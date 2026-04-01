# PHASE 6 — Strategy Compiler & Backtesting Engine

The Strategy Compiler transforms a validated `StrategySpec` into executable signal matrices, and the Backtesting Engine simulates trades using vectorized logic with comprehensive performance metrics.

---

## 6.1 Updated Directory Structure (additions)

```
src/forgequant/core/compiler/
├── __init__.py
├── compiled_strategy.py    # CompiledStrategy data container
├── signal_assembler.py     # Combines entry/exit/filter signals
└── compiler.py             # Main StrategyCompiler

src/forgequant/core/engine/
├── __init__.py
├── results.py              # BacktestResult and TradeRecord
├── metrics.py              # Performance metrics calculator
└── backtester.py           # Vectorized backtesting engine

tests/unit/compiler/
├── __init__.py
├── test_compiled_strategy.py
├── test_signal_assembler.py
└── test_compiler.py

tests/unit/engine/
├── __init__.py
├── test_results.py
├── test_metrics.py
└── test_backtester.py

tests/integration/
└── test_phase6_end_to_end.py
```

---

## 6.2 `src/forgequant/core/compiler/compiled_strategy.py`

```python
"""
Compiled strategy data container.

A CompiledStrategy holds all the intermediate and final signal
DataFrames produced by running a StrategySpec's blocks on OHLCV data.
It is the bridge between the Compiler (which builds it) and the
Backtester (which consumes it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from forgequant.ai_forge.schemas import StrategySpec


@dataclass
class BlockOutput:
    """
    Output from a single block execution.

    Attributes:
        block_name: Name of the block that produced this output.
        category: Block category string.
        params: Validated parameters used.
        result: The DataFrame/Series returned by the block.
    """

    block_name: str
    category: str
    params: dict[str, Any]
    result: pd.DataFrame | pd.Series


@dataclass
class CompiledStrategy:
    """
    A fully compiled strategy ready for backtesting.

    Contains:
        - The original spec for reference
        - Individual block outputs for inspection
        - Assembled signal matrices for direct backtest consumption

    Signal conventions:
        - entry_long / entry_short: Boolean Series, True on signal bar
        - exit_long / exit_short: Boolean Series, True on exit signal bar
        - allow_long / allow_short: Boolean Series, True when filters permit
        - position_size_long / position_size_short: Float Series, units to trade
        - stop_loss_long / stop_loss_short: Float Series, SL price levels
        - take_profit_long / take_profit_short: Float Series, TP price levels
    """

    spec: StrategySpec
    ohlcv: pd.DataFrame
    block_outputs: dict[str, BlockOutput] = field(default_factory=dict)

    # Assembled signal matrices
    entry_long: pd.Series | None = None
    entry_short: pd.Series | None = None
    exit_long: pd.Series | None = None
    exit_short: pd.Series | None = None
    allow_long: pd.Series | None = None
    allow_short: pd.Series | None = None
    position_size_long: pd.Series | None = None
    position_size_short: pd.Series | None = None
    stop_loss_long: pd.Series | None = None
    stop_loss_short: pd.Series | None = None
    take_profit_long: pd.Series | None = None
    take_profit_short: pd.Series | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        """The datetime index shared by all signals."""
        return self.ohlcv.index  # type: ignore[return-value]

    @property
    def close(self) -> pd.Series:
        """Close price series."""
        return self.ohlcv["close"]

    @property
    def n_bars(self) -> int:
        """Number of bars in the data."""
        return len(self.ohlcv)

    def filtered_entry_long(self) -> pd.Series:
        """Entry long signals ANDed with filter permissions."""
        if self.entry_long is None:
            return pd.Series(False, index=self.index)
        entry = self.entry_long.copy()
        if self.allow_long is not None:
            entry = entry & self.allow_long
        return entry

    def filtered_entry_short(self) -> pd.Series:
        """Entry short signals ANDed with filter permissions."""
        if self.entry_short is None:
            return pd.Series(False, index=self.index)
        entry = self.entry_short.copy()
        if self.allow_short is not None:
            entry = entry & self.allow_short
        return entry

    def get_block_output(self, block_name: str) -> BlockOutput | None:
        """Look up a specific block's output by name."""
        return self.block_outputs.get(block_name)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the compiled strategy."""
        def _count(s: pd.Series | None) -> int:
            if s is None:
                return 0
            return int(s.sum()) if s.dtype == bool else 0

        return {
            "strategy_name": self.spec.name,
            "n_bars": self.n_bars,
            "n_blocks": len(self.block_outputs),
            "raw_long_entries": _count(self.entry_long),
            "raw_short_entries": _count(self.entry_short),
            "filtered_long_entries": _count(self.filtered_entry_long()),
            "filtered_short_entries": _count(self.filtered_entry_short()),
            "long_exits": _count(self.exit_long),
            "short_exits": _count(self.exit_short),
        }
```

---

## 6.3 `src/forgequant/core/compiler/signal_assembler.py`

```python
"""
Signal assembler for combining block outputs into unified signal matrices.

Takes the raw outputs from all executed blocks and produces the final
boolean entry/exit signals, filter masks, position sizes, and TP/SL levels
that the backtester consumes.

Assembly rules:
    - Entry signals: OR across all entry rule blocks, then AND with filters
    - Exit signals: OR across all exit rule blocks
    - Filters: AND across all filter blocks (all must agree)
    - Position sizing: from the single money management block
    - TP/SL: from the first exit rule that provides them
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.core.logging import get_logger

logger = get_logger(__name__)


# Maps block output column patterns to signal types
ENTRY_LONG_PATTERNS = [
    "crossover_long_entry",
    "threshold_long_entry",
    "confluence_long_entry",
    "reversal_long_entry",
]

ENTRY_SHORT_PATTERNS = [
    "crossover_short_entry",
    "threshold_short_entry",
    "confluence_short_entry",
    "reversal_short_entry",
]

EXIT_LONG_PATTERNS = [
    "trail_long_exit",
    "time_max_bars_exit",
]

EXIT_SHORT_PATTERNS = [
    "trail_short_exit",
    "time_max_bars_exit",
]

ALLOW_LONG_PATTERNS = [
    "trend_allow_long",
    "session_active",
    "spread_ok",
    "dd_allow_trading",
]

ALLOW_SHORT_PATTERNS = [
    "trend_allow_short",
    "session_active",
    "spread_ok",
    "dd_allow_trading",
]


def _find_column(
    output: BlockOutput,
    patterns: list[str],
) -> pd.Series | None:
    """
    Search a block output for a column matching any of the given patterns.

    Args:
        output: A BlockOutput with a DataFrame result.
        patterns: List of column name patterns to search for.

    Returns:
        The first matching boolean Series, or None.
    """
    if not isinstance(output.result, pd.DataFrame):
        return None

    for pattern in patterns:
        if pattern in output.result.columns:
            series = output.result[pattern]
            if series.dtype == bool or series.dtype == np.bool_:
                return series
            # Try to coerce to bool
            try:
                return series.astype(bool)
            except (ValueError, TypeError):
                continue

    return None


def _find_float_column(
    output: BlockOutput,
    column_name: str,
) -> pd.Series | None:
    """Find a specific float column in a block output."""
    if not isinstance(output.result, pd.DataFrame):
        return None

    if column_name in output.result.columns:
        return output.result[column_name]

    return None


def _or_combine(
    series_list: list[pd.Series],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """OR-combine a list of boolean Series."""
    if not series_list:
        return pd.Series(False, index=index, dtype=bool)

    result = series_list[0].reindex(index, fill_value=False)
    for s in series_list[1:]:
        result = result | s.reindex(index, fill_value=False)

    return result.fillna(False).astype(bool)


def _and_combine(
    series_list: list[pd.Series],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """AND-combine a list of boolean Series."""
    if not series_list:
        return pd.Series(True, index=index, dtype=bool)

    result = series_list[0].reindex(index, fill_value=True)
    for s in series_list[1:]:
        result = result & s.reindex(index, fill_value=True)

    return result.fillna(True).astype(bool)


def assemble_signals(compiled: CompiledStrategy) -> CompiledStrategy:
    """
    Assemble all block outputs into unified signal matrices.

    Modifies the CompiledStrategy in place, populating:
        - entry_long, entry_short
        - exit_long, exit_short
        - allow_long, allow_short
        - position_size_long, position_size_short
        - stop_loss_long, stop_loss_short
        - take_profit_long, take_profit_short

    Args:
        compiled: CompiledStrategy with block_outputs populated.

    Returns:
        The same CompiledStrategy with signal fields populated.
    """
    index = compiled.index

    # ── Entry signals (OR across entry rule blocks) ──
    entry_longs: list[pd.Series] = []
    entry_shorts: list[pd.Series] = []

    for name, output in compiled.block_outputs.items():
        if output.category == "entry_rule":
            el = _find_column(output, ENTRY_LONG_PATTERNS)
            if el is not None:
                entry_longs.append(el)

            es = _find_column(output, ENTRY_SHORT_PATTERNS)
            if es is not None:
                entry_shorts.append(es)

    # Also check price action blocks for entry-like signals
    for name, output in compiled.block_outputs.items():
        if output.category == "price_action":
            # Breakout signals can serve as entries
            bl = _find_column(output, ["breakout_long"])
            if bl is not None:
                # AND with volume confirmation if present
                vol = _find_column(output, ["breakout_volume_confirm"])
                if vol is not None:
                    bl = bl & vol
                entry_longs.append(bl)

            bs = _find_column(output, ["breakout_short"])
            if bs is not None:
                vol = _find_column(output, ["breakout_volume_confirm"])
                if vol is not None:
                    bs = bs & vol
                entry_shorts.append(bs)

            # Pullback signals
            pl = _find_column(output, ["pullback_long"])
            if pl is not None:
                entry_longs.append(pl)

            ps = _find_column(output, ["pullback_short"])
            if ps is not None:
                entry_shorts.append(ps)

    compiled.entry_long = _or_combine(entry_longs, index)
    compiled.entry_short = _or_combine(entry_shorts, index)

    # ── Exit signals (OR across exit rule blocks) ──
    exit_longs: list[pd.Series] = []
    exit_shorts: list[pd.Series] = []

    for name, output in compiled.block_outputs.items():
        if output.category == "exit_rule":
            xl = _find_column(output, EXIT_LONG_PATTERNS)
            if xl is not None:
                exit_longs.append(xl)

            xs = _find_column(output, EXIT_SHORT_PATTERNS)
            if xs is not None:
                exit_shorts.append(xs)

    compiled.exit_long = _or_combine(exit_longs, index)
    compiled.exit_short = _or_combine(exit_shorts, index)

    # ── Filter masks (AND across filter blocks) ──
    allow_longs: list[pd.Series] = []
    allow_shorts: list[pd.Series] = []

    for name, output in compiled.block_outputs.items():
        if output.category == "filter":
            al = _find_column(output, ALLOW_LONG_PATTERNS)
            if al is not None:
                allow_longs.append(al)

            ash = _find_column(output, ALLOW_SHORT_PATTERNS)
            if ash is not None:
                allow_shorts.append(ash)

    compiled.allow_long = _and_combine(allow_longs, index)
    compiled.allow_short = _and_combine(allow_shorts, index)

    # ── TP/SL levels (from exit rules) ──
    for name, output in compiled.block_outputs.items():
        if output.category == "exit_rule":
            if compiled.stop_loss_long is None:
                sl_l = _find_float_column(output, "tpsl_long_sl")
                if sl_l is not None:
                    compiled.stop_loss_long = sl_l

            if compiled.stop_loss_short is None:
                sl_s = _find_float_column(output, "tpsl_short_sl")
                if sl_s is not None:
                    compiled.stop_loss_short = sl_s

            if compiled.take_profit_long is None:
                tp_l = _find_float_column(output, "tpsl_long_tp")
                if tp_l is not None:
                    compiled.take_profit_long = tp_l

            if compiled.take_profit_short is None:
                tp_s = _find_float_column(output, "tpsl_short_tp")
                if tp_s is not None:
                    compiled.take_profit_short = tp_s

    # ── Position sizing (from money management block) ──
    mm_name = compiled.spec.money_management.block_name
    mm_output = compiled.block_outputs.get(mm_name)

    if mm_output is not None and isinstance(mm_output.result, pd.DataFrame):
        # Search for position size columns across different MM blocks
        size_col_candidates = [
            "fr_position_size",
            "vt_position_size",
            "kelly_position_size",
            "atrs_position_size",
        ]
        for col in size_col_candidates:
            if col in mm_output.result.columns:
                compiled.position_size_long = mm_output.result[col]
                compiled.position_size_short = mm_output.result[col]
                break

    logger.info(
        "signals_assembled",
        strategy=compiled.spec.name,
        entry_long_count=int(compiled.entry_long.sum()) if compiled.entry_long is not None else 0,
        entry_short_count=int(compiled.entry_short.sum()) if compiled.entry_short is not None else 0,
        has_tp_sl=compiled.stop_loss_long is not None,
        has_position_sizing=compiled.position_size_long is not None,
    )

    return compiled
```

---

## 6.4 `src/forgequant/core/compiler/compiler.py`

```python
"""
Strategy compiler.

Transforms a validated StrategySpec into a CompiledStrategy by:
    1. Instantiating all blocks from the registry
    2. Executing each block on the OHLCV data
    3. Assembling the raw outputs into unified signal matrices

The compiler is the bridge between the AI Forge (which produces specs)
and the Backtester (which consumes compiled strategies).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.ai_forge.validator import SpecValidator, ValidationResult
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.core.compiler.signal_assembler import assemble_signals
from forgequant.core.exceptions import StrategyCompileError
from forgequant.core.logging import get_logger
from forgequant.core.types import validate_ohlcv

logger = get_logger(__name__)


class StrategyCompiler:
    """
    Compiles a StrategySpec into a CompiledStrategy.

    The compiler:
        1. Optionally validates the spec (skip if already validated)
        2. Instantiates each block from the registry
        3. Executes blocks in dependency order (indicators first, then
           price action, entry rules, exit rules, filters, money mgmt)
        4. Assembles outputs into unified signal matrices

    Usage:
        compiler = StrategyCompiler()
        compiled = compiler.compile(spec, ohlcv_data)
        # compiled is now ready for the backtester
    """

    # Execution order: blocks are run in this category sequence
    EXECUTION_ORDER = [
        "indicators",
        "price_action",
        "entry_rules",
        "exit_rules",
        "filters",
    ]

    def __init__(
        self,
        registry: BlockRegistry | None = None,
        validate: bool = True,
    ) -> None:
        """
        Initialize the compiler.

        Args:
            registry: BlockRegistry to use. Defaults to singleton.
            validate: Whether to validate the spec before compiling.
                      Set to False if the spec was already validated.
        """
        self._registry = registry or BlockRegistry()
        self._validate = validate
        self._validator = SpecValidator(self._registry) if validate else None

    def compile(
        self,
        spec: StrategySpec,
        data: pd.DataFrame,
        validated_params: dict[str, dict[str, Any]] | None = None,
    ) -> CompiledStrategy:
        """
        Compile a StrategySpec into a CompiledStrategy.

        Args:
            spec: The strategy specification to compile.
            data: OHLCV DataFrame with DatetimeIndex.
            validated_params: Pre-validated parameter dict from SpecValidator.
                              If None and validate=True, validation runs first.

        Returns:
            A fully compiled strategy with all signals assembled.

        Raises:
            StrategyCompileError: If compilation fails at any stage.
        """
        logger.info(
            "compilation_start",
            strategy=spec.name,
            n_bars=len(data),
            n_blocks=len(spec.all_blocks()),
        )

        # Step 0: Normalize OHLCV
        data = data.copy()
        data.columns = data.columns.str.lower()

        try:
            validate_ohlcv(data, block_name=f"compiler:{spec.name}")
        except ValueError as e:
            raise StrategyCompileError(spec.name, f"Invalid OHLCV data: {e}") from e

        # Step 1: Optional validation
        val_params = validated_params or {}

        if self._validate and self._validator is not None and not val_params:
            result = self._validator.validate(spec)
            if not result.is_valid:
                raise StrategyCompileError(
                    spec.name,
                    f"Spec validation failed: {'; '.join(result.errors[:5])}",
                )
            val_params = result.validated_params

        # Step 2: Create CompiledStrategy container
        compiled = CompiledStrategy(spec=spec, ohlcv=data)

        # Step 3: Execute blocks in order
        # 3a: Indicators
        for block_spec in spec.indicators:
            self._execute_block(block_spec, "indicator", data, val_params, compiled)

        # 3b: Price action
        for block_spec in spec.price_action:
            self._execute_block(block_spec, "price_action", data, val_params, compiled)

        # 3c: Entry rules
        for block_spec in spec.entry_rules:
            self._execute_block(block_spec, "entry_rule", data, val_params, compiled)

        # 3d: Exit rules
        for block_spec in spec.exit_rules:
            self._execute_block(block_spec, "exit_rule", data, val_params, compiled)

        # 3e: Filters
        for block_spec in spec.filters:
            self._execute_block(block_spec, "filter", data, val_params, compiled)

        # 3f: Money management (single block)
        self._execute_block(
            spec.money_management, "money_management", data, val_params, compiled
        )

        # Step 4: Assemble signals
        compiled = assemble_signals(compiled)

        logger.info(
            "compilation_complete",
            strategy=spec.name,
            **compiled.summary(),
        )

        return compiled

    def _execute_block(
        self,
        block_spec: BlockSpec,
        category: str,
        data: pd.DataFrame,
        validated_params: dict[str, dict[str, Any]],
        compiled: CompiledStrategy,
    ) -> None:
        """
        Execute a single block and store its output.

        Args:
            block_spec: The block specification.
            category: Block category string.
            data: OHLCV DataFrame.
            validated_params: Dict of validated params per block name.
            compiled: CompiledStrategy to store the output in.

        Raises:
            StrategyCompileError: If the block fails to execute.
        """
        name = block_spec.block_name

        # Get the block class
        block_cls = self._registry.get(name)
        if block_cls is None:
            raise StrategyCompileError(
                compiled.spec.name,
                f"Block '{name}' not found in registry",
            )

        # Get validated params (or use raw spec params)
        params = validated_params.get(name, block_spec.params)

        # Instantiate and execute
        try:
            instance = block_cls()
            result = instance.execute(data, params)
        except Exception as e:
            raise StrategyCompileError(
                compiled.spec.name,
                f"Block '{name}' execution failed: {type(e).__name__}: {e}",
            ) from e

        # Ensure result is a DataFrame for uniform handling
        if isinstance(result, pd.Series):
            result = result.to_frame(name=name)

        # Store output
        compiled.block_outputs[name] = BlockOutput(
            block_name=name,
            category=category,
            params=params,
            result=result,
        )

        logger.debug(
            "block_executed",
            block=name,
            category=category,
            result_type=type(result).__name__,
            result_columns=list(result.columns) if isinstance(result, pd.DataFrame) else [],
        )
```

---

## 6.5 `src/forgequant/core/compiler/__init__.py`

```python
"""
Strategy compiler.

Transforms a validated StrategySpec into a runnable CompiledStrategy
by instantiating blocks, executing them on OHLCV data, and assembling
the outputs into unified signal matrices.
"""

from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.compiler.signal_assembler import assemble_signals

__all__ = [
    "BlockOutput",
    "CompiledStrategy",
    "StrategyCompiler",
    "assemble_signals",
]
```

---

## 6.6 `src/forgequant/core/engine/results.py`

```python
"""
Backtest result containers.

TradeRecord holds individual trade details.
BacktestResult holds the full backtest output including equity curve,
trade list, and performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """
    Record of a single completed trade.

    Attributes:
        trade_id: Unique trade identifier.
        direction: "long" or "short".
        entry_time: Datetime of entry bar.
        entry_price: Price at entry.
        exit_time: Datetime of exit bar.
        exit_price: Price at exit.
        exit_reason: Why the trade was closed ("tp", "sl", "signal", "time", "end").
        position_size: Number of units traded.
        pnl: Profit/loss in price units (per unit).
        pnl_pct: Profit/loss as percentage of entry price.
        pnl_dollar: Dollar P&L (pnl * position_size).
        bars_held: Number of bars the position was open.
        mae: Maximum Adverse Excursion (worst drawdown during trade).
        mfe: Maximum Favorable Excursion (best unrealized profit during trade).
    """

    trade_id: int
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    position_size: float
    pnl: float
    pnl_pct: float
    pnl_dollar: float
    bars_held: int
    mae: float = 0.0
    mfe: float = 0.0

    @property
    def is_winner(self) -> bool:
        """True if this trade was profitable."""
        return self.pnl > 0.0

    @property
    def risk_reward_achieved(self) -> float:
        """Actual risk-reward ratio achieved (MFE / MAE)."""
        if self.mae == 0.0:
            return float("inf") if self.mfe > 0 else 0.0
        return abs(self.mfe / self.mae)


@dataclass
class BacktestResult:
    """
    Complete backtest result.

    Attributes:
        strategy_name: Name of the strategy.
        start_date: First bar datetime.
        end_date: Last bar datetime.
        initial_equity: Starting equity.
        final_equity: Ending equity.
        equity_curve: Series of equity values at each bar.
        trades: List of all completed trades.
        metrics: Dictionary of performance metrics.
        drawdown_series: Series of drawdown values at each bar.
        returns_series: Series of bar-by-bar returns.
    """

    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_equity: float
    final_equity: float
    equity_curve: pd.Series
    trades: list[TradeRecord] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    drawdown_series: pd.Series | None = None
    returns_series: pd.Series | None = None

    @property
    def n_trades(self) -> int:
        """Total number of completed trades."""
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        """Total dollar P&L across all trades."""
        return sum(t.pnl_dollar for t in self.trades)

    @property
    def winning_trades(self) -> list[TradeRecord]:
        """List of winning trades."""
        return [t for t in self.trades if t.is_winner]

    @property
    def losing_trades(self) -> list[TradeRecord]:
        """List of losing trades."""
        return [t for t in self.trades if not t.is_winner]

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trade list to a DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "trade_id": t.trade_id,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "position_size": t.position_size,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "pnl_dollar": t.pnl_dollar,
                "bars_held": t.bars_held,
                "mae": t.mae,
                "mfe": t.mfe,
                "is_winner": t.is_winner,
            })

        return pd.DataFrame(records)

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "strategy_name": self.strategy_name,
            "period": f"{self.start_date} to {self.end_date}",
            "initial_equity": self.initial_equity,
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(
                (self.final_equity / self.initial_equity - 1) * 100, 2
            ),
            "n_trades": self.n_trades,
            "total_pnl": round(self.total_pnl, 2),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in self.metrics.items()},
        }
```

---

## 6.7 `src/forgequant/core/engine/metrics.py`

```python
"""
Performance metrics calculator.

Computes a comprehensive set of backtest performance metrics from
an equity curve and trade list. All metrics follow standard
quantitative finance definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.core.engine.results import BacktestResult, TradeRecord
from forgequant.core.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> BacktestResult:
    """
    Compute all performance metrics and populate the BacktestResult.

    Modifies the result in place, populating:
        - result.metrics: dict of all computed metrics
        - result.drawdown_series: per-bar drawdown
        - result.returns_series: per-bar returns

    Args:
        result: BacktestResult with equity_curve and trades populated.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        annualization_factor: Number of bars per year (252 for daily,
                              252*6.5 for hourly US equities, etc.)

    Returns:
        The same BacktestResult with metrics populated.
    """
    equity = result.equity_curve
    trades = result.trades

    if len(equity) < 2:
        result.metrics = {"error": "Insufficient data for metrics"}
        return result

    # ── Returns ──
    returns = equity.pct_change().fillna(0.0)
    result.returns_series = returns

    # ── Drawdown ──
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    result.drawdown_series = drawdown

    metrics: dict[str, float] = {}

    # ── Return metrics ──
    total_return = (result.final_equity / result.initial_equity) - 1.0
    metrics["total_return_pct"] = total_return * 100.0

    n_bars = len(equity)
    if n_bars > 1 and annualization_factor > 0:
        ann_return = (1.0 + total_return) ** (annualization_factor / n_bars) - 1.0
        metrics["annualized_return_pct"] = ann_return * 100.0
    else:
        metrics["annualized_return_pct"] = 0.0

    # ── Risk metrics ──
    metrics["max_drawdown_pct"] = abs(drawdown.min()) * 100.0
    metrics["max_drawdown_duration"] = _max_drawdown_duration(equity)

    # ── Volatility ──
    if returns.std() > 0:
        ann_vol = returns.std() * np.sqrt(annualization_factor)
        metrics["annualized_volatility_pct"] = ann_vol * 100.0
    else:
        ann_vol = 0.0
        metrics["annualized_volatility_pct"] = 0.0

    # ── Sharpe Ratio ──
    if ann_vol > 0 and annualization_factor > 0:
        excess_return = metrics["annualized_return_pct"] / 100.0 - risk_free_rate
        metrics["sharpe_ratio"] = excess_return / ann_vol
    else:
        metrics["sharpe_ratio"] = 0.0

    # ── Sortino Ratio ──
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(annualization_factor)
        if downside_std > 0:
            excess_return = metrics["annualized_return_pct"] / 100.0 - risk_free_rate
            metrics["sortino_ratio"] = excess_return / downside_std
        else:
            metrics["sortino_ratio"] = 0.0
    else:
        metrics["sortino_ratio"] = float("inf") if total_return > 0 else 0.0

    # ── Calmar Ratio ──
    max_dd = metrics["max_drawdown_pct"] / 100.0
    if max_dd > 0:
        metrics["calmar_ratio"] = (metrics["annualized_return_pct"] / 100.0) / max_dd
    else:
        metrics["calmar_ratio"] = float("inf") if total_return > 0 else 0.0

    # ── Trade metrics ──
    n_trades = len(trades)
    metrics["total_trades"] = float(n_trades)

    if n_trades > 0:
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        metrics["win_rate"] = len(winners) / n_trades
        metrics["loss_rate"] = len(losers) / n_trades

        # Average P&L
        metrics["avg_pnl_dollar"] = np.mean([t.pnl_dollar for t in trades])
        metrics["avg_pnl_pct"] = np.mean([t.pnl_pct for t in trades])

        # Average winner / loser
        if winners:
            metrics["avg_winner_dollar"] = np.mean([t.pnl_dollar for t in winners])
            metrics["avg_winner_pct"] = np.mean([t.pnl_pct for t in winners])
        else:
            metrics["avg_winner_dollar"] = 0.0
            metrics["avg_winner_pct"] = 0.0

        if losers:
            metrics["avg_loser_dollar"] = np.mean([t.pnl_dollar for t in losers])
            metrics["avg_loser_pct"] = np.mean([t.pnl_pct for t in losers])
        else:
            metrics["avg_loser_dollar"] = 0.0
            metrics["avg_loser_pct"] = 0.0

        # Profit factor
        gross_profit = sum(t.pnl_dollar for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl_dollar for t in losers)) if losers else 0.0
        if gross_loss > 0:
            metrics["profit_factor"] = gross_profit / gross_loss
        else:
            metrics["profit_factor"] = float("inf") if gross_profit > 0 else 0.0

        # Payoff ratio
        if metrics["avg_loser_dollar"] != 0:
            metrics["payoff_ratio"] = abs(
                metrics["avg_winner_dollar"] / metrics["avg_loser_dollar"]
            )
        else:
            metrics["payoff_ratio"] = float("inf") if metrics["avg_winner_dollar"] > 0 else 0.0

        # Expectancy
        metrics["expectancy_dollar"] = (
            metrics["win_rate"] * metrics["avg_winner_dollar"]
            + metrics["loss_rate"] * metrics["avg_loser_dollar"]
        )

        # Holding period
        metrics["avg_bars_held"] = np.mean([t.bars_held for t in trades])
        metrics["max_bars_held"] = float(max(t.bars_held for t in trades))

        # Consecutive wins/losses
        metrics["max_consecutive_wins"] = float(_max_consecutive(trades, winners=True))
        metrics["max_consecutive_losses"] = float(_max_consecutive(trades, winners=False))

        # Long/short breakdown
        long_trades = [t for t in trades if t.direction == "long"]
        short_trades = [t for t in trades if t.direction == "short"]
        metrics["long_trades"] = float(len(long_trades))
        metrics["short_trades"] = float(len(short_trades))

        if long_trades:
            metrics["long_win_rate"] = len([t for t in long_trades if t.is_winner]) / len(long_trades)
        else:
            metrics["long_win_rate"] = 0.0

        if short_trades:
            metrics["short_win_rate"] = (
                len([t for t in short_trades if t.is_winner]) / len(short_trades)
            )
        else:
            metrics["short_win_rate"] = 0.0

    else:
        # No trades
        for key in [
            "win_rate", "loss_rate", "avg_pnl_dollar", "avg_pnl_pct",
            "avg_winner_dollar", "avg_winner_pct",
            "avg_loser_dollar", "avg_loser_pct",
            "profit_factor", "payoff_ratio", "expectancy_dollar",
            "avg_bars_held", "max_bars_held",
            "max_consecutive_wins", "max_consecutive_losses",
            "long_trades", "short_trades",
            "long_win_rate", "short_win_rate",
        ]:
            metrics[key] = 0.0

    result.metrics = metrics

    logger.info(
        "metrics_computed",
        strategy=result.strategy_name,
        n_trades=n_trades,
        total_return_pct=round(metrics["total_return_pct"], 2),
        sharpe=round(metrics["sharpe_ratio"], 3),
        max_dd_pct=round(metrics["max_drawdown_pct"], 2),
    )

    return result


def _max_drawdown_duration(equity: pd.Series) -> float:
    """
    Compute the maximum drawdown duration in number of bars.

    This is the longest period between an equity peak and its recovery
    to a new peak (or end of data if not recovered).
    """
    running_max = equity.expanding().max()
    in_drawdown = equity < running_max

    if not in_drawdown.any():
        return 0.0

    # Find consecutive drawdown sequences
    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown.values:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return float(max_duration)


def _max_consecutive(trades: list[TradeRecord], winners: bool) -> int:
    """
    Compute maximum consecutive wins or losses.

    Args:
        trades: Ordered list of trades.
        winners: If True, count consecutive wins; else consecutive losses.

    Returns:
        Maximum consecutive count.
    """
    max_streak = 0
    current_streak = 0

    for trade in trades:
        if trade.is_winner == winners:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak
```

---

## 6.8 `src/forgequant/core/engine/backtester.py`

```python
"""
Vectorized backtesting engine.

Simulates trade execution using the compiled strategy's signal matrices.
Operates bar-by-bar with vectorized pre-computation where possible.

Execution model:
    - Signals are evaluated at the CLOSE of each bar
    - Entries are executed at the OPEN of the next bar
    - One position at a time per direction (no pyramiding by default)
    - TP/SL checked against the bar's high/low
    - Position sizing from the money management block

This is a "next-bar execution" model that prevents lookahead bias:
    - Bar N: signals computed from close[N]
    - Bar N+1: entry at open[N+1], TP/SL checked at high[N+1]/low[N+1]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from forgequant.core.compiler.compiled_strategy import CompiledStrategy
from forgequant.core.engine.metrics import compute_metrics
from forgequant.core.engine.results import BacktestResult, TradeRecord
from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for the backtester."""

    initial_equity: float = 100_000.0
    commission_per_unit: float = 0.0
    slippage_pct: float = 0.0
    allow_pyramiding: bool = False
    max_positions: int = 1
    default_position_size: float = 1.0
    annualization_factor: float = 252.0
    risk_free_rate: float = 0.0


class Backtester:
    """
    Vectorized backtesting engine.

    Processes a CompiledStrategy bar-by-bar, managing positions,
    checking TP/SL, and tracking equity.

    Usage:
        backtester = Backtester()
        result = backtester.run(compiled_strategy)
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()

    def run(self, compiled: CompiledStrategy) -> BacktestResult:
        """
        Run the backtest on a compiled strategy.

        Args:
            compiled: CompiledStrategy with signals assembled.

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        cfg = self._config
        data = compiled.ohlcv
        n = len(data)

        logger.info(
            "backtest_start",
            strategy=compiled.spec.name,
            n_bars=n,
            initial_equity=cfg.initial_equity,
        )

        # Pre-extract arrays for performance
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        index = data.index

        entry_long = compiled.filtered_entry_long().values
        entry_short = compiled.filtered_entry_short().values
        exit_long = compiled.exit_long.values if compiled.exit_long is not None else np.zeros(n, dtype=bool)
        exit_short = compiled.exit_short.values if compiled.exit_short is not None else np.zeros(n, dtype=bool)

        # TP/SL arrays (may be None)
        sl_long = compiled.stop_loss_long.values if compiled.stop_loss_long is not None else None
        sl_short = compiled.stop_loss_short.values if compiled.stop_loss_short is not None else None
        tp_long = compiled.take_profit_long.values if compiled.take_profit_long is not None else None
        tp_short = compiled.take_profit_short.values if compiled.take_profit_short is not None else None

        # Position sizing array
        pos_size_arr = (
            compiled.position_size_long.values
            if compiled.position_size_long is not None
            else np.full(n, cfg.default_position_size)
        )

        # State tracking
        equity = cfg.initial_equity
        equity_curve = np.zeros(n)
        trades: list[TradeRecord] = []
        trade_id = 0

        # Current position state
        in_long = False
        in_short = False
        entry_price = 0.0
        entry_bar = 0
        entry_sl = 0.0
        entry_tp = 0.0
        position_size = 0.0
        trade_mae = 0.0
        trade_mfe = 0.0

        for i in range(n):
            bar_pnl = 0.0

            # ── Check exits for open positions ──
            if in_long:
                # Check stop loss (low hits SL)
                if entry_sl > 0 and low_prices[i] <= entry_sl:
                    exit_price = entry_sl
                    exit_reason = "sl"
                    bar_pnl, trade_rec = self._close_trade(
                        trade_id, "long", entry_bar, i, index,
                        entry_price, exit_price, exit_reason,
                        position_size, trade_mae, trade_mfe, cfg,
                    )
                    trades.append(trade_rec)
                    trade_id += 1
                    in_long = False

                # Check take profit (high hits TP)
                elif entry_tp > 0 and high_prices[i] >= entry_tp:
                    exit_price = entry_tp
                    exit_reason = "tp"
                    bar_pnl, trade_rec = self._close_trade(
                        trade_id, "long", entry_bar, i, index,
                        entry_price, exit_price, exit_reason,
                        position_size, trade_mae, trade_mfe, cfg,
                    )
                    trades.append(trade_rec)
                    trade_id += 1
                    in_long = False

                # Check signal-based exit
                elif exit_long[i]:
                    exit_price = close_prices[i]
                    exit_reason = "signal"
                    bar_pnl, trade_rec = self._close_trade(
                        trade_id, "long", entry_bar, i, index,
                        entry_price, exit_price, exit_reason,
                        position_size, trade_mae, trade_mfe, cfg,
                    )
                    trades.append(trade_rec)
                    trade_id += 1
                    in_long = False

                else:
                    # Update MAE/MFE
                    unrealized = low_prices[i] - entry_price
                    trade_mae = min(trade_mae, unrealized)
                    unrealized_best = high_prices[i] - entry_price
                    trade_mfe = max(trade_mfe, unrealized_best)

            if in_short:
                # Check stop loss (high hits SL)
                if entry_sl > 0 and high_prices[i] >= entry_sl:
                    exit_price = entry_sl
                    exit_reason = "sl"
                    bar_pnl, trade_rec = self._close_trade(
                        trade_id, "short", entry_bar, i, index,
                        entry_price, exit_price, exit_reason,
                        position_size, trade_mae, trade_mfe, cfg,
                    )
                    trades.append(trade_rec)
                    trade_id += 1
                    in_short = False

                elif entry_tp > 0 and low_prices[i] <= entry_tp:
                    exit_price = entry_tp
                    exit_reason = "tp"
                    bar_pnl, trade_rec = self._close_trade(
                        trade_id, "short", entry_bar, i, index,
                        entry_price, exit_price, exit_reason,
                        position_size, trade_mae, trade_mfe, cfg,
                    )
                    trades.append(trade_rec)
                    trade_id += 1
                    in_short = False

                elif exit_short[i]:
                    exit_price = close_prices[i]
                    exit_reason = "signal"
                    bar_pnl, trade_rec = self._close_trade(
                        trade_id, "short", entry_bar, i, index,
                        entry_price, exit_price, exit_reason,
                        position_size, trade_mae, trade_mfe, cfg,
                    )
                    trades.append(trade_rec)
                    trade_id += 1
                    in_short = False

                else:
                    unrealized = entry_price - high_prices[i]
                    trade_mae = min(trade_mae, unrealized)
                    unrealized_best = entry_price - low_prices[i]
                    trade_mfe = max(trade_mfe, unrealized_best)

            # ── Check entries (signals from bar i, execute at bar i+1) ──
            # We use the current bar's signal and current bar's close as
            # entry price (next-bar-open would require i+1 to exist).
            # For simplicity and to avoid index issues, we enter at the
            # NEXT bar's open if available, otherwise at current close.
            if not in_long and not in_short:
                if i < n - 1 and entry_long[i]:
                    in_long = True
                    entry_price = open_prices[i + 1]
                    entry_bar = i + 1

                    # Apply slippage
                    entry_price *= (1.0 + cfg.slippage_pct / 100.0)

                    # Position size from signal bar
                    ps = pos_size_arr[i]
                    position_size = ps if not np.isnan(ps) and ps > 0 else cfg.default_position_size

                    # TP/SL from signal bar
                    entry_sl = sl_long[i] if sl_long is not None and not np.isnan(sl_long[i]) else 0.0
                    entry_tp = tp_long[i] if tp_long is not None and not np.isnan(tp_long[i]) else 0.0

                    trade_mae = 0.0
                    trade_mfe = 0.0

                    # Commission on entry
                    equity -= position_size * cfg.commission_per_unit

                elif i < n - 1 and entry_short[i]:
                    in_short = True
                    entry_price = open_prices[i + 1]
                    entry_bar = i + 1

                    # Apply slippage (adverse for shorts = lower entry)
                    entry_price *= (1.0 - cfg.slippage_pct / 100.0)

                    ps = pos_size_arr[i]
                    position_size = ps if not np.isnan(ps) and ps > 0 else cfg.default_position_size

                    entry_sl = sl_short[i] if sl_short is not None and not np.isnan(sl_short[i]) else 0.0
                    entry_tp = tp_short[i] if tp_short is not None and not np.isnan(tp_short[i]) else 0.0

                    trade_mae = 0.0
                    trade_mfe = 0.0

                    equity -= position_size * cfg.commission_per_unit

            equity += bar_pnl
            equity_curve[i] = equity

        # Close any remaining position at end of data
        if in_long:
            exit_price = close_prices[-1]
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "long", entry_bar, n - 1, index,
                entry_price, exit_price, "end",
                position_size, trade_mae, trade_mfe, cfg,
            )
            trades.append(trade_rec)
            equity += bar_pnl
            equity_curve[-1] = equity

        if in_short:
            exit_price = close_prices[-1]
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "short", entry_bar, n - 1, index,
                entry_price, exit_price, "end",
                position_size, trade_mae, trade_mfe, cfg,
            )
            trades.append(trade_rec)
            equity += bar_pnl
            equity_curve[-1] = equity

        # Build result
        equity_series = pd.Series(equity_curve, index=index, name="equity")

        result = BacktestResult(
            strategy_name=compiled.spec.name,
            start_date=index[0].to_pydatetime(),
            end_date=index[-1].to_pydatetime(),
            initial_equity=cfg.initial_equity,
            final_equity=equity,
            equity_curve=equity_series,
            trades=trades,
        )

        # Compute metrics
        result = compute_metrics(
            result,
            risk_free_rate=cfg.risk_free_rate,
            annualization_factor=cfg.annualization_factor,
        )

        logger.info(
            "backtest_complete",
            strategy=compiled.spec.name,
            n_trades=len(trades),
            final_equity=round(equity, 2),
            total_return_pct=round(result.metrics.get("total_return_pct", 0), 2),
        )

        return result

    @staticmethod
    def _close_trade(
        trade_id: int,
        direction: str,
        entry_bar: int,
        exit_bar: int,
        index: pd.DatetimeIndex,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        position_size: float,
        mae: float,
        mfe: float,
        cfg: BacktestConfig,
    ) -> tuple[float, TradeRecord]:
        """
        Close a trade and compute P&L.

        Returns:
            Tuple of (dollar_pnl_adjustment, TradeRecord).
        """
        if direction == "long":
            pnl_per_unit = exit_price - entry_price
        else:
            pnl_per_unit = entry_price - exit_price

        pnl_pct = pnl_per_unit / entry_price * 100.0 if entry_price != 0 else 0.0
        pnl_dollar = pnl_per_unit * position_size

        # Commission on exit
        commission = position_size * cfg.commission_per_unit
        pnl_dollar -= commission

        bars_held = exit_bar - entry_bar
        if bars_held < 0:
            bars_held = 0

        record = TradeRecord(
            trade_id=trade_id,
            direction=direction,
            entry_time=index[entry_bar].to_pydatetime(),
            entry_price=entry_price,
            exit_time=index[exit_bar].to_pydatetime(),
            exit_price=exit_price,
            exit_reason=exit_reason,
            position_size=position_size,
            pnl=pnl_per_unit,
            pnl_pct=pnl_pct,
            pnl_dollar=pnl_dollar,
            bars_held=bars_held,
            mae=mae,
            mfe=mfe,
        )

        return pnl_dollar, record
```

---

## 6.9 `src/forgequant/core/engine/__init__.py`

```python
"""
Backtesting engine.

Provides vectorized trade simulation, performance metrics computation,
and structured result containers.
"""

from forgequant.core.engine.backtester import Backtester, BacktestConfig
from forgequant.core.engine.metrics import compute_metrics
from forgequant.core.engine.results import BacktestResult, TradeRecord

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "TradeRecord",
    "compute_metrics",
]
```

---

## 6.10 Test Suite — Compiler

### `tests/unit/compiler/__init__.py`

```python
"""Tests for strategy compiler."""
```

---

### `tests/unit/compiler/test_compiled_strategy.py`

```python
"""Tests for CompiledStrategy container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="test_compiled",
        description="A test strategy for CompiledStrategy container testing with detail.",
        objective={"style": "trend_following", "timeframe": "1h"},
        indicators=[BlockSpec(block_name="ema")],
        entry_rules=[BlockSpec(block_name="crossover_entry")],
        exit_rules=[BlockSpec(block_name="fixed_tpsl")],
        money_management=BlockSpec(block_name="fixed_risk"),
    )


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(n) * 1000,
        },
        index=dates,
    )


class TestBlockOutput:
    def test_creation(self) -> None:
        df = pd.DataFrame({"ema_20": [1.0, 2.0, 3.0]})
        bo = BlockOutput(
            block_name="ema",
            category="indicator",
            params={"period": 20},
            result=df,
        )
        assert bo.block_name == "ema"
        assert bo.category == "indicator"
        assert len(bo.result) == 3


class TestCompiledStrategy:
    def test_basic_properties(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(50)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        assert compiled.n_bars == 50
        assert len(compiled.close) == 50
        assert len(compiled.index) == 50

    def test_filtered_entry_no_filter(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(10)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)
        compiled.entry_long = pd.Series([True] * 5 + [False] * 5, index=ohlcv.index)

        filtered = compiled.filtered_entry_long()
        # No filter set -> same as raw entries
        assert filtered.sum() == 5

    def test_filtered_entry_with_filter(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(10)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)
        compiled.entry_long = pd.Series([True] * 10, index=ohlcv.index)
        compiled.allow_long = pd.Series(
            [True] * 5 + [False] * 5, index=ohlcv.index
        )

        filtered = compiled.filtered_entry_long()
        assert filtered.sum() == 5

    def test_filtered_entry_none(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(10)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        filtered = compiled.filtered_entry_long()
        assert filtered.sum() == 0

    def test_get_block_output(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(10)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        bo = BlockOutput("ema", "indicator", {}, pd.DataFrame({"x": [1]}))
        compiled.block_outputs["ema"] = bo

        assert compiled.get_block_output("ema") is bo
        assert compiled.get_block_output("nonexistent") is None

    def test_summary(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(20)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)
        compiled.entry_long = pd.Series([True, False] * 10, index=ohlcv.index)
        compiled.entry_short = pd.Series(False, index=ohlcv.index)
        compiled.exit_long = pd.Series(False, index=ohlcv.index)
        compiled.exit_short = pd.Series(False, index=ohlcv.index)

        s = compiled.summary()
        assert s["strategy_name"] == "test_compiled"
        assert s["n_bars"] == 20
        assert s["raw_long_entries"] == 10
```

---

### `tests/unit/compiler/test_signal_assembler.py`

```python
"""Tests for the signal assembler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.core.compiler.signal_assembler import (
    _and_combine,
    _or_combine,
    assemble_signals,
)


def _make_index(n: int = 50) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h")


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="assembler_test",
        description="A test strategy for signal assembler testing with sufficient length.",
        objective={"style": "trend_following", "timeframe": "1h"},
        indicators=[BlockSpec(block_name="ema")],
        entry_rules=[BlockSpec(block_name="crossover_entry")],
        exit_rules=[BlockSpec(block_name="fixed_tpsl")],
        money_management=BlockSpec(block_name="fixed_risk"),
    )


class TestCombineFunctions:
    def test_or_combine_empty(self) -> None:
        idx = _make_index(5)
        result = _or_combine([], idx)
        assert result.sum() == 0

    def test_or_combine_single(self) -> None:
        idx = _make_index(5)
        s = pd.Series([True, False, True, False, True], index=idx)
        result = _or_combine([s], idx)
        assert result.sum() == 3

    def test_or_combine_multiple(self) -> None:
        idx = _make_index(5)
        s1 = pd.Series([True, False, False, False, False], index=idx)
        s2 = pd.Series([False, True, False, False, False], index=idx)
        result = _or_combine([s1, s2], idx)
        assert result.sum() == 2

    def test_and_combine_empty(self) -> None:
        idx = _make_index(5)
        result = _and_combine([], idx)
        assert result.all()  # Default is True (no filters = allow all)

    def test_and_combine_single(self) -> None:
        idx = _make_index(5)
        s = pd.Series([True, True, False, True, True], index=idx)
        result = _and_combine([s], idx)
        assert result.sum() == 4

    def test_and_combine_multiple(self) -> None:
        idx = _make_index(5)
        s1 = pd.Series([True, True, True, False, True], index=idx)
        s2 = pd.Series([True, True, False, True, True], index=idx)
        result = _and_combine([s1, s2], idx)
        # AND: TT, TT, TF, FT, TT -> T, T, F, F, T = 3
        assert result.sum() == 3


class TestAssembleSignals:
    def test_assembles_entry_signals(self) -> None:
        spec = _make_spec()
        n = 20
        idx = _make_index(n)
        ohlcv = pd.DataFrame(
            {
                "open": np.ones(n) * 100,
                "high": np.ones(n) * 101,
                "low": np.ones(n) * 99,
                "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        # Add entry rule output
        entry_df = pd.DataFrame(
            {
                "crossover_long_entry": [False] * 5 + [True] + [False] * 14,
                "crossover_short_entry": [False] * 10 + [True] + [False] * 9,
            },
            index=idx,
        )
        compiled.block_outputs["crossover_entry"] = BlockOutput(
            "crossover_entry", "entry_rule", {}, entry_df
        )

        # Add minimal other outputs
        compiled.block_outputs["fixed_tpsl"] = BlockOutput(
            "fixed_tpsl", "exit_rule", {},
            pd.DataFrame({"tpsl_long_sl": np.ones(n) * 98}, index=idx),
        )
        compiled.block_outputs["fixed_risk"] = BlockOutput(
            "fixed_risk", "money_management", {},
            pd.DataFrame({"fr_position_size": np.ones(n) * 100}, index=idx),
        )

        result = assemble_signals(compiled)

        assert result.entry_long is not None
        assert result.entry_long.sum() == 1
        assert result.entry_short is not None
        assert result.entry_short.sum() == 1

    def test_assembles_filter_masks(self) -> None:
        spec = _make_spec()
        spec = StrategySpec(
            name="filter_test",
            description="Testing filter assembly with trend filter and session filter blocks.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema")],
            entry_rules=[BlockSpec(block_name="crossover_entry")],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            filters=[BlockSpec(block_name="trend_filter")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )
        n = 20
        idx = _make_index(n)
        ohlcv = pd.DataFrame(
            {
                "open": np.ones(n) * 100, "high": np.ones(n) * 101,
                "low": np.ones(n) * 99, "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        # Filter output
        filter_df = pd.DataFrame(
            {
                "trend_allow_long": [True] * 10 + [False] * 10,
                "trend_allow_short": [False] * 10 + [True] * 10,
            },
            index=idx,
        )
        compiled.block_outputs["trend_filter"] = BlockOutput(
            "trend_filter", "filter", {}, filter_df
        )

        result = assemble_signals(compiled)

        assert result.allow_long is not None
        assert result.allow_long.sum() == 10
        assert result.allow_short is not None
        assert result.allow_short.sum() == 10

    def test_assembles_tpsl(self) -> None:
        spec = _make_spec()
        n = 10
        idx = _make_index(n)
        ohlcv = pd.DataFrame(
            {
                "open": np.ones(n) * 100, "high": np.ones(n) * 101,
                "low": np.ones(n) * 99, "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        tpsl_df = pd.DataFrame(
            {
                "tpsl_long_tp": np.ones(n) * 105,
                "tpsl_long_sl": np.ones(n) * 97,
                "tpsl_short_tp": np.ones(n) * 95,
                "tpsl_short_sl": np.ones(n) * 103,
            },
            index=idx,
        )
        compiled.block_outputs["fixed_tpsl"] = BlockOutput(
            "fixed_tpsl", "exit_rule", {}, tpsl_df
        )
        compiled.block_outputs["fixed_risk"] = BlockOutput(
            "fixed_risk", "money_management", {},
            pd.DataFrame({"fr_position_size": np.ones(n)}, index=idx),
        )

        result = assemble_signals(compiled)

        assert result.stop_loss_long is not None
        assert result.take_profit_long is not None
        assert result.stop_loss_long.iloc[0] == 97.0
        assert result.take_profit_long.iloc[0] == 105.0
```

---

### `tests/unit/compiler/test_compiler.py`

```python
"""Tests for the StrategyCompiler."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.exceptions import StrategyCompileError


def _populate_registry(registry: BlockRegistry) -> None:
    for mod_name in [
        "forgequant.blocks.indicators",
        "forgequant.blocks.entry_rules",
        "forgequant.blocks.exit_rules",
        "forgequant.blocks.money_management",
        "forgequant.blocks.filters",
        "forgequant.blocks.price_action",
    ]:
        module = importlib.import_module(mod_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "metadata") and attr_name != "BaseBlock":
                try:
                    registry.register_class(attr)
                except Exception:
                    pass


@pytest.fixture
def full_registry(clean_registry: BlockRegistry) -> BlockRegistry:
    _populate_registry(clean_registry)
    return clean_registry


@pytest.fixture
def compiler(full_registry: BlockRegistry) -> StrategyCompiler:
    return StrategyCompiler(registry=full_registry, validate=True)


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="compiler_test_strategy",
        description="A trend following EMA crossover strategy for compiler testing purposes.",
        objective={"style": "trend_following", "timeframe": "1h"},
        indicators=[
            BlockSpec(block_name="ema", params={"period": 20}),
            BlockSpec(block_name="atr", params={"period": 14}),
        ],
        entry_rules=[
            BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20}),
        ],
        exit_rules=[
            BlockSpec(block_name="fixed_tpsl", params={"atr_period": 14}),
        ],
        money_management=BlockSpec(block_name="fixed_risk", params={"atr_period": 14}),
    )


class TestStrategyCompiler:
    def test_compile_basic_spec(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.spec.name == "compiler_test_strategy"
        assert compiled.n_bars == len(sample_ohlcv)
        assert "ema" in compiled.block_outputs
        assert "atr" in compiled.block_outputs
        assert "crossover_entry" in compiled.block_outputs
        assert "fixed_tpsl" in compiled.block_outputs
        assert "fixed_risk" in compiled.block_outputs

    def test_compile_produces_signals(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.entry_long is not None
        assert compiled.entry_short is not None
        assert compiled.exit_long is not None
        assert compiled.exit_short is not None
        assert len(compiled.entry_long) == len(sample_ohlcv)

    def test_compile_with_filters(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = StrategySpec(
            name="filtered_strategy",
            description="Strategy with trend filter for compiler testing with adequate description.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 20})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            filters=[BlockSpec(block_name="trend_filter", params={"period": 50})],
            money_management=BlockSpec(block_name="fixed_risk"),
        )
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.allow_long is not None
        assert compiled.allow_short is not None
        # Filtered entries should be <= raw entries
        raw_long = compiled.entry_long.sum() if compiled.entry_long is not None else 0
        filtered_long = compiled.filtered_entry_long().sum()
        assert filtered_long <= raw_long

    def test_compile_has_tpsl(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.stop_loss_long is not None
        assert compiled.take_profit_long is not None

    def test_compile_has_position_sizing(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.position_size_long is not None
        valid = compiled.position_size_long.dropna()
        assert (valid > 0).all()

    def test_compile_invalid_spec_raises(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = StrategySpec(
            name="invalid_block_strategy",
            description="Strategy referencing a nonexistent block name for testing error handling.",
            objective={"style": "breakout", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="nonexistent_indicator_xyz")],
            entry_rules=[BlockSpec(block_name="crossover_entry")],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )
        with pytest.raises(StrategyCompileError):
            compiler.compile(spec, sample_ohlcv)

    def test_compile_empty_data_raises(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = _make_spec()
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([]),
        )
        with pytest.raises(StrategyCompileError, match="Invalid OHLCV"):
            compiler.compile(spec, empty)

    def test_compile_skip_validation(
        self, full_registry: BlockRegistry, sample_ohlcv: pd.DataFrame
    ) -> None:
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)
        assert compiled.n_bars == len(sample_ohlcv)

    def test_compile_summary(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)
        summary = compiled.summary()

        assert summary["strategy_name"] == "compiler_test_strategy"
        assert summary["n_bars"] == len(sample_ohlcv)
        assert "raw_long_entries" in summary
        assert "filtered_long_entries" in summary
```

---

## 6.11 Test Suite — Engine

### `tests/unit/engine/__init__.py`

```python
"""Tests for backtesting engine."""
```

---

### `tests/unit/engine/test_results.py`

```python
"""Tests for backtest results containers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from forgequant.core.engine.results import BacktestResult, TradeRecord


class TestTradeRecord:
    def test_winner(self) -> None:
        t = TradeRecord(
            trade_id=0, direction="long",
            entry_time=datetime(2024, 1, 1), entry_price=100.0,
            exit_time=datetime(2024, 1, 2), exit_price=110.0,
            exit_reason="tp", position_size=10.0,
            pnl=10.0, pnl_pct=10.0, pnl_dollar=100.0,
            bars_held=5, mae=-2.0, mfe=12.0,
        )
        assert t.is_winner is True

    def test_loser(self) -> None:
        t = TradeRecord(
            trade_id=1, direction="short",
            entry_time=datetime(2024, 1, 1), entry_price=100.0,
            exit_time=datetime(2024, 1, 2), exit_price=105.0,
            exit_reason="sl", position_size=10.0,
            pnl=-5.0, pnl_pct=-5.0, pnl_dollar=-50.0,
            bars_held=3, mae=-7.0, mfe=2.0,
        )
        assert t.is_winner is False

    def test_risk_reward_achieved(self) -> None:
        t = TradeRecord(
            trade_id=0, direction="long",
            entry_time=datetime(2024, 1, 1), entry_price=100.0,
            exit_time=datetime(2024, 1, 2), exit_price=110.0,
            exit_reason="tp", position_size=1.0,
            pnl=10.0, pnl_pct=10.0, pnl_dollar=10.0,
            bars_held=5, mae=-5.0, mfe=12.0,
        )
        assert abs(t.risk_reward_achieved - 2.4) < 0.01


class TestBacktestResult:
    def test_basic_properties(self) -> None:
        equity = pd.Series([100000, 100500, 101000], name="equity")
        r = BacktestResult(
            strategy_name="test",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_equity=100000,
            final_equity=101000,
            equity_curve=equity,
        )
        assert r.n_trades == 0
        assert r.total_pnl == 0.0

    def test_with_trades(self) -> None:
        equity = pd.Series([100000, 101000])
        winner = TradeRecord(
            trade_id=0, direction="long",
            entry_time=datetime(2024, 1, 1), entry_price=100,
            exit_time=datetime(2024, 1, 2), exit_price=110,
            exit_reason="tp", position_size=10,
            pnl=10.0, pnl_pct=10.0, pnl_dollar=100,
            bars_held=5,
        )
        loser = TradeRecord(
            trade_id=1, direction="short",
            entry_time=datetime(2024, 1, 2), entry_price=110,
            exit_time=datetime(2024, 1, 3), exit_price=115,
            exit_reason="sl", position_size=10,
            pnl=-5.0, pnl_pct=-4.5, pnl_dollar=-50,
            bars_held=3,
        )
        r = BacktestResult(
            strategy_name="test",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_equity=100000,
            final_equity=100050,
            equity_curve=equity,
            trades=[winner, loser],
        )
        assert r.n_trades == 2
        assert len(r.winning_trades) == 1
        assert len(r.losing_trades) == 1
        assert r.total_pnl == 50.0

    def test_trades_to_dataframe(self) -> None:
        trade = TradeRecord(
            trade_id=0, direction="long",
            entry_time=datetime(2024, 1, 1), entry_price=100,
            exit_time=datetime(2024, 1, 2), exit_price=110,
            exit_reason="tp", position_size=1,
            pnl=10.0, pnl_pct=10.0, pnl_dollar=10,
            bars_held=5,
        )
        r = BacktestResult(
            strategy_name="test",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_equity=100000,
            final_equity=100010,
            equity_curve=pd.Series([100000, 100010]),
            trades=[trade],
        )
        df = r.trades_to_dataframe()
        assert len(df) == 1
        assert "is_winner" in df.columns
        assert df["is_winner"].iloc[0] is True

    def test_summary(self) -> None:
        r = BacktestResult(
            strategy_name="test",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_equity=100000,
            final_equity=120000,
            equity_curve=pd.Series([100000, 120000]),
            metrics={"sharpe_ratio": 1.5},
        )
        s = r.summary()
        assert s["total_return_pct"] == 20.0
        assert s["sharpe_ratio"] == 1.5
```

---

### `tests/unit/engine/test_metrics.py`

```python
"""Tests for performance metrics calculator."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from forgequant.core.engine.metrics import compute_metrics, _max_consecutive, _max_drawdown_duration
from forgequant.core.engine.results import BacktestResult, TradeRecord


def _make_result(
    equity_values: list[float],
    trades: list[TradeRecord] | None = None,
) -> BacktestResult:
    n = len(equity_values)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    equity = pd.Series(equity_values, index=dates, name="equity")
    return BacktestResult(
        strategy_name="test",
        start_date=dates[0].to_pydatetime(),
        end_date=dates[-1].to_pydatetime(),
        initial_equity=equity_values[0],
        final_equity=equity_values[-1],
        equity_curve=equity,
        trades=trades or [],
    )


def _make_trade(
    pnl_dollar: float,
    direction: str = "long",
    bars_held: int = 5,
) -> TradeRecord:
    return TradeRecord(
        trade_id=0, direction=direction,
        entry_time=datetime(2024, 1, 1), entry_price=100,
        exit_time=datetime(2024, 1, 2), exit_price=100 + pnl_dollar,
        exit_reason="signal", position_size=1.0,
        pnl=pnl_dollar, pnl_pct=pnl_dollar,
        pnl_dollar=pnl_dollar, bars_held=bars_held,
    )


class TestComputeMetrics:
    def test_basic_uptrend(self) -> None:
        equity = [100000 + i * 100 for i in range(100)]
        result = compute_metrics(_make_result(equity))

        assert result.metrics["total_return_pct"] > 0
        assert result.metrics["max_drawdown_pct"] == 0.0  # Pure uptrend
        assert result.returns_series is not None
        assert result.drawdown_series is not None

    def test_with_drawdown(self) -> None:
        # Up then down then up
        equity = list(np.concatenate([
            np.linspace(100000, 120000, 50),
            np.linspace(120000, 100000, 50),
        ]))
        result = compute_metrics(_make_result(equity))

        assert result.metrics["max_drawdown_pct"] > 0
        assert result.drawdown_series is not None
        assert result.drawdown_series.min() < 0

    def test_with_trades(self) -> None:
        trades = [
            _make_trade(100),    # Winner
            _make_trade(-50),    # Loser
            _make_trade(200),    # Winner
            _make_trade(-30),    # Loser
            _make_trade(80),     # Winner
        ]
        equity = [100000 + i * 50 for i in range(100)]
        result = compute_metrics(_make_result(equity, trades))

        assert result.metrics["total_trades"] == 5
        assert result.metrics["win_rate"] == 3 / 5
        assert result.metrics["loss_rate"] == 2 / 5
        assert result.metrics["profit_factor"] > 0
        assert result.metrics["avg_bars_held"] == 5.0

    def test_no_trades(self) -> None:
        equity = [100000] * 50
        result = compute_metrics(_make_result(equity))

        assert result.metrics["total_trades"] == 0
        assert result.metrics["win_rate"] == 0

    def test_all_winners(self) -> None:
        trades = [_make_trade(100) for _ in range(5)]
        equity = [100000 + i * 100 for i in range(50)]
        result = compute_metrics(_make_result(equity, trades))

        assert result.metrics["win_rate"] == 1.0
        assert result.metrics["loss_rate"] == 0.0

    def test_all_losers(self) -> None:
        trades = [_make_trade(-50) for _ in range(5)]
        equity = list(np.linspace(100000, 99750, 50))
        result = compute_metrics(_make_result(equity, trades))

        assert result.metrics["win_rate"] == 0.0
        assert result.metrics["profit_factor"] == 0.0

    def test_sharpe_positive_return(self) -> None:
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        equity = 100000 * np.exp(np.cumsum(returns))
        result = compute_metrics(_make_result(equity.tolist()))

        assert result.metrics["sharpe_ratio"] > 0

    def test_long_short_breakdown(self) -> None:
        trades = [
            _make_trade(100, "long"),
            _make_trade(-50, "short"),
            _make_trade(80, "long"),
        ]
        equity = [100000 + i * 50 for i in range(50)]
        result = compute_metrics(_make_result(equity, trades))

        assert result.metrics["long_trades"] == 2
        assert result.metrics["short_trades"] == 1
        assert result.metrics["long_win_rate"] == 1.0
        assert result.metrics["short_win_rate"] == 0.0


class TestMaxConsecutive:
    def test_all_winners(self) -> None:
        trades = [_make_trade(100) for _ in range(5)]
        assert _max_consecutive(trades, winners=True) == 5

    def test_alternating(self) -> None:
        trades = [
            _make_trade(100), _make_trade(-50),
            _make_trade(100), _make_trade(-50),
        ]
        assert _max_consecutive(trades, winners=True) == 1
        assert _max_consecutive(trades, winners=False) == 1

    def test_streak(self) -> None:
        trades = [
            _make_trade(100), _make_trade(100), _make_trade(100),
            _make_trade(-50),
            _make_trade(100), _make_trade(100),
        ]
        assert _max_consecutive(trades, winners=True) == 3

    def test_empty(self) -> None:
        assert _max_consecutive([], winners=True) == 0


class TestMaxDrawdownDuration:
    def test_no_drawdown(self) -> None:
        equity = pd.Series([100, 101, 102, 103])
        assert _max_drawdown_duration(equity) == 0

    def test_single_drawdown(self) -> None:
        equity = pd.Series([100, 105, 103, 102, 106])
        # Drawdown bars: index 2, 3 (2 bars)
        assert _max_drawdown_duration(equity) == 2

    def test_long_drawdown(self) -> None:
        equity = pd.Series([100, 105, 104, 103, 102, 101, 106])
        # Drawdown from bar 2 to bar 5 (4 bars)
        assert _max_drawdown_duration(equity) == 4
```

---

### `tests/unit/engine/test_backtester.py`

```python
"""Tests for the backtesting engine."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.engine.backtester import Backtester, BacktestConfig


def _populate_registry(registry: BlockRegistry) -> None:
    for mod_name in [
        "forgequant.blocks.indicators",
        "forgequant.blocks.entry_rules",
        "forgequant.blocks.exit_rules",
        "forgequant.blocks.money_management",
        "forgequant.blocks.filters",
        "forgequant.blocks.price_action",
    ]:
        module = importlib.import_module(mod_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "metadata") and attr_name != "BaseBlock":
                try:
                    registry.register_class(attr)
                except Exception:
                    pass


@pytest.fixture
def full_registry(clean_registry: BlockRegistry) -> BlockRegistry:
    _populate_registry(clean_registry)
    return clean_registry


@pytest.fixture
def compiler(full_registry: BlockRegistry) -> StrategyCompiler:
    return StrategyCompiler(registry=full_registry, validate=True)


def _make_trending_data(n: int = 500, trend: str = "up") -> pd.DataFrame:
    """Generate trending OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    if trend == "up":
        close = 100.0 + np.cumsum(np.random.normal(0.05, 0.3, n))
    elif trend == "down":
        close = 200.0 + np.cumsum(np.random.normal(-0.05, 0.3, n))
    else:
        close = 100.0 + np.cumsum(np.random.normal(0.0, 0.5, n))

    close = np.maximum(close, 50.0)
    spread = np.random.uniform(0.1, 0.5, n)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )


class TestBacktestConfig:
    def test_defaults(self) -> None:
        cfg = BacktestConfig()
        assert cfg.initial_equity == 100_000.0
        assert cfg.commission_per_unit == 0.0
        assert cfg.slippage_pct == 0.0


class TestBacktester:
    def test_basic_backtest(
        self, compiler: StrategyCompiler
    ) -> None:
        """A basic strategy should produce a valid backtest result."""
        spec = StrategySpec(
            name="basic_backtest_test",
            description="A simple EMA crossover strategy for backtesting engine integration testing.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[
                BlockSpec(block_name="ema", params={"period": 20}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            entry_rules=[
                BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20}),
            ],
            exit_rules=[
                BlockSpec(block_name="fixed_tpsl", params={"atr_period": 14, "tp_atr_mult": 3.0, "sl_atr_mult": 1.5}),
            ],
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0}),
        )

        data = _make_trending_data(500, "up")
        compiled = compiler.compile(spec, data)

        backtester = Backtester(BacktestConfig(initial_equity=100000))
        result = backtester.run(compiled)

        assert result.strategy_name == "basic_backtest_test"
        assert result.initial_equity == 100000
        assert len(result.equity_curve) == 500
        assert result.metrics is not None
        assert "total_return_pct" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown_pct" in result.metrics
        assert "total_trades" in result.metrics

    def test_produces_trades(
        self, compiler: StrategyCompiler
    ) -> None:
        """With trending data, the strategy should generate trades."""
        spec = StrategySpec(
            name="trade_generator_test",
            description="Strategy designed to generate trades for testing trade recording behavior.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 10})],
            entry_rules=[
                BlockSpec(block_name="crossover_entry", params={"fast_period": 5, "slow_period": 10}),
            ],
            exit_rules=[
                BlockSpec(block_name="fixed_tpsl", params={"atr_period": 5, "tp_atr_mult": 2.0, "sl_atr_mult": 1.0}),
            ],
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0, "atr_period": 5}),
        )

        data = _make_trending_data(500, "up")
        compiled = compiler.compile(spec, data)

        backtester = Backtester(BacktestConfig(initial_equity=100000))
        result = backtester.run(compiled)

        assert result.n_trades > 0
        # Check trade records are properly formed
        for trade in result.trades:
            assert trade.direction in ("long", "short")
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.bars_held >= 0
            assert trade.exit_reason in ("tp", "sl", "signal", "end")

    def test_equity_curve_starts_at_initial(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="equity_start_test",
            description="Testing that equity curve starts at initial equity for verification.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 50})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 20, "slow_period": 50})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )

        data = _make_trending_data(300)
        compiled = compiler.compile(spec, data)

        cfg = BacktestConfig(initial_equity=50000)
        result = Backtester(cfg).run(compiled)

        assert result.equity_curve.iloc[0] == 50000

    def test_with_commission(
        self, compiler: StrategyCompiler
    ) -> None:
        """Commission should reduce final equity."""
        spec = StrategySpec(
            name="commission_test",
            description="Testing the impact of commissions on final equity in backtesting results.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 10})],
            entry_rules=[
                BlockSpec(block_name="crossover_entry", params={"fast_period": 5, "slow_period": 10}),
            ],
            exit_rules=[
                BlockSpec(block_name="fixed_tpsl", params={"atr_period": 5, "tp_atr_mult": 2.0, "sl_atr_mult": 1.0}),
            ],
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0, "atr_period": 5}),
        )

        data = _make_trending_data(500, "up")
        compiled = compiler.compile(spec, data)

        result_no_comm = Backtester(BacktestConfig(commission_per_unit=0.0)).run(compiled)
        result_with_comm = Backtester(BacktestConfig(commission_per_unit=1.0)).run(compiled)

        if result_no_comm.n_trades > 0:
            assert result_with_comm.final_equity <= result_no_comm.final_equity

    def test_metrics_populated(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="metrics_test",
            description="Verifying all performance metrics are computed and populated after backtesting.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 10})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 5, "slow_period": 10})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl", params={"atr_period": 5})],
            money_management=BlockSpec(block_name="fixed_risk", params={"atr_period": 5}),
        )

        data = _make_trending_data(500)
        compiled = compiler.compile(spec, data)
        result = Backtester().run(compiled)

        required_metrics = [
            "total_return_pct", "annualized_return_pct",
            "max_drawdown_pct", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "total_trades", "win_rate",
            "profit_factor", "expectancy_dollar",
            "avg_bars_held", "long_trades", "short_trades",
        ]
        for metric in required_metrics:
            assert metric in result.metrics, f"Missing metric: {metric}"

    def test_result_summary(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="summary_test",
            description="Testing the summary output format of backtest results for completeness.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 20})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )

        data = _make_trending_data(300)
        compiled = compiler.compile(spec, data)
        result = Backtester().run(compiled)

        summary = result.summary()
        assert "strategy_name" in summary
        assert "total_return_pct" in summary
        assert "n_trades" in summary
```

---

## 6.12 Integration Test — End-to-End

### `tests/integration/test_phase6_end_to_end.py`

```python
"""
End-to-end integration test for the complete Phase 6 pipeline:
StrategySpec → Compiler → Backtester → BacktestResult with metrics.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.ai_forge.validator import SpecValidator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.engine.backtester import Backtester, BacktestConfig


def _populate_registry(registry: BlockRegistry) -> None:
    for mod_name in [
        "forgequant.blocks.indicators",
        "forgequant.blocks.entry_rules",
        "forgequant.blocks.exit_rules",
        "forgequant.blocks.money_management",
        "forgequant.blocks.filters",
        "forgequant.blocks.price_action",
    ]:
        module = importlib.import_module(mod_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "metadata") and attr_name != "BaseBlock":
                try:
                    registry.register_class(attr)
                except Exception:
                    pass


@pytest.fixture
def full_registry(clean_registry: BlockRegistry) -> BlockRegistry:
    _populate_registry(clean_registry)
    return clean_registry


def _make_data(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100.0 + np.cumsum(np.random.normal(0.02, 0.5, n))
    close = np.maximum(close, 50.0)
    spread = np.random.uniform(0.1, 0.5, n)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )


class TestEMACrossoverEndToEnd:
    """Full pipeline test for a classic EMA crossover strategy."""

    def test_full_pipeline(self, full_registry: BlockRegistry) -> None:
        # 1. Define strategy
        spec = StrategySpec(
            name="ema_crossover_e2e",
            description="End to end EMA crossover trend following strategy for integration testing.",
            objective={
                "style": "trend_following",
                "timeframe": "1h",
                "instruments": ["EURUSD"],
            },
            indicators=[
                BlockSpec(block_name="ema", params={"period": 20}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            entry_rules=[
                BlockSpec(
                    block_name="crossover_entry",
                    params={"fast_period": 10, "slow_period": 20},
                ),
            ],
            exit_rules=[
                BlockSpec(
                    block_name="fixed_tpsl",
                    params={"tp_atr_mult": 3.0, "sl_atr_mult": 1.5, "atr_period": 14},
                ),
                BlockSpec(
                    block_name="trailing_stop",
                    params={"atr_period": 14, "trail_atr_mult": 2.5},
                ),
            ],
            filters=[
                BlockSpec(block_name="trend_filter", params={"period": 50}),
            ],
            money_management=BlockSpec(
                block_name="fixed_risk",
                params={"risk_pct": 1.0, "sl_atr_mult": 1.5, "atr_period": 14},
            ),
        )

        # 2. Validate
        validator = SpecValidator(full_registry)
        val_result = validator.validate(spec)
        assert val_result.is_valid, f"Validation errors: {val_result.errors}"

        # 3. Compile
        data = _make_data(500)
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        compiled = compiler.compile(spec, data, val_result.validated_params)

        # Verify compilation
        assert compiled.entry_long is not None
        assert compiled.entry_short is not None
        assert compiled.stop_loss_long is not None
        assert compiled.take_profit_long is not None
        assert compiled.allow_long is not None
        assert compiled.position_size_long is not None

        # 4. Backtest
        config = BacktestConfig(initial_equity=100_000)
        backtester = Backtester(config)
        result = backtester.run(compiled)

        # 5. Verify result
        assert result.strategy_name == "ema_crossover_e2e"
        assert result.initial_equity == 100_000
        assert len(result.equity_curve) == 500
        assert result.drawdown_series is not None
        assert result.returns_series is not None

        # Metrics should be fully populated
        assert "total_return_pct" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown_pct" in result.metrics
        assert "total_trades" in result.metrics
        assert "win_rate" in result.metrics
        assert "profit_factor" in result.metrics

        # Summary should work
        summary = result.summary()
        assert isinstance(summary, dict)
        assert "strategy_name" in summary


class TestMeanReversionEndToEnd:
    """Full pipeline test for a mean-reversion strategy."""

    def test_full_pipeline(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="rsi_mean_reversion_e2e",
            description="End to end RSI mean reversion strategy for integration testing verification.",
            objective={
                "style": "mean_reversion",
                "timeframe": "1h",
            },
            indicators=[
                BlockSpec(block_name="rsi", params={"period": 14}),
                BlockSpec(block_name="bollinger_bands", params={"period": 20}),
            ],
            entry_rules=[
                BlockSpec(
                    block_name="threshold_cross_entry",
                    params={"mode": "mean_reversion"},
                ),
            ],
            exit_rules=[
                BlockSpec(
                    block_name="fixed_tpsl",
                    params={"tp_atr_mult": 2.0, "sl_atr_mult": 1.0},
                ),
                BlockSpec(
                    block_name="time_based_exit",
                    params={"max_bars": 30},
                ),
            ],
            money_management=BlockSpec(
                block_name="atr_based_sizing",
                params={"risk_pct": 1.0},
            ),
        )

        validator = SpecValidator(full_registry)
        val_result = validator.validate(spec)
        assert val_result.is_valid, f"Errors: {val_result.errors}"

        data = _make_data(500)
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        compiled = compiler.compile(spec, data, val_result.validated_params)

        result = Backtester(BacktestConfig(initial_equity=100_000)).run(compiled)

        assert result.n_trades >= 0
        assert "total_return_pct" in result.metrics
        assert result.equity_curve.iloc[0] == 100_000


class TestBreakoutEndToEnd:
    """Full pipeline test for a breakout strategy with price action."""

    def test_full_pipeline(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="breakout_e2e",
            description="End to end breakout strategy using price action and confluence entry for testing.",
            objective={"style": "breakout", "timeframe": "1h"},
            indicators=[
                BlockSpec(block_name="adx", params={"period": 14}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            price_action=[
                BlockSpec(block_name="breakout", params={"lookback": 20}),
            ],
            entry_rules=[
                BlockSpec(block_name="confluence_entry", params={"trend_period": 50}),
            ],
            exit_rules=[
                BlockSpec(block_name="trailing_stop", params={"trail_atr_mult": 3.0}),
                BlockSpec(block_name="breakeven_stop", params={"activation_atr_mult": 2.0}),
            ],
            filters=[
                BlockSpec(block_name="max_drawdown_filter", params={"max_drawdown_pct": 15.0}),
            ],
            money_management=BlockSpec(
                block_name="volatility_targeting",
                params={"target_vol": 0.15},
            ),
        )

        validator = SpecValidator(full_registry)
        val_result = validator.validate(spec)
        assert val_result.is_valid, f"Errors: {val_result.errors}"

        data = _make_data(500)
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        compiled = compiler.compile(spec, data, val_result.validated_params)

        result = Backtester().run(compiled)

        assert "total_return_pct" in result.metrics
        assert "max_drawdown_pct" in result.metrics
        # Breakout block output should be in compiled
        assert "breakout" in compiled.block_outputs
```

---

## 6.13 How to Verify Phase 6

```bash
# From project root with venv activated

# Run all tests
pytest -v

# Run only compiler tests
pytest tests/unit/compiler/ -v

# Run only engine tests
pytest tests/unit/engine/ -v

# Run the integration test
pytest tests/integration/test_phase6_end_to_end.py -v

# Type-check
mypy src/forgequant/core/compiler/ src/forgequant/core/engine/

# Lint
ruff check src/forgequant/core/compiler/ src/forgequant/core/engine/
```

**Expected output:** All tests pass — approximately **60+ new tests** across 7 test modules plus **3 parametrized end-to-end integration tests**.

---

## Phase 6 Summary

### Module Overview

| Module | File | Purpose |
|--------|------|---------|
| **CompiledStrategy** | `compiled_strategy.py` | Data container holding block outputs + assembled signal matrices |
| **SignalAssembler** | `signal_assembler.py` | Combines entry/exit/filter signals: OR for entries/exits, AND for filters |
| **StrategyCompiler** | `compiler.py` | Orchestrates block execution in dependency order → assembled signals |
| **BacktestResult** | `results.py` | Holds equity curve, trade list, metrics; provides DataFrame export |
| **Metrics** | `metrics.py` | 25+ performance metrics: Sharpe, Sortino, Calmar, profit factor, etc. |
| **Backtester** | `backtester.py` | Bar-by-bar vectorized simulation with TP/SL, next-bar execution |

### Architecture Flow

```
StrategySpec (validated)
        │
        ▼
┌─────────────────────┐
│  StrategyCompiler    │
│  ┌───────────────┐  │
│  │ For each block│  │
│  │ in exec order:│  │
│  │  1. Indicators│  │
│  │  2. PriceAct  │  │
│  │  3. Entry     │  │
│  │  4. Exit      │  │
│  │  5. Filters   │  │
│  │  6. MoneyMgmt │  │
│  └───────┬───────┘  │
│          ▼          │
│  ┌───────────────┐  │
│  │ SignalAssembler│  │
│  │  OR entries   │  │
│  │  OR exits     │  │
│  │  AND filters  │  │
│  └───────┬───────┘  │
└──────────┼──────────┘
           │
           ▼
   CompiledStrategy
    (signal matrices)
           │
           ▼
┌──────────────────────┐
│     Backtester       │
│  ┌────────────────┐  │
│  │ Bar-by-bar:    │  │
│  │ 1. Check exits │  │
│  │    (TP/SL/sig) │  │
│  │ 2. Check entry │  │
│  │    (next-bar)  │  │
│  │ 3. Track equity│  │
│  └────────┬───────┘  │
│           ▼          │
│  ┌────────────────┐  │
│  │ compute_metrics│  │
│  └────────┬───────┘  │
└───────────┼──────────┘
            │
            ▼
     BacktestResult
   (equity, trades, 25+ metrics)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Next-bar execution** | Signal on bar N → entry at open[N+1] prevents lookahead |
| **TP/SL from signal bar** | TP/SL levels computed at signal time, not entry time, matching how a trader would place orders |
| **OR for entries, AND for filters** | Multiple entry conditions increase signals; all filters must agree to permit trading |
| **Single position per direction** | No pyramiding by default — avoids complexity and overexposure |
| **Commission on entry AND exit** | Both legs incur transaction costs for realistic simulation |
| **MAE/MFE per trade** | Tracks worst drawdown and best unrealized profit per trade for quality analysis |
| **Hysteresis-free metrics** | Metrics computed from the equity curve directly, independent of trade-level analysis |

### Performance Metrics Computed (25+)

| Category | Metrics |
|----------|---------|
| **Return** | Total return %, annualized return % |
| **Risk** | Max drawdown %, max drawdown duration |
| **Risk-adjusted** | Sharpe ratio, Sortino ratio, Calmar ratio |
| **Volatility** | Annualized volatility % |
| **Trade stats** | Total trades, win rate, loss rate, avg P&L, avg winner/loser |
| **Quality** | Profit factor, payoff ratio, expectancy |
| **Holding** | Avg bars held, max bars held |
| **Streaks** | Max consecutive wins, max consecutive losses |
| **Direction** | Long/short trade counts, long/short win rates |

### Cumulative Project Status

| Phase | Status | Components |
|-------|--------|------------|
| Phase 1 | ✅ | Foundation |
| Phase 2 | ✅ | 8 indicator blocks |
| Phase 3 | ✅ | 4 price action + 4 entry rule blocks |
| Phase 4 | ✅ | 4 exit + 4 money mgmt + 4 filter blocks |
| Phase 5 | ✅ | AI Forge — schemas, validator, prompt, RAG, pipeline |
| **Phase 6** | ✅ | **Compiler + backtesting engine + 25+ metrics** |
| Phase 7 | 🔜 | Robustness suite (walk-forward, Monte Carlo, CPCV) |
| Phase 8 | 🔜 | Execution layer (MT5) |
| Phase 9 | 🔜 | Reflex dashboard |

**Total blocks: 28** | **Total source files: ~60** | **Total test files: ~35**

---

**Ready for Phase 7** — say the word and I'll write the Robustness Suite: walk-forward analysis, Monte Carlo permutation testing, Combinatorial Purged Cross-Validation (CPCV), parameter sensitivity analysis, and equity curve stability checks, all with comprehensive test coverage.
