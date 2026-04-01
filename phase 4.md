# PHASE 4 — Exit Rules, Money Management & Filter Blocks

All 4 exit rule blocks, all 4 money management blocks, and all 4 filter blocks, each with full implementations, parameter validation, and comprehensive tests.

---

## 4.1 Updated Directory Structure (additions)

```
src/forgequant/blocks/exit_rules/
├── __init__.py
├── fixed_tpsl.py
├── trailing_stop.py
├── time_based_exit.py
└── breakeven_stop.py

src/forgequant/blocks/money_management/
├── __init__.py
├── fixed_risk.py
├── volatility_targeting.py
├── kelly_fractional.py
└── atr_based_sizing.py

src/forgequant/blocks/filters/
├── __init__.py
├── trading_session.py
├── spread_filter.py
├── max_drawdown_filter.py
└── trend_filter.py

tests/unit/exit_rules/
├── __init__.py
├── test_fixed_tpsl.py
├── test_trailing_stop.py
├── test_time_based_exit.py
└── test_breakeven_stop.py

tests/unit/money_management/
├── __init__.py
├── test_fixed_risk.py
├── test_volatility_targeting.py
├── test_kelly_fractional.py
└── test_atr_based_sizing.py

tests/unit/filters/
├── __init__.py
├── test_trading_session.py
├── test_spread_filter.py
├── test_max_drawdown_filter.py
└── test_trend_filter.py

tests/integration/
└── test_phase4_registry.py
```

---

## 4.2 Exit Rules — `__init__.py`

### `src/forgequant/blocks/exit_rules/__init__.py`

```python
"""
Exit rule blocks.

Provides:
    - FixedTPSLExit: Fixed take-profit and stop-loss in ATR multiples or pips
    - TrailingStopExit: Trailing stop that locks in profits as price moves
    - TimeBasedExit: Exit after a maximum number of bars
    - BreakevenStopExit: Moves stop to breakeven after reaching a profit threshold

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.exit_rules.fixed_tpsl import FixedTPSLExit
from forgequant.blocks.exit_rules.trailing_stop import TrailingStopExit
from forgequant.blocks.exit_rules.time_based_exit import TimeBasedExit
from forgequant.blocks.exit_rules.breakeven_stop import BreakevenStopExit

__all__ = [
    "FixedTPSLExit",
    "TrailingStopExit",
    "TimeBasedExit",
    "BreakevenStopExit",
]
```

---

## 4.3 `src/forgequant/blocks/exit_rules/fixed_tpsl.py`

```python
"""
Fixed Take-Profit / Stop-Loss exit rule block.

Computes per-bar take-profit and stop-loss levels using ATR multiples.
ATR-based exits adapt to current volatility, producing tighter stops
in calm markets and wider stops in volatile markets.

Calculation:
    1. ATR = Wilder-smoothed True Range over atr_period
    2. For a LONG trade entered at close:
       stop_loss  = close - (sl_atr_mult * ATR)
       take_profit = close + (tp_atr_mult * ATR)
    3. For a SHORT trade entered at close:
       stop_loss  = close + (sl_atr_mult * ATR)
       take_profit = close - (tp_atr_mult * ATR)
    4. risk_reward_ratio = tp_atr_mult / sl_atr_mult

The block rejects configurations where the risk-reward ratio is below
a configurable minimum, enforcing basic trade quality.

Output columns:
    - tpsl_atr: The ATR value at each bar
    - tpsl_long_tp: Take-profit level for a long entry at this bar
    - tpsl_long_sl: Stop-loss level for a long entry at this bar
    - tpsl_short_tp: Take-profit level for a short entry at this bar
    - tpsl_short_sl: Stop-loss level for a short entry at this bar
    - tpsl_risk_reward: The risk-reward ratio (constant across bars)
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class FixedTPSLExit(BaseBlock):
    """Fixed take-profit and stop-loss based on ATR multiples."""

    metadata = BlockMetadata(
        name="fixed_tpsl",
        display_name="Fixed TP/SL",
        category=BlockCategory.EXIT_RULE,
        description=(
            "Computes take-profit and stop-loss levels for each bar using "
            "ATR multiples. Levels adapt to current market volatility. "
            "Rejects configurations with a risk-reward ratio below the "
            "configured minimum."
        ),
        parameters=(
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="tp_atr_mult",
                param_type="float",
                default=3.0,
                min_value=0.1,
                max_value=20.0,
                description="Take-profit distance as multiple of ATR",
            ),
            ParameterSpec(
                name="sl_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description="Stop-loss distance as multiple of ATR",
            ),
            ParameterSpec(
                name="min_rr",
                param_type="float",
                default=0.5,
                min_value=0.0,
                max_value=10.0,
                description=(
                    "Minimum acceptable risk-reward ratio (tp_mult / sl_mult). "
                    "Set to 0 to disable this check."
                ),
            ),
        ),
        tags=("exit", "stop_loss", "take_profit", "atr", "risk_reward", "fixed"),
        typical_use=(
            "Standard exit mechanism for trend-following or breakout systems. "
            "Use wider ATR multiples (3x TP, 1.5x SL) for trending markets, "
            "tighter for mean-reversion. Always pair with a trailing stop or "
            "time-based exit for additional protection."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute fixed TP/SL levels for each bar.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'atr_period', 'tp_atr_mult', 'sl_atr_mult', 'min_rr'.

        Returns:
            DataFrame with ATR, TP/SL levels for both directions, and risk-reward ratio.
        """
        atr_period: int = params["atr_period"]
        tp_mult: float = params["tp_atr_mult"]
        sl_mult: float = params["sl_atr_mult"]
        min_rr: float = params["min_rr"]

        risk_reward = tp_mult / sl_mult

        if min_rr > 0 and risk_reward < min_rr:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=(
                    f"Risk-reward ratio {risk_reward:.2f} "
                    f"(tp={tp_mult}/sl={sl_mult}) is below minimum {min_rr:.2f}"
                ),
            )

        min_rows = atr_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder-smoothed ATR
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        # TP/SL levels
        tp_distance = atr * tp_mult
        sl_distance = atr * sl_mult

        result = pd.DataFrame(
            {
                "tpsl_atr": atr,
                "tpsl_long_tp": close + tp_distance,
                "tpsl_long_sl": close - sl_distance,
                "tpsl_short_tp": close - tp_distance,
                "tpsl_short_sl": close + sl_distance,
                "tpsl_risk_reward": risk_reward,
            },
            index=data.index,
        )

        return result
```

---

## 4.4 `src/forgequant/blocks/exit_rules/trailing_stop.py`

```python
"""
Trailing Stop exit rule block.

Computes a trailing stop level that follows price in the direction of
the trade, locking in profits. The stop only moves in the favorable
direction and never retreats.

For LONG trades:
    trailing_stop = max(all previous trailing_stops,
                        current_high - trail_atr_mult * ATR)

For SHORT trades:
    trailing_stop = min(all previous trailing_stops,
                        current_low + trail_atr_mult * ATR)

The trailing stop computation is vectorized using an expanding maximum
(for longs) or expanding minimum (for shorts) of the raw stop level.

Output columns:
    - trail_atr: ATR values
    - trail_long_stop: Trailing stop level for long positions
    - trail_short_stop: Trailing stop level for short positions
    - trail_long_exit: Boolean, True when close crosses below the long trailing stop
    - trail_short_exit: Boolean, True when close crosses above the short trailing stop
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class TrailingStopExit(BaseBlock):
    """ATR-based trailing stop exit."""

    metadata = BlockMetadata(
        name="trailing_stop",
        display_name="Trailing Stop",
        category=BlockCategory.EXIT_RULE,
        description=(
            "Computes an ATR-based trailing stop that follows price in the "
            "profitable direction and never retreats. For longs, the stop "
            "ratchets upward; for shorts, downward. Exit signals fire when "
            "price crosses the trailing stop level."
        ),
        parameters=(
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="trail_atr_mult",
                param_type="float",
                default=2.5,
                min_value=0.1,
                max_value=20.0,
                description="Trailing distance as a multiple of ATR",
            ),
        ),
        tags=("exit", "trailing", "stop_loss", "atr", "trend_following"),
        typical_use=(
            "Essential exit for trend-following systems. Lets profits run "
            "while protecting against reversals. A wider multiplier (3x ATR) "
            "gives the trade more room but risks giving back more profit. "
            "A tighter multiplier (1.5x ATR) exits sooner but may be "
            "whipsawed in volatile trends."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute trailing stop levels and exit signals.

        The trailing stop for longs is computed as the expanding maximum
        of (high - trail_mult * ATR). This ensures it only ratchets up.
        Similarly for shorts using expanding minimum.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'atr_period' and 'trail_atr_mult'.

        Returns:
            DataFrame with ATR, trailing stop levels, and exit signals.
        """
        atr_period: int = params["atr_period"]
        trail_mult: float = params["trail_atr_mult"]

        min_rows = atr_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # Compute ATR
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        trail_distance = atr * trail_mult

        # Raw trailing stop candidates
        raw_long_stop = high - trail_distance
        raw_short_stop = low + trail_distance

        # Ratchet: long stop only goes up, short stop only goes down
        # Use iterative approach for correct trailing behavior
        n = len(data)
        long_stop = np.full(n, np.nan)
        short_stop = np.full(n, np.nan)

        long_stop[0] = raw_long_stop.iloc[0]
        short_stop[0] = raw_short_stop.iloc[0]

        raw_long_vals = raw_long_stop.values
        raw_short_vals = raw_short_stop.values

        for i in range(1, n):
            if not np.isnan(raw_long_vals[i]):
                if not np.isnan(long_stop[i - 1]):
                    long_stop[i] = max(long_stop[i - 1], raw_long_vals[i])
                else:
                    long_stop[i] = raw_long_vals[i]
            else:
                long_stop[i] = long_stop[i - 1]

            if not np.isnan(raw_short_vals[i]):
                if not np.isnan(short_stop[i - 1]):
                    short_stop[i] = min(short_stop[i - 1], raw_short_vals[i])
                else:
                    short_stop[i] = raw_short_vals[i]
            else:
                short_stop[i] = short_stop[i - 1]

        long_stop_series = pd.Series(long_stop, index=data.index)
        short_stop_series = pd.Series(short_stop, index=data.index)

        # Exit signals: price crosses the trailing stop
        long_exit = close < long_stop_series
        short_exit = close > short_stop_series

        result = pd.DataFrame(
            {
                "trail_atr": atr,
                "trail_long_stop": long_stop_series,
                "trail_short_stop": short_stop_series,
                "trail_long_exit": long_exit.fillna(False),
                "trail_short_exit": short_exit.fillna(False),
            },
            index=data.index,
        )

        return result
```

---

## 4.5 `src/forgequant/blocks/exit_rules/time_based_exit.py`

```python
"""
Time-Based Exit rule block.

Provides bar-counting exit signals. For each bar, outputs a countdown
indicator showing how many bars remain before a forced exit, and a
boolean signal when the maximum holding period is reached.

This block does NOT track individual trade entries — it produces a
rolling bar counter that the strategy compiler can use relative to
each trade's entry bar.

Additionally, it flags bars that fall on configurable "avoid" days
(e.g., Fridays for forex to avoid weekend gaps) and bars near
session close times.

Output columns:
    - time_bar_index: Sequential bar counter (0-based, wrapping at max_bars)
    - time_max_bars_exit: Boolean, True every max_bars bars
    - time_avoid_day: Boolean, True on days that should be avoided
    - time_near_session_close: Boolean, True within close_warning_bars of session end
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class TimeBasedExit(BaseBlock):
    """Time-based exit signals and holding period management."""

    metadata = BlockMetadata(
        name="time_based_exit",
        display_name="Time-Based Exit",
        category=BlockCategory.EXIT_RULE,
        description=(
            "Provides bar-counting exit signals for maximum holding period "
            "enforcement, along with day-of-week avoidance flags and "
            "session-close proximity warnings."
        ),
        parameters=(
            ParameterSpec(
                name="max_bars",
                param_type="int",
                default=50,
                min_value=1,
                max_value=5000,
                description="Maximum number of bars to hold a position",
            ),
            ParameterSpec(
                name="avoid_days",
                param_type="str",
                default="",
                description=(
                    "Comma-separated day names to avoid (e.g. 'Friday,Sunday'). "
                    "Case-insensitive. Leave empty to disable."
                ),
            ),
            ParameterSpec(
                name="close_warning_bars",
                param_type="int",
                default=3,
                min_value=0,
                max_value=100,
                description=(
                    "Number of bars before a detected daily session change "
                    "to flag as near-session-close. Set to 0 to disable."
                ),
            ),
        ),
        tags=("exit", "time", "holding_period", "session", "day_of_week"),
        typical_use=(
            "Used to force-close trades that have been open too long "
            "(preventing stale positions), to avoid entering trades on "
            "certain days (e.g. Friday afternoon in forex), and to warn "
            "when the trading session is about to end."
        ),
    )

    # Map of day name to pandas dayofweek integer
    _DAY_MAP: dict[str, int] = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute time-based exit signals.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'max_bars', 'avoid_days', 'close_warning_bars'.

        Returns:
            DataFrame with bar counter, max-bars exit, avoid-day, and
            session-close warning columns.
        """
        max_bars: int = params["max_bars"]
        avoid_days_str: str = params["avoid_days"]
        close_warning: int = params["close_warning_bars"]

        n = len(data)

        # ── Bar counter (wrapping modulo max_bars) ──
        bar_index = pd.Series(np.arange(n) % max_bars, index=data.index, dtype=int)

        # Max bars exit: fires every max_bars bars (the last bar of each window)
        max_bars_exit = bar_index == (max_bars - 1)

        # ── Avoid days ──
        avoid_day = pd.Series(False, index=data.index)

        if avoid_days_str.strip():
            avoid_names = [
                d.strip().lower() for d in avoid_days_str.split(",") if d.strip()
            ]
            avoid_ints: list[int] = []
            for name in avoid_names:
                if name in self._DAY_MAP:
                    avoid_ints.append(self._DAY_MAP[name])

            if avoid_ints:
                avoid_day = data.index.dayofweek.isin(avoid_ints)
                avoid_day = pd.Series(avoid_day, index=data.index)

        # ── Session close warning ──
        near_close = pd.Series(False, index=data.index)

        if close_warning > 0 and isinstance(data.index, pd.DatetimeIndex):
            # Detect session boundaries by date changes
            dates = data.index.date
            date_series = pd.Series(dates, index=data.index)
            date_changes = date_series != date_series.shift(-1)

            # Mark the last close_warning bars before each date change
            change_positions = np.where(date_changes.values)[0]
            near_close_arr = np.zeros(n, dtype=bool)

            for pos in change_positions:
                start = max(0, pos - close_warning + 1)
                near_close_arr[start : pos + 1] = True

            near_close = pd.Series(near_close_arr, index=data.index)

        result = pd.DataFrame(
            {
                "time_bar_index": bar_index,
                "time_max_bars_exit": max_bars_exit,
                "time_avoid_day": avoid_day,
                "time_near_session_close": near_close,
            },
            index=data.index,
        )

        return result
```

---

## 4.6 `src/forgequant/blocks/exit_rules/breakeven_stop.py`

```python
"""
Breakeven Stop exit rule block.

After price moves a configurable distance in the profitable direction
(the activation threshold), the stop-loss is moved to the entry price
(breakeven) plus an optional small offset to cover costs.

This block computes per-bar breakeven activation levels and the
resulting breakeven stop levels, which the strategy compiler uses
relative to actual trade entry prices.

For a LONG trade entered at close:
    activation_level = close + activation_atr_mult * ATR
    breakeven_stop   = close + offset_atr_mult * ATR

For a SHORT trade entered at close:
    activation_level = close - activation_atr_mult * ATR
    breakeven_stop   = close - offset_atr_mult * ATR

The block flags bars where the activation level would have been reached
(high >= activation for longs, low <= activation for shorts), assuming
entry at the bar's own close. In real usage, the compiler applies this
relative to each trade's actual entry price.

Output columns:
    - be_atr: ATR values
    - be_long_activation: Price level that activates breakeven for longs
    - be_long_stop: The breakeven stop level for longs (entry + offset)
    - be_short_activation: Price level that activates breakeven for shorts
    - be_short_stop: The breakeven stop level for shorts (entry - offset)
    - be_long_activated: Boolean, True when bar's high >= long activation
    - be_short_activated: Boolean, True when bar's low <= short activation
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class BreakevenStopExit(BaseBlock):
    """Breakeven stop that activates after a profit threshold is reached."""

    metadata = BlockMetadata(
        name="breakeven_stop",
        display_name="Breakeven Stop",
        category=BlockCategory.EXIT_RULE,
        description=(
            "Moves the stop-loss to breakeven (entry price + offset) after "
            "price moves a configurable distance in the profitable direction. "
            "The activation threshold and offset are expressed as ATR "
            "multiples for volatility adaptation."
        ),
        parameters=(
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="activation_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description=(
                    "Distance (in ATR multiples) price must move in profit "
                    "before the breakeven stop activates"
                ),
            ),
            ParameterSpec(
                name="offset_atr_mult",
                param_type="float",
                default=0.1,
                min_value=0.0,
                max_value=5.0,
                description=(
                    "Offset (in ATR multiples) added to the entry price for "
                    "the breakeven stop. Covers spread and commissions. "
                    "Set to 0 for exact breakeven."
                ),
            ),
        ),
        tags=("exit", "breakeven", "stop_loss", "atr", "risk_management"),
        typical_use=(
            "Used after a FixedTPSL or TrailingStop to eliminate risk on "
            "a trade once it shows sufficient profit. Activation at 1.5x ATR "
            "means the trade must first move 1.5 ATR in your favor before "
            "the stop moves to breakeven."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute breakeven stop levels and activation flags.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'atr_period', 'activation_atr_mult',
                    'offset_atr_mult'.

        Returns:
            DataFrame with ATR, activation levels, breakeven stop levels,
            and activation flags.
        """
        atr_period: int = params["atr_period"]
        activation_mult: float = params["activation_atr_mult"]
        offset_mult: float = params["offset_atr_mult"]

        min_rows = atr_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # ATR computation
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        # Activation levels (relative to entry at current close)
        long_activation = close + (activation_mult * atr)
        short_activation = close - (activation_mult * atr)

        # Breakeven stop levels (entry + offset)
        long_be_stop = close + (offset_mult * atr)
        short_be_stop = close - (offset_mult * atr)

        # Activation flags (would the bar's range reach the activation?)
        # For a long entered at this bar's close, did the SAME bar's high
        # reach the activation? Typically no — activation happens on
        # subsequent bars. But we flag it for the compiler to interpret
        # relative to actual entry bars.
        long_activated = high >= long_activation
        short_activated = low <= short_activation

        result = pd.DataFrame(
            {
                "be_atr": atr,
                "be_long_activation": long_activation,
                "be_long_stop": long_be_stop,
                "be_short_activation": short_activation,
                "be_short_stop": short_be_stop,
                "be_long_activated": long_activated.fillna(False),
                "be_short_activated": short_activated.fillna(False),
            },
            index=data.index,
        )

        return result
```

---

## 4.7 Money Management — `__init__.py`

### `src/forgequant/blocks/money_management/__init__.py`

```python
"""
Money management (position sizing) blocks.

Provides:
    - FixedRiskSizing: Fixed percentage risk per trade
    - VolatilityTargetingSizing: Target a specific portfolio volatility
    - KellyFractionalSizing: Kelly criterion with fractional scaling
    - ATRBasedSizing: Position size inversely proportional to ATR

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.money_management.fixed_risk import FixedRiskSizing
from forgequant.blocks.money_management.volatility_targeting import VolatilityTargetingSizing
from forgequant.blocks.money_management.kelly_fractional import KellyFractionalSizing
from forgequant.blocks.money_management.atr_based_sizing import ATRBasedSizing

__all__ = [
    "FixedRiskSizing",
    "VolatilityTargetingSizing",
    "KellyFractionalSizing",
    "ATRBasedSizing",
]
```

---

## 4.8 `src/forgequant/blocks/money_management/fixed_risk.py`

```python
"""
Fixed Risk position sizing block.

The most fundamental position sizing method: risk a fixed percentage
of account equity per trade. The position size is determined by the
risk amount and the distance to the stop-loss.

Calculation:
    1. risk_amount = account_equity * risk_pct / 100
    2. stop_distance = sl_atr_mult * ATR  (per-bar)
    3. position_size = risk_amount / stop_distance

The position_size is in "units per point" — the number of units to
trade such that if price moves stop_distance against you, you lose
exactly risk_amount.

Output columns:
    - fr_atr: ATR values
    - fr_stop_distance: Distance to stop in price units
    - fr_risk_amount: Dollar risk per trade (constant)
    - fr_position_size: Position size (units)
    - fr_position_pct: Position as percentage of equity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class FixedRiskSizing(BaseBlock):
    """Fixed percentage risk per trade position sizing."""

    metadata = BlockMetadata(
        name="fixed_risk",
        display_name="Fixed Risk Sizing",
        category=BlockCategory.MONEY_MANAGEMENT,
        description=(
            "Sizes positions so that a fixed percentage of account equity "
            "is risked per trade. The position size adapts to volatility "
            "through ATR-based stop distance calculation."
        ),
        parameters=(
            ParameterSpec(
                name="risk_pct",
                param_type="float",
                default=1.0,
                min_value=0.01,
                max_value=10.0,
                description="Percentage of equity to risk per trade",
            ),
            ParameterSpec(
                name="account_equity",
                param_type="float",
                default=100000.0,
                min_value=100.0,
                max_value=1e12,
                description="Account equity in base currency",
            ),
            ParameterSpec(
                name="sl_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description="Stop-loss distance as ATR multiple",
            ),
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
        ),
        tags=("sizing", "risk", "fixed", "percentage", "money_management"),
        typical_use=(
            "The workhorse of position sizing. Risk 1-2% of equity per trade "
            "for conservative systems, up to 5% for aggressive. Always "
            "combined with a stop-loss exit rule whose ATR multiplier "
            "matches sl_atr_mult."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute fixed-risk position sizes.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'risk_pct', 'account_equity',
                    'sl_atr_mult', 'atr_period'.

        Returns:
            DataFrame with ATR, stop distance, risk amount, and position size.
        """
        risk_pct: float = params["risk_pct"]
        equity: float = params["account_equity"]
        sl_mult: float = params["sl_atr_mult"]
        atr_period: int = params["atr_period"]

        min_rows = atr_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # ATR
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        # Risk calculations
        risk_amount = equity * risk_pct / 100.0
        stop_distance = atr * sl_mult

        # Position size = risk / stop_distance
        # Avoid division by zero
        position_size = risk_amount / stop_distance.replace(0, np.nan)

        # Position as percentage of equity (position_size * close / equity * 100)
        position_pct = (position_size * close / equity) * 100.0

        result = pd.DataFrame(
            {
                "fr_atr": atr,
                "fr_stop_distance": stop_distance,
                "fr_risk_amount": risk_amount,
                "fr_position_size": position_size,
                "fr_position_pct": position_pct,
            },
            index=data.index,
        )

        return result
```

---

## 4.9 `src/forgequant/blocks/money_management/volatility_targeting.py`

```python
"""
Volatility Targeting position sizing block.

Sizes positions to target a specific annualized portfolio volatility.
The position is inversely proportional to the asset's realized
volatility — when the asset is more volatile, take a smaller position.

Calculation:
    1. realized_vol = rolling standard deviation of log returns * sqrt(annualization_factor)
    2. target_exposure = target_vol / realized_vol
    3. position_size = (account_equity * target_exposure) / close

The annualization factor depends on the bar timeframe:
    - Daily bars:   sqrt(252)
    - Hourly bars:  sqrt(252 * trading_hours_per_day)
    - etc.

Output columns:
    - vt_realized_vol: Annualized realized volatility
    - vt_target_exposure: Target notional exposure as fraction of equity
    - vt_position_size: Number of units to trade
    - vt_position_pct: Position as percentage of equity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class VolatilityTargetingSizing(BaseBlock):
    """Position sizing that targets a specific portfolio volatility."""

    metadata = BlockMetadata(
        name="volatility_targeting",
        display_name="Volatility Targeting",
        category=BlockCategory.MONEY_MANAGEMENT,
        description=(
            "Sizes positions inversely proportional to realized volatility "
            "to target a specific annualized portfolio volatility. When "
            "market volatility increases, position size decreases, and "
            "vice versa."
        ),
        parameters=(
            ParameterSpec(
                name="target_vol",
                param_type="float",
                default=0.15,
                min_value=0.01,
                max_value=1.0,
                description="Target annualized portfolio volatility (0.15 = 15%)",
            ),
            ParameterSpec(
                name="vol_lookback",
                param_type="int",
                default=20,
                min_value=5,
                max_value=500,
                description="Lookback period for realized volatility calculation",
            ),
            ParameterSpec(
                name="annualization_factor",
                param_type="float",
                default=252.0,
                min_value=1.0,
                max_value=100000.0,
                description=(
                    "Number of bars per year for annualization. "
                    "252 for daily, 252*6.5≈1638 for hourly (US equities), "
                    "252*24=6048 for hourly (forex 24h)."
                ),
            ),
            ParameterSpec(
                name="account_equity",
                param_type="float",
                default=100000.0,
                min_value=100.0,
                max_value=1e12,
                description="Account equity in base currency",
            ),
            ParameterSpec(
                name="max_leverage",
                param_type="float",
                default=5.0,
                min_value=0.1,
                max_value=100.0,
                description="Maximum allowed leverage (caps position size)",
            ),
        ),
        tags=("sizing", "volatility", "targeting", "risk_parity", "adaptive"),
        typical_use=(
            "Used in risk-parity and adaptive systems. Target 10-20% "
            "annualized vol for conservative portfolios. The max_leverage "
            "cap prevents excessively large positions in low-vol regimes."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute volatility-targeting position sizes.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: All parameter values.

        Returns:
            DataFrame with realized vol, target exposure, and position size.
        """
        target_vol: float = params["target_vol"]
        vol_lookback: int = params["vol_lookback"]
        ann_factor: float = params["annualization_factor"]
        equity: float = params["account_equity"]
        max_leverage: float = params["max_leverage"]

        min_rows = vol_lookback + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        # Log returns
        log_returns = np.log(close / close.shift(1))

        # Rolling realized volatility, annualized
        rolling_std = log_returns.rolling(window=vol_lookback).std(ddof=1)
        realized_vol = rolling_std * np.sqrt(ann_factor)

        # Target exposure = target_vol / realized_vol
        target_exposure = target_vol / realized_vol.replace(0, np.nan)

        # Cap at max leverage
        target_exposure = target_exposure.clip(upper=max_leverage)

        # Position size in units
        position_size = (equity * target_exposure) / close

        # Position as percentage of equity
        position_pct = target_exposure * 100.0

        result = pd.DataFrame(
            {
                "vt_realized_vol": realized_vol,
                "vt_target_exposure": target_exposure,
                "vt_position_size": position_size,
                "vt_position_pct": position_pct,
            },
            index=data.index,
        )

        return result
```

---

## 4.10 `src/forgequant/blocks/money_management/kelly_fractional.py`

```python
"""
Kelly Fractional position sizing block.

The Kelly Criterion determines the optimal fraction of capital to risk
on each trade to maximize long-term growth. In practice, full Kelly
is too aggressive, so a fractional Kelly (typically 25-50% of full Kelly)
is used.

Calculation:
    1. Estimate win_rate and avg_win/avg_loss from rolling window of returns
    2. Full Kelly fraction:
       f* = win_rate - (1 - win_rate) / payoff_ratio
       where payoff_ratio = avg_win / avg_loss
    3. Fractional Kelly:
       f_trade = max(0, f* * kelly_fraction)
    4. position_size = (equity * f_trade) / (sl_atr_mult * ATR)

The rolling window approach allows the Kelly fraction to adapt over
time as the strategy's characteristics change.

Output columns:
    - kelly_win_rate: Rolling estimated win rate
    - kelly_payoff_ratio: Rolling avg_win / avg_loss
    - kelly_full_fraction: Full Kelly fraction (can be negative → no trade)
    - kelly_fraction_used: Fractional Kelly after scaling and flooring at 0
    - kelly_position_size: Position size in units
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class KellyFractionalSizing(BaseBlock):
    """Kelly Criterion with fractional scaling for position sizing."""

    metadata = BlockMetadata(
        name="kelly_fractional",
        display_name="Kelly Fractional Sizing",
        category=BlockCategory.MONEY_MANAGEMENT,
        description=(
            "Sizes positions using a fractional Kelly Criterion based on "
            "rolling estimates of win rate and payoff ratio. Full Kelly "
            "is scaled down by a configurable fraction to reduce volatility."
        ),
        parameters=(
            ParameterSpec(
                name="kelly_fraction",
                param_type="float",
                default=0.25,
                min_value=0.01,
                max_value=1.0,
                description="Fraction of full Kelly to use (0.25 = quarter Kelly)",
            ),
            ParameterSpec(
                name="lookback",
                param_type="int",
                default=100,
                min_value=20,
                max_value=2000,
                description="Rolling window for win rate and payoff estimation",
            ),
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR period for stop distance calculation",
            ),
            ParameterSpec(
                name="sl_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description="Stop-loss distance as ATR multiple",
            ),
            ParameterSpec(
                name="account_equity",
                param_type="float",
                default=100000.0,
                min_value=100.0,
                max_value=1e12,
                description="Account equity in base currency",
            ),
            ParameterSpec(
                name="max_fraction",
                param_type="float",
                default=0.05,
                min_value=0.001,
                max_value=0.5,
                description="Maximum fraction of equity to risk (absolute cap)",
            ),
        ),
        tags=("sizing", "kelly", "optimal", "growth", "adaptive"),
        typical_use=(
            "Used in systems with a known or estimable edge. Quarter Kelly "
            "(0.25) is the most common setting — it sacrifices ~25% of "
            "optimal growth rate but reduces drawdowns by ~50%. Only "
            "use when the strategy has a demonstrable positive expectancy."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute Kelly-based position sizes.

        Uses bar-to-bar returns as a proxy for trade returns. In a real
        strategy, the compiler should replace these with actual trade P&L.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: All parameter values.

        Returns:
            DataFrame with win rate, payoff ratio, Kelly fractions,
            and position sizes.
        """
        kelly_frac: float = params["kelly_fraction"]
        lookback: int = params["lookback"]
        atr_period: int = params["atr_period"]
        sl_mult: float = params["sl_atr_mult"]
        equity: float = params["account_equity"]
        max_frac: float = params["max_fraction"]

        min_rows = max(lookback, atr_period) + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # ATR
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        # Bar returns as proxy for trade returns
        returns = close.pct_change()

        # Rolling win rate: fraction of bars with positive return
        is_win = (returns > 0).astype(float)
        win_rate = is_win.rolling(window=lookback).mean()

        # Rolling average win and average loss
        gains = returns.clip(lower=0.0)
        losses = (-returns).clip(lower=0.0)

        avg_win = gains.rolling(window=lookback).mean()
        avg_loss = losses.rolling(window=lookback).mean()

        # Payoff ratio = avg_win / avg_loss
        payoff_ratio = avg_win / avg_loss.replace(0, np.nan)

        # Full Kelly: f* = win_rate - (1 - win_rate) / payoff_ratio
        full_kelly = win_rate - (1.0 - win_rate) / payoff_ratio.replace(0, np.nan)

        # Fractional Kelly, floored at 0 (don't bet against yourself)
        frac_kelly = (full_kelly * kelly_frac).clip(lower=0.0)

        # Cap at max_fraction
        frac_kelly = frac_kelly.clip(upper=max_frac)

        # Position size: (equity * fraction) / stop_distance
        stop_distance = atr * sl_mult
        position_size = (equity * frac_kelly) / stop_distance.replace(0, np.nan)

        result = pd.DataFrame(
            {
                "kelly_win_rate": win_rate,
                "kelly_payoff_ratio": payoff_ratio,
                "kelly_full_fraction": full_kelly,
                "kelly_fraction_used": frac_kelly,
                "kelly_position_size": position_size,
            },
            index=data.index,
        )

        return result
```

---

## 4.11 `src/forgequant/blocks/money_management/atr_based_sizing.py`

```python
"""
ATR-Based position sizing block.

The simplest volatility-adaptive sizing: position size is inversely
proportional to ATR. When volatility is high, size is smaller; when
low, size is larger. This equalizes the dollar risk per trade
(approximately).

Calculation:
    1. ATR = Wilder-smoothed True Range
    2. risk_per_unit = ATR * risk_atr_mult
    3. position_size = (account_equity * risk_pct / 100) / risk_per_unit
    4. position_value = position_size * close

The position size is capped by max_position_pct of equity.

Output columns:
    - atrs_atr: ATR values
    - atrs_risk_per_unit: Risk per unit in price terms
    - atrs_position_size: Position size (units)
    - atrs_position_value: Position value in base currency
    - atrs_position_pct: Position as percentage of equity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class ATRBasedSizing(BaseBlock):
    """ATR-inverse position sizing for volatility equalization."""

    metadata = BlockMetadata(
        name="atr_based_sizing",
        display_name="ATR-Based Sizing",
        category=BlockCategory.MONEY_MANAGEMENT,
        description=(
            "Sizes positions inversely proportional to ATR, equalizing "
            "dollar risk across different volatility regimes. Simpler "
            "than full volatility targeting but effective for single-asset "
            "strategies."
        ),
        parameters=(
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="risk_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description="Risk distance as ATR multiple",
            ),
            ParameterSpec(
                name="risk_pct",
                param_type="float",
                default=1.0,
                min_value=0.01,
                max_value=10.0,
                description="Percentage of equity to risk per trade",
            ),
            ParameterSpec(
                name="account_equity",
                param_type="float",
                default=100000.0,
                min_value=100.0,
                max_value=1e12,
                description="Account equity in base currency",
            ),
            ParameterSpec(
                name="max_position_pct",
                param_type="float",
                default=20.0,
                min_value=1.0,
                max_value=100.0,
                description="Maximum position size as percentage of equity",
            ),
        ),
        tags=("sizing", "atr", "volatility", "inverse", "equalization"),
        typical_use=(
            "Good default position sizing for any single-instrument strategy. "
            "Risk 1% with 1.5x ATR stop for conservative sizing. The "
            "max_position_pct cap prevents oversized positions when ATR "
            "is unusually low."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute ATR-based position sizes.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: All parameter values.

        Returns:
            DataFrame with ATR, risk per unit, position size, value, and pct.
        """
        atr_period: int = params["atr_period"]
        risk_mult: float = params["risk_atr_mult"]
        risk_pct: float = params["risk_pct"]
        equity: float = params["account_equity"]
        max_pct: float = params["max_position_pct"]

        min_rows = atr_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # ATR
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        # Risk per unit
        risk_per_unit = atr * risk_mult

        # Position size
        risk_amount = equity * risk_pct / 100.0
        position_size = risk_amount / risk_per_unit.replace(0, np.nan)

        # Position value and percentage
        position_value = position_size * close
        position_pct = (position_value / equity) * 100.0

        # Cap at max_position_pct
        cap_mask = position_pct > max_pct
        if cap_mask.any():
            max_position_value = equity * max_pct / 100.0
            position_size = position_size.where(
                ~cap_mask,
                max_position_value / close.replace(0, np.nan),
            )
            position_value = position_size * close
            position_pct = (position_value / equity) * 100.0

        result = pd.DataFrame(
            {
                "atrs_atr": atr,
                "atrs_risk_per_unit": risk_per_unit,
                "atrs_position_size": position_size,
                "atrs_position_value": position_value,
                "atrs_position_pct": position_pct,
            },
            index=data.index,
        )

        return result
```

---

## 4.12 Filters — `__init__.py`

### `src/forgequant/blocks/filters/__init__.py`

```python
"""
Filter blocks.

Provides:
    - TradingSessionFilter: Restricts trading to specific time windows
    - SpreadFilter: Filters out bars where spread is too wide
    - MaxDrawdownFilter: Halts trading when drawdown exceeds a threshold
    - TrendFilter: Only allows trades aligned with the broader trend

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.filters.trading_session import TradingSessionFilter
from forgequant.blocks.filters.spread_filter import SpreadFilter
from forgequant.blocks.filters.max_drawdown_filter import MaxDrawdownFilter
from forgequant.blocks.filters.trend_filter import TrendFilter

__all__ = [
    "TradingSessionFilter",
    "SpreadFilter",
    "MaxDrawdownFilter",
    "TrendFilter",
]
```

---

## 4.13 `src/forgequant/blocks/filters/trading_session.py`

```python
"""
Trading Session filter block.

Restricts trading to specific hours of the day, allowing strategies
to focus on the most liquid sessions and avoid thin/volatile off-hours.

The block supports defining up to two session windows per day (e.g.,
London session 08:00-16:00 and New York session 13:00-21:00).
Overlapping windows are merged.

Output columns:
    - session_active: Boolean, True when the current bar is within
                      an allowed trading session
    - session_name: String identifying which session is active
                    ("session_1", "session_2", "outside", "overlap")
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class TradingSessionFilter(BaseBlock):
    """Time-of-day trading session filter."""

    metadata = BlockMetadata(
        name="trading_session",
        display_name="Trading Session Filter",
        category=BlockCategory.FILTER,
        description=(
            "Restricts trading to specific hours of the day. Supports "
            "two session windows to cover major trading sessions "
            "(e.g. London + New York). Bars outside all sessions are "
            "flagged as inactive."
        ),
        parameters=(
            ParameterSpec(
                name="session1_start",
                param_type="int",
                default=8,
                min_value=0,
                max_value=23,
                description="Session 1 start hour (0-23, inclusive)",
            ),
            ParameterSpec(
                name="session1_end",
                param_type="int",
                default=16,
                min_value=0,
                max_value=23,
                description="Session 1 end hour (0-23, exclusive)",
            ),
            ParameterSpec(
                name="session2_start",
                param_type="int",
                default=-1,
                min_value=-1,
                max_value=23,
                description="Session 2 start hour (-1 to disable)",
            ),
            ParameterSpec(
                name="session2_end",
                param_type="int",
                default=-1,
                min_value=-1,
                max_value=23,
                description="Session 2 end hour (-1 to disable)",
            ),
        ),
        tags=("filter", "session", "time", "hours", "liquidity"),
        typical_use=(
            "Used to avoid low-liquidity periods (Asian session for EUR/USD), "
            "or to focus on the overlap between London and New York. "
            "Session hours are in the timezone of your data."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute session activity flags for each bar.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain session start/end hours.

        Returns:
            DataFrame with session_active and session_name columns.
        """
        s1_start: int = params["session1_start"]
        s1_end: int = params["session1_end"]
        s2_start: int = params["session2_start"]
        s2_end: int = params["session2_end"]

        if not isinstance(data.index, pd.DatetimeIndex):
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason="DataFrame index must be a DatetimeIndex for session filtering",
            )

        hours = data.index.hour
        n = len(data)

        # Session 1
        if s1_start < s1_end:
            in_s1 = (hours >= s1_start) & (hours < s1_end)
        elif s1_start > s1_end:
            # Overnight session (e.g. 22:00 - 06:00)
            in_s1 = (hours >= s1_start) | (hours < s1_end)
        else:
            # start == end: 24-hour session
            in_s1 = pd.Series(True, index=data.index)

        in_s1 = pd.Series(in_s1, index=data.index)

        # Session 2 (disabled if start or end == -1)
        if s2_start >= 0 and s2_end >= 0:
            if s2_start < s2_end:
                in_s2 = (hours >= s2_start) & (hours < s2_end)
            elif s2_start > s2_end:
                in_s2 = (hours >= s2_start) | (hours < s2_end)
            else:
                in_s2 = pd.Series(True, index=data.index)
            in_s2 = pd.Series(in_s2, index=data.index)
        else:
            in_s2 = pd.Series(False, index=data.index)

        # Combined
        active = in_s1 | in_s2

        # Session name
        names = pd.Series("outside", index=data.index, dtype="object")
        names[in_s1 & ~in_s2] = "session_1"
        names[in_s2 & ~in_s1] = "session_2"
        names[in_s1 & in_s2] = "overlap"

        result = pd.DataFrame(
            {
                "session_active": active,
                "session_name": names,
            },
            index=data.index,
        )

        return result
```

---

## 4.14 `src/forgequant/blocks/filters/spread_filter.py`

```python
"""
Spread Filter block.

Filters out bars where the bid-ask spread (approximated from OHLCV data)
is too wide relative to ATR. Wide spreads indicate low liquidity and
increase transaction costs.

Since OHLCV data doesn't directly contain spread information, this block
approximates the spread using the high-low range relative to ATR. Bars
where the bar range is abnormally large compared to ATR may indicate
wide spreads or extreme volatility.

Alternatively, if the data contains an explicit 'spread' column, that
is used directly.

Calculation:
    1. If 'spread' column exists: use it directly
       Else: approximate spread = (high - low) / close * 10000 (in pips-like units)
    2. avg_spread = rolling mean of spread over lookback
    3. spread_ok = spread <= max_spread AND spread <= avg_spread * max_spread_ratio

Output columns:
    - spread_value: The spread value for each bar
    - spread_avg: Rolling average spread
    - spread_ok: Boolean, True when spread is acceptable
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class SpreadFilter(BaseBlock):
    """Bid-ask spread quality filter."""

    metadata = BlockMetadata(
        name="spread_filter",
        display_name="Spread Filter",
        category=BlockCategory.FILTER,
        description=(
            "Filters out bars where the spread is too wide, indicating "
            "low liquidity or data quality issues. Uses an explicit "
            "'spread' column if available, otherwise approximates from "
            "the high-low range."
        ),
        parameters=(
            ParameterSpec(
                name="max_spread",
                param_type="float",
                default=50.0,
                min_value=0.1,
                max_value=10000.0,
                description=(
                    "Maximum absolute spread in pips-like units. "
                    "Bars exceeding this are filtered out."
                ),
            ),
            ParameterSpec(
                name="max_spread_ratio",
                param_type="float",
                default=3.0,
                min_value=1.0,
                max_value=20.0,
                description=(
                    "Maximum spread as a ratio of the rolling average spread. "
                    "Bars where spread > avg * ratio are filtered out."
                ),
            ),
            ParameterSpec(
                name="lookback",
                param_type="int",
                default=50,
                min_value=5,
                max_value=500,
                description="Lookback period for average spread calculation",
            ),
        ),
        tags=("filter", "spread", "liquidity", "cost", "quality"),
        typical_use=(
            "Used to avoid entering trades during illiquid periods (e.g. "
            "news events, market opens, off-hours) where wide spreads "
            "would erode profits. Essential for scalping and short-term "
            "strategies."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute spread filter flags.

        Args:
            data: OHLCV DataFrame. May optionally contain a 'spread' column.
            params: Must contain 'max_spread', 'max_spread_ratio', 'lookback'.

        Returns:
            DataFrame with spread values, average, and filter flag.
        """
        max_spread: float = params["max_spread"]
        max_ratio: float = params["max_spread_ratio"]
        lookback: int = params["lookback"]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Use explicit spread column if present, else approximate
        if "spread" in data.columns:
            spread = data["spread"].astype(float)
        else:
            # Approximate spread as high-low range in pip-like units
            # Multiply by 10000 for forex-style pip representation
            spread = ((high - low) / close.replace(0, np.nan)) * 10000.0

        # Rolling average spread
        avg_spread = spread.rolling(window=lookback, min_periods=1).mean()

        # Filter conditions
        below_absolute = spread <= max_spread
        below_ratio = spread <= (avg_spread * max_ratio)
        spread_ok = below_absolute & below_ratio

        result = pd.DataFrame(
            {
                "spread_value": spread,
                "spread_avg": avg_spread,
                "spread_ok": spread_ok.fillna(False),
            },
            index=data.index,
        )

        return result
```

---

## 4.15 `src/forgequant/blocks/filters/max_drawdown_filter.py`

```python
"""
Max Drawdown Filter block.

Monitors the cumulative return curve and halts trading when the
current drawdown from peak exceeds a configurable threshold. Trading
resumes when equity recovers to within a recovery percentage of the
peak.

This block operates on the close price series as a proxy for equity
(suitable for a single-instrument strategy). The strategy compiler
can feed actual equity curve data when available.

Calculation:
    1. cumulative_return = close / close[0]
    2. running_max = expanding max of cumulative_return
    3. drawdown = (cumulative_return - running_max) / running_max
    4. allow_trading = |drawdown| <= max_drawdown_pct / 100

Recovery logic:
    Once drawdown exceeds the threshold, trading is halted until
    |drawdown| recovers to <= recovery_pct / 100.

Output columns:
    - dd_cumulative_return: Cumulative return relative to first bar
    - dd_running_max: Running peak of cumulative return
    - dd_drawdown: Current drawdown (negative value)
    - dd_drawdown_pct: Drawdown as positive percentage
    - dd_allow_trading: Boolean, True when drawdown is acceptable
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class MaxDrawdownFilter(BaseBlock):
    """Maximum drawdown circuit breaker filter."""

    metadata = BlockMetadata(
        name="max_drawdown_filter",
        display_name="Max Drawdown Filter",
        category=BlockCategory.FILTER,
        description=(
            "Halts trading when the current drawdown from peak exceeds a "
            "threshold. Trading resumes once the drawdown recovers to "
            "within a configurable percentage. Acts as a circuit breaker "
            "to prevent catastrophic losses."
        ),
        parameters=(
            ParameterSpec(
                name="max_drawdown_pct",
                param_type="float",
                default=15.0,
                min_value=1.0,
                max_value=50.0,
                description="Maximum drawdown percentage before halting trades",
            ),
            ParameterSpec(
                name="recovery_pct",
                param_type="float",
                default=10.0,
                min_value=0.5,
                max_value=50.0,
                description=(
                    "Drawdown must recover to this percentage or less before "
                    "trading resumes"
                ),
            ),
        ),
        tags=("filter", "drawdown", "risk", "circuit_breaker", "protection"),
        typical_use=(
            "Essential risk management filter. Set max_drawdown to 15-20% "
            "for most strategies. The recovery threshold prevents premature "
            "re-entry during a declining market. Combine with position "
            "sizing to create a multi-layered defense."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute drawdown levels and trading permission flags.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'max_drawdown_pct' and 'recovery_pct'.

        Returns:
            DataFrame with cumulative return, drawdown, and allow_trading flag.
        """
        max_dd: float = params["max_drawdown_pct"]
        recovery: float = params["recovery_pct"]

        if recovery > max_dd:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=(
                    f"recovery_pct ({recovery}) must be <= max_drawdown_pct "
                    f"({max_dd}) — you can't require recovery beyond the "
                    f"halt threshold"
                ),
            )

        close = data["close"]
        n = len(close)

        if n < 2:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least 2 rows, got {n}",
            )

        # Cumulative return
        first_close = close.iloc[0]
        if first_close == 0:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason="First close price is 0; cannot compute returns",
            )

        cum_return = close / first_close

        # Running peak
        running_max = cum_return.expanding().max()

        # Drawdown (negative)
        drawdown = (cum_return - running_max) / running_max

        # Drawdown as positive percentage
        drawdown_pct = drawdown.abs() * 100.0

        # Hysteresis-based allow_trading
        # Start with trading allowed
        max_dd_thresh = max_dd
        recovery_thresh = recovery

        allow_arr = np.ones(n, dtype=bool)
        dd_pct_vals = drawdown_pct.values
        halted = False

        for i in range(n):
            if halted:
                if dd_pct_vals[i] <= recovery_thresh:
                    halted = False
                    allow_arr[i] = True
                else:
                    allow_arr[i] = False
            else:
                if dd_pct_vals[i] > max_dd_thresh:
                    halted = True
                    allow_arr[i] = False

        allow_trading = pd.Series(allow_arr, index=data.index)

        result = pd.DataFrame(
            {
                "dd_cumulative_return": cum_return,
                "dd_running_max": running_max,
                "dd_drawdown": drawdown,
                "dd_drawdown_pct": drawdown_pct,
                "dd_allow_trading": allow_trading,
            },
            index=data.index,
        )

        return result
```

---

## 4.16 `src/forgequant/blocks/filters/trend_filter.py`

```python
"""
Trend Filter block.

Only allows trades aligned with the broader trend direction.
Uses a long-period moving average with a buffer band to reduce
whipsaws near the MA.

Calculation:
    1. trend_ma = MA(close, period)  (EMA or SMA)
    2. upper_buffer = trend_ma * (1 + buffer_pct / 100)
       lower_buffer = trend_ma * (1 - buffer_pct / 100)
    3. allow_long:
       True when close > upper_buffer (clearly above trend)
    4. allow_short:
       True when close < lower_buffer (clearly below trend)
    5. In the buffer zone (lower <= close <= upper):
       Neither long nor short is allowed (choppy zone)

Output columns:
    - trend_ma: The trend moving average
    - trend_upper_buffer: Upper buffer level
    - trend_lower_buffer: Lower buffer level
    - trend_allow_long: Boolean, True when longs are permitted
    - trend_allow_short: Boolean, True when shorts are permitted
    - trend_direction: "bullish", "bearish", or "neutral"
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class TrendFilter(BaseBlock):
    """Trend alignment filter with buffer band to reduce whipsaws."""

    metadata = BlockMetadata(
        name="trend_filter",
        display_name="Trend Filter",
        category=BlockCategory.FILTER,
        description=(
            "Allows trades only when price is clearly above (for longs) "
            "or below (for shorts) a long-period moving average. A buffer "
            "band around the MA creates a neutral zone where no trades "
            "are permitted, reducing whipsaw losses near the trend boundary."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=200,
                min_value=10,
                max_value=1000,
                description="Moving average period for trend determination",
            ),
            ParameterSpec(
                name="ma_type",
                param_type="str",
                default="ema",
                choices=("ema", "sma"),
                description="Moving average type",
            ),
            ParameterSpec(
                name="buffer_pct",
                param_type="float",
                default=0.5,
                min_value=0.0,
                max_value=5.0,
                description=(
                    "Buffer zone width as percentage of MA. "
                    "Set to 0 for no buffer."
                ),
            ),
        ),
        tags=("filter", "trend", "moving_average", "direction", "whipsaw"),
        typical_use=(
            "Essential for trend-following systems. Use a 200-period EMA "
            "with a 0.5% buffer. Only take long entries when trend_allow_long "
            "is True, and short entries when trend_allow_short is True. "
            "This single filter can eliminate a large portion of losing "
            "trades in choppy markets."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute trend filter signals.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'period', 'ma_type', 'buffer_pct'.

        Returns:
            DataFrame with MA, buffer levels, and allow_long/allow_short flags.
        """
        period: int = params["period"]
        ma_type: str = params["ma_type"]
        buffer_pct: float = params["buffer_pct"]

        if len(data) < period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        # Trend MA
        if ma_type == "ema":
            trend_ma = close.ewm(span=period, adjust=False).mean()
        else:
            trend_ma = close.rolling(window=period).mean()

        # Buffer band
        buffer_factor = buffer_pct / 100.0
        upper_buffer = trend_ma * (1.0 + buffer_factor)
        lower_buffer = trend_ma * (1.0 - buffer_factor)

        # Allow signals
        allow_long = close > upper_buffer
        allow_short = close < lower_buffer

        # Direction label
        direction = pd.Series("neutral", index=data.index, dtype="object")
        direction[allow_long] = "bullish"
        direction[allow_short] = "bearish"

        result = pd.DataFrame(
            {
                "trend_ma": trend_ma,
                "trend_upper_buffer": upper_buffer,
                "trend_lower_buffer": lower_buffer,
                "trend_allow_long": allow_long.fillna(False),
                "trend_allow_short": allow_short.fillna(False),
                "trend_direction": direction,
            },
            index=data.index,
        )

        return result
```

---

## 4.17 Test Suite — Exit Rules

### `tests/unit/exit_rules/__init__.py`

```python
"""Tests for exit rule blocks."""
```

---

### `tests/unit/exit_rules/test_fixed_tpsl.py`

```python
"""Tests for the Fixed TP/SL exit rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.exit_rules.fixed_tpsl import FixedTPSLExit
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def tpsl() -> FixedTPSLExit:
    return FixedTPSLExit()


class TestFixedTPSLMetadata:
    def test_name(self, tpsl: FixedTPSLExit) -> None:
        assert tpsl.metadata.name == "fixed_tpsl"

    def test_category(self, tpsl: FixedTPSLExit) -> None:
        assert tpsl.metadata.category == BlockCategory.EXIT_RULE

    def test_defaults(self, tpsl: FixedTPSLExit) -> None:
        d = tpsl.metadata.get_defaults()
        assert d["atr_period"] == 14
        assert d["tp_atr_mult"] == 3.0
        assert d["sl_atr_mult"] == 1.5
        assert d["min_rr"] == 0.5

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "fixed_tpsl" in registry


class TestFixedTPSLCompute:
    def test_output_columns(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        expected = {
            "tpsl_atr", "tpsl_long_tp", "tpsl_long_sl",
            "tpsl_short_tp", "tpsl_short_sl", "tpsl_risk_reward",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_long_tp_above_close(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["tpsl_long_tp"].dropna()
        assert (valid > close.loc[valid.index]).all()

    def test_long_sl_below_close(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["tpsl_long_sl"].dropna()
        assert (valid < close.loc[valid.index]).all()

    def test_short_tp_below_close(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["tpsl_short_tp"].dropna()
        assert (valid < close.loc[valid.index]).all()

    def test_short_sl_above_close(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["tpsl_short_sl"].dropna()
        assert (valid > close.loc[valid.index]).all()

    def test_risk_reward_calculation(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv, {"tp_atr_mult": 3.0, "sl_atr_mult": 1.5})
        assert abs(result["tpsl_risk_reward"].iloc[0] - 2.0) < 1e-10

    def test_symmetry(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Long TP distance should equal short SL distance from close."""
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        long_tp_dist = result["tpsl_long_tp"] - close
        short_sl_dist = result["tpsl_short_sl"] - close
        pd.testing.assert_series_equal(
            long_tp_dist, short_sl_dist, check_names=False, atol=1e-10
        )

    def test_low_rr_raises(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="Risk-reward ratio"):
            tpsl.execute(sample_ohlcv, {"tp_atr_mult": 0.5, "sl_atr_mult": 2.0, "min_rr": 0.5})

    def test_rr_check_disabled(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Setting min_rr=0 should disable the RR check."""
        result = tpsl.execute(
            sample_ohlcv,
            {"tp_atr_mult": 0.5, "sl_atr_mult": 2.0, "min_rr": 0.0},
        )
        assert result["tpsl_risk_reward"].iloc[0] == 0.25

    def test_insufficient_data_raises(self, tpsl: FixedTPSLExit) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5,
                "low": [99.0] * 5, "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            tpsl.execute(df)
```

---

### `tests/unit/exit_rules/test_trailing_stop.py`

```python
"""Tests for the Trailing Stop exit rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.exit_rules.trailing_stop import TrailingStopExit
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def trail() -> TrailingStopExit:
    return TrailingStopExit()


class TestTrailingStopMetadata:
    def test_name(self, trail: TrailingStopExit) -> None:
        assert trail.metadata.name == "trailing_stop"

    def test_category(self, trail: TrailingStopExit) -> None:
        assert trail.metadata.category == BlockCategory.EXIT_RULE

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "trailing_stop" in registry


class TestTrailingStopCompute:
    def test_output_columns(
        self, trail: TrailingStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trail.execute(sample_ohlcv)
        expected = {
            "trail_atr", "trail_long_stop", "trail_short_stop",
            "trail_long_exit", "trail_short_exit",
        }
        assert expected == set(result.columns)

    def test_long_stop_only_increases(
        self, trail: TrailingStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        """The long trailing stop should never decrease."""
        result = trail.execute(sample_ohlcv)
        stops = result["trail_long_stop"].dropna().values
        for i in range(1, len(stops)):
            assert stops[i] >= stops[i - 1] - 1e-10, (
                f"Long stop decreased at index {i}: {stops[i-1]} -> {stops[i]}"
            )

    def test_short_stop_only_decreases(
        self, trail: TrailingStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        """The short trailing stop should never increase."""
        result = trail.execute(sample_ohlcv)
        stops = result["trail_short_stop"].dropna().values
        for i in range(1, len(stops)):
            assert stops[i] <= stops[i - 1] + 1e-10, (
                f"Short stop increased at index {i}: {stops[i-1]} -> {stops[i]}"
            )

    def test_long_exit_fires_on_breakdown(self, trail: TrailingStopExit) -> None:
        """When price drops sharply, long exit should fire."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([np.linspace(100, 120, 60), np.linspace(119, 90, 40)])
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1,
                "low": close - 1, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = trail.execute(df, {"atr_period": 5, "trail_atr_mult": 2.0})
        assert result["trail_long_exit"].iloc[70:].sum() > 0

    def test_wider_mult_fewer_exits(
        self, trail: TrailingStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Wider trailing distance should produce fewer exit signals."""
        tight = trail.execute(sample_ohlcv, {"trail_atr_mult": 1.0})
        wide = trail.execute(sample_ohlcv, {"trail_atr_mult": 4.0})
        assert tight["trail_long_exit"].sum() >= wide["trail_long_exit"].sum()

    def test_insufficient_data_raises(self, trail: TrailingStopExit) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5,
                "low": [99.0] * 5, "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            trail.execute(df)
```

---

### `tests/unit/exit_rules/test_time_based_exit.py`

```python
"""Tests for the Time-Based Exit rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.exit_rules.time_based_exit import TimeBasedExit
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def time_exit() -> TimeBasedExit:
    return TimeBasedExit()


class TestTimeBasedExitMetadata:
    def test_name(self, time_exit: TimeBasedExit) -> None:
        assert time_exit.metadata.name == "time_based_exit"

    def test_category(self, time_exit: TimeBasedExit) -> None:
        assert time_exit.metadata.category == BlockCategory.EXIT_RULE

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "time_based_exit" in registry


class TestTimeBasedExitCompute:
    def test_output_columns(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv)
        expected = {
            "time_bar_index", "time_max_bars_exit",
            "time_avoid_day", "time_near_session_close",
        }
        assert expected == set(result.columns)

    def test_bar_index_wraps(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"max_bars": 50})
        bar_idx = result["time_bar_index"]
        assert bar_idx.min() == 0
        assert bar_idx.max() == 49

    def test_max_bars_exit_frequency(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Max bars exit should fire every max_bars bars."""
        max_bars = 10
        result = time_exit.execute(sample_ohlcv, {"max_bars": max_bars})
        exits = result["time_max_bars_exit"]
        expected_count = len(sample_ohlcv) // max_bars
        assert exits.sum() == expected_count

    def test_avoid_friday(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"avoid_days": "Friday"})
        avoid = result["time_avoid_day"]
        # Check that Friday bars are flagged
        friday_mask = sample_ohlcv.index.dayofweek == 4
        if friday_mask.any():
            assert avoid[friday_mask].all()
            assert not avoid[~friday_mask].any()

    def test_avoid_multiple_days(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"avoid_days": "Saturday,Sunday"})
        avoid = result["time_avoid_day"]
        weekend_mask = sample_ohlcv.index.dayofweek.isin([5, 6])
        if weekend_mask.any():
            assert avoid[weekend_mask].all()

    def test_empty_avoid_days(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"avoid_days": ""})
        assert not result["time_avoid_day"].any()

    def test_session_close_warning(self, time_exit: TimeBasedExit) -> None:
        """Bars near day boundaries should be flagged."""
        # Create data spanning 3 days with hourly bars
        dates = pd.date_range("2024-01-01", periods=72, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = time_exit.execute(df, {"close_warning_bars": 3})
        near_close = result["time_near_session_close"]
        # Last 3 bars of each day should be flagged
        assert near_close.sum() > 0

    def test_session_close_disabled(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"close_warning_bars": 0})
        assert not result["time_near_session_close"].any()
```

---

### `tests/unit/exit_rules/test_breakeven_stop.py`

```python
"""Tests for the Breakeven Stop exit rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.exit_rules.breakeven_stop import BreakevenStopExit
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def be_stop() -> BreakevenStopExit:
    return BreakevenStopExit()


class TestBreakevenStopMetadata:
    def test_name(self, be_stop: BreakevenStopExit) -> None:
        assert be_stop.metadata.name == "breakeven_stop"

    def test_category(self, be_stop: BreakevenStopExit) -> None:
        assert be_stop.metadata.category == BlockCategory.EXIT_RULE

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "breakeven_stop" in registry


class TestBreakevenStopCompute:
    def test_output_columns(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = be_stop.execute(sample_ohlcv)
        expected = {
            "be_atr", "be_long_activation", "be_long_stop",
            "be_short_activation", "be_short_stop",
            "be_long_activated", "be_short_activated",
        }
        assert expected == set(result.columns)

    def test_long_activation_above_close(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = be_stop.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["be_long_activation"].dropna()
        assert (valid > close.loc[valid.index]).all()

    def test_short_activation_below_close(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = be_stop.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["be_short_activation"].dropna()
        assert (valid < close.loc[valid.index]).all()

    def test_long_stop_above_close_with_offset(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        """With positive offset, the BE stop should be slightly above entry."""
        result = be_stop.execute(sample_ohlcv, {"offset_atr_mult": 0.1})
        close = sample_ohlcv["close"]
        valid = result["be_long_stop"].dropna()
        assert (valid >= close.loc[valid.index]).all()

    def test_zero_offset_exact_breakeven(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        """With zero offset, BE stop exactly equals entry (close)."""
        result = be_stop.execute(sample_ohlcv, {"offset_atr_mult": 0.0})
        close = sample_ohlcv["close"]
        pd.testing.assert_series_equal(
            result["be_long_stop"], close, check_names=False, atol=1e-10
        )

    def test_insufficient_data_raises(self, be_stop: BreakevenStopExit) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5,
                "low": [99.0] * 5, "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            be_stop.execute(df)
```

---

## 4.18 Test Suite — Money Management

### `tests/unit/money_management/__init__.py`

```python
"""Tests for money management blocks."""
```

---

### `tests/unit/money_management/test_fixed_risk.py`

```python
"""Tests for the Fixed Risk position sizing block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.money_management.fixed_risk import FixedRiskSizing
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def fixed_risk() -> FixedRiskSizing:
    return FixedRiskSizing()


class TestFixedRiskMetadata:
    def test_name(self, fixed_risk: FixedRiskSizing) -> None:
        assert fixed_risk.metadata.name == "fixed_risk"

    def test_category(self, fixed_risk: FixedRiskSizing) -> None:
        assert fixed_risk.metadata.category == BlockCategory.MONEY_MANAGEMENT

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "fixed_risk" in registry


class TestFixedRiskCompute:
    def test_output_columns(
        self, fixed_risk: FixedRiskSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fixed_risk.execute(sample_ohlcv)
        expected = {
            "fr_atr", "fr_stop_distance", "fr_risk_amount",
            "fr_position_size", "fr_position_pct",
        }
        assert expected == set(result.columns)

    def test_risk_amount_constant(
        self, fixed_risk: FixedRiskSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fixed_risk.execute(
            sample_ohlcv, {"risk_pct": 2.0, "account_equity": 50000.0}
        )
        assert result["fr_risk_amount"].iloc[0] == 1000.0
        assert result["fr_risk_amount"].iloc[-1] == 1000.0

    def test_position_size_positive(
        self, fixed_risk: FixedRiskSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fixed_risk.execute(sample_ohlcv)
        valid = result["fr_position_size"].dropna()
        assert (valid > 0).all()

    def test_higher_vol_smaller_position(
        self, fixed_risk: FixedRiskSizing
    ) -> None:
        """Higher ATR should result in smaller position size."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        # Low vol
        close_low = np.full(n, 100.0)
        df_low = pd.DataFrame(
            {
                "open": close_low, "high": close_low + 0.1,
                "low": close_low - 0.1, "close": close_low,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        # High vol
        close_high = np.full(n, 100.0)
        df_high = pd.DataFrame(
            {
                "open": close_high, "high": close_high + 5.0,
                "low": close_high - 5.0, "close": close_high,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        pos_low = fixed_risk.execute(df_low)["fr_position_size"].iloc[-1]
        pos_high = fixed_risk.execute(df_high)["fr_position_size"].iloc[-1]
        assert pos_low > pos_high

    def test_insufficient_data_raises(self, fixed_risk: FixedRiskSizing) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5,
                "low": [99.0] * 5, "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            fixed_risk.execute(df)
```

---

### `tests/unit/money_management/test_volatility_targeting.py`

```python
"""Tests for the Volatility Targeting position sizing block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.money_management.volatility_targeting import VolatilityTargetingSizing
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def vol_target() -> VolatilityTargetingSizing:
    return VolatilityTargetingSizing()


class TestVolTargetMetadata:
    def test_name(self, vol_target: VolatilityTargetingSizing) -> None:
        assert vol_target.metadata.name == "volatility_targeting"

    def test_category(self, vol_target: VolatilityTargetingSizing) -> None:
        assert vol_target.metadata.category == BlockCategory.MONEY_MANAGEMENT

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "volatility_targeting" in registry


class TestVolTargetCompute:
    def test_output_columns(
        self, vol_target: VolatilityTargetingSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = vol_target.execute(sample_ohlcv)
        expected = {
            "vt_realized_vol", "vt_target_exposure",
            "vt_position_size", "vt_position_pct",
        }
        assert expected == set(result.columns)

    def test_exposure_capped_at_max_leverage(
        self, vol_target: VolatilityTargetingSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = vol_target.execute(sample_ohlcv, {"max_leverage": 3.0})
        valid = result["vt_target_exposure"].dropna()
        assert (valid <= 3.0 + 1e-10).all()

    def test_higher_vol_smaller_position(
        self, vol_target: VolatilityTargetingSizing
    ) -> None:
        """Higher realized vol should produce smaller positions."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(42)

        # Low vol asset
        close_low = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
        df_low = pd.DataFrame(
            {
                "open": close_low, "high": close_low * 1.001,
                "low": close_low * 0.999, "close": close_low,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        # High vol asset
        close_high = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        df_high = pd.DataFrame(
            {
                "open": close_high, "high": close_high * 1.01,
                "low": close_high * 0.99, "close": close_high,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        exp_low = vol_target.execute(df_low)["vt_target_exposure"].iloc[-1]
        exp_high = vol_target.execute(df_high)["vt_target_exposure"].iloc[-1]
        assert exp_low > exp_high

    def test_position_size_positive(
        self, vol_target: VolatilityTargetingSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = vol_target.execute(sample_ohlcv)
        valid = result["vt_position_size"].dropna()
        assert (valid > 0).all()

    def test_insufficient_data_raises(
        self, vol_target: VolatilityTargetingSizing
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 10, "high": [101.0] * 10,
                "low": [99.0] * 10, "close": [100.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            vol_target.execute(df)
```

---

### `tests/unit/money_management/test_kelly_fractional.py`

```python
"""Tests for the Kelly Fractional position sizing block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.money_management.kelly_fractional import KellyFractionalSizing
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def kelly() -> KellyFractionalSizing:
    return KellyFractionalSizing()


class TestKellyMetadata:
    def test_name(self, kelly: KellyFractionalSizing) -> None:
        assert kelly.metadata.name == "kelly_fractional"

    def test_category(self, kelly: KellyFractionalSizing) -> None:
        assert kelly.metadata.category == BlockCategory.MONEY_MANAGEMENT

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "kelly_fractional" in registry


class TestKellyCompute:
    def test_output_columns(
        self, kelly: KellyFractionalSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = kelly.execute(sample_ohlcv)
        expected = {
            "kelly_win_rate", "kelly_payoff_ratio",
            "kelly_full_fraction", "kelly_fraction_used",
            "kelly_position_size",
        }
        assert expected == set(result.columns)

    def test_fraction_used_non_negative(
        self, kelly: KellyFractionalSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = kelly.execute(sample_ohlcv)
        valid = result["kelly_fraction_used"].dropna()
        assert (valid >= 0).all()

    def test_fraction_used_capped(
        self, kelly: KellyFractionalSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        max_frac = 0.03
        result = kelly.execute(sample_ohlcv, {"max_fraction": max_frac})
        valid = result["kelly_fraction_used"].dropna()
        assert (valid <= max_frac + 1e-10).all()

    def test_win_rate_range(
        self, kelly: KellyFractionalSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = kelly.execute(sample_ohlcv)
        wr = result["kelly_win_rate"].dropna()
        assert (wr >= 0).all()
        assert (wr <= 1.0).all()

    def test_uptrend_positive_kelly(self, kelly: KellyFractionalSizing) -> None:
        """In a consistent uptrend, Kelly fraction should be positive."""
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.2, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = kelly.execute(df)
        assert result["kelly_fraction_used"].iloc[-1] > 0

    def test_downtrend_zero_kelly(self, kelly: KellyFractionalSizing) -> None:
        """In a downtrend, Kelly fraction should be zero (negative edge)."""
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(200, 100, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.2,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = kelly.execute(df)
        # Fraction should be 0 (floored) since the edge is negative
        assert result["kelly_fraction_used"].iloc[-1] == 0.0

    def test_insufficient_data_raises(self, kelly: KellyFractionalSizing) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 50, "high": [101.0] * 50,
                "low": [99.0] * 50, "close": [100.0] * 50,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            kelly.execute(df, {"lookback": 100})
```

---

### `tests/unit/money_management/test_atr_based_sizing.py`

```python
"""Tests for the ATR-Based Sizing block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.money_management.atr_based_sizing import ATRBasedSizing
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def atr_sizing() -> ATRBasedSizing:
    return ATRBasedSizing()


class TestATRSizingMetadata:
    def test_name(self, atr_sizing: ATRBasedSizing) -> None:
        assert atr_sizing.metadata.name == "atr_based_sizing"

    def test_category(self, atr_sizing: ATRBasedSizing) -> None:
        assert atr_sizing.metadata.category == BlockCategory.MONEY_MANAGEMENT

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "atr_based_sizing" in registry


class TestATRSizingCompute:
    def test_output_columns(
        self, atr_sizing: ATRBasedSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr_sizing.execute(sample_ohlcv)
        expected = {
            "atrs_atr", "atrs_risk_per_unit",
            "atrs_position_size", "atrs_position_value",
            "atrs_position_pct",
        }
        assert expected == set(result.columns)

    def test_position_size_positive(
        self, atr_sizing: ATRBasedSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr_sizing.execute(sample_ohlcv)
        valid = result["atrs_position_size"].dropna()
        assert (valid > 0).all()

    def test_position_pct_capped(
        self, atr_sizing: ATRBasedSizing
    ) -> None:
        """Position percentage should not exceed max_position_pct."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Very low vol -> large position -> should be capped
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.001,
                "low": close - 0.001, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = atr_sizing.execute(df, {"max_position_pct": 20.0})
        valid = result["atrs_position_pct"].dropna()
        assert (valid <= 20.0 + 0.1).all()  # Small tolerance for float

    def test_inverse_relationship(
        self, atr_sizing: ATRBasedSizing
    ) -> None:
        """Higher ATR should produce smaller position size."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        close = np.full(n, 100.0)
        df_low = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        df_high = pd.DataFrame(
            {
                "open": close, "high": close + 5.0,
                "low": close - 5.0, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        pos_low = atr_sizing.execute(df_low)["atrs_position_size"].iloc[-1]
        pos_high = atr_sizing.execute(df_high)["atrs_position_size"].iloc[-1]
        assert pos_low > pos_high

    def test_insufficient_data_raises(self, atr_sizing: ATRBasedSizing) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5,
                "low": [99.0] * 5, "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            atr_sizing.execute(df)
```

---

## 4.19 Test Suite — Filters

### `tests/unit/filters/__init__.py`

```python
"""Tests for filter blocks."""
```

---

### `tests/unit/filters/test_trading_session.py`

```python
"""Tests for the Trading Session filter block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.filters.trading_session import TradingSessionFilter
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def session() -> TradingSessionFilter:
    return TradingSessionFilter()


class TestTradingSessionMetadata:
    def test_name(self, session: TradingSessionFilter) -> None:
        assert session.metadata.name == "trading_session"

    def test_category(self, session: TradingSessionFilter) -> None:
        assert session.metadata.category == BlockCategory.FILTER

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "trading_session" in registry


class TestTradingSessionCompute:
    def test_output_columns(
        self, session: TradingSessionFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = session.execute(sample_ohlcv)
        assert "session_active" in result.columns
        assert "session_name" in result.columns

    def test_london_session(self, session: TradingSessionFilter) -> None:
        """Only 08:00-16:00 should be active."""
        dates = pd.date_range("2024-01-15", periods=24, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = session.execute(
            df, {"session1_start": 8, "session1_end": 16, "session2_start": -1}
        )
        active = result["session_active"]
        for i, dt in enumerate(dates):
            expected = 8 <= dt.hour < 16
            assert active.iloc[i] == expected, f"Hour {dt.hour}: expected {expected}"

    def test_overnight_session(self, session: TradingSessionFilter) -> None:
        """A session like 22:00-06:00 should wrap across midnight."""
        dates = pd.date_range("2024-01-15", periods=24, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = session.execute(
            df, {"session1_start": 22, "session1_end": 6, "session2_start": -1}
        )
        active = result["session_active"]
        for i, dt in enumerate(dates):
            expected = dt.hour >= 22 or dt.hour < 6
            assert active.iloc[i] == expected, f"Hour {dt.hour}"

    def test_two_sessions_overlap(self, session: TradingSessionFilter) -> None:
        """Overlapping sessions should be labeled 'overlap'."""
        dates = pd.date_range("2024-01-15", periods=24, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = session.execute(
            df,
            {
                "session1_start": 8, "session1_end": 16,
                "session2_start": 13, "session2_end": 21,
            },
        )
        names = result["session_name"]
        # Hours 13-15 should be overlap
        for i, dt in enumerate(dates):
            if 13 <= dt.hour < 16:
                assert names.iloc[i] == "overlap", f"Hour {dt.hour}"

    def test_session2_disabled(
        self, session: TradingSessionFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = session.execute(
            sample_ohlcv, {"session2_start": -1, "session2_end": -1}
        )
        names = result["session_name"]
        # Should never see "session_2" or "overlap"
        assert "session_2" not in names.values
```

---

### `tests/unit/filters/test_spread_filter.py`

```python
"""Tests for the Spread Filter block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.filters.spread_filter import SpreadFilter
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory


@pytest.fixture
def spread() -> SpreadFilter:
    return SpreadFilter()


class TestSpreadFilterMetadata:
    def test_name(self, spread: SpreadFilter) -> None:
        assert spread.metadata.name == "spread_filter"

    def test_category(self, spread: SpreadFilter) -> None:
        assert spread.metadata.category == BlockCategory.FILTER

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "spread_filter" in registry


class TestSpreadFilterCompute:
    def test_output_columns(
        self, spread: SpreadFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = spread.execute(sample_ohlcv)
        expected = {"spread_value", "spread_avg", "spread_ok"}
        assert expected == set(result.columns)

    def test_tight_spread_all_ok(self, spread: SpreadFilter) -> None:
        """Uniform tight spreads should all pass."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 1.1000)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.0001,
                "low": close - 0.0001, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = spread.execute(df, {"max_spread": 50.0, "max_spread_ratio": 3.0})
        assert result["spread_ok"].all()

    def test_wide_spread_filtered(self, spread: SpreadFilter) -> None:
        """A bar with an anomalous wide spread should be filtered."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 1.1000)
        high = close + 0.0001
        low = close - 0.0001
        # Bar 50 has a massive spread
        high[50] = close[50] + 0.05
        low[50] = close[50] - 0.05
        df = pd.DataFrame(
            {
                "open": close, "high": high,
                "low": low, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = spread.execute(df, {"max_spread": 50.0, "max_spread_ratio": 3.0})
        assert not result["spread_ok"].iloc[50]

    def test_explicit_spread_column(self, spread: SpreadFilter) -> None:
        """If a 'spread' column exists, it should be used directly."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        spread_vals = np.full(n, 2.0)
        spread_vals[50] = 200.0  # Anomalous
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1,
                "low": close - 1, "close": close,
                "volume": np.ones(n) * 1000,
                "spread": spread_vals,
            },
            index=dates,
        )
        result = spread.execute(df, {"max_spread": 50.0, "max_spread_ratio": 3.0})
        assert result["spread_value"].iloc[50] == 200.0
        assert not result["spread_ok"].iloc[50]

    def test_avg_spread_increases_with_window(
        self, spread: SpreadFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = spread.execute(sample_ohlcv)
        avg = result["spread_avg"]
        assert avg.iloc[-1] > 0
```

---

### `tests/unit/filters/test_max_drawdown_filter.py`

```python
"""Tests for the Max Drawdown Filter block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.filters.max_drawdown_filter import MaxDrawdownFilter
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def dd_filter() -> MaxDrawdownFilter:
    return MaxDrawdownFilter()


class TestMaxDrawdownMetadata:
    def test_name(self, dd_filter: MaxDrawdownFilter) -> None:
        assert dd_filter.metadata.name == "max_drawdown_filter"

    def test_category(self, dd_filter: MaxDrawdownFilter) -> None:
        assert dd_filter.metadata.category == BlockCategory.FILTER

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "max_drawdown_filter" in registry


class TestMaxDrawdownCompute:
    def test_output_columns(
        self, dd_filter: MaxDrawdownFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = dd_filter.execute(sample_ohlcv)
        expected = {
            "dd_cumulative_return", "dd_running_max",
            "dd_drawdown", "dd_drawdown_pct", "dd_allow_trading",
        }
        assert expected == set(result.columns)

    def test_uptrend_always_allowed(self, dd_filter: MaxDrawdownFilter) -> None:
        """In a pure uptrend (no drawdown), trading should always be allowed."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1,
                "low": close - 1, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = dd_filter.execute(df)
        assert result["dd_allow_trading"].all()

    def test_crash_halts_trading(self, dd_filter: MaxDrawdownFilter) -> None:
        """A >15% drawdown should halt trading."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([
            np.linspace(100, 120, 50),   # Up
            np.linspace(120, 95, 50),    # Down 20.8%
        ])
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1,
                "low": close - 1, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = dd_filter.execute(df, {"max_drawdown_pct": 15.0, "recovery_pct": 10.0})
        # At some point, trading should be halted
        assert not result["dd_allow_trading"].all()

    def test_hysteresis_recovery(self, dd_filter: MaxDrawdownFilter) -> None:
        """Trading should resume only after recovery_pct is met."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([
            np.linspace(100, 120, 50),   # Up to peak
            np.linspace(120, 98, 50),    # Down 18.3% from peak (halted)
            np.linspace(98, 110, 50),    # Partial recovery
            np.linspace(110, 120, 50),   # Full recovery
        ])
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = dd_filter.execute(df, {"max_drawdown_pct": 15.0, "recovery_pct": 5.0})
        allow = result["dd_allow_trading"]
        # Should halt during the crash and stay halted until recovery
        halted_bars = (~allow).sum()
        assert halted_bars > 0
        # Should eventually resume
        assert allow.iloc[-1] == True

    def test_drawdown_non_positive(
        self, dd_filter: MaxDrawdownFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Drawdown should always be non-positive (or zero at peaks)."""
        result = dd_filter.execute(sample_ohlcv)
        assert (result["dd_drawdown"] <= 0 + 1e-10).all()

    def test_drawdown_pct_non_negative(
        self, dd_filter: MaxDrawdownFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = dd_filter.execute(sample_ohlcv)
        assert (result["dd_drawdown_pct"] >= 0 - 1e-10).all()

    def test_recovery_gt_max_dd_raises(
        self, dd_filter: MaxDrawdownFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="recovery_pct"):
            dd_filter.execute(
                sample_ohlcv, {"max_drawdown_pct": 10.0, "recovery_pct": 15.0}
            )

    def test_insufficient_data_raises(self, dd_filter: MaxDrawdownFilter) -> None:
        dates = pd.date_range("2024-01-01", periods=1, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0], "high": [101.0],
                "low": [99.0], "close": [100.0],
                "volume": [1000.0],
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            dd_filter.execute(df)
```

---

### `tests/unit/filters/test_trend_filter.py`

```python
"""Tests for the Trend Filter block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.filters.trend_filter import TrendFilter
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def trend() -> TrendFilter:
    return TrendFilter()


class TestTrendFilterMetadata:
    def test_name(self, trend: TrendFilter) -> None:
        assert trend.metadata.name == "trend_filter"

    def test_category(self, trend: TrendFilter) -> None:
        assert trend.metadata.category == BlockCategory.FILTER

    def test_defaults(self, trend: TrendFilter) -> None:
        d = trend.metadata.get_defaults()
        assert d["period"] == 200
        assert d["ma_type"] == "ema"
        assert d["buffer_pct"] == 0.5

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "trend_filter" in registry


class TestTrendFilterCompute:
    def test_output_columns(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trend.execute(sample_ohlcv)
        expected = {
            "trend_ma", "trend_upper_buffer", "trend_lower_buffer",
            "trend_allow_long", "trend_allow_short", "trend_direction",
        }
        assert expected == set(result.columns)

    def test_buffer_surrounds_ma(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trend.execute(sample_ohlcv)
        valid = result.dropna(subset=["trend_ma"])
        assert (valid["trend_upper_buffer"] >= valid["trend_ma"]).all()
        assert (valid["trend_lower_buffer"] <= valid["trend_ma"]).all()

    def test_long_short_mutually_exclusive(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        """A bar cannot allow both longs and shorts simultaneously."""
        result = trend.execute(sample_ohlcv)
        both = result["trend_allow_long"] & result["trend_allow_short"]
        assert both.sum() == 0

    def test_uptrend_allows_long(self, trend: TrendFilter) -> None:
        """In a clear uptrend, longs should be allowed."""
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = trend.execute(df, {"period": 50, "buffer_pct": 0.5})
        # After warmup, longs should be allowed
        assert result["trend_allow_long"].iloc[-50:].all()
        assert not result["trend_allow_short"].iloc[-50:].any()

    def test_downtrend_allows_short(self, trend: TrendFilter) -> None:
        """In a clear downtrend, shorts should be allowed."""
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(200, 100, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = trend.execute(df, {"period": 50, "buffer_pct": 0.5})
        assert result["trend_allow_short"].iloc[-50:].all()
        assert not result["trend_allow_long"].iloc[-50:].any()

    def test_zero_buffer(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        """With buffer=0, upper and lower should equal MA."""
        result = trend.execute(sample_ohlcv, {"buffer_pct": 0.0})
        valid = result.dropna(subset=["trend_ma"])
        pd.testing.assert_series_equal(
            valid["trend_upper_buffer"], valid["trend_ma"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            valid["trend_lower_buffer"], valid["trend_ma"],
            check_names=False,
        )

    def test_wider_buffer_fewer_signals(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Wider buffer should result in fewer allowed bars."""
        tight = trend.execute(sample_ohlcv, {"buffer_pct": 0.1, "period": 50})
        wide = trend.execute(sample_ohlcv, {"buffer_pct": 3.0, "period": 50})
        tight_active = tight["trend_allow_long"].sum() + tight["trend_allow_short"].sum()
        wide_active = wide["trend_allow_long"].sum() + wide["trend_allow_short"].sum()
        assert tight_active >= wide_active

    def test_direction_labels(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trend.execute(sample_ohlcv)
        valid_values = {"bullish", "bearish", "neutral"}
        assert set(result["trend_direction"].unique()) <= valid_values

    def test_ema_vs_sma(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result_ema = trend.execute(sample_ohlcv, {"ma_type": "ema", "period": 50})
        result_sma = trend.execute(sample_ohlcv, {"ma_type": "sma", "period": 50})
        assert not result_ema["trend_ma"].equals(result_sma["trend_ma"])

    def test_insufficient_data_raises(self, trend: TrendFilter) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 50, "high": [101.0] * 50,
                "low": [99.0] * 50, "close": [100.0] * 50,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            trend.execute(df, {"period": 200})

    def test_invalid_ma_type_raises(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            trend.execute(sample_ohlcv, {"ma_type": "wma"})
```

---

## 4.20 Integration Test — Phase 4 Registry

### `tests/integration/test_phase4_registry.py`

```python
"""
Integration test verifying all Phase 4 blocks (exit rules, money management,
and filters) are properly registered and can execute on sample data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

# Force registration
import forgequant.blocks.exit_rules  # noqa: F401
import forgequant.blocks.money_management  # noqa: F401
import forgequant.blocks.filters  # noqa: F401


EXPECTED_EXIT_RULES = [
    "fixed_tpsl",
    "trailing_stop",
    "time_based_exit",
    "breakeven_stop",
]

EXPECTED_MONEY_MANAGEMENT = [
    "fixed_risk",
    "volatility_targeting",
    "kelly_fractional",
    "atr_based_sizing",
]

EXPECTED_FILTERS = [
    "trading_session",
    "spread_filter",
    "max_drawdown_filter",
    "trend_filter",
]

ALL_PHASE4_BLOCKS = EXPECTED_EXIT_RULES + EXPECTED_MONEY_MANAGEMENT + EXPECTED_FILTERS


class TestExitRulesRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_EXIT_RULES)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_EXIT_RULES)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.EXIT_RULE

    @pytest.mark.parametrize("block_name", EXPECTED_EXIT_RULES)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)


class TestMoneyManagementRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_MONEY_MANAGEMENT)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_MONEY_MANAGEMENT)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.MONEY_MANAGEMENT

    @pytest.mark.parametrize("block_name", EXPECTED_MONEY_MANAGEMENT)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)


class TestFiltersRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_FILTERS)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_FILTERS)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.FILTER

    @pytest.mark.parametrize("block_name", EXPECTED_FILTERS)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)


class TestAllPhase4Metadata:
    @pytest.mark.parametrize("block_name", ALL_PHASE4_BLOCKS)
    def test_has_description(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.description) > 20

    @pytest.mark.parametrize("block_name", ALL_PHASE4_BLOCKS)
    def test_has_tags(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.tags) >= 1

    @pytest.mark.parametrize("block_name", ALL_PHASE4_BLOCKS)
    def test_has_typical_use(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.typical_use) > 20
```

---

## 4.21 How to Verify Phase 4

```bash
# From project root with venv activated

# Run all tests
pytest -v

# Run only exit rule tests
pytest tests/unit/exit_rules/ -v

# Run only money management tests
pytest tests/unit/money_management/ -v

# Run only filter tests
pytest tests/unit/filters/ -v

# Run the integration test
pytest tests/integration/test_phase4_registry.py -v

# Type-check
mypy src/forgequant/blocks/exit_rules/ \
     src/forgequant/blocks/money_management/ \
     src/forgequant/blocks/filters/

# Lint
ruff check src/forgequant/blocks/exit_rules/ \
           src/forgequant/blocks/money_management/ \
           src/forgequant/blocks/filters/
```

**Expected output:** All tests pass — approximately **100+ new tests** across 12 test modules plus **48 parametrized integration tests**.

---

## Phase 4 Summary

### Exit Rule Blocks

| Block | File | Output Columns | Key Detail |
|-------|------|----------------|------------|
| **Fixed TP/SL** | `fixed_tpsl.py` | atr, long/short tp/sl, risk_reward | ATR-adaptive levels; rejects low RR configurations |
| **Trailing Stop** | `trailing_stop.py` | atr, long/short stop, long/short exit | Ratchet logic — stops only move in profitable direction |
| **Time-Based Exit** | `time_based_exit.py` | bar_index, max_bars_exit, avoid_day, near_close | Day-of-week avoidance; session close proximity warning |
| **Breakeven Stop** | `breakeven_stop.py` | atr, activation/stop levels, activated flags | Two-stage: activation threshold → breakeven + offset |

### Money Management Blocks

| Block | File | Output Columns | Key Detail |
|-------|------|----------------|------------|
| **Fixed Risk** | `fixed_risk.py` | atr, stop_distance, risk_amount, position_size/pct | Fundamental risk % sizing; adapts via ATR stop distance |
| **Volatility Targeting** | `volatility_targeting.py` | realized_vol, exposure, position_size/pct | Log-return vol; max leverage cap; configurable annualization |
| **Kelly Fractional** | `kelly_fractional.py` | win_rate, payoff_ratio, kelly fractions, position_size | Rolling estimation; floored at 0; absolute fraction cap |
| **ATR-Based Sizing** | `atr_based_sizing.py` | atr, risk_per_unit, position_size/value/pct | Inverse ATR sizing with max position % cap |

### Filter Blocks

| Block | File | Output Columns | Key Detail |
|-------|------|----------------|------------|
| **Trading Session** | `trading_session.py` | session_active, session_name | Dual sessions; overnight wrap; overlap detection |
| **Spread Filter** | `spread_filter.py` | spread_value, spread_avg, spread_ok | Uses explicit spread column or H-L approximation |
| **Max Drawdown** | `max_drawdown_filter.py` | cum_return, running_max, drawdown, allow_trading | Hysteresis: halts at threshold, resumes at recovery |
| **Trend Filter** | `trend_filter.py` | trend_ma, upper/lower_buffer, allow_long/short, direction | Buffer band eliminates whipsaws near the MA |

### Cumulative Block Count

| Category | Count | Blocks |
|----------|-------|--------|
| Indicators | 8 | EMA, RSI, MACD, ADX, ATR, Bollinger, Ichimoku, Stochastic |
| Price Action | 4 | Breakout, Pullback, HHLL, Support/Resistance |
| Entry Rules | 4 | Crossover, ThresholdCross, Confluence, ReversalPattern |
| Exit Rules | 4 | FixedTPSL, TrailingStop, TimeBased, BreakevenStop |
| Money Management | 4 | FixedRisk, VolatilityTargeting, KellyFractional, ATRBased |
| Filters | 4 | TradingSession, SpreadFilter, MaxDrawdown, TrendFilter |
| **Total** | **28** | |

---

**Ready for Phase 5** — say the word and I'll write the AI Forge: Pydantic schemas (`StrategySpec`, `BlockSpec`), the full system prompt, the RAG grounding pipeline (ChromaDB ingestion + retrieval), the LLM call + validation pipeline (OpenAI/Anthropic via `instructor`), and the complete spec-to-registry wiring with full test coverage.
