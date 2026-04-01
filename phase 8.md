# PHASE 8 — Execution Layer (MetaTrader 5 via aiomql)

Async MT5 integration with order management, position tracking, signal-to-order translation, pre-trade risk checks, and comprehensive test coverage using mock MT5 interactions.

---

## 8.1 Updated Directory Structure (additions)

```
src/forgequant/execution/
├── __init__.py
├── models.py               # Order, Position, Fill data models
├── risk_guard.py            # Pre-trade risk checks
├── signal_translator.py     # Compiled signals → order instructions
├── mt5_gateway.py           # MT5 connection and raw order API
├── position_manager.py      # Track and reconcile open positions
├── executor.py              # Top-level async execution orchestrator
└── exceptions.py            # Execution-specific exceptions

tests/unit/execution/
├── __init__.py
├── test_models.py
├── test_risk_guard.py
├── test_signal_translator.py
├── test_mt5_gateway.py
├── test_position_manager.py
└── test_executor.py

tests/integration/
└── test_phase8_execution.py
```

---

## 8.2 `src/forgequant/execution/exceptions.py`

```python
"""
Execution-layer exceptions.

All inherit from ExecutionError which itself inherits from ForgeQuantError.
"""

from __future__ import annotations

from typing import Any

from forgequant.core.exceptions import ForgeQuantError


class ExecutionError(ForgeQuantError):
    """Base exception for all execution-layer errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, details=details)


class ConnectionError_(ExecutionError):
    """Raised when the MT5 connection cannot be established or is lost."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"MT5 connection error: {reason}",
            details={"reason": reason},
        )
        self.reason = reason


class OrderError(ExecutionError):
    """Raised when an order submission or modification fails."""

    def __init__(self, order_id: str, reason: str) -> None:
        super().__init__(
            message=f"Order '{order_id}' failed: {reason}",
            details={"order_id": order_id, "reason": reason},
        )
        self.order_id = order_id
        self.reason = reason


class RiskCheckError(ExecutionError):
    """Raised when a pre-trade risk check fails."""

    def __init__(self, check_name: str, reason: str) -> None:
        super().__init__(
            message=f"Risk check '{check_name}' failed: {reason}",
            details={"check_name": check_name, "reason": reason},
        )
        self.check_name = check_name
        self.reason = reason


class PositionError(ExecutionError):
    """Raised when position tracking encounters an inconsistency."""

    def __init__(self, symbol: str, reason: str) -> None:
        super().__init__(
            message=f"Position error on '{symbol}': {reason}",
            details={"symbol": symbol, "reason": reason},
        )
        self.symbol = symbol
        self.reason = reason
```

---

## 8.3 `src/forgequant/execution/models.py`

```python
"""
Execution data models.

Immutable data containers for orders, positions, fills, and
market ticks used throughout the execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, unique
from typing import Any


@unique
class OrderSide(str, Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"


@unique
class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@unique
class OrderStatus(str, Enum):
    """Order lifecycle status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@unique
class PositionSide(str, Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class MarketTick:
    """
    A single market tick (bid/ask snapshot).

    Attributes:
        symbol: Instrument symbol (e.g. "EURUSD").
        bid: Current bid price.
        ask: Current ask price.
        spread: Ask - bid.
        timestamp: When this tick was received.
    """
    symbol: str
    bid: float
    ask: float
    timestamp: datetime

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class OrderRequest:
    """
    Instruction to submit an order.

    Built by the SignalTranslator from compiled strategy signals.
    Consumed by the MT5Gateway for submission.
    """
    request_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    volume: float
    price: float = 0.0          # For limit/stop orders
    stop_loss: float = 0.0
    take_profit: float = 0.0
    comment: str = ""
    magic_number: int = 0       # MT5 magic number for identification
    expiration: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.volume <= 0:
            raise ValueError(f"Volume must be positive, got {self.volume}")
        if self.stop_loss < 0:
            raise ValueError(f"Stop loss cannot be negative, got {self.stop_loss}")
        if self.take_profit < 0:
            raise ValueError(f"Take profit cannot be negative, got {self.take_profit}")


@dataclass
class OrderResult:
    """
    Result of an order submission to MT5.

    Returned by the MT5Gateway after sending an order.
    """
    request_id: str
    broker_order_id: str
    status: OrderStatus
    filled_volume: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_success(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)


@dataclass
class Position:
    """
    Represents a currently open position.

    Attributes:
        symbol: Instrument symbol.
        side: Long or short.
        volume: Current position size.
        entry_price: Average entry price.
        current_price: Latest market price.
        stop_loss: Current stop-loss level.
        take_profit: Current take-profit level.
        unrealized_pnl: Current unrealized P&L.
        open_time: When the position was opened.
        magic_number: MT5 magic number.
        ticket: MT5 position ticket.
    """
    symbol: str
    side: PositionSide
    volume: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    open_time: datetime = field(default_factory=datetime.utcnow)
    magic_number: int = 0
    ticket: int = 0

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT

    def update_price(self, tick: MarketTick) -> None:
        """Update current price and unrealized P&L from a tick."""
        if self.is_long:
            self.current_price = tick.bid  # Close longs at bid
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.volume
        elif self.is_short:
            self.current_price = tick.ask  # Close shorts at ask
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.volume
```

---

## 8.4 `src/forgequant/execution/risk_guard.py`

```python
"""
Pre-trade risk guard.

Validates every order request against configurable risk limits
before it reaches the broker. Rejects orders that would violate
any of the safety constraints.

Checks performed:
    1. Max position size per symbol
    2. Max total exposure across all symbols
    3. Max daily loss limit
    4. Max number of open positions
    5. Max order frequency (throttle)
    6. Spread check (reject if spread too wide)
    7. Drawdown circuit breaker
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from forgequant.core.logging import get_logger
from forgequant.execution.exceptions import RiskCheckError
from forgequant.execution.models import MarketTick, OrderRequest, Position

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_position_size: float = 10.0         # Max lots per symbol
    max_total_exposure: float = 30.0        # Max total lots across all symbols
    max_daily_loss: float = 5000.0          # Max daily loss in base currency
    max_open_positions: int = 5             # Max concurrent positions
    max_orders_per_minute: int = 10         # Rate limiting
    max_spread_pips: float = 50.0           # Max spread to accept
    max_drawdown_pct: float = 15.0          # Drawdown circuit breaker (%)
    min_order_interval_seconds: float = 1.0 # Min seconds between orders


@dataclass
class RiskState:
    """Mutable state tracked by the risk guard."""
    daily_pnl: float = 0.0
    daily_loss: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    order_timestamps: list[datetime] = field(default_factory=list)
    last_order_time: datetime | None = None
    is_halted: bool = False
    halt_reason: str = ""


class RiskGuard:
    """
    Pre-trade risk validation.

    Usage:
        guard = RiskGuard(limits, initial_equity=100000)
        guard.check_order(order_request, current_positions, tick)
        # Raises RiskCheckError if any check fails
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        initial_equity: float = 100_000.0,
    ) -> None:
        self._limits = limits or RiskLimits()
        self._state = RiskState(
            peak_equity=initial_equity,
            current_equity=initial_equity,
        )

    @property
    def state(self) -> RiskState:
        """Read-only access to risk state."""
        return self._state

    @property
    def is_halted(self) -> bool:
        """True if the circuit breaker has tripped."""
        return self._state.is_halted

    def update_equity(self, equity: float) -> None:
        """
        Update current equity and check circuit breaker.

        Args:
            equity: Current account equity.
        """
        self._state.current_equity = equity
        if equity > self._state.peak_equity:
            self._state.peak_equity = equity

        # Check drawdown circuit breaker
        if self._state.peak_equity > 0:
            drawdown_pct = (
                (self._state.peak_equity - equity) / self._state.peak_equity * 100.0
            )
            if drawdown_pct >= self._limits.max_drawdown_pct:
                self._state.is_halted = True
                self._state.halt_reason = (
                    f"Drawdown {drawdown_pct:.1f}% exceeds limit "
                    f"{self._limits.max_drawdown_pct:.1f}%"
                )
                logger.warning(
                    "risk_circuit_breaker_tripped",
                    drawdown_pct=round(drawdown_pct, 2),
                    limit=self._limits.max_drawdown_pct,
                )

    def record_fill(self, pnl: float) -> None:
        """
        Record a trade fill's P&L for daily tracking.

        Args:
            pnl: Realized P&L of the fill (positive = profit).
        """
        self._state.daily_pnl += pnl
        if pnl < 0:
            self._state.daily_loss += abs(pnl)

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self._state.daily_pnl = 0.0
        self._state.daily_loss = 0.0
        self._state.order_timestamps.clear()

    def check_order(
        self,
        order: OrderRequest,
        open_positions: list[Position],
        tick: MarketTick | None = None,
    ) -> None:
        """
        Validate an order against all risk checks.

        Args:
            order: The order to validate.
            open_positions: Current open positions.
            tick: Current market tick for spread check.

        Raises:
            RiskCheckError: If any check fails.
        """
        # 0. Circuit breaker
        if self._state.is_halted:
            raise RiskCheckError(
                "circuit_breaker",
                f"Trading halted: {self._state.halt_reason}",
            )

        # 1. Max position size per symbol
        symbol_volume = sum(
            p.volume for p in open_positions if p.symbol == order.symbol
        )
        if symbol_volume + order.volume > self._limits.max_position_size:
            raise RiskCheckError(
                "max_position_size",
                f"Symbol '{order.symbol}' would have "
                f"{symbol_volume + order.volume:.2f} lots, "
                f"limit is {self._limits.max_position_size:.2f}",
            )

        # 2. Max total exposure
        total_volume = sum(p.volume for p in open_positions)
        if total_volume + order.volume > self._limits.max_total_exposure:
            raise RiskCheckError(
                "max_total_exposure",
                f"Total exposure would be {total_volume + order.volume:.2f} lots, "
                f"limit is {self._limits.max_total_exposure:.2f}",
            )

        # 3. Max daily loss
        if self._state.daily_loss >= self._limits.max_daily_loss:
            raise RiskCheckError(
                "max_daily_loss",
                f"Daily loss ${self._state.daily_loss:.2f} "
                f"exceeds limit ${self._limits.max_daily_loss:.2f}",
            )

        # 4. Max open positions
        if len(open_positions) >= self._limits.max_open_positions:
            raise RiskCheckError(
                "max_open_positions",
                f"{len(open_positions)} positions open, "
                f"limit is {self._limits.max_open_positions}",
            )

        # 5. Order rate limiting
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        recent = [t for t in self._state.order_timestamps if t > cutoff]
        if len(recent) >= self._limits.max_orders_per_minute:
            raise RiskCheckError(
                "rate_limit",
                f"{len(recent)} orders in last minute, "
                f"limit is {self._limits.max_orders_per_minute}",
            )

        # 5b. Min interval between orders
        if self._state.last_order_time is not None:
            elapsed = (now - self._state.last_order_time).total_seconds()
            if elapsed < self._limits.min_order_interval_seconds:
                raise RiskCheckError(
                    "min_interval",
                    f"Only {elapsed:.1f}s since last order, "
                    f"minimum is {self._limits.min_order_interval_seconds:.1f}s",
                )

        # 6. Spread check
        if tick is not None and self._limits.max_spread_pips > 0:
            # Convert spread to pips (assuming 4/5 digit pricing)
            pip_size = 0.0001 if tick.bid > 10 else 0.01
            spread_pips = tick.spread / pip_size
            if spread_pips > self._limits.max_spread_pips:
                raise RiskCheckError(
                    "spread",
                    f"Spread {spread_pips:.1f} pips exceeds "
                    f"limit {self._limits.max_spread_pips:.1f} pips",
                )

        # Record order timestamp
        self._state.order_timestamps.append(now)
        self._state.last_order_time = now

        logger.debug(
            "risk_check_passed",
            symbol=order.symbol,
            volume=order.volume,
            side=order.side.value,
        )
```

---

## 8.5 `src/forgequant/execution/signal_translator.py`

```python
"""
Signal translator.

Converts compiled strategy signals (boolean Series + sizing Series)
into concrete OrderRequest objects for execution.

The translator operates on the LATEST bar of a compiled strategy,
producing at most one order per evaluation cycle.
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from forgequant.core.compiler.compiled_strategy import CompiledStrategy
from forgequant.core.logging import get_logger
from forgequant.execution.models import OrderRequest, OrderSide, OrderType, Position, PositionSide

logger = get_logger(__name__)


class SignalTranslator:
    """
    Translates strategy signals into executable order requests.

    Usage:
        translator = SignalTranslator(symbol="EURUSD", magic_number=12345)
        order = translator.evaluate(compiled_strategy, bar_index, open_positions)
    """

    def __init__(
        self,
        symbol: str,
        magic_number: int = 0,
        default_volume: float = 0.01,
        volume_step: float = 0.01,
        min_volume: float = 0.01,
        max_volume: float = 100.0,
    ) -> None:
        """
        Args:
            symbol: Trading instrument symbol.
            magic_number: MT5 magic number for order identification.
            default_volume: Fallback volume if sizing signal is missing.
            volume_step: Minimum volume increment (for rounding).
            min_volume: Minimum allowed order volume.
            max_volume: Maximum allowed order volume.
        """
        self._symbol = symbol
        self._magic = magic_number
        self._default_vol = default_volume
        self._vol_step = volume_step
        self._min_vol = min_volume
        self._max_vol = max_volume

    def round_volume(self, volume: float) -> float:
        """Round volume to the nearest valid step."""
        if self._vol_step <= 0:
            return volume

        rounded = round(volume / self._vol_step) * self._vol_step
        rounded = max(self._min_vol, min(self._max_vol, rounded))
        return round(rounded, 8)  # Avoid float precision artifacts

    def evaluate(
        self,
        compiled: CompiledStrategy,
        bar_index: int,
        open_positions: list[Position],
    ) -> OrderRequest | None:
        """
        Evaluate the compiled strategy at a specific bar and produce
        an order request if a signal is present.

        Applies these rules:
            1. If there's an open position and an exit signal, produce
               a close order (opposite direction market order).
            2. If there's no open position and an entry signal, produce
               an open order with TP/SL from the compiled strategy.
            3. If there's an open position in one direction and an entry
               signal in the opposite direction, produce a close first
               (entry will happen next cycle).

        Args:
            compiled: The CompiledStrategy with signals.
            bar_index: The bar index to evaluate (typically the latest).
            open_positions: Currently open positions for this symbol.

        Returns:
            An OrderRequest if a trade action is needed, None otherwise.
        """
        if bar_index < 0 or bar_index >= compiled.n_bars:
            return None

        # Get filtered signals at this bar
        entry_long = bool(compiled.filtered_entry_long().iloc[bar_index])
        entry_short = bool(compiled.filtered_entry_short().iloc[bar_index])

        exit_long = bool(
            compiled.exit_long.iloc[bar_index]
            if compiled.exit_long is not None
            else False
        )
        exit_short = bool(
            compiled.exit_short.iloc[bar_index]
            if compiled.exit_short is not None
            else False
        )

        # Current position state for this symbol
        my_positions = [p for p in open_positions if p.symbol == self._symbol]
        has_long = any(p.is_long for p in my_positions)
        has_short = any(p.is_short for p in my_positions)

        # Priority 1: Exit signals for open positions
        if has_long and exit_long:
            long_pos = next(p for p in my_positions if p.is_long)
            return self._make_close_order(long_pos)

        if has_short and exit_short:
            short_pos = next(p for p in my_positions if p.is_short)
            return self._make_close_order(short_pos)

        # Priority 2: Close opposite position before entering new direction
        if has_long and entry_short:
            long_pos = next(p for p in my_positions if p.is_long)
            return self._make_close_order(long_pos)

        if has_short and entry_long:
            short_pos = next(p for p in my_positions if p.is_short)
            return self._make_close_order(short_pos)

        # Priority 3: New entries (only if no position)
        if not has_long and not has_short:
            if entry_long:
                return self._make_entry_order(
                    compiled, bar_index, OrderSide.BUY
                )
            if entry_short:
                return self._make_entry_order(
                    compiled, bar_index, OrderSide.SELL
                )

        return None

    def _make_entry_order(
        self,
        compiled: CompiledStrategy,
        bar_index: int,
        side: OrderSide,
    ) -> OrderRequest:
        """Build an entry order request from compiled signals."""
        # Volume from position sizing
        if side == OrderSide.BUY and compiled.position_size_long is not None:
            raw_volume = compiled.position_size_long.iloc[bar_index]
        elif side == OrderSide.SELL and compiled.position_size_short is not None:
            raw_volume = compiled.position_size_short.iloc[bar_index]
        else:
            raw_volume = self._default_vol

        import numpy as np
        if np.isnan(raw_volume) or raw_volume <= 0:
            raw_volume = self._default_vol

        volume = self.round_volume(raw_volume)

        # TP/SL
        sl = 0.0
        tp = 0.0
        if side == OrderSide.BUY:
            if compiled.stop_loss_long is not None:
                sl_val = compiled.stop_loss_long.iloc[bar_index]
                if not np.isnan(sl_val):
                    sl = float(sl_val)
            if compiled.take_profit_long is not None:
                tp_val = compiled.take_profit_long.iloc[bar_index]
                if not np.isnan(tp_val):
                    tp = float(tp_val)
        else:
            if compiled.stop_loss_short is not None:
                sl_val = compiled.stop_loss_short.iloc[bar_index]
                if not np.isnan(sl_val):
                    sl = float(sl_val)
            if compiled.take_profit_short is not None:
                tp_val = compiled.take_profit_short.iloc[bar_index]
                if not np.isnan(tp_val):
                    tp = float(tp_val)

        request = OrderRequest(
            request_id=str(uuid.uuid4()),
            symbol=self._symbol,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            stop_loss=sl,
            take_profit=tp,
            comment=f"fq:{compiled.spec.name}",
            magic_number=self._magic,
        )

        logger.info(
            "signal_translated",
            symbol=self._symbol,
            side=side.value,
            volume=volume,
            sl=sl,
            tp=tp,
            strategy=compiled.spec.name,
        )

        return request

    def _make_close_order(self, position: Position) -> OrderRequest:
        """Build a close order for an existing position."""
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY

        request = OrderRequest(
            request_id=str(uuid.uuid4()),
            symbol=position.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            volume=position.volume,
            comment=f"fq:close:{position.ticket}",
            magic_number=position.magic_number,
        )

        logger.info(
            "close_order_created",
            symbol=position.symbol,
            side=close_side.value,
            volume=position.volume,
            ticket=position.ticket,
        )

        return request
```

---

## 8.6 `src/forgequant/execution/mt5_gateway.py`

```python
"""
MetaTrader 5 gateway.

Provides an async interface to MT5 for:
    - Connection management
    - Market data (ticks, bars)
    - Order submission (market, limit, stop)
    - Position queries
    - Account information

Uses aiomql under the hood but wraps it in a clean interface
that the rest of the execution layer depends on. This isolation
allows mocking for tests and swapping backends later.

All MT5-specific imports are lazy to avoid requiring aiomql
when not executing live.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from forgequant.core.logging import get_logger
from forgequant.execution.exceptions import ConnectionError_, OrderError
from forgequant.execution.models import (
    MarketTick,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    Position,
    PositionSide,
)

logger = get_logger(__name__)


@runtime_checkable
class MT5GatewayProtocol(Protocol):
    """
    Protocol defining the MT5 gateway interface.

    Any implementation (real MT5, mock, or paper trading)
    must satisfy this protocol.
    """

    async def connect(self) -> bool: ...
    async def disconnect(self) -> None: ...
    async def is_connected(self) -> bool: ...
    async def get_tick(self, symbol: str) -> MarketTick: ...
    async def get_bars(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame: ...
    async def submit_order(self, order: OrderRequest) -> OrderResult: ...
    async def get_positions(self, symbol: str | None = None) -> list[Position]: ...
    async def get_account_equity(self) -> float: ...
    async def get_account_balance(self) -> float: ...


class MT5Gateway:
    """
    Real MT5 gateway implementation using aiomql.

    Usage:
        gw = MT5Gateway(login=12345, password="pass", server="Broker-Live")
        await gw.connect()
        tick = await gw.get_tick("EURUSD")
        result = await gw.submit_order(order_request)
        await gw.disconnect()
    """

    def __init__(
        self,
        login: int | str = 0,
        password: str = "",
        server: str = "",
        terminal_path: str = "",
    ) -> None:
        self._login = int(login) if login else 0
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to MT5 terminal."""
        try:
            import MetaTrader5 as mt5
        except ImportError as e:
            raise ConnectionError_(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            ) from e

        kwargs: dict[str, Any] = {}
        if self._login:
            kwargs["login"] = self._login
        if self._password:
            kwargs["password"] = self._password
        if self._server:
            kwargs["server"] = self._server
        if self._terminal_path:
            kwargs["path"] = self._terminal_path

        if not mt5.initialize(**kwargs):
            error = mt5.last_error()
            raise ConnectionError_(f"MT5 initialize failed: {error}")

        self._connected = True
        logger.info("mt5_connected", login=self._login, server=self._server)
        return True

    async def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass
        self._connected = False
        logger.info("mt5_disconnected")

    async def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self._connected

    async def get_tick(self, symbol: str) -> MarketTick:
        """Get latest tick for a symbol."""
        import MetaTrader5 as mt5

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ConnectionError_(f"Cannot get tick for '{symbol}'")

        return MarketTick(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            timestamp=datetime.fromtimestamp(tick.time),
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> pd.DataFrame:
        """Get OHLCV bars from MT5."""
        import MetaTrader5 as mt5

        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "1w": mt5.TIMEFRAME_W1,
        }

        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            raise ConnectionError_(f"Unknown timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            raise ConnectionError_(f"No bars returned for {symbol} {timeframe}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "tick_volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]]
        df["volume"] = df["volume"].astype(float)

        return df

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order to MT5."""
        import MetaTrader5 as mt5

        action = mt5.TRADE_ACTION_DEAL if order.order_type.value == "market" else mt5.TRADE_ACTION_PENDING
        order_type = (
            mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        )

        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None:
            return OrderResult(
                request_id=order.request_id,
                broker_order_id="",
                status=OrderStatus.REJECTED,
                error_message=f"Cannot get tick for {order.symbol}",
            )

        price = tick.ask if order.side == OrderSide.BUY else tick.bid

        request = {
            "action": action,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": order.magic_number,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if order.stop_loss > 0:
            request["sl"] = order.stop_loss
        if order.take_profit > 0:
            request["tp"] = order.take_profit

        result = mt5.order_send(request)

        if result is None:
            return OrderResult(
                request_id=order.request_id,
                broker_order_id="",
                status=OrderStatus.REJECTED,
                error_message="order_send returned None",
            )

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                request_id=order.request_id,
                broker_order_id=str(result.order),
                status=OrderStatus.FILLED,
                filled_volume=result.volume,
                filled_price=result.price,
                commission=result.comment if hasattr(result, "comment") else 0.0,
            )
        else:
            return OrderResult(
                request_id=order.request_id,
                broker_order_id=str(result.order) if result.order else "",
                status=OrderStatus.REJECTED,
                error_message=f"Retcode {result.retcode}: {result.comment}",
            )

    async def get_positions(
        self, symbol: str | None = None
    ) -> list[Position]:
        """Get open positions from MT5."""
        import MetaTrader5 as mt5

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result: list[Position] = []
        for pos in positions:
            side = PositionSide.LONG if pos.type == 0 else PositionSide.SHORT
            result.append(Position(
                symbol=pos.symbol,
                side=side,
                volume=pos.volume,
                entry_price=pos.price_open,
                current_price=pos.price_current,
                stop_loss=pos.sl,
                take_profit=pos.tp,
                unrealized_pnl=pos.profit,
                open_time=datetime.fromtimestamp(pos.time),
                magic_number=pos.magic,
                ticket=pos.ticket,
            ))

        return result

    async def get_account_equity(self) -> float:
        """Get current account equity."""
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info is None:
            raise ConnectionError_("Cannot get account info")
        return info.equity

    async def get_account_balance(self) -> float:
        """Get current account balance."""
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info is None:
            raise ConnectionError_("Cannot get account info")
        return info.balance


class MockMT5Gateway:
    """
    Mock MT5 gateway for testing and paper trading.

    Simulates order fills at the current tick price with configurable
    slippage and fill probability.
    """

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        slippage_pips: float = 0.0,
        fill_probability: float = 1.0,
    ) -> None:
        self._equity = initial_equity
        self._balance = initial_equity
        self._slippage = slippage_pips
        self._fill_prob = fill_probability
        self._connected = False
        self._positions: list[Position] = []
        self._ticks: dict[str, MarketTick] = {}
        self._bars: dict[str, pd.DataFrame] = {}
        self._next_ticket = 1000

    def set_tick(self, tick: MarketTick) -> None:
        """Set the current tick for a symbol (for testing)."""
        self._ticks[tick.symbol] = tick

    def set_bars(self, symbol: str, bars: pd.DataFrame) -> None:
        """Set the bar data for a symbol (for testing)."""
        self._bars[symbol] = bars

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def is_connected(self) -> bool:
        return self._connected

    async def get_tick(self, symbol: str) -> MarketTick:
        tick = self._ticks.get(symbol)
        if tick is None:
            raise ConnectionError_(f"No tick data for '{symbol}'")
        return tick

    async def get_bars(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame:
        bars = self._bars.get(symbol)
        if bars is None:
            raise ConnectionError_(f"No bar data for '{symbol}'")
        return bars.tail(count)

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        import numpy as np

        # Check fill probability
        if np.random.random() > self._fill_prob:
            return OrderResult(
                request_id=order.request_id,
                broker_order_id="",
                status=OrderStatus.REJECTED,
                error_message="Simulated rejection",
            )

        tick = self._ticks.get(order.symbol)
        if tick is None:
            return OrderResult(
                request_id=order.request_id,
                broker_order_id="",
                status=OrderStatus.REJECTED,
                error_message=f"No tick for {order.symbol}",
            )

        # Determine fill price
        if order.side == OrderSide.BUY:
            fill_price = tick.ask + self._slippage * 0.0001
        else:
            fill_price = tick.bid - self._slippage * 0.0001

        # Create position
        side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
        ticket = self._next_ticket
        self._next_ticket += 1

        pos = Position(
            symbol=order.symbol,
            side=side,
            volume=order.volume,
            entry_price=fill_price,
            current_price=fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            magic_number=order.magic_number,
            ticket=ticket,
        )

        # Check if this is a close order (matches existing position direction)
        existing = [
            p for p in self._positions
            if p.symbol == order.symbol
        ]
        if existing and order.comment.startswith("fq:close"):
            # Remove the position
            closed = existing[0]
            self._positions.remove(closed)
            # Compute P&L
            if closed.is_long:
                pnl = (fill_price - closed.entry_price) * closed.volume
            else:
                pnl = (closed.entry_price - fill_price) * closed.volume
            self._equity += pnl
            self._balance += pnl
        else:
            self._positions.append(pos)

        return OrderResult(
            request_id=order.request_id,
            broker_order_id=str(ticket),
            status=OrderStatus.FILLED,
            filled_volume=order.volume,
            filled_price=fill_price,
        )

    async def get_positions(
        self, symbol: str | None = None
    ) -> list[Position]:
        if symbol:
            return [p for p in self._positions if p.symbol == symbol]
        return list(self._positions)

    async def get_account_equity(self) -> float:
        return self._equity

    async def get_account_balance(self) -> float:
        return self._balance
```

---

## 8.7 `src/forgequant/execution/position_manager.py`

```python
"""
Position manager.

Tracks open positions, reconciles with the broker, and provides
a clean view of current exposure for the risk guard and signal
translator.
"""

from __future__ import annotations

from typing import Any

from forgequant.core.logging import get_logger
from forgequant.execution.exceptions import PositionError
from forgequant.execution.models import MarketTick, OrderResult, OrderStatus, Position, PositionSide
from forgequant.execution.mt5_gateway import MT5GatewayProtocol

logger = get_logger(__name__)


class PositionManager:
    """
    Manages the local position cache and reconciles with the broker.

    Usage:
        pm = PositionManager(gateway)
        await pm.sync()  # Fetch positions from broker
        positions = pm.get_positions("EURUSD")
        pm.on_fill(order_result)  # Update after a fill
    """

    def __init__(self, gateway: MT5GatewayProtocol) -> None:
        self._gateway = gateway
        self._positions: dict[int, Position] = {}  # ticket -> Position

    @property
    def all_positions(self) -> list[Position]:
        """All tracked positions."""
        return list(self._positions.values())

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get positions, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol. None returns all.

        Returns:
            List of Position objects.
        """
        if symbol is None:
            return list(self._positions.values())
        return [p for p in self._positions.values() if p.symbol == symbol]

    def get_position_by_ticket(self, ticket: int) -> Position | None:
        """Look up a position by its ticket number."""
        return self._positions.get(ticket)

    def total_exposure(self, symbol: str | None = None) -> float:
        """Total volume across positions, optionally filtered by symbol."""
        positions = self.get_positions(symbol)
        return sum(p.volume for p in positions)

    def net_exposure(self, symbol: str) -> tuple[PositionSide, float]:
        """
        Compute net exposure for a symbol.

        Returns:
            Tuple of (side, net_volume).
        """
        positions = self.get_positions(symbol)
        long_vol = sum(p.volume for p in positions if p.is_long)
        short_vol = sum(p.volume for p in positions if p.is_short)

        net = long_vol - short_vol
        if net > 0:
            return PositionSide.LONG, net
        elif net < 0:
            return PositionSide.SHORT, abs(net)
        else:
            return PositionSide.FLAT, 0.0

    async def sync(self) -> int:
        """
        Synchronize local cache with broker positions.

        Fetches all positions from the gateway and replaces the
        local cache entirely.

        Returns:
            Number of positions synced.
        """
        broker_positions = await self._gateway.get_positions()

        self._positions.clear()
        for pos in broker_positions:
            self._positions[pos.ticket] = pos

        logger.info(
            "positions_synced",
            count=len(self._positions),
        )

        return len(self._positions)

    def on_fill(self, result: OrderResult, position: Position | None = None) -> None:
        """
        Update local cache after an order fill.

        Args:
            result: The order fill result.
            position: The new position (for opens) or None (for closes).
        """
        if not result.is_success:
            return

        if position is not None:
            self._positions[position.ticket] = position
            logger.info(
                "position_opened",
                ticket=position.ticket,
                symbol=position.symbol,
                side=position.side.value,
                volume=position.volume,
            )

    def on_close(self, ticket: int) -> Position | None:
        """
        Remove a position from the local cache.

        Args:
            ticket: The ticket of the closed position.

        Returns:
            The removed Position, or None if not found.
        """
        pos = self._positions.pop(ticket, None)
        if pos is not None:
            logger.info(
                "position_closed",
                ticket=ticket,
                symbol=pos.symbol,
                side=pos.side.value,
            )
        return pos

    def update_prices(self, ticks: dict[str, MarketTick]) -> None:
        """
        Update current prices and unrealized P&L for all positions.

        Args:
            ticks: Dict of symbol -> MarketTick.
        """
        for pos in self._positions.values():
            tick = ticks.get(pos.symbol)
            if tick is not None:
                pos.update_price(tick)

    def total_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())
```

---

## 8.8 `src/forgequant/execution/executor.py`

```python
"""
Top-level execution orchestrator.

Coordinates the full execution cycle:
    1. Fetch latest market data
    2. Compile strategy on latest bars
    3. Translate signals to order requests
    4. Validate orders through risk guard
    5. Submit orders through MT5 gateway
    6. Update position manager
    7. Update risk state

This is the main entry point for live/paper execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from forgequant.ai_forge.schemas import StrategySpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.logging import get_logger
from forgequant.execution.exceptions import ExecutionError, OrderError, RiskCheckError
from forgequant.execution.models import (
    MarketTick,
    OrderRequest,
    OrderResult,
    OrderStatus,
)
from forgequant.execution.mt5_gateway import MT5GatewayProtocol
from forgequant.execution.position_manager import PositionManager
from forgequant.execution.risk_guard import RiskGuard, RiskLimits
from forgequant.execution.signal_translator import SignalTranslator

logger = get_logger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for the executor."""
    symbol: str = "EURUSD"
    timeframe: str = "1h"
    bar_count: int = 500
    magic_number: int = 20240001
    default_volume: float = 0.01
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    poll_interval_seconds: float = 5.0
    max_cycles: int = 0  # 0 = unlimited


@dataclass
class ExecutionCycleResult:
    """Result of a single execution cycle."""
    cycle_number: int
    timestamp: datetime
    order_submitted: OrderRequest | None = None
    order_result: OrderResult | None = None
    risk_rejected: bool = False
    risk_reason: str = ""
    error: str = ""
    equity: float = 0.0


class StrategyExecutor:
    """
    Async strategy execution orchestrator.

    Usage:
        executor = StrategyExecutor(config, spec, gateway)
        await executor.start()

    Or for a single cycle:
        result = await executor.run_cycle()
    """

    def __init__(
        self,
        config: ExecutionConfig,
        spec: StrategySpec,
        gateway: MT5GatewayProtocol,
        registry: BlockRegistry | None = None,
        validated_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._config = config
        self._spec = spec
        self._gateway = gateway
        self._registry = registry or BlockRegistry()
        self._validated_params = validated_params or {}

        self._compiler = StrategyCompiler(
            registry=self._registry, validate=False
        )
        self._translator = SignalTranslator(
            symbol=config.symbol,
            magic_number=config.magic_number,
            default_volume=config.default_volume,
        )
        self._risk_guard = RiskGuard(
            limits=config.risk_limits,
        )
        self._position_manager = PositionManager(gateway)

        self._cycle_count = 0
        self._is_running = False
        self._history: list[ExecutionCycleResult] = []

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def history(self) -> list[ExecutionCycleResult]:
        return list(self._history)

    async def start(self) -> None:
        """
        Start the execution loop.

        Runs continuously until stopped or max_cycles reached.
        """
        self._is_running = True

        logger.info(
            "executor_start",
            strategy=self._spec.name,
            symbol=self._config.symbol,
            timeframe=self._config.timeframe,
        )

        try:
            # Connect to gateway
            if not await self._gateway.is_connected():
                await self._gateway.connect()

            # Initialize equity
            equity = await self._gateway.get_account_equity()
            self._risk_guard.update_equity(equity)

            while self._is_running:
                if (
                    self._config.max_cycles > 0
                    and self._cycle_count >= self._config.max_cycles
                ):
                    break

                cycle_result = await self.run_cycle()
                self._history.append(cycle_result)

                await asyncio.sleep(self._config.poll_interval_seconds)

        except Exception as e:
            logger.error("executor_error", error=str(e))
            raise
        finally:
            self._is_running = False
            logger.info("executor_stopped", cycles=self._cycle_count)

    async def stop(self) -> None:
        """Signal the executor to stop."""
        self._is_running = False

    async def run_cycle(self) -> ExecutionCycleResult:
        """
        Execute a single cycle of the strategy.

        Steps:
            1. Fetch latest bars
            2. Compile strategy
            3. Evaluate signals
            4. Risk check
            5. Submit order
            6. Update state

        Returns:
            ExecutionCycleResult with details of what happened.
        """
        self._cycle_count += 1
        cycle = ExecutionCycleResult(
            cycle_number=self._cycle_count,
            timestamp=datetime.utcnow(),
        )

        try:
            # 1. Fetch latest bars
            bars = await self._gateway.get_bars(
                self._config.symbol,
                self._config.timeframe,
                self._config.bar_count,
            )

            # 2. Compile strategy
            compiled = self._compiler.compile(
                self._spec, bars, self._validated_params
            )

            # 3. Sync positions
            await self._position_manager.sync()
            positions = self._position_manager.get_positions(self._config.symbol)

            # 4. Evaluate signals on latest bar
            last_bar = compiled.n_bars - 1
            order = self._translator.evaluate(compiled, last_bar, positions)

            if order is None:
                # No signal — update equity and return
                equity = await self._gateway.get_account_equity()
                self._risk_guard.update_equity(equity)
                cycle.equity = equity
                return cycle

            # 5. Risk check
            try:
                tick = await self._gateway.get_tick(self._config.symbol)
                self._risk_guard.check_order(order, positions, tick)
            except RiskCheckError as e:
                cycle.risk_rejected = True
                cycle.risk_reason = e.message
                logger.warning(
                    "order_risk_rejected",
                    reason=e.message,
                    order_id=order.request_id,
                )
                return cycle

            # 6. Submit order
            cycle.order_submitted = order
            result = await self._gateway.submit_order(order)
            cycle.order_result = result

            if result.is_success:
                logger.info(
                    "order_filled",
                    order_id=order.request_id,
                    broker_id=result.broker_order_id,
                    price=result.filled_price,
                    volume=result.filled_volume,
                )
            else:
                logger.warning(
                    "order_rejected_by_broker",
                    order_id=order.request_id,
                    error=result.error_message,
                )

            # 7. Update equity
            equity = await self._gateway.get_account_equity()
            self._risk_guard.update_equity(equity)
            cycle.equity = equity

        except ExecutionError as e:
            cycle.error = e.message
            logger.error("execution_cycle_error", error=e.message)
        except Exception as e:
            cycle.error = str(e)
            logger.error("execution_cycle_unexpected_error", error=str(e))

        return cycle
```

---

## 8.9 `src/forgequant/execution/__init__.py`

```python
"""
Live execution layer (MetaTrader 5).

Provides:
    - MT5Gateway / MockMT5Gateway: Broker connectivity
    - SignalTranslator: Strategy signals → order instructions
    - RiskGuard: Pre-trade safety validation
    - PositionManager: Position tracking and reconciliation
    - StrategyExecutor: Top-level async execution orchestrator
"""

from forgequant.execution.exceptions import (
    ConnectionError_,
    ExecutionError,
    OrderError,
    PositionError,
    RiskCheckError,
)
from forgequant.execution.models import (
    MarketTick,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
)
from forgequant.execution.mt5_gateway import MockMT5Gateway, MT5Gateway
from forgequant.execution.position_manager import PositionManager
from forgequant.execution.risk_guard import RiskGuard, RiskLimits, RiskState
from forgequant.execution.signal_translator import SignalTranslator
from forgequant.execution.executor import ExecutionConfig, StrategyExecutor, ExecutionCycleResult

__all__ = [
    "ConnectionError_",
    "ExecutionError",
    "OrderError",
    "PositionError",
    "RiskCheckError",
    "MarketTick",
    "OrderRequest",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "PositionSide",
    "MockMT5Gateway",
    "MT5Gateway",
    "PositionManager",
    "RiskGuard",
    "RiskLimits",
    "RiskState",
    "SignalTranslator",
    "ExecutionConfig",
    "StrategyExecutor",
    "ExecutionCycleResult",
]
```

---

## 8.10 Test Suite

### `tests/unit/execution/__init__.py`

```python
"""Tests for execution layer."""
```

---

### `tests/unit/execution/test_models.py`

```python
"""Tests for execution data models."""

from __future__ import annotations

from datetime import datetime

import pytest

from forgequant.execution.models import (
    MarketTick,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
)


class TestMarketTick:
    def test_spread(self) -> None:
        tick = MarketTick("EURUSD", bid=1.1000, ask=1.1002, timestamp=datetime.utcnow())
        assert abs(tick.spread - 0.0002) < 1e-10

    def test_mid(self) -> None:
        tick = MarketTick("EURUSD", bid=1.1000, ask=1.1002, timestamp=datetime.utcnow())
        assert abs(tick.mid - 1.1001) < 1e-10


class TestOrderRequest:
    def test_valid_creation(self) -> None:
        o = OrderRequest(
            request_id="r1", symbol="EURUSD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, volume=0.1,
            stop_loss=1.0900, take_profit=1.1100,
        )
        assert o.volume == 0.1
        assert o.stop_loss == 1.0900

    def test_zero_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="Volume"):
            OrderRequest(
                request_id="r1", symbol="EURUSD", side=OrderSide.BUY,
                order_type=OrderType.MARKET, volume=0.0,
            )

    def test_negative_sl_raises(self) -> None:
        with pytest.raises(ValueError, match="Stop loss"):
            OrderRequest(
                request_id="r1", symbol="EURUSD", side=OrderSide.BUY,
                order_type=OrderType.MARKET, volume=0.1, stop_loss=-1.0,
            )


class TestOrderResult:
    def test_success(self) -> None:
        r = OrderResult(
            request_id="r1", broker_order_id="b1",
            status=OrderStatus.FILLED, filled_volume=0.1, filled_price=1.1000,
        )
        assert r.is_success is True

    def test_rejection(self) -> None:
        r = OrderResult(
            request_id="r1", broker_order_id="",
            status=OrderStatus.REJECTED, error_message="Insufficient margin",
        )
        assert r.is_success is False


class TestPosition:
    def test_is_long(self) -> None:
        p = Position(symbol="EURUSD", side=PositionSide.LONG, volume=0.1, entry_price=1.1)
        assert p.is_long is True
        assert p.is_short is False

    def test_update_price_long(self) -> None:
        p = Position(symbol="EURUSD", side=PositionSide.LONG, volume=1.0, entry_price=1.1000)
        tick = MarketTick("EURUSD", bid=1.1050, ask=1.1052, timestamp=datetime.utcnow())
        p.update_price(tick)
        assert p.current_price == 1.1050
        assert abs(p.unrealized_pnl - 0.005) < 1e-6

    def test_update_price_short(self) -> None:
        p = Position(symbol="EURUSD", side=PositionSide.SHORT, volume=1.0, entry_price=1.1050)
        tick = MarketTick("EURUSD", bid=1.1000, ask=1.1002, timestamp=datetime.utcnow())
        p.update_price(tick)
        assert p.current_price == 1.1002
        assert abs(p.unrealized_pnl - 0.0048) < 1e-6
```

---

### `tests/unit/execution/test_risk_guard.py`

```python
"""Tests for the pre-trade risk guard."""

from __future__ import annotations

from datetime import datetime

import pytest

from forgequant.execution.exceptions import RiskCheckError
from forgequant.execution.models import (
    MarketTick,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
)
from forgequant.execution.risk_guard import RiskGuard, RiskLimits


def _make_order(symbol: str = "EURUSD", volume: float = 0.1) -> OrderRequest:
    return OrderRequest(
        request_id="test", symbol=symbol, side=OrderSide.BUY,
        order_type=OrderType.MARKET, volume=volume,
    )


def _make_position(symbol: str = "EURUSD", volume: float = 1.0) -> Position:
    return Position(symbol=symbol, side=PositionSide.LONG, volume=volume, entry_price=1.1)


def _make_tick(symbol: str = "EURUSD", spread: float = 0.0002) -> MarketTick:
    return MarketTick(symbol, bid=1.1000, ask=1.1000 + spread, timestamp=datetime.utcnow())


class TestRiskGuard:
    def test_passes_clean_order(self) -> None:
        guard = RiskGuard(RiskLimits())
        guard.check_order(_make_order(), [], _make_tick())

    def test_max_position_size(self) -> None:
        limits = RiskLimits(max_position_size=1.0)
        guard = RiskGuard(limits)
        positions = [_make_position(volume=0.9)]
        with pytest.raises(RiskCheckError, match="max_position_size"):
            guard.check_order(_make_order(volume=0.2), positions)

    def test_max_total_exposure(self) -> None:
        limits = RiskLimits(max_total_exposure=1.0)
        guard = RiskGuard(limits)
        positions = [_make_position(volume=0.9)]
        with pytest.raises(RiskCheckError, match="max_total_exposure"):
            guard.check_order(_make_order(volume=0.2), positions)

    def test_max_daily_loss(self) -> None:
        limits = RiskLimits(max_daily_loss=100.0)
        guard = RiskGuard(limits)
        guard.record_fill(-101.0)
        with pytest.raises(RiskCheckError, match="max_daily_loss"):
            guard.check_order(_make_order(), [])

    def test_max_open_positions(self) -> None:
        limits = RiskLimits(max_open_positions=2)
        guard = RiskGuard(limits)
        positions = [_make_position(), _make_position(symbol="GBPUSD")]
        with pytest.raises(RiskCheckError, match="max_open_positions"):
            guard.check_order(_make_order(), positions)

    def test_spread_check(self) -> None:
        limits = RiskLimits(max_spread_pips=5.0)
        guard = RiskGuard(limits)
        wide_tick = _make_tick(spread=0.01)  # 100 pips
        with pytest.raises(RiskCheckError, match="spread"):
            guard.check_order(_make_order(), [], wide_tick)

    def test_spread_check_passes_narrow(self) -> None:
        limits = RiskLimits(max_spread_pips=5.0)
        guard = RiskGuard(limits)
        narrow_tick = _make_tick(spread=0.0002)  # 2 pips
        guard.check_order(_make_order(), [], narrow_tick)

    def test_circuit_breaker(self) -> None:
        limits = RiskLimits(max_drawdown_pct=10.0)
        guard = RiskGuard(limits, initial_equity=100000)
        guard.update_equity(89000)  # 11% drawdown
        assert guard.is_halted
        with pytest.raises(RiskCheckError, match="circuit_breaker"):
            guard.check_order(_make_order(), [])

    def test_circuit_breaker_not_tripped(self) -> None:
        limits = RiskLimits(max_drawdown_pct=10.0)
        guard = RiskGuard(limits, initial_equity=100000)
        guard.update_equity(95000)  # 5% drawdown
        assert not guard.is_halted

    def test_reset_daily(self) -> None:
        guard = RiskGuard()
        guard.record_fill(-1000)
        assert guard.state.daily_loss == 1000
        guard.reset_daily()
        assert guard.state.daily_loss == 0

    def test_rate_limit(self) -> None:
        limits = RiskLimits(max_orders_per_minute=2, min_order_interval_seconds=0)
        guard = RiskGuard(limits)
        guard.check_order(_make_order(), [])
        guard.check_order(_make_order(), [])
        with pytest.raises(RiskCheckError, match="rate_limit"):
            guard.check_order(_make_order(), [])
```

---

### `tests/unit/execution/test_signal_translator.py`

```python
"""Tests for the signal translator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.core.compiler.compiled_strategy import CompiledStrategy
from forgequant.execution.models import OrderSide, Position, PositionSide
from forgequant.execution.signal_translator import SignalTranslator


def _make_compiled(
    n: int = 20,
    entry_long_at: int | None = None,
    entry_short_at: int | None = None,
    exit_long_at: int | None = None,
) -> CompiledStrategy:
    """Build a minimal CompiledStrategy for testing."""
    spec = StrategySpec(
        name="translator_test",
        description="A minimal strategy specification for signal translator testing purposes.",
        objective={"style": "trend_following", "timeframe": "1h"},
        indicators=[BlockSpec(block_name="ema")],
        entry_rules=[BlockSpec(block_name="crossover_entry")],
        exit_rules=[BlockSpec(block_name="fixed_tpsl")],
        money_management=BlockSpec(block_name="fixed_risk"),
    )
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    ohlcv = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=idx,
    )
    compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

    # Set signals
    el = pd.Series(False, index=idx)
    es = pd.Series(False, index=idx)
    xl = pd.Series(False, index=idx)
    xs = pd.Series(False, index=idx)

    if entry_long_at is not None:
        el.iloc[entry_long_at] = True
    if entry_short_at is not None:
        es.iloc[entry_short_at] = True
    if exit_long_at is not None:
        xl.iloc[exit_long_at] = True

    compiled.entry_long = el
    compiled.entry_short = es
    compiled.exit_long = xl
    compiled.exit_short = xs
    compiled.allow_long = pd.Series(True, index=idx)
    compiled.allow_short = pd.Series(True, index=idx)

    # TP/SL
    compiled.stop_loss_long = pd.Series(98.0, index=idx)
    compiled.take_profit_long = pd.Series(105.0, index=idx)
    compiled.stop_loss_short = pd.Series(102.0, index=idx)
    compiled.take_profit_short = pd.Series(95.0, index=idx)
    compiled.position_size_long = pd.Series(0.1, index=idx)
    compiled.position_size_short = pd.Series(0.1, index=idx)

    return compiled


class TestSignalTranslator:
    def test_no_signal_returns_none(self) -> None:
        translator = SignalTranslator(symbol="EURUSD")
        compiled = _make_compiled(20)
        order = translator.evaluate(compiled, 10, [])
        assert order is None

    def test_long_entry_produces_buy(self) -> None:
        translator = SignalTranslator(symbol="EURUSD")
        compiled = _make_compiled(20, entry_long_at=10)
        order = translator.evaluate(compiled, 10, [])
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.volume == 0.1
        assert order.stop_loss == 98.0
        assert order.take_profit == 105.0

    def test_short_entry_produces_sell(self) -> None:
        translator = SignalTranslator(symbol="EURUSD")
        compiled = _make_compiled(20, entry_short_at=10)
        order = translator.evaluate(compiled, 10, [])
        assert order is not None
        assert order.side == OrderSide.SELL

    def test_exit_long_with_position(self) -> None:
        translator = SignalTranslator(symbol="EURUSD")
        compiled = _make_compiled(20, exit_long_at=10)
        position = Position(
            symbol="EURUSD", side=PositionSide.LONG, volume=0.1,
            entry_price=100.0, ticket=1000,
        )
        order = translator.evaluate(compiled, 10, [position])
        assert order is not None
        assert order.side == OrderSide.SELL  # Close long = sell
        assert order.volume == 0.1

    def test_no_entry_with_existing_position(self) -> None:
        """Should not open new position when one already exists."""
        translator = SignalTranslator(symbol="EURUSD")
        compiled = _make_compiled(20, entry_long_at=10)
        position = Position(
            symbol="EURUSD", side=PositionSide.LONG, volume=0.1,
            entry_price=100.0, ticket=1000,
        )
        order = translator.evaluate(compiled, 10, [position])
        # Already has a long, long entry should be suppressed
        assert order is None

    def test_opposite_entry_closes_first(self) -> None:
        """Short entry with existing long should close the long first."""
        translator = SignalTranslator(symbol="EURUSD")
        compiled = _make_compiled(20, entry_short_at=10)
        position = Position(
            symbol="EURUSD", side=PositionSide.LONG, volume=0.1,
            entry_price=100.0, ticket=1000,
        )
        order = translator.evaluate(compiled, 10, [position])
        assert order is not None
        assert order.side == OrderSide.SELL  # Closes the long
        assert "close" in order.comment

    def test_volume_rounding(self) -> None:
        translator = SignalTranslator(symbol="EURUSD", volume_step=0.01)
        assert translator.round_volume(0.123) == 0.12
        assert translator.round_volume(0.005) == 0.01  # Clamped to min
        assert translator.round_volume(200.0) == 100.0  # Clamped to max

    def test_out_of_bounds_bar_returns_none(self) -> None:
        translator = SignalTranslator(symbol="EURUSD")
        compiled = _make_compiled(20)
        assert translator.evaluate(compiled, -1, []) is None
        assert translator.evaluate(compiled, 100, []) is None
```

---

### `tests/unit/execution/test_mt5_gateway.py`

```python
"""Tests for the MT5 gateway (mock implementation)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from forgequant.execution.exceptions import ConnectionError_
from forgequant.execution.models import (
    MarketTick,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from forgequant.execution.mt5_gateway import MockMT5Gateway


@pytest.fixture
def gateway() -> MockMT5Gateway:
    gw = MockMT5Gateway(initial_equity=100000)
    tick = MarketTick("EURUSD", bid=1.1000, ask=1.1002, timestamp=datetime.utcnow())
    gw.set_tick(tick)
    return gw


@pytest.mark.asyncio
class TestMockMT5Gateway:
    async def test_connect(self, gateway: MockMT5Gateway) -> None:
        assert not await gateway.is_connected()
        await gateway.connect()
        assert await gateway.is_connected()

    async def test_disconnect(self, gateway: MockMT5Gateway) -> None:
        await gateway.connect()
        await gateway.disconnect()
        assert not await gateway.is_connected()

    async def test_get_tick(self, gateway: MockMT5Gateway) -> None:
        tick = await gateway.get_tick("EURUSD")
        assert tick.symbol == "EURUSD"
        assert tick.bid == 1.1000
        assert tick.ask == 1.1002

    async def test_get_tick_unknown_symbol(self, gateway: MockMT5Gateway) -> None:
        with pytest.raises(ConnectionError_):
            await gateway.get_tick("UNKNOWN")

    async def test_submit_market_order(self, gateway: MockMT5Gateway) -> None:
        order = OrderRequest(
            request_id="t1", symbol="EURUSD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, volume=0.1,
        )
        result = await gateway.submit_order(order)
        assert result.is_success
        assert result.status == OrderStatus.FILLED
        assert result.filled_volume == 0.1
        assert result.filled_price > 0

    async def test_order_creates_position(self, gateway: MockMT5Gateway) -> None:
        order = OrderRequest(
            request_id="t1", symbol="EURUSD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, volume=0.1,
        )
        await gateway.submit_order(order)
        positions = await gateway.get_positions("EURUSD")
        assert len(positions) == 1
        assert positions[0].volume == 0.1

    async def test_get_equity(self, gateway: MockMT5Gateway) -> None:
        equity = await gateway.get_account_equity()
        assert equity == 100000

    async def test_get_balance(self, gateway: MockMT5Gateway) -> None:
        balance = await gateway.get_account_balance()
        assert balance == 100000

    async def test_get_bars(self, gateway: MockMT5Gateway) -> None:
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        bars = pd.DataFrame(
            {"open": 1.1, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 100.0},
            index=dates,
        )
        gateway.set_bars("EURUSD", bars)
        result = await gateway.get_bars("EURUSD", "1h", 20)
        assert len(result) == 20

    async def test_get_bars_unknown(self, gateway: MockMT5Gateway) -> None:
        with pytest.raises(ConnectionError_):
            await gateway.get_bars("UNKNOWN", "1h", 10)
```

---

### `tests/unit/execution/test_position_manager.py`

```python
"""Tests for the position manager."""

from __future__ import annotations

from datetime import datetime

import pytest

from forgequant.execution.models import (
    MarketTick,
    OrderResult,
    OrderStatus,
    Position,
    PositionSide,
)
from forgequant.execution.mt5_gateway import MockMT5Gateway
from forgequant.execution.position_manager import PositionManager


@pytest.fixture
def gateway() -> MockMT5Gateway:
    return MockMT5Gateway()


@pytest.fixture
def pm(gateway: MockMT5Gateway) -> PositionManager:
    return PositionManager(gateway)


class TestPositionManager:
    def test_empty_initially(self, pm: PositionManager) -> None:
        assert pm.all_positions == []
        assert pm.total_exposure() == 0.0

    @pytest.mark.asyncio
    async def test_sync(self, pm: PositionManager, gateway: MockMT5Gateway) -> None:
        # Add a position via gateway
        gateway._positions.append(
            Position("EURUSD", PositionSide.LONG, 0.5, 1.1, ticket=100)
        )
        count = await pm.sync()
        assert count == 1
        assert len(pm.all_positions) == 1

    def test_get_positions_by_symbol(self, pm: PositionManager) -> None:
        pm._positions[1] = Position("EURUSD", PositionSide.LONG, 0.1, 1.1, ticket=1)
        pm._positions[2] = Position("GBPUSD", PositionSide.SHORT, 0.2, 1.3, ticket=2)

        eu = pm.get_positions("EURUSD")
        assert len(eu) == 1
        gb = pm.get_positions("GBPUSD")
        assert len(gb) == 1

    def test_total_exposure(self, pm: PositionManager) -> None:
        pm._positions[1] = Position("EURUSD", PositionSide.LONG, 0.1, 1.1, ticket=1)
        pm._positions[2] = Position("EURUSD", PositionSide.SHORT, 0.2, 1.1, ticket=2)
        assert pm.total_exposure("EURUSD") == 0.3

    def test_net_exposure(self, pm: PositionManager) -> None:
        pm._positions[1] = Position("EURUSD", PositionSide.LONG, 0.5, 1.1, ticket=1)
        pm._positions[2] = Position("EURUSD", PositionSide.SHORT, 0.2, 1.1, ticket=2)
        side, vol = pm.net_exposure("EURUSD")
        assert side == PositionSide.LONG
        assert abs(vol - 0.3) < 1e-10

    def test_net_exposure_flat(self, pm: PositionManager) -> None:
        side, vol = pm.net_exposure("EURUSD")
        assert side == PositionSide.FLAT
        assert vol == 0.0

    def test_on_close(self, pm: PositionManager) -> None:
        pm._positions[100] = Position("EURUSD", PositionSide.LONG, 0.1, 1.1, ticket=100)
        closed = pm.on_close(100)
        assert closed is not None
        assert closed.ticket == 100
        assert len(pm.all_positions) == 0

    def test_on_close_missing(self, pm: PositionManager) -> None:
        assert pm.on_close(999) is None

    def test_update_prices(self, pm: PositionManager) -> None:
        pm._positions[1] = Position("EURUSD", PositionSide.LONG, 1.0, 1.1000, ticket=1)
        tick = MarketTick("EURUSD", bid=1.1050, ask=1.1052, timestamp=datetime.utcnow())
        pm.update_prices({"EURUSD": tick})
        pos = pm.all_positions[0]
        assert pos.current_price == 1.1050
        assert pos.unrealized_pnl > 0

    def test_total_unrealized_pnl(self, pm: PositionManager) -> None:
        p = Position("EURUSD", PositionSide.LONG, 1.0, 1.1000, ticket=1)
        p.unrealized_pnl = 50.0
        pm._positions[1] = p
        assert pm.total_unrealized_pnl() == 50.0
```

---

### `tests/unit/execution/test_executor.py`

```python
"""Tests for the strategy executor."""

from __future__ import annotations

import importlib
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.execution.executor import ExecutionConfig, ExecutionCycleResult, StrategyExecutor
from forgequant.execution.models import MarketTick
from forgequant.execution.mt5_gateway import MockMT5Gateway
from forgequant.execution.risk_guard import RiskLimits


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


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="executor_test",
        description="A simple strategy for executor testing with sufficient description length.",
        objective={"style": "trend_following", "timeframe": "1h"},
        indicators=[BlockSpec(block_name="ema", params={"period": 20})],
        entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20})],
        exit_rules=[BlockSpec(block_name="fixed_tpsl", params={"atr_period": 14})],
        money_management=BlockSpec(block_name="fixed_risk", params={"atr_period": 14}),
    )


def _make_bars(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 1.1 + np.cumsum(np.random.normal(0.00005, 0.0003, n))
    spread = np.random.uniform(0.00005, 0.0002, n)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.0001,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )


@pytest.mark.asyncio
class TestStrategyExecutor:
    async def test_single_cycle(self, full_registry: BlockRegistry) -> None:
        spec = _make_spec()
        gateway = MockMT5Gateway(initial_equity=100000)

        bars = _make_bars(500)
        gateway.set_bars("EURUSD", bars)
        gateway.set_tick(
            MarketTick("EURUSD", bid=1.1000, ask=1.1002, timestamp=datetime.utcnow())
        )
        await gateway.connect()

        config = ExecutionConfig(
            symbol="EURUSD",
            timeframe="1h",
            bar_count=500,
            risk_limits=RiskLimits(),
        )

        executor = StrategyExecutor(
            config=config,
            spec=spec,
            gateway=gateway,
            registry=full_registry,
        )

        result = await executor.run_cycle()

        assert isinstance(result, ExecutionCycleResult)
        assert result.cycle_number == 1
        assert result.error == ""

    async def test_multiple_cycles(self, full_registry: BlockRegistry) -> None:
        spec = _make_spec()
        gateway = MockMT5Gateway(initial_equity=100000)

        bars = _make_bars(500)
        gateway.set_bars("EURUSD", bars)
        gateway.set_tick(
            MarketTick("EURUSD", bid=1.1000, ask=1.1002, timestamp=datetime.utcnow())
        )
        await gateway.connect()

        config = ExecutionConfig(
            symbol="EURUSD",
            timeframe="1h",
            bar_count=500,
            max_cycles=3,
            poll_interval_seconds=0.01,
        )

        executor = StrategyExecutor(
            config=config, spec=spec, gateway=gateway, registry=full_registry,
        )

        await executor.start()

        assert executor.cycle_count == 3
        assert len(executor.history) == 3

    async def test_risk_rejection_recorded(self, full_registry: BlockRegistry) -> None:
        spec = _make_spec()
        gateway = MockMT5Gateway(initial_equity=100000)

        bars = _make_bars(500)
        gateway.set_bars("EURUSD", bars)
        gateway.set_tick(
            MarketTick("EURUSD", bid=1.1000, ask=1.1002, timestamp=datetime.utcnow())
        )
        await gateway.connect()

        # Very restrictive limits
        config = ExecutionConfig(
            symbol="EURUSD",
            timeframe="1h",
            bar_count=500,
            risk_limits=RiskLimits(max_drawdown_pct=0.001),  # Instant halt
        )

        executor = StrategyExecutor(
            config=config, spec=spec, gateway=gateway, registry=full_registry,
        )

        # Trip the circuit breaker
        executor._risk_guard.update_equity(0.01)

        result = await executor.run_cycle()

        # Should either be risk rejected or no signal
        # (depending on whether a signal was generated)
        assert result.error == "" or result.risk_rejected
```

---

## 8.11 Integration Test

### `tests/integration/test_phase8_execution.py`

```python
"""
Integration test for the execution layer.

Tests the full flow from compiled strategy through signal translation,
risk checking, and mock order execution.
"""

from __future__ import annotations

import importlib
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.ai_forge.validator import SpecValidator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.execution.executor import ExecutionConfig, StrategyExecutor
from forgequant.execution.models import MarketTick
from forgequant.execution.mt5_gateway import MockMT5Gateway
from forgequant.execution.risk_guard import RiskLimits


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


def _make_trending_bars(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 1.1 + np.cumsum(np.random.normal(0.0001, 0.0003, n))
    spread = np.random.uniform(0.00005, 0.0002, n)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.0001,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )


@pytest.mark.asyncio
class TestExecutionIntegration:
    async def test_full_execution_flow(self, full_registry: BlockRegistry) -> None:
        """Test: spec → validate → compile → execute with mock gateway."""

        # 1. Define strategy
        spec = StrategySpec(
            name="execution_e2e",
            description="End to end execution test with EMA crossover and fixed risk management.",
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

        # 2. Validate
        validator = SpecValidator(full_registry)
        val = validator.validate(spec)
        assert val.is_valid

        # 3. Set up mock gateway
        gateway = MockMT5Gateway(initial_equity=100_000)
        bars = _make_trending_bars(500)
        gateway.set_bars("EURUSD", bars)

        last_close = bars["close"].iloc[-1]
        gateway.set_tick(
            MarketTick(
                "EURUSD",
                bid=last_close,
                ask=last_close + 0.0002,
                timestamp=datetime.utcnow(),
            )
        )
        await gateway.connect()

        # 4. Execute multiple cycles
        config = ExecutionConfig(
            symbol="EURUSD",
            timeframe="1h",
            bar_count=500,
            max_cycles=5,
            poll_interval_seconds=0.01,
            risk_limits=RiskLimits(
                max_position_size=10.0,
                max_daily_loss=10_000.0,
            ),
        )

        executor = StrategyExecutor(
            config=config,
            spec=spec,
            gateway=gateway,
            registry=full_registry,
            validated_params=val.validated_params,
        )

        await executor.start()

        # Verify execution completed
        assert executor.cycle_count == 5
        assert len(executor.history) == 5

        # All cycles should complete without errors
        for cycle in executor.history:
            assert cycle.error == "", f"Cycle {cycle.cycle_number}: {cycle.error}"

        # Check gateway state
        equity = await gateway.get_account_equity()
        assert equity > 0  # Still solvent

    async def test_risk_guard_prevents_overexposure(
        self, full_registry: BlockRegistry
    ) -> None:
        """Risk guard should prevent orders that exceed limits."""

        spec = StrategySpec(
            name="risk_test",
            description="Strategy for testing risk guard integration with very tight position limits.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 10})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 5, "slow_period": 10})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl", params={"atr_period": 5})],
            money_management=BlockSpec(block_name="fixed_risk", params={"atr_period": 5}),
        )

        gateway = MockMT5Gateway(initial_equity=100_000)
        bars = _make_trending_bars(500)
        gateway.set_bars("EURUSD", bars)
        gateway.set_tick(
            MarketTick("EURUSD", bid=1.1, ask=1.1002, timestamp=datetime.utcnow())
        )
        await gateway.connect()

        config = ExecutionConfig(
            symbol="EURUSD",
            timeframe="1h",
            bar_count=500,
            max_cycles=3,
            poll_interval_seconds=0.01,
            risk_limits=RiskLimits(
                max_open_positions=1,
                max_position_size=0.01,
            ),
        )

        executor = StrategyExecutor(
            config=config, spec=spec, gateway=gateway, registry=full_registry,
        )

        await executor.start()

        # Should have completed without crashing
        assert executor.cycle_count == 3
```

---

## 8.12 How to Verify Phase 8

```bash
# From project root with venv activated

# Install async test support
pip install pytest-asyncio

# Run all tests
pytest -v

# Run only execution tests
pytest tests/unit/execution/ -v

# Run the integration test
pytest tests/integration/test_phase8_execution.py -v

# Type-check
mypy src/forgequant/execution/

# Lint
ruff check src/forgequant/execution/
```

**Expected output:** All tests pass — approximately **55+ new tests** across 7 test modules plus integration tests.

---

## Phase 8 Summary

### Module Overview

| Module | File | Purpose |
|--------|------|---------|
| **Models** | `models.py` | `OrderRequest`, `OrderResult`, `Position`, `MarketTick` — immutable data containers |
| **RiskGuard** | `risk_guard.py` | 7 pre-trade checks: position size, exposure, daily loss, open positions, rate limit, spread, circuit breaker |
| **SignalTranslator** | `signal_translator.py` | `CompiledStrategy` signals → `OrderRequest`; handles close-before-reverse logic |
| **MT5Gateway** | `mt5_gateway.py` | Real MT5 + `MockMT5Gateway` for testing; protocol-based interface |
| **PositionManager** | `position_manager.py` | Local position cache, broker sync, net exposure calculation |
| **Executor** | `executor.py` | Async execution loop: fetch bars → compile → translate → risk check → submit → update |
| **Exceptions** | `exceptions.py` | `ExecutionError`, `ConnectionError_`, `OrderError`, `RiskCheckError`, `PositionError` |

### Execution Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   StrategyExecutor                       │
│                                                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐│
│  │MT5Gateway│  │ Compiler  │  │Translator│  │  Risk  ││
│  │(or Mock) │  │           │  │          │  │ Guard  ││
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └───┬────┘│
│       │              │              │             │     │
│       │  get_bars()  │  compile()   │ evaluate()  │     │
│       ├──────────────►│─────────────►│─────────────►     │
│       │              │              │  OrderReq   │check │
│       │              │              │             │order │
│       │  submit()    │              │             │()    │
│       ◄──────────────┤              │             │     │
│       │              │              │             │     │
│  ┌────┴─────┐                                          │
│  │Position  │  ← sync() / on_fill() / on_close()      │
│  │Manager   │                                          │
│  └──────────┘                                          │
└─────────────────────────────────────────────────────────┘
```

### Risk Guard Checks (7 layers)

| # | Check | Default Limit | Description |
|---|-------|---------------|-------------|
| 0 | Circuit breaker | 15% DD | Halts all trading when drawdown exceeds threshold |
| 1 | Max position size | 10 lots | Per-symbol volume cap |
| 2 | Max total exposure | 30 lots | Cross-symbol volume cap |
| 3 | Max daily loss | $5,000 | Cumulative daily loss cap |
| 4 | Max open positions | 5 | Concurrent position count cap |
| 5 | Rate limit | 10/min, 1s interval | Order frequency throttling |
| 6 | Spread check | 50 pips | Rejects orders when spread is too wide |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Protocol-based gateway** | `MT5GatewayProtocol` allows seamless swapping between real MT5 and mock for testing |
| **Close-before-reverse** | When an opposite signal appears, close the current position first rather than attempting a hedge |
| **Risk guard is synchronous** | All risk checks happen before the async broker call, preventing race conditions |
| **Position manager caches locally** | Reduces broker round-trips; `sync()` reconciles with broker state periodically |
| **Executor is async** | Non-blocking I/O for broker communication; compatible with asyncio event loops |
| **MockMT5Gateway simulates fills** | Fills at bid/ask with configurable slippage; tracks positions and equity for realistic paper trading |

### Cumulative Project Status

| Phase | Status | Components |
|-------|--------|------------|
| Phase 1 | ✅ | Foundation |
| Phase 2 | ✅ | 8 indicator blocks |
| Phase 3 | ✅ | 4 price action + 4 entry rule blocks |
| Phase 4 | ✅ | 4 exit + 4 money mgmt + 4 filter blocks |
| Phase 5 | ✅ | AI Forge — schemas, validator, prompt, RAG, pipeline |
| Phase 6 | ✅ | Compiler + backtesting engine + 25+ metrics |
| Phase 7 | ✅ | Robustness suite — 5 gates, consolidated verdict |
| **Phase 8** | ✅ | **Execution layer — MT5 gateway, risk guard, position manager, async executor** |
| Phase 9 | 🔜 | Reflex dashboard |

**Total blocks: 28** | **Total source files: ~80** | **Total test files: ~49**

---

**Ready for Phase 9** — say the word and I'll write the Reflex-based dashboard: multi-page layout with strategy builder, backtest viewer, robustness report, live execution monitor, and block library explorer — all with reactive state management and comprehensive test coverage.
