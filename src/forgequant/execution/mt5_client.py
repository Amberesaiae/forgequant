"""
MetaTrader 5 client abstraction.

Wraps the MT5 connection and provides a clean async interface
for market data, order placement, and account info.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MT5Config:
    """Configuration for MT5 connection."""

    login: int = 0
    password: str = ""
    server: str = ""
    timeout: int = 60000
    path: str = ""
    portable: bool = False


@dataclass
class MQLTick:
    """Market tick data."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    time: int
    time_msc: int
    flags: int


@dataclass
class MQLRates:
    """OHLCV bar data from MT5."""

    time: list[int]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    tick_volume: list[int]
    spread: list[int]
    real_volume: list[int]


class MT5Client:
    """
    MetaTrader 5 client wrapper.

    Provides methods for:
    - Connection management (initialize, shutdown)
    - Market data (ticks, rates, symbols)
    - Order placement (market, pending)
    - Position management (close, modify)
    - Account info (balance, equity, margin)

    Usage:
        client = MT5Client(config)
        await client.initialize()
        tick = await client.get_tick("EURUSD")
        await client.shutdown()
    """

    def __init__(self, config: MT5Config | None = None) -> None:
        self._config = config or MT5Config()
        self._initialized = False
        self._mt5: Any = None

    async def initialize(self) -> bool:
        """
        Initialize the MT5 terminal connection.

        Returns:
            True if initialization was successful.
        """
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.error("mt5_import_failed", error="MetaTrader5 package not installed")
            return False

        self._mt5 = mt5

        if not mt5.initialize(
            login=self._config.login if self._config.login else None,
            password=self._config.password if self._config.password else None,
            server=self._config.server if self._config.server else None,
            path=self._config.path if self._config.path else None,
            timeout=self._config.timeout,
            portable=self._config.portable,
        ):
            logger.error("mt5_init_failed", error_code=mt5.last_error())
            return False

        self._initialized = True
        account_info = mt5.account_info()
        if account_info:
            logger.info(
                "mt5_connected",
                login=account_info.login,
                server=account_info.server,
                balance=account_info.balance,
            )
        return True

    async def shutdown(self) -> None:
        """Shutdown the MT5 connection."""
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._initialized = False
            logger.info("mt5_disconnected")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def get_tick(self, symbol: str) -> MQLTick | None:
        """Get the latest tick for a symbol."""
        if not self._initialized or self._mt5 is None:
            return None

        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return MQLTick(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            time=tick.time,
            time_msc=tick.time_msc,
            flags=tick.flags,
        )

    async def get_rates(
        self,
        symbol: str,
        timeframe: int,
        count: int = 1000,
    ) -> MQLRates | None:
        """Get OHLCV rates for a symbol."""
        if not self._initialized or self._mt5 is None:
            return None

        rates = self._mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None

        return MQLRates(
            time=[r[0] for r in rates],
            open=[r[1] for r in rates],
            high=[r[2] for r in rates],
            low=[r[3] for r in rates],
            close=[r[4] for r in rates],
            tick_volume=[r[5] for r in rates],
            spread=[r[6] for r in rates],
            real_volume=[r[7] for r in rates],
        )

    async def get_account_balance(self) -> float:
        """Get current account balance."""
        if not self._initialized or self._mt5 is None:
            return 0.0

        info = self._mt5.account_info()
        if info is None:
            return 0.0
        return float(info.balance)

    async def get_account_equity(self) -> float:
        """Get current account equity."""
        if not self._initialized or self._mt5 is None:
            return 0.0

        info = self._mt5.account_info()
        if info is None:
            return 0.0
        return float(info.equity)

    async def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open positions."""
        if not self._initialized or self._mt5 is None:
            return []

        if symbol:
            positions = self._mt5.positions_get(symbol=symbol)
        else:
            positions = self._mt5.positions_get()

        if positions is None:
            return []

        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": p.type,
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "time": p.time,
            }
            for p in positions
        ]
