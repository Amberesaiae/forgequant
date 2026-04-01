"""
Signal translator.

Converts compiled strategy signals into executable trade orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any

import pandas as pd

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@unique
class SignalType(str, Enum):
    """Type of trading signal."""

    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    MODIFY_SL = "modify_sl"
    MODIFY_TP = "modify_tp"


@dataclass
class TradeSignal:
    """A trade signal ready for execution."""

    signal_type: SignalType
    symbol: str
    timestamp: pd.Timestamp
    price: float = 0.0
    volume: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0
    comment: str = ""
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class SignalTranslator:
    """
    Translates compiled strategy signals into TradeSignal objects.

    Takes the CompiledStrategy's signal matrices and produces
    a list of TradeSignal objects for the OrderManager to execute.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        magic: int = 123456,
        default_volume: float = 0.01,
        use_position_sizing: bool = True,
    ) -> None:
        """
        Args:
            symbol: Trading symbol.
            magic: Magic number for order identification.
            default_volume: Default lot size if position sizing is disabled.
            use_position_sizing: Whether to use position size from signals.
        """
        self._symbol = symbol
        self._magic = magic
        self._default_volume = default_volume
        self._use_position_sizing = use_position_sizing

    def translate(
        self,
        entry_long: pd.Series | None = None,
        entry_short: pd.Series | None = None,
        exit_long: pd.Series | None = None,
        exit_short: pd.Series | None = None,
        stop_loss_long: pd.Series | None = None,
        stop_loss_short: pd.Series | None = None,
        take_profit_long: pd.Series | None = None,
        take_profit_short: pd.Series | None = None,
        position_size_long: pd.Series | None = None,
        position_size_short: pd.Series | None = None,
        close_prices: pd.Series | None = None,
    ) -> list[TradeSignal]:
        """
        Translate signal matrices into TradeSignal objects.

        Args:
            entry_long: Boolean Series, True on bars with long entry signal.
            entry_short: Boolean Series, True on bars with short entry signal.
            exit_long: Boolean Series, True on bars with long exit signal.
            exit_short: Boolean Series, True on bars with short exit signal.
            stop_loss_long: Float Series, SL levels for longs.
            stop_loss_short: Float Series, SL levels for shorts.
            take_profit_long: Float Series, TP levels for longs.
            take_profit_short: Float Series, TP levels for shorts.
            position_size_long: Float Series, position sizes for longs.
            position_size_short: Float Series, position sizes for shorts.
            close_prices: Float Series, close prices for reference.

        Returns:
            List of TradeSignal objects.
        """
        signals: list[TradeSignal] = []

        # Process entry long signals
        if entry_long is not None:
            for timestamp, is_signal in entry_long.items():
                if is_signal:
                    price = float(close_prices.loc[timestamp]) if close_prices is not None else 0.0
                    sl = float(stop_loss_long.loc[timestamp]) if stop_loss_long is not None else 0.0
                    tp = float(take_profit_long.loc[timestamp]) if take_profit_long is not None else 0.0
                    volume = self._get_volume(position_size_long, timestamp, price)

                    signals.append(TradeSignal(
                        signal_type=SignalType.ENTRY_LONG,
                        symbol=self._symbol,
                        timestamp=timestamp,
                        price=price,
                        volume=volume,
                        sl=sl,
                        tp=tp,
                        magic=self._magic,
                        comment=f"ForgeQuant long entry at {timestamp}",
                    ))

        # Process entry short signals
        if entry_short is not None:
            for timestamp, is_signal in entry_short.items():
                if is_signal:
                    price = float(close_prices.loc[timestamp]) if close_prices is not None else 0.0
                    sl = float(stop_loss_short.loc[timestamp]) if stop_loss_short is not None else 0.0
                    tp = float(take_profit_short.loc[timestamp]) if take_profit_short is not None else 0.0
                    volume = self._get_volume(position_size_short, timestamp, price)

                    signals.append(TradeSignal(
                        signal_type=SignalType.ENTRY_SHORT,
                        symbol=self._symbol,
                        timestamp=timestamp,
                        price=price,
                        volume=volume,
                        sl=sl,
                        tp=tp,
                        magic=self._magic,
                        comment=f"ForgeQuant short entry at {timestamp}",
                    ))

        # Process exit long signals
        if exit_long is not None:
            for timestamp, is_signal in exit_long.items():
                if is_signal:
                    price = float(close_prices.loc[timestamp]) if close_prices is not None else 0.0
                    signals.append(TradeSignal(
                        signal_type=SignalType.EXIT_LONG,
                        symbol=self._symbol,
                        timestamp=timestamp,
                        price=price,
                        magic=self._magic,
                        comment=f"ForgeQuant long exit at {timestamp}",
                    ))

        # Process exit short signals
        if exit_short is not None:
            for timestamp, is_signal in exit_short.items():
                if is_signal:
                    price = float(close_prices.loc[timestamp]) if close_prices is not None else 0.0
                    signals.append(TradeSignal(
                        signal_type=SignalType.EXIT_SHORT,
                        symbol=self._symbol,
                        timestamp=timestamp,
                        price=price,
                        magic=self._magic,
                        comment=f"ForgeQuant short exit at {timestamp}",
                    ))

        logger.info(
            "signals_translated",
            symbol=self._symbol,
            n_signals=len(signals),
            n_entries=sum(1 for s in signals if s.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT)),
            n_exits=sum(1 for s in signals if s.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT)),
        )

        return signals

    def _get_volume(
        self,
        position_size: pd.Series | None,
        timestamp: pd.Timestamp,
        price: float,
    ) -> float:
        """Get trade volume from position size series or use default."""
        if not self._use_position_sizing or position_size is None:
            return self._default_volume

        try:
            size = float(position_size.loc[timestamp])
            if size > 0 and price > 0:
                # Convert position size (units) to lots (100000 units per standard lot)
                return max(0.01, round(size / 100000, 2))
        except (KeyError, TypeError, ValueError):
            pass

        return self._default_volume
