"""
Position tracker for managing open trades.

Tracks all open positions, their entry details, and P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, unique
from typing import Any

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@unique
class PositionDirection(str, Enum):
    """Position direction."""

    LONG = "long"
    SHORT = "short"


@dataclass
class PositionRecord:
    """Record of an open or closed position."""

    ticket: int
    symbol: str
    direction: PositionDirection
    entry_price: float
    volume: float
    sl: float = 0.0
    tp: float = 0.0
    entry_time: datetime | None = None
    exit_price: float = 0.0
    exit_time: datetime | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    magic: int = 0
    comment: str = ""
    is_open: bool = True

    @property
    def bars_held(self) -> int:
        """Number of bars held (approximate, based on time)."""
        if self.entry_time is None or self.exit_time is None:
            return 0
        delta = self.exit_time - self.entry_time
        return int(delta.total_seconds() / 3600)  # Approximate hourly bars

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0.0


class PositionTracker:
    """
    Tracks all positions managed by the execution layer.

    Provides:
    - Position lifecycle tracking (open → modify → close)
    - P&L calculation
    - Exposure monitoring
    - Position lookup by symbol/ticket
    """

    def __init__(self) -> None:
        self._positions: dict[int, PositionRecord] = {}
        self._closed: list[PositionRecord] = []

    def add_position(self, record: PositionRecord) -> None:
        """Add a new open position."""
        self._positions[record.ticket] = record
        logger.info(
            "position_opened",
            ticket=record.ticket,
            symbol=record.symbol,
            direction=record.direction.value,
            volume=record.volume,
            price=record.entry_price,
        )

    def close_position(
        self,
        ticket: int,
        exit_price: float,
        exit_time: datetime | None = None,
    ) -> PositionRecord | None:
        """
        Mark a position as closed and calculate P&L.

        Args:
            ticket: Position ticket number.
            exit_price: Exit price.
            exit_time: Exit time.

        Returns:
            The closed PositionRecord, or None if not found.
        """
        record = self._positions.pop(ticket, None)
        if record is None:
            logger.warning("position_not_found", ticket=ticket)
            return None

        record.exit_price = exit_price
        record.exit_time = exit_time or datetime.now()
        record.is_open = False

        if record.direction == PositionDirection.LONG:
            record.pnl = (exit_price - record.entry_price) * record.volume
        else:
            record.pnl = (record.entry_price - exit_price) * record.volume

        if record.entry_price > 0:
            record.pnl_pct = (record.pnl / (record.entry_price * record.volume)) * 100.0

        self._closed.append(record)

        logger.info(
            "position_closed",
            ticket=ticket,
            symbol=record.symbol,
            pnl=round(record.pnl, 2),
            pnl_pct=round(record.pnl_pct, 2),
        )

        return record

    def update_position(
        self,
        ticket: int,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> bool:
        """Update SL/TP of an open position."""
        record = self._positions.get(ticket)
        if record is None:
            return False

        if sl > 0:
            record.sl = sl
        if tp > 0:
            record.tp = tp

        return True

    def get_open_positions(self, symbol: str | None = None) -> list[PositionRecord]:
        """Get all open positions, optionally filtered by symbol."""
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def get_closed_positions(self, symbol: str | None = None) -> list[PositionRecord]:
        """Get all closed positions, optionally filtered by symbol."""
        positions = self._closed
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def get_position(self, ticket: int) -> PositionRecord | None:
        """Get a position by ticket number."""
        return self._positions.get(ticket)

    @property
    def total_open(self) -> int:
        """Number of open positions."""
        return len(self._positions)

    @property
    def total_exposure(self) -> float:
        """Total exposure across all open positions (volume * entry_price)."""
        return sum(p.volume * p.entry_price for p in self._positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all open positions."""
        return sum(p.pnl for p in self._positions.values())

    def summary(self) -> dict[str, Any]:
        """Return a summary of position tracking state."""
        total_pnl = sum(p.pnl for p in self._closed)
        winners = [p for p in self._closed if p.is_winner]
        losers = [p for p in self._closed if not p.is_winner]

        return {
            "open_positions": self.total_open,
            "closed_positions": len(self._closed),
            "total_exposure": round(self.total_exposure, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(total_pnl, 2),
            "win_rate": len(winners) / len(self._closed) if self._closed else 0.0,
            "avg_winner": sum(p.pnl for p in winners) / len(winners) if winners else 0.0,
            "avg_loser": sum(p.pnl for p in losers) / len(losers) if losers else 0.0,
        }
