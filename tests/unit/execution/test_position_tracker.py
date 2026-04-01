"""Tests for position tracker."""

from __future__ import annotations

from datetime import datetime

import pytest

from forgequant.execution.position_tracker import PositionDirection, PositionRecord, PositionTracker


class TestPositionRecord:
    def test_is_winner_long(self) -> None:
        r = PositionRecord(
            ticket=1, symbol="EURUSD", direction=PositionDirection.LONG,
            entry_price=1.1000, volume=100000,
            exit_price=1.1100, pnl=1000.0,
        )
        assert r.is_winner is True

    def test_is_loser_short(self) -> None:
        r = PositionRecord(
            ticket=2, symbol="EURUSD", direction=PositionDirection.SHORT,
            entry_price=1.1000, volume=100000,
            exit_price=1.1200, pnl=-2000.0,
        )
        assert r.is_winner is False


class TestPositionTracker:
    def test_add_position(self) -> None:
        tracker = PositionTracker()
        record = PositionRecord(
            ticket=1, symbol="EURUSD", direction=PositionDirection.LONG,
            entry_price=1.1000, volume=100000,
            entry_time=datetime(2024, 1, 1),
        )
        tracker.add_position(record)
        assert tracker.total_open == 1
        assert tracker.get_position(1) is record

    def test_close_position(self) -> None:
        tracker = PositionTracker()
        record = PositionRecord(
            ticket=1, symbol="EURUSD", direction=PositionDirection.LONG,
            entry_price=1.1000, volume=100000,
            entry_time=datetime(2024, 1, 1),
        )
        tracker.add_position(record)

        closed = tracker.close_position(
            1, exit_price=1.1100, exit_time=datetime(2024, 1, 2),
        )
        assert closed is not None
        assert abs(closed.pnl - 1000.0) < 1.0  # (1.11 - 1.10) * 100000 = 1000
        assert closed.is_open is False
        assert tracker.total_open == 0

    def test_close_nonexistent(self) -> None:
        tracker = PositionTracker()
        result = tracker.close_position(999, exit_price=1.1000)
        assert result is None

    def test_update_position(self) -> None:
        tracker = PositionTracker()
        record = PositionRecord(
            ticket=1, symbol="EURUSD", direction=PositionDirection.LONG,
            entry_price=1.1000, volume=100000,
        )
        tracker.add_position(record)
        assert tracker.update_position(1, sl=1.0900, tp=1.1200)
        assert record.sl == 1.0900
        assert record.tp == 1.1200

    def test_update_nonexistent(self) -> None:
        tracker = PositionTracker()
        assert tracker.update_position(999, sl=1.0900) is False

    def test_get_open_positions(self) -> None:
        tracker = PositionTracker()
        tracker.add_position(PositionRecord(
            ticket=1, symbol="EURUSD", direction=PositionDirection.LONG,
            entry_price=1.1000, volume=100000,
        ))
        tracker.add_position(PositionRecord(
            ticket=2, symbol="GBPUSD", direction=PositionDirection.SHORT,
            entry_price=1.2500, volume=100000,
        ))
        assert len(tracker.get_open_positions()) == 2
        assert len(tracker.get_open_positions("EURUSD")) == 1

    def test_total_exposure(self) -> None:
        tracker = PositionTracker()
        tracker.add_position(PositionRecord(
            ticket=1, symbol="EURUSD", direction=PositionDirection.LONG,
            entry_price=1.1000, volume=100000,
        ))
        assert abs(tracker.total_exposure - 110000.0) < 1.0

    def test_summary(self) -> None:
        tracker = PositionTracker()
        tracker.add_position(PositionRecord(
            ticket=1, symbol="EURUSD", direction=PositionDirection.LONG,
            entry_price=1.1000, volume=100000,
        ))
        tracker.close_position(1, exit_price=1.1100, exit_time=datetime(2024, 1, 2))

        s = tracker.summary()
        assert s["open_positions"] == 0
        assert s["closed_positions"] == 1
        assert s["win_rate"] == 1.0
