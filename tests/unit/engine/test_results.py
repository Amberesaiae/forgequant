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
        assert df["is_winner"].iloc[0] == True

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
