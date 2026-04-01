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
        assert result.metrics["max_drawdown_pct"] == 0.0
        assert result.returns_series is not None
        assert result.drawdown_series is not None

    def test_with_drawdown(self) -> None:
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
            _make_trade(100),
            _make_trade(-50),
            _make_trade(200),
            _make_trade(-30),
            _make_trade(80),
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
        assert _max_drawdown_duration(equity) == 2

    def test_long_drawdown(self) -> None:
        equity = pd.Series([100, 105, 104, 103, 102, 101, 106])
        assert _max_drawdown_duration(equity) == 4
