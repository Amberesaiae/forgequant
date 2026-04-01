"""Tests for equity curve stability analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.core.robustness.stability import EquityStability, StabilityResult


def _make_equity(
    n: int = 500,
    drift: float = 0.001,
    volatility: float = 0.01,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = np.random.normal(drift, volatility, n)
    equity = 100000 * np.cumprod(1.0 + returns)
    return pd.Series(equity, index=dates), pd.Series(returns, index=dates)


class TestEquityStability:
    def test_straight_line_high_r2(self) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        equity = pd.Series(np.linspace(100000, 120000, n), index=dates)
        returns = equity.pct_change().fillna(0)

        es = EquityStability(min_r_squared=0.9)
        result = es.analyse(equity, returns)

        assert result.r_squared > 0.99

    def test_noisy_equity_lower_r2(self) -> None:
        equity, returns = _make_equity(500, drift=0.001, volatility=0.05)
        es = EquityStability()
        result = es.analyse(equity, returns)

        assert result.r_squared < 0.99

    def test_tail_ratio(self) -> None:
        equity, returns = _make_equity(500, drift=0.001)
        es = EquityStability()
        result = es.analyse(equity, returns)

        assert result.tail_ratio > 0

    def test_recovery_factor(self) -> None:
        equity, returns = _make_equity(500, drift=0.001)
        es = EquityStability()
        result = es.analyse(equity, returns)

        assert result.recovery_factor > 0

    def test_regime_consistency(self) -> None:
        equity, returns = _make_equity(500, drift=0.001)
        es = EquityStability(n_regimes=3)
        result = es.analyse(equity, returns)

        assert result.n_regimes == 3
        assert len(result.regime_sharpes) == 3
        assert 0 <= result.regime_consistency_score <= 1.0

    def test_hurst_exponent_range(self) -> None:
        equity, returns = _make_equity(500)
        es = EquityStability()
        result = es.analyse(equity, returns)

        assert 0.0 <= result.hurst_exponent <= 1.0

    def test_strong_uptrend_passes(self) -> None:
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.003, n)
        equity = 100000 * np.cumprod(1.0 + returns)
        equity_s = pd.Series(equity, index=dates)
        returns_s = pd.Series(returns, index=dates)

        es = EquityStability(
            min_r_squared=0.7,
            min_tail_ratio=0.5,
            min_recovery_factor=0.5,
            min_regime_consistency=0.3,
        )
        result = es.analyse(equity_s, returns_s)

        assert result.is_passed

    def test_summary(self) -> None:
        equity, returns = _make_equity(500)
        es = EquityStability()
        result = es.analyse(equity, returns)

        s = result.summary()
        assert "r_squared" in s
        assert "tail_ratio" in s
        assert "hurst_exponent" in s
        assert "is_passed" in s

    def test_insufficient_data_raises(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        equity = pd.Series(np.ones(10) * 100000, index=dates)
        es = EquityStability()
        with pytest.raises(ValueError, match="Insufficient"):
            es.analyse(equity)
