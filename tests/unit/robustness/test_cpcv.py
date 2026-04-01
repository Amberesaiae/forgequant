"""Tests for Combinatorial Purged Cross-Validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.core.robustness.cpcv import CPCVAnalysis, CPCVResult


def _make_equity(n: int = 500, drift: float = 0.001) -> tuple[pd.Series, pd.Series]:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = np.random.normal(drift, 0.01, n)
    equity = 100000 * np.cumprod(1.0 + returns)
    return pd.Series(equity, index=dates), pd.Series(returns, index=dates)


class TestCPCVInit:
    def test_invalid_n_groups(self) -> None:
        with pytest.raises(ValueError, match="n_groups"):
            CPCVAnalysis(n_groups=2)

    def test_invalid_n_test_groups(self) -> None:
        with pytest.raises(ValueError, match="n_test_groups"):
            CPCVAnalysis(n_groups=5, n_test_groups=5)

    def test_valid_init(self) -> None:
        cpcv = CPCVAnalysis(n_groups=6, n_test_groups=2)
        assert cpcv._n_groups == 6


class TestCPCVAnalyse:
    def test_basic_analysis(self) -> None:
        equity, returns = _make_equity(500)
        cpcv = CPCVAnalysis(n_groups=5, n_test_groups=1, random_seed=42)
        result = cpcv.analyse(equity, returns)

        assert result.n_groups == 5
        assert result.n_test_groups == 1
        assert result.n_combinations_evaluated > 0
        assert len(result.folds) > 0

    def test_positive_drift_low_pbo(self) -> None:
        equity, returns = _make_equity(500, drift=0.002)
        cpcv = CPCVAnalysis(n_groups=5, n_test_groups=1, random_seed=42, max_pbo=0.8)
        result = cpcv.analyse(equity, returns)

        assert result.pbo_probability < 0.8

    def test_purge_gap_applied(self) -> None:
        equity, returns = _make_equity(500)
        cpcv = CPCVAnalysis(n_groups=5, n_test_groups=1, purge_gap=10, random_seed=42)
        result = cpcv.analyse(equity, returns)

        assert result.purge_gap == 10
        assert result.n_combinations_evaluated > 0

    def test_max_combinations_cap(self) -> None:
        equity, returns = _make_equity(500)
        cpcv = CPCVAnalysis(
            n_groups=10, n_test_groups=2, max_combinations=10, random_seed=42
        )
        result = cpcv.analyse(equity, returns)

        assert result.n_combinations_evaluated <= 10
        assert result.n_combinations_total >= result.n_combinations_evaluated

    def test_summary(self) -> None:
        equity, returns = _make_equity(500)
        cpcv = CPCVAnalysis(n_groups=5, n_test_groups=1, random_seed=42)
        result = cpcv.analyse(equity, returns)

        s = result.summary()
        assert "pbo_probability" in s
        assert "avg_oos_sharpe" in s
        assert "is_passed" in s

    def test_insufficient_data_raises(self) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        equity = pd.Series(np.ones(20) * 100000, index=dates)
        returns = pd.Series(np.zeros(20), index=dates)
        cpcv = CPCVAnalysis(n_groups=10)
        with pytest.raises(ValueError, match="Insufficient"):
            cpcv.analyse(equity, returns)
