"""Tests for walk-forward analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.core.robustness.walk_forward import WalkForwardAnalysis, WalkForwardResult


def _make_equity(n: int = 500, drift: float = 0.0005) -> tuple[pd.Series, pd.Series]:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = np.random.normal(drift, 0.01, n)
    equity = 100000 * np.cumprod(1.0 + returns)
    equity_s = pd.Series(equity, index=dates, name="equity")
    returns_s = pd.Series(returns, index=dates, name="returns")
    return equity_s, returns_s


class TestWalkForwardInit:
    def test_invalid_n_folds(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            WalkForwardAnalysis(n_folds=1)

    def test_invalid_train_pct(self) -> None:
        with pytest.raises(ValueError, match="train_pct"):
            WalkForwardAnalysis(train_pct=0.0)


class TestWalkForwardSplits:
    def test_generates_correct_count(self) -> None:
        wfa = WalkForwardAnalysis(n_folds=5)
        splits = wfa.generate_splits(500)
        assert len(splits) == 4

    def test_splits_non_overlapping_test(self) -> None:
        wfa = WalkForwardAnalysis(n_folds=5)
        splits = wfa.generate_splits(500)
        for i in range(len(splits) - 1):
            _, _, ts1, te1 = splits[i]
            _, _, ts2, te2 = splits[i + 1]
            assert te1 <= ts2, "Test windows must not overlap"

    def test_train_before_test(self) -> None:
        wfa = WalkForwardAnalysis(n_folds=5)
        splits = wfa.generate_splits(500)
        for tr_s, tr_e, te_s, te_e in splits:
            assert tr_e <= te_s, "Train must end before test starts"

    def test_insufficient_data_raises(self) -> None:
        wfa = WalkForwardAnalysis(n_folds=10)
        with pytest.raises(ValueError, match="Insufficient"):
            wfa.generate_splits(50)


class TestWalkForwardAnalyse:
    def test_positive_drift_passes(self) -> None:
        equity, returns = _make_equity(500, drift=0.001)
        wfa = WalkForwardAnalysis(n_folds=5, min_consistency=0.3, min_oos_sharpe=-1.0)
        result = wfa.analyse(equity, returns)

        assert result.n_folds > 0
        assert len(result.folds) > 0
        assert result.total_oos_trades > 0

    def test_negative_drift_low_consistency(self) -> None:
        equity, returns = _make_equity(500, drift=-0.002)
        wfa = WalkForwardAnalysis(n_folds=5, min_consistency=0.8, min_oos_sharpe=0.5)
        result = wfa.analyse(equity, returns)

        assert result.oos_returns_consistency < 0.8

    def test_result_has_folds(self) -> None:
        equity, returns = _make_equity(500)
        wfa = WalkForwardAnalysis(n_folds=4)
        result = wfa.analyse(equity, returns)

        for fold in result.folds:
            assert fold.test_start >= fold.train_end
            assert fold.test_end > fold.test_start
            assert fold.oos_n_trades >= 0

    def test_summary(self) -> None:
        equity, returns = _make_equity(500)
        wfa = WalkForwardAnalysis(n_folds=5)
        result = wfa.analyse(equity, returns)

        s = result.summary()
        assert "n_folds" in s
        assert "avg_oos_return_pct" in s
        assert "is_passed" in s

    def test_auto_compute_returns(self) -> None:
        equity, _ = _make_equity(500)
        wfa = WalkForwardAnalysis(n_folds=5)
        result = wfa.analyse(equity)
        assert result.n_folds > 0
