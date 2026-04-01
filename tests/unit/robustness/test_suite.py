"""Tests for the robustness suite orchestrator."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from forgequant.core.engine.results import BacktestResult
from forgequant.core.robustness.suite import RobustnessSuite, RobustnessVerdict, SuiteConfig


def _make_backtest_result(
    n: int = 500,
    drift: float = 0.001,
) -> BacktestResult:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = np.random.normal(drift, 0.01, n)
    equity = 100000 * np.cumprod(1.0 + returns)
    equity_s = pd.Series(equity, index=dates, name="equity")
    returns_s = pd.Series(returns, index=dates, name="returns")

    return BacktestResult(
        strategy_name="test_strategy",
        start_date=dates[0].to_pydatetime(),
        end_date=dates[-1].to_pydatetime(),
        initial_equity=100000,
        final_equity=float(equity[-1]),
        equity_curve=equity_s,
        returns_series=returns_s,
        metrics={"sharpe_ratio": 1.5},
    )


class TestSuiteConfig:
    def test_defaults(self) -> None:
        cfg = SuiteConfig()
        assert cfg.run_walk_forward is True
        assert cfg.run_monte_carlo is True
        assert cfg.run_cpcv is True
        assert cfg.run_stability is True
        assert cfg.run_parameter_sensitivity is False


class TestRobustnessSuite:
    def test_runs_all_gates(self) -> None:
        result = _make_backtest_result(500, drift=0.002)
        config = SuiteConfig(
            mc_n_simulations=200,
            mc_random_seed=42,
            wf_min_consistency=0.2,
            wf_min_oos_sharpe=-5.0,
            stab_min_r_squared=0.3,
            stab_min_tail_ratio=0.3,
            stab_min_recovery_factor=0.1,
            stab_n_regimes=3,
            cpcv_max_pbo=0.9,
        )
        suite = RobustnessSuite(config)
        verdict = suite.evaluate(result)

        assert verdict.strategy_name == "test_strategy"
        assert verdict.gates_total == 4
        assert verdict.walk_forward is not None
        assert verdict.monte_carlo is not None
        assert verdict.cpcv is not None
        assert verdict.stability is not None

    def test_gates_can_be_disabled(self) -> None:
        result = _make_backtest_result()
        config = SuiteConfig(
            run_walk_forward=False,
            run_monte_carlo=False,
            run_cpcv=False,
            run_stability=True,
            stab_min_r_squared=0.1,
            stab_min_tail_ratio=0.1,
            stab_min_recovery_factor=0.01,
        )
        suite = RobustnessSuite(config)
        verdict = suite.evaluate(result)

        assert verdict.gates_total == 1
        assert verdict.walk_forward is None
        assert verdict.monte_carlo is None
        assert verdict.stability is not None

    def test_summary(self) -> None:
        result = _make_backtest_result()
        config = SuiteConfig(
            mc_n_simulations=100,
            mc_random_seed=42,
            cpcv_n_groups=5,
            cpcv_n_test_groups=1,
        )
        suite = RobustnessSuite(config)
        verdict = suite.evaluate(result)

        s = verdict.summary()
        assert "strategy_name" in s
        assert "is_passed" in s
        assert "gates_passed" in s
        assert "gates_total" in s

    def test_with_parameter_sensitivity(self) -> None:
        result = _make_backtest_result()

        def backtest_fn(block: str, param: str, value: float) -> float:
            return 1.5

        config = SuiteConfig(
            run_walk_forward=False,
            run_monte_carlo=False,
            run_cpcv=False,
            run_stability=False,
            run_parameter_sensitivity=True,
        )
        suite = RobustnessSuite(config)
        verdict = suite.evaluate(
            result,
            sensitivity_backtest_fn=backtest_fn,
            sensitivity_params={"ema": {"period": 20.0}},
        )

        assert verdict.gates_total == 1
        assert verdict.parameter_sensitivity is not None
        assert verdict.parameter_sensitivity.is_passed

    def test_error_handling(self) -> None:
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        equity = pd.Series(np.linspace(100000, 101000, 30), index=dates)
        returns = equity.pct_change().fillna(0)

        result = BacktestResult(
            strategy_name="short_data",
            start_date=dates[0].to_pydatetime(),
            end_date=dates[-1].to_pydatetime(),
            initial_equity=100000,
            final_equity=101000,
            equity_curve=equity,
            returns_series=returns,
            metrics={"sharpe_ratio": 1.0},
        )

        config = SuiteConfig(
            wf_n_folds=10,
            mc_n_simulations=100,
            mc_random_seed=42,
            cpcv_n_groups=10,
        )
        suite = RobustnessSuite(config)
        verdict = suite.evaluate(result)

        assert verdict.strategy_name == "short_data"
        if verdict.errors:
            assert any("failed" in e.lower() or "insufficient" in e.lower() for e in verdict.errors)
