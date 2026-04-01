"""Tests for parameter sensitivity analysis."""

from __future__ import annotations

import pytest

from forgequant.core.robustness.parameter_sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
)


class TestParameterSensitivity:
    def test_stable_strategy_passes(self) -> None:
        def backtest_fn(block: str, param: str, value: float) -> float:
            return 1.5

        ps = ParameterSensitivity(
            perturbation_pcts=[10, 20],
            sensitivity_threshold_pct=30.0,
            max_sensitive_ratio=0.3,
        )
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={"ema": {"period": 20.0}, "atr": {"period": 14.0}},
            backtest_fn=backtest_fn,
        )

        assert result.is_passed
        assert result.n_sensitive == 0
        assert result.sensitivity_ratio == 0.0

    def test_sensitive_strategy_fails(self) -> None:
        def backtest_fn(block: str, param: str, value: float) -> float:
            return 0.0

        ps = ParameterSensitivity(
            perturbation_pcts=[10],
            sensitivity_threshold_pct=30.0,
            max_sensitive_ratio=0.3,
        )
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={"ema": {"period": 20.0}},
            backtest_fn=backtest_fn,
        )

        assert not result.is_passed
        assert result.n_sensitive > 0

    def test_partial_sensitivity(self) -> None:
        call_log: list[tuple[str, str, float]] = []

        def backtest_fn(block: str, param: str, value: float) -> float:
            call_log.append((block, param, value))
            if block == "ema":
                return 1.5
            return 0.0

        ps = ParameterSensitivity(
            perturbation_pcts=[20],
            sensitivity_threshold_pct=30.0,
        )
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={
                "ema": {"period": 20.0},
                "tpsl": {"tp_mult": 2.5},
            },
            backtest_fn=backtest_fn,
        )

        assert result.n_perturbations > 0
        assert result.n_sensitive > 0
        # ema perturbations should NOT be sensitive (always returns 1.5)
        ema_perts = [p for p in result.perturbations if p.block_name == "ema"]
        assert all(not p.is_sensitive for p in ema_perts)
        # tpsl perturbations SHOULD be sensitive (returns 0.0)
        tpsl_perts = [p for p in result.perturbations if p.block_name == "tpsl"]
        assert all(p.is_sensitive for p in tpsl_perts)

    def test_backtest_fn_failure_counts_as_sensitive(self) -> None:
        def backtest_fn(block: str, param: str, value: float) -> float:
            raise RuntimeError("Backtest failed")

        ps = ParameterSensitivity(perturbation_pcts=[10])
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={"ema": {"period": 20.0}},
            backtest_fn=backtest_fn,
        )

        assert result.n_sensitive > 0

    def test_zero_original_value_skipped(self) -> None:
        def backtest_fn(block: str, param: str, value: float) -> float:
            return 1.5

        ps = ParameterSensitivity(perturbation_pcts=[10])
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={"block": {"param": 0.0}},
            backtest_fn=backtest_fn,
        )

        assert result.n_perturbations == 0

    def test_summary(self) -> None:
        def backtest_fn(b: str, p: str, v: float) -> float:
            return 1.5

        ps = ParameterSensitivity(perturbation_pcts=[10])
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={"ema": {"period": 20.0}},
            backtest_fn=backtest_fn,
        )

        s = result.summary()
        assert "n_perturbations" in s
        assert "is_passed" in s
        assert "most_sensitive" in s
