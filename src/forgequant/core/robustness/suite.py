"""
Robustness Suite orchestrator.

Runs all robustness gates on a BacktestResult and produces a
consolidated pass/fail verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from forgequant.core.engine.results import BacktestResult
from forgequant.core.logging import get_logger
from forgequant.core.robustness.cpcv import CPCVAnalysis, CPCVResult
from forgequant.core.robustness.monte_carlo import MonteCarloAnalysis, MonteCarloResult
from forgequant.core.robustness.parameter_sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
)
from forgequant.core.robustness.stability import EquityStability, StabilityResult
from forgequant.core.robustness.walk_forward import WalkForwardAnalysis, WalkForwardResult

logger = get_logger(__name__)


@dataclass
class RobustnessVerdict:
    """Consolidated robustness verdict across all gates."""

    strategy_name: str
    is_passed: bool = False
    gates_passed: int = 0
    gates_total: int = 0

    walk_forward: WalkForwardResult | None = None
    monte_carlo: MonteCarloResult | None = None
    cpcv: CPCVResult | None = None
    parameter_sensitivity: SensitivityResult | None = None
    stability: StabilityResult | None = None

    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "strategy_name": self.strategy_name,
            "is_passed": self.is_passed,
            "gates_passed": self.gates_passed,
            "gates_total": self.gates_total,
        }

        if self.walk_forward:
            result["walk_forward"] = self.walk_forward.summary()
        if self.monte_carlo:
            result["monte_carlo"] = self.monte_carlo.summary()
        if self.cpcv:
            result["cpcv"] = self.cpcv.summary()
        if self.parameter_sensitivity:
            result["parameter_sensitivity"] = self.parameter_sensitivity.summary()
        if self.stability:
            result["stability"] = self.stability.summary()

        if self.errors:
            result["errors"] = self.errors

        return result


@dataclass
class SuiteConfig:
    """Configuration for the robustness suite."""

    wf_n_folds: int = 5
    wf_train_pct: float = 0.70
    wf_min_consistency: float = 0.5
    wf_min_oos_sharpe: float = 0.0

    mc_n_simulations: int = 1000
    mc_p_value_threshold: float = 0.05
    mc_random_seed: int | None = 42

    cpcv_n_groups: int = 10
    cpcv_n_test_groups: int = 2
    cpcv_purge_gap: int = 5
    cpcv_max_pbo: float = 0.5

    stab_min_r_squared: float = 0.7
    stab_min_tail_ratio: float = 0.8
    stab_min_recovery_factor: float = 1.0
    stab_n_regimes: int = 3
    stab_min_regime_consistency: float = 0.5

    annualization_factor: float = 252.0

    run_walk_forward: bool = True
    run_monte_carlo: bool = True
    run_cpcv: bool = True
    run_stability: bool = True
    run_parameter_sensitivity: bool = False


class RobustnessSuite:
    """Orchestrates all robustness tests."""

    def __init__(self, config: SuiteConfig | None = None) -> None:
        self._config = config or SuiteConfig()

    def evaluate(
        self,
        result: BacktestResult,
        sensitivity_backtest_fn: Callable[[str, str, float], float] | None = None,
        sensitivity_params: dict[str, dict[str, float]] | None = None,
    ) -> RobustnessVerdict:
        cfg = self._config
        verdict = RobustnessVerdict(strategy_name=result.strategy_name)

        equity = result.equity_curve
        returns = result.returns_series

        gates_passed = 0
        gates_total = 0

        if cfg.run_walk_forward:
            gates_total += 1
            try:
                wfa = WalkForwardAnalysis(
                    n_folds=cfg.wf_n_folds,
                    train_pct=cfg.wf_train_pct,
                    min_consistency=cfg.wf_min_consistency,
                    min_oos_sharpe=cfg.wf_min_oos_sharpe,
                )
                verdict.walk_forward = wfa.analyse(
                    equity, returns,
                    annualization_factor=cfg.annualization_factor,
                )
                if verdict.walk_forward.is_passed:
                    gates_passed += 1
            except Exception as e:
                verdict.errors.append(f"Walk-forward failed: {e}")

        if cfg.run_monte_carlo:
            gates_total += 1
            try:
                mc = MonteCarloAnalysis(
                    n_simulations=cfg.mc_n_simulations,
                    p_value_threshold=cfg.mc_p_value_threshold,
                    random_seed=cfg.mc_random_seed,
                )
                verdict.monte_carlo = mc.analyse(
                    equity, returns,
                    annualization_factor=cfg.annualization_factor,
                )
                if verdict.monte_carlo.is_passed:
                    gates_passed += 1
            except Exception as e:
                verdict.errors.append(f"Monte Carlo failed: {e}")

        if cfg.run_cpcv:
            gates_total += 1
            try:
                cpcv = CPCVAnalysis(
                    n_groups=cfg.cpcv_n_groups,
                    n_test_groups=cfg.cpcv_n_test_groups,
                    purge_gap=cfg.cpcv_purge_gap,
                    max_pbo=cfg.cpcv_max_pbo,
                )
                verdict.cpcv = cpcv.analyse(
                    equity, returns,
                    annualization_factor=cfg.annualization_factor,
                )
                if verdict.cpcv.is_passed:
                    gates_passed += 1
            except Exception as e:
                verdict.errors.append(f"CPCV failed: {e}")

        if cfg.run_stability:
            gates_total += 1
            try:
                stab = EquityStability(
                    min_r_squared=cfg.stab_min_r_squared,
                    min_tail_ratio=cfg.stab_min_tail_ratio,
                    min_recovery_factor=cfg.stab_min_recovery_factor,
                    n_regimes=cfg.stab_n_regimes,
                    min_regime_consistency=cfg.stab_min_regime_consistency,
                    annualization_factor=cfg.annualization_factor,
                )
                verdict.stability = stab.analyse(equity, returns)
                if verdict.stability.is_passed:
                    gates_passed += 1
            except Exception as e:
                verdict.errors.append(f"Stability failed: {e}")

        if cfg.run_parameter_sensitivity and sensitivity_backtest_fn and sensitivity_params:
            gates_total += 1
            try:
                ps = ParameterSensitivity()
                original_sharpe = result.metrics.get("sharpe_ratio", 0.0)
                verdict.parameter_sensitivity = ps.analyse(
                    original_sharpe=original_sharpe,
                    parameter_specs=sensitivity_params,
                    backtest_fn=sensitivity_backtest_fn,
                )
                if verdict.parameter_sensitivity.is_passed:
                    gates_passed += 1
            except Exception as e:
                verdict.errors.append(f"Parameter sensitivity failed: {e}")

        verdict.gates_passed = gates_passed
        verdict.gates_total = gates_total
        verdict.is_passed = gates_total > 0 and gates_passed == gates_total

        logger.info(
            "robustness_suite_complete",
            strategy=result.strategy_name,
            gates_passed=gates_passed,
            gates_total=gates_total,
            overall_passed=verdict.is_passed,
        )

        return verdict
