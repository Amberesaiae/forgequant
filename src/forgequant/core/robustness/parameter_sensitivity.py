"""
Parameter Sensitivity Analysis.

Tests whether small perturbations to parameters cause large
changes in performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerturbationResult:
    """Result of a single parameter perturbation."""

    block_name: str
    param_name: str
    original_value: float
    perturbed_value: float
    perturbation_pct: float
    original_sharpe: float
    perturbed_sharpe: float
    sharpe_change_pct: float
    is_sensitive: bool


@dataclass
class SensitivityResult:
    """Aggregated parameter sensitivity result."""

    n_perturbations: int
    n_sensitive: int
    sensitivity_ratio: float
    perturbations: list[PerturbationResult] = field(default_factory=list)
    most_sensitive_param: str = ""
    most_sensitive_block: str = ""
    is_passed: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "n_perturbations": self.n_perturbations,
            "n_sensitive": self.n_sensitive,
            "sensitivity_ratio": round(self.sensitivity_ratio, 4),
            "most_sensitive": f"{self.most_sensitive_block}.{self.most_sensitive_param}",
            "is_passed": self.is_passed,
        }


class ParameterSensitivity:
    """Parameter sensitivity analysis."""

    def __init__(
        self,
        perturbation_pcts: list[float] | None = None,
        sensitivity_threshold_pct: float = 30.0,
        max_sensitive_ratio: float = 0.3,
    ) -> None:
        self._perts = perturbation_pcts or [10.0, 20.0]
        self._threshold = sensitivity_threshold_pct
        self._max_ratio = max_sensitive_ratio

    def analyse(
        self,
        original_sharpe: float,
        parameter_specs: dict[str, dict[str, float]],
        backtest_fn: Callable[[str, str, float], float],
    ) -> SensitivityResult:
        perturbations: list[PerturbationResult] = []
        max_change = 0.0
        most_sensitive_block = ""
        most_sensitive_param = ""

        for block_name, params in parameter_specs.items():
            for param_name, original_value in params.items():
                if original_value == 0:
                    continue

                for pct in self._perts:
                    for sign in [-1, 1]:
                        pert_pct = sign * pct
                        new_value = original_value * (1.0 + pert_pct / 100.0)

                        if isinstance(original_value, int) or (
                            isinstance(original_value, float) and original_value == int(original_value)
                        ):
                            new_value = round(new_value)
                            if new_value == original_value:
                                continue
                            if new_value < 1:
                                new_value = 1

                        try:
                            perturbed_sharpe = backtest_fn(block_name, param_name, new_value)
                        except Exception:
                            perturbed_sharpe = 0.0

                        if abs(original_sharpe) > 1e-10:
                            change_pct = abs(perturbed_sharpe - original_sharpe) / abs(original_sharpe) * 100.0
                        else:
                            change_pct = abs(perturbed_sharpe) * 100.0

                        is_sensitive = change_pct > self._threshold

                        pr = PerturbationResult(
                            block_name=block_name,
                            param_name=param_name,
                            original_value=original_value,
                            perturbed_value=new_value,
                            perturbation_pct=pert_pct,
                            original_sharpe=original_sharpe,
                            perturbed_sharpe=perturbed_sharpe,
                            sharpe_change_pct=change_pct,
                            is_sensitive=is_sensitive,
                        )
                        perturbations.append(pr)

                        if change_pct > max_change:
                            max_change = change_pct
                            most_sensitive_block = block_name
                            most_sensitive_param = param_name

        n_total = len(perturbations)
        n_sensitive = sum(1 for p in perturbations if p.is_sensitive)
        sensitivity_ratio = n_sensitive / n_total if n_total > 0 else 0.0

        result = SensitivityResult(
            n_perturbations=n_total,
            n_sensitive=n_sensitive,
            sensitivity_ratio=sensitivity_ratio,
            perturbations=perturbations,
            most_sensitive_param=most_sensitive_param,
            most_sensitive_block=most_sensitive_block,
            is_passed=sensitivity_ratio <= self._max_ratio,
        )

        logger.info(
            "parameter_sensitivity_complete",
            n_perturbations=n_total,
            n_sensitive=n_sensitive,
            ratio=round(sensitivity_ratio, 4),
            passed=result.is_passed,
        )

        return result
