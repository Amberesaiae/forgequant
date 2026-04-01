"""
Monte Carlo Analysis.

Performs return shuffling for p-values and bootstrapping for
confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo analysis."""

    n_simulations: int
    original_return_pct: float
    original_sharpe: float
    original_max_dd_pct: float

    simulated_returns_pct: list[float] = field(default_factory=list)
    simulated_sharpes: list[float] = field(default_factory=list)
    simulated_max_dds_pct: list[float] = field(default_factory=list)

    return_p_value: float = 1.0
    sharpe_p_value: float = 1.0
    return_percentile: float = 0.0
    sharpe_percentile: float = 0.0

    return_ci_lower: float = 0.0
    return_ci_upper: float = 0.0
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    max_dd_ci_lower: float = 0.0
    max_dd_ci_upper: float = 0.0

    is_passed: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "original_return_pct": round(self.original_return_pct, 4),
            "original_sharpe": round(self.original_sharpe, 4),
            "return_p_value": round(self.return_p_value, 4),
            "sharpe_p_value": round(self.sharpe_p_value, 4),
            "return_percentile": round(self.return_percentile, 2),
            "sharpe_percentile": round(self.sharpe_percentile, 2),
            "return_ci_95": (round(self.return_ci_lower, 2), round(self.return_ci_upper, 2)),
            "max_dd_ci_95": (round(self.max_dd_ci_lower, 2), round(self.max_dd_ci_upper, 2)),
            "is_passed": self.is_passed,
        }


class MonteCarloAnalysis:
    """Monte Carlo robustness analysis."""

    def __init__(
        self,
        n_simulations: int = 1000,
        p_value_threshold: float = 0.05,
        confidence_level: float = 0.95,
        random_seed: int | None = None,
    ) -> None:
        if n_simulations < 100:
            raise ValueError("n_simulations must be >= 100")

        self._n_sims = n_simulations
        self._p_threshold = p_value_threshold
        self._ci_level = confidence_level
        self._rng = np.random.RandomState(random_seed)

    def analyse(
        self,
        equity_curve: pd.Series,
        returns_series: pd.Series | None = None,
        annualization_factor: float = 252.0,
    ) -> MonteCarloResult:
        if returns_series is None:
            returns_series = equity_curve.pct_change().fillna(0.0)

        returns_arr = returns_series.values
        n = len(returns_arr)

        if n < 20:
            raise ValueError(f"Insufficient data: need >= 20 bars, got {n}")

        initial_equity = equity_curve.iloc[0]

        orig_return = (equity_curve.iloc[-1] / initial_equity - 1.0) * 100.0
        ret_std = returns_series.std()
        if ret_std > 0 and annualization_factor > 0:
            orig_sharpe = (returns_series.mean() / ret_std) * np.sqrt(annualization_factor)
        else:
            orig_sharpe = 0.0

        running_max = equity_curve.expanding().max()
        orig_max_dd = abs(((equity_curve - running_max) / running_max).min()) * 100.0

        sim_returns: list[float] = []
        sim_sharpes: list[float] = []
        sim_max_dds: list[float] = []

        for _ in range(self._n_sims):
            shuffled = self._rng.permutation(returns_arr)
            sim_equity = initial_equity * np.cumprod(1.0 + shuffled)

            sim_ret = (sim_equity[-1] / initial_equity - 1.0) * 100.0
            sim_returns.append(sim_ret)

            s_std = np.std(shuffled)
            if s_std > 0 and annualization_factor > 0:
                s_sharpe = (np.mean(shuffled) / s_std) * np.sqrt(annualization_factor)
            else:
                s_sharpe = 0.0
            sim_sharpes.append(s_sharpe)

            sim_running_max = np.maximum.accumulate(sim_equity)
            sim_dd = np.min((sim_equity - sim_running_max) / sim_running_max)
            sim_max_dds.append(abs(sim_dd) * 100.0)

        return_p = float(np.mean([1 if sr >= orig_return else 0 for sr in sim_returns]))
        sharpe_p = float(np.mean([1 if ss >= orig_sharpe else 0 for ss in sim_sharpes]))

        return_pctile = float(np.mean([1 if sr < orig_return else 0 for sr in sim_returns]) * 100)
        sharpe_pctile = float(np.mean([1 if ss < orig_sharpe else 0 for ss in sim_sharpes]) * 100)

        alpha = (1.0 - self._ci_level) / 2.0
        lower_q = alpha * 100.0
        upper_q = (1.0 - alpha) * 100.0

        result = MonteCarloResult(
            n_simulations=self._n_sims,
            original_return_pct=orig_return,
            original_sharpe=orig_sharpe,
            original_max_dd_pct=orig_max_dd,
            simulated_returns_pct=sim_returns,
            simulated_sharpes=sim_sharpes,
            simulated_max_dds_pct=sim_max_dds,
            return_p_value=return_p,
            sharpe_p_value=sharpe_p,
            return_percentile=return_pctile,
            sharpe_percentile=sharpe_pctile,
            return_ci_lower=float(np.percentile(sim_returns, lower_q)),
            return_ci_upper=float(np.percentile(sim_returns, upper_q)),
            sharpe_ci_lower=float(np.percentile(sim_sharpes, lower_q)),
            sharpe_ci_upper=float(np.percentile(sim_sharpes, upper_q)),
            max_dd_ci_lower=float(np.percentile(sim_max_dds, lower_q)),
            max_dd_ci_upper=float(np.percentile(sim_max_dds, upper_q)),
        )

        result.is_passed = (
            result.return_p_value <= self._p_threshold
            and result.sharpe_p_value <= self._p_threshold
        )

        logger.info(
            "monte_carlo_complete",
            n_sims=self._n_sims,
            return_p=round(return_p, 4),
            sharpe_p=round(sharpe_p, 4),
            passed=result.is_passed,
        )

        return result
