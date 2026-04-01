"""
Combinatorial Purged Cross-Validation (CPCV).

Generates all C(N, N-k) combinations of contiguous groups for testing,
with a purge gap to prevent information leakage.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CPCVFold:
    """Result of a single CPCV combination."""

    combination_index: int
    test_groups: tuple[int, ...]
    oos_return_pct: float
    oos_sharpe: float


@dataclass
class CPCVResult:
    """Aggregated CPCV result."""

    n_groups: int
    n_test_groups: int
    n_combinations_total: int
    n_combinations_evaluated: int
    purge_gap: int
    folds: list[CPCVFold] = field(default_factory=list)
    avg_oos_return_pct: float = 0.0
    avg_oos_sharpe: float = 0.0
    std_oos_return_pct: float = 0.0
    std_oos_sharpe: float = 0.0
    pbo_probability: float = 0.0
    is_passed: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "n_test_groups": self.n_test_groups,
            "n_combinations_evaluated": self.n_combinations_evaluated,
            "purge_gap": self.purge_gap,
            "avg_oos_return_pct": round(self.avg_oos_return_pct, 4),
            "avg_oos_sharpe": round(self.avg_oos_sharpe, 4),
            "std_oos_sharpe": round(self.std_oos_sharpe, 4),
            "pbo_probability": round(self.pbo_probability, 4),
            "is_passed": self.is_passed,
        }


class CPCVAnalysis:
    """Combinatorial Purged Cross-Validation."""

    def __init__(
        self,
        n_groups: int = 10,
        n_test_groups: int = 2,
        purge_gap: int = 5,
        max_combinations: int = 200,
        min_avg_oos_sharpe: float = 0.0,
        max_pbo: float = 0.5,
        random_seed: int | None = None,
    ) -> None:
        if n_groups < 3:
            raise ValueError("n_groups must be >= 3")
        if n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be >= 1 and < n_groups")

        self._n_groups = n_groups
        self._n_test = n_test_groups
        self._purge_gap = purge_gap
        self._max_combos = max_combinations
        self._min_sharpe = min_avg_oos_sharpe
        self._max_pbo = max_pbo
        self._rng = np.random.RandomState(random_seed)

    def analyse(
        self,
        equity_curve: pd.Series,
        returns_series: pd.Series | None = None,
        annualization_factor: float = 252.0,
    ) -> CPCVResult:
        if returns_series is None:
            returns_series = equity_curve.pct_change().fillna(0.0)

        n = len(returns_series)
        group_size = n // self._n_groups

        if group_size < 10:
            raise ValueError(
                f"Insufficient data: {n} bars / {self._n_groups} groups "
                f"= {group_size} bars per group (need >= 10)"
            )

        group_bounds: list[tuple[int, int]] = []
        for g in range(self._n_groups):
            start = g * group_size
            end = min((g + 1) * group_size, n) if g < self._n_groups - 1 else n
            group_bounds.append((start, end))

        all_combos = list(itertools.combinations(range(self._n_groups), self._n_test))
        total_combos = len(all_combos)

        if total_combos > self._max_combos:
            indices = self._rng.choice(total_combos, self._max_combos, replace=False)
            combos = [all_combos[i] for i in sorted(indices)]
        else:
            combos = all_combos

        folds: list[CPCVFold] = []
        returns_arr = returns_series.values

        for combo_idx, test_groups in enumerate(combos):
            test_set = set(test_groups)

            test_indices: list[int] = []
            for g in test_groups:
                gs, ge = group_bounds[g]
                test_indices.extend(range(gs, ge))

            purged: set[int] = set()
            for g in test_groups:
                gs, ge = group_bounds[g]
                for p in range(max(0, gs - self._purge_gap), gs):
                    purged.add(p)
                for p in range(ge, min(n, ge + self._purge_gap)):
                    purged.add(p)

            oos_returns = returns_arr[test_indices]

            if len(oos_returns) < 5:
                continue

            cum_return = np.prod(1.0 + oos_returns) - 1.0
            oos_return_pct = cum_return * 100.0

            oos_std = np.std(oos_returns)
            if oos_std > 0 and annualization_factor > 0:
                oos_sharpe = (np.mean(oos_returns) / oos_std) * np.sqrt(annualization_factor)
            else:
                oos_sharpe = 0.0

            folds.append(CPCVFold(
                combination_index=combo_idx,
                test_groups=test_groups,
                oos_return_pct=oos_return_pct,
                oos_sharpe=oos_sharpe,
            ))

        result = CPCVResult(
            n_groups=self._n_groups,
            n_test_groups=self._n_test,
            n_combinations_total=total_combos,
            n_combinations_evaluated=len(folds),
            purge_gap=self._purge_gap,
            folds=folds,
        )

        if folds:
            sharpes = [f.oos_sharpe for f in folds]
            returns = [f.oos_return_pct for f in folds]

            result.avg_oos_return_pct = float(np.mean(returns))
            result.avg_oos_sharpe = float(np.mean(sharpes))
            result.std_oos_return_pct = float(np.std(returns))
            result.std_oos_sharpe = float(np.std(sharpes))

            negative_sharpe_count = sum(1 for s in sharpes if s <= 0)
            result.pbo_probability = negative_sharpe_count / len(folds)

            result.is_passed = (
                result.avg_oos_sharpe >= self._min_sharpe
                and result.pbo_probability <= self._max_pbo
            )

        logger.info(
            "cpcv_complete",
            n_evaluated=len(folds),
            avg_oos_sharpe=round(result.avg_oos_sharpe, 4),
            pbo=round(result.pbo_probability, 4),
            passed=result.is_passed,
        )

        return result
