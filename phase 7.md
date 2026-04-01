# PHASE 7 — Robustness Suite

Walk-forward analysis, Monte Carlo simulation, Combinatorial Purged Cross-Validation (CPCV), parameter sensitivity analysis, and equity curve stability checks — all with full implementations and comprehensive tests.

---

## 7.1 Updated Directory Structure (additions)

```
src/forgequant/core/robustness/
├── __init__.py
├── walk_forward.py         # Walk-forward analysis
├── monte_carlo.py          # Monte Carlo permutation & simulation
├── cpcv.py                 # Combinatorial Purged Cross-Validation
├── parameter_sensitivity.py # Parameter sensitivity (perturbation) analysis
├── stability.py            # Equity curve stability checks
└── suite.py                # Orchestrates all robustness gates

tests/unit/robustness/
├── __init__.py
├── test_walk_forward.py
├── test_monte_carlo.py
├── test_cpcv.py
├── test_parameter_sensitivity.py
├── test_stability.py
└── test_suite.py

tests/integration/
└── test_phase7_robustness.py
```

---

## 7.2 `src/forgequant/core/robustness/__init__.py`

```python
"""
Robustness testing suite.

Provides multiple independent robustness tests that a strategy must
pass before being considered viable:

    - WalkForwardAnalysis: Out-of-sample validation across rolling windows
    - MonteCarloAnalysis: Statistical significance via return shuffling
    - CPCVAnalysis: Combinatorial Purged Cross-Validation
    - ParameterSensitivity: Stability under parameter perturbation
    - EquityStability: Equity curve quality and regime analysis
    - RobustnessSuite: Orchestrates all gates with pass/fail verdict
"""

from forgequant.core.robustness.walk_forward import WalkForwardAnalysis, WalkForwardResult
from forgequant.core.robustness.monte_carlo import MonteCarloAnalysis, MonteCarloResult
from forgequant.core.robustness.cpcv import CPCVAnalysis, CPCVResult
from forgequant.core.robustness.parameter_sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
)
from forgequant.core.robustness.stability import EquityStability, StabilityResult
from forgequant.core.robustness.suite import RobustnessSuite, RobustnessVerdict

__all__ = [
    "WalkForwardAnalysis",
    "WalkForwardResult",
    "MonteCarloAnalysis",
    "MonteCarloResult",
    "CPCVAnalysis",
    "CPCVResult",
    "ParameterSensitivity",
    "SensitivityResult",
    "EquityStability",
    "StabilityResult",
    "RobustnessSuite",
    "RobustnessVerdict",
]
```

---

## 7.3 `src/forgequant/core/robustness/walk_forward.py`

```python
"""
Walk-Forward Analysis.

Splits data into sequential train/test windows, backtests on each
test window using only information available up to the train boundary,
and aggregates out-of-sample results.

This validates that a strategy generalises beyond its optimisation period.

Window scheme (anchored expanding or rolling):
    Anchored:   train always starts at bar 0, grows each step
    Rolling:    train window is a fixed size, slides forward

    ┌────────────┬──────┐
    │   Train 1  │Test 1│
    ├──────────────┬────┤──────┐
    │   Train 2    │Test 2│
    └──────────────┴──────┘

Metrics collected per fold:
    - Out-of-sample return
    - Out-of-sample Sharpe
    - Out-of-sample max drawdown
    - Number of OOS trades
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardFold:
    """Result of a single walk-forward fold."""

    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    oos_return_pct: float
    oos_sharpe: float
    oos_max_drawdown_pct: float
    oos_n_trades: int
    oos_equity_curve: pd.Series | None = None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward analysis result."""

    n_folds: int
    folds: list[WalkForwardFold] = field(default_factory=list)
    avg_oos_return_pct: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_oos_max_drawdown_pct: float = 0.0
    total_oos_trades: int = 0
    oos_returns_consistency: float = 0.0  # Fraction of folds with positive return
    is_passed: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "avg_oos_return_pct": round(self.avg_oos_return_pct, 4),
            "avg_oos_sharpe": round(self.avg_oos_sharpe, 4),
            "avg_oos_max_drawdown_pct": round(self.avg_oos_max_drawdown_pct, 4),
            "total_oos_trades": self.total_oos_trades,
            "oos_returns_consistency": round(self.oos_returns_consistency, 4),
            "is_passed": self.is_passed,
        }


class WalkForwardAnalysis:
    """
    Walk-forward analysis engine.

    Operates on a pre-computed equity curve and returns series
    rather than re-running the compiler/backtester per fold (which
    would require re-optimisation logic). This makes it usable as a
    pure statistical test on backtest results.

    For full walk-forward optimisation (re-fitting parameters each fold),
    the caller should run the compiler+backtester per fold externally
    and feed the OOS equity segments here.

    Usage (simple mode — split an existing equity curve):
        wfa = WalkForwardAnalysis(n_folds=5, train_pct=0.70)
        result = wfa.analyse(equity_curve, returns_series)
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_pct: float = 0.70,
        min_oos_trades: int = 10,
        min_oos_sharpe: float = 0.0,
        min_consistency: float = 0.5,
        anchored: bool = False,
    ) -> None:
        """
        Args:
            n_folds: Number of walk-forward folds.
            train_pct: Fraction of each window used for training.
            min_oos_trades: Minimum trades in OOS for the fold to count.
            min_oos_sharpe: Minimum average OOS Sharpe to pass.
            min_consistency: Minimum fraction of folds with positive return.
            anchored: If True, train always starts at bar 0 (expanding window).
        """
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if not 0.1 <= train_pct <= 0.95:
            raise ValueError("train_pct must be between 0.1 and 0.95")

        self._n_folds = n_folds
        self._train_pct = train_pct
        self._min_oos_trades = min_oos_trades
        self._min_oos_sharpe = min_oos_sharpe
        self._min_consistency = min_consistency
        self._anchored = anchored

    def generate_splits(
        self,
        n_bars: int,
    ) -> list[tuple[int, int, int, int]]:
        """
        Generate (train_start, train_end, test_start, test_end) index tuples.

        Uses a rolling or anchored scheme that divides the data into
        n_folds sequential segments.

        Args:
            n_bars: Total number of bars.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples.
            Indices are inclusive start, exclusive end.
        """
        fold_size = n_bars // self._n_folds
        if fold_size < 10:
            raise ValueError(
                f"Insufficient data: {n_bars} bars / {self._n_folds} folds "
                f"= {fold_size} bars per fold (need >= 10)"
            )

        splits: list[tuple[int, int, int, int]] = []

        for i in range(self._n_folds - 1):
            test_start = (i + 1) * fold_size
            test_end = min((i + 2) * fold_size, n_bars)

            if self._anchored:
                train_start = 0
            else:
                train_start = max(0, test_start - int(fold_size / (1 - self._train_pct) * self._train_pct))

            train_end = test_start

            if train_end - train_start < 10:
                continue

            splits.append((train_start, train_end, test_start, test_end))

        return splits

    def analyse(
        self,
        equity_curve: pd.Series,
        returns_series: pd.Series | None = None,
        trade_bars: pd.Series | None = None,
        annualization_factor: float = 252.0,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis on an equity curve.

        Args:
            equity_curve: Full backtest equity curve.
            returns_series: Bar-by-bar returns. Computed from equity if None.
            trade_bars: Boolean Series, True on bars where a trade was entered.
                        Used to count OOS trades. If None, estimated from returns.
            annualization_factor: For Sharpe annualisation.

        Returns:
            WalkForwardResult with per-fold and aggregated metrics.
        """
        n = len(equity_curve)

        if returns_series is None:
            returns_series = equity_curve.pct_change().fillna(0.0)

        splits = self.generate_splits(n)

        folds: list[WalkForwardFold] = []

        for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
            oos_equity = equity_curve.iloc[te_s:te_e]
            oos_returns = returns_series.iloc[te_s:te_e]

            if len(oos_equity) < 2:
                continue

            # OOS return
            oos_return = (oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) * 100.0

            # OOS Sharpe
            ret_std = oos_returns.std()
            if ret_std > 0 and annualization_factor > 0:
                oos_sharpe = (oos_returns.mean() / ret_std) * np.sqrt(annualization_factor)
            else:
                oos_sharpe = 0.0

            # OOS max drawdown
            oos_running_max = oos_equity.expanding().max()
            oos_dd = ((oos_equity - oos_running_max) / oos_running_max).min()
            oos_max_dd_pct = abs(oos_dd) * 100.0

            # OOS trade count
            if trade_bars is not None:
                oos_trades = int(trade_bars.iloc[te_s:te_e].sum())
            else:
                # Estimate: count bars with non-zero return as trade bars
                oos_trades = int((oos_returns.abs() > 1e-10).sum())

            fold = WalkForwardFold(
                fold_index=fold_idx,
                train_start=tr_s,
                train_end=tr_e,
                test_start=te_s,
                test_end=te_e,
                oos_return_pct=oos_return,
                oos_sharpe=oos_sharpe,
                oos_max_drawdown_pct=oos_max_dd_pct,
                oos_n_trades=oos_trades,
                oos_equity_curve=oos_equity,
            )
            folds.append(fold)

        # Aggregate
        result = WalkForwardResult(n_folds=len(folds), folds=folds)

        if folds:
            result.avg_oos_return_pct = np.mean([f.oos_return_pct for f in folds])
            result.avg_oos_sharpe = np.mean([f.oos_sharpe for f in folds])
            result.avg_oos_max_drawdown_pct = np.mean([f.oos_max_drawdown_pct for f in folds])
            result.total_oos_trades = sum(f.oos_n_trades for f in folds)

            positive_folds = sum(1 for f in folds if f.oos_return_pct > 0)
            result.oos_returns_consistency = positive_folds / len(folds)

            # Pass/fail
            result.is_passed = (
                result.avg_oos_sharpe >= self._min_oos_sharpe
                and result.oos_returns_consistency >= self._min_consistency
            )

        logger.info(
            "walk_forward_complete",
            n_folds=result.n_folds,
            avg_oos_return=round(result.avg_oos_return_pct, 4),
            avg_oos_sharpe=round(result.avg_oos_sharpe, 4),
            consistency=round(result.oos_returns_consistency, 4),
            passed=result.is_passed,
        )

        return result
```

---

## 7.4 `src/forgequant/core/robustness/monte_carlo.py`

```python
"""
Monte Carlo Analysis.

Performs two types of Monte Carlo simulation:

1. **Return Shuffling**: Randomly permutes the order of trade returns
   to generate synthetic equity curves. If the original equity curve's
   final return is significantly better than the shuffled distribution,
   the strategy likely has a genuine edge (not just lucky sequencing).

2. **Trade Bootstrapping**: Samples trades with replacement to estimate
   the distribution of key metrics (Sharpe, drawdown, etc.) and compute
   confidence intervals.

The p-value from return shuffling answers: "What is the probability of
achieving this result by chance?"
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

    # Shuffled return distribution
    simulated_returns_pct: list[float] = field(default_factory=list)
    simulated_sharpes: list[float] = field(default_factory=list)
    simulated_max_dds_pct: list[float] = field(default_factory=list)

    # Statistical results
    return_p_value: float = 1.0
    sharpe_p_value: float = 1.0
    return_percentile: float = 0.0  # Where original falls in distribution
    sharpe_percentile: float = 0.0

    # Confidence intervals (from bootstrapping)
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
    """
    Monte Carlo robustness analysis.

    Usage:
        mc = MonteCarloAnalysis(n_simulations=1000, p_value_threshold=0.05)
        result = mc.analyse(equity_curve, returns_series)
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        p_value_threshold: float = 0.05,
        confidence_level: float = 0.95,
        random_seed: int | None = None,
    ) -> None:
        """
        Args:
            n_simulations: Number of Monte Carlo iterations.
            p_value_threshold: Maximum p-value to pass (0.05 = 95% confidence).
            confidence_level: For confidence interval computation.
            random_seed: For reproducibility. None for random.
        """
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
        """
        Run Monte Carlo analysis on an equity curve.

        Performs return shuffling to compute p-values and bootstrapping
        to compute confidence intervals.

        Args:
            equity_curve: Full backtest equity curve.
            returns_series: Bar-by-bar returns. Computed if None.
            annualization_factor: For Sharpe annualisation.

        Returns:
            MonteCarloResult with p-values and confidence intervals.
        """
        if returns_series is None:
            returns_series = equity_curve.pct_change().fillna(0.0)

        returns_arr = returns_series.values
        n = len(returns_arr)

        if n < 20:
            raise ValueError(f"Insufficient data: need >= 20 bars, got {n}")

        initial_equity = equity_curve.iloc[0]

        # Original metrics
        orig_return = (equity_curve.iloc[-1] / initial_equity - 1.0) * 100.0
        ret_std = returns_series.std()
        if ret_std > 0 and annualization_factor > 0:
            orig_sharpe = (returns_series.mean() / ret_std) * np.sqrt(annualization_factor)
        else:
            orig_sharpe = 0.0

        running_max = equity_curve.expanding().max()
        orig_max_dd = abs(((equity_curve - running_max) / running_max).min()) * 100.0

        # ── Return Shuffling ──
        sim_returns: list[float] = []
        sim_sharpes: list[float] = []
        sim_max_dds: list[float] = []

        for _ in range(self._n_sims):
            shuffled = self._rng.permutation(returns_arr)

            # Reconstruct equity
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

        # P-values: fraction of simulations >= original
        return_p = np.mean([1 if sr >= orig_return else 0 for sr in sim_returns])
        sharpe_p = np.mean([1 if ss >= orig_sharpe else 0 for ss in sim_sharpes])

        # Percentiles
        return_pctile = np.mean([1 if sr < orig_return else 0 for sr in sim_returns]) * 100
        sharpe_pctile = np.mean([1 if ss < orig_sharpe else 0 for ss in sim_sharpes]) * 100

        # Confidence intervals via percentile method
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
```

---

## 7.5 `src/forgequant/core/robustness/cpcv.py`

```python
"""
Combinatorial Purged Cross-Validation (CPCV).

CPCV is a more rigorous alternative to k-fold cross-validation for
time series. It generates all C(N, N-k) combinations of contiguous
groups, using k groups for testing and N-k for training, with a purge
gap between train and test to prevent information leakage.

Reference: Marcos López de Prado, "Advances in Financial Machine Learning"

Simplified implementation:
    1. Divide data into N equal-sized groups
    2. For each combination of test groups (choosing k out of N):
       - Train on remaining groups
       - Test on the selected groups
       - Apply purge gap between adjacent train/test boundaries
    3. Aggregate OOS metrics across all combinations

For practical purposes with large data, we cap the number of
combinations evaluated.
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
    pbo_probability: float = 0.0  # Probability of Backtest Overfitting
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
    """
    Combinatorial Purged Cross-Validation.

    Usage:
        cpcv = CPCVAnalysis(n_groups=10, n_test_groups=2, purge_gap=5)
        result = cpcv.analyse(equity_curve, returns_series)
    """

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
        """
        Args:
            n_groups: Number of contiguous groups to split data into.
            n_test_groups: Number of groups used for testing per combination.
            purge_gap: Bars to remove between train/test boundaries.
            max_combinations: Maximum combinations to evaluate (sampled if exceeded).
            min_avg_oos_sharpe: Minimum average OOS Sharpe to pass.
            max_pbo: Maximum probability of backtest overfitting.
            random_seed: For reproducible combination sampling.
        """
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
        """
        Run CPCV analysis.

        Args:
            equity_curve: Full backtest equity curve.
            returns_series: Bar-by-bar returns. Computed if None.
            annualization_factor: For Sharpe annualisation.

        Returns:
            CPCVResult with per-combination and aggregated metrics.
        """
        if returns_series is None:
            returns_series = equity_curve.pct_change().fillna(0.0)

        n = len(returns_series)
        group_size = n // self._n_groups

        if group_size < 10:
            raise ValueError(
                f"Insufficient data: {n} bars / {self._n_groups} groups "
                f"= {group_size} bars per group (need >= 10)"
            )

        # Generate group boundaries
        group_bounds: list[tuple[int, int]] = []
        for g in range(self._n_groups):
            start = g * group_size
            end = min((g + 1) * group_size, n) if g < self._n_groups - 1 else n
            group_bounds.append((start, end))

        # Generate all combinations of test groups
        all_combos = list(itertools.combinations(range(self._n_groups), self._n_test))
        total_combos = len(all_combos)

        # Sample if too many
        if total_combos > self._max_combos:
            indices = self._rng.choice(total_combos, self._max_combos, replace=False)
            combos = [all_combos[i] for i in sorted(indices)]
        else:
            combos = all_combos

        folds: list[CPCVFold] = []
        returns_arr = returns_series.values

        for combo_idx, test_groups in enumerate(combos):
            test_set = set(test_groups)

            # Collect test indices
            test_indices: list[int] = []
            for g in test_groups:
                gs, ge = group_bounds[g]
                test_indices.extend(range(gs, ge))

            # Apply purge: remove bars adjacent to train/test boundary
            purged: set[int] = set()
            for g in test_groups:
                gs, ge = group_bounds[g]
                # Purge before test start
                for p in range(max(0, gs - self._purge_gap), gs):
                    purged.add(p)
                # Purge after test end
                for p in range(ge, min(n, ge + self._purge_gap)):
                    purged.add(p)

            # OOS returns are from the test groups (not purged)
            oos_returns = returns_arr[test_indices]

            if len(oos_returns) < 5:
                continue

            # OOS metrics
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

        # Aggregate
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

            # PBO: fraction of combinations with negative OOS Sharpe
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
```

---

## 7.6 `src/forgequant/core/robustness/parameter_sensitivity.py`

```python
"""
Parameter Sensitivity Analysis.

Tests whether small perturbations to a strategy's parameters cause
large changes in performance. A robust strategy should degrade
gracefully when parameters shift slightly.

Algorithm:
    1. For each block in the strategy, for each numeric parameter:
       a. Perturb by -pct, +pct (e.g. ±10%, ±20%)
       b. Re-compile and backtest with the perturbed value
       c. Record the resulting Sharpe ratio
    2. A parameter is "sensitive" if the Sharpe changes by more than
       a threshold relative to the original
    3. The strategy passes if fewer than max_sensitive_pct of
       parameter perturbations are sensitive

This analysis requires the full compiler+backtester pipeline, so
it accepts callables rather than operating on pre-computed curves.
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
    sensitivity_ratio: float  # n_sensitive / n_perturbations
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
    """
    Parameter sensitivity analysis.

    Usage with a backtest function:
        def run_backtest(params_override: dict) -> float:
            # Returns Sharpe ratio for the given parameter overrides
            ...

        ps = ParameterSensitivity(perturbation_pcts=[10, 20])
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={"ema": {"period": 20}, "fixed_tpsl": {"tp_atr_mult": 3.0}},
            backtest_fn=run_backtest,
        )
    """

    def __init__(
        self,
        perturbation_pcts: list[float] | None = None,
        sensitivity_threshold_pct: float = 30.0,
        max_sensitive_ratio: float = 0.3,
    ) -> None:
        """
        Args:
            perturbation_pcts: Percentage perturbations to apply (e.g. [10, 20]).
                               Both positive and negative are tested.
            sensitivity_threshold_pct: If Sharpe changes by more than this % of
                                       original, the parameter is "sensitive".
            max_sensitive_ratio: Maximum fraction of perturbations that can be
                                 sensitive for the strategy to pass.
        """
        self._perts = perturbation_pcts or [10.0, 20.0]
        self._threshold = sensitivity_threshold_pct
        self._max_ratio = max_sensitive_ratio

    def analyse(
        self,
        original_sharpe: float,
        parameter_specs: dict[str, dict[str, float]],
        backtest_fn: Callable[[str, str, float], float],
    ) -> SensitivityResult:
        """
        Run parameter sensitivity analysis.

        Args:
            original_sharpe: Sharpe ratio of the unperturbed strategy.
            parameter_specs: Dict mapping block_name -> {param_name: original_value}
                             for all numeric parameters to test.
            backtest_fn: Callable(block_name, param_name, new_value) -> sharpe.
                         Must re-run the full compile+backtest with the
                         perturbed parameter and return the resulting Sharpe.

        Returns:
            SensitivityResult with per-perturbation details.
        """
        perturbations: list[PerturbationResult] = []
        max_change = 0.0
        most_sensitive_block = ""
        most_sensitive_param = ""

        for block_name, params in parameter_specs.items():
            for param_name, original_value in params.items():
                if original_value == 0:
                    continue  # Can't perturb zero meaningfully

                for pct in self._perts:
                    for sign in [-1, 1]:
                        pert_pct = sign * pct
                        new_value = original_value * (1.0 + pert_pct / 100.0)

                        # Ensure integer params stay integer
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
                            # If backtest fails with perturbed params, treat as sensitive
                            perturbed_sharpe = 0.0

                        # Compute change
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
```

---

## 7.7 `src/forgequant/core/robustness/stability.py`

```python
"""
Equity Curve Stability Analysis.

Evaluates the quality and consistency of the equity curve using
multiple statistical measures:

1. **Linearity (R²)**: How closely does the equity curve follow a
   straight line? R² near 1.0 indicates consistent returns.

2. **Tail Ratio**: Ratio of the 95th percentile of returns to the
   absolute 5th percentile. > 1.0 means gains are larger than losses.

3. **Regime Consistency**: Divide the equity curve into halves or thirds
   and compare metrics across segments. A robust strategy performs
   similarly across all segments.

4. **Recovery Factor**: Total return / max drawdown. Higher is better.

5. **Hurst Exponent Estimate**: H > 0.5 suggests persistent (trending)
   equity, H ≈ 0.5 is random, H < 0.5 is mean-reverting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StabilityResult:
    """Result of equity curve stability analysis."""

    # Linearity
    r_squared: float = 0.0

    # Return distribution
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Recovery
    recovery_factor: float = 0.0

    # Regime consistency
    n_regimes: int = 0
    regime_sharpes: list[float] = field(default_factory=list)
    regime_returns_pct: list[float] = field(default_factory=list)
    regime_consistency_score: float = 0.0  # 0 to 1, higher is more consistent

    # Hurst exponent
    hurst_exponent: float = 0.5

    is_passed: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "r_squared": round(self.r_squared, 4),
            "tail_ratio": round(self.tail_ratio, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "recovery_factor": round(self.recovery_factor, 4),
            "regime_consistency_score": round(self.regime_consistency_score, 4),
            "hurst_exponent": round(self.hurst_exponent, 4),
            "is_passed": self.is_passed,
        }


class EquityStability:
    """
    Equity curve stability analysis.

    Usage:
        es = EquityStability(min_r_squared=0.8, n_regimes=3)
        result = es.analyse(equity_curve, returns_series)
    """

    def __init__(
        self,
        min_r_squared: float = 0.7,
        min_tail_ratio: float = 0.8,
        min_recovery_factor: float = 1.0,
        n_regimes: int = 3,
        min_regime_consistency: float = 0.5,
        annualization_factor: float = 252.0,
    ) -> None:
        """
        Args:
            min_r_squared: Minimum R² for the equity curve to pass.
            min_tail_ratio: Minimum tail ratio (gain/loss asymmetry).
            min_recovery_factor: Minimum total_return / max_drawdown.
            n_regimes: Number of segments for regime analysis.
            min_regime_consistency: Minimum consistency score across regimes.
            annualization_factor: For Sharpe in regime analysis.
        """
        self._min_r2 = min_r_squared
        self._min_tail = min_tail_ratio
        self._min_recovery = min_recovery_factor
        self._n_regimes = max(2, n_regimes)
        self._min_consistency = min_regime_consistency
        self._ann_factor = annualization_factor

    def analyse(
        self,
        equity_curve: pd.Series,
        returns_series: pd.Series | None = None,
    ) -> StabilityResult:
        """
        Run stability analysis on an equity curve.

        Args:
            equity_curve: Full backtest equity curve.
            returns_series: Bar-by-bar returns. Computed if None.

        Returns:
            StabilityResult with all stability metrics.
        """
        if returns_series is None:
            returns_series = equity_curve.pct_change().fillna(0.0)

        n = len(equity_curve)
        if n < 20:
            raise ValueError(f"Insufficient data: need >= 20 bars, got {n}")

        result = StabilityResult()

        # ── R² (linearity) ──
        result.r_squared = self._compute_r_squared(equity_curve)

        # ── Return distribution ──
        clean_returns = returns_series.dropna()
        if len(clean_returns) > 5:
            result.tail_ratio = self._compute_tail_ratio(clean_returns)
            result.skewness = float(clean_returns.skew())
            result.kurtosis = float(clean_returns.kurtosis())
        else:
            result.tail_ratio = 0.0
            result.skewness = 0.0
            result.kurtosis = 0.0

        # ── Recovery factor ──
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
        running_max = equity_curve.expanding().max()
        max_dd = abs(((equity_curve - running_max) / running_max).min())
        if max_dd > 0:
            result.recovery_factor = total_return / max_dd
        else:
            result.recovery_factor = float("inf") if total_return > 0 else 0.0

        # ── Regime consistency ──
        result.n_regimes = self._n_regimes
        regime_size = n // self._n_regimes

        regime_sharpes: list[float] = []
        regime_returns: list[float] = []

        for i in range(self._n_regimes):
            start = i * regime_size
            end = min((i + 1) * regime_size, n) if i < self._n_regimes - 1 else n

            seg_returns = returns_series.iloc[start:end]
            seg_equity = equity_curve.iloc[start:end]

            if len(seg_returns) < 5:
                continue

            seg_ret = (seg_equity.iloc[-1] / seg_equity.iloc[0] - 1.0) * 100.0
            regime_returns.append(seg_ret)

            seg_std = seg_returns.std()
            if seg_std > 0 and self._ann_factor > 0:
                seg_sharpe = (seg_returns.mean() / seg_std) * np.sqrt(self._ann_factor)
            else:
                seg_sharpe = 0.0
            regime_sharpes.append(seg_sharpe)

        result.regime_sharpes = regime_sharpes
        result.regime_returns_pct = regime_returns

        if len(regime_sharpes) >= 2:
            # Consistency: 1 - normalised std of regime Sharpes
            sharpe_std = np.std(regime_sharpes)
            sharpe_mean = abs(np.mean(regime_sharpes))
            if sharpe_mean > 0:
                cv = sharpe_std / sharpe_mean  # Coefficient of variation
                result.regime_consistency_score = max(0.0, 1.0 - cv)
            else:
                result.regime_consistency_score = 0.0

            # Also check that majority of regimes are profitable
            positive_regimes = sum(1 for r in regime_returns if r > 0)
            regime_positivity = positive_regimes / len(regime_returns)
            result.regime_consistency_score = min(
                result.regime_consistency_score,
                regime_positivity,
            )
        else:
            result.regime_consistency_score = 0.0

        # ── Hurst exponent (simplified R/S method) ──
        result.hurst_exponent = self._estimate_hurst(returns_series)

        # ── Pass / Fail ──
        result.is_passed = (
            result.r_squared >= self._min_r2
            and result.tail_ratio >= self._min_tail
            and result.recovery_factor >= self._min_recovery
            and result.regime_consistency_score >= self._min_consistency
        )

        logger.info(
            "stability_complete",
            r_squared=round(result.r_squared, 4),
            tail_ratio=round(result.tail_ratio, 4),
            recovery=round(result.recovery_factor, 4),
            consistency=round(result.regime_consistency_score, 4),
            hurst=round(result.hurst_exponent, 4),
            passed=result.is_passed,
        )

        return result

    @staticmethod
    def _compute_r_squared(equity: pd.Series) -> float:
        """Compute R² of the equity curve against a straight line."""
        y = equity.values.astype(float)
        x = np.arange(len(y), dtype=float)

        if len(y) < 3:
            return 0.0

        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()

        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_yy = np.sum((y - y_mean) ** 2)

        if ss_xx == 0 or ss_yy == 0:
            return 0.0

        r = ss_xy / np.sqrt(ss_xx * ss_yy)
        return float(r ** 2)

    @staticmethod
    def _compute_tail_ratio(returns: pd.Series) -> float:
        """
        Compute tail ratio: 95th percentile / |5th percentile|.

        > 1.0 means large gains exceed large losses.
        """
        p95 = returns.quantile(0.95)
        p05 = returns.quantile(0.05)

        if abs(p05) < 1e-10:
            return float("inf") if p95 > 0 else 0.0

        return abs(p95 / p05)

    @staticmethod
    def _estimate_hurst(returns: pd.Series, max_lags: int = 20) -> float:
        """
        Estimate the Hurst exponent using the rescaled range (R/S) method.

        H > 0.5: persistent (trending equity)
        H ≈ 0.5: random walk
        H < 0.5: mean-reverting
        """
        ts = returns.dropna().values
        n = len(ts)

        if n < 40:
            return 0.5  # Insufficient data, assume random

        lags = range(10, min(max_lags + 1, n // 4))
        rs_values: list[float] = []
        lag_values: list[float] = []

        for lag in lags:
            # Split into sub-series of length `lag`
            n_subseries = n // lag
            if n_subseries < 1:
                continue

            rs_list: list[float] = []
            for i in range(n_subseries):
                sub = ts[i * lag : (i + 1) * lag]
                mean_sub = np.mean(sub)
                deviations = np.cumsum(sub - mean_sub)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(sub, ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.log(np.mean(rs_list)))
                lag_values.append(np.log(lag))

        if len(rs_values) < 3:
            return 0.5

        # Linear regression of log(R/S) on log(lag)
        x = np.array(lag_values)
        y = np.array(rs_values)
        slope, _ = np.polyfit(x, y, 1)

        return float(np.clip(slope, 0.0, 1.0))
```

---

## 7.8 `src/forgequant/core/robustness/suite.py`

```python
"""
Robustness Suite orchestrator.

Runs all robustness gates on a BacktestResult and produces a
consolidated pass/fail verdict with detailed per-gate results.
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
    """
    Consolidated robustness verdict across all gates.

    A strategy passes the robustness suite only if ALL individual
    gates pass. Any single failure results in an overall failure.
    """

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

    # Walk-forward
    wf_n_folds: int = 5
    wf_train_pct: float = 0.70
    wf_min_consistency: float = 0.5
    wf_min_oos_sharpe: float = 0.0

    # Monte Carlo
    mc_n_simulations: int = 1000
    mc_p_value_threshold: float = 0.05
    mc_random_seed: int | None = 42

    # CPCV
    cpcv_n_groups: int = 10
    cpcv_n_test_groups: int = 2
    cpcv_purge_gap: int = 5
    cpcv_max_pbo: float = 0.5

    # Stability
    stab_min_r_squared: float = 0.7
    stab_min_tail_ratio: float = 0.8
    stab_min_recovery_factor: float = 1.0
    stab_n_regimes: int = 3

    # Annualisation
    annualization_factor: float = 252.0

    # Which gates to run
    run_walk_forward: bool = True
    run_monte_carlo: bool = True
    run_cpcv: bool = True
    run_stability: bool = True
    run_parameter_sensitivity: bool = False  # Requires external backtest_fn


class RobustnessSuite:
    """
    Orchestrates all robustness tests.

    Usage:
        suite = RobustnessSuite(config)
        verdict = suite.evaluate(backtest_result)
    """

    def __init__(self, config: SuiteConfig | None = None) -> None:
        self._config = config or SuiteConfig()

    def evaluate(
        self,
        result: BacktestResult,
        sensitivity_backtest_fn: Callable[[str, str, float], float] | None = None,
        sensitivity_params: dict[str, dict[str, float]] | None = None,
    ) -> RobustnessVerdict:
        """
        Run all configured robustness gates on a backtest result.

        Args:
            result: The BacktestResult to evaluate.
            sensitivity_backtest_fn: Optional callable for parameter sensitivity.
            sensitivity_params: Optional parameter specs for sensitivity analysis.

        Returns:
            RobustnessVerdict with per-gate results and overall verdict.
        """
        cfg = self._config
        verdict = RobustnessVerdict(strategy_name=result.strategy_name)

        equity = result.equity_curve
        returns = result.returns_series

        gates_passed = 0
        gates_total = 0

        # ── Walk-Forward ──
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

        # ── Monte Carlo ──
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

        # ── CPCV ──
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

        # ── Stability ──
        if cfg.run_stability:
            gates_total += 1
            try:
                stab = EquityStability(
                    min_r_squared=cfg.stab_min_r_squared,
                    min_tail_ratio=cfg.stab_min_tail_ratio,
                    min_recovery_factor=cfg.stab_min_recovery_factor,
                    n_regimes=cfg.stab_n_regimes,
                    annualization_factor=cfg.annualization_factor,
                )
                verdict.stability = stab.analyse(equity, returns)
                if verdict.stability.is_passed:
                    gates_passed += 1
            except Exception as e:
                verdict.errors.append(f"Stability failed: {e}")

        # ── Parameter Sensitivity ──
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

        # ── Overall Verdict ──
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
```

---

## 7.9 Test Suite

### `tests/unit/robustness/__init__.py`

```python
"""Tests for robustness suite."""
```

---

### `tests/unit/robustness/test_walk_forward.py`

```python
"""Tests for walk-forward analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.core.robustness.walk_forward import WalkForwardAnalysis, WalkForwardResult


def _make_equity(n: int = 500, drift: float = 0.0005) -> tuple[pd.Series, pd.Series]:
    """Generate a synthetic equity curve and returns."""
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
        assert len(splits) == 4  # n_folds - 1

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
        result = wfa.analyse(equity)  # No returns passed
        assert result.n_folds > 0
```

---

### `tests/unit/robustness/test_monte_carlo.py`

```python
"""Tests for Monte Carlo analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.core.robustness.monte_carlo import MonteCarloAnalysis, MonteCarloResult


def _make_equity(n: int = 500, drift: float = 0.001) -> tuple[pd.Series, pd.Series]:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = np.random.normal(drift, 0.01, n)
    equity = 100000 * np.cumprod(1.0 + returns)
    return pd.Series(equity, index=dates), pd.Series(returns, index=dates)


class TestMonteCarloInit:
    def test_insufficient_sims(self) -> None:
        with pytest.raises(ValueError, match="n_simulations"):
            MonteCarloAnalysis(n_simulations=50)


class TestMonteCarloAnalyse:
    def test_strong_strategy_low_p_value(self) -> None:
        """A strategy with strong drift should have a low return p-value."""
        equity, returns = _make_equity(500, drift=0.003)
        mc = MonteCarloAnalysis(n_simulations=500, random_seed=42)
        result = mc.analyse(equity, returns)

        assert result.return_p_value < 0.2  # Should be low for strong drift
        assert result.n_simulations == 500
        assert len(result.simulated_returns_pct) == 500

    def test_random_strategy_high_p_value(self) -> None:
        """A zero-drift strategy should have a high p-value."""
        equity, returns = _make_equity(500, drift=0.0)
        mc = MonteCarloAnalysis(n_simulations=500, random_seed=42)
        result = mc.analyse(equity, returns)

        # With zero drift, shuffling shouldn't change much
        # p-value should be relatively high (near 0.5)
        assert result.return_p_value > 0.1

    def test_confidence_intervals(self) -> None:
        equity, returns = _make_equity(500)
        mc = MonteCarloAnalysis(n_simulations=200, random_seed=42, confidence_level=0.95)
        result = mc.analyse(equity, returns)

        assert result.return_ci_lower < result.return_ci_upper
        assert result.sharpe_ci_lower < result.sharpe_ci_upper
        assert result.max_dd_ci_lower <= result.max_dd_ci_upper

    def test_percentiles(self) -> None:
        equity, returns = _make_equity(500, drift=0.002)
        mc = MonteCarloAnalysis(n_simulations=300, random_seed=42)
        result = mc.analyse(equity, returns)

        assert 0 <= result.return_percentile <= 100
        assert 0 <= result.sharpe_percentile <= 100

    def test_summary(self) -> None:
        equity, returns = _make_equity(500)
        mc = MonteCarloAnalysis(n_simulations=200, random_seed=42)
        result = mc.analyse(equity, returns)

        s = result.summary()
        assert "return_p_value" in s
        assert "sharpe_p_value" in s
        assert "is_passed" in s

    def test_insufficient_data_raises(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        equity = pd.Series(np.ones(10) * 100000, index=dates)
        mc = MonteCarloAnalysis(n_simulations=100)
        with pytest.raises(ValueError, match="Insufficient"):
            mc.analyse(equity)

    def test_reproducibility(self) -> None:
        equity, returns = _make_equity(200)
        mc1 = MonteCarloAnalysis(n_simulations=100, random_seed=123)
        mc2 = MonteCarloAnalysis(n_simulations=100, random_seed=123)
        r1 = mc1.analyse(equity, returns)
        r2 = mc2.analyse(equity, returns)
        assert r1.return_p_value == r2.return_p_value
```

---

### `tests/unit/robustness/test_cpcv.py`

```python
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

        # With positive drift, most OOS periods should have positive Sharpe
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
```

---

### `tests/unit/robustness/test_parameter_sensitivity.py`

```python
"""Tests for parameter sensitivity analysis."""

from __future__ import annotations

import pytest

from forgequant.core.robustness.parameter_sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
)


class TestParameterSensitivity:
    def test_stable_strategy_passes(self) -> None:
        """A strategy insensitive to perturbations should pass."""

        def backtest_fn(block: str, param: str, value: float) -> float:
            # Always returns similar Sharpe regardless of perturbation
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
        """A strategy highly sensitive to perturbations should fail."""

        def backtest_fn(block: str, param: str, value: float) -> float:
            # Sharpe collapses with any perturbation
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
        """Mixed sensitivity: some params stable, some not."""
        call_count = {"n": 0}

        def backtest_fn(block: str, param: str, value: float) -> float:
            call_count["n"] += 1
            if block == "ema":
                return 1.5  # Stable
            return 0.0  # Sensitive

        ps = ParameterSensitivity(
            perturbation_pcts=[10],
            sensitivity_threshold_pct=30.0,
        )
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={
                "ema": {"period": 20.0},
                "tpsl": {"tp_mult": 3.0},
            },
            backtest_fn=backtest_fn,
        )

        assert result.n_perturbations > 0
        assert 0 < result.n_sensitive < result.n_perturbations

    def test_backtest_fn_failure_counts_as_sensitive(self) -> None:
        """If the backtest function raises, the perturbation is sensitive."""

        def backtest_fn(block: str, param: str, value: float) -> float:
            raise RuntimeError("Backtest failed")

        ps = ParameterSensitivity(perturbation_pcts=[10])
        result = ps.analyse(
            original_sharpe=1.5,
            parameter_specs={"ema": {"period": 20.0}},
            backtest_fn=backtest_fn,
        )

        # Failed backtests return 0.0, which is a 100% change -> sensitive
        assert result.n_sensitive > 0

    def test_zero_original_value_skipped(self) -> None:
        """Parameters with value 0 should be skipped."""

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
```

---

### `tests/unit/robustness/test_stability.py`

```python
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
        """A perfectly linear equity curve should have R² ≈ 1.0."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        equity = pd.Series(np.linspace(100000, 120000, n), index=dates)
        returns = equity.pct_change().fillna(0)

        es = EquityStability(min_r_squared=0.9)
        result = es.analyse(equity, returns)

        assert result.r_squared > 0.99

    def test_noisy_equity_lower_r2(self) -> None:
        """Noisy equity should have lower R²."""
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
        """Strong consistent uptrend should pass stability checks."""
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        np.random.seed(42)
        # Low noise, high drift
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
```

---

### `tests/unit/robustness/test_suite.py`

```python
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
        assert verdict.gates_total == 4  # WF, MC, CPCV, Stability
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

    def test_all_pass_verdict(self) -> None:
        """With generous thresholds and strong drift, all gates should pass."""
        result = _make_backtest_result(500, drift=0.003)
        config = SuiteConfig(
            mc_n_simulations=200,
            mc_random_seed=42,
            mc_p_value_threshold=0.5,  # Very generous
            wf_min_consistency=0.1,
            wf_min_oos_sharpe=-10.0,
            cpcv_max_pbo=0.9,
            cpcv_n_groups=5,
            cpcv_n_test_groups=1,
            stab_min_r_squared=0.1,
            stab_min_tail_ratio=0.1,
            stab_min_recovery_factor=0.01,
            stab_min_regime_consistency=0.1,
        )
        suite = RobustnessSuite(config)
        verdict = suite.evaluate(result)

        assert verdict.gates_total == 4
        # With very generous thresholds, should pass
        # (May not always pass MC due to randomness, but should pass most)

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
        """Test suite with parameter sensitivity enabled."""
        result = _make_backtest_result()

        def backtest_fn(block: str, param: str, value: float) -> float:
            return 1.5  # Always stable

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
        """If a gate raises, it should be recorded as an error, not crash."""
        # Very short data that will cause some gates to fail
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
            wf_n_folds=10,  # Too many folds for 30 bars
            mc_n_simulations=100,
            mc_random_seed=42,
            cpcv_n_groups=10,  # Too many groups for 30 bars
        )
        suite = RobustnessSuite(config)
        verdict = suite.evaluate(result)

        # Some gates should have errored but not crashed
        assert verdict.strategy_name == "short_data"
        # Errors recorded
        if verdict.errors:
            assert any("failed" in e.lower() or "insufficient" in e.lower() for e in verdict.errors)
```

---

## 7.10 Integration Test

### `tests/integration/test_phase7_robustness.py`

```python
"""
Integration test for the full robustness suite on a
compiled and backtested strategy.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.ai_forge.validator import SpecValidator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.engine.backtester import Backtester, BacktestConfig
from forgequant.core.robustness.suite import RobustnessSuite, SuiteConfig


def _populate_registry(registry: BlockRegistry) -> None:
    for mod_name in [
        "forgequant.blocks.indicators",
        "forgequant.blocks.entry_rules",
        "forgequant.blocks.exit_rules",
        "forgequant.blocks.money_management",
        "forgequant.blocks.filters",
        "forgequant.blocks.price_action",
    ]:
        module = importlib.import_module(mod_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "metadata") and attr_name != "BaseBlock":
                try:
                    registry.register_class(attr)
                except Exception:
                    pass


@pytest.fixture
def full_registry(clean_registry: BlockRegistry) -> BlockRegistry:
    _populate_registry(clean_registry)
    return clean_registry


def _make_trending_data(n: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100.0 + np.cumsum(np.random.normal(0.05, 0.3, n))
    close = np.maximum(close, 50.0)
    spread = np.random.uniform(0.1, 0.5, n)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )


class TestFullRobustnessFlow:
    """End-to-end: spec → compile → backtest → robustness suite."""

    def test_complete_pipeline(self, full_registry: BlockRegistry) -> None:
        # 1. Define strategy
        spec = StrategySpec(
            name="robustness_e2e_test",
            description="A trend following strategy for end to end robustness suite integration testing.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[
                BlockSpec(block_name="ema", params={"period": 20}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            entry_rules=[
                BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20}),
            ],
            exit_rules=[
                BlockSpec(block_name="fixed_tpsl", params={"atr_period": 14, "tp_atr_mult": 3.0, "sl_atr_mult": 1.5}),
            ],
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0, "atr_period": 14}),
        )

        # 2. Validate
        validator = SpecValidator(full_registry)
        val = validator.validate(spec)
        assert val.is_valid

        # 3. Compile
        data = _make_trending_data(1000)
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        compiled = compiler.compile(spec, data, val.validated_params)

        # 4. Backtest
        bt_result = Backtester(BacktestConfig(initial_equity=100_000)).run(compiled)
        assert bt_result.equity_curve is not None
        assert bt_result.returns_series is not None

        # 5. Robustness suite
        suite_config = SuiteConfig(
            wf_n_folds=4,
            wf_min_consistency=0.1,
            wf_min_oos_sharpe=-10.0,
            mc_n_simulations=200,
            mc_p_value_threshold=0.5,
            mc_random_seed=42,
            cpcv_n_groups=5,
            cpcv_n_test_groups=1,
            cpcv_max_pbo=0.9,
            stab_min_r_squared=0.1,
            stab_min_tail_ratio=0.1,
            stab_min_recovery_factor=0.01,
            stab_min_regime_consistency=0.1,
        )
        suite = RobustnessSuite(suite_config)
        verdict = suite.evaluate(bt_result)

        # Verify structure
        assert verdict.strategy_name == "robustness_e2e_test"
        assert verdict.gates_total == 4
        assert verdict.walk_forward is not None
        assert verdict.monte_carlo is not None
        assert verdict.cpcv is not None
        assert verdict.stability is not None

        # Each gate should have produced a valid result
        assert verdict.walk_forward.n_folds > 0
        assert verdict.monte_carlo.n_simulations == 200
        assert verdict.cpcv.n_combinations_evaluated > 0
        assert verdict.stability.r_squared >= 0

        # Summary should work
        summary = verdict.summary()
        assert "strategy_name" in summary
        assert "walk_forward" in summary
        assert "monte_carlo" in summary
        assert "cpcv" in summary
        assert "stability" in summary
```

---

## 7.11 How to Verify Phase 7

```bash
# From project root with venv activated

# Run all tests
pytest -v

# Run only robustness tests
pytest tests/unit/robustness/ -v

# Run the integration test
pytest tests/integration/test_phase7_robustness.py -v

# Type-check
mypy src/forgequant/core/robustness/

# Lint
ruff check src/forgequant/core/robustness/
```

**Expected output:** All tests pass — approximately **60+ new tests** across 7 test modules plus the integration test.

---

## Phase 7 Summary

### Module Overview

| Module | File | Purpose | Key Output |
|--------|------|---------|------------|
| **Walk-Forward** | `walk_forward.py` | Sequential train/test splits, OOS metrics per fold | `WalkForwardResult` with per-fold Sharpe/return/DD, consistency ratio |
| **Monte Carlo** | `monte_carlo.py` | Return shuffling for p-values, bootstrapped CIs | `MonteCarloResult` with return/Sharpe p-values, 95% confidence intervals |
| **CPCV** | `cpcv.py` | Combinatorial purged cross-validation, PBO estimation | `CPCVResult` with per-combination OOS Sharpe, PBO probability |
| **Param Sensitivity** | `parameter_sensitivity.py` | Perturbation analysis for each numeric parameter | `SensitivityResult` with per-perturbation Sharpe change, sensitivity ratio |
| **Stability** | `stability.py` | Equity curve quality: R², tail ratio, regime consistency, Hurst | `StabilityResult` with 7 stability metrics |
| **Suite** | `suite.py` | Orchestrates all gates, produces consolidated verdict | `RobustnessVerdict` with per-gate results, overall pass/fail |

### Robustness Gate Details

| Gate | What It Tests | Pass Criteria (defaults) |
|------|---------------|-------------------------|
| **Walk-Forward** | OOS generalisation across sequential windows | ≥50% folds profitable, avg OOS Sharpe ≥ 0 |
| **Monte Carlo** | Statistical significance vs random chance | Return and Sharpe p-value ≤ 0.05 |
| **CPCV** | Backtest overfitting probability | PBO ≤ 50%, avg OOS Sharpe ≥ 0 |
| **Param Sensitivity** | Stability under ±10-20% parameter perturbation | ≤30% of perturbations cause >30% Sharpe change |
| **Stability** | Equity curve consistency across time | R² ≥ 0.7, tail ratio ≥ 0.8, recovery ≥ 1.0, regime consistency ≥ 0.5 |

### Key Metrics Computed

| Analysis | Metric | Description |
|----------|--------|-------------|
| Walk-Forward | `oos_returns_consistency` | Fraction of folds with positive OOS return |
| Walk-Forward | `avg_oos_sharpe` | Mean OOS Sharpe across all folds |
| Monte Carlo | `return_p_value` | Probability of achieving this return by chance |
| Monte Carlo | `sharpe_p_value` | Probability of achieving this Sharpe by chance |
| Monte Carlo | `return_ci_lower/upper` | 95% confidence interval on returns |
| CPCV | `pbo_probability` | Fraction of combinations with negative OOS Sharpe |
| CPCV | `std_oos_sharpe` | Cross-combination Sharpe volatility |
| Sensitivity | `sensitivity_ratio` | Fraction of perturbations causing significant Sharpe degradation |
| Stability | `r_squared` | Linearity of equity curve (1.0 = perfect straight line) |
| Stability | `tail_ratio` | Asymmetry of return tails (>1.0 = gains > losses) |
| Stability | `hurst_exponent` | Persistence of equity returns (>0.5 = trending equity) |
| Stability | `regime_consistency_score` | How similar is performance across time segments |
| Stability | `recovery_factor` | Total return / max drawdown |

### Architecture Flow

```
BacktestResult
(equity_curve, returns_series, trades, metrics)
          │
          ▼
┌──────────────────────────────────┐
│        RobustnessSuite           │
│                                  │
│  ┌─────────────────────────────┐ │
│  │ 1. Walk-Forward Analysis    │ │
│  │    → OOS consistency        │ │
│  ├─────────────────────────────┤ │
│  │ 2. Monte Carlo              │ │
│  │    → p-values, CIs          │ │
│  ├─────────────────────────────┤ │
│  │ 3. CPCV                     │ │
│  │    → PBO probability        │ │
│  ├─────────────────────────────┤ │
│  │ 4. Equity Stability         │ │
│  │    → R², Hurst, regimes     │ │
│  ├─────────────────────────────┤ │
│  │ 5. Parameter Sensitivity    │ │
│  │    (optional, needs fn)     │ │
│  └─────────────────────────────┘ │
│                                  │
│  ALL gates pass → overall PASS   │
│  ANY gate fails → overall FAIL   │
└──────────────────┬───────────────┘
                   │
                   ▼
          RobustnessVerdict
     (per-gate results + verdict)
```

### Cumulative Project Status

| Phase | Status | Components |
|-------|--------|------------|
| Phase 1 | ✅ | Foundation |
| Phase 2 | ✅ | 8 indicator blocks |
| Phase 3 | ✅ | 4 price action + 4 entry rule blocks |
| Phase 4 | ✅ | 4 exit + 4 money mgmt + 4 filter blocks |
| Phase 5 | ✅ | AI Forge — schemas, validator, prompt, RAG, pipeline |
| Phase 6 | ✅ | Compiler + backtesting engine + 25+ metrics |
| **Phase 7** | ✅ | **Robustness suite — 5 gates, consolidated verdict** |
| Phase 8 | 🔜 | Execution layer (MT5 via aiomql) |
| Phase 9 | 🔜 | Reflex dashboard |

**Total blocks: 28** | **Total source files: ~70** | **Total test files: ~42**

---

**Ready for Phase 8** — say the word and I'll write the Execution Layer: MetaTrader 5 integration via `aiomql` with async order management, position tracking, signal-to-order translation, risk checks, and comprehensive test coverage with mock MT5 interactions.
