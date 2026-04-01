"""
Equity Curve Stability Analysis.

Evaluates quality and consistency using R², tail ratio, regime
consistency, recovery factor, and Hurst exponent.
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

    r_squared: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    recovery_factor: float = 0.0
    n_regimes: int = 0
    regime_sharpes: list[float] = field(default_factory=list)
    regime_returns_pct: list[float] = field(default_factory=list)
    regime_consistency_score: float = 0.0
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
    """Equity curve stability analysis."""

    def __init__(
        self,
        min_r_squared: float = 0.7,
        min_tail_ratio: float = 0.8,
        min_recovery_factor: float = 1.0,
        n_regimes: int = 3,
        min_regime_consistency: float = 0.5,
        annualization_factor: float = 252.0,
    ) -> None:
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
        if returns_series is None:
            returns_series = equity_curve.pct_change().fillna(0.0)

        n = len(equity_curve)
        if n < 20:
            raise ValueError(f"Insufficient data: need >= 20 bars, got {n}")

        result = StabilityResult()

        result.r_squared = self._compute_r_squared(equity_curve)

        clean_returns = returns_series.dropna()
        if len(clean_returns) > 5:
            result.tail_ratio = self._compute_tail_ratio(clean_returns)
            result.skewness = float(clean_returns.skew())
            result.kurtosis = float(clean_returns.kurtosis())
        else:
            result.tail_ratio = 0.0
            result.skewness = 0.0
            result.kurtosis = 0.0

        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
        running_max = equity_curve.expanding().max()
        max_dd = abs(((equity_curve - running_max) / running_max).min())
        if max_dd > 0:
            result.recovery_factor = total_return / max_dd
        else:
            result.recovery_factor = float("inf") if total_return > 0 else 0.0

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
            sharpe_std = np.std(regime_sharpes)
            sharpe_mean = abs(np.mean(regime_sharpes))
            if sharpe_mean > 0:
                cv = sharpe_std / sharpe_mean
                result.regime_consistency_score = max(0.0, 1.0 - cv)
            else:
                result.regime_consistency_score = 0.0

            positive_regimes = sum(1 for r in regime_returns if r > 0)
            regime_positivity = positive_regimes / len(regime_returns)
            result.regime_consistency_score = min(
                result.regime_consistency_score,
                regime_positivity,
            )
        else:
            result.regime_consistency_score = 0.0

        result.hurst_exponent = self._estimate_hurst(returns_series)

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
        y = equity.values.astype(float)
        x = np.arange(len(y), dtype=float)

        if len(y) < 3:
            return 0.0

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
        p95 = returns.quantile(0.95)
        p05 = returns.quantile(0.05)

        if abs(p05) < 1e-10:
            return float("inf") if p95 > 0 else 0.0

        return abs(p95 / p05)

    @staticmethod
    def _estimate_hurst(returns: pd.Series, max_lags: int = 20) -> float:
        ts = returns.dropna().values
        n = len(ts)

        if n < 40:
            return 0.5

        lags = range(10, min(max_lags + 1, n // 4))
        rs_values: list[float] = []
        lag_values: list[float] = []

        for lag in lags:
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

        x = np.array(lag_values)
        y = np.array(rs_values)
        slope, _ = np.polyfit(x, y, 1)

        return float(np.clip(slope, 0.0, 1.0))
