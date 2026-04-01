"""
Walk-Forward Analysis.

Splits data into sequential train/test windows and aggregates
out-of-sample results to validate generalisation.
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
    oos_returns_consistency: float = 0.0
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
    """Walk-forward analysis engine."""

    def __init__(
        self,
        n_folds: int = 5,
        train_pct: float = 0.70,
        min_oos_trades: int = 10,
        min_oos_sharpe: float = 0.0,
        min_consistency: float = 0.5,
        anchored: bool = False,
    ) -> None:
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

            oos_return = (oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) * 100.0

            ret_std = oos_returns.std()
            if ret_std > 0 and annualization_factor > 0:
                oos_sharpe = (oos_returns.mean() / ret_std) * np.sqrt(annualization_factor)
            else:
                oos_sharpe = 0.0

            oos_running_max = oos_equity.expanding().max()
            oos_dd = ((oos_equity - oos_running_max) / oos_running_max).min()
            oos_max_dd_pct = abs(oos_dd) * 100.0

            if trade_bars is not None:
                oos_trades = int(trade_bars.iloc[te_s:te_e].sum())
            else:
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

        result = WalkForwardResult(n_folds=len(folds), folds=folds)

        if folds:
            result.avg_oos_return_pct = float(np.mean([f.oos_return_pct for f in folds]))
            result.avg_oos_sharpe = float(np.mean([f.oos_sharpe for f in folds]))
            result.avg_oos_max_drawdown_pct = float(np.mean([f.oos_max_drawdown_pct for f in folds]))
            result.total_oos_trades = sum(f.oos_n_trades for f in folds)

            positive_folds = sum(1 for f in folds if f.oos_return_pct > 0)
            result.oos_returns_consistency = positive_folds / len(folds)

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
