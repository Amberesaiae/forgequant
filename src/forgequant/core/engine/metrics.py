"""
Performance metrics calculator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.core.engine.results import BacktestResult, TradeRecord
from forgequant.core.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> BacktestResult:
    """Compute all performance metrics and populate the BacktestResult."""
    equity = result.equity_curve
    trades = result.trades

    if len(equity) < 2:
        result.metrics = {"error": "Insufficient data for metrics"}
        return result

    returns = equity.pct_change().fillna(0.0)
    result.returns_series = returns

    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    result.drawdown_series = drawdown

    metrics: dict[str, float] = {}

    total_return = (result.final_equity / result.initial_equity) - 1.0
    metrics["total_return_pct"] = total_return * 100.0

    n_bars = len(equity)
    if n_bars > 1 and annualization_factor > 0:
        ann_return = (1.0 + total_return) ** (annualization_factor / n_bars) - 1.0
        metrics["annualized_return_pct"] = ann_return * 100.0
    else:
        metrics["annualized_return_pct"] = 0.0

    metrics["max_drawdown_pct"] = abs(drawdown.min()) * 100.0
    metrics["max_drawdown_duration"] = _max_drawdown_duration(equity)

    if returns.std() > 0:
        ann_vol = returns.std() * np.sqrt(annualization_factor)
        metrics["annualized_volatility_pct"] = ann_vol * 100.0
    else:
        ann_vol = 0.0
        metrics["annualized_volatility_pct"] = 0.0

    if ann_vol > 0 and annualization_factor > 0:
        excess_return = metrics["annualized_return_pct"] / 100.0 - risk_free_rate
        metrics["sharpe_ratio"] = excess_return / ann_vol
    else:
        metrics["sharpe_ratio"] = 0.0

    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(annualization_factor)
        if downside_std > 0:
            excess_return = metrics["annualized_return_pct"] / 100.0 - risk_free_rate
            metrics["sortino_ratio"] = excess_return / downside_std
        else:
            metrics["sortino_ratio"] = 0.0
    else:
        metrics["sortino_ratio"] = float("inf") if total_return > 0 else 0.0

    max_dd = metrics["max_drawdown_pct"] / 100.0
    if max_dd > 0:
        metrics["calmar_ratio"] = (metrics["annualized_return_pct"] / 100.0) / max_dd
    else:
        metrics["calmar_ratio"] = float("inf") if total_return > 0 else 0.0

    n_trades = len(trades)
    metrics["total_trades"] = float(n_trades)

    if n_trades > 0:
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        metrics["win_rate"] = len(winners) / n_trades
        metrics["loss_rate"] = len(losers) / n_trades

        metrics["avg_pnl_dollar"] = np.mean([t.pnl_dollar for t in trades])
        metrics["avg_pnl_pct"] = np.mean([t.pnl_pct for t in trades])

        if winners:
            metrics["avg_winner_dollar"] = np.mean([t.pnl_dollar for t in winners])
            metrics["avg_winner_pct"] = np.mean([t.pnl_pct for t in winners])
        else:
            metrics["avg_winner_dollar"] = 0.0
            metrics["avg_winner_pct"] = 0.0

        if losers:
            metrics["avg_loser_dollar"] = np.mean([t.pnl_dollar for t in losers])
            metrics["avg_loser_pct"] = np.mean([t.pnl_pct for t in losers])
        else:
            metrics["avg_loser_dollar"] = 0.0
            metrics["avg_loser_pct"] = 0.0

        gross_profit = sum(t.pnl_dollar for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl_dollar for t in losers)) if losers else 0.0
        if gross_loss > 0:
            metrics["profit_factor"] = gross_profit / gross_loss
        else:
            metrics["profit_factor"] = float("inf") if gross_profit > 0 else 0.0

        if metrics["avg_loser_dollar"] != 0:
            metrics["payoff_ratio"] = abs(
                metrics["avg_winner_dollar"] / metrics["avg_loser_dollar"]
            )
        else:
            metrics["payoff_ratio"] = float("inf") if metrics["avg_winner_dollar"] > 0 else 0.0

        metrics["expectancy_dollar"] = (
            metrics["win_rate"] * metrics["avg_winner_dollar"]
            + metrics["loss_rate"] * metrics["avg_loser_dollar"]
        )

        metrics["avg_bars_held"] = np.mean([t.bars_held for t in trades])
        metrics["max_bars_held"] = float(max(t.bars_held for t in trades))

        metrics["max_consecutive_wins"] = float(_max_consecutive(trades, winners=True))
        metrics["max_consecutive_losses"] = float(_max_consecutive(trades, winners=False))

        long_trades = [t for t in trades if t.direction == "long"]
        short_trades = [t for t in trades if t.direction == "short"]
        metrics["long_trades"] = float(len(long_trades))
        metrics["short_trades"] = float(len(short_trades))

        if long_trades:
            metrics["long_win_rate"] = len([t for t in long_trades if t.is_winner]) / len(long_trades)
        else:
            metrics["long_win_rate"] = 0.0

        if short_trades:
            metrics["short_win_rate"] = (
                len([t for t in short_trades if t.is_winner]) / len(short_trades)
            )
        else:
            metrics["short_win_rate"] = 0.0

    else:
        for key in [
            "win_rate", "loss_rate", "avg_pnl_dollar", "avg_pnl_pct",
            "avg_winner_dollar", "avg_winner_pct",
            "avg_loser_dollar", "avg_loser_pct",
            "profit_factor", "payoff_ratio", "expectancy_dollar",
            "avg_bars_held", "max_bars_held",
            "max_consecutive_wins", "max_consecutive_losses",
            "long_trades", "short_trades",
            "long_win_rate", "short_win_rate",
        ]:
            metrics[key] = 0.0

    result.metrics = metrics

    logger.info(
        "metrics_computed",
        strategy=result.strategy_name,
        n_trades=n_trades,
        total_return_pct=round(metrics["total_return_pct"], 2),
        sharpe=round(metrics["sharpe_ratio"], 3),
        max_dd_pct=round(metrics["max_drawdown_pct"], 2),
    )

    return result


def _max_drawdown_duration(equity: pd.Series) -> float:
    running_max = equity.expanding().max()
    in_drawdown = equity < running_max

    if not in_drawdown.any():
        return 0.0

    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown.values:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return float(max_duration)


def _max_consecutive(trades: list[TradeRecord], winners: bool) -> int:
    max_streak = 0
    current_streak = 0

    for trade in trades:
        if trade.is_winner == winners:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak
