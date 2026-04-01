"""
Backtest result containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """Record of a single completed trade."""

    trade_id: int
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    position_size: float
    pnl: float
    pnl_pct: float
    pnl_dollar: float
    bars_held: int
    mae: float = 0.0
    mfe: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0.0

    @property
    def risk_reward_achieved(self) -> float:
        if self.mae == 0.0:
            return float("inf") if self.mfe > 0 else 0.0
        return abs(self.mfe / self.mae)


@dataclass
class BacktestResult:
    """Complete backtest result."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_equity: float
    final_equity: float
    equity_curve: pd.Series
    trades: list[TradeRecord] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    drawdown_series: pd.Series | None = None
    returns_series: pd.Series | None = None

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_dollar for t in self.trades)

    @property
    def winning_trades(self) -> list[TradeRecord]:
        return [t for t in self.trades if t.is_winner]

    @property
    def losing_trades(self) -> list[TradeRecord]:
        return [t for t in self.trades if not t.is_winner]

    def trades_to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "trade_id": t.trade_id,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "position_size": t.position_size,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "pnl_dollar": t.pnl_dollar,
                "bars_held": t.bars_held,
                "mae": t.mae,
                "mfe": t.mfe,
                "is_winner": t.is_winner,
            })

        return pd.DataFrame(records)

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "period": f"{self.start_date} to {self.end_date}",
            "initial_equity": self.initial_equity,
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(
                (self.final_equity / self.initial_equity - 1) * 100, 2
            ),
            "n_trades": self.n_trades,
            "total_pnl": round(self.total_pnl, 2),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in self.metrics.items()},
        }
