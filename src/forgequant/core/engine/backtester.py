"""
Vectorized backtesting engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from forgequant.core.compiler.compiled_strategy import CompiledStrategy
from forgequant.core.engine.metrics import compute_metrics
from forgequant.core.engine.results import BacktestResult, TradeRecord
from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for the backtester."""

    initial_equity: float = 100_000.0
    commission_per_unit: float = 0.0
    slippage_pct: float = 0.0
    allow_pyramiding: bool = False
    max_positions: int = 1
    default_position_size: float = 1.0
    annualization_factor: float = 252.0
    risk_free_rate: float = 0.0


class Backtester:
    """Vectorized backtesting engine."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()

    def run(self, compiled: CompiledStrategy) -> BacktestResult:
        """Run the backtest on a compiled strategy."""
        cfg = self._config
        data = compiled.ohlcv
        n = len(data)

        logger.info(
            "backtest_start",
            strategy=compiled.spec.name,
            n_bars=n,
            initial_equity=cfg.initial_equity,
        )

        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        index = data.index

        entry_long = compiled.filtered_entry_long().values
        entry_short = compiled.filtered_entry_short().values
        exit_long = compiled.exit_long.values if compiled.exit_long is not None else np.zeros(n, dtype=bool)
        exit_short = compiled.exit_short.values if compiled.exit_short is not None else np.zeros(n, dtype=bool)

        sl_long = compiled.stop_loss_long.values if compiled.stop_loss_long is not None else None
        sl_short = compiled.stop_loss_short.values if compiled.stop_loss_short is not None else None
        tp_long = compiled.take_profit_long.values if compiled.take_profit_long is not None else None
        tp_short = compiled.take_profit_short.values if compiled.take_profit_short is not None else None

        pos_size_arr = (
            compiled.position_size_long.values
            if compiled.position_size_long is not None
            else np.full(n, cfg.default_position_size)
        )

        equity = cfg.initial_equity
        equity_curve = np.zeros(n)
        trades: list[TradeRecord] = []
        trade_id = 0

        in_long = False
        in_short = False
        entry_price = 0.0
        entry_bar = 0
        entry_sl = 0.0
        entry_tp = 0.0
        position_size = 0.0
        trade_mae = 0.0
        trade_mfe = 0.0

        for i in range(n):
            bar_pnl = 0.0

            bar_pnl, trade_rec, closed, trade_mae, trade_mfe = self._process_long_bar(
                i, in_long, entry_price, entry_bar, entry_sl, entry_tp,
                position_size, trade_mae, trade_mfe, trade_id,
                low_prices, high_prices, close_prices, exit_long, index, cfg,
            )
            if trade_rec is not None:
                trades.append(trade_rec)
                trade_id += 1
            if in_long:
                in_long = not closed

            bar_pnl_s, trade_rec_s, closed_s, trade_mae, trade_mfe = self._process_short_bar(
                i, in_short, entry_price, entry_bar, entry_sl, entry_tp,
                position_size, trade_mae, trade_mfe, trade_id,
                low_prices, high_prices, close_prices, exit_short, index, cfg,
            )
            if trade_rec_s is not None:
                trades.append(trade_rec_s)
                trade_id += 1
                bar_pnl = bar_pnl_s
            if in_short:
                in_short = not closed_s

            if not in_long and not in_short:
                if i < n - 1 and entry_long[i]:
                    in_long = True
                    entry_price = open_prices[i + 1]
                    entry_bar = i + 1
                    entry_price *= (1.0 + cfg.slippage_pct / 100.0)
                    ps = pos_size_arr[i]
                    position_size = ps if not np.isnan(ps) and ps > 0 else cfg.default_position_size
                    entry_sl = sl_long[i] if sl_long is not None and not np.isnan(sl_long[i]) else 0.0
                    entry_tp = tp_long[i] if tp_long is not None and not np.isnan(tp_long[i]) else 0.0
                    trade_mae = 0.0
                    trade_mfe = 0.0
                    equity -= position_size * cfg.commission_per_unit

                elif i < n - 1 and entry_short[i]:
                    in_short = True
                    entry_price = open_prices[i + 1]
                    entry_bar = i + 1
                    entry_price *= (1.0 - cfg.slippage_pct / 100.0)
                    ps = pos_size_arr[i]
                    position_size = ps if not np.isnan(ps) and ps > 0 else cfg.default_position_size
                    entry_sl = sl_short[i] if sl_short is not None and not np.isnan(sl_short[i]) else 0.0
                    entry_tp = tp_short[i] if tp_short is not None and not np.isnan(tp_short[i]) else 0.0
                    trade_mae = 0.0
                    trade_mfe = 0.0
                    equity -= position_size * cfg.commission_per_unit

            equity += bar_pnl
            equity_curve[i] = equity

        if in_long:
            exit_price = close_prices[-1]
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "long", entry_bar, n - 1, index,
                entry_price, exit_price, "end",
                position_size, trade_mae, trade_mfe, cfg,
            )
            trades.append(trade_rec)
            equity += bar_pnl
            equity_curve[-1] = equity

        if in_short:
            exit_price = close_prices[-1]
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "short", entry_bar, n - 1, index,
                entry_price, exit_price, "end",
                position_size, trade_mae, trade_mfe, cfg,
            )
            trades.append(trade_rec)
            equity += bar_pnl
            equity_curve[-1] = equity

        equity_series = pd.Series(equity_curve, index=index, name="equity")

        result = BacktestResult(
            strategy_name=compiled.spec.name,
            start_date=index[0].to_pydatetime(),
            end_date=index[-1].to_pydatetime(),
            initial_equity=cfg.initial_equity,
            final_equity=equity,
            equity_curve=equity_series,
            trades=trades,
        )

        result = compute_metrics(
            result,
            risk_free_rate=cfg.risk_free_rate,
            annualization_factor=cfg.annualization_factor,
        )

        logger.info(
            "backtest_complete",
            strategy=compiled.spec.name,
            n_trades=len(trades),
            final_equity=round(equity, 2),
            total_return_pct=round(result.metrics.get("total_return_pct", 0), 2),
        )

        return result

    def _process_long_bar(
        self,
        i: int,
        in_long: bool,
        entry_price: float,
        entry_bar: int,
        entry_sl: float,
        entry_tp: float,
        position_size: float,
        trade_mae: float,
        trade_mfe: float,
        trade_id: int,
        low_prices: np.ndarray,
        high_prices: np.ndarray,
        close_prices: np.ndarray,
        exit_long: np.ndarray,
        index: pd.DatetimeIndex,
        cfg: BacktestConfig,
    ) -> tuple[float, TradeRecord | None, bool, float, float]:
        """Process a bar when in a long position.

        Returns: (bar_pnl, trade_record_or_None, position_closed, updated_mae, updated_mfe)
        """
        if not in_long:
            return 0.0, None, False, trade_mae, trade_mfe

        if entry_sl > 0 and low_prices[i] <= entry_sl:
            exit_price = entry_sl
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "long", entry_bar, i, index,
                entry_price, exit_price, "sl",
                position_size, trade_mae, trade_mfe, cfg,
            )
            return bar_pnl, trade_rec, True, trade_mae, trade_mfe

        if entry_tp > 0 and high_prices[i] >= entry_tp:
            exit_price = entry_tp
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "long", entry_bar, i, index,
                entry_price, exit_price, "tp",
                position_size, trade_mae, trade_mfe, cfg,
            )
            return bar_pnl, trade_rec, True, trade_mae, trade_mfe

        if exit_long[i]:
            exit_price = close_prices[i]
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "long", entry_bar, i, index,
                entry_price, exit_price, "signal",
                position_size, trade_mae, trade_mfe, cfg,
            )
            return bar_pnl, trade_rec, True, trade_mae, trade_mfe

        unrealized = low_prices[i] - entry_price
        trade_mae = min(trade_mae, unrealized)
        unrealized_best = high_prices[i] - entry_price
        trade_mfe = max(trade_mfe, unrealized_best)
        return 0.0, None, False, trade_mae, trade_mfe

    def _process_short_bar(
        self,
        i: int,
        in_short: bool,
        entry_price: float,
        entry_bar: int,
        entry_sl: float,
        entry_tp: float,
        position_size: float,
        trade_mae: float,
        trade_mfe: float,
        trade_id: int,
        low_prices: np.ndarray,
        high_prices: np.ndarray,
        close_prices: np.ndarray,
        exit_short: np.ndarray,
        index: pd.DatetimeIndex,
        cfg: BacktestConfig,
    ) -> tuple[float, TradeRecord | None, bool, float, float]:
        """Process a bar when in a short position.

        Returns: (bar_pnl, trade_record_or_None, position_closed, updated_mae, updated_mfe)
        """
        if not in_short:
            return 0.0, None, False, trade_mae, trade_mfe

        if entry_sl > 0 and high_prices[i] >= entry_sl:
            exit_price = entry_sl
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "short", entry_bar, i, index,
                entry_price, exit_price, "sl",
                position_size, trade_mae, trade_mfe, cfg,
            )
            return bar_pnl, trade_rec, True, trade_mae, trade_mfe

        if entry_tp > 0 and low_prices[i] <= entry_tp:
            exit_price = entry_tp
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "short", entry_bar, i, index,
                entry_price, exit_price, "tp",
                position_size, trade_mae, trade_mfe, cfg,
            )
            return bar_pnl, trade_rec, True, trade_mae, trade_mfe

        if exit_short[i]:
            exit_price = close_prices[i]
            bar_pnl, trade_rec = self._close_trade(
                trade_id, "short", entry_bar, i, index,
                entry_price, exit_price, "signal",
                position_size, trade_mae, trade_mfe, cfg,
            )
            return bar_pnl, trade_rec, True, trade_mae, trade_mfe

        unrealized = entry_price - high_prices[i]
        trade_mae = min(trade_mae, unrealized)
        unrealized_best = entry_price - low_prices[i]
        trade_mfe = max(trade_mfe, unrealized_best)
        return 0.0, None, False, trade_mae, trade_mfe

    @staticmethod
    def _close_trade(
        trade_id: int,
        direction: str,
        entry_bar: int,
        exit_bar: int,
        index: pd.DatetimeIndex,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        position_size: float,
        mae: float,
        mfe: float,
        cfg: BacktestConfig,
    ) -> tuple[float, TradeRecord]:
        if direction == "long":
            pnl_per_unit = exit_price - entry_price
        else:
            pnl_per_unit = entry_price - exit_price

        pnl_pct = pnl_per_unit / entry_price * 100.0 if entry_price != 0 else 0.0
        pnl_dollar = pnl_per_unit * position_size

        commission = position_size * cfg.commission_per_unit
        pnl_dollar -= commission

        bars_held = exit_bar - entry_bar
        if bars_held < 0:  # pragma: no cover
            bars_held = 0

        record = TradeRecord(
            trade_id=trade_id,
            direction=direction,
            entry_time=index[entry_bar].to_pydatetime(),
            entry_price=entry_price,
            exit_time=index[exit_bar].to_pydatetime(),
            exit_price=exit_price,
            exit_reason=exit_reason,
            position_size=position_size,
            pnl=pnl_per_unit,
            pnl_pct=pnl_pct,
            pnl_dollar=pnl_dollar,
            bars_held=bars_held,
            mae=mae,
            mfe=mfe,
        )

        return pnl_dollar, record
