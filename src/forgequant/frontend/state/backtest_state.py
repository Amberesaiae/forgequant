"""
forgequant.frontend.state.backtest_state — State for the Backtest page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx


class BacktestState(rx.State):
    selected_strategy_name: str = ""
    symbol: str = "EURUSD"
    timeframe: str = "1H"
    date_from: str = "2023-01-01"
    date_to: str = "2024-12-31"
    initial_capital: float = 10000.0

    is_running: bool = False
    has_result: bool = False

    _strategy_names: list[str] = []

    metrics: dict[str, Any] = {}
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    drawdown_curve: list[dict[str, Any]] = []
    error_message: str = ""

    @rx.event
    def load_strategy_names(self) -> None:
        from forgequant.frontend.state.app_state import AppState

        try:
            app_state = self.get_state(AppState)
            self._strategy_names = [
                s.get("name", "?") for s in app_state.strategies
            ]
        except Exception:
            self._strategy_names = []

    @rx.var(auto_deps=False, deps=["_strategy_names"])
    def available_strategies(self) -> list[str]:
        return self._strategy_names

    @rx.var(auto_deps=False, deps=["metrics"])
    def sharpe_ratio(self) -> str:
        return f"{self.metrics.get('sharpe_ratio', 0.0):.2f}"

    @rx.var(auto_deps=False, deps=["metrics"])
    def max_drawdown(self) -> str:
        val = self.metrics.get("max_drawdown", 0.0)
        return f"{val:.1%}"

    @rx.var(auto_deps=False, deps=["metrics"])
    def profit_factor(self) -> str:
        return f"{self.metrics.get('profit_factor', 0.0):.2f}"

    @rx.var(auto_deps=False, deps=["metrics"])
    def total_trades(self) -> int:
        return self.metrics.get("total_trades", 0)

    @rx.var(auto_deps=False, deps=["metrics"])
    def win_rate(self) -> str:
        val = self.metrics.get("win_rate", 0.0)
        return f"{val:.1%}"

    @rx.var(auto_deps=False, deps=["metrics"])
    def net_profit(self) -> str:
        val = self.metrics.get("net_profit", 0.0)
        return f"${val:,.2f}"

    @rx.var(auto_deps=False, deps=["trades"])
    def trades_with_colors(self) -> list[dict[str, Any]]:
        green = "#22c55e"
        red = "#ef4444"
        result = []
        for t in self.trades:
            row = dict(t)
            row["color"] = green if t.get("pnl", 0) >= 0 else red
            result.append(row)
        return result

    @rx.event
    def set_strategy(self, value: str) -> None:
        self.selected_strategy_name = value

    @rx.event
    def set_symbol(self, value: str) -> None:
        self.symbol = value

    @rx.event
    def set_timeframe(self, value: str) -> None:
        self.timeframe = value

    @rx.event
    def set_date_from(self, value: str) -> None:
        self.date_from = value

    @rx.event
    def set_date_to(self, value: str) -> None:
        self.date_to = value

    @rx.event
    def set_initial_capital(self, value: str) -> None:
        try:
            self.initial_capital = float(value)
        except ValueError:
            pass

    @rx.event
    def run_backtest(self) -> None:
        if not self.selected_strategy_name:
            self.error_message = "Select a strategy first."
            return

        self.error_message = "Backtest not yet wired to real engine — coming soon."
        self.has_result = False
        self.metrics = {}
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
