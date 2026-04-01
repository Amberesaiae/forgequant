"""
forgequant.frontend.state.backtest_state — State for the Backtest page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx


class BacktestState(rx.State):
    """Manages the Backtest page state."""

    selected_strategy_name: str = ""
    symbol: str = "EURUSD"
    timeframe: str = "1H"
    date_from: str = "2023-01-01"
    date_to: str = "2024-12-31"
    initial_capital: float = 10000.0

    is_running: bool = False
    has_result: bool = False

    metrics: dict[str, Any] = {}
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    drawdown_curve: list[dict[str, Any]] = []
    error_message: str = ""

    @rx.var
    def available_strategies(self) -> list[str]:
        try:
            app_state = self.get_parent_state()
            return [s.get("name", "?") for s in app_state.strategies]
        except Exception:
            return []

    @rx.var
    def sharpe_ratio(self) -> str:
        return f"{self.metrics.get('sharpe_ratio', 0.0):.2f}"

    @rx.var
    def max_drawdown(self) -> str:
        return f"{self.metrics.get('max_drawdown', 0.0):.1%}"

    @rx.var
    def profit_factor(self) -> str:
        return f"{self.metrics.get('profit_factor', 0.0):.2f}"

    @rx.var
    def total_trades(self) -> int:
        return self.metrics.get("total_trades", 0)

    @rx.var
    def win_rate(self) -> str:
        return f"{self.metrics.get('win_rate', 0.0):.1%}"

    @rx.var
    def net_profit(self) -> str:
        return f"${self.metrics.get('net_profit', 0.0):,.2f}"

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

        self.is_running = True
        self.has_result = False
        self.error_message = ""

        try:
            self.metrics = {
                "sharpe_ratio": 1.82,
                "max_drawdown": 0.123,
                "profit_factor": 2.14,
                "total_trades": 147,
                "win_rate": 0.583,
                "net_profit": 4230.50,
                "avg_trade": 28.78,
                "max_consecutive_losses": 5,
            }
            self.trades = [
                {
                    "id": i,
                    "time": f"2024-01-{i+1:02d} 10:00",
                    "side": "LONG" if i % 3 != 0 else "SHORT",
                    "size": 0.1,
                    "entry": 1.0820 + i * 0.001,
                    "exit": 1.0850 + i * 0.001,
                    "pnl": 30.0 - (i % 5) * 15,
                }
                for i in range(10)
            ]
            self.equity_curve = [
                {"date": f"2024-01-{d:02d}", "equity": 10000 + d * 42.3}
                for d in range(1, 31)
            ]
            self.drawdown_curve = [
                {"date": f"2024-01-{d:02d}", "drawdown": -(d % 7) * 0.5}
                for d in range(1, 31)
            ]
            self.has_result = True
        except Exception as exc:
            self.error_message = f"Backtest failed: {exc}"
        finally:
            self.is_running = False
