"""
forgequant.frontend.state.execution_state — State for the Execution page.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import reflex as rx


class ExecutionState(rx.State):
    """Manages the Execution monitoring page."""

    is_connected: bool = False
    connection_error: str = ""
    account_balance: float = 0.0
    account_equity: float = 0.0
    account_margin: float = 0.0
    deployed_strategies: list[dict[str, Any]] = []
    open_positions: list[dict[str, Any]] = []
    order_journal: list[dict[str, Any]] = []
    execution_logs: list[str] = []

    @rx.var
    def position_count(self) -> int:
        return len(self.open_positions)

    @rx.var
    def unrealised_pnl(self) -> float:
        return sum(p.get("unrealised_pnl", 0.0) for p in self.open_positions)

    @rx.var
    def connection_status_text(self) -> str:
        return "Connected" if self.is_connected else "Disconnected"

    @rx.event
    def connect_mt5(self) -> None:
        self.is_connected = True
        self.account_balance = 10000.00
        self.account_equity = 10245.30
        self.account_margin = 120.00
        self.connection_error = ""
        self._add_log("Connected to MT5 terminal")

    @rx.event
    def disconnect_mt5(self) -> None:
        self.is_connected = False
        self.account_balance = 0.0
        self.account_equity = 0.0
        self.open_positions = []
        self._add_log("Disconnected from MT5 terminal")

    @rx.event
    def deploy_strategy(self, strategy_name: str) -> None:
        self.deployed_strategies.append({
            "name": strategy_name, "status": "active",
            "started": "2024-01-01 10:00:00", "trades_today": 0,
        })
        self._add_log(f"Deployed strategy: {strategy_name}")

    @rx.event
    def stop_strategy(self, strategy_name: str) -> None:
        self.deployed_strategies = [
            {**s, "status": "stopped"} if s["name"] == strategy_name else s
            for s in self.deployed_strategies
        ]
        self._add_log(f"Stopped strategy: {strategy_name}")

    def _add_log(self, message: str) -> None:
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        self.execution_logs = [f"[{timestamp}] {message}"] + self.execution_logs[:99]
