"""
forgequant.frontend.state.settings_state — State for the Settings page.
"""

from __future__ import annotations

import reflex as rx


class SettingsState(rx.State):
    """Manages application settings."""

    openai_key: str = ""
    anthropic_key: str = ""
    groq_key: str = ""
    mt5_terminal_path: str = ""
    mt5_login: str = ""
    mt5_password: str = ""
    mt5_server: str = ""
    min_sharpe: float = 0.5
    max_drawdown: float = 0.30
    min_profit_factor: float = 1.2
    min_trades: int = 30
    save_message: str = ""

    @rx.event
    def load_settings(self) -> None:
        try:
            from forgequant.core.config import get_settings
            settings = get_settings()
            self.openai_key = settings.openai_api_key or ""
            self.anthropic_key = settings.anthropic_api_key or ""
            self.groq_key = settings.groq_api_key or ""
            self.mt5_terminal_path = settings.mt5_terminal_path or ""
            self.mt5_login = str(settings.mt5_login or "")
            self.mt5_password = settings.mt5_password or ""
            self.mt5_server = settings.mt5_server or ""
        except Exception as exc:
            self.save_message = f"Error loading settings: {exc}"

    @rx.event
    def set_openai_key(self, value: str) -> None:
        self.openai_key = value

    @rx.event
    def set_anthropic_key(self, value: str) -> None:
        self.anthropic_key = value

    @rx.event
    def set_groq_key(self, value: str) -> None:
        self.groq_key = value

    @rx.event
    def set_mt5_path(self, value: str) -> None:
        self.mt5_terminal_path = value

    @rx.event
    def set_mt5_login(self, value: str) -> None:
        self.mt5_login = value

    @rx.event
    def set_mt5_password(self, value: str) -> None:
        self.mt5_password = value

    @rx.event
    def set_mt5_server(self, value: str) -> None:
        self.mt5_server = value

    @rx.event
    def set_min_sharpe(self, value: str) -> None:
        try:
            self.min_sharpe = float(value)
        except ValueError:
            pass

    @rx.event
    def set_max_drawdown(self, value: str) -> None:
        try:
            self.max_drawdown = float(value)
        except ValueError:
            pass

    @rx.event
    def set_min_profit_factor(self, value: str) -> None:
        try:
            self.min_profit_factor = float(value)
        except ValueError:
            pass

    @rx.event
    def set_min_trades(self, value: str) -> None:
        try:
            self.min_trades = int(value)
        except ValueError:
            pass

    @rx.event
    def save_settings(self) -> None:
        self.save_message = "Settings saved (in-memory only)."
