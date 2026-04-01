"""
forgequant.frontend.pages.settings — Application settings page.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.components.layout import page_layout
from forgequant.frontend.state.settings_state import SettingsState
from forgequant.frontend.styles import CARD_STYLE, COLORS, INPUT_STYLE


def _section(title: str, *children: rx.Component) -> rx.Component:
    return rx.box(rx.vstack(rx.text(title, font_size="16px", font_weight="600", color=COLORS["text_primary"]), *children, spacing="4", width="100%"), **CARD_STYLE)


def _field(label: str, value, on_change, input_type: str = "text", placeholder: str = "") -> rx.Component:
    return rx.vstack(
        rx.text(label, font_size="13px", color=COLORS["text_muted"]),
        rx.input(value=value, on_change=on_change, type=input_type, placeholder=placeholder, style=INPUT_STYLE),
        spacing="1",
        width="100%",
    )


@rx.page(route="/settings", title="ForgeQuant — Settings", on_load=SettingsState.load_settings)
def settings_page() -> rx.Component:
    return page_layout(
        "Settings",
        _section(
            "API Keys",
            _field("OpenAI API Key", SettingsState.openai_key, SettingsState.set_openai_key, input_type="password", placeholder="sk-..."),
            _field("Anthropic API Key", SettingsState.anthropic_key, SettingsState.set_anthropic_key, input_type="password", placeholder="sk-ant-..."),
            _field("Groq API Key", SettingsState.groq_key, SettingsState.set_groq_key, input_type="password", placeholder="gsk_..."),
        ),
        _section(
            "MetaTrader 5",
            _field("Terminal Path", SettingsState.mt5_terminal_path, SettingsState.set_mt5_path),
            rx.hstack(
                rx.box(_field("Login", SettingsState.mt5_login, SettingsState.set_mt5_login), flex="1"),
                rx.box(_field("Password", SettingsState.mt5_password, SettingsState.set_mt5_password, input_type="password"), flex="1"),
                rx.box(_field("Server", SettingsState.mt5_server, SettingsState.set_mt5_server), flex="1"),
                spacing="4",
                width="100%",
            ),
        ),
        _section(
            "Robustness Thresholds",
            rx.hstack(
                rx.box(_field("Min Sharpe", SettingsState.min_sharpe.to(str), SettingsState.set_min_sharpe), flex="1"),
                rx.box(_field("Max Drawdown", SettingsState.max_drawdown.to(str), SettingsState.set_max_drawdown), flex="1"),
                rx.box(_field("Min Profit Factor", SettingsState.min_profit_factor.to(str), SettingsState.set_min_profit_factor), flex="1"),
                rx.box(_field("Min Trades", SettingsState.min_trades.to(str), SettingsState.set_min_trades), flex="1"),
                spacing="4",
                width="100%",
            ),
        ),
        rx.button("Save Settings", on_click=SettingsState.save_settings, color_scheme="blue", width="100%", size="3"),
        rx.cond(SettingsState.save_message != "", rx.callout(SettingsState.save_message, icon="info", color_scheme="blue", width="100%"), rx.fragment()),
    )
