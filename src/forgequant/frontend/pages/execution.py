"""
forgequant.frontend.pages.execution — Live execution monitoring page.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.components.layout import page_layout
from forgequant.frontend.components.stat_card import stat_card
from forgequant.frontend.state.execution_state import ExecutionState
from forgequant.frontend.styles import CARD_STYLE, COLORS


def _connection_panel() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.text(ExecutionState.connection_status_text, font_size="16px", font_weight="600"),
            rx.spacer(),
            rx.cond(
                ExecutionState.is_connected,
                rx.button("Disconnect", on_click=ExecutionState.disconnect_mt5, color_scheme="red", variant="outline", size="2"),
                rx.button("Connect to MT5", on_click=ExecutionState.connect_mt5, color_scheme="green", size="2"),
            ),
            width="100%",
            align="center",
        ),
        **CARD_STYLE,
    )


def _account_panel() -> rx.Component:
    return rx.cond(
        ExecutionState.is_connected,
        rx.hstack(
            stat_card("Balance", f"${ExecutionState.account_balance:,.2f}", COLORS["accent_blue"]),
            stat_card("Equity", f"${ExecutionState.account_equity:,.2f}", COLORS["accent_green"]),
            stat_card("Open Positions", ExecutionState.position_count.to(str), COLORS["accent_purple"]),
            stat_card("Unrealised P&L", f"${ExecutionState.unrealised_pnl:,.2f}", rx.cond(ExecutionState.unrealised_pnl >= 0, COLORS["accent_green"], COLORS["accent_red"])),
            spacing="4",
            flex_wrap="wrap",
            width="100%",
        ),
        rx.fragment(),
    )


def _logs_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text("Execution Logs", font_size="16px", font_weight="600", color=COLORS["text_primary"]),
            rx.box(
                rx.vstack(
                    rx.foreach(ExecutionState.execution_logs, lambda entry: rx.text(entry, font_size="12px", font_family="monospace", color=COLORS["text_secondary"])),
                    spacing="1",
                    width="100%",
                ),
                max_height="300px",
                overflow_y="auto",
                padding="12px",
                background=COLORS["bg_secondary"],
                border_radius="8px",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        **CARD_STYLE,
    )


@rx.page(route="/execution", title="ForgeQuant — Execution")
def execution_page() -> rx.Component:
    return page_layout(
        "Execution",
        rx.callout("Live trading is not yet production-ready. Use paper accounts only.", icon="alert_triangle", color_scheme="yellow", width="100%"),
        _connection_panel(),
        _account_panel(),
        _logs_panel(),
    )
