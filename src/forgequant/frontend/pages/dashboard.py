"""
forgequant.frontend.pages.dashboard — Main dashboard page.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.components.layout import page_layout
from forgequant.frontend.components.stat_card import stat_card
from forgequant.frontend.state.app_state import AppState
from forgequant.frontend.styles import CARD_STYLE, COLORS


def _strategy_table() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text("Strategy Overview", font_size="16px", font_weight="600", color=COLORS["text_primary"]),
            rx.cond(
                AppState.total_strategies > 0,
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Name"),
                            rx.table.column_header_cell("Timeframe"),
                            rx.table.column_header_cell("Symbols"),
                            rx.table.column_header_cell("Blocks"),
                        ),
                    ),
                    rx.table.body(
                        rx.foreach(
                            AppState.strategies,
                            lambda s: rx.table.row(
                                rx.table.cell(rx.text(s["name"], font_weight="600", color=COLORS["text_primary"])),
                                rx.table.cell(s.get("timeframe", "—")),
                                rx.table.cell(rx.text(", ".join(s.get("symbols", [])), color=COLORS["text_secondary"])),
                                rx.table.cell(rx.text(str(
                                    len(s.get("indicators", [])) + len(s.get("entry_rules", []))
                                    + len(s.get("exit_rules", [])) + len(s.get("filters", []))
                                    + len(s.get("money_management", [])) + len(s.get("price_action", []))
                                ), color=COLORS["text_secondary"])),
                            ),
                        ),
                    ),
                    width="100%",
                ),
                rx.box(
                    rx.vstack(
                        rx.text("📋", font_size="48px"),
                        rx.text("No strategies yet", color=COLORS["text_muted"], font_size="16px"),
                        rx.link(rx.button("Open AI Forge", size="2"), href="/forge"),
                        spacing="3",
                        align="center",
                        padding="40px",
                    ),
                    width="100%",
                ),
            ),
            spacing="4",
            width="100%",
        ),
        **CARD_STYLE,
    )


@rx.page(route="/", title="ForgeQuant — Dashboard", on_load=AppState.initialise_app)
def dashboard_page() -> rx.Component:
    return page_layout(
        "Dashboard",
        rx.hstack(
            stat_card("Total Strategies", AppState.total_strategies.to(str), COLORS["accent_blue"]),
            stat_card("Available Blocks", AppState.total_blocks.to(str), COLORS["accent_green"]),
            stat_card("Block Categories", rx.cond(
                AppState.total_blocks > 0,
                AppState.block_categories.length().to(str),
                "0",
            ), COLORS["accent_purple"]),
            spacing="4",
            flex_wrap="wrap",
            width="100%",
        ),
        _strategy_table(),
    )
