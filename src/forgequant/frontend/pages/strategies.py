"""
forgequant.frontend.pages.strategies — Saved strategies list page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from forgequant.frontend.components.layout import page_layout
from forgequant.frontend.state.app_state import AppState
from forgequant.frontend.styles import CARD_STYLE, COLORS


def _strategy_card(strategy: dict[str, Any]) -> rx.Component:
    block_count = (
        len(strategy.get("indicators", [])) + len(strategy.get("price_action", []))
        + len(strategy.get("entry_rules", [])) + len(strategy.get("exit_rules", []))
        + len(strategy.get("filters", [])) + len(strategy.get("money_management", []))
    )
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading(strategy.get("name", "Unnamed"), size="3", color=COLORS["text_primary"]),
                rx.spacer(),
                rx.button("🗑", on_click=AppState.delete_strategy(strategy.get("name", "")), variant="ghost", color_scheme="red", size="1"),
                width="100%",
                align="center",
            ),
            rx.text(strategy.get("description", "No description."), color=COLORS["text_secondary"], font_size="13px", no_of_lines=2),
            rx.hstack(
                rx.badge(strategy.get("timeframe", "?"), variant="soft"),
                rx.text(f"{block_count} blocks", font_size="12px", color=COLORS["text_muted"]),
                spacing="3",
            ),
            rx.hstack(
                rx.link(rx.button("Backtest", size="1", variant="outline"), href="/backtest"),
                rx.link(rx.button("Robustness", size="1", variant="outline"), href="/robustness"),
                spacing="2",
            ),
            spacing="3",
            width="100%",
        ),
        **CARD_STYLE,
    )


@rx.page(route="/strategies", title="ForgeQuant — Strategies")
def strategies_page() -> rx.Component:
    return page_layout(
        "Strategies",
        rx.cond(
            AppState.total_strategies > 0,
            rx.vstack(rx.foreach(AppState.strategies, _strategy_card), spacing="4", width="100%"),
            rx.box(
                rx.vstack(
                    rx.text("📋", font_size="48px"),
                    rx.text("No strategies saved yet.", color=COLORS["text_muted"]),
                    rx.link(rx.button("Generate one in AI Forge", size="2"), href="/forge"),
                    spacing="3",
                    align="center",
                    padding="60px",
                ),
                **CARD_STYLE,
            ),
        ),
    )
