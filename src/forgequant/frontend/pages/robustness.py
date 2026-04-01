"""
forgequant.frontend.pages.robustness — Robustness testing page.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.components.layout import page_layout
from forgequant.frontend.styles import CARD_STYLE, COLORS


def _test_card(name: str, description: str, status: str = "pending") -> rx.Component:
    status_icon = {"pending": "⏳", "running": "🔄", "pass": "✅", "fail": "❌"}
    status_color = {"pending": COLORS["text_muted"], "running": COLORS["accent_blue"], "pass": COLORS["accent_green"], "fail": COLORS["accent_red"]}
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(status_icon.get(status, "⏳"), font_size="20px"),
                rx.text(name, font_size="16px", font_weight="600", color=COLORS["text_primary"]),
                rx.spacer(),
                rx.text(status.upper(), font_size="12px", font_weight="700", color=status_color.get(status, COLORS["text_muted"])),
                width="100%",
                align="center",
            ),
            rx.text(description, font_size="13px", color=COLORS["text_secondary"]),
            spacing="2",
            width="100%",
        ),
        **CARD_STYLE,
    )


@rx.page(route="/robustness", title="ForgeQuant — Robustness")
def robustness_page() -> rx.Component:
    return page_layout(
        "Robustness Suite",
        rx.text("Run the full robustness suite to guard against overfitting.", color=COLORS["text_secondary"], font_size="14px"),
        rx.button("Run All Tests", size="3", color_scheme="purple", width="100%"),
        _test_card("Walk-Forward Analysis", "Tests out-of-sample stability by training on rolling windows and testing on unseen data."),
        _test_card("Combinatorial Purged Cross-Validation", "Creates multiple train/test splits with purging to prevent data leakage."),
        _test_card("Monte Carlo Simulation", "Randomises trade order and slippage to test robustness to execution variation."),
        _test_card("Parameter Sensitivity", "Varies each parameter ±20% to ensure performance is not brittle."),
        _test_card("Stability Analysis", "Splits the backtest period into sub-periods and checks for consistent performance."),
    )
