"""
forgequant.frontend.components.sidebar — Application sidebar navigation.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.styles import COLORS, SIDEBAR_WIDTH


def sidebar_link(text: str, href: str, icon: str) -> rx.Component:
    return rx.link(
        rx.hstack(
            rx.text(icon, font_size="18px"),
            rx.text(text, font_size="14px", font_weight="500"),
            spacing="3",
            align="center",
            padding="10px 16px",
            border_radius="8px",
            width="100%",
            _hover={"background": COLORS["bg_card_hover"]},
        ),
        href=href,
        underline="none",
        color=COLORS["text_secondary"],
        _hover={"color": COLORS["text_primary"]},
        width="100%",
    )


def sidebar() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("⚡", font_size="24px"),
                rx.text("ForgeQuant", font_size="20px", font_weight="700", color=COLORS["text_primary"]),
                spacing="2",
                align="center",
                padding="20px 16px 30px 16px",
            ),
            rx.vstack(
                sidebar_link("Dashboard", "/", "📊"),
                sidebar_link("AI Forge", "/forge", "🤖"),
                sidebar_link("Blocks", "/blocks", "🧱"),
                sidebar_link("Strategies", "/strategies", "📋"),
                sidebar_link("Backtest", "/backtest", "📈"),
                sidebar_link("Robustness", "/robustness", "🛡️"),
                sidebar_link("Execution", "/execution", "⚡"),
                spacing="1",
                width="100%",
            ),
            rx.spacer(),
            rx.box(sidebar_link("Settings", "/settings", "⚙️"), padding_bottom="20px", width="100%"),
            height="100vh",
            width=SIDEBAR_WIDTH,
            padding="0 8px",
        ),
        position="fixed",
        left="0",
        top="0",
        height="100vh",
        width=SIDEBAR_WIDTH,
        background=COLORS["bg_secondary"],
        border_right=f"1px solid {COLORS['border']}",
        overflow_y="auto",
    )
