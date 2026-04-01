"""
forgequant.frontend.components.stat_card — Metric display card.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.styles import COLORS, STAT_CARD_STYLE


def stat_card(label: str, value: str, color: str = COLORS["accent_blue"]) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(label, font_size="12px", color=COLORS["text_muted"], text_transform="uppercase", letter_spacing="0.05em", font_weight="600"),
            rx.text(value, font_size="28px", font_weight="700", color=color),
            spacing="1",
            align="start",
        ),
        **STAT_CARD_STYLE,
    )
