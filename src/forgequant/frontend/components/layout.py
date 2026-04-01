"""
forgequant.frontend.components.layout — Page layout wrapper with sidebar.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.components.sidebar import sidebar
from forgequant.frontend.styles import COLORS, CONTENT_MAX_WIDTH, SIDEBAR_WIDTH


def page_layout(title: str, *children: rx.Component) -> rx.Component:
    return rx.box(
        sidebar(),
        rx.box(
            rx.vstack(
                rx.heading(title, size="6", color=COLORS["text_primary"], font_weight="700"),
                rx.separator(color=COLORS["border"]),
                *children,
                spacing="5",
                width="100%",
                max_width=CONTENT_MAX_WIDTH,
                padding="30px",
            ),
            margin_left=SIDEBAR_WIDTH,
            min_height="100vh",
            background=COLORS["bg_primary"],
            width=f"calc(100% - {SIDEBAR_WIDTH})",
        ),
        width="100%",
        min_height="100vh",
        background=COLORS["bg_primary"],
        color=COLORS["text_primary"],
    )
