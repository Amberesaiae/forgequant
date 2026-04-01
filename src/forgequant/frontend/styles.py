"""
forgequant.frontend.styles — Shared style constants and theme tokens.
"""

from __future__ import annotations

COLORS = {
    "bg_primary": "#0f1117",
    "bg_secondary": "#1a1d27",
    "bg_card": "#1e2130",
    "bg_card_hover": "#252840",
    "border": "#2a2d3e",
    "border_focus": "#4a6cf7",
    "text_primary": "#e4e4e7",
    "text_secondary": "#a1a1aa",
    "text_muted": "#71717a",
    "accent_blue": "#4a6cf7",
    "accent_green": "#22c55e",
    "accent_red": "#ef4444",
    "accent_amber": "#f59e0b",
    "accent_purple": "#a855f7",
}

CATEGORY_COLORS: dict[str, str] = {
    "indicator": "#3b82f6",
    "price_action": "#8b5cf6",
    "entry_rule": "#22c55e",
    "exit_rule": "#ef4444",
    "filter": "#f59e0b",
    "money_management": "#06b6d4",
}

SIDEBAR_WIDTH = "240px"
CONTENT_MAX_WIDTH = "1200px"
CARD_PADDING = "20px"
CARD_BORDER_RADIUS = "12px"

CARD_STYLE: dict = {
    "background": COLORS["bg_card"],
    "border": f"1px solid {COLORS['border']}",
    "border_radius": CARD_BORDER_RADIUS,
    "padding": CARD_PADDING,
    "width": "100%",
}

STAT_CARD_STYLE: dict = {
    **CARD_STYLE,
    "min_width": "180px",
    "cursor": "pointer",
}

INPUT_STYLE: dict = {
    "background": COLORS["bg_secondary"],
    "border": f"1px solid {COLORS['border']}",
    "color": COLORS["text_primary"],
    "width": "100%",
}
