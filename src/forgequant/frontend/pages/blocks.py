"""
forgequant.frontend.pages.blocks — Block Explorer page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from forgequant.frontend.components.layout import page_layout
from forgequant.frontend.state.app_state import AppState
from forgequant.frontend.state.blocks_state import BlocksState
from forgequant.frontend.styles import CARD_STYLE, CATEGORY_COLORS, COLORS, INPUT_STYLE


def _search_bar() -> rx.Component:
    return rx.hstack(
        rx.input(placeholder="Search blocks...", value=BlocksState.search_query, on_change=BlocksState.set_search, style=INPUT_STYLE, width="300px"),
        rx.hstack(
            rx.button("All", on_click=BlocksState.set_category_filter("all"), variant=rx.cond(BlocksState.selected_category == "all", "solid", "outline"), size="1"),
            rx.foreach(AppState.block_categories, lambda cat: rx.button(
                cat, on_click=BlocksState.set_category_filter(cat),
                variant=rx.cond(BlocksState.selected_category == cat, "solid", "outline"), size="1",
            )),
            spacing="2",
            flex_wrap="wrap",
        ),
        spacing="4",
        width="100%",
        align="center",
    )


def _block_card(block: dict[str, Any]) -> rx.Component:
    category = block.get("category", "indicator")
    color = CATEGORY_COLORS.get(category, COLORS["accent_blue"])
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.box(width="8px", height="8px", border_radius="50%", background=color),
                rx.text(block["name"], font_size="14px", font_weight="600", color=COLORS["text_primary"]),
                spacing="2",
                align="center",
            ),
            rx.text(block.get("category", ""), font_size="11px", color=COLORS["text_muted"]),
            rx.text(block.get("description", "")[:80], font_size="12px", color=COLORS["text_secondary"], no_of_lines=2),
            spacing="1",
        ),
        padding="12px",
        border=f"1px solid {COLORS['border']}",
        border_radius="8px",
        background=COLORS["bg_card"],
        cursor="pointer",
        _hover={"border_color": color},
        on_click=BlocksState.select_block(block["name"]),
        min_width="200px",
    )


def _block_grid() -> rx.Component:
    return rx.hstack(rx.foreach(BlocksState.filtered_blocks, _block_card), spacing="3", flex_wrap="wrap", width="100%")


def _detail_panel() -> rx.Component:
    block = BlocksState.selected_block_detail
    return rx.cond(
        BlocksState.has_selection,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.heading(block["name"], size="4", color=COLORS["text_primary"]),
                    rx.spacer(),
                    rx.badge(block["category"], color_scheme="blue"),
                    width="100%",
                    align="center",
                ),
                rx.text(block.get("description", ""), color=COLORS["text_secondary"], font_size="14px"),
                rx.separator(color=COLORS["border"]),
                rx.text("Parameters", font_size="14px", font_weight="600", color=COLORS["text_primary"]),
                rx.cond(
                    block["parameters"].length() > 0,
                    rx.table.root(
                        rx.table.header(rx.table.row(
                            rx.table.column_header_cell("Name"), rx.table.column_header_cell("Type"),
                            rx.table.column_header_cell("Default"), rx.table.column_header_cell("Min"),
                            rx.table.column_header_cell("Max"),
                        )),
                        rx.table.body(rx.foreach(block["parameters"], lambda p: rx.table.row(
                            rx.table.cell(rx.text(p["name"], font_weight="600")),
                            rx.table.cell(p["param_type"]),
                            rx.table.cell(str(p.get("default", "—"))),
                            rx.table.cell(str(p.get("min_value", "—"))),
                            rx.table.cell(str(p.get("max_value", "—"))),
                        ))),
                        width="100%",
                    ),
                    rx.text("No parameters", color=COLORS["text_muted"]),
                ),
                spacing="4",
                width="100%",
            ),
            **CARD_STYLE,
        ),
        rx.fragment(),
    )


@rx.page(route="/blocks", title="ForgeQuant — Block Explorer", on_load=AppState.initialise_app)
def blocks_page() -> rx.Component:
    return page_layout("Block Explorer", _search_bar(), _block_grid(), _detail_panel())
