"""
forgequant.frontend.state.blocks_state — State for the Block Explorer page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx


class BlocksState(rx.State):
    """Manages the Block Explorer page state."""

    search_query: str = ""
    selected_category: str = "all"
    selected_block_name: str = ""

    @rx.event
    def set_search(self, value: str) -> None:
        self.search_query = value
        self.selected_block_name = ""

    @rx.event
    def set_category_filter(self, category: str) -> None:
        self.selected_category = category
        self.selected_block_name = ""

    @rx.event
    def select_block(self, name: str) -> None:
        self.selected_block_name = name

    @rx.var
    def filtered_blocks(self) -> list[dict[str, Any]]:
        from forgequant.frontend.state.app_state import AppState
        try:
            app_state = self.get_parent_state()
            catalog = app_state.block_catalog
        except Exception:
            catalog = {}

        results: list[dict[str, Any]] = []
        query = self.search_query.lower()

        for name, info in sorted(catalog.items()):
            if self.selected_category != "all":
                if info.get("category") != self.selected_category:
                    continue
            if query:
                searchable = " ".join([
                    name.lower(),
                    info.get("display_name", "").lower(),
                    info.get("description", "").lower(),
                    " ".join(info.get("tags", [])).lower(),
                ])
                if query not in searchable:
                    continue
            results.append({"name": name, **info})
        return results

    @rx.var
    def selected_block_detail(self) -> dict[str, Any]:
        if not self.selected_block_name:
            return {}
        try:
            app_state = self.get_parent_state()
            catalog = app_state.block_catalog
        except Exception:
            catalog = {}
        info = catalog.get(self.selected_block_name, {})
        if info:
            return {"name": self.selected_block_name, **info}
        return {}

    @rx.var
    def has_selection(self) -> bool:
        return len(self.selected_block_name) > 0
