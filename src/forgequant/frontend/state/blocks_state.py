"""
forgequant.frontend.state.blocks_state — State for the Block Explorer page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx


class BlocksState(rx.State):
    search_query: str = ""
    selected_category: str = "all"
    selected_block_name: str = ""

    _local_catalog: dict[str, dict[str, Any]] = {}

    detail_name: str = ""
    detail_description: str = ""
    detail_category: str = ""
    detail_tags: list[str] = []
    detail_parameters: list[dict[str, Any]] = []
    detail_outputs: list[str] = []

    def load_catalog(self) -> None:
        from forgequant.frontend.state.app_state import AppState

        try:
            app_state = self.get_state(AppState)
            self._local_catalog = dict(app_state.block_catalog)
        except Exception:
            self._local_catalog = {}

    def _reset_detail(self) -> None:
        """Clear all detail-view fields."""
        self.selected_block_name = ""
        self.detail_name = ""
        self.detail_description = ""
        self.detail_category = ""
        self.detail_tags = []
        self.detail_parameters = []
        self.detail_outputs = []

    @rx.event
    def set_search(self, value: str) -> None:
        self.search_query = value
        self._reset_detail()

    @rx.event
    def set_category_filter(self, category: str) -> None:
        self.selected_category = category
        self._reset_detail()

    @rx.event
    def select_block(self, name: str) -> None:
        self.selected_block_name = name
        info = self._local_catalog.get(name, {})
        self.detail_name = info.get("display_name", name)
        self.detail_description = info.get("description", "")
        self.detail_category = info.get("category", "")
        self.detail_tags = info.get("tags", [])
        self.detail_parameters = info.get("parameters", [])
        self.detail_outputs = info.get("outputs", [])

    @rx.var(auto_deps=False, deps=["_local_catalog", "selected_category", "search_query"])
    def filtered_blocks(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        query = self.search_query.lower()

        for name, info in sorted(self._local_catalog.items()):
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

            row = {"name": name, **info}
            row["_desc_short"] = info.get("description", "")[:80]
            results.append(row)

        return results

    @rx.var(auto_deps=False, deps=["selected_block_name"])
    def has_selection(self) -> bool:
        return len(self.selected_block_name) > 0
