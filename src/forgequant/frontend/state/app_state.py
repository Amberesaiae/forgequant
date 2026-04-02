"""
forgequant.frontend.state.app_state — Root application state.
"""

from __future__ import annotations

from typing import Any

import reflex as rx


class AppState(rx.State):
    """Root state for the ForgeQuant dashboard."""

    strategies: list[dict[str, Any]] = []
    block_catalog: dict[str, dict[str, Any]] = {}
    is_initialised: bool = False

    @rx.event
    def initialise_app(self) -> None:
        """Load the block catalog on first page load."""
        if self.is_initialised:
            return
        try:
            import forgequant.blocks  # noqa: F401 — eager registration

            from forgequant.blocks.registry import BlockRegistry
            self.block_catalog = BlockRegistry().to_catalog_dict()
        except Exception:
            self.block_catalog = {}
        self.strategies = []
        self.is_initialised = True

    @rx.var
    def total_strategies(self) -> int:
        return len(self.strategies)

    @rx.var
    def total_blocks(self) -> int:
        return len(self.block_catalog)

    @rx.var
    def block_categories(self) -> list[str]:
        cats: set[str] = set()
        for block_info in self.block_catalog.values():
            cats.add(block_info.get("category", "unknown"))
        return sorted(cats)

    @rx.var
    def block_categories_count(self) -> int:
        return len(self.block_categories)

    @rx.event
    def save_strategy(self, spec_dict: dict[str, Any]) -> None:
        self.strategies = [
            s for s in self.strategies if s.get("name") != spec_dict.get("name")
        ]
        self.strategies.append(spec_dict)

    @rx.event
    def delete_strategy(self, name: str) -> None:
        self.strategies = [s for s in self.strategies if s.get("name") != name]

    @rx.var
    def strategy_display_list(self) -> list[dict[str, Any]]:
        result = []
        for s in self.strategies:
            symbols = s.get("symbols", [])
            block_count = (
                len(s.get("indicators", []))
                + len(s.get("entry_rules", []))
                + len(s.get("exit_rules", []))
                + len(s.get("filters", []))
                + len(s.get("money_management", []))
                + len(s.get("price_action", []))
            )
            result.append({
                **s,
                "_symbols_str": ", ".join(symbols) if symbols else "—",
                "_block_count": block_count,
            })
        return result
