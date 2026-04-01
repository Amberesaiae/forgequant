"""Tests for AI Forge prompt builder."""

from __future__ import annotations

import pytest

from forgequant.ai_forge.exceptions import PromptBuildError
from forgequant.ai_forge.prompt import (
    build_system_prompt,
    build_user_message,
    _format_block_catalog,
)
from forgequant.blocks.registry import BlockRegistry

# Import to register blocks
import forgequant.blocks.indicators  # noqa: F401
import forgequant.blocks.entry_rules  # noqa: F401
import forgequant.blocks.exit_rules  # noqa: F401
import forgequant.blocks.money_management  # noqa: F401
import forgequant.blocks.filters  # noqa: F401
import forgequant.blocks.price_action  # noqa: F401


@pytest.fixture
def populated_registry() -> BlockRegistry:
    registry = BlockRegistry()
    for module_name in [
        "forgequant.blocks.indicators",
        "forgequant.blocks.entry_rules",
        "forgequant.blocks.exit_rules",
        "forgequant.blocks.money_management",
        "forgequant.blocks.filters",
        "forgequant.blocks.price_action",
    ]:
        import importlib
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and hasattr(attr, "metadata")
                and attr_name not in ("BaseBlock",)
            ):
                try:
                    registry.register_class(attr)
                except Exception:
                    pass
    return registry


class TestBlockCatalog:
    def test_format_includes_all_categories(
        self, populated_registry: BlockRegistry
    ) -> None:
        catalog = _format_block_catalog(populated_registry)
        assert "INDICATOR" in catalog
        assert "ENTRY RULE" in catalog
        assert "EXIT RULE" in catalog
        assert "MONEY MANAGEMENT" in catalog
        assert "FILTER" in catalog

    def test_format_includes_block_names(
        self, populated_registry: BlockRegistry
    ) -> None:
        catalog = _format_block_catalog(populated_registry)
        assert "ema" in catalog
        assert "rsi" in catalog
        assert "crossover_entry" in catalog
        assert "fixed_tpsl" in catalog
        assert "fixed_risk" in catalog
        assert "trend_filter" in catalog

    def test_format_includes_parameters(
        self, populated_registry: BlockRegistry
    ) -> None:
        catalog = _format_block_catalog(populated_registry)
        assert "period" in catalog
        assert "default=" in catalog


class TestBuildSystemPrompt:
    def test_builds_successfully(
        self, populated_registry: BlockRegistry
    ) -> None:
        prompt = build_system_prompt(registry=populated_registry)
        assert len(prompt) > 1000
        assert "ForgeQuant" in prompt
        assert "AVAILABLE BLOCKS" in prompt

    def test_includes_output_schema(
        self, populated_registry: BlockRegistry
    ) -> None:
        prompt = build_system_prompt(registry=populated_registry)
        assert "StrategySpec" in prompt or "properties" in prompt

    def test_includes_rag_context(
        self, populated_registry: BlockRegistry
    ) -> None:
        prompt = build_system_prompt(
            registry=populated_registry,
            rag_context="Use EMA crossover for trend identification.",
        )
        assert "EMA crossover" in prompt

    def test_no_rag_context(
        self, populated_registry: BlockRegistry
    ) -> None:
        prompt = build_system_prompt(registry=populated_registry)
        assert "No additional context" in prompt

    def test_empty_registry_raises(self) -> None:
        empty_reg = BlockRegistry()
        empty_reg.clear()
        with pytest.raises(PromptBuildError, match="No blocks"):
            build_system_prompt(registry=empty_reg)


class TestBuildUserMessage:
    def test_basic_message(self) -> None:
        msg = build_user_message(idea="EMA crossover system")
        assert "EMA crossover" in msg
        assert "Timeframe:" in msg

    def test_includes_timeframe(self) -> None:
        msg = build_user_message(idea="Test", timeframe="4h")
        assert "4h" in msg

    def test_includes_instruments(self) -> None:
        msg = build_user_message(
            idea="Test", instruments=["EURUSD", "GBPUSD"]
        )
        assert "EURUSD" in msg
        assert "GBPUSD" in msg

    def test_includes_style(self) -> None:
        msg = build_user_message(idea="Test", style="mean_reversion")
        assert "mean_reversion" in msg

    def test_includes_additional_requirements(self) -> None:
        msg = build_user_message(
            idea="Test",
            additional_requirements="Must have max 2% drawdown",
        )
        assert "2% drawdown" in msg
