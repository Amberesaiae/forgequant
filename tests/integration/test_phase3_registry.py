"""
Integration test verifying all Phase 3 blocks (price action + entry rules)
are properly registered and can execute on sample data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

# Force registration
import forgequant.blocks.price_action  # noqa: F401
import forgequant.blocks.entry_rules  # noqa: F401


EXPECTED_PRICE_ACTION = [
    "breakout",
    "pullback",
    "higher_high_lower_low",
    "support_resistance",
]

EXPECTED_ENTRY_RULES = [
    "crossover_entry",
    "threshold_cross_entry",
    "confluence_entry",
    "reversal_pattern_entry",
]


class TestPriceActionRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_PRICE_ACTION)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_PRICE_ACTION)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.PRICE_ACTION

    @pytest.mark.parametrize("block_name", EXPECTED_PRICE_ACTION)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)


class TestEntryRulesRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.ENTRY_RULE

    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)

    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_has_description(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.description) > 20

    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_has_tags(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.tags) >= 1
