"""
Integration test verifying all indicator blocks are properly
registered and can execute on sample data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

# Force import of all indicator blocks so they register
import forgequant.blocks.indicators  # noqa: F401


EXPECTED_INDICATORS = [
    "adx",
    "atr",
    "bollinger_bands",
    "ema",
    "ichimoku",
    "macd",
    "rsi",
    "stochastic",
]


class TestAllIndicatorsRegistered:
    def test_all_present(self) -> None:
        registry = BlockRegistry()
        for name in EXPECTED_INDICATORS:
            assert name in registry, f"Indicator '{name}' not found in registry"

    def test_count(self) -> None:
        registry = BlockRegistry()
        indicators = registry.list_by_category(BlockCategory.INDICATOR)
        assert len(indicators) >= len(EXPECTED_INDICATORS)

    def test_all_are_indicator_category(self) -> None:
        registry = BlockRegistry()
        for name in EXPECTED_INDICATORS:
            cls = registry.get_or_raise(name)
            assert cls.metadata.category == BlockCategory.INDICATOR


class TestAllIndicatorsExecute:
    @pytest.mark.parametrize("block_name", EXPECTED_INDICATORS)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)

        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)
            assert not result.columns.empty
        elif isinstance(result, pd.Series):
            assert len(result) == len(sample_ohlcv)

    @pytest.mark.parametrize("block_name", EXPECTED_INDICATORS)
    def test_has_description(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.description) > 20

    @pytest.mark.parametrize("block_name", EXPECTED_INDICATORS)
    def test_has_tags(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.tags) >= 1

    @pytest.mark.parametrize("block_name", EXPECTED_INDICATORS)
    def test_has_typical_use(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.typical_use) > 20
