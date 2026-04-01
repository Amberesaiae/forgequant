"""
Integration test verifying all Phase 4 blocks (exit rules, money management,
and filters) are properly registered and can execute on sample data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

# Force registration
import forgequant.blocks.exit_rules  # noqa: F401
import forgequant.blocks.money_management  # noqa: F401
import forgequant.blocks.filters  # noqa: F401


EXPECTED_EXIT_RULES = [
    "fixed_tpsl",
    "trailing_stop",
    "time_based_exit",
    "breakeven_stop",
]

EXPECTED_MONEY_MANAGEMENT = [
    "fixed_risk",
    "volatility_targeting",
    "kelly_fractional",
    "atr_based_sizing",
]

EXPECTED_FILTERS = [
    "trading_session",
    "spread_filter",
    "max_drawdown_filter",
    "trend_filter",
]

ALL_PHASE4_BLOCKS = EXPECTED_EXIT_RULES + EXPECTED_MONEY_MANAGEMENT + EXPECTED_FILTERS


class TestExitRulesRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_EXIT_RULES)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_EXIT_RULES)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.EXIT_RULE

    @pytest.mark.parametrize("block_name", EXPECTED_EXIT_RULES)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)


class TestMoneyManagementRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_MONEY_MANAGEMENT)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_MONEY_MANAGEMENT)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.MONEY_MANAGEMENT

    @pytest.mark.parametrize("block_name", EXPECTED_MONEY_MANAGEMENT)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)


class TestFiltersRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_FILTERS)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_FILTERS)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.FILTER

    @pytest.mark.parametrize("block_name", EXPECTED_FILTERS)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)


class TestAllPhase4Metadata:
    @pytest.mark.parametrize("block_name", ALL_PHASE4_BLOCKS)
    def test_has_description(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.description) > 20

    @pytest.mark.parametrize("block_name", ALL_PHASE4_BLOCKS)
    def test_has_tags(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.tags) >= 1

    @pytest.mark.parametrize("block_name", ALL_PHASE4_BLOCKS)
    def test_has_typical_use(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.typical_use) > 20
