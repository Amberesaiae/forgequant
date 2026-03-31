"""
Verification test to confirm all 28 blocks are registered correctly.

Run with:
    uv run pytest tests/unit/test_all_blocks.py -v
"""

import pytest

from strategies_library.registry import BlockRegistry

# Import at module level to trigger @BlockRegistry.register decorators
import strategies_library.blocks  # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state for each test."""
    # Save current state
    saved_blocks = dict(BlockRegistry._blocks)
    saved_metadata = dict(BlockRegistry._metadata)
    yield
    # Restore
    BlockRegistry._blocks.clear()
    BlockRegistry._metadata.clear()
    BlockRegistry._blocks.update(saved_blocks)
    BlockRegistry._metadata.update(saved_metadata)


class TestAllBlocksRegistered:
    """Verify all 28 blocks are properly registered."""

    def test_total_block_count(self):
        assert BlockRegistry.count() == 28

    def test_indicator_count(self):
        indicators = BlockRegistry.list_by_category("indicator")
        assert len(indicators) == 8

    def test_price_action_count(self):
        pa = BlockRegistry.list_by_category("price_action")
        assert len(pa) == 4

    def test_entry_count(self):
        entries = BlockRegistry.list_by_category("entry")
        assert len(entries) == 4

    def test_exit_count(self):
        exits = BlockRegistry.list_by_category("exit")
        assert len(exits) == 4

    def test_money_management_count(self):
        mm = BlockRegistry.list_by_category("money_management")
        assert len(mm) == 4

    def test_filter_count(self):
        filters = BlockRegistry.list_by_category("filter")
        assert len(filters) == 4

    def test_all_expected_names_present(self):
        expected = [
            "EMA", "RSI", "ATR", "BollingerBands", "MACD", "ADX", "Stochastic", "Ichimoku",
            "Breakout", "Pullback", "HigherHighLowerLow", "SupportResistance",
            "Crossover", "ThresholdCross", "Confluence", "ReversalPattern",
            "FixedTPSL", "TrailingStop", "TimeBasedExit", "BreakevenStop",
            "FixedRisk", "VolatilityTargeting", "KellyFractional", "ATRBasedSizing",
            "TradingSessionFilter", "SpreadFilter", "MaxDrawdownFilter", "TrendFilter",
        ]
        all_names = BlockRegistry.get_all_names()
        for name in expected:
            assert name in all_names, f"Missing block: {name}"
