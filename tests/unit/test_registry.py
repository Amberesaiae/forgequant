"""
Unit tests for the Block Registry.

Run with:
    uv run pytest tests/unit/test_registry.py -v
"""

import pandas as pd
import pytest

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


class DummyBlock(BaseBlock):
    """A simple test block for registry verification."""

    metadata = BlockMetadata(
        name="DummyTest",
        category="indicator",
        description="A dummy block for testing purposes",
        complexity=1,
        typical_use=["testing"],
        required_columns=["close"],
        tags=["test", "dummy"],
    )

    def compute(
        self, data: pd.DataFrame, params: dict | None = None
    ) -> pd.Series:
        params = params or {"period": 10}
        period = int(params.get("period", 10))
        return data["close"].rolling(window=period).mean()


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure a clean registry for each test."""
    BlockRegistry.clear()
    yield
    BlockRegistry.clear()


class TestBlockRegistry:
    """Tests for BlockRegistry functionality."""

    def test_register_block(self):
        """Block should be registered and retrievable."""
        BlockRegistry.register(DummyBlock)
        assert BlockRegistry.count() == 1
        assert BlockRegistry.get("DummyTest") is DummyBlock

    def test_get_nonexistent_block_returns_none(self):
        """Getting a nonexistent block should return None."""
        result = BlockRegistry.get("NonExistent")
        assert result is None

    def test_get_or_raise_nonexistent_block(self):
        """Getting a nonexistent block with get_or_raise should raise KeyError."""
        with pytest.raises(KeyError):
            BlockRegistry.get_or_raise("NonExistent")

    def test_list_by_category(self):
        """Listing by category should return correct blocks."""
        BlockRegistry.register(DummyBlock)
        indicators = BlockRegistry.list_by_category("indicator")
        assert len(indicators) == 1
        assert indicators[0].name == "DummyTest"

        # Different category should be empty
        entries = BlockRegistry.list_by_category("entry")
        assert len(entries) == 0

    def test_search_by_name(self):
        """Search should find blocks by name."""
        BlockRegistry.register(DummyBlock)
        results = BlockRegistry.search("Dummy")
        assert len(results) == 1

    def test_search_by_tag(self):
        """Search should find blocks by tag."""
        BlockRegistry.register(DummyBlock)
        results = BlockRegistry.search("test")
        assert len(results) == 1

    def test_search_no_results(self):
        """Search should return empty list when nothing matches."""
        BlockRegistry.register(DummyBlock)
        results = BlockRegistry.search("nonexistent_query")
        assert len(results) == 0

    def test_get_all(self):
        """get_all should return complete metadata catalog."""
        BlockRegistry.register(DummyBlock)
        catalog = BlockRegistry.get_all()
        assert "DummyTest" in catalog
        assert catalog["DummyTest"].category == "indicator"

    def test_get_all_names(self):
        """get_all_names should return list of registered block names."""
        BlockRegistry.register(DummyBlock)
        names = BlockRegistry.get_all_names()
        assert names == ["DummyTest"]

    def test_clear_registry(self):
        """clear should remove all blocks."""
        BlockRegistry.register(DummyBlock)
        assert BlockRegistry.count() == 1
        BlockRegistry.clear()
        assert BlockRegistry.count() == 0

    def test_block_compute(self):
        """Registered block should compute correctly."""
        BlockRegistry.register(DummyBlock)
        block_class = BlockRegistry.get("DummyTest")
        assert block_class is not None

        block = block_class()
        data = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
        )
        result = block.compute(data, {"period": 3})

        assert isinstance(result, pd.Series)
        assert len(result) == 10
        # First two values should be NaN (not enough data for period=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Third value should be mean of [1, 2, 3] = 2.0
        assert result.iloc[2] == pytest.approx(2.0)

    def test_block_metadata_access(self):
        """Block metadata should be accessible via get_metadata()."""
        block = DummyBlock()
        meta = block.get_metadata()
        assert meta.name == "DummyTest"
        assert meta.category == "indicator"
        assert meta.complexity == 1
        assert "testing" in meta.typical_use

    def test_block_repr(self):
        """Block __repr__ should be readable."""
        block = DummyBlock()
        assert repr(block) == "<Block: DummyTest (indicator)>"
