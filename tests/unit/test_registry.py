"""Tests for forgequant.blocks.registry.BlockRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockNotFoundError, BlockRegistrationError
from forgequant.core.types import BlockCategory


class TestBlockRegistrySingleton:
    """Tests for the singleton behavior."""

    def test_singleton(self) -> None:
        r1 = BlockRegistry()
        r2 = BlockRegistry()
        assert r1 is r2

    def test_len_empty(self, clean_registry: BlockRegistry) -> None:
        assert len(clean_registry) == 0

    def test_contains_empty(self, clean_registry: BlockRegistry) -> None:
        assert "nonexistent" not in clean_registry

    def test_repr(self, clean_registry: BlockRegistry) -> None:
        r = repr(clean_registry)
        assert "BlockRegistry" in r
        assert "blocks=0" in r


class TestBlockRegistration:
    """Tests for registering blocks."""

    def test_register_decorator(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("decorator_test")
        BlockRegistry.register(MyBlock)
        assert "decorator_test" in clean_registry
        assert clean_registry.count() == 1

    def test_register_class_method(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("method_test")
        clean_registry.register_class(MyBlock)
        assert "method_test" in clean_registry

    def test_register_duplicate_raises(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("dupe_test")
        clean_registry.register_class(MyBlock)
        MyBlock2 = sample_block_class("dupe_test")
        with pytest.raises(BlockRegistrationError, match="Already registered"):
            clean_registry.register_class(MyBlock2)

    def test_register_non_baseblock_raises(self, clean_registry: BlockRegistry) -> None:
        with pytest.raises(BlockRegistrationError, match="subclass of BaseBlock"):
            clean_registry.register_class(str)  # type: ignore[arg-type]

    def test_register_without_metadata_raises(self, clean_registry: BlockRegistry) -> None:
        with pytest.raises(TypeError, match="must define a 'metadata'"):

            class BareBlock(BaseBlock):
                pass  # No metadata, no compute


class TestBlockLookup:
    """Tests for get() and get_or_raise()."""

    def test_get_found(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("lookup_test")
        clean_registry.register_class(MyBlock)
        result = clean_registry.get("lookup_test")
        assert result is MyBlock

    def test_get_not_found(self, clean_registry: BlockRegistry) -> None:
        assert clean_registry.get("ghost") is None

    def test_get_or_raise_found(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("raise_test")
        clean_registry.register_class(MyBlock)
        result = clean_registry.get_or_raise("raise_test")
        assert result is MyBlock

    def test_get_or_raise_not_found(self, clean_registry: BlockRegistry) -> None:
        with pytest.raises(BlockNotFoundError, match="ghost"):
            clean_registry.get_or_raise("ghost")

    def test_instantiate(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("inst_test")
        clean_registry.register_class(MyBlock)
        instance = clean_registry.instantiate("inst_test")
        assert isinstance(instance, BaseBlock)
        assert instance.metadata.name == "inst_test"


class TestBlockFiltering:
    """Tests for list_by_category() and search()."""

    def test_list_by_category(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        Ind1 = sample_block_class("ind_alpha", BlockCategory.INDICATOR)
        Ind2 = sample_block_class("ind_beta", BlockCategory.INDICATOR)
        Flt1 = sample_block_class("flt_one", BlockCategory.FILTER)

        clean_registry.register_class(Ind1)
        clean_registry.register_class(Ind2)
        clean_registry.register_class(Flt1)

        indicators = clean_registry.list_by_category(BlockCategory.INDICATOR)
        assert len(indicators) == 2
        assert indicators[0].metadata.name == "ind_alpha"
        assert indicators[1].metadata.name == "ind_beta"

        filters = clean_registry.list_by_category(BlockCategory.FILTER)
        assert len(filters) == 1

        exits = clean_registry.list_by_category(BlockCategory.EXIT_RULE)
        assert len(exits) == 0

    def test_search_by_name(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("ema_indicator")
        clean_registry.register_class(MyBlock)
        results = clean_registry.search("ema")
        assert len(results) == 1
        assert results[0].metadata.name == "ema_indicator"

    def test_search_case_insensitive(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("bollinger_bands")
        clean_registry.register_class(MyBlock)
        results = clean_registry.search("BOLLINGER")
        assert len(results) == 1

    def test_search_empty_query(self, clean_registry: BlockRegistry) -> None:
        assert clean_registry.search("") == []
        assert clean_registry.search("   ") == []

    def test_search_no_match(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("rsi_indicator")
        clean_registry.register_class(MyBlock)
        results = clean_registry.search("ichimoku")
        assert len(results) == 0


class TestBlockIteration:
    """Tests for all_blocks(), all_names(), count()."""

    def test_all_blocks_sorted(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        B1 = sample_block_class("zebra")
        B2 = sample_block_class("alpha")
        clean_registry.register_class(B1)
        clean_registry.register_class(B2)

        names = [cls.metadata.name for cls in clean_registry.all_blocks()]
        assert names == ["alpha", "zebra"]

    def test_all_names(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        B1 = sample_block_class("bravo")
        B2 = sample_block_class("alpha")
        clean_registry.register_class(B1)
        clean_registry.register_class(B2)
        assert clean_registry.all_names() == ["alpha", "bravo"]

    def test_count(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        assert clean_registry.count() == 0
        B1 = sample_block_class("one")
        clean_registry.register_class(B1)
        assert clean_registry.count() == 1

    def test_count_by_category(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        I1 = sample_block_class("i1", BlockCategory.INDICATOR)
        I2 = sample_block_class("i2", BlockCategory.INDICATOR)
        F1 = sample_block_class("f1", BlockCategory.FILTER)
        clean_registry.register_class(I1)
        clean_registry.register_class(I2)
        clean_registry.register_class(F1)

        counts = clean_registry.count_by_category()
        assert counts[BlockCategory.INDICATOR] == 2
        assert counts[BlockCategory.FILTER] == 1
        assert BlockCategory.EXIT_RULE not in counts


class TestBlockRegistryMaintenance:
    """Tests for clear() and unregister()."""

    def test_clear(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        B1 = sample_block_class("clear_me")
        clean_registry.register_class(B1)
        assert clean_registry.count() == 1
        clean_registry.clear()
        assert clean_registry.count() == 0

    def test_unregister_existing(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        B1 = sample_block_class("removable")
        clean_registry.register_class(B1)
        result = clean_registry.unregister("removable")
        assert result is True
        assert "removable" not in clean_registry

    def test_unregister_nonexistent(self, clean_registry: BlockRegistry) -> None:
        result = clean_registry.unregister("ghost")
        assert result is False
