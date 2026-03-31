"""
ForgeQuant Block Registry.

Central registry for all building blocks. Provides:
- Registration via decorator (@BlockRegistry.register)
- Lookup by name
- Listing by category
- Search by keyword (used by RAG and AI Forge)
- Full catalog export (used by genetic engine)

Usage:
    from strategies_library.registry import BlockRegistry

    # Register a block (typically done via decorator)
    @BlockRegistry.register
    class MyBlock(BaseBlock):
        ...

    # Look up a block
    block_class = BlockRegistry.get("MyBlock")
    instance = block_class()
    result = instance.compute(data, params)

    # Search for blocks
    results = BlockRegistry.search("momentum")

    # List all indicators
    indicators = BlockRegistry.list_by_category("indicator")
"""

from typing import Dict, List, Type

from strategies_library.base import BaseBlock, BlockMetadata
from core.logging import get_logger

logger = get_logger(__name__)


class BlockRegistry:
    """Central registry for all strategy building blocks."""

    _blocks: Dict[str, Type[BaseBlock]] = {}
    _metadata: Dict[str, BlockMetadata] = {}

    @classmethod
    def register(cls, block_class: Type[BaseBlock]) -> Type[BaseBlock]:
        """Register a block class in the registry.

        Intended to be used as a class decorator:
            @BlockRegistry.register
            class EMA(BaseBlock):
                ...

        Args:
            block_class: The block class to register. Must inherit from BaseBlock
                         and have a `metadata` class attribute.

        Returns:
            The same block_class (unchanged), allowing use as a decorator.

        Raises:
            TypeError: If block_class does not have a metadata attribute.
        """
        instance = block_class()

        if not hasattr(instance, "metadata"):
            raise TypeError(
                f"Block class {block_class.__name__} must have a 'metadata' attribute "
                f"of type BlockMetadata."
            )

        name = instance.metadata.name

        if name in cls._blocks:
            logger.warning(
                "Block already registered, overwriting",
                block_name=name,
                old_class=cls._blocks[name].__name__,
                new_class=block_class.__name__,
            )

        cls._blocks[name] = block_class
        cls._metadata[name] = instance.metadata

        logger.debug("Block registered", block_name=name, category=instance.metadata.category)

        return block_class

    @classmethod
    def get(cls, name: str) -> Type[BaseBlock] | None:
        """Retrieve a block class by name.

        Args:
            name: The unique name of the block (as defined in its metadata).

        Returns:
            The block class if found, None otherwise.
        """
        return cls._blocks.get(name)

    @classmethod
    def get_or_raise(cls, name: str) -> Type[BaseBlock]:
        """Retrieve a block class by name, raising an error if not found.

        Args:
            name: The unique name of the block.

        Returns:
            The block class.

        Raises:
            KeyError: If no block with that name is registered.
        """
        block = cls._blocks.get(name)
        if block is None:
            available = list(cls._blocks.keys())
            raise KeyError(
                f"Block '{name}' not found in registry. "
                f"Available blocks: {available}"
            )
        return block

    @classmethod
    def list_by_category(cls, category: str) -> List[BlockMetadata]:
        """List all blocks in a given category.

        Args:
            category: One of 'indicator', 'price_action', 'entry', 'exit',
                      'money_management', 'filter'.

        Returns:
            List of BlockMetadata objects for blocks in that category.
        """
        return [
            metadata
            for metadata in cls._metadata.values()
            if metadata.category == category
        ]

    @classmethod
    def search(cls, query: str) -> List[BlockMetadata]:
        """Search blocks by keyword across name, description, and tags.

        Args:
            query: Search string (case-insensitive).

        Returns:
            List of matching BlockMetadata objects.
        """
        q = query.lower()
        results = []
        for metadata in cls._metadata.values():
            if (
                q in metadata.name.lower()
                or q in metadata.description.lower()
                or any(q in tag.lower() for tag in metadata.tags)
                or any(q in use.lower() for use in metadata.typical_use)
            ):
                results.append(metadata)
        return results

    @classmethod
    def get_all(cls) -> Dict[str, BlockMetadata]:
        """Return the complete metadata catalog.

        Returns:
            Dictionary mapping block names to their BlockMetadata.
        """
        return dict(cls._metadata)

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Return a list of all registered block names.

        Returns:
            List of block name strings.
        """
        return list(cls._blocks.keys())

    @classmethod
    def count(cls) -> int:
        """Return the total number of registered blocks.

        Returns:
            Integer count.
        """
        return len(cls._blocks)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered blocks. Used primarily in testing.

        WARNING: This removes all blocks from the registry.
        """
        cls._blocks.clear()
        cls._metadata.clear()
        logger.warning("Block registry cleared")
