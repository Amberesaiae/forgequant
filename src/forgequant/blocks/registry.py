"""
Central registry for strategy building blocks.

The BlockRegistry is a singleton that stores all available block classes,
keyed by their metadata.name. It provides:

    - register(): Class decorator to auto-register blocks
    - get() / get_or_raise(): Lookup by name
    - list_by_category(): Filter by BlockCategory
    - search(): Full-text search across name, description, tags, typical_use
    - all_blocks(): Iterator over all registered blocks
    - count(): Number of registered blocks
    - clear(): Remove all registrations (for testing)
"""

from __future__ import annotations

from typing import Iterator

from forgequant.blocks.base import BaseBlock
from forgequant.core.exceptions import BlockNotFoundError, BlockRegistrationError
from forgequant.core.logging import get_logger
from forgequant.core.types import BlockCategory

logger = get_logger(__name__)


class _BlockRegistryMeta(type):
    """
    Metaclass that ensures BlockRegistry is a singleton.

    No matter how many times BlockRegistry() is called, the same instance
    is returned.
    """

    _instance: BlockRegistry | None = None

    def __call__(cls, *args: object, **kwargs: object) -> BlockRegistry:
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class BlockRegistry(metaclass=_BlockRegistryMeta):
    """
    Singleton registry of all available strategy building blocks.

    Blocks register themselves via the @BlockRegistry.register decorator
    or by calling BlockRegistry().register_class(cls) explicitly.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, type[BaseBlock]] = {}

    # ── Registration ─────────────────────────────────────────────────────

    @staticmethod
    def register(cls: type[BaseBlock]) -> type[BaseBlock]:
        """
        Class decorator that registers a block in the global registry.

        Usage:
            @BlockRegistry.register
            class EMAIndicator(BaseBlock):
                metadata = BlockMetadata(...)
                def compute(self, data, params): ...

        Args:
            cls: The block class to register. Must be a concrete subclass
                 of BaseBlock with a valid metadata attribute.

        Returns:
            The class unchanged (so it can be used as a decorator).

        Raises:
            BlockRegistrationError: If the class is invalid or a block
                                     with the same name already exists.
        """
        registry = BlockRegistry()
        registry.register_class(cls)
        return cls

    def register_class(self, cls: type[BaseBlock]) -> None:
        """
        Register a block class explicitly (non-decorator form).

        Args:
            cls: The block class to register.

        Raises:
            BlockRegistrationError: If validation fails or name collides.
        """
        # Validate it's a proper BaseBlock subclass
        if not isinstance(cls, type) or not issubclass(cls, BaseBlock):
            raise BlockRegistrationError(
                block_name=getattr(cls, "__name__", str(cls)),
                reason="Must be a subclass of BaseBlock",
            )

        if not hasattr(cls, "metadata"):
            raise BlockRegistrationError(
                block_name=cls.__name__,
                reason="Missing 'metadata' class attribute",
            )

        name = cls.metadata.name

        # Check for duplicates
        if name in self._blocks:
            existing = self._blocks[name]
            raise BlockRegistrationError(
                block_name=name,
                reason=(
                    f"Already registered by {existing.__name__}. "
                    f"Cannot register {cls.__name__} with the same name."
                ),
            )

        self._blocks[name] = cls

        logger.info(
            "block_registered",
            block=name,
            category=str(cls.metadata.category),
            class_name=cls.__name__,
        )

    # ── Lookup ───────────────────────────────────────────────────────────

    def get(self, name: str) -> type[BaseBlock] | None:
        """
        Look up a block class by name.

        Args:
            name: The block's metadata.name.

        Returns:
            The block class, or None if not found.
        """
        return self._blocks.get(name)

    def get_or_raise(self, name: str) -> type[BaseBlock]:
        """
        Look up a block class by name, raising if not found.

        Args:
            name: The block's metadata.name.

        Returns:
            The block class.

        Raises:
            BlockNotFoundError: If the name is not in the registry.
        """
        cls = self._blocks.get(name)
        if cls is None:
            raise BlockNotFoundError(block_name=name)
        return cls

    def instantiate(self, name: str) -> BaseBlock:
        """
        Look up and instantiate a block by name.

        Args:
            name: The block's metadata.name.

        Returns:
            A new instance of the block.

        Raises:
            BlockNotFoundError: If the name is not in the registry.
        """
        cls = self.get_or_raise(name)
        return cls()

    # ── Filtering & Search ───────────────────────────────────────────────

    def list_by_category(self, category: BlockCategory) -> list[type[BaseBlock]]:
        """
        Return all registered block classes in a given category.

        Args:
            category: The category to filter by.

        Returns:
            List of block classes, sorted by name.
        """
        return sorted(
            (cls for cls in self._blocks.values() if cls.metadata.category == category),
            key=lambda cls: cls.metadata.name,
        )

    def search(self, query: str) -> list[type[BaseBlock]]:
        """
        Search for blocks matching a query string.

        The query is matched (case-insensitive substring) against:
            - metadata.name
            - metadata.display_name
            - metadata.description
            - metadata.tags (each tag)
            - metadata.typical_use

        Args:
            query: The search string.

        Returns:
            List of matching block classes, sorted by name.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        results: list[type[BaseBlock]] = []

        for cls in self._blocks.values():
            meta = cls.metadata
            searchable = " ".join(
                [
                    meta.name,
                    meta.display_name,
                    meta.description,
                    " ".join(meta.tags),
                    meta.typical_use,
                ]
            ).lower()

            if query_lower in searchable:
                results.append(cls)

        return sorted(results, key=lambda c: c.metadata.name)

    # ── Iteration & Info ─────────────────────────────────────────────────

    def all_blocks(self) -> Iterator[type[BaseBlock]]:
        """Iterate over all registered block classes, sorted by name."""
        yield from sorted(self._blocks.values(), key=lambda cls: cls.metadata.name)

    def all_names(self) -> list[str]:
        """Return sorted list of all registered block names."""
        return sorted(self._blocks.keys())

    def count(self) -> int:
        """Return the number of registered blocks."""
        return len(self._blocks)

    def count_by_category(self) -> dict[BlockCategory, int]:
        """Return a count of blocks per category."""
        counts: dict[BlockCategory, int] = {}
        for cls in self._blocks.values():
            cat = cls.metadata.category
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    # ── Maintenance ──────────────────────────────────────────────────────

    def clear(self) -> None:
        """
        Remove all registered blocks.

        WARNING: This is intended only for testing. Do not call in production.
        """
        self._blocks.clear()
        logger.warning("block_registry_cleared")

    def unregister(self, name: str) -> bool:
        """
        Remove a single block from the registry.

        Args:
            name: The block's metadata.name.

        Returns:
            True if the block was found and removed, False if it wasn't registered.
        """
        if name in self._blocks:
            del self._blocks[name]
            logger.info("block_unregistered", block=name)
            return True
        return False

    # ── Dunder ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, name: str) -> bool:
        return name in self._blocks

    def __repr__(self) -> str:
        return f"<BlockRegistry blocks={self.count()}>"
