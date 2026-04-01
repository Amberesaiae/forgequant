"""
Composable strategy building blocks.

This package provides the base abstractions (BaseBlock, BlockMetadata),
the central BlockRegistry, and all concrete block implementations
organized by category.

Usage:
    from forgequant.blocks import BlockRegistry, BaseBlock, BlockMetadata
    from forgequant.blocks import BlockCategory
"""

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

__all__ = [
    "BaseBlock",
    "BlockMetadata",
    "BlockRegistry",
    "BlockCategory",
    "ParameterSpec",
]
