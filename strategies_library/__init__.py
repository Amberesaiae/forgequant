"""
ForgeQuant Strategies Library.
High-quality modular building blocks for systematic strategy generation.
"""

from .base import BaseBlock, BlockMetadata
from .registry import BlockRegistry

__all__ = [
    "BlockRegistry",
    "BaseBlock",
    "BlockMetadata",
]
