"""
ForgeQuant Base Block.

Every building block in the strategies library inherits from BaseBlock.
This ensures a consistent interface for the AI Forge, Genetic Engine,
VectorBT evaluator, and compiler to work with.

Key design decisions:
- compute() is the only required method.
- It accepts a pandas DataFrame and optional params dict.
- It returns a pandas Series (for signals/conditions), a dict (for complex
  indicators like Bollinger Bands), or a float/dict (for money management).
- validate_params() is optional but recommended for safety.
- BlockMetadata provides information for the registry, RAG, and UI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import pandas as pd


@dataclass
class BlockMetadata:
    """Metadata describing a building block.

    Attributes:
        name: Unique identifier for the block. Used in registry lookups.
        category: One of 'indicator', 'price_action', 'entry', 'exit',
                  'money_management', or 'filter'.
        description: Human-readable description of what this block does.
        complexity: Integer from 1 (simple) to 5 (advanced). Used by the
                    genetic engine to control strategy complexity.
        typical_use: List of strategy types this block is commonly used in.
                     Examples: ['trend_following', 'mean_reversion'].
        required_columns: DataFrame columns this block needs to compute.
                          Examples: ['close'], ['high', 'low', 'close'].
        version: Semantic version string. Increment when logic changes.
        tags: Additional searchable keywords for RAG retrieval.
    """

    name: str
    category: str
    description: str
    complexity: int
    typical_use: List[str]
    required_columns: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)


class BaseBlock(ABC):
    """Abstract base class for all strategy building blocks.

    Every block must:
    1. Define a `metadata` class attribute of type BlockMetadata.
    2. Implement the `compute()` method.

    Optionally:
    3. Override `validate_params()` for parameter range checking.
    """

    metadata: BlockMetadata

    @abstractmethod
    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Union[pd.Series, Dict[str, Any], float]:
        """Core computation method.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume.
                  Index should be DatetimeIndex.
            params: Optional dictionary of parameters. Each block defines
                    its own expected keys and defaults.

        Returns:
            - pd.Series: For indicators and boolean conditions.
            - dict: For complex indicators (e.g., Bollinger returns upper/middle/lower).
            - float or dict: For money management configuration.
        """
        ...

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate that params are within acceptable ranges.

        Override this in subclasses to enforce parameter constraints.
        The safety review and genetic engine use this to reject
        invalid parameter combinations.

        Args:
            params: The parameter dictionary to validate.

        Returns:
            True if all parameters are valid, False otherwise.
        """
        return True

    def get_metadata(self) -> BlockMetadata:
        """Return this block's metadata."""
        return self.metadata

    def __repr__(self) -> str:
        return f"<Block: {self.metadata.name} ({self.metadata.category})>"
