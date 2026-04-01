"""
Abstract base class for all strategy building blocks.

Every concrete block (indicator, price action pattern, entry rule, exit rule,
money management, filter) must subclass BaseBlock and implement:

    1. metadata (class attribute): a BlockMetadata instance
    2. compute(data, params) -> BlockResult

The base class provides:
    - Automatic OHLCV validation
    - Parameter validation against metadata specs
    - Structured logging on entry/exit of compute()
    - Consistent error wrapping via BlockComputeError
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from forgequant.blocks.metadata import BlockMetadata
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.logging import get_logger
from forgequant.core.types import BlockParams, BlockResult, validate_ohlcv

logger = get_logger(__name__)


class BaseBlock(ABC):
    """
    Abstract base class for all ForgeQuant strategy building blocks.

    Subclasses MUST define:
        metadata: ClassVar[BlockMetadata] — describes the block's identity and params.

    Subclasses MUST implement:
        compute(data, params) — the core computation logic.
    """

    metadata: ClassVar[BlockMetadata]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate that subclasses properly define metadata.

        Called automatically when a class inherits from BaseBlock.
        """
        super().__init_subclass__(**kwargs)

        # Skip validation for intermediate abstract classes
        if getattr(cls, "__abstractmethods__", None):
            return

        if not hasattr(cls, "metadata") or not isinstance(cls.metadata, BlockMetadata):
            raise TypeError(
                f"Block class '{cls.__name__}' must define a 'metadata' "
                f"class attribute of type BlockMetadata"
            )

    @abstractmethod
    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Execute the block's core computation.

        This is the method that subclasses MUST implement. It receives
        already-validated OHLCV data and already-validated parameters.

        Args:
            data: OHLCV DataFrame with lowercase column names and DatetimeIndex.
            params: Validated parameter dictionary matching this block's metadata.

        Returns:
            Computation result — typically a DataFrame with new columns,
            a Series, or a dict of Series/floats depending on block category.

        Raises:
            BlockComputeError: If the computation fails for any reason.
        """
        ...

    def execute(
        self,
        data: pd.DataFrame,
        params: BlockParams | None = None,
    ) -> BlockResult:
        """
        Validate inputs, run compute(), and return results with logging.

        This is the public API that external code should call. It wraps
        compute() with:
            1. OHLCV DataFrame validation
            2. Parameter validation (with defaults for missing params)
            3. Structured log entry with timing
            4. Error wrapping in BlockComputeError

        Args:
            data: OHLCV DataFrame. Columns are normalized to lowercase.
            params: Parameter dictionary. Missing params use defaults.
                    If None, all defaults are used.

        Returns:
            The result from compute().

        Raises:
            ValueError: If OHLCV data is invalid.
            BlockValidationError: If parameters fail validation.
            BlockComputeError: If compute() raises any exception.
        """
        block_name = self.metadata.name
        raw_params = params or {}

        # Step 1: Validate OHLCV data
        data = data.copy()
        if not data.empty:
            data.columns = data.columns.str.lower()
        validate_ohlcv(data, block_name=block_name)

        # Step 2: Validate and fill parameters
        try:
            validated_params = self.metadata.validate_params(raw_params)
        except ValueError as e:
            raise BlockValidationError(
                block_name=block_name,
                param_name="(multiple)",
                value=raw_params,
                constraint=str(e),
            ) from e

        # Step 3: Execute compute with logging
        logger.debug(
            "block_execute_start",
            block=block_name,
            category=str(self.metadata.category),
            params=validated_params,
            data_rows=len(data),
        )

        start_time = time.perf_counter()

        try:
            result = self.compute(data, validated_params)
        except BlockComputeError:
            # Re-raise without wrapping
            raise
        except Exception as e:
            raise BlockComputeError(
                block_name=block_name,
                reason=f"{type(e).__name__}: {e}",
            ) from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            "block_execute_complete",
            block=block_name,
            elapsed_ms=round(elapsed_ms, 2),
            result_type=type(result).__name__,
        )

        return result

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} "
            f"name={self.metadata.name!r} "
            f"category={self.metadata.category.value!r}>"
        )

    def __str__(self) -> str:
        return f"{self.metadata.display_name} ({self.metadata.category.value})"
