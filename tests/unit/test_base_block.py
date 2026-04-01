"""Tests for forgequant.blocks.base.BaseBlock."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


class TestBaseBlockSubclassing:
    """Tests for BaseBlock's __init_subclass__ validation."""

    def test_concrete_without_metadata_raises(self) -> None:
        """A concrete subclass without metadata should raise TypeError."""
        with pytest.raises(TypeError, match="must define a 'metadata'"):

            class BadBlock(BaseBlock):
                def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
                    return data["close"]

            # Force the check by trying to use the class
            _ = BadBlock

    def test_concrete_with_metadata_succeeds(self, sample_block_class: Any) -> None:
        """A properly defined concrete subclass should work fine."""
        MyBlock = sample_block_class("valid_block")
        instance = MyBlock()
        assert instance.metadata.name == "valid_block"


class TestBaseBlockExecute:
    """Tests for the execute() public API."""

    def test_execute_with_defaults(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """Execute with no params should use defaults and return a result."""
        MyBlock = sample_block_class("exec_test")
        instance = MyBlock()
        result = instance.execute(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_execute_validates_ohlcv(self, sample_block_class: Any) -> None:
        """Execute should reject an empty DataFrame."""
        MyBlock = sample_block_class("empty_test")
        instance = MyBlock()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            instance.execute(empty_df)

    def test_execute_validates_params(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """Execute should reject unknown parameters."""
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
        )
        MyBlock = sample_block_class("param_test", parameters=params)
        instance = MyBlock()
        with pytest.raises(BlockValidationError):
            instance.execute(sample_ohlcv, {"unknown_param": 42})

    def test_execute_fills_defaults(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """Missing params should be filled with defaults."""
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
        )

        def compute_fn(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            # Return the period value as verification
            return pd.Series([params["period"]] * len(data), index=data.index)

        MyBlock = sample_block_class(
            "default_test",
            parameters=params,
            compute_fn=compute_fn,
        )
        instance = MyBlock()
        result = instance.execute(sample_ohlcv)
        assert result.iloc[0] == 14

    def test_execute_wraps_exception(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """Exceptions in compute() should be wrapped in BlockComputeError."""

        def bad_compute(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            raise ZeroDivisionError("boom")

        MyBlock = sample_block_class("error_test", compute_fn=bad_compute)
        instance = MyBlock()
        with pytest.raises(BlockComputeError, match="boom"):
            instance.execute(sample_ohlcv)

    def test_execute_preserves_block_compute_error(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """BlockComputeError raised in compute() should not be double-wrapped."""

        def deliberate_error(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            raise BlockComputeError("deliberate", "intentional failure")

        MyBlock = sample_block_class("preserve_test", compute_fn=deliberate_error)
        instance = MyBlock()
        with pytest.raises(BlockComputeError, match="intentional failure"):
            instance.execute(sample_ohlcv)

    def test_execute_normalizes_columns(self, sample_block_class: Any) -> None:
        """Uppercase column names in input should be lowered automatically."""
        import numpy as np

        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "Open": np.ones(10),
                "High": np.ones(10) * 1.1,
                "Low": np.ones(10) * 0.9,
                "Close": np.ones(10),
                "Volume": np.ones(10) * 100,
            },
            index=dates,
        )

        MyBlock = sample_block_class("case_test")
        instance = MyBlock()
        result = instance.execute(df)
        assert isinstance(result, pd.Series)


class TestBaseBlockRepr:
    """Tests for __repr__ and __str__."""

    def test_repr(self, sample_block_class: Any) -> None:
        MyBlock = sample_block_class("repr_test")
        instance = MyBlock()
        r = repr(instance)
        assert "repr_test" in r
        assert "indicator" in r

    def test_str(self, sample_block_class: Any) -> None:
        MyBlock = sample_block_class("str_test")
        instance = MyBlock()
        s = str(instance)
        assert "Str Test" in s
        assert "indicator" in s
