"""
Shared pytest fixtures for the ForgeQuant test suite.

Provides:
    - sample_ohlcv: A realistic synthetic OHLCV DataFrame
    - clean_registry: Auto-clears the BlockRegistry before/after each test
    - sample_block_class: A minimal concrete BaseBlock for testing
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    Generate a realistic synthetic OHLCV DataFrame with 500 bars.

    Returns a DataFrame with:
        - DatetimeIndex at 1-hour intervals
        - Columns: open, high, low, close, volume
        - Prices in a random-walk pattern starting at 1.1000
        - Realistic high/low spreads
        - Volume as random integers
    """
    np.random.seed(42)
    n_bars = 500

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")

    # Generate a random walk for close prices
    returns = np.random.normal(loc=0.0, scale=0.001, size=n_bars)
    close = 1.1000 * np.exp(np.cumsum(returns))

    # Generate realistic OHLC from close
    spread = np.random.uniform(0.0005, 0.002, size=n_bars)
    high = close + spread * np.random.uniform(0.3, 1.0, size=n_bars)
    low = close - spread * np.random.uniform(0.3, 1.0, size=n_bars)

    # Open is previous close with some gap
    open_prices = np.roll(close, 1) + np.random.normal(0, 0.0002, size=n_bars)
    open_prices[0] = close[0]

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))

    volume = np.random.randint(100, 10000, size=n_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return df


@pytest.fixture
def clean_registry():
    """
    Clear the BlockRegistry before each test and restore after.

    Use this fixture explicitly in tests that manipulate the registry.
    """
    registry = BlockRegistry()
    saved_blocks = dict(registry._blocks)
    registry.clear()
    yield registry
    registry._blocks.clear()
    registry._blocks.update(saved_blocks)


@pytest.fixture
def sample_block_class():
    """
    Return a factory that creates minimal concrete BaseBlock classes.

    Usage:
        def test_something(sample_block_class):
            MyBlock = sample_block_class("my_block", BlockCategory.INDICATOR)
            instance = MyBlock()
            result = instance.execute(ohlcv_data)
    """

    def _factory(
        name: str = "test_block",
        category: BlockCategory = BlockCategory.INDICATOR,
        parameters: tuple[ParameterSpec, ...] = (),
        compute_fn: Any = None,
    ) -> type[BaseBlock]:
        """
        Create a concrete BaseBlock subclass with the given configuration.

        Args:
            name: Block name.
            category: Block category.
            parameters: Parameter specifications.
            compute_fn: Optional custom compute function.
                        Signature: (self, data, params) -> BlockResult
                        Defaults to returning data["close"].rename(name).
        """
        meta = BlockMetadata(
            name=name,
            display_name=name.replace("_", " ").title(),
            category=category,
            description=f"Test block: {name}",
            parameters=parameters,
            tags=("test",),
        )

        def default_compute(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            return data["close"].rename(name)

        fn = compute_fn if compute_fn is not None else default_compute

        # Dynamically create the class
        block_cls = type(
            f"TestBlock_{name}",
            (BaseBlock,),
            {
                "metadata": meta,
                "compute": fn,
            },
        )

        return block_cls

    return _factory
