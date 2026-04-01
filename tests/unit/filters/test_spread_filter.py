"""Tests for the Spread Filter block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.filters.spread_filter import SpreadFilter
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory


@pytest.fixture
def spread() -> SpreadFilter:
    return SpreadFilter()


class TestSpreadFilterMetadata:
    def test_name(self, spread: SpreadFilter) -> None:
        assert spread.metadata.name == "spread_filter"

    def test_category(self, spread: SpreadFilter) -> None:
        assert spread.metadata.category == BlockCategory.FILTER

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "spread_filter" in registry


class TestSpreadFilterCompute:
    def test_output_columns(
        self, spread: SpreadFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = spread.execute(sample_ohlcv)
        expected = {"spread_value", "spread_avg", "spread_ok"}
        assert expected == set(result.columns)

    def test_tight_spread_all_ok(self, spread: SpreadFilter) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 1.1000)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.0001,
                "low": close - 0.0001, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = spread.execute(df, {"max_spread": 50.0, "max_spread_ratio": 3.0})
        assert result["spread_ok"].all()

    def test_wide_spread_filtered(self, spread: SpreadFilter) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 1.1000)
        high = close + 0.0001
        low = close - 0.0001
        high[50] = close[50] + 0.05
        low[50] = close[50] - 0.05
        df = pd.DataFrame(
            {
                "open": close, "high": high,
                "low": low, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = spread.execute(df, {"max_spread": 50.0, "max_spread_ratio": 3.0})
        assert not result["spread_ok"].iloc[50]

    def test_explicit_spread_column(self, spread: SpreadFilter) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        spread_vals = np.full(n, 2.0)
        spread_vals[50] = 200.0
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1,
                "low": close - 1, "close": close,
                "volume": np.ones(n) * 1000,
                "spread": spread_vals,
            },
            index=dates,
        )
        result = spread.execute(df, {"max_spread": 50.0, "max_spread_ratio": 3.0})
        assert result["spread_value"].iloc[50] == 200.0
        assert not result["spread_ok"].iloc[50]

    def test_avg_spread_increases_with_window(
        self, spread: SpreadFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = spread.execute(sample_ohlcv)
        avg = result["spread_avg"]
        assert avg.iloc[-1] > 0
