"""Tests for the Trailing Stop exit rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.exit_rules.trailing_stop import TrailingStopExit
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def trail() -> TrailingStopExit:
    return TrailingStopExit()


class TestTrailingStopMetadata:
    def test_name(self, trail: TrailingStopExit) -> None:
        assert trail.metadata.name == "trailing_stop"

    def test_category(self, trail: TrailingStopExit) -> None:
        assert trail.metadata.category == BlockCategory.EXIT_RULE

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "trailing_stop" in registry


class TestTrailingStopCompute:
    def test_output_columns(
        self, trail: TrailingStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trail.execute(sample_ohlcv)
        expected = {
            "trail_atr", "trail_long_stop", "trail_short_stop",
            "trail_long_exit", "trail_short_exit",
        }
        assert expected == set(result.columns)

    def test_long_stop_only_increases(
        self, trail: TrailingStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trail.execute(sample_ohlcv)
        stops = result["trail_long_stop"].dropna().values
        for i in range(1, len(stops)):
            assert stops[i] >= stops[i - 1] - 1e-10

    def test_short_stop_only_decreases(
        self, trail: TrailingStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trail.execute(sample_ohlcv)
        stops = result["trail_short_stop"].dropna().values
        for i in range(1, len(stops)):
            assert stops[i] <= stops[i - 1] + 1e-10

    def test_long_exit_fires_on_breakdown(self, trail: TrailingStopExit) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([np.linspace(100, 120, 60), np.linspace(119, 90, 40)])
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1,
                "low": close - 1, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = trail.execute(df, {"atr_period": 5, "trail_atr_mult": 2.0})
        assert result["trail_long_exit"].iloc[70:].sum() > 0

    def test_wider_mult_fewer_exits(
        self, trail: TrailingStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        tight = trail.execute(sample_ohlcv, {"trail_atr_mult": 1.0})
        wide = trail.execute(sample_ohlcv, {"trail_atr_mult": 4.0})
        assert tight["trail_long_exit"].sum() >= wide["trail_long_exit"].sum()

    def test_insufficient_data_raises(self, trail: TrailingStopExit) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5,
                "low": [99.0] * 5, "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            trail.execute(df)
