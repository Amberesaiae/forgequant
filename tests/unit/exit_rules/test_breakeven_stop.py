"""Tests for the Breakeven Stop exit rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.exit_rules.breakeven_stop import BreakevenStopExit
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def be_stop() -> BreakevenStopExit:
    return BreakevenStopExit()


class TestBreakevenStopMetadata:
    def test_name(self, be_stop: BreakevenStopExit) -> None:
        assert be_stop.metadata.name == "breakeven_stop"

    def test_category(self, be_stop: BreakevenStopExit) -> None:
        assert be_stop.metadata.category == BlockCategory.EXIT_RULE

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "breakeven_stop" in registry


class TestBreakevenStopCompute:
    def test_output_columns(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = be_stop.execute(sample_ohlcv)
        expected = {
            "be_atr", "be_long_activation", "be_long_stop",
            "be_short_activation", "be_short_stop",
            "be_long_activated", "be_short_activated",
        }
        assert expected == set(result.columns)

    def test_long_activation_above_close(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = be_stop.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["be_long_activation"].dropna()
        assert (valid > close.loc[valid.index]).all()

    def test_short_activation_below_close(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = be_stop.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["be_short_activation"].dropna()
        assert (valid < close.loc[valid.index]).all()

    def test_long_stop_above_close_with_offset(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = be_stop.execute(sample_ohlcv, {"offset_atr_mult": 0.1})
        close = sample_ohlcv["close"]
        valid = result["be_long_stop"].dropna()
        assert (valid >= close.loc[valid.index]).all()

    def test_zero_offset_exact_breakeven(
        self, be_stop: BreakevenStopExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = be_stop.execute(sample_ohlcv, {"offset_atr_mult": 0.0})
        close = sample_ohlcv["close"]
        pd.testing.assert_series_equal(
            result["be_long_stop"], close, check_names=False, atol=1e-10
        )

    def test_insufficient_data_raises(self, be_stop: BreakevenStopExit) -> None:
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
            be_stop.execute(df)
