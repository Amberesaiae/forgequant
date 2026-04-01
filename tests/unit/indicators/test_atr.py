"""Tests for the ATR indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.atr import ATRIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def atr() -> ATRIndicator:
    return ATRIndicator()


class TestATRMetadata:
    def test_name(self, atr: ATRIndicator) -> None:
        assert atr.metadata.name == "atr"

    def test_category(self, atr: ATRIndicator) -> None:
        assert atr.metadata.category == BlockCategory.INDICATOR

    def test_default_period(self, atr: ATRIndicator) -> None:
        assert atr.metadata.get_defaults()["period"] == 14

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "atr" in registry


class TestATRCompute:
    def test_default_params(
        self, atr: ATRIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr.execute(sample_ohlcv)
        assert "atr_14" in result.columns
        assert "atr_14_pct" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_atr_positive(
        self, atr: ATRIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr.execute(sample_ohlcv)
        atr_vals = result["atr_14"].dropna()
        assert (atr_vals > 0).all()

    def test_atr_pct_positive(
        self, atr: ATRIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr.execute(sample_ohlcv)
        pct_vals = result["atr_14_pct"].dropna()
        assert (pct_vals > 0).all()

    def test_high_volatility_high_atr(self, atr: ATRIndicator) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 10.0,
                "low": close - 10.0,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = atr.execute(df)
        assert result["atr_14"].iloc[-1] > 15.0

    def test_low_volatility_low_atr(self, atr: ATRIndicator) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.01,
                "low": close - 0.01,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = atr.execute(df)
        assert result["atr_14"].iloc[-1] < 0.1

    def test_gap_increases_atr(self, atr: ATRIndicator) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        high = close + 0.5
        low = close - 0.5
        close[50:] = 120.0
        high[50:] = 120.5
        low[50:] = 119.5
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = atr.execute(df)
        assert result["atr_14"].iloc[55] > result["atr_14"].iloc[49]

    def test_known_true_range(self, atr: ATRIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        close = np.full(20, 100.0)
        high = np.full(20, 105.0)
        low = np.full(20, 95.0)
        close[0] = 100.0
        high[1] = 108.0
        low[1] = 97.0
        close[1] = 106.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(20) * 1000,
            },
            index=dates,
        )
        result = atr.execute(df, {"period": 1})
        assert abs(result["atr_1"].iloc[1] - 11.0) < 1e-6

    def test_insufficient_data_raises(self, atr: ATRIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            atr.execute(df, {"period": 14})

    def test_custom_period(
        self, atr: ATRIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr.execute(sample_ohlcv, {"period": 7})
        assert "atr_7" in result.columns
        assert "atr_7_pct" in result.columns
