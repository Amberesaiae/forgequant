"""Tests for the ADX indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.adx import ADXIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def adx() -> ADXIndicator:
    return ADXIndicator()


class TestADXMetadata:
    def test_name(self, adx: ADXIndicator) -> None:
        assert adx.metadata.name == "adx"

    def test_category(self, adx: ADXIndicator) -> None:
        assert adx.metadata.category == BlockCategory.INDICATOR

    def test_default_period(self, adx: ADXIndicator) -> None:
        assert adx.metadata.get_defaults()["period"] == 14

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "adx" in registry


class TestADXCompute:
    def test_default_params(
        self, adx: ADXIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = adx.execute(sample_ohlcv)
        assert "adx" in result.columns
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_adx_non_negative(
        self, adx: ADXIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = adx.execute(sample_ohlcv)
        adx_vals = result["adx"].dropna()
        assert (adx_vals >= 0).all()

    def test_di_non_negative(
        self, adx: ADXIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = adx.execute(sample_ohlcv)
        assert (result["plus_di"].dropna() >= 0).all()
        assert (result["minus_di"].dropna() >= 0).all()

    def test_strong_trend_high_adx(self, adx: ADXIndicator) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        trend = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": trend,
                "high": trend + 1.0,
                "low": trend - 0.1,
                "close": trend + 0.5,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = adx.execute(df)
        assert result["adx"].iloc[-1] > 20

    def test_uptrend_plus_di_gt_minus_di(self, adx: ADXIndicator) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        trend = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": trend,
                "high": trend + 1.0,
                "low": trend - 0.1,
                "close": trend + 0.5,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = adx.execute(df)
        assert result["plus_di"].iloc[-1] > result["minus_di"].iloc[-1]

    def test_insufficient_data_raises(self, adx: ADXIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        close = np.ones(20) * 100
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(20) * 1000,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            adx.execute(df)

    def test_custom_period(
        self, adx: ADXIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = adx.execute(sample_ohlcv, {"period": 7})
        assert "adx" in result.columns
