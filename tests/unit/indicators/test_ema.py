"""Tests for the EMA indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.ema import EMAIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def ema() -> EMAIndicator:
    return EMAIndicator()


class TestEMAMetadata:
    def test_name(self, ema: EMAIndicator) -> None:
        assert ema.metadata.name == "ema"

    def test_category(self, ema: EMAIndicator) -> None:
        assert ema.metadata.category == BlockCategory.INDICATOR

    def test_display_name(self, ema: EMAIndicator) -> None:
        assert ema.metadata.display_name == "Exponential Moving Average"

    def test_default_parameters(self, ema: EMAIndicator) -> None:
        defaults = ema.metadata.get_defaults()
        assert defaults["period"] == 20
        assert defaults["source"] == "close"

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "ema" in registry


class TestEMACompute:
    def test_default_params(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ema.execute(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert "ema_20" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_custom_period(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ema.execute(sample_ohlcv, {"period": 50})
        assert "ema_50" in result.columns

    def test_source_high(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ema.execute(sample_ohlcv, {"period": 10, "source": "high"})
        assert "ema_10" in result.columns

    def test_no_nans_with_adjust_false(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ema.execute(sample_ohlcv, {"period": 20})
        assert result["ema_20"].isna().sum() == 0

    def test_ema_responds_to_price_changes(
        self, ema: EMAIndicator
    ) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([np.full(50, 100.0), np.full(50, 200.0)])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = ema.execute(df, {"period": 10})
        ema_vals = result["ema_10"]
        assert abs(ema_vals.iloc[49] - 100.0) < 1.0
        assert ema_vals.iloc[99] > 190.0

    def test_ema_with_known_values(self, ema: EMAIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        close = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [100.0] * 5,
            },
            index=dates,
        )
        result = ema.execute(df, {"period": 3})
        ema_vals = result["ema_3"]
        assert abs(ema_vals.iloc[0] - 10.0) < 1e-10
        assert abs(ema_vals.iloc[1] - 10.5) < 1e-10
        assert abs(ema_vals.iloc[2] - 11.25) < 1e-10
        assert abs(ema_vals.iloc[3] - 11.125) < 1e-10
        assert abs(ema_vals.iloc[4] - 10.5625) < 1e-10

    def test_insufficient_data_raises(self, ema: EMAIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [1.0] * 5,
                "high": [1.1] * 5,
                "low": [0.9] * 5,
                "close": [1.0] * 5,
                "volume": [100.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            ema.execute(df, {"period": 10})

    def test_invalid_source_raises(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            ema.execute(sample_ohlcv, {"source": "vwap"})

    def test_period_below_min_raises(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            ema.execute(sample_ohlcv, {"period": 1})

    def test_period_above_max_raises(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            ema.execute(sample_ohlcv, {"period": 501})
