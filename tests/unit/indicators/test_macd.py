"""Tests for the MACD indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.macd import MACDIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def macd() -> MACDIndicator:
    return MACDIndicator()


class TestMACDMetadata:
    def test_name(self, macd: MACDIndicator) -> None:
        assert macd.metadata.name == "macd"

    def test_category(self, macd: MACDIndicator) -> None:
        assert macd.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, macd: MACDIndicator) -> None:
        defaults = macd.metadata.get_defaults()
        assert defaults["fast_period"] == 12
        assert defaults["slow_period"] == 26
        assert defaults["signal_period"] == 9

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "macd" in registry


class TestMACDCompute:
    def test_default_params(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = macd.execute(sample_ohlcv)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_histogram_equals_line_minus_signal(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = macd.execute(sample_ohlcv)
        computed_hist = result["macd_line"] - result["macd_signal"]
        pd.testing.assert_series_equal(
            result["macd_histogram"], computed_hist, check_names=False
        )

    def test_flat_price_macd_zero(self, macd: MACDIndicator) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = macd.execute(df)
        assert abs(result["macd_line"].iloc[-1]) < 1e-10

    def test_uptrend_positive_macd(self, macd: MACDIndicator) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = macd.execute(df)
        assert result["macd_line"].iloc[-1] > 0

    def test_fast_ge_slow_raises(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be less than"):
            macd.execute(sample_ohlcv, {"fast_period": 26, "slow_period": 12})

    def test_fast_eq_slow_raises(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be less than"):
            macd.execute(sample_ohlcv, {"fast_period": 20, "slow_period": 20})

    def test_insufficient_data_raises(self, macd: MACDIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=30, freq="h")
        close = np.random.randn(30) + 100
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(30) * 1000,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            macd.execute(df)

    def test_custom_periods(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = macd.execute(
            sample_ohlcv,
            {"fast_period": 8, "slow_period": 21, "signal_period": 5},
        )
        assert "macd_line" in result.columns
        assert len(result) == len(sample_ohlcv)
