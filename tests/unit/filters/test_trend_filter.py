"""Tests for the Trend Filter block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.filters.trend_filter import TrendFilter
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def trend() -> TrendFilter:
    return TrendFilter()


class TestTrendFilterMetadata:
    def test_name(self, trend: TrendFilter) -> None:
        assert trend.metadata.name == "trend_filter"

    def test_category(self, trend: TrendFilter) -> None:
        assert trend.metadata.category == BlockCategory.FILTER

    def test_defaults(self, trend: TrendFilter) -> None:
        d = trend.metadata.get_defaults()
        assert d["period"] == 200
        assert d["ma_type"] == "ema"
        assert d["buffer_pct"] == 0.5

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "trend_filter" in registry


class TestTrendFilterCompute:
    def test_output_columns(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trend.execute(sample_ohlcv)
        expected = {
            "trend_ma", "trend_upper_buffer", "trend_lower_buffer",
            "trend_allow_long", "trend_allow_short", "trend_direction",
        }
        assert expected == set(result.columns)

    def test_buffer_surrounds_ma(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trend.execute(sample_ohlcv)
        valid = result.dropna(subset=["trend_ma"])
        assert (valid["trend_upper_buffer"] >= valid["trend_ma"]).all()
        assert (valid["trend_lower_buffer"] <= valid["trend_ma"]).all()

    def test_long_short_mutually_exclusive(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trend.execute(sample_ohlcv)
        both = result["trend_allow_long"] & result["trend_allow_short"]
        assert both.sum() == 0

    def test_uptrend_allows_long(self, trend: TrendFilter) -> None:
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = trend.execute(df, {"period": 50, "buffer_pct": 0.5})
        assert result["trend_allow_long"].iloc[-50:].all()
        assert not result["trend_allow_short"].iloc[-50:].any()

    def test_downtrend_allows_short(self, trend: TrendFilter) -> None:
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(200, 100, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = trend.execute(df, {"period": 50, "buffer_pct": 0.5})
        assert result["trend_allow_short"].iloc[-50:].all()
        assert not result["trend_allow_long"].iloc[-50:].any()

    def test_zero_buffer(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trend.execute(sample_ohlcv, {"buffer_pct": 0.0})
        valid = result.dropna(subset=["trend_ma"])
        pd.testing.assert_series_equal(
            valid["trend_upper_buffer"], valid["trend_ma"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            valid["trend_lower_buffer"], valid["trend_ma"],
            check_names=False,
        )

    def test_wider_buffer_fewer_signals(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        tight = trend.execute(sample_ohlcv, {"buffer_pct": 0.1, "period": 50})
        wide = trend.execute(sample_ohlcv, {"buffer_pct": 3.0, "period": 50})
        tight_active = tight["trend_allow_long"].sum() + tight["trend_allow_short"].sum()
        wide_active = wide["trend_allow_long"].sum() + wide["trend_allow_short"].sum()
        assert tight_active >= wide_active

    def test_direction_labels(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = trend.execute(sample_ohlcv)
        valid_values = {"bullish", "bearish", "neutral"}
        assert set(result["trend_direction"].unique()) <= valid_values

    def test_ema_vs_sma(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result_ema = trend.execute(sample_ohlcv, {"ma_type": "ema", "period": 50})
        result_sma = trend.execute(sample_ohlcv, {"ma_type": "sma", "period": 50})
        assert not result_ema["trend_ma"].equals(result_sma["trend_ma"])

    def test_insufficient_data_raises(self, trend: TrendFilter) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 50, "high": [101.0] * 50,
                "low": [99.0] * 50, "close": [100.0] * 50,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            trend.execute(df, {"period": 200})

    def test_invalid_ma_type_raises(
        self, trend: TrendFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            trend.execute(sample_ohlcv, {"ma_type": "wma"})
