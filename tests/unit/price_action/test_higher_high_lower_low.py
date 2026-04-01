"""Tests for the Higher High / Lower Low price action block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.price_action.higher_high_lower_low import HigherHighLowerLowBlock
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def hhll() -> HigherHighLowerLowBlock:
    return HigherHighLowerLowBlock()


class TestHHLLMetadata:
    def test_name(self, hhll: HigherHighLowerLowBlock) -> None:
        assert hhll.metadata.name == "higher_high_lower_low"

    def test_category(self, hhll: HigherHighLowerLowBlock) -> None:
        assert hhll.metadata.category == BlockCategory.PRICE_ACTION

    def test_defaults(self, hhll: HigherHighLowerLowBlock) -> None:
        d = hhll.metadata.get_defaults()
        assert d["left_bars"] == 5
        assert d["right_bars"] == 5

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "higher_high_lower_low" in registry


class TestHHLLCompute:
    def test_output_columns(
        self, hhll: HigherHighLowerLowBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = hhll.execute(sample_ohlcv)
        expected = {
            "hhll_swing_high", "hhll_swing_low",
            "hhll_is_hh", "hhll_is_lh",
            "hhll_is_hl", "hhll_is_ll",
            "hhll_trend",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_uptrend_detection(self, hhll: HigherHighLowerLowBlock) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.zeros(n)
        for i in range(n):
            base = 100.0 + i * 0.5
            cycle = np.sin(i * 2 * np.pi / 20) * 5.0
            close[i] = base + cycle
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = hhll.execute(df, {"left_bars": 3, "right_bars": 3})
        assert result["hhll_is_hh"].sum() > 0
        assert result["hhll_is_hl"].sum() > 0
        assert (result["hhll_trend"] == "bullish").any()

    def test_downtrend_detection(self, hhll: HigherHighLowerLowBlock) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.zeros(n)
        for i in range(n):
            base = 200.0 - i * 0.5
            cycle = np.sin(i * 2 * np.pi / 20) * 5.0
            close[i] = base + cycle
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = hhll.execute(df, {"left_bars": 3, "right_bars": 3})
        assert result["hhll_is_lh"].sum() > 0
        assert result["hhll_is_ll"].sum() > 0
        assert (result["hhll_trend"] == "bearish").any()

    def test_swing_high_requires_left_right(
        self, hhll: HigherHighLowerLowBlock
    ) -> None:
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        high = np.full(n, 100.0)
        high[10] = 120.0
        low = np.full(n, 90.0)
        close = np.full(n, 95.0)
        close[10] = 115.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = hhll.execute(df, {"left_bars": 3, "right_bars": 3})
        assert not np.isnan(result["hhll_swing_high"].iloc[10])
        assert result["hhll_swing_high"].iloc[10] == 120.0

    def test_no_swings_in_flat_market(
        self, hhll: HigherHighLowerLowBlock
    ) -> None:
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = hhll.execute(df, {"left_bars": 3, "right_bars": 3})
        assert result["hhll_swing_high"].isna().all()
        assert result["hhll_swing_low"].isna().all()

    def test_trend_forward_fills(
        self, hhll: HigherHighLowerLowBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = hhll.execute(sample_ohlcv)
        trend = result["hhll_trend"]
        valid_values = {"bullish", "bearish", "neutral"}
        assert set(trend.unique()) <= valid_values

    def test_insufficient_data_raises(
        self, hhll: HigherHighLowerLowBlock
    ) -> None:
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
            hhll.execute(df, {"left_bars": 5, "right_bars": 5})
