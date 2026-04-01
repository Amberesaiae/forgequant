"""Tests for the Pullback price action block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.price_action.pullback import PullbackBlock
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def pullback() -> PullbackBlock:
    return PullbackBlock()


class TestPullbackMetadata:
    def test_name(self, pullback: PullbackBlock) -> None:
        assert pullback.metadata.name == "pullback"

    def test_category(self, pullback: PullbackBlock) -> None:
        assert pullback.metadata.category == BlockCategory.PRICE_ACTION

    def test_defaults(self, pullback: PullbackBlock) -> None:
        d = pullback.metadata.get_defaults()
        assert d["ma_period"] == 20
        assert d["ma_type"] == "ema"
        assert d["proximity_pct"] == 0.5

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "pullback" in registry


class TestPullbackCompute:
    def test_output_columns(
        self, pullback: PullbackBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = pullback.execute(sample_ohlcv)
        expected = {
            "pullback_ma", "pullback_upper_zone", "pullback_lower_zone",
            "pullback_long", "pullback_short",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_zone_surrounds_ma(
        self, pullback: PullbackBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = pullback.execute(sample_ohlcv)
        valid = result.dropna()
        assert (valid["pullback_upper_zone"] >= valid["pullback_ma"]).all()
        assert (valid["pullback_lower_zone"] <= valid["pullback_ma"]).all()

    def test_ema_vs_sma(
        self, pullback: PullbackBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result_ema = pullback.execute(sample_ohlcv, {"ma_type": "ema"})
        result_sma = pullback.execute(sample_ohlcv, {"ma_type": "sma"})
        assert not result_ema["pullback_ma"].equals(result_sma["pullback_ma"])

    def test_long_pullback_detection(self, pullback: PullbackBlock) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 150, n)
        close[60] = close[59] - 5.0
        close[61] = close[59] - 3.0
        close[62] = close[59] + 1.0
        high = close + 1.0
        low = close - 1.0
        low[62] = close[59] - 2.0
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
        result = pullback.execute(df, {"ma_period": 10, "proximity_pct": 3.0})
        assert result["pullback_long"].sum() > 0

    def test_no_pullback_in_flat_market(self, pullback: PullbackBlock) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.01,
                "low": close - 0.01,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = pullback.execute(df, {"ma_period": 10, "proximity_pct": 0.5})
        assert result["pullback_long"].sum() == 0
        assert result["pullback_short"].sum() == 0

    def test_insufficient_data_raises(self, pullback: PullbackBlock) -> None:
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
            pullback.execute(df, {"ma_period": 20})

    def test_invalid_ma_type_raises(
        self, pullback: PullbackBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            pullback.execute(sample_ohlcv, {"ma_type": "wma"})
