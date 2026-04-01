"""Tests for the Bollinger Bands indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.bollinger_bands import BollingerBandsIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def bb() -> BollingerBandsIndicator:
    return BollingerBandsIndicator()


class TestBollingerMetadata:
    def test_name(self, bb: BollingerBandsIndicator) -> None:
        assert bb.metadata.name == "bollinger_bands"

    def test_category(self, bb: BollingerBandsIndicator) -> None:
        assert bb.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, bb: BollingerBandsIndicator) -> None:
        defaults = bb.metadata.get_defaults()
        assert defaults["period"] == 20
        assert defaults["num_std"] == 2.0

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "bollinger_bands" in registry


class TestBollingerCompute:
    def test_output_columns(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = bb.execute(sample_ohlcv)
        expected_cols = {"bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "bb_bandwidth"}
        assert expected_cols == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_upper_gt_middle_gt_lower(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = bb.execute(sample_ohlcv)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_middle_is_sma(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = bb.execute(sample_ohlcv, {"period": 20})
        expected_sma = sample_ohlcv["close"].rolling(20).mean()
        pd.testing.assert_series_equal(
            result["bb_middle"], expected_sma, check_names=False
        )

    def test_band_width_matches_std(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        period = 20
        num_std = 2.0
        result = bb.execute(sample_ohlcv, {"period": period, "num_std": num_std})
        std = sample_ohlcv["close"].rolling(period).std(ddof=0)
        expected_width = 2.0 * num_std * std
        actual_width = result["bb_upper"] - result["bb_lower"]
        pd.testing.assert_series_equal(
            actual_width, expected_width, check_names=False, atol=1e-10
        )

    def test_pct_b_at_middle(self, bb: BollingerBandsIndicator) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(42)
        close = 100.0 + np.random.randn(n) * 0.01
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = bb.execute(df, {"period": 20, "num_std": 2.0})
        pct_b = result["bb_pct_b"].dropna()
        assert abs(pct_b.mean() - 0.5) < 0.15

    def test_bandwidth_positive(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = bb.execute(sample_ohlcv)
        bw = result["bb_bandwidth"].dropna()
        assert (bw >= 0).all()

    def test_high_volatility_wide_bands(
        self, bb: BollingerBandsIndicator
    ) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(42)
        close_low = 100.0 + np.random.randn(n) * 0.1
        df_low = pd.DataFrame(
            {
                "open": close_low,
                "high": close_low + 0.2,
                "low": close_low - 0.2,
                "close": close_low,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        close_high = 100.0 + np.random.randn(n) * 5.0
        df_high = pd.DataFrame(
            {
                "open": close_high,
                "high": close_high + 1,
                "low": close_high - 1,
                "close": close_high,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        bw_low = bb.execute(df_low)["bb_bandwidth"].iloc[-1]
        bw_high = bb.execute(df_high)["bb_bandwidth"].iloc[-1]
        assert bw_high > bw_low

    def test_insufficient_data_raises(
        self, bb: BollingerBandsIndicator
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            bb.execute(df, {"period": 20})

    def test_custom_std(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result_2 = bb.execute(sample_ohlcv, {"num_std": 2.0})
        result_3 = bb.execute(sample_ohlcv, {"num_std": 3.0})
        width_2 = (result_2["bb_upper"] - result_2["bb_lower"]).dropna()
        width_3 = (result_3["bb_upper"] - result_3["bb_lower"]).dropna()
        assert (width_3 >= width_2 - 1e-10).all()
