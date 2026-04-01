"""Tests for the Ichimoku Kinko Hyo indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.ichimoku import IchimokuIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def ichimoku() -> IchimokuIndicator:
    return IchimokuIndicator()


class TestIchimokuMetadata:
    def test_name(self, ichimoku: IchimokuIndicator) -> None:
        assert ichimoku.metadata.name == "ichimoku"

    def test_category(self, ichimoku: IchimokuIndicator) -> None:
        assert ichimoku.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, ichimoku: IchimokuIndicator) -> None:
        defaults = ichimoku.metadata.get_defaults()
        assert defaults["tenkan_period"] == 9
        assert defaults["kijun_period"] == 26
        assert defaults["senkou_b_period"] == 52

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "ichimoku" in registry


class TestIchimokuCompute:
    def test_output_columns(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(sample_ohlcv)
        expected = {
            "ichimoku_tenkan",
            "ichimoku_kijun",
            "ichimoku_senkou_a",
            "ichimoku_senkou_b",
            "ichimoku_chikou",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_tenkan_is_donchian_midline(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(sample_ohlcv)
        expected = (
            sample_ohlcv["high"].rolling(9).max()
            + sample_ohlcv["low"].rolling(9).min()
        ) / 2.0
        pd.testing.assert_series_equal(
            result["ichimoku_tenkan"], expected, check_names=False
        )

    def test_kijun_is_donchian_midline(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(sample_ohlcv)
        expected = (
            sample_ohlcv["high"].rolling(26).max()
            + sample_ohlcv["low"].rolling(26).min()
        ) / 2.0
        pd.testing.assert_series_equal(
            result["ichimoku_kijun"], expected, check_names=False
        )

    def test_senkou_a_shifted_forward(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(sample_ohlcv)
        assert result["ichimoku_senkou_a"].iloc[:26].isna().all()

    def test_senkou_b_shifted_forward(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(sample_ohlcv)
        first_valid = result["ichimoku_senkou_b"].first_valid_index()
        assert first_valid is not None
        first_valid_pos = result.index.get_loc(first_valid)
        assert first_valid_pos >= 26 + 52 - 1

    def test_chikou_shifted_backward(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(sample_ohlcv)
        expected = sample_ohlcv["close"].shift(-26)
        pd.testing.assert_series_equal(
            result["ichimoku_chikou"], expected, check_names=False
        )

    def test_chikou_last_values_nan(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(sample_ohlcv)
        assert result["ichimoku_chikou"].iloc[-26:].isna().all()

    def test_uptrend_price_above_cloud(self, ichimoku: IchimokuIndicator) -> None:
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        trend = np.linspace(100, 300, n)
        df = pd.DataFrame(
            {
                "open": trend,
                "high": trend + 1.0,
                "low": trend - 0.5,
                "close": trend,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = ichimoku.execute(df)
        check_idx = -30
        close_check = df["close"].iloc[check_idx]
        senkou_a = result["ichimoku_senkou_a"].iloc[check_idx]
        senkou_b = result["ichimoku_senkou_b"].iloc[check_idx]
        if not (np.isnan(senkou_a) or np.isnan(senkou_b)):
            cloud_top = max(senkou_a, senkou_b)
            assert close_check > cloud_top

    def test_insufficient_data_raises(self, ichimoku: IchimokuIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        close = np.full(50, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(50) * 1000,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            ichimoku.execute(df)

    def test_custom_periods(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(
            sample_ohlcv,
            {"tenkan_period": 7, "kijun_period": 22, "senkou_b_period": 44},
        )
        assert "ichimoku_tenkan" in result.columns
