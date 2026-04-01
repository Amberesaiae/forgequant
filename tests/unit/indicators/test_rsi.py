"""Tests for the RSI indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.rsi import RSIIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def rsi() -> RSIIndicator:
    return RSIIndicator()


def _make_df(close: list[float] | np.ndarray, n: int | None = None) -> pd.DataFrame:
    close_arr = np.array(close, dtype=float)
    length = n or len(close_arr)
    dates = pd.date_range("2024-01-01", periods=length, freq="h")
    return pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr + 0.5,
            "low": close_arr - 0.5,
            "close": close_arr,
            "volume": np.ones(length) * 1000.0,
        },
        index=dates,
    )


class TestRSIMetadata:
    def test_name(self, rsi: RSIIndicator) -> None:
        assert rsi.metadata.name == "rsi"

    def test_category(self, rsi: RSIIndicator) -> None:
        assert rsi.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, rsi: RSIIndicator) -> None:
        defaults = rsi.metadata.get_defaults()
        assert defaults["period"] == 14
        assert defaults["overbought"] == 70.0
        assert defaults["oversold"] == 30.0

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "rsi" in registry


class TestRSICompute:
    def test_default_params(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = rsi.execute(sample_ohlcv)
        assert "rsi_14" in result.columns
        assert "rsi_14_overbought" in result.columns
        assert "rsi_14_oversold" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_rsi_range(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = rsi.execute(sample_ohlcv)
        rsi_vals = result["rsi_14"].dropna()
        assert rsi_vals.min() >= 0.0
        assert rsi_vals.max() <= 100.0

    def test_all_gains_rsi_near_100(self, rsi: RSIIndicator) -> None:
        close = np.linspace(100, 200, 100)
        df = _make_df(close)
        result = rsi.execute(df)
        assert result["rsi_14"].iloc[-1] > 95.0

    def test_all_losses_rsi_near_0(self, rsi: RSIIndicator) -> None:
        close = np.linspace(200, 100, 100)
        df = _make_df(close)
        result = rsi.execute(df)
        assert result["rsi_14"].iloc[-1] < 5.0

    def test_flat_price_rsi_50(self, rsi: RSIIndicator) -> None:
        close = np.full(100, 150.0)
        df = _make_df(close)
        result = rsi.execute(df)
        rsi_last = result["rsi_14"].iloc[-1]
        assert rsi_last == 100.0

    def test_overbought_oversold_flags(self, rsi: RSIIndicator) -> None:
        close = np.linspace(100, 200, 100)
        df = _make_df(close)
        result = rsi.execute(df, {"period": 14, "overbought": 70.0, "oversold": 30.0})
        rsi_vals = result["rsi_14"]
        ob_flags = result["rsi_14_overbought"]
        os_flags = result["rsi_14_oversold"]
        for i in range(len(rsi_vals)):
            if not np.isnan(rsi_vals.iloc[i]):
                assert ob_flags.iloc[i] == (rsi_vals.iloc[i] >= 70.0)
                assert os_flags.iloc[i] == (rsi_vals.iloc[i] <= 30.0)

    def test_custom_period(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = rsi.execute(sample_ohlcv, {"period": 7})
        assert "rsi_7" in result.columns

    def test_overbought_lte_oversold_raises(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            rsi.execute(sample_ohlcv, {"overbought": 30.0, "oversold": 70.0})

    def test_insufficient_data_raises(self, rsi: RSIIndicator) -> None:
        close = [100.0] * 10
        df = _make_df(close)
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            rsi.execute(df, {"period": 14})

    def test_period_below_min_raises(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            rsi.execute(sample_ohlcv, {"period": 1})
