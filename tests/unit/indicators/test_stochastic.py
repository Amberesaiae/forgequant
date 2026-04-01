"""Tests for the Stochastic Oscillator indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.stochastic import StochasticIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def stoch() -> StochasticIndicator:
    return StochasticIndicator()


class TestStochasticMetadata:
    def test_name(self, stoch: StochasticIndicator) -> None:
        assert stoch.metadata.name == "stochastic"

    def test_category(self, stoch: StochasticIndicator) -> None:
        assert stoch.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, stoch: StochasticIndicator) -> None:
        defaults = stoch.metadata.get_defaults()
        assert defaults["k_period"] == 14
        assert defaults["k_smooth"] == 3
        assert defaults["d_period"] == 3
        assert defaults["overbought"] == 80.0
        assert defaults["oversold"] == 20.0

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "stochastic" in registry


class TestStochasticCompute:
    def test_output_columns(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(sample_ohlcv)
        expected = {"stoch_k", "stoch_d", "stoch_overbought", "stoch_oversold"}
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_stoch_k_range(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(sample_ohlcv)
        k_vals = result["stoch_k"].dropna()
        assert k_vals.min() >= -1e-10
        assert k_vals.max() <= 100.0 + 1e-10

    def test_stoch_d_range(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(sample_ohlcv)
        d_vals = result["stoch_d"].dropna()
        assert d_vals.min() >= -1e-10
        assert d_vals.max() <= 100.0 + 1e-10

    def test_d_is_sma_of_k(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(sample_ohlcv, {"d_period": 3})
        expected_d = result["stoch_k"].rolling(3).mean()
        pd.testing.assert_series_equal(
            result["stoch_d"], expected_d, check_names=False
        )

    def test_close_at_high_k_near_100(self, stoch: StochasticIndicator) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = stoch.execute(df)
        k_tail = result["stoch_k"].iloc[-10:]
        assert k_tail.dropna().mean() > 90.0

    def test_close_at_low_k_near_0(self, stoch: StochasticIndicator) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(200, 100, n)
        df = pd.DataFrame(
            {
                "open": close + 0.5,
                "high": close + 1.0,
                "low": close,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = stoch.execute(df)
        k_tail = result["stoch_k"].iloc[-10:]
        assert k_tail.dropna().mean() < 10.0

    def test_overbought_oversold_flags(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(sample_ohlcv)
        k_vals = result["stoch_k"]
        ob = result["stoch_overbought"]
        os_flag = result["stoch_oversold"]
        for i in range(len(k_vals)):
            if not np.isnan(k_vals.iloc[i]):
                assert ob.iloc[i] == (k_vals.iloc[i] >= 80.0)
                assert os_flag.iloc[i] == (k_vals.iloc[i] <= 20.0)

    def test_fast_stochastic_k_smooth_1(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(sample_ohlcv, {"k_smooth": 1})
        ll = sample_ohlcv["low"].rolling(14).min()
        hh = sample_ohlcv["high"].rolling(14).max()
        raw_k = 100.0 * (sample_ohlcv["close"] - ll) / (hh - ll).replace(0, np.nan)
        pd.testing.assert_series_equal(
            result["stoch_k"], raw_k, check_names=False
        )

    def test_overbought_lte_oversold_raises(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            stoch.execute(sample_ohlcv, {"overbought": 20.0, "oversold": 80.0})

    def test_insufficient_data_raises(self, stoch: StochasticIndicator) -> None:
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
            stoch.execute(df)

    def test_custom_periods(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(
            sample_ohlcv,
            {"k_period": 5, "k_smooth": 2, "d_period": 2},
        )
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns
