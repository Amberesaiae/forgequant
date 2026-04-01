"""Tests for the Kelly Fractional position sizing block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.money_management.kelly_fractional import KellyFractionalSizing
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def kelly() -> KellyFractionalSizing:
    return KellyFractionalSizing()


class TestKellyMetadata:
    def test_name(self, kelly: KellyFractionalSizing) -> None:
        assert kelly.metadata.name == "kelly_fractional"

    def test_category(self, kelly: KellyFractionalSizing) -> None:
        assert kelly.metadata.category == BlockCategory.MONEY_MANAGEMENT

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "kelly_fractional" in registry


class TestKellyCompute:
    def test_output_columns(
        self, kelly: KellyFractionalSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = kelly.execute(sample_ohlcv)
        expected = {
            "kelly_win_rate", "kelly_payoff_ratio",
            "kelly_full_fraction", "kelly_fraction_used",
            "kelly_position_size",
        }
        assert expected == set(result.columns)

    def test_fraction_used_non_negative(
        self, kelly: KellyFractionalSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = kelly.execute(sample_ohlcv)
        valid = result["kelly_fraction_used"].dropna()
        assert (valid >= 0).all()

    def test_fraction_used_capped(
        self, kelly: KellyFractionalSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        max_frac = 0.03
        result = kelly.execute(sample_ohlcv, {"max_fraction": max_frac})
        valid = result["kelly_fraction_used"].dropna()
        assert (valid <= max_frac + 1e-10).all()

    def test_win_rate_range(
        self, kelly: KellyFractionalSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = kelly.execute(sample_ohlcv)
        wr = result["kelly_win_rate"].dropna()
        assert (wr >= 0).all()
        assert (wr <= 1.0).all()

    def test_uptrend_positive_kelly(self, kelly: KellyFractionalSizing) -> None:
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.2, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = kelly.execute(df)
        assert result["kelly_fraction_used"].iloc[-1] > 0

    def test_downtrend_zero_kelly(self, kelly: KellyFractionalSizing) -> None:
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(200, 100, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.2,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = kelly.execute(df)
        assert result["kelly_fraction_used"].iloc[-1] == 0.0

    def test_insufficient_data_raises(self, kelly: KellyFractionalSizing) -> None:
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
            kelly.execute(df, {"lookback": 100})
