"""Tests for the ATR-Based Sizing block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.money_management.atr_based_sizing import ATRBasedSizing
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def atr_sizing() -> ATRBasedSizing:
    return ATRBasedSizing()


class TestATRSizingMetadata:
    def test_name(self, atr_sizing: ATRBasedSizing) -> None:
        assert atr_sizing.metadata.name == "atr_based_sizing"

    def test_category(self, atr_sizing: ATRBasedSizing) -> None:
        assert atr_sizing.metadata.category == BlockCategory.MONEY_MANAGEMENT

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "atr_based_sizing" in registry


class TestATRSizingCompute:
    def test_output_columns(
        self, atr_sizing: ATRBasedSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr_sizing.execute(sample_ohlcv)
        expected = {
            "atrs_atr", "atrs_risk_per_unit",
            "atrs_position_size", "atrs_position_value",
            "atrs_position_pct",
        }
        assert expected == set(result.columns)

    def test_position_size_positive(
        self, atr_sizing: ATRBasedSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr_sizing.execute(sample_ohlcv)
        valid = result["atrs_position_size"].dropna()
        assert (valid > 0).all()

    def test_position_pct_capped(
        self, atr_sizing: ATRBasedSizing
    ) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.001,
                "low": close - 0.001, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = atr_sizing.execute(df, {"max_position_pct": 20.0})
        valid = result["atrs_position_pct"].dropna()
        assert (valid <= 20.0 + 0.1).all()

    def test_inverse_relationship(
        self, atr_sizing: ATRBasedSizing
    ) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        close = np.full(n, 100.0)
        df_low = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        df_high = pd.DataFrame(
            {
                "open": close, "high": close + 5.0,
                "low": close - 5.0, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        pos_low = atr_sizing.execute(df_low)["atrs_position_size"].iloc[-1]
        pos_high = atr_sizing.execute(df_high)["atrs_position_size"].iloc[-1]
        assert pos_low > pos_high

    def test_insufficient_data_raises(self, atr_sizing: ATRBasedSizing) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5,
                "low": [99.0] * 5, "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            atr_sizing.execute(df)
