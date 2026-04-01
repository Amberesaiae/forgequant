"""Tests for the Fixed Risk position sizing block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.money_management.fixed_risk import FixedRiskSizing
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def fixed_risk() -> FixedRiskSizing:
    return FixedRiskSizing()


class TestFixedRiskMetadata:
    def test_name(self, fixed_risk: FixedRiskSizing) -> None:
        assert fixed_risk.metadata.name == "fixed_risk"

    def test_category(self, fixed_risk: FixedRiskSizing) -> None:
        assert fixed_risk.metadata.category == BlockCategory.MONEY_MANAGEMENT

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "fixed_risk" in registry


class TestFixedRiskCompute:
    def test_output_columns(
        self, fixed_risk: FixedRiskSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fixed_risk.execute(sample_ohlcv)
        expected = {
            "fr_atr", "fr_stop_distance", "fr_risk_amount",
            "fr_position_size", "fr_position_pct",
        }
        assert expected == set(result.columns)

    def test_risk_amount_constant(
        self, fixed_risk: FixedRiskSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fixed_risk.execute(
            sample_ohlcv, {"risk_pct": 2.0, "account_equity": 50000.0}
        )
        assert result["fr_risk_amount"].iloc[0] == 1000.0
        assert result["fr_risk_amount"].iloc[-1] == 1000.0

    def test_position_size_positive(
        self, fixed_risk: FixedRiskSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fixed_risk.execute(sample_ohlcv)
        valid = result["fr_position_size"].dropna()
        assert (valid > 0).all()

    def test_higher_vol_smaller_position(
        self, fixed_risk: FixedRiskSizing
    ) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        close_low = np.full(n, 100.0)
        df_low = pd.DataFrame(
            {
                "open": close_low, "high": close_low + 0.1,
                "low": close_low - 0.1, "close": close_low,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        close_high = np.full(n, 100.0)
        df_high = pd.DataFrame(
            {
                "open": close_high, "high": close_high + 5.0,
                "low": close_high - 5.0, "close": close_high,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        pos_low = fixed_risk.execute(df_low)["fr_position_size"].iloc[-1]
        pos_high = fixed_risk.execute(df_high)["fr_position_size"].iloc[-1]
        assert pos_low > pos_high

    def test_insufficient_data_raises(self, fixed_risk: FixedRiskSizing) -> None:
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
            fixed_risk.execute(df)
