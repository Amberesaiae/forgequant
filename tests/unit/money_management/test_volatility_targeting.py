"""Tests for the Volatility Targeting position sizing block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.money_management.volatility_targeting import VolatilityTargetingSizing
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def vol_target() -> VolatilityTargetingSizing:
    return VolatilityTargetingSizing()


class TestVolTargetMetadata:
    def test_name(self, vol_target: VolatilityTargetingSizing) -> None:
        assert vol_target.metadata.name == "volatility_targeting"

    def test_category(self, vol_target: VolatilityTargetingSizing) -> None:
        assert vol_target.metadata.category == BlockCategory.MONEY_MANAGEMENT

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "volatility_targeting" in registry


class TestVolTargetCompute:
    def test_output_columns(
        self, vol_target: VolatilityTargetingSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = vol_target.execute(sample_ohlcv)
        expected = {
            "vt_realized_vol", "vt_target_exposure",
            "vt_position_size", "vt_position_pct",
        }
        assert expected == set(result.columns)

    def test_exposure_capped_at_max_leverage(
        self, vol_target: VolatilityTargetingSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = vol_target.execute(sample_ohlcv, {"max_leverage": 3.0})
        valid = result["vt_target_exposure"].dropna()
        assert (valid <= 3.0 + 1e-10).all()

    def test_higher_vol_smaller_position(
        self, vol_target: VolatilityTargetingSizing
    ) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(42)

        close_low = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
        df_low = pd.DataFrame(
            {
                "open": close_low, "high": close_low * 1.001,
                "low": close_low * 0.999, "close": close_low,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        close_high = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        df_high = pd.DataFrame(
            {
                "open": close_high, "high": close_high * 1.01,
                "low": close_high * 0.99, "close": close_high,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        exp_low = vol_target.execute(df_low)["vt_target_exposure"].iloc[-1]
        exp_high = vol_target.execute(df_high)["vt_target_exposure"].iloc[-1]
        assert exp_low > exp_high

    def test_position_size_positive(
        self, vol_target: VolatilityTargetingSizing, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = vol_target.execute(sample_ohlcv)
        valid = result["vt_position_size"].dropna()
        assert (valid > 0).all()

    def test_insufficient_data_raises(
        self, vol_target: VolatilityTargetingSizing
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 10, "high": [101.0] * 10,
                "low": [99.0] * 10, "close": [100.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            vol_target.execute(df)
