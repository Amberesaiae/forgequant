"""Tests for the Support & Resistance price action block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.price_action.support_resistance import SupportResistanceBlock
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def sr() -> SupportResistanceBlock:
    return SupportResistanceBlock()


class TestSRMetadata:
    def test_name(self, sr: SupportResistanceBlock) -> None:
        assert sr.metadata.name == "support_resistance"

    def test_category(self, sr: SupportResistanceBlock) -> None:
        assert sr.metadata.category == BlockCategory.PRICE_ACTION

    def test_defaults(self, sr: SupportResistanceBlock) -> None:
        d = sr.metadata.get_defaults()
        assert d["left_bars"] == 5
        assert d["right_bars"] == 5
        assert d["merge_pct"] == 0.5
        assert d["max_zones"] == 20

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "support_resistance" in registry


class TestSRCompute:
    def test_output_columns(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = sr.execute(sample_ohlcv)
        expected = {
            "sr_nearest_support", "sr_nearest_resistance",
            "sr_support_strength", "sr_resistance_strength",
            "sr_distance_to_support_pct", "sr_distance_to_resistance_pct",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_support_below_close(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = sr.execute(sample_ohlcv)
        valid = result.dropna(subset=["sr_nearest_support"])
        close = sample_ohlcv.loc[valid.index, "close"]
        assert (valid["sr_nearest_support"] < close).all()

    def test_resistance_above_close(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = sr.execute(sample_ohlcv)
        valid = result.dropna(subset=["sr_nearest_resistance"])
        close = sample_ohlcv.loc[valid.index, "close"]
        assert (valid["sr_nearest_resistance"] > close).all()

    def test_distance_positive(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = sr.execute(sample_ohlcv)
        sup_dist = result["sr_distance_to_support_pct"].dropna()
        res_dist = result["sr_distance_to_resistance_pct"].dropna()
        if len(sup_dist) > 0:
            assert (sup_dist > 0).all()
        if len(res_dist) > 0:
            assert (res_dist > 0).all()

    def test_strength_at_least_1(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = sr.execute(sample_ohlcv)
        sup_str = result["sr_support_strength"].dropna()
        res_str = result["sr_resistance_strength"].dropna()
        if len(sup_str) > 0:
            assert (sup_str >= 1).all()
        if len(res_str) > 0:
            assert (res_str >= 1).all()

    def test_merge_levels(self, sr: SupportResistanceBlock) -> None:
        levels = [100.0, 100.2, 100.3, 110.0, 110.1]
        zones = SupportResistanceBlock._merge_levels(levels, merge_pct=0.5)
        assert len(zones) == 2
        assert abs(zones[0][0] - 100.167) < 0.5
        assert zones[0][1] == 3
        assert abs(zones[1][0] - 110.05) < 0.5
        assert zones[1][1] == 2

    def test_empty_levels_merge(self, sr: SupportResistanceBlock) -> None:
        zones = SupportResistanceBlock._merge_levels([], merge_pct=0.5)
        assert zones == []

    def test_known_swing_levels(self, sr: SupportResistanceBlock) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.zeros(n)
        for i in range(n):
            close[i] = 100 + 10 * np.sin(i * 2 * np.pi / 20)
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
        result = sr.execute(df, {"left_bars": 3, "right_bars": 3})
        has_support = result["sr_nearest_support"].notna().any()
        has_resistance = result["sr_nearest_resistance"].notna().any()
        assert has_support or has_resistance

    def test_insufficient_data_raises(self, sr: SupportResistanceBlock) -> None:
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
            sr.execute(df, {"left_bars": 5, "right_bars": 5})
