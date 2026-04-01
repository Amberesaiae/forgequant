"""Tests for the Threshold Cross entry rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.entry_rules.threshold_cross import ThresholdCrossEntry
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def threshold() -> ThresholdCrossEntry:
    return ThresholdCrossEntry()


class TestThresholdMetadata:
    def test_name(self, threshold: ThresholdCrossEntry) -> None:
        assert threshold.metadata.name == "threshold_cross_entry"

    def test_category(self, threshold: ThresholdCrossEntry) -> None:
        assert threshold.metadata.category == BlockCategory.ENTRY_RULE

    def test_defaults(self, threshold: ThresholdCrossEntry) -> None:
        d = threshold.metadata.get_defaults()
        assert d["rsi_period"] == 14
        assert d["upper_threshold"] == 70.0
        assert d["lower_threshold"] == 30.0
        assert d["mode"] == "mean_reversion"

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "threshold_cross_entry" in registry


class TestThresholdCompute:
    def test_output_columns(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = threshold.execute(sample_ohlcv)
        expected = {
            "threshold_indicator",
            "threshold_long_entry",
            "threshold_short_entry",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_indicator_is_rsi(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = threshold.execute(sample_ohlcv)
        rsi = result["threshold_indicator"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_mean_reversion_long_on_oversold_recovery(
        self, threshold: ThresholdCrossEntry
    ) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([
            np.linspace(200, 100, 100),
            np.linspace(100, 150, 100),
        ])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = threshold.execute(df, {"mode": "mean_reversion"})
        long_signals = result["threshold_long_entry"].iloc[100:].sum()
        assert long_signals >= 1

    def test_momentum_long_on_overbought_entry(
        self, threshold: ThresholdCrossEntry
    ) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([
            np.full(100, 100.0),
            np.linspace(100, 200, 100),
        ])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = threshold.execute(df, {"mode": "momentum"})
        long_signals = result["threshold_long_entry"].iloc[100:].sum()
        assert long_signals >= 1

    def test_signals_are_state_changes(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = threshold.execute(sample_ohlcv)
        total = result["threshold_long_entry"].sum() + result["threshold_short_entry"].sum()
        assert total < len(sample_ohlcv) / 2

    def test_upper_lte_lower_raises(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be greater than"):
            threshold.execute(
                sample_ohlcv,
                {"upper_threshold": 50.0, "lower_threshold": 50.0},
            )

    def test_insufficient_data_raises(self, threshold: ThresholdCrossEntry) -> None:
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
            threshold.execute(df)

    def test_invalid_mode_raises(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            threshold.execute(sample_ohlcv, {"mode": "invalid"})
