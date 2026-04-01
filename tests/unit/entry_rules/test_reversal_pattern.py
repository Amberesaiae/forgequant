"""Tests for the Reversal Pattern entry rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.entry_rules.reversal_pattern import ReversalPatternEntry
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def reversal() -> ReversalPatternEntry:
    return ReversalPatternEntry()


def _make_bar(
    open_: float, high: float, low: float, close: float
) -> dict[str, float]:
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0}


class TestReversalMetadata:
    def test_name(self, reversal: ReversalPatternEntry) -> None:
        assert reversal.metadata.name == "reversal_pattern_entry"

    def test_category(self, reversal: ReversalPatternEntry) -> None:
        assert reversal.metadata.category == BlockCategory.ENTRY_RULE

    def test_defaults(self, reversal: ReversalPatternEntry) -> None:
        d = reversal.metadata.get_defaults()
        assert d["pin_bar_ratio"] == 2.0
        assert d["max_opposite_wick_ratio"] == 0.5
        assert d["star_body_pct"] == 30.0

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "reversal_pattern_entry" in registry


class TestReversalCompute:
    def test_output_columns(
        self, reversal: ReversalPatternEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = reversal.execute(sample_ohlcv)
        expected = {
            "reversal_bullish_engulfing", "reversal_bearish_engulfing",
            "reversal_hammer", "reversal_shooting_star",
            "reversal_morning_star", "reversal_evening_star",
            "reversal_long_entry", "reversal_short_entry",
        }
        assert expected == set(result.columns)

    def test_bullish_engulfing(self, reversal: ReversalPatternEntry) -> None:
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101, 99, 100),
            _make_bar(102, 103, 99, 100),
            _make_bar(99, 104, 98, 103),
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_bullish_engulfing"].iloc[-1] == True

    def test_bearish_engulfing(self, reversal: ReversalPatternEntry) -> None:
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 103, 99, 102),
            _make_bar(103, 104, 98, 99),
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_bearish_engulfing"].iloc[-1] == True

    def test_hammer(self, reversal: ReversalPatternEntry) -> None:
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101.2, 95, 101),
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_hammer"].iloc[-1] == True

    def test_shooting_star(self, reversal: ReversalPatternEntry) -> None:
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 105, 98.8, 99),
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_shooting_star"].iloc[-1] == True

    def test_morning_star(self, reversal: ReversalPatternEntry) -> None:
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(110, 111, 99, 100),
            _make_bar(100, 101, 99, 100.5),
            _make_bar(101, 107, 100, 106),
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_morning_star"].iloc[-1] == True

    def test_evening_star(self, reversal: ReversalPatternEntry) -> None:
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 111, 99, 110),
            _make_bar(110, 111, 109, 110.5),
            _make_bar(110, 111, 103, 104),
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_evening_star"].iloc[-1] == True

    def test_combined_long_entry(self, reversal: ReversalPatternEntry) -> None:
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101.2, 95, 101),
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_long_entry"].iloc[-1] == True

    def test_combined_short_entry(self, reversal: ReversalPatternEntry) -> None:
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 105, 98.8, 99),
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_short_entry"].iloc[-1] == True

    def test_no_pattern_in_flat_market(
        self, reversal: ReversalPatternEntry
    ) -> None:
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.1] * n,
                "low": [99.9] * n,
                "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = reversal.execute(df)
        assert result["reversal_long_entry"].sum() == 0
        assert result["reversal_short_entry"].sum() == 0

    def test_on_sample_data_finds_patterns(
        self, reversal: ReversalPatternEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = reversal.execute(sample_ohlcv)
        total = result["reversal_long_entry"].sum() + result["reversal_short_entry"].sum()
        assert total > 0

    def test_insufficient_data_raises(
        self, reversal: ReversalPatternEntry
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "volume": [1000.0, 1000.0],
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            reversal.execute(df)
