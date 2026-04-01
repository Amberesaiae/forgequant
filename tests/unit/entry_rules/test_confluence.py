"""Tests for the Confluence entry rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.entry_rules.confluence import ConfluenceEntry
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def confluence() -> ConfluenceEntry:
    return ConfluenceEntry()


class TestConfluenceMetadata:
    def test_name(self, confluence: ConfluenceEntry) -> None:
        assert confluence.metadata.name == "confluence_entry"

    def test_category(self, confluence: ConfluenceEntry) -> None:
        assert confluence.metadata.category == BlockCategory.ENTRY_RULE

    def test_defaults(self, confluence: ConfluenceEntry) -> None:
        d = confluence.metadata.get_defaults()
        assert d["trend_period"] == 50
        assert d["rsi_period"] == 14
        assert d["atr_period"] == 14

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "confluence_entry" in registry


class TestConfluenceCompute:
    def test_output_columns(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = confluence.execute(sample_ohlcv)
        expected = {
            "confluence_trend_ok", "confluence_momentum_ok",
            "confluence_volatility_ok",
            "confluence_long_entry", "confluence_short_entry",
            "confluence_score",
        }
        assert expected == set(result.columns)

    def test_score_range(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = confluence.execute(sample_ohlcv)
        scores = result["confluence_score"]
        assert scores.min() >= 0
        assert scores.max() <= 3

    def test_long_requires_all_three(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = confluence.execute(sample_ohlcv)
        long_bars = result[result["confluence_long_entry"]]
        if len(long_bars) > 0:
            assert long_bars["confluence_trend_ok"].all()
            assert long_bars["confluence_momentum_ok"].all()
            assert long_bars["confluence_volatility_ok"].all()

    def test_long_score_3_matches_long_entry(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = confluence.execute(sample_ohlcv)
        score_3 = result["confluence_score"] == 3
        long_entry = result["confluence_long_entry"]
        pd.testing.assert_series_equal(score_3, long_entry, check_names=False)

    def test_long_short_mutually_exclusive(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = confluence.execute(sample_ohlcv)
        both = result["confluence_long_entry"] & result["confluence_short_entry"]
        assert both.sum() == 0

    def test_strong_uptrend_produces_signals(
        self, confluence: ConfluenceEntry
    ) -> None:
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.normal(0.1, 0.3, n))
        close = np.maximum(close, 50.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + abs(np.random.randn(n)),
                "low": close - abs(np.random.randn(n)),
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = confluence.execute(df)
        assert result["confluence_long_entry"].sum() > 0

    def test_insufficient_data_raises(
        self, confluence: ConfluenceEntry
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=30, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
                "volume": [1000.0] * 30,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            confluence.execute(df, {"trend_period": 50})
