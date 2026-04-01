"""Tests for the Trading Session filter block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.filters.trading_session import TradingSessionFilter
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def session() -> TradingSessionFilter:
    return TradingSessionFilter()


class TestTradingSessionMetadata:
    def test_name(self, session: TradingSessionFilter) -> None:
        assert session.metadata.name == "trading_session"

    def test_category(self, session: TradingSessionFilter) -> None:
        assert session.metadata.category == BlockCategory.FILTER

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "trading_session" in registry


class TestTradingSessionCompute:
    def test_output_columns(
        self, session: TradingSessionFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = session.execute(sample_ohlcv)
        assert "session_active" in result.columns
        assert "session_name" in result.columns

    def test_london_session(self, session: TradingSessionFilter) -> None:
        dates = pd.date_range("2024-01-15", periods=24, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = session.execute(
            df, {"session1_start": 8, "session1_end": 16, "session2_start": -1}
        )
        active = result["session_active"]
        for i, dt in enumerate(dates):
            expected = 8 <= dt.hour < 16
            assert active.iloc[i] == expected, f"Hour {dt.hour}: expected {expected}"

    def test_overnight_session(self, session: TradingSessionFilter) -> None:
        dates = pd.date_range("2024-01-15", periods=24, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = session.execute(
            df, {"session1_start": 22, "session1_end": 6, "session2_start": -1}
        )
        active = result["session_active"]
        for i, dt in enumerate(dates):
            expected = dt.hour >= 22 or dt.hour < 6
            assert active.iloc[i] == expected, f"Hour {dt.hour}"

    def test_two_sessions_overlap(self, session: TradingSessionFilter) -> None:
        dates = pd.date_range("2024-01-15", periods=24, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = session.execute(
            df,
            {
                "session1_start": 8, "session1_end": 16,
                "session2_start": 13, "session2_end": 21,
            },
        )
        names = result["session_name"]
        for i, dt in enumerate(dates):
            if 13 <= dt.hour < 16:
                assert names.iloc[i] == "overlap", f"Hour {dt.hour}"

    def test_session2_disabled(
        self, session: TradingSessionFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = session.execute(
            sample_ohlcv, {"session2_start": -1, "session2_end": -1}
        )
        names = result["session_name"]
        assert "session_2" not in names.values
