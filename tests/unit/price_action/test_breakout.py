"""Tests for the Breakout price action block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.price_action.breakout import BreakoutBlock
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def breakout() -> BreakoutBlock:
    return BreakoutBlock()


def _make_ohlcv(close: np.ndarray, spread: float = 0.5) -> pd.DataFrame:
    n = len(close)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.ones(n) * 1000.0,
        },
        index=dates,
    )


class TestBreakoutMetadata:
    def test_name(self, breakout: BreakoutBlock) -> None:
        assert breakout.metadata.name == "breakout"

    def test_category(self, breakout: BreakoutBlock) -> None:
        assert breakout.metadata.category == BlockCategory.PRICE_ACTION

    def test_defaults(self, breakout: BreakoutBlock) -> None:
        d = breakout.metadata.get_defaults()
        assert d["lookback"] == 20
        assert d["volume_multiplier"] == 1.5
        assert d["volume_lookback"] == 20

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "breakout" in registry


class TestBreakoutCompute:
    def test_output_columns(
        self, breakout: BreakoutBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = breakout.execute(sample_ohlcv)
        expected = {
            "breakout_resistance", "breakout_support",
            "breakout_long", "breakout_short",
            "breakout_volume_confirm",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_anti_lookahead_shift(self, breakout: BreakoutBlock) -> None:
        n = 30
        close = np.full(n, 100.0)
        close[25] = 200.0
        df = _make_ohlcv(close)
        result = breakout.execute(df, {"lookback": 5, "volume_multiplier": 0.0})
        res_at_25 = result["breakout_resistance"].iloc[25]
        assert res_at_25 < 200.0

    def test_upward_breakout_detected(self, breakout: BreakoutBlock) -> None:
        n = 50
        close = np.full(n, 100.0)
        close[30] = 110.0
        df = _make_ohlcv(close)
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 0.0})
        assert result["breakout_long"].iloc[30] == True

    def test_downward_breakout_detected(self, breakout: BreakoutBlock) -> None:
        n = 50
        close = np.full(n, 100.0)
        close[30] = 90.0
        df = _make_ohlcv(close)
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 0.0})
        assert result["breakout_short"].iloc[30] == True

    def test_no_breakout_in_flat_market(self, breakout: BreakoutBlock) -> None:
        n = 100
        close = np.full(n, 100.0)
        df = _make_ohlcv(close, spread=0.0)
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 0.0})
        assert result["breakout_long"].iloc[15:].sum() == 0
        assert result["breakout_short"].iloc[15:].sum() == 0

    def test_volume_confirmation(self, breakout: BreakoutBlock) -> None:
        n = 50
        close = np.full(n, 100.0)
        close[30] = 110.0
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        volume = np.ones(n) * 1000.0
        volume[30] = 500.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 1.5})
        assert result["breakout_long"].iloc[30] == True
        assert result["breakout_volume_confirm"].iloc[30] == False

    def test_volume_disabled(self, breakout: BreakoutBlock) -> None:
        n = 50
        close = np.full(n, 100.0)
        df = _make_ohlcv(close)
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 0.0})
        assert result["breakout_volume_confirm"].all()

    def test_insufficient_data_raises(self, breakout: BreakoutBlock) -> None:
        close = np.full(10, 100.0)
        df = _make_ohlcv(close)
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            breakout.execute(df, {"lookback": 20})

    def test_resistance_above_support(
        self, breakout: BreakoutBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = breakout.execute(sample_ohlcv)
        valid = result.dropna(subset=["breakout_resistance", "breakout_support"])
        assert (valid["breakout_resistance"] >= valid["breakout_support"]).all()
