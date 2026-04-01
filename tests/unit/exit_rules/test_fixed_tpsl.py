"""Tests for the Fixed TP/SL exit rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.exit_rules.fixed_tpsl import FixedTPSLExit
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def tpsl() -> FixedTPSLExit:
    return FixedTPSLExit()


class TestFixedTPSLMetadata:
    def test_name(self, tpsl: FixedTPSLExit) -> None:
        assert tpsl.metadata.name == "fixed_tpsl"

    def test_category(self, tpsl: FixedTPSLExit) -> None:
        assert tpsl.metadata.category == BlockCategory.EXIT_RULE

    def test_defaults(self, tpsl: FixedTPSLExit) -> None:
        d = tpsl.metadata.get_defaults()
        assert d["atr_period"] == 14
        assert d["tp_atr_mult"] == 3.0
        assert d["sl_atr_mult"] == 1.5
        assert d["min_rr"] == 0.5

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "fixed_tpsl" in registry


class TestFixedTPSLCompute:
    def test_output_columns(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        expected = {
            "tpsl_atr", "tpsl_long_tp", "tpsl_long_sl",
            "tpsl_short_tp", "tpsl_short_sl", "tpsl_risk_reward",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_long_tp_above_close(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["tpsl_long_tp"].dropna()
        assert (valid > close.loc[valid.index]).all()

    def test_long_sl_below_close(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["tpsl_long_sl"].dropna()
        assert (valid < close.loc[valid.index]).all()

    def test_short_tp_below_close(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["tpsl_short_tp"].dropna()
        assert (valid < close.loc[valid.index]).all()

    def test_short_sl_above_close(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        valid = result["tpsl_short_sl"].dropna()
        assert (valid > close.loc[valid.index]).all()

    def test_risk_reward_calculation(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv, {"tp_atr_mult": 3.0, "sl_atr_mult": 1.5})
        assert abs(result["tpsl_risk_reward"].iloc[0] - 2.0) < 1e-10

    def test_symmetry(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(sample_ohlcv)
        close = sample_ohlcv["close"]
        long_tp_dist = result["tpsl_long_tp"] - close
        short_tp_dist = close - result["tpsl_short_tp"]
        pd.testing.assert_series_equal(
            long_tp_dist, short_tp_dist, check_names=False, atol=1e-10
        )
        long_sl_dist = close - result["tpsl_long_sl"]
        short_sl_dist = result["tpsl_short_sl"] - close
        pd.testing.assert_series_equal(
            long_sl_dist, short_sl_dist, check_names=False, atol=1e-10
        )

    def test_low_rr_raises(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="Risk-reward ratio"):
            tpsl.execute(sample_ohlcv, {"tp_atr_mult": 0.5, "sl_atr_mult": 2.0, "min_rr": 0.5})

    def test_rr_check_disabled(
        self, tpsl: FixedTPSLExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = tpsl.execute(
            sample_ohlcv,
            {"tp_atr_mult": 0.5, "sl_atr_mult": 2.0, "min_rr": 0.0},
        )
        assert result["tpsl_risk_reward"].iloc[0] == 0.25

    def test_insufficient_data_raises(self, tpsl: FixedTPSLExit) -> None:
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
            tpsl.execute(df)
