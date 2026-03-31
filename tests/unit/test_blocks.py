"""
Comprehensive tests for all 28 strategy blocks.

Run with:
    uv run pytest tests/unit/test_blocks.py -v
"""

import pandas as pd
import numpy as np
import pytest

from strategies_library.registry import BlockRegistry
from strategies_library.base import BaseBlock, BlockMetadata

# Import all blocks at module level to trigger registration
from strategies_library.blocks import *  # noqa: F401


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate realistic OHLCV test data."""
    np.random.seed(42)
    n = 500
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_p = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, size=n).astype(float)

    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    df["high"] = df[["high", "close"]].max(axis=1)
    df["low"] = df[["low", "close"]].min(axis=1)

    return df


class TestBlockRegistration:
    """Test that all blocks are properly registered."""

    EXPECTED_BLOCKS = {
        "EMA", "RSI", "ATR", "MACD", "BollingerBands", "ADX", "Stochastic", "Ichimoku",
        "Breakout", "Pullback", "HigherHighLowerLow", "SupportResistance",
        "Crossover", "ThresholdCross", "Confluence", "ReversalPattern",
        "FixedTPSL", "TrailingStop", "TimeBasedExit", "BreakevenStop",
        "FixedRisk", "VolatilityTargeting", "KellyFractional", "ATRBasedSizing",
        "TradingSessionFilter", "SpreadFilter", "MaxDrawdownFilter", "TrendFilter",
    }

    def test_all_blocks_registered(self):
        registered = set(BlockRegistry.get_all_names())
        missing = self.EXPECTED_BLOCKS - registered
        assert not missing, f"Missing blocks: {missing}"

    def test_block_count(self):
        assert BlockRegistry.count() == 28

    def test_category_counts(self):
        indicators = BlockRegistry.list_by_category("indicator")
        price_action = BlockRegistry.list_by_category("price_action")
        entries = BlockRegistry.list_by_category("entry")
        exits = BlockRegistry.list_by_category("exit")
        mm = BlockRegistry.list_by_category("money_management")
        filters = BlockRegistry.list_by_category("filter")

        assert len(indicators) == 8
        assert len(price_action) == 4
        assert len(entries) == 4
        assert len(exits) == 4
        assert len(mm) == 4
        assert len(filters) == 4


class TestIndicatorBlocks:
    def test_ema_returns_series(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("EMA")()
        result = block.compute(sample_ohlcv, {"period": 20})
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert pd.notna(result.iloc[-1])

    def test_rsi_bounded(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("RSI")()
        result = block.compute(sample_ohlcv, {"period": 14})
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("ATR")()
        result = block.compute(sample_ohlcv, {"period": 14})
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_macd_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("MACD")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, dict)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result
        assert isinstance(result["macd"], pd.Series)

    def test_bollinger_bands_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("BollingerBands")()
        result = block.compute(sample_ohlcv, {"period": 20, "std_dev": 2.0})
        assert isinstance(result, dict)
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result
        valid = pd.DataFrame(result).dropna()
        assert (valid["upper"] >= valid["middle"]).all()
        assert (valid["lower"] <= valid["middle"]).all()

    def test_adx_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("ADX")()
        result = block.compute(sample_ohlcv, {"period": 14})
        assert isinstance(result, dict)
        assert "adx" in result
        assert "plus_di" in result
        assert "minus_di" in result

    def test_stochastic_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("Stochastic")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, dict)
        assert "k" in result
        assert "d" in result
        valid = pd.DataFrame(result).dropna()
        assert (valid["k"] >= 0).all()
        assert (valid["k"] <= 100).all()

    def test_ichimoku_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("Ichimoku")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, dict)
        assert "tenkan" in result
        assert "kijun" in result
        assert "senkou_a" in result
        assert "senkou_b" in result
        assert "chikou" in result


class TestPriceActionBlocks:
    def test_breakout_returns_bool(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("Breakout")()
        result = block.compute(sample_ohlcv, {"lookback": 20, "direction": "long"})
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_pullback_returns_bool(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("Pullback")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_hhll_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("HigherHighLowerLow")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, dict)
        assert "higher_highs" in result
        assert "lower_lows" in result

    def test_support_resistance_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("SupportResistance")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, dict)
        assert "support" in result
        assert "resistance" in result
        assert "near_support" in result
        assert "near_resistance" in result


class TestEntryRuleBlocks:
    def test_crossover_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("Crossover")()
        result = block.compute(sample_ohlcv, {"fast_period": 9, "slow_period": 21})
        assert isinstance(result, dict)
        assert "long_entry" in result
        assert "short_entry" in result
        assert result["long_entry"].dtype == bool

    def test_threshold_cross_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("ThresholdCross")()
        result = block.compute(sample_ohlcv, {"indicator_name": "RSI", "threshold": 30.0})
        assert isinstance(result, dict)
        assert "cross_above" in result
        assert "cross_below" in result

    def test_confluence_returns_bool(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("Confluence")()
        result = block.compute(sample_ohlcv, {
            "conditions": [
                {"block_name": "RSI", "params": {"period": 14}},
                {"block_name": "EMA", "params": {"period": 20}},
            ],
            "min_conditions": 2,
        })
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_reversal_pattern_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("ReversalPattern")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, dict)
        assert "bullish_reversal" in result
        assert "bearish_reversal" in result


class TestExitRuleBlocks:
    def test_fixed_tp_sl_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("FixedTPSL")()
        result = block.compute(sample_ohlcv, {"tp_pips": 50.0, "sl_pips": 30.0})
        assert isinstance(result, dict)
        assert result["tp_pips"] == 50.0
        assert result["sl_pips"] == 30.0
        assert result["risk_reward"] == pytest.approx(50.0 / 30.0)

    def test_trailing_stop_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("TrailingStop")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, dict)
        assert "trailing_stop_long" in result
        assert "trailing_stop_short" in result

    def test_time_based_exit_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("TimeBasedExit")()
        result = block.compute(sample_ohlcv, {"max_bars": 12})
        assert isinstance(result, dict)
        assert result["max_bars"] == 12
        assert "exit_signal" in result

    def test_breakeven_stop_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("BreakevenStop")()
        result = block.compute(sample_ohlcv, {"activation_pips": 20.0, "offset_pips": 2.0})
        assert isinstance(result, dict)
        assert result["activation_pips"] == 20.0
        assert result["offset_pips"] == 2.0


class TestMoneyManagementBlocks:
    def test_fixed_risk_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("FixedRisk")()
        result = block.compute(sample_ohlcv, {"risk_percent": 1.0})
        assert isinstance(result, dict)
        assert result["risk_percent"] == 0.01

    def test_volatility_targeting_returns_series(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("VolatilityTargeting")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert pd.notna(result.iloc[-1])

    def test_kelly_fractional_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("KellyFractional")()
        result = block.compute(sample_ohlcv, {
            "win_rate": 0.55,
            "avg_win_loss_ratio": 1.5,
            "kelly_fraction": 0.25,
        })
        assert isinstance(result, dict)
        assert "full_kelly_pct" in result
        assert "adjusted_kelly_pct" in result
        assert "risk_pct" in result
        assert result["risk_pct"] > 0

    def test_atr_based_sizing_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("ATRBasedSizing")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, dict)
        assert "atr" in result
        assert "stop_distance" in result
        assert "risk_percent" in result


class TestFilterBlocks:
    def test_trend_filter_returns_dict(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("TrendFilter")()
        result = block.compute(sample_ohlcv, {"period": 50})
        assert isinstance(result, dict)
        assert "allow_long" in result
        assert "allow_short" in result

    def test_spread_filter_allows_when_no_spread(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("SpreadFilter")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert result.all()

    def test_spread_filter_blocks_high_spread(self):
        df = pd.DataFrame({
            "spread": [1.0, 2.0, 5.0, 1.5, 0.5],
        }, index=pd.date_range("2024-01-01", periods=5, freq="h"))
        block = BlockRegistry.get("SpreadFilter")()
        result = block.compute(df, {"max_spread_pips": 3.0})
        assert list(result) == [True, True, False, True, True]

    def test_trading_session_filter_returns_bool(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("TradingSessionFilter")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_max_drawdown_filter_returns_bool(self, sample_ohlcv: pd.DataFrame):
        block = BlockRegistry.get("MaxDrawdownFilter")()
        result = block.compute(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool


class TestParamValidation:
    def test_ema_valid_params(self):
        block = BlockRegistry.get("EMA")()
        assert block.validate_params({"period": 20}) is True

    def test_rsi_valid_params(self):
        block = BlockRegistry.get("RSI")()
        assert block.validate_params({"period": 14}) is True

    def test_bollinger_invalid_params(self):
        block = BlockRegistry.get("BollingerBands")()
        assert block.validate_params({"period": 3, "std_dev": 2.0}) is False
        assert block.validate_params({"period": 20, "std_dev": 5.0}) is False

    def test_crossover_invalid_params(self):
        block = BlockRegistry.get("Crossover")()
        assert block.validate_params({"fast_period": 21, "slow_period": 9}) is False
        assert block.validate_params({"fast_period": 9, "slow_period": 21, "ma_type": "invalid"}) is False

    def test_fixed_tp_sl_invalid_params(self):
        block = BlockRegistry.get("FixedTPSL")()
        assert block.validate_params({"tp_pips": -10, "sl_pips": 30}) is False
        assert block.validate_params({"tp_pips": 50, "sl_pips": 0}) is False

    def test_kelly_invalid_params(self):
        block = BlockRegistry.get("KellyFractional")()
        assert block.validate_params({"win_rate": 1.5, "kelly_fraction": 0.25}) is False
        assert block.validate_params({"win_rate": 0.55, "kelly_fraction": 2.0}) is False

    def test_trend_filter_invalid_params(self):
        block = BlockRegistry.get("TrendFilter")()
        assert block.validate_params({"period": 10, "ma_type": "ema"}) is False
        assert block.validate_params({"period": 200, "ma_type": "invalid"}) is False


class TestBlockMetadata:
    def test_all_blocks_have_metadata(self):
        for name in BlockRegistry.get_all_names():
            block_class = BlockRegistry.get(name)
            assert block_class is not None
            block = block_class()
            meta = block.get_metadata()
            assert meta.name == name
            assert meta.category in [
                "indicator", "price_action", "entry", "exit",
                "money_management", "filter",
            ]
            assert meta.complexity >= 1
            assert meta.complexity <= 5
            assert len(meta.description) > 0
            assert len(meta.typical_use) > 0

    def test_block_repr(self):
        block = BlockRegistry.get("EMA")()
        assert "EMA" in repr(block)
        assert "indicator" in repr(block)

    def test_search_returns_results(self):
        results = BlockRegistry.search("trend")
        assert len(results) > 0

    def test_search_no_results(self):
        results = BlockRegistry.search("xyznonexistent")
        assert len(results) == 0

    def test_get_or_raise(self):
        block_class = BlockRegistry.get_or_raise("EMA")
        assert block_class is not None

    def test_get_or_raise_missing(self):
        with pytest.raises(KeyError):
            BlockRegistry.get_or_raise("NonExistentBlock")
