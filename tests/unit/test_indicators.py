"""
Unit tests for all 8 indicator blocks.

Run with:
    uv run pytest tests/unit/test_indicators.py -v
"""

import pandas as pd
import numpy as np
import pytest

from strategies_library.registry import BlockRegistry


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create realistic sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200

    # Generate a random walk for close prices
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)

    # Generate realistic OHLC from close
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_price = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 100000, size=n).astype(float)

    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


# Import at module level to trigger @BlockRegistry.register decorators
import strategies_library.blocks.indicators  # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state for each test."""
    # Save current state
    saved_blocks = dict(BlockRegistry._blocks)
    saved_metadata = dict(BlockRegistry._metadata)
    yield
    # Restore
    BlockRegistry._blocks.clear()
    BlockRegistry._metadata.clear()
    BlockRegistry._blocks.update(saved_blocks)
    BlockRegistry._metadata.update(saved_metadata)


class TestEMA:
    """Tests for the EMA indicator block."""

    def test_ema_exists_in_registry(self):
        assert BlockRegistry.get("EMA") is not None

    def test_ema_returns_series(self, sample_ohlcv_data):
        block = BlockRegistry.get("EMA")()
        result = block.compute(sample_ohlcv_data, {"period": 20})
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)

    def test_ema_default_params(self, sample_ohlcv_data):
        block = BlockRegistry.get("EMA")()
        result = block.compute(sample_ohlcv_data)
        assert not result.isna().all()

    def test_ema_validate_params_valid(self):
        block = BlockRegistry.get("EMA")()
        assert block.validate_params({"period": 20}) is True

    def test_ema_validate_params_invalid(self):
        block = BlockRegistry.get("EMA")()
        assert block.validate_params({"period": 1}) is False
        assert block.validate_params({"period": 600}) is False


class TestRSI:
    """Tests for the RSI indicator block."""

    def test_rsi_exists_in_registry(self):
        assert BlockRegistry.get("RSI") is not None

    def test_rsi_returns_series(self, sample_ohlcv_data):
        block = BlockRegistry.get("RSI")()
        result = block.compute(sample_ohlcv_data, {"period": 14})
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)

    def test_rsi_values_in_range(self, sample_ohlcv_data):
        block = BlockRegistry.get("RSI")()
        result = block.compute(sample_ohlcv_data, {"period": 14})
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_validate_params_valid(self):
        block = BlockRegistry.get("RSI")()
        assert block.validate_params({"period": 14}) is True

    def test_rsi_validate_params_invalid(self):
        block = BlockRegistry.get("RSI")()
        assert block.validate_params({"period": 1}) is False
        assert block.validate_params({"period": 200}) is False


class TestATR:
    """Tests for the ATR indicator block."""

    def test_atr_exists_in_registry(self):
        assert BlockRegistry.get("ATR") is not None

    def test_atr_returns_series(self, sample_ohlcv_data):
        block = BlockRegistry.get("ATR")()
        result = block.compute(sample_ohlcv_data, {"period": 14})
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)

    def test_atr_values_positive(self, sample_ohlcv_data):
        block = BlockRegistry.get("ATR")()
        result = block.compute(sample_ohlcv_data, {"period": 14})
        valid_values = result.dropna()
        assert (valid_values >= 0).all()


class TestBollingerBands:
    """Tests for the Bollinger Bands indicator block."""

    def test_bollinger_exists_in_registry(self):
        assert BlockRegistry.get("BollingerBands") is not None

    def test_bollinger_returns_dict(self, sample_ohlcv_data):
        block = BlockRegistry.get("BollingerBands")()
        result = block.compute(sample_ohlcv_data, {"period": 20, "std_dev": 2.0})
        assert isinstance(result, dict)
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result
        assert "bandwidth" in result

    def test_bollinger_upper_above_lower(self, sample_ohlcv_data):
        block = BlockRegistry.get("BollingerBands")()
        result = block.compute(sample_ohlcv_data, {"period": 20, "std_dev": 2.0})
        valid_idx = result["upper"].dropna().index
        assert (result["upper"][valid_idx] >= result["lower"][valid_idx]).all()

    def test_bollinger_validate_params_invalid(self):
        block = BlockRegistry.get("BollingerBands")()
        assert block.validate_params({"period": 3, "std_dev": 2.0}) is False
        assert block.validate_params({"period": 20, "std_dev": 5.0}) is False


class TestMACD:
    """Tests for the MACD indicator block."""

    def test_macd_exists_in_registry(self):
        assert BlockRegistry.get("MACD") is not None

    def test_macd_returns_dict(self, sample_ohlcv_data):
        block = BlockRegistry.get("MACD")()
        result = block.compute(sample_ohlcv_data)
        assert isinstance(result, dict)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_macd_histogram_equals_difference(self, sample_ohlcv_data):
        block = BlockRegistry.get("MACD")()
        result = block.compute(sample_ohlcv_data)
        diff = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], diff)

    def test_macd_validate_fast_must_be_less_than_slow(self):
        block = BlockRegistry.get("MACD")()
        assert block.validate_params({"fast_period": 26, "slow_period": 12}) is False
        assert block.validate_params({"fast_period": 12, "slow_period": 26}) is True


class TestADX:
    """Tests for the ADX indicator block."""

    def test_adx_exists_in_registry(self):
        assert BlockRegistry.get("ADX") is not None

    def test_adx_returns_dict(self, sample_ohlcv_data):
        block = BlockRegistry.get("ADX")()
        result = block.compute(sample_ohlcv_data, {"period": 14})
        assert isinstance(result, dict)
        assert "adx" in result
        assert "plus_di" in result
        assert "minus_di" in result

    def test_adx_values_reasonable(self, sample_ohlcv_data):
        block = BlockRegistry.get("ADX")()
        result = block.compute(sample_ohlcv_data, {"period": 14})
        valid_adx = result["adx"].dropna()
        assert (valid_adx >= 0).all()


class TestStochastic:
    """Tests for the Stochastic Oscillator indicator block."""

    def test_stochastic_exists_in_registry(self):
        assert BlockRegistry.get("Stochastic") is not None

    def test_stochastic_returns_dict(self, sample_ohlcv_data):
        block = BlockRegistry.get("Stochastic")()
        result = block.compute(sample_ohlcv_data)
        assert isinstance(result, dict)
        assert "k" in result
        assert "d" in result

    def test_stochastic_values_in_range(self, sample_ohlcv_data):
        block = BlockRegistry.get("Stochastic")()
        result = block.compute(sample_ohlcv_data)
        valid_k = result["k"].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()


class TestIchimoku:
    """Tests for the Ichimoku Cloud indicator block."""

    def test_ichimoku_exists_in_registry(self):
        assert BlockRegistry.get("Ichimoku") is not None

    def test_ichimoku_returns_dict(self, sample_ohlcv_data):
        block = BlockRegistry.get("Ichimoku")()
        result = block.compute(sample_ohlcv_data)
        assert isinstance(result, dict)
        assert "tenkan" in result
        assert "kijun" in result
        assert "senkou_a" in result
        assert "senkou_b" in result
        assert "chikou" in result

    def test_ichimoku_validate_tenkan_less_than_kijun(self):
        block = BlockRegistry.get("Ichimoku")()
        assert block.validate_params({"tenkan_period": 30, "kijun_period": 26}) is False
        assert block.validate_params({"tenkan_period": 9, "kijun_period": 26}) is True


class TestAllIndicatorsRegistered:
    """Verify all 8 indicators are properly registered."""

    def test_all_eight_indicators_registered(self):
        indicators = BlockRegistry.list_by_category("indicator")
        names = [m.name for m in indicators]
        expected = ["EMA", "RSI", "ATR", "BollingerBands", "MACD", "ADX", "Stochastic", "Ichimoku"]
        for name in expected:
            assert name in names, f"Missing indicator: {name}"
        assert len(indicators) == 8
