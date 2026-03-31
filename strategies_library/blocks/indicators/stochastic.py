"""
Stochastic Oscillator Indicator Block.

The Stochastic Oscillator compares a security's closing price
to its price range over a given period.

Components:
    - %K (fast line): Measures where the close is relative to the high-low range
    - %D (slow line): Smoothed version of %K

Formula:
    Raw %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %K = SMA(Raw %K, smooth_k)
    %D = SMA(%K, d_period)

Interpretation:
    - %K < 20: Oversold zone
    - %K > 80: Overbought zone
    - %K crosses above %D in oversold zone: Buy signal
    - %K crosses below %D in overbought zone: Sell signal

Default Parameters:
    k_period: 14
    d_period: 3
    smooth_k: 3
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class Stochastic(BaseBlock):
    """Stochastic Oscillator indicator."""

    metadata = BlockMetadata(
        name="Stochastic",
        category="indicator",
        description="Stochastic Oscillator — measures momentum via price position in range",
        complexity=3,
        typical_use=["mean_reversion", "overbought_oversold", "momentum"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["oscillator", "momentum", "mean_reversion", "overbought"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute Stochastic %K and %D.

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns.
            params: Optional dict with:
                - k_period (int): Lookback period for raw %K. Default 14.
                - d_period (int): SMA period for %D. Default 3.
                - smooth_k (int): SMA period for smoothing raw %K. Default 3.

        Returns:
            Dict with keys:
                - 'k': %K line Series (0 to 100)
                - 'd': %D line Series (0 to 100)
        """
        params = params or {}
        k_period = int(params.get("k_period", 14))
        d_period = int(params.get("d_period", 3))
        smooth_k = int(params.get("smooth_k", 3))

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Highest high and lowest low over k_period
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()

        # Prevent division by zero
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0.0, 1e-10)

        # Raw %K
        raw_k = 100.0 * (close - lowest_low) / range_diff

        # Smoothed %K
        k_line = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()

        # %D is the SMA of %K
        d_line = k_line.rolling(window=d_period, min_periods=d_period).mean()

        return {
            "k": k_line,
            "d": d_line,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate Stochastic parameters.

        Rules:
            - k_period must be between 5 and 50.
            - d_period must be between 2 and 10.
            - smooth_k must be between 1 and 10.
        """
        k_period = params.get("k_period", 14)
        d_period = params.get("d_period", 3)
        smooth_k = params.get("smooth_k", 3)

        if int(k_period) < 5 or int(k_period) > 50:
            return False
        if int(d_period) < 2 or int(d_period) > 10:
            return False
        if int(smooth_k) < 1 or int(smooth_k) > 10:
            return False

        return True
