"""
Ichimoku Cloud (Ichimoku Kinko Hyo) Indicator Block.

A comprehensive indicator that defines support/resistance, identifies
trend direction, gauges momentum, and provides trading signals.

Components:
    - Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 over tenkan_period
    - Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 over kijun_period
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 'displacement' bars ahead
    - Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 over senkou_b_period,
                                       plotted 'displacement' bars ahead
    - Chikou Span (Lagging Span): Close price plotted 'displacement' bars behind

The area between Senkou A and Senkou B forms the "cloud" (kumo).

Interpretation:
    - Price above cloud: Bullish
    - Price below cloud: Bearish
    - Price inside cloud: No clear trend
    - Tenkan crosses above Kijun above cloud: Strong buy
    - Tenkan crosses below Kijun below cloud: Strong sell

Default Parameters:
    tenkan_period: 9
    kijun_period: 26
    senkou_b_period: 52
    displacement: 26
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class Ichimoku(BaseBlock):
    """Ichimoku Cloud indicator — comprehensive trend analysis system."""

    metadata = BlockMetadata(
        name="Ichimoku",
        category="indicator",
        description="Ichimoku Cloud — all-in-one trend, momentum, support/resistance system",
        complexity=5,
        typical_use=["trend_following", "support_resistance", "momentum"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["trend", "cloud", "support", "resistance", "japanese"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute all five Ichimoku components.

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns.
            params: Optional dict with:
                - tenkan_period (int): Conversion line period. Default 9.
                - kijun_period (int): Base line period. Default 26.
                - senkou_b_period (int): Leading Span B period. Default 52.
                - displacement (int): Number of bars to shift leading/lagging spans. Default 26.

        Returns:
            Dict with keys:
                - 'tenkan': Tenkan-sen (Conversion Line) Series
                - 'kijun': Kijun-sen (Base Line) Series
                - 'senkou_a': Senkou Span A Series (shifted forward)
                - 'senkou_b': Senkou Span B Series (shifted forward)
                - 'chikou': Chikou Span Series (shifted backward)
        """
        params = params or {}
        tenkan_period = int(params.get("tenkan_period", 9))
        kijun_period = int(params.get("kijun_period", 26))
        senkou_b_period = int(params.get("senkou_b_period", 52))
        displacement = int(params.get("displacement", 26))

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period, min_periods=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2.0

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period, min_periods=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period, min_periods=kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2.0

        # Senkou Span A (Leading Span A) — shifted forward
        senkou_a = ((tenkan + kijun) / 2.0).shift(displacement)

        # Senkou Span B (Leading Span B) — shifted forward
        senkou_b_high = high.rolling(window=senkou_b_period, min_periods=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period, min_periods=senkou_b_period).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2.0).shift(displacement)

        # Chikou Span (Lagging Span) — shifted backward
        chikou = close.shift(-displacement)

        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate Ichimoku parameters.

        Rules:
            - tenkan_period must be less than kijun_period.
            - kijun_period must be less than senkou_b_period.
            - All periods must be positive and within reasonable ranges.
            - displacement must be positive.
        """
        tenkan = int(params.get("tenkan_period", 9))
        kijun = int(params.get("kijun_period", 26))
        senkou_b = int(params.get("senkou_b_period", 52))
        displacement = int(params.get("displacement", 26))

        if tenkan >= kijun:
            return False
        if kijun >= senkou_b:
            return False
        if tenkan < 2 or tenkan > 50:
            return False
        if kijun < 5 or kijun > 100:
            return False
        if senkou_b < 10 or senkou_b > 200:
            return False
        if displacement < 1 or displacement > 100:
            return False

        return True
