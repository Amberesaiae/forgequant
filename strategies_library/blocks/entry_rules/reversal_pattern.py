"""
Candlestick Reversal Pattern Detection Block.

Detects common reversal candlestick patterns:
    - Engulfing (bullish and bearish)
    - Pin Bar / Hammer
    - Doji at extremes

These patterns suggest a potential change in trend direction.

Implementation details:
    - Body = abs(close - open)
    - Candle range = high - low
    - Upper wick = high - max(open, close)
    - Lower wick = min(open, close) - low

Bullish Engulfing:
    Previous bar is bearish (open > close), current bar is bullish (close > open),
    current body engulfs the previous body.

Pin Bar / Hammer:
    Small body with a long lower wick (bullish) or long upper wick (bearish),
    indicating rejection of a price level.

Doji:
    Very small body relative to the candle range, indicating indecision.

Default Parameters:
    pattern_type: 'all'  ('engulfing', 'pin_bar', 'doji', or 'all')
    body_ratio: 0.3
    wick_ratio: 2.5
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class ReversalPattern(BaseBlock):
    """Candlestick reversal pattern detection."""

    metadata = BlockMetadata(
        name="ReversalPattern",
        category="entry",
        description="Detects bullish and bearish candlestick reversal patterns",
        complexity=3,
        typical_use=["mean_reversion", "reversal", "pattern_trading"],
        required_columns=["open", "high", "low", "close"],
        version="1.0.0",
        tags=["candlestick", "reversal", "pattern", "engulfing", "pin_bar", "doji"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Detect reversal patterns.

        Args:
            data: OHLCV DataFrame with 'open', 'high', 'low', 'close' columns.
            params: Optional dict with:
                - pattern_type (str): 'engulfing', 'pin_bar', 'doji', or 'all'. Default 'all'.
                - body_ratio (float): Min body/range ratio for engulfing. Default 0.3.
                - wick_ratio (float): Min wick/body ratio for pin bars. Default 2.5.

        Returns:
            Dict with keys:
                - 'bullish_reversal': Boolean Series (True on bullish reversal patterns)
                - 'bearish_reversal': Boolean Series (True on bearish reversal patterns)
        """
        params = params or {}
        pattern_type = str(params.get("pattern_type", "all"))
        body_ratio = float(params.get("body_ratio", 0.3))
        wick_ratio = float(params.get("wick_ratio", 2.5))

        open_price = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Core candle components
        body = (close - open_price).abs()
        candle_range = high - low
        candle_range_safe = candle_range.replace(0, 1e-10)

        # Wick calculations
        candle_max = pd.concat([close, open_price], axis=1).max(axis=1)
        candle_min = pd.concat([close, open_price], axis=1).min(axis=1)
        upper_wick = high - candle_max
        lower_wick = candle_min - low

        # Safe body for division
        body_safe = body.replace(0, 1e-10)

        # Initialize signals
        bullish = pd.Series(False, index=data.index)
        bearish = pd.Series(False, index=data.index)

        # Engulfing Pattern
        if pattern_type in ["engulfing", "all"]:
            prev_bearish = open_price.shift(1) > close.shift(1)
            curr_bullish = close > open_price
            body_engulfs_up = (close > open_price.shift(1)) & (open_price < close.shift(1))
            good_body = (body / candle_range_safe) >= body_ratio
            bullish_engulfing = prev_bearish & curr_bullish & body_engulfs_up & good_body
            bullish = bullish | bullish_engulfing

            prev_bullish = close.shift(1) > open_price.shift(1)
            curr_bearish = open_price > close
            body_engulfs_down = (open_price > close.shift(1)) & (close < open_price.shift(1))
            bearish_engulfing = prev_bullish & curr_bearish & body_engulfs_down & good_body
            bearish = bearish | bearish_engulfing

        # Pin Bar / Hammer
        if pattern_type in ["pin_bar", "all"]:
            small_body = body < candle_range_safe * 0.35
            bullish_pin = small_body & (lower_wick >= body_safe * wick_ratio) & (upper_wick < body)
            bearish_pin = small_body & (upper_wick >= body_safe * wick_ratio) & (lower_wick < body)
            bullish = bullish | bullish_pin
            bearish = bearish | bearish_pin

        # Doji
        if pattern_type in ["doji", "all"]:
            is_doji = body <= candle_range_safe * 0.1
            # Doji is ambiguous — assign to both
            bullish = bullish | is_doji
            bearish = bearish | is_doji

        return {
            "bullish_reversal": bullish,
            "bearish_reversal": bearish,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate ReversalPattern parameters.

        Rules:
            - pattern_type must be one of 'engulfing', 'pin_bar', 'doji', 'all'.
            - body_ratio must be between 0.1 and 0.8.
            - wick_ratio must be between 1.0 and 5.0.
        """
        pattern_type = params.get("pattern_type", "all")
        body_ratio = params.get("body_ratio", 0.3)
        wick_ratio = params.get("wick_ratio", 2.5)

        if pattern_type not in ["engulfing", "pin_bar", "doji", "all"]:
            return False
        if float(body_ratio) < 0.1 or float(body_ratio) > 0.8:
            return False
        if float(wick_ratio) < 1.0 or float(wick_ratio) > 5.0:
            return False

        return True
