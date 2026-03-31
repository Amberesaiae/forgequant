"""
Moving Average Convergence Divergence (MACD) Indicator Block.

MACD shows the relationship between two EMAs of a price series.

Components:
    - MACD Line: Fast EMA - Slow EMA
    - Signal Line: EMA of the MACD Line
    - Histogram: MACD Line - Signal Line

Interpretation:
    - MACD crosses above Signal: Bullish signal
    - MACD crosses below Signal: Bearish signal
    - Histogram increasing: Momentum strengthening
    - Histogram decreasing: Momentum weakening
    - Zero line crossover: Trend change confirmation

Default Parameters:
    fast_period: 12
    slow_period: 26
    signal_period: 9
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class MACD(BaseBlock):
    """Moving Average Convergence Divergence indicator."""

    metadata = BlockMetadata(
        name="MACD",
        category="indicator",
        description="MACD — trend following momentum indicator",
        complexity=3,
        typical_use=["trend_following", "momentum", "crossover"],
        required_columns=["close"],
        version="1.0.0",
        tags=["trend", "momentum", "crossover", "histogram"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute MACD, Signal line, and Histogram.

        Args:
            data: OHLCV DataFrame with a 'close' column.
            params: Optional dict with:
                - fast_period (int): Fast EMA period. Default 12.
                - slow_period (int): Slow EMA period. Default 26.
                - signal_period (int): Signal line EMA period. Default 9.

        Returns:
            Dict with keys:
                - 'macd': MACD line Series
                - 'signal': Signal line Series
                - 'histogram': Histogram Series (MACD - Signal)
        """
        params = params or {}
        fast_period = int(params.get("fast_period", 12))
        slow_period = int(params.get("slow_period", 26))
        signal_period = int(params.get("signal_period", 9))

        close = data["close"]

        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate MACD parameters.

        Rules:
            - fast_period must be less than slow_period.
            - fast_period must be between 2 and 100.
            - slow_period must be between 5 and 200.
            - signal_period must be between 2 and 50.
        """
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        signal = params.get("signal_period", 9)

        if int(fast) >= int(slow):
            return False
        if int(fast) < 2 or int(fast) > 100:
            return False
        if int(slow) < 5 or int(slow) > 200:
            return False
        if int(signal) < 2 or int(signal) > 50:
            return False

        return True
