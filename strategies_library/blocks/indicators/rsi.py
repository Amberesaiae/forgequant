"""
Relative Strength Index (RSI) Indicator Block.

The RSI measures the speed and magnitude of recent price changes
to evaluate overbought or oversold conditions.

Formula:
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss (over 'period' bars)

Interpretation:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    - RSI divergence from price: Potential reversal warning

Default Parameters:
    period: 14
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class RSI(BaseBlock):
    """Relative Strength Index indicator."""

    metadata = BlockMetadata(
        name="RSI",
        category="indicator",
        description="Relative Strength Index — measures overbought and oversold conditions",
        complexity=2,
        typical_use=["mean_reversion", "overbought_oversold", "divergence"],
        required_columns=["close"],
        version="1.0.0",
        tags=["oscillator", "momentum", "mean_reversion"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> pd.Series:
        """Compute RSI on the close price.

        Args:
            data: OHLCV DataFrame with a 'close' column.
            params: Optional dict with:
                - period (int): RSI lookback period. Default 14.

        Returns:
            pd.Series with RSI values ranging from 0 to 100.
            First 'period' values will be NaN due to insufficient data.
        """
        params = params or {}
        period = int(params.get("period", 14))

        delta = data["close"].diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Prevent division by zero
        avg_loss = avg_loss.replace(0.0, 1e-10)

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate RSI parameters.

        Rules:
            - period must be between 2 and 100.
        """
        period = params.get("period", 14)
        if not isinstance(period, (int, float)):
            return False
        if int(period) < 2 or int(period) > 100:
            return False
        return True
