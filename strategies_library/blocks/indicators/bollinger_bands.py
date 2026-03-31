"""
Bollinger Bands Indicator Block.

Bollinger Bands consist of three lines:
    - Middle Band: Simple Moving Average (SMA) of close prices
    - Upper Band: Middle Band + (std_dev × standard deviation)
    - Lower Band: Middle Band - (std_dev × standard deviation)

Interpretation:
    - Price near upper band: Potentially overbought / strong momentum
    - Price near lower band: Potentially oversold / weak momentum
    - Band squeeze (bands narrowing): Low volatility, breakout expected
    - Band expansion: High volatility

Default Parameters:
    period: 20
    std_dev: 2.0
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class BollingerBands(BaseBlock):
    """Bollinger Bands volatility and mean reversion indicator."""

    metadata = BlockMetadata(
        name="BollingerBands",
        category="indicator",
        description="Bollinger Bands — volatility bands around a moving average",
        complexity=3,
        typical_use=["mean_reversion", "volatility_breakout", "squeeze"],
        required_columns=["close"],
        version="1.0.0",
        tags=["volatility", "bands", "mean_reversion", "squeeze"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute Bollinger Bands.

        Args:
            data: OHLCV DataFrame with a 'close' column.
            params: Optional dict with:
                - period (int): SMA lookback period. Default 20.
                - std_dev (float): Number of standard deviations. Default 2.0.

        Returns:
            Dict with keys:
                - 'upper': Upper band Series
                - 'middle': Middle band (SMA) Series
                - 'lower': Lower band Series
                - 'bandwidth': (upper - lower) / middle — measures squeeze
        """
        params = params or {}
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))

        close = data["close"]

        middle = close.rolling(window=period, min_periods=period).mean()
        rolling_std = close.rolling(window=period, min_periods=period).std()

        upper = middle + std_dev * rolling_std
        lower = middle - std_dev * rolling_std

        # Bandwidth: measures how wide the bands are relative to the middle
        # Low bandwidth = squeeze (consolidation)
        # High bandwidth = expansion (trending/volatile)
        bandwidth = (upper - lower) / middle

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": bandwidth,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate Bollinger Bands parameters.

        Rules:
            - period must be between 5 and 200.
            - std_dev must be between 0.5 and 4.0.
        """
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2.0)

        if not isinstance(period, (int, float)):
            return False
        if int(period) < 5 or int(period) > 200:
            return False
        if not isinstance(std_dev, (int, float)):
            return False
        if float(std_dev) < 0.5 or float(std_dev) > 4.0:
            return False

        return True
