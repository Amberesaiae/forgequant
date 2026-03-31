from typing import Dict, Any
import pandas as pd
from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class TrendFilter(BaseBlock):
    """Trend Direction Filter.

    Only allows trades in the direction of the prevailing trend.
    Uses a long-period moving average to determine trend direction.

    Returns a dictionary:
    - allow_long: True when price is above the trend MA (uptrend)
    - allow_short: True when price is below the trend MA (downtrend)

    Default Parameters:
        period: 200
        ma_type: 'ema'   (or 'sma')
        buffer_pct: 0.001  (0.1% buffer to avoid whipsaws at the MA)
    """

    metadata = BlockMetadata(
        name="TrendFilter",
        category="filter",
        description="Only allow trades in the direction of the major trend",
        complexity=2,
        typical_use=["trend_following", "direction_filter"],
        required_columns=["close"],
        tags=["trend", "direction", "filter"]
    )

    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> dict:
        params = params or {}
        period = int(params.get("period", 200))
        ma_type = str(params.get("ma_type", "ema"))
        buffer_pct = float(params.get("buffer_pct", 0.001))

        close = data["close"]

        if ma_type == "ema":
            trend_ma = close.ewm(span=period, adjust=False).mean()
        else:
            trend_ma = close.rolling(window=period).mean()

        upper_band = trend_ma * (1 + buffer_pct)
        lower_band = trend_ma * (1 - buffer_pct)

        allow_long = close > upper_band
        allow_short = close < lower_band

        return {
            "allow_long": allow_long,
            "allow_short": allow_short,
            "trend_ma": trend_ma,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        period = params.get("period", 200)
        ma_type = params.get("ma_type", "ema")
        if period < 50 or period > 500:
            return False
        if ma_type not in ["ema", "sma"]:
            return False
        return True
