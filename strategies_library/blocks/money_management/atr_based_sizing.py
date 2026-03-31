from typing import Dict, Any
import pandas as pd
from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class ATRBasedSizing(BaseBlock):
    """ATR-Based Position Sizing.

    Uses current ATR to determine stop loss distance, then calculates
    position size to risk a fixed dollar amount or percentage.

    Volume = risk_amount / (atr_multiplier * atr * pip_value)

    This naturally adjusts position size to current volatility conditions.

    Default Parameters:
        atr_period: 14
        atr_multiplier: 2.0
        risk_percent: 1.0
    """

    metadata = BlockMetadata(
        name="ATRBasedSizing",
        category="money_management",
        description="Position sizing based on ATR for dynamic risk adjustment",
        complexity=3,
        typical_use=["risk_management", "volatility_adjusted"],
        required_columns=["high", "low", "close"],
        tags=["atr", "dynamic_sizing"]
    )

    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> dict:
        params = params or {}
        atr_period = int(params.get("atr_period", 14))
        atr_multiplier = float(params.get("atr_multiplier", 2.0))
        risk_percent = float(params.get("risk_percent", 1.0))

        # Compute ATR
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift(1))
        low_close = abs(data["low"] - data["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()

        # Stop distance in price units
        stop_distance = atr * atr_multiplier

        return {
            "atr": atr,
            "stop_distance": stop_distance,
            "risk_percent": risk_percent / 100.0,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        atr_period = params.get("atr_period", 14)
        atr_multiplier = params.get("atr_multiplier", 2.0)
        risk_percent = params.get("risk_percent", 1.0)
        if atr_period < 5 or atr_period > 50:
            return False
        if atr_multiplier < 0.5 or atr_multiplier > 5.0:
            return False
        if risk_percent <= 0 or risk_percent > 5.0:
            return False
        return True
