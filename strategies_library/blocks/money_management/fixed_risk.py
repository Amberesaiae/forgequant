from typing import Dict, Any
import pandas as pd
from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class FixedRisk(BaseBlock):
    """Fixed Percentage Risk Per Trade.

    The simplest and most common money management approach.
    Risks a fixed percentage of account equity on each trade.

    Volume is calculated as:
        volume = (equity * risk_percent) / (sl_pips * pip_value)

    Returns:
    - risk_percent: The fixed risk percentage
    - max_risk_per_trade: Maximum dollar risk per trade (based on equity input)

    Default Parameters:
        risk_percent: 1.0   (1% of equity per trade)
        min_volume: 0.01
        max_volume: 10.0
    """

    metadata = BlockMetadata(
        name="FixedRisk",
        category="money_management",
        description="Fixed percentage risk per trade",
        complexity=1,
        typical_use=["risk_management"],
        required_columns=[],
        tags=["position_sizing", "fixed_risk"]
    )

    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> dict:
        params = params or {}
        risk_percent = float(params.get("risk_percent", 1.0))
        min_volume = float(params.get("min_volume", 0.01))
        max_volume = float(params.get("max_volume", 10.0))

        return {
            "risk_percent": risk_percent / 100.0,
            "min_volume": min_volume,
            "max_volume": max_volume,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        risk = params.get("risk_percent", 1.0)
        if risk <= 0 or risk > 5.0:
            return False
        return True
