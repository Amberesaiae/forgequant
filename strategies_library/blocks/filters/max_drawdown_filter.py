from typing import Dict, Any
import pandas as pd
import numpy as np
from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class MaxDrawdownFilter(BaseBlock):
    """Maximum Drawdown Filter.

    Pauses trading when the running drawdown of the equity curve
    exceeds a specified threshold. Acts as a portfolio-level safety net.

    Requires an equity column or computes from close prices as a proxy.

    Returns a boolean Series where True = allowed to trade.

    Default Parameters:
        max_drawdown_pct: 0.08   (8% drawdown limit)
        lookback: 252            (bars to compute running peak)
    """

    metadata = BlockMetadata(
        name="MaxDrawdownFilter",
        category="filter",
        description="Pause trading when drawdown exceeds threshold",
        complexity=3,
        typical_use=["risk_management", "circuit_breaker"],
        required_columns=["close"],
        tags=["drawdown", "safety", "filter"]
    )

    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
        params = params or {}
        max_dd_pct = float(params.get("max_drawdown_pct", 0.08))
        lookback = int(params.get("lookback", 252))

        # Use equity column if available, otherwise use close as proxy
        if "equity" in data.columns:
            equity = data["equity"]
        else:
            equity = data["close"]

        running_peak = equity.rolling(window=lookback, min_periods=1).max()
        drawdown = (equity - running_peak) / running_peak
        
        # True when drawdown is within acceptable limits
        return drawdown > -max_dd_pct

    def validate_params(self, params: Dict[str, Any]) -> bool:
        max_dd = params.get("max_drawdown_pct", 0.08)
        if max_dd <= 0 or max_dd > 0.5:
            return False
        return True
