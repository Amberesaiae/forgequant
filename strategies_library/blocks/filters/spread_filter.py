from typing import Dict, Any
import pandas as pd
from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class SpreadFilter(BaseBlock):
    """Spread Filter.

    Prevents trading when the spread exceeds a specified threshold.
    High spreads eat into profits and indicate low liquidity.

    If spread data is not available in the DataFrame, this filter
    always returns True (allows trading) and logs a warning.

    Default Parameters:
        max_spread_pips: 3.0
    """

    metadata = BlockMetadata(
        name="SpreadFilter",
        category="filter",
        description="Block trading when spread exceeds threshold",
        complexity=1,
        typical_use=["cost_control", "liquidity"],
        required_columns=[],
        tags=["spread", "cost", "filter"]
    )

    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
        params = params or {}
        max_spread_pips = float(params.get("max_spread_pips", 3.0))

        if "spread" in data.columns:
            return data["spread"] <= max_spread_pips
        else:
            # No spread data available — allow all trades
            return pd.Series(True, index=data.index)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        max_spread = params.get("max_spread_pips", 3.0)
        if max_spread <= 0:
            return False
        return True
