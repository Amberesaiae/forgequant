from typing import Dict, Any
import pandas as pd
from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class TradingSessionFilter(BaseBlock):
    """Trading Session Time Filter.

    Restricts trading to specific market sessions when liquidity is highest.
    Returns a boolean Series where True = allowed to trade.

    Predefined sessions (UTC):
    - london: 07:00 - 16:00
    - new_york: 13:00 - 22:00
    - tokyo: 00:00 - 09:00
    - london_ny_overlap: 13:00 - 16:00  (highest liquidity)
    - custom: use start_hour and end_hour params

    Default Parameters:
        session: 'london_ny_overlap'
        start_hour: 13      (for custom session)
        end_hour: 16         (for custom session)
    """

    SESSIONS = {
        "london": (7, 16),
        "new_york": (13, 22),
        "tokyo": (0, 9),
        "london_ny_overlap": (13, 16),
        "sydney": (22, 7),
    }

    metadata = BlockMetadata(
        name="TradingSessionFilter",
        category="filter",
        description="Restrict trading to specific market sessions",
        complexity=2,
        typical_use=["time_filter", "liquidity"],
        required_columns=[],
        tags=["session", "time", "filter"]
    )

    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
        params = params or {}
        session = str(params.get("session", "london_ny_overlap"))

        if session == "custom":
            start_hour = int(params.get("start_hour", 13))
            end_hour = int(params.get("end_hour", 16))
        elif session in self.SESSIONS:
            start_hour, end_hour = self.SESSIONS[session]
        else:
            start_hour, end_hour = self.SESSIONS["london_ny_overlap"]

        hour = pd.to_datetime(data.index).hour

        if start_hour <= end_hour:
            return pd.Series((hour >= start_hour) & (hour < end_hour), index=data.index)
        else:
            # Handles wrap-around (e.g., Sydney 22:00 - 07:00)
            return pd.Series((hour >= start_hour) | (hour < end_hour), index=data.index)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        session = params.get("session", "london_ny_overlap")
        if session not in list(self.SESSIONS.keys()) + ["custom"]:
            return False
        return True
