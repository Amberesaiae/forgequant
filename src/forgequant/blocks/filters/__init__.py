"""
Filter blocks.

Provides:
    - TradingSessionFilter: Restricts trading to specific time windows
    - SpreadFilter: Filters out bars where spread is too wide
    - MaxDrawdownFilter: Halts trading when drawdown exceeds a threshold
    - TrendFilter: Only allows trades aligned with the broader trend

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.filters.trading_session import TradingSessionFilter
from forgequant.blocks.filters.spread_filter import SpreadFilter
from forgequant.blocks.filters.max_drawdown_filter import MaxDrawdownFilter
from forgequant.blocks.filters.trend_filter import TrendFilter

__all__ = [
    "TradingSessionFilter",
    "SpreadFilter",
    "MaxDrawdownFilter",
    "TrendFilter",
]
