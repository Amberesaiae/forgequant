"""
ForgeQuant Price Action Building Blocks.

4 price action blocks covering:
- Breakout: Price breaks above/below recent extremes
- Pullback: Price retraces to support within a trend
- HigherHighLowerLow: Trend structure analysis via swing points
- SupportResistance: Dynamic support and resistance level detection

All blocks return either a boolean pd.Series (condition met or not)
or a dict containing multiple Series for complex analysis.
"""

from .breakout import Breakout
from .pullback import Pullback
from .higher_high_lower_low import HigherHighLowerLow
from .support_resistance import SupportResistance

__all__ = [
    "Breakout",
    "Pullback",
    "HigherHighLowerLow",
    "SupportResistance",
]
