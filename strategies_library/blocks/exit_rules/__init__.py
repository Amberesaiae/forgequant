"""
ForgeQuant Exit Rule Building Blocks.

4 exit blocks covering:
- FixedTPSL: Fixed take profit and stop loss in pips
- TrailingStop: ATR-based dynamic trailing stop
- TimeBasedExit: Force exit after N bars
- BreakevenStop: Move stop to entry after specified profit
"""

from .fixed_tp_sl import FixedTPSL
from .trailing_stop import TrailingStop
from .time_based_exit import TimeBasedExit
from .breakeven_stop import BreakevenStop

__all__ = [
    "FixedTPSL",
    "TrailingStop",
    "TimeBasedExit",
    "BreakevenStop",
]
