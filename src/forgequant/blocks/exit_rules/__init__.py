"""
Exit rule blocks.

Provides:
    - FixedTPSLExit: Fixed take-profit and stop-loss in ATR multiples or pips
    - TrailingStopExit: Trailing stop that locks in profits as price moves
    - TimeBasedExit: Exit after a maximum number of bars
    - BreakevenStopExit: Moves stop to breakeven after reaching a profit threshold

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.exit_rules.fixed_tpsl import FixedTPSLExit
from forgequant.blocks.exit_rules.trailing_stop import TrailingStopExit
from forgequant.blocks.exit_rules.time_based_exit import TimeBasedExit
from forgequant.blocks.exit_rules.breakeven_stop import BreakevenStopExit

__all__ = [
    "FixedTPSLExit",
    "TrailingStopExit",
    "TimeBasedExit",
    "BreakevenStopExit",
]
