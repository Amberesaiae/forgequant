"""
Price action pattern blocks.

Provides:
    - BreakoutBlock: Detects breakouts above/below recent highs/lows
    - PullbackBlock: Detects pullbacks to moving averages or key levels
    - HigherHighLowerLowBlock: Identifies higher highs, higher lows (and vice versa)
    - SupportResistanceBlock: Identifies horizontal support and resistance zones

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.price_action.breakout import BreakoutBlock
from forgequant.blocks.price_action.pullback import PullbackBlock
from forgequant.blocks.price_action.higher_high_lower_low import HigherHighLowerLowBlock
from forgequant.blocks.price_action.support_resistance import SupportResistanceBlock

__all__ = [
    "BreakoutBlock",
    "PullbackBlock",
    "HigherHighLowerLowBlock",
    "SupportResistanceBlock",
]
