"""
Money management (position sizing) blocks.

Provides:
    - FixedRiskSizing: Fixed percentage risk per trade
    - VolatilityTargetingSizing: Target a specific portfolio volatility
    - KellyFractionalSizing: Kelly criterion with fractional scaling
    - ATRBasedSizing: Position size inversely proportional to ATR

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.money_management.fixed_risk import FixedRiskSizing
from forgequant.blocks.money_management.volatility_targeting import VolatilityTargetingSizing
from forgequant.blocks.money_management.kelly_fractional import KellyFractionalSizing
from forgequant.blocks.money_management.atr_based_sizing import ATRBasedSizing

__all__ = [
    "FixedRiskSizing",
    "VolatilityTargetingSizing",
    "KellyFractionalSizing",
    "ATRBasedSizing",
]
