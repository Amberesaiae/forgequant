"""
ForgeQuant Pre-built Strategy Templates.

Composed from building blocks, these templates provide ready-to-use
strategy configurations that can be backtested, optimized, and deployed.

Each template is a dict specifying which blocks to use and their parameters.
"""

from .trend_follower import TREND_FOLLOWER
from .mean_reversion import MEAN_REVERSION
from .breakout_momentum import BREAKOUT_MOMENTUM
from .scalper import SCALPER

__all__ = [
    "TREND_FOLLOWER",
    "MEAN_REVERSION",
    "BREAKOUT_MOMENTUM",
    "SCALPER",
]
