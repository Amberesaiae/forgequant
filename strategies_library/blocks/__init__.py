"""
ForgeQuant Building Blocks.

28 modular blocks organized by category:
- indicators/       (8 blocks): EMA, RSI, ATR, BollingerBands, MACD, ADX, Stochastic, Ichimoku
- price_action/     (4 blocks): Breakout, Pullback, HigherHighLowerLow, SupportResistance
- entry_rules/      (4 blocks): Crossover, ThresholdCross, Confluence, ReversalPattern
- exit_rules/       (4 blocks): FixedTPSL, TrailingStop, TimeBasedExit, BreakevenStop
- money_management/ (4 blocks): FixedRisk, VolatilityTargeting, KellyFractional, ATRBasedSizing
- filters/          (4 blocks): TradingSessionFilter, SpreadFilter, MaxDrawdownFilter, TrendFilter
"""

# Indicators
from .indicators import EMA, RSI, ATR, BollingerBands, MACD, ADX, Stochastic, Ichimoku

# Price Action
from .price_action import Breakout, Pullback, HigherHighLowerLow, SupportResistance

# Entry Rules
from .entry_rules import Crossover, ThresholdCross, Confluence, ReversalPattern

# Exit Rules
from .exit_rules import FixedTPSL, TrailingStop, TimeBasedExit, BreakevenStop

# Money Management
from .money_management import FixedRisk, VolatilityTargeting, KellyFractional, ATRBasedSizing

# Filters
from .filters import TradingSessionFilter, SpreadFilter, MaxDrawdownFilter, TrendFilter

__all__ = [
    # Indicators
    "EMA", "RSI", "ATR", "BollingerBands", "MACD", "ADX", "Stochastic", "Ichimoku",
    # Price Action
    "Breakout", "Pullback", "HigherHighLowerLow", "SupportResistance",
    # Entry Rules
    "Crossover", "ThresholdCross", "Confluence", "ReversalPattern",
    # Exit Rules
    "FixedTPSL", "TrailingStop", "TimeBasedExit", "BreakevenStop",
    # Money Management
    "FixedRisk", "VolatilityTargeting", "KellyFractional", "ATRBasedSizing",
    # Filters
    "TradingSessionFilter", "SpreadFilter", "MaxDrawdownFilter", "TrendFilter",
]
