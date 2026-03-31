"""
ForgeQuant Entry Rule Building Blocks.

4 entry blocks covering:
- Crossover: Moving average crossover signals
- ThresholdCross: Indicator crosses above/below a level
- Confluence: Multiple conditions must align simultaneously
- ReversalPattern: Candlestick reversal pattern detection
"""

from .crossover import Crossover
from .threshold_cross import ThresholdCross
from .confluence import Confluence
from .reversal_pattern import ReversalPattern

__all__ = [
    "Crossover",
    "ThresholdCross",
    "Confluence",
    "ReversalPattern",
]
