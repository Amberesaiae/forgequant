"""
Entry rule blocks.

Provides:
    - CrossoverEntry: Moving average crossover entry signals
    - ThresholdCrossEntry: Indicator crosses above/below threshold levels
    - ConfluenceEntry: Requires multiple conditions to be true simultaneously
    - ReversalPatternEntry: Detects candlestick reversal patterns

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.entry_rules.crossover import CrossoverEntry
from forgequant.blocks.entry_rules.threshold_cross import ThresholdCrossEntry
from forgequant.blocks.entry_rules.confluence import ConfluenceEntry
from forgequant.blocks.entry_rules.reversal_pattern import ReversalPatternEntry

__all__ = [
    "CrossoverEntry",
    "ThresholdCrossEntry",
    "ConfluenceEntry",
    "ReversalPatternEntry",
]
