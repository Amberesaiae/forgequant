"""
Moving Average Convergence Divergence (MACD) indicator block.

The MACD shows the relationship between two EMAs of the close price.

Calculation:
    1. MACD line    = EMA(close, fast_period) - EMA(close, slow_period)
    2. Signal line  = EMA(MACD line, signal_period)
    3. Histogram    = MACD line - Signal line

Output columns:
    - macd_line: The MACD line
    - macd_signal: The signal line
    - macd_histogram: The MACD histogram
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class MACDIndicator(BaseBlock):
    """Moving Average Convergence Divergence indicator."""

    metadata = BlockMetadata(
        name="macd",
        display_name="MACD",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Moving Average Convergence Divergence indicator. "
            "The MACD line is the difference between a fast and slow EMA. "
            "The signal line is an EMA of the MACD line. The histogram is "
            "the difference between the MACD and signal lines."
        ),
        parameters=(
            ParameterSpec(
                name="fast_period",
                param_type="int",
                default=12,
                min_value=2,
                max_value=100,
                description="Period for the fast EMA",
            ),
            ParameterSpec(
                name="slow_period",
                param_type="int",
                default=26,
                min_value=2,
                max_value=200,
                description="Period for the slow EMA",
            ),
            ParameterSpec(
                name="signal_period",
                param_type="int",
                default=9,
                min_value=2,
                max_value=100,
                description="Period for the signal line EMA",
            ),
        ),
        tags=("trend", "momentum", "crossover", "histogram", "convergence"),
        typical_use=(
            "Used for trend-following entries via MACD/signal crossovers, "
            "momentum confirmation via histogram direction, and divergence "
            "detection between price and MACD for reversal signals."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        fast_period: int = params["fast_period"]
        slow_period: int = params["slow_period"]
        signal_period: int = params["signal_period"]

        if fast_period >= slow_period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"fast_period ({fast_period}) must be less than "
                       f"slow_period ({slow_period})",
            )

        min_rows = slow_period + signal_period
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - macd_signal

        result = pd.DataFrame(
            {
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
            },
            index=data.index,
        )

        return result
