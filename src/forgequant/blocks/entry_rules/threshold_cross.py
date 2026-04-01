"""
Threshold Cross entry rule block.

Generates entry signals when a computed indicator value crosses
above or below configurable threshold levels. Supports mean-reversion
and momentum modes.
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class ThresholdCrossEntry(BaseBlock):
    """Threshold-crossing entry signals (RSI-based)."""

    metadata = BlockMetadata(
        name="threshold_cross_entry",
        display_name="Threshold Cross Entry",
        category=BlockCategory.ENTRY_RULE,
        description=(
            "Generates entry signals when RSI crosses configurable threshold "
            "levels. Supports mean-reversion mode (buy on oversold recovery) "
            "and momentum mode (buy on overbought entry)."
        ),
        parameters=(
            ParameterSpec(
                name="rsi_period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="RSI calculation period",
            ),
            ParameterSpec(
                name="upper_threshold",
                param_type="float",
                default=70.0,
                min_value=50.0,
                max_value=95.0,
                description="Upper threshold (overbought level)",
            ),
            ParameterSpec(
                name="lower_threshold",
                param_type="float",
                default=30.0,
                min_value=5.0,
                max_value=50.0,
                description="Lower threshold (oversold level)",
            ),
            ParameterSpec(
                name="mode",
                param_type="str",
                default="mean_reversion",
                choices=("mean_reversion", "momentum"),
                description=(
                    "Signal mode: mean_reversion (buy on oversold recovery) "
                    "or momentum (buy on overbought breakout)"
                ),
            ),
        ),
        tags=("threshold", "rsi", "overbought", "oversold", "entry", "signal"),
        typical_use=(
            "Mean-reversion: enter long when RSI recovers from oversold, "
            "enter short when RSI falls from overbought. Momentum: enter "
            "long when RSI surges above overbought (strong trend). Best "
            "combined with a trend or volatility filter."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        rsi_period: int = params["rsi_period"]
        upper: float = params["upper_threshold"]
        lower: float = params["lower_threshold"]
        mode: str = params["mode"]

        if upper <= lower:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"upper_threshold ({upper}) must be greater than "
                       f"lower_threshold ({lower})",
            )

        min_rows = rsi_period + 2
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        alpha = 1.0 / rsi_period
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(50.0)

        prev_rsi = rsi.shift(1)

        if mode == "mean_reversion":
            long_entry = (rsi >= lower) & (prev_rsi < lower)
            short_entry = (rsi <= upper) & (prev_rsi > upper)
        else:
            long_entry = (rsi >= upper) & (prev_rsi < upper)
            short_entry = (rsi <= lower) & (prev_rsi > lower)

        result = pd.DataFrame(
            {
                "threshold_indicator": rsi,
                "threshold_long_entry": long_entry.fillna(False),
                "threshold_short_entry": short_entry.fillna(False),
            },
            index=data.index,
        )

        return result
