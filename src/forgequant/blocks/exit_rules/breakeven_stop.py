"""
Breakeven Stop exit rule block.

After price moves a configurable distance in the profitable direction,
the stop-loss is moved to the entry price plus an optional small offset.
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class BreakevenStopExit(BaseBlock):
    """Breakeven stop that activates after a profit threshold is reached."""

    metadata = BlockMetadata(
        name="breakeven_stop",
        display_name="Breakeven Stop",
        category=BlockCategory.EXIT_RULE,
        description=(
            "Moves the stop-loss to breakeven (entry price + offset) after "
            "price moves a configurable distance in the profitable direction. "
            "The activation threshold and offset are expressed as ATR "
            "multiples for volatility adaptation."
        ),
        parameters=(
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="activation_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description=(
                    "Distance (in ATR multiples) price must move in profit "
                    "before the breakeven stop activates"
                ),
            ),
            ParameterSpec(
                name="offset_atr_mult",
                param_type="float",
                default=0.1,
                min_value=0.0,
                max_value=5.0,
                description=(
                    "Offset (in ATR multiples) added to the entry price for "
                    "the breakeven stop. Covers spread and commissions. "
                    "Set to 0 for exact breakeven."
                ),
            ),
        ),
        tags=("exit", "breakeven", "stop_loss", "atr", "risk_management"),
        typical_use=(
            "Used after a FixedTPSL or TrailingStop to eliminate risk on "
            "a trade once it shows sufficient profit. Activation at 1.5x ATR "
            "means the trade must first move 1.5 ATR in your favor before "
            "the stop moves to breakeven."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        atr_period: int = params["atr_period"]
        activation_mult: float = params["activation_atr_mult"]
        offset_mult: float = params["offset_atr_mult"]

        min_rows = atr_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        long_activation = close + (activation_mult * atr)
        short_activation = close - (activation_mult * atr)

        long_be_stop = close + (offset_mult * atr)
        short_be_stop = close - (offset_mult * atr)

        long_activated = high >= long_activation
        short_activated = low <= short_activation

        result = pd.DataFrame(
            {
                "be_atr": atr,
                "be_long_activation": long_activation,
                "be_long_stop": long_be_stop,
                "be_short_activation": short_activation,
                "be_short_stop": short_be_stop,
                "be_long_activated": long_activated.fillna(False),
                "be_short_activated": short_activated.fillna(False),
            },
            index=data.index,
        )

        return result
