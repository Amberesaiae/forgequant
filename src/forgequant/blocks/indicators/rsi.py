"""
Relative Strength Index (RSI) indicator block.

The RSI measures the speed and magnitude of recent price changes to
evaluate overbought or oversold conditions.

Calculation (Wilder's smoothed method):
    1. delta = close_t - close_{t-1}
    2. gain = max(delta, 0),  loss = abs(min(delta, 0))
    3. avg_gain = ewm(gain, alpha=1/period)   [Wilder smoothing]
       avg_loss = ewm(loss, alpha=1/period)
    4. RS = avg_gain / avg_loss
    5. RSI = 100 - (100 / (1 + RS))

Wilder's smoothing is equivalent to an EMA with alpha = 1/period
(NOT the standard EMA where alpha = 2/(period+1)). We use pandas
ewm(alpha=1/period, adjust=False) to match this exactly.

Output columns:
    - rsi_{period}: RSI values in [0, 100]
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class RSIIndicator(BaseBlock):
    """Relative Strength Index indicator using Wilder's smoothing method."""

    metadata = BlockMetadata(
        name="rsi",
        display_name="Relative Strength Index",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Relative Strength Index using Wilder's smoothing "
            "method. RSI oscillates between 0 and 100, with readings above 70 "
            "typically considered overbought and below 30 considered oversold."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="Lookback period for RSI calculation",
            ),
            ParameterSpec(
                name="overbought",
                param_type="float",
                default=70.0,
                min_value=50.0,
                max_value=95.0,
                description="Overbought threshold level",
            ),
            ParameterSpec(
                name="oversold",
                param_type="float",
                default=30.0,
                min_value=5.0,
                max_value=50.0,
                description="Oversold threshold level",
            ),
        ),
        tags=("momentum", "oscillator", "overbought", "oversold", "mean_reversion"),
        typical_use=(
            "Used to identify overbought/oversold conditions for mean-reversion "
            "entries, as a divergence signal when price makes new highs/lows "
            "but RSI does not, or as a filter to avoid buying into overbought "
            "conditions in trend-following systems."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        period: int = params["period"]
        overbought: float = params["overbought"]
        oversold: float = params["oversold"]

        if overbought <= oversold:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Overbought ({overbought}) must be greater than "
                       f"oversold ({oversold})",
            )

        if len(data) < period + 1:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period + 1} rows, "
                       f"got {len(data)}",
            )

        delta = data["close"].diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)

        alpha = 1.0 / period
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(100.0)

        col = f"rsi_{period}"
        result = pd.DataFrame(
            {
                col: rsi,
                f"{col}_overbought": rsi >= overbought,
                f"{col}_oversold": rsi <= oversold,
            },
            index=data.index,
        )

        return result
