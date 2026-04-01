"""
Bollinger Bands indicator block.

Bollinger Bands consist of a middle band (SMA) with an upper and lower
band placed a specified number of standard deviations away.

Calculation:
    1. Middle Band = SMA(close, period)
    2. Std Dev     = rolling standard deviation of close over period
    3. Upper Band  = Middle Band + (num_std * Std Dev)
    4. Lower Band  = Middle Band - (num_std * Std Dev)
    5. %B          = (close - Lower Band) / (Upper Band - Lower Band)
    6. Bandwidth   = (Upper Band - Lower Band) / Middle Band * 100

Output columns:
    - bb_upper: Upper band
    - bb_middle: Middle band (SMA)
    - bb_lower: Lower band
    - bb_pct_b: %B (percent B)
    - bb_bandwidth: Bandwidth percentage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class BollingerBandsIndicator(BaseBlock):
    """Bollinger Bands volatility and mean-reversion indicator."""

    metadata = BlockMetadata(
        name="bollinger_bands",
        display_name="Bollinger Bands",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates Bollinger Bands: a middle SMA band with upper and "
            "lower bands at a configurable number of standard deviations. "
            "Also computes %B (relative position within bands) and "
            "Bandwidth (band width as a percentage of the middle band)."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=20,
                min_value=2,
                max_value=500,
                description="Lookback period for the middle band SMA and standard deviation",
            ),
            ParameterSpec(
                name="num_std",
                param_type="float",
                default=2.0,
                min_value=0.5,
                max_value=5.0,
                description="Number of standard deviations for upper/lower bands",
            ),
        ),
        tags=(
            "volatility", "bands", "mean_reversion", "squeeze",
            "standard_deviation", "overbought", "oversold",
        ),
        typical_use=(
            "Used for mean-reversion entries when price touches or exceeds "
            "the bands, volatility squeeze detection (narrow bandwidth "
            "often precedes breakouts), and trend-following when price walks "
            "along the upper or lower band."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        period: int = params["period"]
        num_std: float = params["num_std"]

        if len(data) < period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std(ddof=0)
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        band_width_raw = upper - lower
        pct_b = (close - lower) / band_width_raw.replace(0, np.nan)
        bandwidth = (band_width_raw / middle.replace(0, np.nan)) * 100.0

        result = pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "bb_pct_b": pct_b,
                "bb_bandwidth": bandwidth,
            },
            index=data.index,
        )

        return result
