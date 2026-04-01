"""
Stochastic Oscillator indicator block.

The Stochastic Oscillator compares a closing price to its price range
over a given lookback period.

Calculation:
    1. %K (raw):
       %K = 100 * (close - lowest_low(k_period)) / (highest_high(k_period) - lowest_low(k_period))
    2. %K (smoothed):
       %K smoothed = SMA(%K raw, k_smooth)
    3. %D (signal):
       %D = SMA(%K smoothed, d_period)

Output columns:
    - stoch_k: The %K line (smoothed)
    - stoch_d: The %D line (signal)
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
class StochasticIndicator(BaseBlock):
    """Stochastic Oscillator momentum indicator."""

    metadata = BlockMetadata(
        name="stochastic",
        display_name="Stochastic Oscillator",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Stochastic Oscillator, a momentum indicator "
            "that compares the closing price to its range over a lookback "
            "period. %K above 80 is typically considered overbought, and "
            "below 20 is considered oversold."
        ),
        parameters=(
            ParameterSpec(
                name="k_period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="Lookback period for %K calculation",
            ),
            ParameterSpec(
                name="k_smooth",
                param_type="int",
                default=3,
                min_value=1,
                max_value=50,
                description="Smoothing period for %K (1 = Fast Stochastic, 3 = Slow)",
            ),
            ParameterSpec(
                name="d_period",
                param_type="int",
                default=3,
                min_value=1,
                max_value=50,
                description="Period for the %D signal line (SMA of %K)",
            ),
            ParameterSpec(
                name="overbought",
                param_type="float",
                default=80.0,
                min_value=50.0,
                max_value=95.0,
                description="Overbought threshold level",
            ),
            ParameterSpec(
                name="oversold",
                param_type="float",
                default=20.0,
                min_value=5.0,
                max_value=50.0,
                description="Oversold threshold level",
            ),
        ),
        tags=("momentum", "oscillator", "overbought", "oversold", "mean_reversion"),
        typical_use=(
            "Used for mean-reversion entries in ranging markets (buy oversold, "
            "sell overbought), %K/%D crossover signals, and divergence "
            "detection. Often combined with a trend filter — only take "
            "oversold signals in uptrends and overbought signals in downtrends."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        k_period: int = params["k_period"]
        k_smooth: int = params["k_smooth"]
        d_period: int = params["d_period"]
        overbought: float = params["overbought"]
        oversold: float = params["oversold"]

        if overbought <= oversold:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Overbought ({overbought}) must be greater than "
                       f"oversold ({oversold})",
            )

        min_rows = k_period + k_smooth + d_period
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        range_hl = highest_high - lowest_low
        k_raw = 100.0 * (close - lowest_low) / range_hl.replace(0, np.nan)

        k_smooth_series = k_raw.rolling(window=k_smooth).mean()
        d_series = k_smooth_series.rolling(window=d_period).mean()

        result = pd.DataFrame(
            {
                "stoch_k": k_smooth_series,
                "stoch_d": d_series,
                "stoch_overbought": k_smooth_series >= overbought,
                "stoch_oversold": k_smooth_series <= oversold,
            },
            index=data.index,
        )

        return result
