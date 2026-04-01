"""
Ichimoku Kinko Hyo indicator block.

Ichimoku is a comprehensive indicator system that defines support/resistance,
trend direction, and momentum at a glance.

Components:
    1. Tenkan-sen (Conversion Line):
       (highest_high + lowest_low) / 2  over tenkan_period
    2. Kijun-sen (Base Line):
       (highest_high + lowest_low) / 2  over kijun_period
    3. Senkou Span A (Leading Span A):
       (Tenkan-sen + Kijun-sen) / 2, plotted kijun_period bars ahead
    4. Senkou Span B (Leading Span B):
       (highest_high + lowest_low) / 2 over senkou_b_period,
       plotted kijun_period bars ahead
    5. Chikou Span (Lagging Span):
       Current close, plotted kijun_period bars behind (i.e. shifted back)

Output columns:
    - ichimoku_tenkan: Tenkan-sen (Conversion Line)
    - ichimoku_kijun: Kijun-sen (Base Line)
    - ichimoku_senkou_a: Senkou Span A (shifted forward)
    - ichimoku_senkou_b: Senkou Span B (shifted forward)
    - ichimoku_chikou: Chikou Span (shifted backward)
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class IchimokuIndicator(BaseBlock):
    """Ichimoku Kinko Hyo full indicator system."""

    metadata = BlockMetadata(
        name="ichimoku",
        display_name="Ichimoku Kinko Hyo",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates all five Ichimoku Kinko Hyo components: Tenkan-sen, "
            "Kijun-sen, Senkou Span A, Senkou Span B, and Chikou Span. "
            "The cloud (Kumo) between Senkou Span A and B provides dynamic "
            "support/resistance and trend context."
        ),
        parameters=(
            ParameterSpec(
                name="tenkan_period",
                param_type="int",
                default=9,
                min_value=2,
                max_value=100,
                description="Period for Tenkan-sen (Conversion Line)",
            ),
            ParameterSpec(
                name="kijun_period",
                param_type="int",
                default=26,
                min_value=2,
                max_value=200,
                description="Period for Kijun-sen (Base Line) and displacement",
            ),
            ParameterSpec(
                name="senkou_b_period",
                param_type="int",
                default=52,
                min_value=2,
                max_value=300,
                description="Period for Senkou Span B",
            ),
        ),
        tags=("trend", "cloud", "support", "resistance", "japanese", "comprehensive"),
        typical_use=(
            "Used for trend identification (price vs cloud), entry signals "
            "(Tenkan/Kijun cross), support/resistance levels (cloud edges, "
            "Kijun-sen), and momentum confirmation (Chikou Span vs price)."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        tenkan_period: int = params["tenkan_period"]
        kijun_period: int = params["kijun_period"]
        senkou_b_period: int = params["senkou_b_period"]

        min_rows = senkou_b_period + kijun_period
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]

        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2.0

        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2.0

        senkou_a = ((tenkan + kijun) / 2.0).shift(kijun_period)

        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2.0).shift(kijun_period)

        chikou = close.shift(-kijun_period)

        result = pd.DataFrame(
            {
                "ichimoku_tenkan": tenkan,
                "ichimoku_kijun": kijun,
                "ichimoku_senkou_a": senkou_a,
                "ichimoku_senkou_b": senkou_b,
                "ichimoku_chikou": chikou,
            },
            index=data.index,
        )

        return result
