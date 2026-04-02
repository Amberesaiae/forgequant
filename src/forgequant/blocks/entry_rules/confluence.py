"""
Confluence entry rule block.

Generates entry signals only when multiple independent conditions are
simultaneously true. Acts as a logical AND gate across conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks._utils import _compute_atr, _compute_rsi
from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult, SIGNAL_COLUMNS as SC


@BlockRegistry.register
class ConfluenceEntry(BaseBlock):
    """Multi-condition confluence entry signals."""

    metadata = BlockMetadata(
        name="confluence_entry",
        display_name="Confluence Entry",
        category=BlockCategory.ENTRY_RULE,
        description=(
            "Generates entry signals only when trend alignment, momentum "
            "confirmation, and volatility conditions are all met. Acts as "
            "a quality gate to filter out low-conviction setups."
        ),
        parameters=(
            ParameterSpec(
                name="trend_period",
                param_type="int",
                default=50,
                min_value=5,
                max_value=500,
                description="EMA period for trend alignment",
            ),
            ParameterSpec(
                name="rsi_period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="RSI period for momentum confirmation",
            ),
            ParameterSpec(
                name="rsi_long_min",
                param_type="float",
                default=40.0,
                min_value=5.0,
                max_value=80.0,
                description="Minimum RSI for long entry (momentum not too weak)",
            ),
            ParameterSpec(
                name="rsi_long_max",
                param_type="float",
                default=70.0,
                min_value=20.0,
                max_value=95.0,
                description="Maximum RSI for long entry (not already overbought)",
            ),
            ParameterSpec(
                name="rsi_short_min",
                param_type="float",
                default=30.0,
                min_value=5.0,
                max_value=80.0,
                description="Minimum RSI for short entry (not already oversold)",
            ),
            ParameterSpec(
                name="rsi_short_max",
                param_type="float",
                default=60.0,
                min_value=20.0,
                max_value=95.0,
                description="Maximum RSI for short entry (momentum not too strong)",
            ),
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR period for volatility measurement",
            ),
            ParameterSpec(
                name="min_atr_pct",
                param_type="float",
                default=0.05,
                min_value=0.0,
                max_value=10.0,
                description="Minimum ATR as % of close (avoid dead markets)",
            ),
            ParameterSpec(
                name="max_atr_pct",
                param_type="float",
                default=5.0,
                min_value=0.1,
                max_value=50.0,
                description="Maximum ATR as % of close (avoid excessive volatility)",
            ),
        ),
        tags=("confluence", "multi_condition", "quality", "filter", "entry"),
        typical_use=(
            "Used as a high-quality entry filter that requires trend, "
            "momentum, and volatility to all agree before taking a trade. "
            "Reduces false signals compared to single-condition entries."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        trend_period: int = params["trend_period"]
        rsi_period: int = params["rsi_period"]
        rsi_long_min: float = params["rsi_long_min"]
        rsi_long_max: float = params["rsi_long_max"]
        rsi_short_min: float = params["rsi_short_min"]
        rsi_short_max: float = params["rsi_short_max"]
        atr_period: int = params["atr_period"]
        min_atr_pct: float = params["min_atr_pct"]
        max_atr_pct: float = params["max_atr_pct"]

        min_rows = max(trend_period, rsi_period, atr_period) + 2
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]
        high = data["high"]
        low = data["low"]

        trend_ema = close.ewm(span=trend_period, adjust=False).mean()
        trend_bullish = close > trend_ema
        trend_bearish = close < trend_ema

        rsi = _compute_rsi(data, rsi_period)

        momentum_long = (rsi >= rsi_long_min) & (rsi <= rsi_long_max)
        momentum_short = (rsi >= rsi_short_min) & (rsi <= rsi_short_max)

        atr = _compute_atr(data, atr_period)
        atr_pct = (atr / close) * 100.0

        volatility_ok = (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)

        long_entry = trend_bullish & momentum_long & volatility_ok
        short_entry = trend_bearish & momentum_short & volatility_ok

        score = (
            trend_bullish.astype(int)
            + momentum_long.astype(int)
            + volatility_ok.astype(int)
        )

        result = pd.DataFrame(
            {
                "confluence_trend_ok": trend_bullish.fillna(False),
                "confluence_momentum_ok": momentum_long.fillna(False),
                "confluence_volatility_ok": volatility_ok.fillna(False),
                SC.confluence_long_entry: long_entry.fillna(False),
                SC.confluence_short_entry: short_entry.fillna(False),
                "confluence_score": score.fillna(0).astype(int),
            },
            index=data.index,
        )

        return result
