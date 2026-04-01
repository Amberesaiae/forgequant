"""
Kelly Fractional position sizing block.

Uses the Kelly Criterion with fractional scaling to determine optimal
position size based on rolling estimates of win rate and payoff ratio.
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
class KellyFractionalSizing(BaseBlock):
    """Kelly Criterion with fractional scaling for position sizing."""

    metadata = BlockMetadata(
        name="kelly_fractional",
        display_name="Kelly Fractional Sizing",
        category=BlockCategory.MONEY_MANAGEMENT,
        description=(
            "Sizes positions using a fractional Kelly Criterion based on "
            "rolling estimates of win rate and payoff ratio. Full Kelly "
            "is scaled down by a configurable fraction to reduce volatility."
        ),
        parameters=(
            ParameterSpec(
                name="kelly_fraction",
                param_type="float",
                default=0.25,
                min_value=0.01,
                max_value=1.0,
                description="Fraction of full Kelly to use (0.25 = quarter Kelly)",
            ),
            ParameterSpec(
                name="lookback",
                param_type="int",
                default=100,
                min_value=20,
                max_value=2000,
                description="Rolling window for win rate and payoff estimation",
            ),
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR period for stop distance calculation",
            ),
            ParameterSpec(
                name="sl_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description="Stop-loss distance as ATR multiple",
            ),
            ParameterSpec(
                name="account_equity",
                param_type="float",
                default=100000.0,
                min_value=100.0,
                max_value=1e12,
                description="Account equity in base currency",
            ),
            ParameterSpec(
                name="max_fraction",
                param_type="float",
                default=0.05,
                min_value=0.001,
                max_value=0.5,
                description="Maximum fraction of equity to risk (absolute cap)",
            ),
        ),
        tags=("sizing", "kelly", "optimal", "growth", "adaptive"),
        typical_use=(
            "Used in systems with a known or estimable edge. Quarter Kelly "
            "(0.25) is the most common setting — it sacrifices ~25% of "
            "optimal growth rate but reduces drawdowns by ~50%. Only "
            "use when the strategy has a demonstrable positive expectancy."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        kelly_frac: float = params["kelly_fraction"]
        lookback: int = params["lookback"]
        atr_period: int = params["atr_period"]
        sl_mult: float = params["sl_atr_mult"]
        equity: float = params["account_equity"]
        max_frac: float = params["max_fraction"]

        min_rows = max(lookback, atr_period) + 1
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

        returns = close.pct_change()

        is_win = (returns > 0).astype(float)
        win_rate = is_win.rolling(window=lookback).mean()

        gains = returns.clip(lower=0.0)
        losses = (-returns).clip(lower=0.0)

        avg_win = gains.rolling(window=lookback).mean()
        avg_loss = losses.rolling(window=lookback).mean()

        payoff_ratio = avg_win / avg_loss.replace(0, np.nan)
        payoff_ratio = payoff_ratio.fillna(np.inf)

        full_kelly = win_rate - (1.0 - win_rate) / payoff_ratio.replace(0, np.nan).replace(np.inf, np.nan)
        full_kelly = full_kelly.fillna(win_rate)

        frac_kelly = (full_kelly * kelly_frac).clip(lower=0.0)
        frac_kelly = frac_kelly.clip(upper=max_frac)

        stop_distance = atr * sl_mult
        position_size = (equity * frac_kelly) / stop_distance.replace(0, np.nan)

        result = pd.DataFrame(
            {
                "kelly_win_rate": win_rate,
                "kelly_payoff_ratio": payoff_ratio,
                "kelly_full_fraction": full_kelly,
                "kelly_fraction_used": frac_kelly,
                "kelly_position_size": position_size,
            },
            index=data.index,
        )

        return result
