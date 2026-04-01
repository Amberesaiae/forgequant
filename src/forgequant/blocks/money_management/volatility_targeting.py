"""
Volatility Targeting position sizing block.

Sizes positions to target a specific annualized portfolio volatility.
Position is inversely proportional to realized volatility.
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
class VolatilityTargetingSizing(BaseBlock):
    """Position sizing that targets a specific portfolio volatility."""

    metadata = BlockMetadata(
        name="volatility_targeting",
        display_name="Volatility Targeting",
        category=BlockCategory.MONEY_MANAGEMENT,
        description=(
            "Sizes positions inversely proportional to realized volatility "
            "to target a specific annualized portfolio volatility. When "
            "market volatility increases, position size decreases, and "
            "vice versa."
        ),
        parameters=(
            ParameterSpec(
                name="target_vol",
                param_type="float",
                default=0.15,
                min_value=0.01,
                max_value=1.0,
                description="Target annualized portfolio volatility (0.15 = 15%)",
            ),
            ParameterSpec(
                name="vol_lookback",
                param_type="int",
                default=20,
                min_value=5,
                max_value=500,
                description="Lookback period for realized volatility calculation",
            ),
            ParameterSpec(
                name="annualization_factor",
                param_type="float",
                default=252.0,
                min_value=1.0,
                max_value=100000.0,
                description=(
                    "Number of bars per year for annualization. "
                    "252 for daily, 252*6.5≈1638 for hourly (US equities), "
                    "252*24=6048 for hourly (forex 24h)."
                ),
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
                name="max_leverage",
                param_type="float",
                default=5.0,
                min_value=0.1,
                max_value=100.0,
                description="Maximum allowed leverage (caps position size)",
            ),
        ),
        tags=("sizing", "volatility", "targeting", "risk_parity", "adaptive"),
        typical_use=(
            "Used in risk-parity and adaptive systems. Target 10-20% "
            "annualized vol for conservative portfolios. The max_leverage "
            "cap prevents excessively large positions in low-vol regimes."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        target_vol: float = params["target_vol"]
        vol_lookback: int = params["vol_lookback"]
        ann_factor: float = params["annualization_factor"]
        equity: float = params["account_equity"]
        max_leverage: float = params["max_leverage"]

        min_rows = vol_lookback + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        log_returns = np.log(close / close.shift(1))

        rolling_std = log_returns.rolling(window=vol_lookback).std(ddof=1)
        realized_vol = rolling_std * np.sqrt(ann_factor)

        target_exposure = target_vol / realized_vol.replace(0, np.nan)
        target_exposure = target_exposure.clip(upper=max_leverage)

        position_size = (equity * target_exposure) / close
        position_pct = target_exposure * 100.0

        result = pd.DataFrame(
            {
                "vt_realized_vol": realized_vol,
                "vt_target_exposure": target_exposure,
                "vt_position_size": position_size,
                "vt_position_pct": position_pct,
            },
            index=data.index,
        )

        return result
