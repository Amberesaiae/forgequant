"""
Compiled strategy data container.

A CompiledStrategy holds all the intermediate and final signal
DataFrames produced by running a StrategySpec's blocks on OHLCV data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from forgequant.ai_forge.schemas import StrategySpec


@dataclass
class BlockOutput:
    """Output from a single block execution."""

    block_name: str
    category: str
    params: dict[str, Any]
    result: pd.DataFrame | pd.Series


@dataclass
class CompiledStrategy:
    """A fully compiled strategy ready for backtesting."""

    spec: StrategySpec
    ohlcv: pd.DataFrame
    block_outputs: dict[str, BlockOutput] = field(default_factory=dict)

    entry_long: pd.Series | None = None
    entry_short: pd.Series | None = None
    exit_long: pd.Series | None = None
    exit_short: pd.Series | None = None
    allow_long: pd.Series | None = None
    allow_short: pd.Series | None = None
    position_size_long: pd.Series | None = None
    position_size_short: pd.Series | None = None
    stop_loss_long: pd.Series | None = None
    stop_loss_short: pd.Series | None = None
    take_profit_long: pd.Series | None = None
    take_profit_short: pd.Series | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.ohlcv.index  # type: ignore[return-value]

    @property
    def close(self) -> pd.Series:
        return self.ohlcv["close"]

    @property
    def n_bars(self) -> int:
        return len(self.ohlcv)

    def filtered_entry_long(self) -> pd.Series:
        if self.entry_long is None:
            return pd.Series(False, index=self.index)
        entry = self.entry_long.copy()
        if self.allow_long is not None:
            entry = entry & self.allow_long
        return entry

    def filtered_entry_short(self) -> pd.Series:
        if self.entry_short is None:
            return pd.Series(False, index=self.index)
        entry = self.entry_short.copy()
        if self.allow_short is not None:
            entry = entry & self.allow_short
        return entry

    def get_block_output(self, block_name: str) -> BlockOutput | None:
        return self.block_outputs.get(block_name)

    def summary(self) -> dict[str, Any]:
        def _count(s: pd.Series | None) -> int:
            if s is None:
                return 0
            return int(s.sum()) if s.dtype == bool else 0

        return {
            "strategy_name": self.spec.name,
            "n_bars": self.n_bars,
            "n_blocks": len(self.block_outputs),
            "raw_long_entries": _count(self.entry_long),
            "raw_short_entries": _count(self.entry_short),
            "filtered_long_entries": _count(self.filtered_entry_long()),
            "filtered_short_entries": _count(self.filtered_entry_short()),
            "long_exits": _count(self.exit_long),
            "short_exits": _count(self.exit_short),
        }
