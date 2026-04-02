"""
Shared type definitions, enumerations, and type aliases used across ForgeQuant.

These provide a single source of truth for categorical values and common
type shapes, preventing stringly-typed code.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from enum import Enum, unique
from typing import Any

import pandas as pd


# ── Enumerations ─────────────────────────────────────────────────────────────


@unique
class BlockCategory(str, Enum):
    """
    Categories for strategy building blocks.

    Each block belongs to exactly one category, which determines its role
    in strategy assembly and the interface contract it must fulfil.
    """

    INDICATOR = "indicator"
    PRICE_ACTION = "price_action"
    ENTRY_RULE = "entry_rule"
    EXIT_RULE = "exit_rule"
    MONEY_MANAGEMENT = "money_management"
    FILTER = "filter"

    def __str__(self) -> str:
        return self.value


@unique
class TimeFrame(str, Enum):
    """Supported OHLCV bar timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

    def __str__(self) -> str:
        return self.value


@unique
class TradeDirection(str, Enum):
    """Direction of a trade signal or position."""

    LONG = "long"
    SHORT = "short"
    BOTH = "both"

    def __str__(self) -> str:
        return self.value  # pragma: no cover


@unique
class MovingAverageType(str, Enum):
    """Supported moving average calculation methods."""

    SMA = "sma"
    EMA = "ema"

    def __str__(self) -> str:
        return self.value  # pragma: no cover


# ── Type Aliases ─────────────────────────────────────────────────────────────

# Standard OHLCV DataFrame: must have columns [open, high, low, close, volume]
# with a DatetimeIndex.
OHLCVDataFrame = pd.DataFrame

# Parameters passed to block compute() methods.
BlockParams = dict[str, Any]

# Result from a block compute() — typically a DataFrame or Series,
# but some blocks (like exit rules) may return a dict of Series.
BlockResult = pd.DataFrame | pd.Series | dict[str, pd.Series | float]

# Mapping of column name to required dtype for OHLCV validation.
OHLCV_REQUIRED_COLUMNS: dict[str, str] = {
    "open": "float",
    "high": "float",
    "low": "float",
    "close": "float",
    "volume": "float",
}


def validate_ohlcv(df: pd.DataFrame, block_name: str = "unknown") -> pd.DataFrame:
    """
    Validate that a DataFrame conforms to the expected OHLCV shape.

    Checks:
        1. Not empty
        2. Has all required columns (case-insensitive; columns are lowered)
        3. Index is a DatetimeIndex
        4. No fully-null required columns

    Args:
        df: The DataFrame to validate.
        block_name: Name of the calling block (for error messages).

    Returns:
        A normalized copy of the DataFrame with lowercase column names.

    Raises:
        ValueError: If any validation check fails.
    """
    if df.empty:
        raise ValueError(f"[{block_name}] Input DataFrame is empty")

    # Normalize column names to lowercase for consistent access
    df = df.copy()
    df.columns = df.columns.str.lower()

    missing = set(OHLCV_REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{block_name}] Missing required OHLCV columns: {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"[{block_name}] DataFrame index must be a DatetimeIndex, "
            f"got {type(df.index).__name__}"
        )

    for col in OHLCV_REQUIRED_COLUMNS:
        if df[col].isna().all():
            raise ValueError(f"[{block_name}] Column '{col}' is entirely NaN")

    return df


@_dataclass(frozen=True)
class SignalColumns:
    """Typed string constants for the signal contract between blocks and the assembler.

    Single source of truth for column names that the signal assembler recognizes.
    """

    # ── Entry long ──────────────────────────────────────────────────
    crossover_long_entry: str = "crossover_long_entry"
    threshold_long_entry: str = "threshold_long_entry"
    confluence_long_entry: str = "confluence_long_entry"
    reversal_long_entry: str = "reversal_long_entry"
    breakout_long: str = "breakout_long"
    pullback_long: str = "pullback_long"

    # ── Entry short ─────────────────────────────────────────────────
    crossover_short_entry: str = "crossover_short_entry"
    threshold_short_entry: str = "threshold_short_entry"
    confluence_short_entry: str = "confluence_short_entry"
    reversal_short_entry: str = "reversal_short_entry"
    breakout_short: str = "breakout_short"
    pullback_short: str = "pullback_short"

    # ── Exit long ───────────────────────────────────────────────────
    trail_long_exit: str = "trail_long_exit"
    time_max_bars_exit: str = "time_max_bars_exit"

    # ── Exit short ──────────────────────────────────────────────────
    trail_short_exit: str = "trail_short_exit"

    # ── Filter allow long ───────────────────────────────────────────
    trend_allow_long: str = "trend_allow_long"

    # ── Filter allow short ──────────────────────────────────────────
    trend_allow_short: str = "trend_allow_short"

    # ── Filter allow (both) ─────────────────────────────────────────
    session_active: str = "session_active"
    spread_ok: str = "spread_ok"
    dd_allow_trading: str = "dd_allow_trading"

    # ── TP/SL ──────────────────────────────────────────────────────
    tpsl_long_tp: str = "tpsl_long_tp"
    tpsl_long_sl: str = "tpsl_long_sl"
    tpsl_short_tp: str = "tpsl_short_tp"
    tpsl_short_sl: str = "tpsl_short_sl"

    # ── Position size ───────────────────────────────────────────────
    fr_position_size: str = "fr_position_size"
    vt_position_size: str = "vt_position_size"
    kelly_position_size: str = "kelly_position_size"
    atrs_position_size: str = "atrs_position_size"

    # ── Price action helpers ────────────────────────────────────────
    breakout_volume_confirm: str = "breakout_volume_confirm"

    # ── Derived lists for the assembler ─────────────────────────────
    @property
    def entry_long_patterns(self) -> list[str]:
        return [
            self.crossover_long_entry,
            self.threshold_long_entry,
            self.confluence_long_entry,
            self.reversal_long_entry,
        ]

    @property
    def entry_short_patterns(self) -> list[str]:
        return [
            self.crossover_short_entry,
            self.threshold_short_entry,
            self.confluence_short_entry,
            self.reversal_short_entry,
        ]

    @property
    def exit_long_patterns(self) -> list[str]:
        return [self.trail_long_exit, self.time_max_bars_exit]

    @property
    def exit_short_patterns(self) -> list[str]:
        return [self.trail_short_exit, self.time_max_bars_exit]

    @property
    def allow_long_patterns(self) -> list[str]:
        return [
            self.trend_allow_long,
            self.session_active,
            self.spread_ok,
            self.dd_allow_trading,
        ]

    @property
    def allow_short_patterns(self) -> list[str]:
        return [
            self.trend_allow_short,
            self.session_active,
            self.spread_ok,
            self.dd_allow_trading,
        ]

    @property
    def position_size_candidates(self) -> list[str]:
        return [
            self.fr_position_size,
            self.vt_position_size,
            self.kelly_position_size,
            self.atrs_position_size,
        ]


SIGNAL_COLUMNS = SignalColumns()
