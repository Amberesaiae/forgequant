"""
Shared type definitions, enumerations, and type aliases used across ForgeQuant.

These provide a single source of truth for categorical values and common
type shapes, preventing stringly-typed code.
"""

from __future__ import annotations

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
        return self.value


@unique
class MovingAverageType(str, Enum):
    """Supported moving average calculation methods."""

    SMA = "sma"
    EMA = "ema"

    def __str__(self) -> str:
        return self.value


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


def validate_ohlcv(df: pd.DataFrame, block_name: str = "unknown") -> None:
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

    Raises:
        ValueError: If any validation check fails.
    """
    if df.empty:
        raise ValueError(f"[{block_name}] Input DataFrame is empty")

    # Normalize column names to lowercase for consistent access
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
