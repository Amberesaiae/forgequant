from __future__ import annotations

import pandas as pd


def _compute_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Returns ATR series using Wilder smoothing.

    ``data`` must have lowercase ``high``, ``low``, ``close`` columns.
    """
    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def _compute_rsi(data: pd.DataFrame, period: int) -> pd.Series:
    """Returns RSI series using Wilder smoothing.

    ``data`` must have lowercase ``close`` column.
    """
    delta = data["close"].diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
