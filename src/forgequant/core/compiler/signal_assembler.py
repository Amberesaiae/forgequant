"""
Signal assembler for combining block outputs into unified signal matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.core.logging import get_logger

logger = get_logger(__name__)

ENTRY_LONG_PATTERNS = [
    "crossover_long_entry",
    "threshold_long_entry",
    "confluence_long_entry",
    "reversal_long_entry",
]

ENTRY_SHORT_PATTERNS = [
    "crossover_short_entry",
    "threshold_short_entry",
    "confluence_short_entry",
    "reversal_short_entry",
]

EXIT_LONG_PATTERNS = [
    "trail_long_exit",
    "time_max_bars_exit",
]

EXIT_SHORT_PATTERNS = [
    "trail_short_exit",
    "time_max_bars_exit",
]

ALLOW_LONG_PATTERNS = [
    "trend_allow_long",
    "session_active",
    "spread_ok",
    "dd_allow_trading",
]

ALLOW_SHORT_PATTERNS = [
    "trend_allow_short",
    "session_active",
    "spread_ok",
    "dd_allow_trading",
]


def _find_column(
    output: BlockOutput,
    patterns: list[str],
) -> pd.Series | None:
    if not isinstance(output.result, pd.DataFrame):
        return None

    for pattern in patterns:
        if pattern in output.result.columns:
            series = output.result[pattern]
            if series.dtype == bool or series.dtype == np.bool_:
                return series
            try:
                return series.astype(bool)
            except (ValueError, TypeError):
                continue

    return None


def _find_float_column(
    output: BlockOutput,
    column_name: str,
) -> pd.Series | None:
    if not isinstance(output.result, pd.DataFrame):
        return None

    if column_name in output.result.columns:
        return output.result[column_name]

    return None


def _or_combine(
    series_list: list[pd.Series],
    index: pd.DatetimeIndex,
) -> pd.Series:
    if not series_list:
        return pd.Series(False, index=index, dtype=bool)

    result = series_list[0].reindex(index, fill_value=False)
    for s in series_list[1:]:
        result = result | s.reindex(index, fill_value=False)

    return result.fillna(False).astype(bool)


def _and_combine(
    series_list: list[pd.Series],
    index: pd.DatetimeIndex,
) -> pd.Series:
    if not series_list:
        return pd.Series(True, index=index, dtype=bool)

    result = series_list[0].reindex(index, fill_value=True)
    for s in series_list[1:]:
        result = result & s.reindex(index, fill_value=True)

    return result.fillna(True).astype(bool)


def assemble_signals(compiled: CompiledStrategy) -> CompiledStrategy:
    """Assemble all block outputs into unified signal matrices."""
    index = compiled.index

    entry_longs: list[pd.Series] = []
    entry_shorts: list[pd.Series] = []

    for name, output in compiled.block_outputs.items():
        if output.category == "entry_rule":
            el = _find_column(output, ENTRY_LONG_PATTERNS)
            if el is not None:
                entry_longs.append(el)

            es = _find_column(output, ENTRY_SHORT_PATTERNS)
            if es is not None:
                entry_shorts.append(es)

    for name, output in compiled.block_outputs.items():
        if output.category == "price_action":
            bl = _find_column(output, ["breakout_long"])
            if bl is not None:
                vol = _find_column(output, ["breakout_volume_confirm"])
                if vol is not None:
                    bl = bl & vol
                entry_longs.append(bl)

            bs = _find_column(output, ["breakout_short"])
            if bs is not None:
                vol = _find_column(output, ["breakout_volume_confirm"])
                if vol is not None:
                    bs = bs & vol
                entry_shorts.append(bs)

            pl = _find_column(output, ["pullback_long"])
            if pl is not None:
                entry_longs.append(pl)

            ps = _find_column(output, ["pullback_short"])
            if ps is not None:
                entry_shorts.append(ps)

    compiled.entry_long = _or_combine(entry_longs, index)
    compiled.entry_short = _or_combine(entry_shorts, index)

    exit_longs: list[pd.Series] = []
    exit_shorts: list[pd.Series] = []

    for name, output in compiled.block_outputs.items():
        if output.category == "exit_rule":
            xl = _find_column(output, EXIT_LONG_PATTERNS)
            if xl is not None:
                exit_longs.append(xl)

            xs = _find_column(output, EXIT_SHORT_PATTERNS)
            if xs is not None:
                exit_shorts.append(xs)

    compiled.exit_long = _or_combine(exit_longs, index)
    compiled.exit_short = _or_combine(exit_shorts, index)

    allow_longs: list[pd.Series] = []
    allow_shorts: list[pd.Series] = []

    for name, output in compiled.block_outputs.items():
        if output.category == "filter":
            al = _find_column(output, ALLOW_LONG_PATTERNS)
            if al is not None:
                allow_longs.append(al)

            ash = _find_column(output, ALLOW_SHORT_PATTERNS)
            if ash is not None:
                allow_shorts.append(ash)

    compiled.allow_long = _and_combine(allow_longs, index)
    compiled.allow_short = _and_combine(allow_shorts, index)

    for name, output in compiled.block_outputs.items():
        if output.category == "exit_rule":
            if compiled.stop_loss_long is None:
                sl_l = _find_float_column(output, "tpsl_long_sl")
                if sl_l is not None:
                    compiled.stop_loss_long = sl_l

            if compiled.stop_loss_short is None:
                sl_s = _find_float_column(output, "tpsl_short_sl")
                if sl_s is not None:
                    compiled.stop_loss_short = sl_s

            if compiled.take_profit_long is None:
                tp_l = _find_float_column(output, "tpsl_long_tp")
                if tp_l is not None:
                    compiled.take_profit_long = tp_l

            if compiled.take_profit_short is None:
                tp_s = _find_float_column(output, "tpsl_short_tp")
                if tp_s is not None:
                    compiled.take_profit_short = tp_s

    mm_name = compiled.spec.money_management.block_name
    mm_output = compiled.block_outputs.get(mm_name)

    if mm_output is not None and isinstance(mm_output.result, pd.DataFrame):
        size_col_candidates = [
            "fr_position_size",
            "vt_position_size",
            "kelly_position_size",
            "atrs_position_size",
        ]
        for col in size_col_candidates:
            if col in mm_output.result.columns:
                compiled.position_size_long = mm_output.result[col]
                compiled.position_size_short = mm_output.result[col]
                break

    logger.info(
        "signals_assembled",
        strategy=compiled.spec.name,
        entry_long_count=int(compiled.entry_long.sum()) if compiled.entry_long is not None else 0,
        entry_short_count=int(compiled.entry_short.sum()) if compiled.entry_short is not None else 0,
        has_tp_sl=compiled.stop_loss_long is not None,
        has_position_sizing=compiled.position_size_long is not None,
    )

    return compiled
