"""Tests for the StrategyCompiler."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.exceptions import StrategyCompileError


def _populate_registry(registry: BlockRegistry) -> None:
    for mod_name in [
        "forgequant.blocks.indicators",
        "forgequant.blocks.entry_rules",
        "forgequant.blocks.exit_rules",
        "forgequant.blocks.money_management",
        "forgequant.blocks.filters",
        "forgequant.blocks.price_action",
    ]:
        module = importlib.import_module(mod_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "metadata") and attr_name != "BaseBlock":
                try:
                    registry.register_class(attr)
                except Exception:
                    pass


@pytest.fixture
def full_registry(clean_registry: BlockRegistry) -> BlockRegistry:
    _populate_registry(clean_registry)
    return clean_registry


@pytest.fixture
def compiler(full_registry: BlockRegistry) -> StrategyCompiler:
    return StrategyCompiler(registry=full_registry, validate=True)


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="compiler_test_strategy",
        description="A trend following EMA crossover strategy for compiler testing purposes.",
        objective={"style": "trend_following", "timeframe": "1h"},
        indicators=[
            BlockSpec(block_name="ema", params={"period": 20}),
            BlockSpec(block_name="atr", params={"period": 14}),
        ],
        entry_rules=[
            BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20}),
        ],
        exit_rules=[
            BlockSpec(block_name="fixed_tpsl", params={"atr_period": 14}),
        ],
        money_management=BlockSpec(block_name="fixed_risk", params={"atr_period": 14}),
    )


class TestStrategyCompiler:
    def test_compile_basic_spec(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.spec.name == "compiler_test_strategy"
        assert compiled.n_bars == len(sample_ohlcv)
        assert "ema" in compiled.block_outputs
        assert "atr" in compiled.block_outputs
        assert "crossover_entry" in compiled.block_outputs
        assert "fixed_tpsl" in compiled.block_outputs
        assert "fixed_risk" in compiled.block_outputs

    def test_compile_produces_signals(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.entry_long is not None
        assert compiled.entry_short is not None
        assert compiled.exit_long is not None
        assert compiled.exit_short is not None
        assert len(compiled.entry_long) == len(sample_ohlcv)

    def test_compile_with_filters(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = StrategySpec(
            name="filtered_strategy",
            description="Strategy with trend filter for compiler testing with adequate description.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 20})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            filters=[BlockSpec(block_name="trend_filter", params={"period": 50})],
            money_management=BlockSpec(block_name="fixed_risk"),
        )
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.allow_long is not None
        assert compiled.allow_short is not None
        raw_long = compiled.entry_long.sum() if compiled.entry_long is not None else 0
        filtered_long = compiled.filtered_entry_long().sum()
        assert filtered_long <= raw_long

    def test_compile_has_tpsl(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.stop_loss_long is not None
        assert compiled.take_profit_long is not None

    def test_compile_has_position_sizing(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)

        assert compiled.position_size_long is not None
        valid = compiled.position_size_long.dropna()
        assert (valid > 0).all()

    def test_compile_invalid_spec_raises(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = StrategySpec(
            name="invalid_block_strategy",
            description="Strategy referencing a nonexistent block name for testing error handling.",
            objective={"style": "breakout", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="nonexistent_indicator_xyz")],
            entry_rules=[BlockSpec(block_name="crossover_entry")],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )
        with pytest.raises(StrategyCompileError):
            compiler.compile(spec, sample_ohlcv)

    def test_compile_empty_data_raises(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = _make_spec()
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([]),
        )
        with pytest.raises(StrategyCompileError, match="Invalid OHLCV"):
            compiler.compile(spec, empty)

    def test_compile_skip_validation(
        self, full_registry: BlockRegistry, sample_ohlcv: pd.DataFrame
    ) -> None:
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)
        assert compiled.n_bars == len(sample_ohlcv)

    def test_compile_summary(
        self, compiler: StrategyCompiler, sample_ohlcv: pd.DataFrame
    ) -> None:
        spec = _make_spec()
        compiled = compiler.compile(spec, sample_ohlcv)
        summary = compiled.summary()

        assert summary["strategy_name"] == "compiler_test_strategy"
        assert summary["n_bars"] == len(sample_ohlcv)
        assert "raw_long_entries" in summary
        assert "filtered_long_entries" in summary
