"""
Strategy compiler.

Transforms a validated StrategySpec into a CompiledStrategy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.ai_forge.validator import SpecValidator, ValidationResult
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.core.compiler.signal_assembler import assemble_signals
from forgequant.core.exceptions import StrategyCompileError
from forgequant.core.logging import get_logger
from forgequant.core.types import validate_ohlcv

logger = get_logger(__name__)


class StrategyCompiler:
    """Compiles a StrategySpec into a CompiledStrategy."""

    def __init__(
        self,
        registry: BlockRegistry | None = None,
        validate: bool = True,
    ) -> None:
        self._registry = registry or BlockRegistry()
        self._validate = validate
        self._validator = SpecValidator(self._registry) if validate else None

    def compile(
        self,
        spec: StrategySpec,
        data: pd.DataFrame,
        validated_params: dict[str, dict[str, Any]] | None = None,
    ) -> CompiledStrategy:
        logger.info(
            "compilation_start",
            strategy=spec.name,
            n_bars=len(data),
            n_blocks=len(spec.all_blocks()),
        )

        data = data.copy()
        data.columns = data.columns.str.lower()

        try:
            validate_ohlcv(data, block_name=f"compiler:{spec.name}")
        except ValueError as e:
            raise StrategyCompileError(spec.name, f"Invalid OHLCV data: {e}") from e

        val_params = validated_params or {}

        if self._validate and self._validator is not None and not val_params:
            result = self._validator.validate(spec)
            if not result.is_valid:
                raise StrategyCompileError(
                    spec.name,
                    f"Spec validation failed: {'; '.join(result.errors[:5])}",
                )
            val_params = result.validated_params

        compiled = CompiledStrategy(spec=spec, ohlcv=data)

        for block_spec in spec.indicators:
            self._execute_block(block_spec, "indicator", data, val_params, compiled)

        for block_spec in spec.price_action:
            self._execute_block(block_spec, "price_action", data, val_params, compiled)

        for block_spec in spec.entry_rules:
            self._execute_block(block_spec, "entry_rule", data, val_params, compiled)

        for block_spec in spec.exit_rules:
            self._execute_block(block_spec, "exit_rule", data, val_params, compiled)

        for block_spec in spec.filters:
            self._execute_block(block_spec, "filter", data, val_params, compiled)

        self._execute_block(
            spec.money_management, "money_management", data, val_params, compiled
        )

        compiled = assemble_signals(compiled)

        logger.info(
            "compilation_complete",
            strategy=spec.name,
            **compiled.summary(),
        )

        return compiled

    def _execute_block(
        self,
        block_spec: BlockSpec,
        category: str,
        data: pd.DataFrame,
        validated_params: dict[str, dict[str, Any]],
        compiled: CompiledStrategy,
    ) -> None:
        name = block_spec.block_name

        block_cls = self._registry.get(name)
        if block_cls is None:
            raise StrategyCompileError(
                compiled.spec.name,
                f"Block '{name}' not found in registry",
            )

        params = validated_params.get(name, block_spec.params)

        try:
            instance = block_cls()
            result = instance.execute(data, params)
        except Exception as e:
            raise StrategyCompileError(
                compiled.spec.name,
                f"Block '{name}' execution failed: {type(e).__name__}: {e}",
            ) from e

        if isinstance(result, pd.Series):
            result = result.to_frame(name=name)

        compiled.block_outputs[name] = BlockOutput(
            block_name=name,
            category=category,
            params=params,
            result=result,
        )

        logger.debug(
            "block_executed",
            block=name,
            category=category,
            result_type=type(result).__name__,
            result_columns=list(result.columns) if isinstance(result, pd.DataFrame) else [],
        )
