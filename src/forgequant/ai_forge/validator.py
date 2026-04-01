"""
Strategy spec validator.

Validates a StrategySpec against the live BlockRegistry to ensure:
    1. Every block_name exists in the registry
    2. Every block is in the correct category for its role
    3. Every param dict passes the block's metadata validation
    4. Cross-block constraints are satisfied
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.logging import get_logger
from forgequant.core.types import BlockCategory

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class SpecValidator:
    """Validates a StrategySpec against the BlockRegistry."""

    def __init__(self, registry: BlockRegistry | None = None) -> None:
        self._registry = registry or BlockRegistry()

    def validate(self, spec: StrategySpec) -> ValidationResult:
        result = ValidationResult()

        logger.info(
            "spec_validation_start",
            strategy=spec.name,
            total_blocks=len(spec.all_blocks()),
        )

        self._validate_block_list(
            spec.indicators, "indicators", BlockCategory.INDICATOR, result
        )
        self._validate_block_list(
            spec.price_action, "price_action", BlockCategory.PRICE_ACTION, result
        )
        self._validate_block_list(
            spec.entry_rules, "entry_rules", BlockCategory.ENTRY_RULE, result
        )
        self._validate_block_list(
            spec.exit_rules, "exit_rules", BlockCategory.EXIT_RULE, result
        )
        self._validate_block_list(
            spec.filters, "filters", BlockCategory.FILTER, result
        )

        self._validate_single_block(
            spec.money_management,
            "money_management",
            BlockCategory.MONEY_MANAGEMENT,
            result,
        )

        self._check_cross_block_consistency(spec, result)
        self._check_quality_warnings(spec, result)

        logger.info(
            "spec_validation_complete",
            strategy=spec.name,
            is_valid=result.is_valid,
            error_count=len(result.errors),
            warning_count=len(result.warnings),
        )

        return result

    def _validate_block_list(
        self,
        blocks: list[BlockSpec],
        field_name: str,
        expected_category: BlockCategory,
        result: ValidationResult,
    ) -> None:
        for block_spec in blocks:
            self._validate_single_block(
                block_spec, field_name, expected_category, result
            )

    def _validate_single_block(
        self,
        block_spec: BlockSpec,
        field_name: str,
        expected_category: BlockCategory,
        result: ValidationResult,
    ) -> None:
        name = block_spec.block_name

        block_cls = self._registry.get(name)
        if block_cls is None:
            result.add_error(
                f"[{field_name}] Block '{name}' not found in registry. "
                f"Available blocks: {self._registry.all_names()}"
            )
            return

        actual_category = block_cls.metadata.category
        if actual_category != expected_category:
            result.add_error(
                f"[{field_name}] Block '{name}' is category "
                f"'{actual_category.value}', expected '{expected_category.value}'"
            )
            return

        try:
            validated = block_cls.metadata.validate_params(block_spec.params)
            result.validated_params[name] = validated
        except ValueError as e:
            result.add_error(
                f"[{field_name}] Block '{name}' parameter validation failed: {e}"
            )

    def _check_cross_block_consistency(
        self,
        spec: StrategySpec,
        result: ValidationResult,
    ) -> None:
        mm_params = result.validated_params.get(spec.money_management.block_name, {})
        mm_atr_period = mm_params.get("atr_period")

        if mm_atr_period is not None:
            for exit_spec in spec.exit_rules:
                exit_params = result.validated_params.get(exit_spec.block_name, {})
                exit_atr_period = exit_params.get("atr_period")

                if exit_atr_period is not None and exit_atr_period != mm_atr_period:
                    result.add_warning(
                        f"ATR period mismatch: money_management "
                        f"'{spec.money_management.block_name}' uses atr_period="
                        f"{mm_atr_period}, but exit rule '{exit_spec.block_name}' "
                        f"uses atr_period={exit_atr_period}. Consider aligning them."
                    )

        mm_sl_mult = mm_params.get("sl_atr_mult")
        if mm_sl_mult is not None:
            for exit_spec in spec.exit_rules:
                exit_params = result.validated_params.get(exit_spec.block_name, {})
                exit_sl_mult = exit_params.get("sl_atr_mult")
                if exit_sl_mult is not None and exit_sl_mult != mm_sl_mult:
                    result.add_warning(
                        f"SL multiplier mismatch: money_management uses "
                        f"sl_atr_mult={mm_sl_mult}, but exit rule "
                        f"'{exit_spec.block_name}' uses sl_atr_mult={exit_sl_mult}."
                    )

    def _check_quality_warnings(
        self,
        spec: StrategySpec,
        result: ValidationResult,
    ) -> None:
        if len(spec.filters) == 0:
            result.add_warning(
                "No filter blocks defined. Consider adding a trend filter "
                "or session filter to improve signal quality."
            )

        entry_names = {e.block_name for e in spec.entry_rules}
        if "reversal_pattern_entry" in entry_names and len(spec.filters) == 0:
            result.add_warning(
                "Reversal pattern entries work best when combined with "
                "support/resistance or trend filters."
            )

        if spec.constraints.max_drawdown > 0.30:
            result.add_warning(
                f"Max drawdown constraint of {spec.constraints.max_drawdown:.0%} "
                f"is very permissive. Consider tightening to 15-20%."
            )

        if spec.constraints.min_profit_factor > 3.0:
            result.add_warning(
                f"Min profit factor of {spec.constraints.min_profit_factor} "
                f"is very ambitious. Most robust strategies achieve 1.3-2.0."
            )
