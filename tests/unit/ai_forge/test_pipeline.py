"""Tests for AI Forge pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from forgequant.ai_forge.exceptions import LLMCallError
from forgequant.ai_forge.pipeline import (
    ForgeQuantPipeline,
    PipelineConfig,
    PipelineResult,
)
from forgequant.ai_forge.providers import BaseLLMClient
from forgequant.ai_forge.schemas import StrategySpec
from forgequant.blocks.registry import BlockRegistry

# Import to register blocks
import forgequant.blocks.indicators  # noqa: F401
import forgequant.blocks.entry_rules  # noqa: F401
import forgequant.blocks.exit_rules  # noqa: F401
import forgequant.blocks.money_management  # noqa: F401
import forgequant.blocks.filters  # noqa: F401
import forgequant.blocks.price_action  # noqa: F401


def _populate_registry(registry: BlockRegistry) -> None:
    import importlib

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


def _make_valid_spec_obj() -> StrategySpec:
    return StrategySpec(
        name="test_ema_crossover",
        description="A simple EMA crossover trend-following strategy for testing purposes.",
        objective={
            "style": "trend_following",
            "timeframe": "1h",
            "instruments": ["EURUSD"],
        },
        indicators=[
            {"block_name": "ema", "params": {"period": 20}},
            {"block_name": "atr", "params": {"period": 14}},
        ],
        entry_rules=[
            {"block_name": "crossover_entry", "params": {"fast_period": 10, "slow_period": 20}},
        ],
        exit_rules=[
            {"block_name": "fixed_tpsl", "params": {"atr_period": 14}},
        ],
        money_management={"block_name": "fixed_risk", "params": {"atr_period": 14}},
        filters=[
            {"block_name": "trend_filter", "params": {"period": 200}},
        ],
    )


class MockLLMClient(BaseLLMClient):
    """Mock LLM client that returns a pre-configured spec."""

    provider_name = "mock"

    def __init__(
        self,
        spec: StrategySpec | None = None,
        error: Exception | None = None,
        call_count_to_succeed: int = 1,
    ) -> None:
        self._spec = spec
        self._error = error
        self._call_count = 0
        self._succeed_at = call_count_to_succeed

    def generate_strategy(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> StrategySpec:
        self._call_count += 1

        if self._error and self._call_count < self._succeed_at:
            raise self._error

        if self._spec is None:
            raise LLMCallError(provider="mock", reason="No spec configured")

        return self._spec


class TestPipelineResult:
    def test_default(self) -> None:
        r = PipelineResult()
        assert r.success is False
        assert r.spec is None

    def test_success(self) -> None:
        from forgequant.ai_forge.validator import ValidationResult

        r = PipelineResult(
            spec=_make_valid_spec_obj(),
            validation=ValidationResult(is_valid=True),
        )
        assert r.success is True


class TestForgeQuantPipeline:
    @pytest.fixture(autouse=True)
    def setup_registry(self, clean_registry: BlockRegistry) -> None:
        _populate_registry(clean_registry)

    def test_successful_generation(self) -> None:
        spec = _make_valid_spec_obj()
        mock_client = MockLLMClient(spec=spec)

        pipeline = ForgeQuantPipeline(
            config=PipelineConfig(),
            llm_client=mock_client,
        )

        result = pipeline.generate(
            idea="Simple EMA crossover for EURUSD",
            timeframe="1h",
        )

        assert result.success, f"Expected success, got errors: {result.errors}"
        assert result.spec is not None
        assert result.spec.name == "test_ema_crossover"
        assert result.attempts == 1

    def test_llm_failure_retries(self) -> None:
        spec = _make_valid_spec_obj()
        mock_client = MockLLMClient(
            spec=spec,
            error=LLMCallError("mock", "temporary failure"),
            call_count_to_succeed=2,
        )

        pipeline = ForgeQuantPipeline(
            config=PipelineConfig(max_attempts=3),
            llm_client=mock_client,
        )

        result = pipeline.generate(idea="Test strategy")

        assert result.success
        assert result.attempts == 2

    def test_all_attempts_fail(self) -> None:
        mock_client = MockLLMClient(
            error=LLMCallError("mock", "persistent failure"),
            call_count_to_succeed=999,
        )

        pipeline = ForgeQuantPipeline(
            config=PipelineConfig(max_attempts=2),
            llm_client=mock_client,
        )

        result = pipeline.generate(idea="Test strategy")

        assert not result.success
        assert result.attempts == 2
        assert len(result.errors) == 2

    def test_invalid_spec_triggers_retry(self) -> None:
        valid_spec = _make_valid_spec_obj()

        call_count = {"n": 0}

        class RetryMockClient(BaseLLMClient):
            provider_name = "mock"

            def generate_strategy(self, system_prompt, user_message, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return StrategySpec(
                        name="bad_strategy",
                        description="Strategy that uses a block name that does not exist in registry.",
                        objective={"style": "breakout", "timeframe": "1h"},
                        indicators=[{"block_name": "fake_indicator_xyz"}],
                        entry_rules=[{"block_name": "crossover_entry"}],
                        exit_rules=[{"block_name": "fixed_tpsl"}],
                        money_management={"block_name": "fixed_risk"},
                    )
                return valid_spec

        pipeline = ForgeQuantPipeline(
            config=PipelineConfig(max_attempts=3),
            llm_client=RetryMockClient(),
        )

        result = pipeline.generate(idea="Test retry")
        assert result.success
        assert result.attempts == 2

    def test_pipeline_result_contains_raw_specs(self) -> None:
        spec = _make_valid_spec_obj()
        mock_client = MockLLMClient(spec=spec)

        pipeline = ForgeQuantPipeline(
            config=PipelineConfig(),
            llm_client=mock_client,
        )

        result = pipeline.generate(idea="Test")
        assert len(result.raw_specs) == 1
