"""
End-to-end LLM strategy generation pipeline.

Orchestrates:
    1. (Optional) RAG retrieval from knowledge base
    2. System prompt construction with block catalog
    3. User message formatting
    4. LLM call via the configured provider
    5. StrategySpec validation against the BlockRegistry
    6. Retry with error feedback if validation fails
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forgequant.ai_forge.exceptions import (
    AIForgeError,
    LLMCallError,
    SpecValidationError,
)
from forgequant.ai_forge.grounding import KnowledgeBase
from forgequant.ai_forge.prompt import build_system_prompt, build_user_message
from forgequant.ai_forge.providers import BaseLLMClient, LLMProvider, get_llm_client
from forgequant.ai_forge.schemas import StrategySpec
from forgequant.ai_forge.validator import SpecValidator, ValidationResult
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    spec: StrategySpec | None = None
    validation: ValidationResult | None = None
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    raw_specs: list[StrategySpec] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.spec is not None and self.validation is not None and self.validation.is_valid


@dataclass
class PipelineConfig:
    provider: LLMProvider | str = LLMProvider.OPENAI
    model: str | None = None
    temperature: float = 0.7
    max_attempts: int = 3
    max_llm_retries: int = 2
    use_rag: bool = False
    rag_n_results: int = 5
    rag_persist_dir: str = "./data/chromadb"
    rag_collection: str = "trading_knowledge"


class ForgeQuantPipeline:
    """End-to-end strategy generation pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        registry: BlockRegistry | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._registry = registry or BlockRegistry()
        self._validator = SpecValidator(self._registry)
        self._llm_client = llm_client
        self._knowledge_base: KnowledgeBase | None = None

    def _get_llm_client(self) -> BaseLLMClient:
        if self._llm_client is not None:
            return self._llm_client

        kwargs: dict[str, Any] = {}
        if self._config.model:
            kwargs["model"] = self._config.model

        self._llm_client = get_llm_client(
            provider=self._config.provider,
            **kwargs,
        )
        return self._llm_client

    def _get_rag_context(self, query: str) -> str:
        if not self._config.use_rag:
            return ""

        try:
            if self._knowledge_base is None:
                self._knowledge_base = KnowledgeBase(
                    persist_directory=self._config.rag_persist_dir,
                    collection_name=self._config.rag_collection,
                )

            context = self._knowledge_base.retrieve(
                query=query,
                n_results=self._config.rag_n_results,
            )
            return context

        except Exception as e:
            logger.warning(
                "rag_retrieval_failed",
                error=str(e),
            )
            return ""

    def generate(
        self,
        idea: str,
        timeframe: str = "1h",
        instruments: list[str] | None = None,
        style: str | None = None,
        additional_requirements: str = "",
    ) -> PipelineResult:
        result = PipelineResult()

        logger.info(
            "pipeline_start",
            idea_length=len(idea),
            timeframe=timeframe,
            provider=str(self._config.provider),
        )

        rag_context = self._get_rag_context(idea)

        try:
            system_prompt = build_system_prompt(
                registry=self._registry,
                rag_context=rag_context,
            )
        except Exception as e:
            result.errors.append(f"System prompt build failed: {e}")
            return result

        user_message = build_user_message(
            idea=idea,
            timeframe=timeframe,
            instruments=instruments,
            style=style,
            additional_requirements=additional_requirements,
        )

        llm_client = self._get_llm_client()
        validation_errors_feedback = ""

        for attempt in range(1, self._config.max_attempts + 1):
            result.attempts = attempt

            logger.info(
                "pipeline_attempt",
                attempt=attempt,
                max_attempts=self._config.max_attempts,
            )

            if validation_errors_feedback:
                augmented_message = (
                    f"{user_message}\n\n"
                    f"IMPORTANT: Your previous attempt had validation errors. "
                    f"Please fix these issues:\n{validation_errors_feedback}"
                )
            else:
                augmented_message = user_message

            try:
                spec = llm_client.generate_strategy(
                    system_prompt=system_prompt,
                    user_message=augmented_message,
                    temperature=self._config.temperature,
                    max_retries=self._config.max_llm_retries,
                )
            except LLMCallError as e:
                result.errors.append(f"Attempt {attempt}: LLM call failed: {e.reason}")
                continue
            except Exception as e:
                result.errors.append(f"Attempt {attempt}: Unexpected error: {e}")
                continue

            result.raw_specs.append(spec)

            validation = self._validator.validate(spec)

            if validation.is_valid:
                result.spec = spec
                result.validation = validation

                logger.info(
                    "pipeline_success",
                    strategy=spec.name,
                    attempt=attempt,
                    warnings=len(validation.warnings),
                )

                return result
            else:
                validation_errors_feedback = "\n".join(
                    f"- {err}" for err in validation.errors
                )

                result.errors.append(
                    f"Attempt {attempt}: Validation failed with "
                    f"{len(validation.errors)} error(s)"
                )

                logger.warning(
                    "pipeline_validation_failed",
                    attempt=attempt,
                    error_count=len(validation.errors),
                    errors=validation.errors[:5],
                )

        logger.error(
            "pipeline_failed",
            attempts=result.attempts,
            total_errors=len(result.errors),
        )

        return result
