"""
LLM provider abstraction.

Provides a unified interface for calling different LLM APIs
(OpenAI, Anthropic, Groq) with structured output via the
instructor library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Any

from forgequant.ai_forge.exceptions import LLMCallError
from forgequant.ai_forge.schemas import StrategySpec
from forgequant.core.config import get_settings
from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@unique
class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    provider_name: str = "unknown"

    @abstractmethod
    def generate_strategy(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> StrategySpec:
        ...


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client using instructor for structured output."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
    ) -> None:
        self._api_key = api_key or get_settings().openai_api_key
        self._model = model

        if not self._api_key:
            raise LLMCallError(
                provider=self.provider_name,
                reason="OpenAI API key not configured. Set OPENAI_API_KEY.",
            )

    def generate_strategy(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> StrategySpec:
        try:
            import instructor
            from openai import OpenAI
        except ImportError as e:
            raise LLMCallError(
                provider=self.provider_name,
                reason=(
                    "openai and instructor are not installed. "
                    "Install with: pip install forgequant[ai]"
                ),
            ) from e

        try:
            client = instructor.from_openai(
                OpenAI(api_key=self._api_key),
            )

            logger.info(
                "llm_call_start",
                provider=self.provider_name,
                model=self._model,
                temperature=temperature,
            )

            spec = client.chat.completions.create(
                model=self._model,
                response_model=StrategySpec,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_retries=max_retries,
            )

            logger.info(
                "llm_call_complete",
                provider=self.provider_name,
                strategy_name=spec.name,
                block_count=len(spec.all_blocks()),
            )

            return spec

        except Exception as e:
            raise LLMCallError(
                provider=self.provider_name,
                reason=str(e),
            ) from e


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client using instructor for structured output."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._api_key = api_key or get_settings().anthropic_api_key
        self._model = model

        if not self._api_key:
            raise LLMCallError(
                provider=self.provider_name,
                reason="Anthropic API key not configured. Set ANTHROPIC_API_KEY.",
            )

    def generate_strategy(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> StrategySpec:
        try:
            import instructor
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMCallError(
                provider=self.provider_name,
                reason=(
                    "anthropic and instructor are not installed. "
                    "Install with: pip install forgequant[ai]"
                ),
            ) from e

        try:
            client = instructor.from_anthropic(
                Anthropic(api_key=self._api_key),
            )

            logger.info(
                "llm_call_start",
                provider=self.provider_name,
                model=self._model,
                temperature=temperature,
            )

            spec = client.messages.create(
                model=self._model,
                response_model=StrategySpec,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=4096,
                max_retries=max_retries,
            )

            logger.info(
                "llm_call_complete",
                provider=self.provider_name,
                strategy_name=spec.name,
                block_count=len(spec.all_blocks()),
            )

            return spec

        except Exception as e:
            raise LLMCallError(
                provider=self.provider_name,
                reason=str(e),
            ) from e


class GroqClient(BaseLLMClient):
    """Groq client using instructor for structured output."""

    provider_name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self._api_key = api_key or get_settings().groq_api_key
        self._model = model

        if not self._api_key:
            raise LLMCallError(
                provider=self.provider_name,
                reason="Groq API key not configured. Set GROQ_API_KEY.",
            )

    def generate_strategy(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> StrategySpec:
        try:
            import instructor
            from groq import Groq
        except ImportError as e:
            raise LLMCallError(
                provider=self.provider_name,
                reason=(
                    "groq and instructor are not installed. "
                    "Install with: pip install forgequant[ai]"
                ),
            ) from e

        try:
            client = instructor.from_groq(
                Groq(api_key=self._api_key),
            )

            logger.info(
                "llm_call_start",
                provider=self.provider_name,
                model=self._model,
                temperature=temperature,
            )

            spec = client.chat.completions.create(
                model=self._model,
                response_model=StrategySpec,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_retries=max_retries,
            )

            logger.info(
                "llm_call_complete",
                provider=self.provider_name,
                strategy_name=spec.name,
                block_count=len(spec.all_blocks()),
            )

            return spec

        except Exception as e:
            raise LLMCallError(
                provider=self.provider_name,
                reason=str(e),
            ) from e


def get_llm_client(
    provider: LLMProvider | str = LLMProvider.OPENAI,
    **kwargs: Any,
) -> BaseLLMClient:
    if isinstance(provider, str):
        provider = provider.strip().lower()

    client_map: dict[str, type[BaseLLMClient]] = {
        "openai": OpenAIClient,
        LLMProvider.OPENAI: OpenAIClient,
        "anthropic": AnthropicClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        "groq": GroqClient,
        LLMProvider.GROQ: GroqClient,
    }

    client_cls = client_map.get(provider)
    if client_cls is None:
        raise LLMCallError(
            provider=str(provider),
            reason=f"Unknown provider '{provider}'. Supported: openai, anthropic, groq",
        )

    return client_cls(**kwargs)
