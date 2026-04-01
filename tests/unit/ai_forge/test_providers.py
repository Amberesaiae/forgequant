"""Tests for AI Forge LLM providers."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from forgequant.ai_forge.exceptions import LLMCallError
from forgequant.ai_forge.providers import (
    LLMProvider,
    OpenAIClient,
    AnthropicClient,
    GroqClient,
    get_llm_client,
)
from forgequant.core.config import get_settings


class TestLLMProvider:
    def test_values(self) -> None:
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GROQ.value == "groq"


class TestGetLLMClient:
    def test_openai_from_string(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            client = get_llm_client("openai")
            assert isinstance(client, OpenAIClient)

    def test_openai_from_enum(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            client = get_llm_client(LLMProvider.OPENAI)
            assert isinstance(client, OpenAIClient)

    def test_anthropic_from_string(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True):
            client = get_llm_client("anthropic")
            assert isinstance(client, AnthropicClient)

    def test_groq_from_string(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk-test"}, clear=True):
            client = get_llm_client("groq")
            assert isinstance(client, GroqClient)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(LLMCallError, match="Unknown provider"):
            get_llm_client("not_a_real_provider")


class TestOpenAIClient:
    def test_no_api_key_raises(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMCallError, match="API key"):
                OpenAIClient(api_key="")

    def test_explicit_api_key(self) -> None:
        client = OpenAIClient(api_key="sk-test-key")
        assert client.provider_name == "openai"

    def test_custom_model(self) -> None:
        client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
        assert client._model == "gpt-4o-mini"


class TestAnthropicClient:
    def test_no_api_key_raises(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMCallError, match="API key"):
                AnthropicClient(api_key="")

    def test_explicit_api_key(self) -> None:
        client = AnthropicClient(api_key="sk-test-key")
        assert client.provider_name == "anthropic"


class TestGroqClient:
    def test_no_api_key_raises(self) -> None:
        get_settings.cache_clear()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMCallError, match="API key"):
                GroqClient(api_key="")

    def test_explicit_api_key(self) -> None:
        client = GroqClient(api_key="gsk-test-key")
        assert client.provider_name == "groq"
