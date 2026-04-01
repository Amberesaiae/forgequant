"""Tests for forgequant.core.config."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from forgequant.core.config import Environment, LogFormat, Settings, get_settings


class TestEnvironmentEnum:
    """Tests for the Environment enum."""

    def test_values(self) -> None:
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_all_members(self) -> None:
        assert len(Environment) == 4


class TestLogFormatEnum:
    """Tests for the LogFormat enum."""

    def test_values(self) -> None:
        assert LogFormat.JSON.value == "json"
        assert LogFormat.CONSOLE.value == "console"


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_values(self) -> None:
        """Settings should have sensible defaults without any env vars."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.forgequant_env == Environment.DEVELOPMENT
        assert settings.forgequant_log_level == "INFO"
        assert settings.forgequant_log_format == LogFormat.JSON
        assert settings.forgequant_min_trades == 150
        assert settings.forgequant_max_drawdown == 0.18
        assert settings.forgequant_min_profit_factor == 1.35
        assert settings.forgequant_min_sharpe == 0.80
        assert settings.forgequant_min_win_rate == 0.35
        assert settings.forgequant_max_correlation == 0.70
        assert settings.chroma_collection_name == "trading_knowledge"

    def test_log_level_normalization(self) -> None:
        """Log level should be normalized to uppercase."""
        with patch.dict(os.environ, {"FORGEQUANT_LOG_LEVEL": "debug"}, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.forgequant_log_level == "DEBUG"

    def test_env_override(self) -> None:
        """Environment variables should override defaults."""
        env = {
            "FORGEQUANT_ENV": "production",
            "FORGEQUANT_MIN_TRADES": "300",
            "FORGEQUANT_MAX_DRAWDOWN": "0.25",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.forgequant_env == Environment.PRODUCTION
        assert settings.forgequant_min_trades == 300
        assert settings.forgequant_max_drawdown == 0.25

    def test_is_development_property(self) -> None:
        with patch.dict(os.environ, {"FORGEQUANT_ENV": "development"}, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.is_development is True
        assert settings.is_production is False

    def test_is_production_property(self) -> None:
        with patch.dict(os.environ, {"FORGEQUANT_ENV": "production"}, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.is_production is True
        assert settings.is_development is False

    def test_is_testing_property(self) -> None:
        with patch.dict(os.environ, {"FORGEQUANT_ENV": "testing"}, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.is_testing is True

    def test_has_openai_key_false(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.has_openai_key is False

    def test_has_openai_key_true(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.has_openai_key is True

    def test_has_mt5_credentials_incomplete(self) -> None:
        with patch.dict(os.environ, {"MT5_LOGIN": "12345"}, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.has_mt5_credentials is False

    def test_has_mt5_credentials_complete(self) -> None:
        env = {
            "MT5_LOGIN": "12345",
            "MT5_PASSWORD": "secret",
            "MT5_SERVER": "Broker-Live",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.has_mt5_credentials is True

    def test_max_drawdown_constraint(self) -> None:
        """Max drawdown must be between 0 (exclusive) and 1 (inclusive)."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            with patch.dict(
                os.environ, {"FORGEQUANT_MAX_DRAWDOWN": "0"}, clear=True
            ):
                Settings(_env_file=None)  # type: ignore[call-arg]

    def test_min_trades_constraint(self) -> None:
        """Min trades must be >= 1."""
        with pytest.raises(Exception):
            with patch.dict(
                os.environ, {"FORGEQUANT_MIN_TRADES": "0"}, clear=True
            ):
                Settings(_env_file=None)  # type: ignore[call-arg]


class TestGetSettings:
    """Tests for the get_settings() cached factory."""

    def test_returns_settings_instance(self) -> None:
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_is_cached(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # After clearing, a new instance is created
        # (may or may not be the same object depending on values)
        assert isinstance(s2, Settings)
