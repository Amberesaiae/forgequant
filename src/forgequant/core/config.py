"""
Application configuration using Pydantic Settings.

Loads values from environment variables and/or a .env file.
Provides a singleton accessor via get_settings() with LRU caching.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application runtime environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(str, Enum):
    """Supported structured log output formats."""

    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    """
    Central configuration for the ForgeQuant platform.

    All values can be overridden via environment variables prefixed with
    FORGEQUANT_ (for app-level settings) or via their exact name
    (for API keys and external service credentials).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    forgequant_env: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Runtime environment",
    )
    forgequant_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level",
    )
    forgequant_log_format: LogFormat = Field(
        default=LogFormat.JSON,
        description="Structured log output format",
    )

    # ── LLM API Keys ────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key",
    )
    groq_api_key: str = Field(
        default="",
        description="Groq API key",
    )

    # ── MetaTrader 5 ────────────────────────────────────────────────────
    mt5_login: str = Field(default="", description="MT5 account login")
    mt5_password: str = Field(default="", description="MT5 account password")
    mt5_server: str = Field(default="", description="MT5 server name")
    mt5_terminal_path: str = Field(default="", description="Path to MT5 terminal executable")

    # ── Robustness Thresholds ────────────────────────────────────────────
    forgequant_min_trades: int = Field(
        default=150,
        ge=1,
        description="Minimum number of trades required for strategy validation",
    )
    forgequant_max_drawdown: float = Field(
        default=0.18,
        gt=0.0,
        le=1.0,
        description="Maximum acceptable drawdown as a decimal fraction (0.18 = 18%)",
    )
    forgequant_min_profit_factor: float = Field(
        default=1.35,
        gt=0.0,
        description="Minimum acceptable profit factor (gross_profit / gross_loss)",
    )
    forgequant_min_sharpe: float = Field(
        default=0.80,
        description="Minimum acceptable annualized Sharpe ratio",
    )
    forgequant_min_win_rate: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description="Minimum acceptable win rate as a decimal fraction",
    )
    forgequant_max_correlation: float = Field(
        default=0.70,
        gt=0.0,
        le=1.0,
        description="Maximum acceptable equity curve correlation with existing strategies",
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_persist_directory: Path = Field(
        default=Path("./data/chromadb"),
        description="Directory for ChromaDB persistent storage",
    )
    chroma_collection_name: str = Field(
        default="trading_knowledge",
        description="ChromaDB collection name for knowledge base",
    )

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("forgequant_log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Ensure log level is always uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v

    # ── Derived Properties ───────────────────────────────────────────────

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.forgequant_env == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.forgequant_env == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.forgequant_env == Environment.TESTING

    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def has_mt5_credentials(self) -> bool:
        """Check if all MT5 credentials are configured."""
        return bool(self.mt5_login and self.mt5_password and self.mt5_server)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Uses LRU cache so the .env file is read only once per process.
    Call get_settings.cache_clear() if you need to reload configuration.
    """
    return Settings()
