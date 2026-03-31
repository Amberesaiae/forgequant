"""
ForgeQuant Configuration.

Loads settings from environment variables and .env file.
Uses Pydantic v2 Settings for validation and type safety.

Usage:
    from core.config import settings
    print(settings.mt5_login)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from .env file and environment variables."""

    # --- MetaTrader 5 ---
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_path: str | None = None

    # --- AI / LLM ---
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None

    # --- Project Paths ---
    log_level: str = "INFO"
    data_path: Path = Path("./data")
    knowledge_base_path: Path = Path("./knowledge_base")

    # --- Robustness Testing Defaults ---
    min_trades: int = 150
    max_drawdown: float = 0.18
    min_profit_factor: float = 1.35
    min_sharpe: float = 0.85
    max_oos_degradation: float = 0.32

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton instance — import this everywhere
settings = Settings()
