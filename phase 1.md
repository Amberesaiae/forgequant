I'll now build this project from scratch in carefully structured phases. Based on my earlier analysis of the `Amberesaiae/forgequant` repo and the StrategyQuant X–inspired architecture, here is the complete **Phase 1: Foundation** — the project skeleton, configuration system, structured logging, base abstractions, block metadata, and registry.

---

# ForgeQuant — Full Rewrite Plan

## Phase Overview

| Phase | Scope |
|-------|-------|
| **Phase 1** | Project skeleton, `pyproject.toml`, env config, structured logging, base block abstraction, block metadata, block registry, and full test suite |
| **Phase 2** | All indicator blocks (EMA, RSI, MACD, ADX, ATR, Bollinger, Ichimoku, Stochastic) |
| **Phase 3** | Price action blocks, entry rule blocks, exit rule blocks |
| **Phase 4** | Money management blocks, filter blocks |
| **Phase 5** | AI Forge — Pydantic schemas, system prompt, RAG grounding, LLM pipeline |
| **Phase 6** | Strategy compiler + VectorBT backtesting engine |
| **Phase 7** | Robustness suite (walk-forward, Monte Carlo, CPCV) |
| **Phase 8** | Execution layer (MT5 via aiomql) |
| **Phase 9** | Reflex dashboard frontend |

---

# PHASE 1 — Foundation

## 1.1 Directory Structure

Create this exact tree:

```
forgequant/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── src/
│   └── forgequant/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── logging.py
│       │   ├── exceptions.py
│       │   ├── types.py
│       │   ├── generator/
│       │   │   └── __init__.py
│       │   ├── robustness/
│       │   │   └── __init__.py
│       │   └── compiler/
│       │       └── __init__.py
│       ├── blocks/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── metadata.py
│       │   ├── registry.py
│       │   ├── indicators/
│       │   │   └── __init__.py
│       │   ├── price_action/
│       │   │   └── __init__.py
│       │   ├── entry_rules/
│       │   │   └── __init__.py
│       │   ├── exit_rules/
│       │   │   └── __init__.py
│       │   ├── money_management/
│       │   │   └── __init__.py
│       │   └── filters/
│       │       └── __init__.py
│       ├── ai_forge/
│       │   ├── __init__.py
│       │   ├── prompt.py
│       │   └── grounding.py
│       ├── execution/
│       │   └── __init__.py
│       ├── frontend/
│       │   ├── __init__.py
│       │   ├── pages/
│       │   │   └── __init__.py
│       │   └── components/
│       │       └── __init__.py
│       ├── knowledge_base/
│       │   └── __init__.py
│       ├── scripts/
│       │   └── __init__.py
│       └── notebooks/
│           └── .gitkeep
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── unit/
    │   ├── __init__.py
    │   ├── test_config.py
    │   ├── test_logging.py
    │   ├── test_exceptions.py
    │   ├── test_types.py
    │   ├── test_metadata.py
    │   ├── test_base_block.py
    │   └── test_registry.py
    └── integration/
        └── __init__.py
```

---

## 1.2 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling>=1.21.0"]
build-backend = "hatchling.build"

[project]
name = "forgequant"
version = "0.1.0"
description = "Open-source StrategyQuant X–style systematic strategy generation platform"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.12"
authors = [
    { name = "ForgeQuant Contributors" },
]
keywords = [
    "quantitative-finance",
    "algorithmic-trading",
    "strategy-generation",
    "backtesting",
    "metatrader5",
]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Office/Business :: Financial :: Investment",
    "Typing :: Typed",
]

dependencies = [
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.1.0",
    "structlog>=24.1.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
ai = [
    "openai>=1.12.0",
    "instructor>=1.0.0",
    "chromadb>=0.4.22",
]
backtest = [
    "vectorbt>=0.26.0",
]
execution = [
    "aiomql>=1.0.0",
]
frontend = [
    "reflex>=0.4.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "pandas-stubs>=2.1.0",
    "pre-commit>=3.6.0",
]
all = [
    "forgequant[ai,backtest,execution,frontend,dev]",
]

[project.urls]
Homepage = "https://github.com/Amberesaiae/forgequant"
Repository = "https://github.com/Amberesaiae/forgequant"
Issues = "https://github.com/Amberesaiae/forgequant/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/forgequant"]

[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # ruff-specific rules
]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["forgequant"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["vectorbt.*", "aiomql.*", "chromadb.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--tb=short",
    "--cov=forgequant",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
```

---

## 1.3 `.env.example`

```dotenv
# ============================================================================
# ForgeQuant Environment Configuration
# ============================================================================
# Copy this file to .env and fill in your values.
# Never commit .env to version control.
# ============================================================================

# --- Application ---
FORGEQUANT_ENV=development
FORGEQUANT_LOG_LEVEL=DEBUG
FORGEQUANT_LOG_FORMAT=json

# --- LLM API Keys ---
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=

# --- MetaTrader 5 ---
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
MT5_TERMINAL_PATH=

# --- Robustness Thresholds ---
FORGEQUANT_MIN_TRADES=150
FORGEQUANT_MAX_DRAWDOWN=0.18
FORGEQUANT_MIN_PROFIT_FACTOR=1.35
FORGEQUANT_MIN_SHARPE=0.80
FORGEQUANT_MIN_WIN_RATE=0.35
FORGEQUANT_MAX_CORRELATION=0.70

# --- ChromaDB ---
CHROMA_PERSIST_DIRECTORY=./data/chromadb
CHROMA_COLLECTION_NAME=trading_knowledge
```

---

## 1.4 `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env

# Data
data/
*.parquet
*.h5
*.db

# Notebooks
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Coverage
htmlcov/
.coverage
.coverage.*
coverage.xml

# mypy
.mypy_cache/

# ruff
.ruff_cache/

# ChromaDB
chromadb/
```

---

## 1.5 `README.md`

```markdown
# ForgeQuant

**Open-source StrategyQuant X–style systematic strategy generation platform.**

ForgeQuant provides a modular, AI-assisted framework for generating, evaluating, and
deploying systematic trading strategies. It uses composable building blocks — indicators,
price action patterns, entry/exit rules, money management, and filters — that can be
assembled by humans or by LLM-driven pipelines.

## Status

🚧 **Pre-Alpha** — under active development. Not suitable for live trading yet.

## Quick Start

```bash
# Clone
git clone https://github.com/Amberesaiae/forgequant.git
cd forgequant

# Create virtual environment (Python 3.12+)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# Run tests
pytest
```

## Architecture

```
forgequant/
├── core/         # Config, logging, exceptions, type definitions
├── blocks/       # Composable strategy building blocks
│   ├── indicators/        # Technical indicators (EMA, RSI, MACD, etc.)
│   ├── price_action/      # Price action patterns
│   ├── entry_rules/       # Entry signal generators
│   ├── exit_rules/        # Exit signal generators
│   ├── money_management/  # Position sizing
│   └── filters/           # Trade filters
├── ai_forge/     # LLM-driven strategy specification
├── execution/    # Live execution (MetaTrader 5)
└── frontend/     # Reflex-based dashboard
```

## License

MIT
```

---

## 1.6 `src/forgequant/__init__.py`

```python
"""
ForgeQuant — Open-source systematic strategy generation platform.

Provides composable building blocks, AI-assisted strategy specification,
vectorized backtesting, robustness testing, and live execution capabilities.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
```

---

## 1.7 `src/forgequant/core/__init__.py`

```python
"""
Core infrastructure: configuration, logging, exceptions, and shared types.
"""

from forgequant.core.config import Settings, get_settings
from forgequant.core.exceptions import (
    ForgeQuantError,
    BlockNotFoundError,
    BlockRegistrationError,
    BlockComputeError,
    BlockValidationError,
    ConfigurationError,
)
from forgequant.core.logging import get_logger, configure_logging
from forgequant.core.types import BlockCategory, TimeFrame, TradeDirection

__all__ = [
    "Settings",
    "get_settings",
    "ForgeQuantError",
    "BlockNotFoundError",
    "BlockRegistrationError",
    "BlockComputeError",
    "BlockValidationError",
    "ConfigurationError",
    "get_logger",
    "configure_logging",
    "BlockCategory",
    "TimeFrame",
    "TradeDirection",
]
```

---

## 1.8 `src/forgequant/core/config.py`

```python
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
```

---

## 1.9 `src/forgequant/core/exceptions.py`

```python
"""
Structured exception hierarchy for ForgeQuant.

All custom exceptions inherit from ForgeQuantError so callers can catch
the entire tree with a single except clause when appropriate.
"""

from __future__ import annotations

from typing import Any


class ForgeQuantError(Exception):
    """
    Base exception for all ForgeQuant errors.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary of structured context for logging.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        if self.details:
            return f"{cls_name}(message={self.message!r}, details={self.details!r})"
        return f"{cls_name}(message={self.message!r})"


# ── Block Errors ─────────────────────────────────────────────────────────────


class BlockNotFoundError(ForgeQuantError):
    """Raised when a requested block is not in the registry."""

    def __init__(self, block_name: str) -> None:
        super().__init__(
            message=f"Block '{block_name}' not found in registry",
            details={"block_name": block_name},
        )
        self.block_name = block_name


class BlockRegistrationError(ForgeQuantError):
    """Raised when a block fails to register (e.g., duplicate name, invalid class)."""

    def __init__(self, block_name: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to register block '{block_name}': {reason}",
            details={"block_name": block_name, "reason": reason},
        )
        self.block_name = block_name
        self.reason = reason


class BlockComputeError(ForgeQuantError):
    """Raised when a block's compute() method encounters a runtime error."""

    def __init__(self, block_name: str, reason: str) -> None:
        super().__init__(
            message=f"Block '{block_name}' compute failed: {reason}",
            details={"block_name": block_name, "reason": reason},
        )
        self.block_name = block_name
        self.reason = reason


class BlockValidationError(ForgeQuantError):
    """Raised when block parameters fail validation."""

    def __init__(
        self,
        block_name: str,
        param_name: str,
        value: Any,
        constraint: str,
    ) -> None:
        super().__init__(
            message=(
                f"Block '{block_name}' parameter '{param_name}' value {value!r} "
                f"violates constraint: {constraint}"
            ),
            details={
                "block_name": block_name,
                "param_name": param_name,
                "value": value,
                "constraint": constraint,
            },
        )
        self.block_name = block_name
        self.param_name = param_name
        self.value = value
        self.constraint = constraint


# ── Configuration Errors ─────────────────────────────────────────────────────


class ConfigurationError(ForgeQuantError):
    """Raised when the application configuration is invalid or incomplete."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message)


# ── Strategy Errors ──────────────────────────────────────────────────────────


class StrategyCompileError(ForgeQuantError):
    """Raised when a strategy specification cannot be compiled into runnable code."""

    def __init__(self, strategy_name: str, reason: str) -> None:
        super().__init__(
            message=f"Strategy '{strategy_name}' compilation failed: {reason}",
            details={"strategy_name": strategy_name, "reason": reason},
        )
        self.strategy_name = strategy_name
        self.reason = reason


class RobustnessError(ForgeQuantError):
    """Raised when a strategy fails a robustness gate."""

    def __init__(self, strategy_name: str, gate_name: str, reason: str) -> None:
        super().__init__(
            message=(
                f"Strategy '{strategy_name}' failed robustness gate "
                f"'{gate_name}': {reason}"
            ),
            details={
                "strategy_name": strategy_name,
                "gate_name": gate_name,
                "reason": reason,
            },
        )
        self.strategy_name = strategy_name
        self.gate_name = gate_name
        self.reason = reason
```

---

## 1.10 `src/forgequant/core/types.py`

```python
"""
Shared type definitions, enumerations, and type aliases used across ForgeQuant.

These provide a single source of truth for categorical values and common
type shapes, preventing stringly-typed code.
"""

from __future__ import annotations

from enum import Enum, unique
from typing import Any

import pandas as pd


# ── Enumerations ─────────────────────────────────────────────────────────────


@unique
class BlockCategory(str, Enum):
    """
    Categories for strategy building blocks.

    Each block belongs to exactly one category, which determines its role
    in strategy assembly and the interface contract it must fulfil.
    """

    INDICATOR = "indicator"
    PRICE_ACTION = "price_action"
    ENTRY_RULE = "entry_rule"
    EXIT_RULE = "exit_rule"
    MONEY_MANAGEMENT = "money_management"
    FILTER = "filter"

    def __str__(self) -> str:
        return self.value


@unique
class TimeFrame(str, Enum):
    """Supported OHLCV bar timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

    def __str__(self) -> str:
        return self.value


@unique
class TradeDirection(str, Enum):
    """Direction of a trade signal or position."""

    LONG = "long"
    SHORT = "short"
    BOTH = "both"

    def __str__(self) -> str:
        return self.value


@unique
class MovingAverageType(str, Enum):
    """Supported moving average calculation methods."""

    SMA = "sma"
    EMA = "ema"

    def __str__(self) -> str:
        return self.value


# ── Type Aliases ─────────────────────────────────────────────────────────────

# Standard OHLCV DataFrame: must have columns [open, high, low, close, volume]
# with a DatetimeIndex.
OHLCVDataFrame = pd.DataFrame

# Parameters passed to block compute() methods.
BlockParams = dict[str, Any]

# Result from a block compute() — typically a DataFrame or Series,
# but some blocks (like exit rules) may return a dict of Series.
BlockResult = pd.DataFrame | pd.Series | dict[str, pd.Series | float]

# Mapping of column name to required dtype for OHLCV validation.
OHLCV_REQUIRED_COLUMNS: dict[str, str] = {
    "open": "float",
    "high": "float",
    "low": "float",
    "close": "float",
    "volume": "float",
}


def validate_ohlcv(df: pd.DataFrame, block_name: str = "unknown") -> None:
    """
    Validate that a DataFrame conforms to the expected OHLCV shape.

    Checks:
        1. Not empty
        2. Has all required columns (case-insensitive; columns are lowered)
        3. Index is a DatetimeIndex
        4. No fully-null required columns

    Args:
        df: The DataFrame to validate.
        block_name: Name of the calling block (for error messages).

    Raises:
        ValueError: If any validation check fails.
    """
    if df.empty:
        raise ValueError(f"[{block_name}] Input DataFrame is empty")

    # Normalize column names to lowercase for consistent access
    df.columns = df.columns.str.lower()

    missing = set(OHLCV_REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{block_name}] Missing required OHLCV columns: {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"[{block_name}] DataFrame index must be a DatetimeIndex, "
            f"got {type(df.index).__name__}"
        )

    for col in OHLCV_REQUIRED_COLUMNS:
        if df[col].isna().all():
            raise ValueError(f"[{block_name}] Column '{col}' is entirely NaN")
```

---

## 1.11 `src/forgequant/core/logging.py`

```python
"""
Structured logging configuration using structlog.

Provides JSON-formatted or human-readable console output depending on
environment settings. All log entries include timestamps, log level,
and caller information.

Usage:
    from forgequant.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("strategy_compiled", strategy_name="ema_crossover", blocks=5)
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

import structlog
from structlog.types import Processor

from forgequant.core.config import LogFormat, get_settings


def _build_shared_processors() -> list[Processor]:
    """
    Build the processor chain shared by both stdlib pre-chain
    and structlog's own chain.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def _build_renderer(log_format: LogFormat) -> Processor:
    """Return the final renderer based on the configured format."""
    if log_format == LogFormat.JSON:
        return structlog.processors.JSONRenderer()
    else:
        return structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=40,
        )


def configure_logging(
    log_level: str | None = None,
    log_format: LogFormat | None = None,
) -> None:
    """
    Configure structlog and stdlib logging for the application.

    Should be called once at application startup. If called without
    arguments, reads configuration from Settings.

    Args:
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Override log format (json or console).
    """
    settings = get_settings()
    level = log_level or settings.forgequant_log_level
    fmt = log_format or settings.forgequant_log_format

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors = _build_shared_processors()

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib root logger
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _build_renderer(fmt),
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Suppress noisy third-party loggers
    for noisy_logger in ("urllib3", "httpcore", "httpx", "chromadb", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


@lru_cache(maxsize=128)
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a named structlog logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
              If None, returns the root logger.

    Returns:
        A bound structlog logger with the configured processors.
    """
    return structlog.get_logger(name or "forgequant")
```

---

## 1.12 `src/forgequant/core/generator/__init__.py`

```python
"""
Strategy generation engine.

Will contain the core loop that:
1. Receives a StrategySpec (from AI Forge or manual input)
2. Assembles blocks from the registry
3. Compiles into a backtestable pipeline
4. Evaluates and applies robustness gates

(Placeholder — implementation in Phase 6)
"""
```

---

## 1.13 `src/forgequant/core/robustness/__init__.py`

```python
"""
Robustness testing suite.

Will contain:
- Walk-forward analysis
- Combinatorial Purged Cross-Validation (CPCV)
- Monte Carlo permutation tests
- Parameter sensitivity analysis
- Equity curve stability checks

(Placeholder — implementation in Phase 7)
"""
```

---

## 1.14 `src/forgequant/core/compiler/__init__.py`

```python
"""
Strategy compiler.

Transforms a validated StrategySpec into a runnable backtest pipeline
by wiring together blocks from the registry and producing vectorized
signal/sizing logic compatible with VectorBT.

(Placeholder — implementation in Phase 6)
"""
```

---

## 1.15 `src/forgequant/blocks/__init__.py`

```python
"""
Composable strategy building blocks.

This package provides the base abstractions (BaseBlock, BlockMetadata),
the central BlockRegistry, and all concrete block implementations
organized by category.

Usage:
    from forgequant.blocks import BlockRegistry, BaseBlock, BlockMetadata
    from forgequant.blocks import BlockCategory
"""

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

__all__ = [
    "BaseBlock",
    "BlockMetadata",
    "BlockRegistry",
    "BlockCategory",
    "ParameterSpec",
]
```

---

## 1.16 `src/forgequant/blocks/metadata.py`

```python
"""
Block metadata and parameter specification.

Every block declares a BlockMetadata instance that describes its identity,
category, parameters, and usage characteristics. This metadata is used by
the registry for search/filtering, by the AI Forge for prompt grounding,
and by the frontend for dynamic UI generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forgequant.core.types import BlockCategory


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """
    Specification for a single block parameter.

    Attributes:
        name: Parameter identifier (must be a valid Python identifier).
        param_type: Expected Python type as a string ("int", "float", "str", "bool").
        default: Default value if the parameter is not provided.
        min_value: Minimum allowed value (for numeric types). None means unbounded.
        max_value: Maximum allowed value (for numeric types). None means unbounded.
        description: Human-readable explanation of what this parameter controls.
        choices: If set, the parameter must be one of these values.
    """

    name: str
    param_type: str
    default: Any
    min_value: float | int | None = None
    max_value: float | int | None = None
    description: str = ""
    choices: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the parameter spec at creation time."""
        if not self.name.isidentifier():
            raise ValueError(
                f"Parameter name '{self.name}' is not a valid Python identifier"
            )

        valid_types = {"int", "float", "str", "bool"}
        if self.param_type not in valid_types:
            raise ValueError(
                f"Parameter type '{self.param_type}' must be one of {valid_types}"
            )

        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(
                    f"Parameter '{self.name}': min_value ({self.min_value}) "
                    f"cannot exceed max_value ({self.max_value})"
                )

        if self.choices is not None and len(self.choices) == 0:
            raise ValueError(
                f"Parameter '{self.name}': choices must be non-empty if provided"
            )

    def validate_value(self, value: Any) -> Any:
        """
        Validate and coerce a value against this parameter's constraints.

        Args:
            value: The value to validate.

        Returns:
            The validated (and possibly coerced) value.

        Raises:
            ValueError: If the value violates any constraint.
        """
        # Type coercion
        type_map: dict[str, type] = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }
        target_type = type_map[self.param_type]

        try:
            coerced = target_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Parameter '{self.name}': cannot convert {value!r} to {self.param_type}"
            ) from e

        # Range validation for numeric types
        if self.param_type in ("int", "float"):
            if self.min_value is not None and coerced < self.min_value:
                raise ValueError(
                    f"Parameter '{self.name}': value {coerced} is below "
                    f"minimum {self.min_value}"
                )
            if self.max_value is not None and coerced > self.max_value:
                raise ValueError(
                    f"Parameter '{self.name}': value {coerced} exceeds "
                    f"maximum {self.max_value}"
                )

        # Choices validation
        if self.choices is not None and coerced not in self.choices:
            raise ValueError(
                f"Parameter '{self.name}': value {coerced!r} is not in "
                f"allowed choices {self.choices}"
            )

        return coerced


@dataclass(frozen=True, slots=True)
class BlockMetadata:
    """
    Complete metadata for a strategy building block.

    Attributes:
        name: Unique block identifier (lowercase, underscore-separated).
        display_name: Human-friendly display name.
        category: Which category this block belongs to.
        description: Detailed description of block behavior.
        parameters: Ordered list of parameter specifications.
        tags: Searchable tags for discovery.
        typical_use: Prose description of when/how this block is typically used.
        version: Semantic version of this block's implementation.
        author: Block author or "forgequant" for built-in blocks.
    """

    name: str
    display_name: str
    category: BlockCategory
    description: str
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    typical_use: str = ""
    version: str = "1.0.0"
    author: str = "forgequant"

    def __post_init__(self) -> None:
        """Validate metadata at creation time."""
        if not self.name:
            raise ValueError("Block name cannot be empty")

        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Block name '{self.name}' must contain only lowercase letters, "
                f"digits, and underscores"
            )

        if self.name != self.name.lower():
            raise ValueError(
                f"Block name '{self.name}' must be lowercase"
            )

        if not self.display_name:
            raise ValueError("Block display_name cannot be empty")

        if not self.description:
            raise ValueError("Block description cannot be empty")

        # Ensure parameter names are unique
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            duplicates = [n for n in param_names if param_names.count(n) > 1]
            raise ValueError(
                f"Block '{self.name}' has duplicate parameter names: {set(duplicates)}"
            )

    def get_parameter(self, name: str) -> ParameterSpec | None:
        """Look up a parameter spec by name. Returns None if not found."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def get_defaults(self) -> dict[str, Any]:
        """Return a dict of all parameter defaults."""
        return {p.name: p.default for p in self.parameters}

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a full set of parameters against this metadata's specs.

        Missing parameters are filled with defaults. Unknown parameters
        are rejected.

        Args:
            params: User-provided parameter values.

        Returns:
            A complete, validated parameter dictionary.

        Raises:
            ValueError: If any parameter is invalid or unknown.
        """
        validated: dict[str, Any] = {}

        # Check for unknown parameters
        known_names = {p.name for p in self.parameters}
        unknown = set(params.keys()) - known_names
        if unknown:
            raise ValueError(
                f"Block '{self.name}': unknown parameters {sorted(unknown)}. "
                f"Known: {sorted(known_names)}"
            )

        # Validate or fill defaults for each spec
        for spec in self.parameters:
            if spec.name in params:
                validated[spec.name] = spec.validate_value(params[spec.name])
            else:
                validated[spec.name] = spec.default

        return validated
```

---

## 1.17 `src/forgequant/blocks/base.py`

```python
"""
Abstract base class for all strategy building blocks.

Every concrete block (indicator, price action pattern, entry rule, exit rule,
money management, filter) must subclass BaseBlock and implement:

    1. metadata (class attribute): a BlockMetadata instance
    2. compute(data, params) -> BlockResult

The base class provides:
    - Automatic OHLCV validation
    - Parameter validation against metadata specs
    - Structured logging on entry/exit of compute()
    - Consistent error wrapping via BlockComputeError
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from forgequant.blocks.metadata import BlockMetadata
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.logging import get_logger
from forgequant.core.types import BlockParams, BlockResult, validate_ohlcv

logger = get_logger(__name__)


class BaseBlock(ABC):
    """
    Abstract base class for all ForgeQuant strategy building blocks.

    Subclasses MUST define:
        metadata: ClassVar[BlockMetadata] — describes the block's identity and params.

    Subclasses MUST implement:
        compute(data, params) — the core computation logic.
    """

    metadata: ClassVar[BlockMetadata]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate that subclasses properly define metadata.

        Called automatically when a class inherits from BaseBlock.
        """
        super().__init_subclass__(**kwargs)

        # Skip validation for intermediate abstract classes
        if getattr(cls, "__abstractmethods__", None):
            return

        if not hasattr(cls, "metadata") or not isinstance(cls.metadata, BlockMetadata):
            raise TypeError(
                f"Block class '{cls.__name__}' must define a 'metadata' "
                f"class attribute of type BlockMetadata"
            )

    @abstractmethod
    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Execute the block's core computation.

        This is the method that subclasses MUST implement. It receives
        already-validated OHLCV data and already-validated parameters.

        Args:
            data: OHLCV DataFrame with lowercase column names and DatetimeIndex.
            params: Validated parameter dictionary matching this block's metadata.

        Returns:
            Computation result — typically a DataFrame with new columns,
            a Series, or a dict of Series/floats depending on block category.

        Raises:
            BlockComputeError: If the computation fails for any reason.
        """
        ...

    def execute(
        self,
        data: pd.DataFrame,
        params: BlockParams | None = None,
    ) -> BlockResult:
        """
        Validate inputs, run compute(), and return results with logging.

        This is the public API that external code should call. It wraps
        compute() with:
            1. OHLCV DataFrame validation
            2. Parameter validation (with defaults for missing params)
            3. Structured log entry with timing
            4. Error wrapping in BlockComputeError

        Args:
            data: OHLCV DataFrame. Columns are normalized to lowercase.
            params: Parameter dictionary. Missing params use defaults.
                    If None, all defaults are used.

        Returns:
            The result from compute().

        Raises:
            ValueError: If OHLCV data is invalid.
            BlockValidationError: If parameters fail validation.
            BlockComputeError: If compute() raises any exception.
        """
        block_name = self.metadata.name
        raw_params = params or {}

        # Step 1: Validate OHLCV data
        data = data.copy()
        data.columns = data.columns.str.lower()
        validate_ohlcv(data, block_name=block_name)

        # Step 2: Validate and fill parameters
        try:
            validated_params = self.metadata.validate_params(raw_params)
        except ValueError as e:
            raise BlockValidationError(
                block_name=block_name,
                param_name="(multiple)",
                value=raw_params,
                constraint=str(e),
            ) from e

        # Step 3: Execute compute with logging
        logger.debug(
            "block_execute_start",
            block=block_name,
            category=str(self.metadata.category),
            params=validated_params,
            data_rows=len(data),
        )

        start_time = time.perf_counter()

        try:
            result = self.compute(data, validated_params)
        except BlockComputeError:
            # Re-raise without wrapping
            raise
        except Exception as e:
            raise BlockComputeError(
                block_name=block_name,
                reason=f"{type(e).__name__}: {e}",
            ) from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            "block_execute_complete",
            block=block_name,
            elapsed_ms=round(elapsed_ms, 2),
            result_type=type(result).__name__,
        )

        return result

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} "
            f"name={self.metadata.name!r} "
            f"category={self.metadata.category.value!r}>"
        )

    def __str__(self) -> str:
        return f"{self.metadata.display_name} ({self.metadata.category.value})"
```

---

## 1.18 `src/forgequant/blocks/registry.py`

```python
"""
Central registry for strategy building blocks.

The BlockRegistry is a singleton that stores all available block classes,
keyed by their metadata.name. It provides:

    - register(): Class decorator to auto-register blocks
    - get() / get_or_raise(): Lookup by name
    - list_by_category(): Filter by BlockCategory
    - search(): Full-text search across name, description, tags, typical_use
    - all_blocks(): Iterator over all registered blocks
    - count(): Number of registered blocks
    - clear(): Remove all registrations (for testing)
"""

from __future__ import annotations

from typing import Iterator

from forgequant.blocks.base import BaseBlock
from forgequant.core.exceptions import BlockNotFoundError, BlockRegistrationError
from forgequant.core.logging import get_logger
from forgequant.core.types import BlockCategory

logger = get_logger(__name__)


class _BlockRegistryMeta(type):
    """
    Metaclass that ensures BlockRegistry is a singleton.

    No matter how many times BlockRegistry() is called, the same instance
    is returned.
    """

    _instance: BlockRegistry | None = None

    def __call__(cls, *args: object, **kwargs: object) -> BlockRegistry:
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class BlockRegistry(metaclass=_BlockRegistryMeta):
    """
    Singleton registry of all available strategy building blocks.

    Blocks register themselves via the @BlockRegistry.register decorator
    or by calling BlockRegistry().register_class(cls) explicitly.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, type[BaseBlock]] = {}

    # ── Registration ─────────────────────────────────────────────────────

    @staticmethod
    def register(cls: type[BaseBlock]) -> type[BaseBlock]:
        """
        Class decorator that registers a block in the global registry.

        Usage:
            @BlockRegistry.register
            class EMAIndicator(BaseBlock):
                metadata = BlockMetadata(...)
                def compute(self, data, params): ...

        Args:
            cls: The block class to register. Must be a concrete subclass
                 of BaseBlock with a valid metadata attribute.

        Returns:
            The class unchanged (so it can be used as a decorator).

        Raises:
            BlockRegistrationError: If the class is invalid or a block
                                     with the same name already exists.
        """
        registry = BlockRegistry()
        registry.register_class(cls)
        return cls

    def register_class(self, cls: type[BaseBlock]) -> None:
        """
        Register a block class explicitly (non-decorator form).

        Args:
            cls: The block class to register.

        Raises:
            BlockRegistrationError: If validation fails or name collides.
        """
        # Validate it's a proper BaseBlock subclass
        if not isinstance(cls, type) or not issubclass(cls, BaseBlock):
            raise BlockRegistrationError(
                block_name=getattr(cls, "__name__", str(cls)),
                reason="Must be a subclass of BaseBlock",
            )

        if not hasattr(cls, "metadata"):
            raise BlockRegistrationError(
                block_name=cls.__name__,
                reason="Missing 'metadata' class attribute",
            )

        name = cls.metadata.name

        # Check for duplicates
        if name in self._blocks:
            existing = self._blocks[name]
            raise BlockRegistrationError(
                block_name=name,
                reason=(
                    f"Already registered by {existing.__name__}. "
                    f"Cannot register {cls.__name__} with the same name."
                ),
            )

        self._blocks[name] = cls

        logger.info(
            "block_registered",
            block=name,
            category=str(cls.metadata.category),
            class_name=cls.__name__,
        )

    # ── Lookup ───────────────────────────────────────────────────────────

    def get(self, name: str) -> type[BaseBlock] | None:
        """
        Look up a block class by name.

        Args:
            name: The block's metadata.name.

        Returns:
            The block class, or None if not found.
        """
        return self._blocks.get(name)

    def get_or_raise(self, name: str) -> type[BaseBlock]:
        """
        Look up a block class by name, raising if not found.

        Args:
            name: The block's metadata.name.

        Returns:
            The block class.

        Raises:
            BlockNotFoundError: If the name is not in the registry.
        """
        cls = self._blocks.get(name)
        if cls is None:
            raise BlockNotFoundError(block_name=name)
        return cls

    def instantiate(self, name: str) -> BaseBlock:
        """
        Look up and instantiate a block by name.

        Args:
            name: The block's metadata.name.

        Returns:
            A new instance of the block.

        Raises:
            BlockNotFoundError: If the name is not in the registry.
        """
        cls = self.get_or_raise(name)
        return cls()

    # ── Filtering & Search ───────────────────────────────────────────────

    def list_by_category(self, category: BlockCategory) -> list[type[BaseBlock]]:
        """
        Return all registered block classes in a given category.

        Args:
            category: The category to filter by.

        Returns:
            List of block classes, sorted by name.
        """
        return sorted(
            (cls for cls in self._blocks.values() if cls.metadata.category == category),
            key=lambda cls: cls.metadata.name,
        )

    def search(self, query: str) -> list[type[BaseBlock]]:
        """
        Search for blocks matching a query string.

        The query is matched (case-insensitive substring) against:
            - metadata.name
            - metadata.display_name
            - metadata.description
            - metadata.tags (each tag)
            - metadata.typical_use

        Args:
            query: The search string.

        Returns:
            List of matching block classes, sorted by name.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        results: list[type[BaseBlock]] = []

        for cls in self._blocks.values():
            meta = cls.metadata
            searchable = " ".join(
                [
                    meta.name,
                    meta.display_name,
                    meta.description,
                    " ".join(meta.tags),
                    meta.typical_use,
                ]
            ).lower()

            if query_lower in searchable:
                results.append(cls)

        return sorted(results, key=lambda c: c.metadata.name)

    # ── Iteration & Info ─────────────────────────────────────────────────

    def all_blocks(self) -> Iterator[type[BaseBlock]]:
        """Iterate over all registered block classes, sorted by name."""
        yield from sorted(self._blocks.values(), key=lambda cls: cls.metadata.name)

    def all_names(self) -> list[str]:
        """Return sorted list of all registered block names."""
        return sorted(self._blocks.keys())

    def count(self) -> int:
        """Return the number of registered blocks."""
        return len(self._blocks)

    def count_by_category(self) -> dict[BlockCategory, int]:
        """Return a count of blocks per category."""
        counts: dict[BlockCategory, int] = {}
        for cls in self._blocks.values():
            cat = cls.metadata.category
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    # ── Maintenance ──────────────────────────────────────────────────────

    def clear(self) -> None:
        """
        Remove all registered blocks.

        WARNING: This is intended only for testing. Do not call in production.
        """
        self._blocks.clear()
        logger.warning("block_registry_cleared")

    def unregister(self, name: str) -> bool:
        """
        Remove a single block from the registry.

        Args:
            name: The block's metadata.name.

        Returns:
            True if the block was found and removed, False if it wasn't registered.
        """
        if name in self._blocks:
            del self._blocks[name]
            logger.info("block_unregistered", block=name)
            return True
        return False

    # ── Dunder ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, name: str) -> bool:
        return name in self._blocks

    def __repr__(self) -> str:
        return f"<BlockRegistry blocks={self.count()}>"
```

---

## 1.19 Empty sub-package `__init__.py` files

These are all the same pattern — a docstring stub. Create each one:

### `src/forgequant/blocks/indicators/__init__.py`

```python
"""
Technical indicator blocks: EMA, RSI, MACD, ADX, ATR, Bollinger Bands,
Ichimoku, Stochastic. (Implementation in Phase 2)
"""
```

### `src/forgequant/blocks/price_action/__init__.py`

```python
"""
Price action pattern blocks: Breakout, Pullback, HigherHighLowerLow,
SupportResistance. (Implementation in Phase 3)
"""
```

### `src/forgequant/blocks/entry_rules/__init__.py`

```python
"""
Entry rule blocks: Crossover, ThresholdCross, Confluence,
ReversalPattern. (Implementation in Phase 3)
"""
```

### `src/forgequant/blocks/exit_rules/__init__.py`

```python
"""
Exit rule blocks: FixedTPSL, TrailingStop, TimeBasedExit,
BreakevenStop. (Implementation in Phase 3)
"""
```

### `src/forgequant/blocks/money_management/__init__.py`

```python
"""
Money management blocks: FixedRisk, VolatilityTargeting,
KellyFractional, ATRBasedSizing. (Implementation in Phase 4)
"""
```

### `src/forgequant/blocks/filters/__init__.py`

```python
"""
Filter blocks: TradingSession, SpreadFilter, MaxDrawdownFilter,
TrendFilter. (Implementation in Phase 4)
"""
```

### `src/forgequant/ai_forge/__init__.py`

```python
"""
AI Forge: LLM-driven strategy specification and RAG grounding.
(Implementation in Phase 5)
"""
```

### `src/forgequant/execution/__init__.py`

```python
"""
Live execution layer (MetaTrader 5 via aiomql).
(Implementation in Phase 8)
"""
```

### `src/forgequant/frontend/__init__.py`

```python
"""
Reflex-based dashboard frontend.
(Implementation in Phase 9)
"""
```

### `src/forgequant/frontend/pages/__init__.py`

```python
"""Dashboard page components."""
```

### `src/forgequant/frontend/components/__init__.py`

```python
"""Reusable UI components."""
```

### `src/forgequant/knowledge_base/__init__.py`

```python
"""
Knowledge base JSON documents for RAG grounding.
(Populate with trading knowledge documents)
"""
```

### `src/forgequant/scripts/__init__.py`

```python
"""Utility scripts for data download, batch runs, etc."""
```

---

## 1.20 Test Suite

### `tests/__init__.py`

```python
"""ForgeQuant test suite."""
```

### `tests/unit/__init__.py`

```python
"""Unit tests."""
```

### `tests/integration/__init__.py`

```python
"""Integration tests."""
```

### `tests/conftest.py`

```python
"""
Shared pytest fixtures for the ForgeQuant test suite.

Provides:
    - sample_ohlcv: A realistic synthetic OHLCV DataFrame
    - clean_registry: Auto-clears the BlockRegistry before/after each test
    - sample_block_class: A minimal concrete BaseBlock for testing
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    Generate a realistic synthetic OHLCV DataFrame with 500 bars.

    Returns a DataFrame with:
        - DatetimeIndex at 1-hour intervals
        - Columns: open, high, low, close, volume
        - Prices in a random-walk pattern starting at 1.1000
        - Realistic high/low spreads
        - Volume as random integers
    """
    np.random.seed(42)
    n_bars = 500

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")

    # Generate a random walk for close prices
    returns = np.random.normal(loc=0.0, scale=0.001, size=n_bars)
    close = 1.1000 * np.exp(np.cumsum(returns))

    # Generate realistic OHLC from close
    spread = np.random.uniform(0.0005, 0.002, size=n_bars)
    high = close + spread * np.random.uniform(0.3, 1.0, size=n_bars)
    low = close - spread * np.random.uniform(0.3, 1.0, size=n_bars)

    # Open is previous close with some gap
    open_prices = np.roll(close, 1) + np.random.normal(0, 0.0002, size=n_bars)
    open_prices[0] = close[0]

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))

    volume = np.random.randint(100, 10000, size=n_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return df


@pytest.fixture(autouse=True)
def clean_registry():
    """
    Automatically clear the BlockRegistry before and after each test.

    This ensures tests don't pollute each other's registry state.
    """
    registry = BlockRegistry()
    registry.clear()
    yield registry
    registry.clear()


@pytest.fixture
def sample_block_class():
    """
    Return a factory that creates minimal concrete BaseBlock classes.

    Usage:
        def test_something(sample_block_class):
            MyBlock = sample_block_class("my_block", BlockCategory.INDICATOR)
            instance = MyBlock()
            result = instance.execute(ohlcv_data)
    """

    def _factory(
        name: str = "test_block",
        category: BlockCategory = BlockCategory.INDICATOR,
        parameters: tuple[ParameterSpec, ...] = (),
        compute_fn: Any = None,
    ) -> type[BaseBlock]:
        """
        Create a concrete BaseBlock subclass with the given configuration.

        Args:
            name: Block name.
            category: Block category.
            parameters: Parameter specifications.
            compute_fn: Optional custom compute function.
                        Signature: (self, data, params) -> BlockResult
                        Defaults to returning data["close"].rename(name).
        """
        meta = BlockMetadata(
            name=name,
            display_name=name.replace("_", " ").title(),
            category=category,
            description=f"Test block: {name}",
            parameters=parameters,
            tags=("test",),
        )

        def default_compute(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            return data["close"].rename(name)

        fn = compute_fn if compute_fn is not None else default_compute

        # Dynamically create the class
        block_cls = type(
            f"TestBlock_{name}",
            (BaseBlock,),
            {
                "metadata": meta,
                "compute": fn,
            },
        )

        return block_cls

    return _factory
```

---

### `tests/unit/test_config.py`

```python
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
```

---

### `tests/unit/test_logging.py`

```python
"""Tests for forgequant.core.logging."""

from __future__ import annotations

import structlog

from forgequant.core.config import LogFormat
from forgequant.core.logging import configure_logging, get_logger


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_configure_json_format(self) -> None:
        """Should configure without errors in JSON mode."""
        configure_logging(log_level="DEBUG", log_format=LogFormat.JSON)

    def test_configure_console_format(self) -> None:
        """Should configure without errors in console mode."""
        configure_logging(log_level="INFO", log_format=LogFormat.CONSOLE)

    def test_configure_defaults(self) -> None:
        """Should work with default settings."""
        configure_logging()


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_bound_logger(self) -> None:
        get_logger.cache_clear()
        logger = get_logger("test_module")
        assert isinstance(logger, structlog.stdlib.BoundLogger)

    def test_returns_root_logger_when_none(self) -> None:
        get_logger.cache_clear()
        logger = get_logger(None)
        assert isinstance(logger, structlog.stdlib.BoundLogger)

    def test_caching(self) -> None:
        get_logger.cache_clear()
        l1 = get_logger("cached_test")
        l2 = get_logger("cached_test")
        assert l1 is l2

    def test_different_names_different_loggers(self) -> None:
        get_logger.cache_clear()
        l1 = get_logger("module_a")
        l2 = get_logger("module_b")
        # They should be distinct objects
        assert l1 is not l2

    def test_logger_can_log(self, capsys: object) -> None:
        """Verify the logger can actually emit a log entry without crashing."""
        configure_logging(log_level="DEBUG", log_format=LogFormat.CONSOLE)
        get_logger.cache_clear()
        logger = get_logger("emit_test")
        # Should not raise
        logger.info("test_event", key="value")
```

---

### `tests/unit/test_exceptions.py`

```python
"""Tests for forgequant.core.exceptions."""

from __future__ import annotations

import pytest

from forgequant.core.exceptions import (
    BlockComputeError,
    BlockNotFoundError,
    BlockRegistrationError,
    BlockValidationError,
    ConfigurationError,
    ForgeQuantError,
    RobustnessError,
    StrategyCompileError,
)


class TestForgeQuantError:
    """Tests for the base exception."""

    def test_message(self) -> None:
        err = ForgeQuantError("something broke")
        assert str(err) == "something broke"
        assert err.message == "something broke"

    def test_details_default_empty(self) -> None:
        err = ForgeQuantError("oops")
        assert err.details == {}

    def test_details_provided(self) -> None:
        err = ForgeQuantError("oops", details={"foo": "bar"})
        assert err.details == {"foo": "bar"}

    def test_repr_without_details(self) -> None:
        err = ForgeQuantError("msg")
        assert "ForgeQuantError" in repr(err)
        assert "msg" in repr(err)

    def test_repr_with_details(self) -> None:
        err = ForgeQuantError("msg", details={"k": "v"})
        r = repr(err)
        assert "details" in r
        assert "k" in r

    def test_is_exception(self) -> None:
        err = ForgeQuantError("x")
        assert isinstance(err, Exception)

    def test_catch_as_base(self) -> None:
        """All custom exceptions should be catchable as ForgeQuantError."""
        with pytest.raises(ForgeQuantError):
            raise BlockNotFoundError("test")


class TestBlockNotFoundError:
    def test_attributes(self) -> None:
        err = BlockNotFoundError("ema")
        assert err.block_name == "ema"
        assert "ema" in err.message
        assert err.details["block_name"] == "ema"

    def test_inherits_base(self) -> None:
        assert issubclass(BlockNotFoundError, ForgeQuantError)


class TestBlockRegistrationError:
    def test_attributes(self) -> None:
        err = BlockRegistrationError("rsi", "duplicate name")
        assert err.block_name == "rsi"
        assert err.reason == "duplicate name"
        assert "rsi" in err.message
        assert "duplicate" in err.message


class TestBlockComputeError:
    def test_attributes(self) -> None:
        err = BlockComputeError("macd", "division by zero")
        assert err.block_name == "macd"
        assert err.reason == "division by zero"


class TestBlockValidationError:
    def test_attributes(self) -> None:
        err = BlockValidationError(
            block_name="ema",
            param_name="period",
            value=-5,
            constraint="must be >= 2",
        )
        assert err.block_name == "ema"
        assert err.param_name == "period"
        assert err.value == -5
        assert err.constraint == "must be >= 2"
        assert "ema" in err.message
        assert "period" in err.message


class TestConfigurationError:
    def test_message(self) -> None:
        err = ConfigurationError("missing API key")
        assert err.message == "missing API key"


class TestStrategyCompileError:
    def test_attributes(self) -> None:
        err = StrategyCompileError("trend_follow", "missing exit block")
        assert err.strategy_name == "trend_follow"
        assert err.reason == "missing exit block"


class TestRobustnessError:
    def test_attributes(self) -> None:
        err = RobustnessError("trend_follow", "walk_forward", "degraded sharpe")
        assert err.strategy_name == "trend_follow"
        assert err.gate_name == "walk_forward"
        assert err.reason == "degraded sharpe"
```

---

### `tests/unit/test_types.py`

```python
"""Tests for forgequant.core.types."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.core.types import (
    BlockCategory,
    MovingAverageType,
    TimeFrame,
    TradeDirection,
    validate_ohlcv,
)


class TestBlockCategory:
    def test_all_values(self) -> None:
        assert BlockCategory.INDICATOR.value == "indicator"
        assert BlockCategory.PRICE_ACTION.value == "price_action"
        assert BlockCategory.ENTRY_RULE.value == "entry_rule"
        assert BlockCategory.EXIT_RULE.value == "exit_rule"
        assert BlockCategory.MONEY_MANAGEMENT.value == "money_management"
        assert BlockCategory.FILTER.value == "filter"

    def test_count(self) -> None:
        assert len(BlockCategory) == 6

    def test_str(self) -> None:
        assert str(BlockCategory.INDICATOR) == "indicator"


class TestTimeFrame:
    def test_all_values(self) -> None:
        expected = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"}
        actual = {tf.value for tf in TimeFrame}
        assert actual == expected

    def test_str(self) -> None:
        assert str(TimeFrame.H1) == "1h"


class TestTradeDirection:
    def test_values(self) -> None:
        assert TradeDirection.LONG.value == "long"
        assert TradeDirection.SHORT.value == "short"
        assert TradeDirection.BOTH.value == "both"


class TestMovingAverageType:
    def test_values(self) -> None:
        assert MovingAverageType.SMA.value == "sma"
        assert MovingAverageType.EMA.value == "ema"


class TestValidateOhlcv:
    """Tests for the validate_ohlcv function."""

    def _make_valid_df(self, n: int = 50) -> pd.DataFrame:
        """Create a minimal valid OHLCV DataFrame."""
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(0)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame(
            {
                "open": close + np.random.randn(n) * 0.1,
                "high": close + abs(np.random.randn(n) * 0.5),
                "low": close - abs(np.random.randn(n) * 0.5),
                "close": close,
                "volume": np.random.randint(100, 1000, n).astype(float),
            },
            index=dates,
        )

    def test_valid_df_passes(self) -> None:
        df = self._make_valid_df()
        validate_ohlcv(df)  # Should not raise

    def test_empty_df_raises(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            validate_ohlcv(df)

    def test_missing_column_raises(self) -> None:
        df = self._make_valid_df()
        df = df.drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            validate_ohlcv(df)

    def test_wrong_index_type_raises(self) -> None:
        df = self._make_valid_df()
        df.index = range(len(df))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_ohlcv(df)

    def test_all_nan_column_raises(self) -> None:
        df = self._make_valid_df()
        df["close"] = np.nan
        with pytest.raises(ValueError, match="entirely NaN"):
            validate_ohlcv(df)

    def test_uppercase_columns_normalized(self) -> None:
        """Columns should be lowered automatically."""
        df = self._make_valid_df()
        df.columns = df.columns.str.upper()
        validate_ohlcv(df)  # Should not raise

    def test_block_name_in_error(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="my_block"):
            validate_ohlcv(df, block_name="my_block")
```

---

### `tests/unit/test_metadata.py`

```python
"""Tests for forgequant.blocks.metadata."""

from __future__ import annotations

import pytest

from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.core.types import BlockCategory


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_valid_creation(self) -> None:
        p = ParameterSpec(
            name="period",
            param_type="int",
            default=14,
            min_value=2,
            max_value=500,
            description="Lookback period",
        )
        assert p.name == "period"
        assert p.param_type == "int"
        assert p.default == 14
        assert p.min_value == 2
        assert p.max_value == 500

    def test_invalid_name_not_identifier(self) -> None:
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            ParameterSpec(name="123bad", param_type="int", default=1)

    def test_invalid_param_type(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            ParameterSpec(name="x", param_type="list", default=[])

    def test_min_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            ParameterSpec(name="x", param_type="int", default=5, min_value=10, max_value=5)

    def test_empty_choices(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ParameterSpec(name="x", param_type="str", default="a", choices=())

    def test_validate_value_int(self) -> None:
        p = ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500)
        assert p.validate_value(20) == 20
        assert p.validate_value("30") == 30  # Coercion from string

    def test_validate_value_below_min(self) -> None:
        p = ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500)
        with pytest.raises(ValueError, match="below minimum"):
            p.validate_value(1)

    def test_validate_value_above_max(self) -> None:
        p = ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500)
        with pytest.raises(ValueError, match="exceeds maximum"):
            p.validate_value(501)

    def test_validate_value_choices(self) -> None:
        p = ParameterSpec(
            name="method",
            param_type="str",
            default="ema",
            choices=("sma", "ema"),
        )
        assert p.validate_value("sma") == "sma"
        with pytest.raises(ValueError, match="not in allowed choices"):
            p.validate_value("wma")

    def test_validate_value_type_error(self) -> None:
        p = ParameterSpec(name="count", param_type="int", default=5)
        with pytest.raises(ValueError, match="cannot convert"):
            p.validate_value("not_a_number")

    def test_validate_float(self) -> None:
        p = ParameterSpec(
            name="multiplier", param_type="float", default=2.0, min_value=0.5, max_value=5.0
        )
        assert p.validate_value(3.5) == 3.5
        assert p.validate_value(1) == 1.0  # int -> float

    def test_validate_bool(self) -> None:
        p = ParameterSpec(name="use_ema", param_type="bool", default=True)
        assert p.validate_value(True) is True
        assert p.validate_value(False) is False


class TestBlockMetadata:
    """Tests for BlockMetadata dataclass."""

    def _make_metadata(self, **kwargs: object) -> BlockMetadata:
        """Helper to create BlockMetadata with sensible defaults."""
        defaults = dict(
            name="test_block",
            display_name="Test Block",
            category=BlockCategory.INDICATOR,
            description="A test block for testing",
        )
        defaults.update(kwargs)
        return BlockMetadata(**defaults)  # type: ignore[arg-type]

    def test_valid_creation(self) -> None:
        meta = self._make_metadata()
        assert meta.name == "test_block"
        assert meta.display_name == "Test Block"
        assert meta.category == BlockCategory.INDICATOR
        assert meta.version == "1.0.0"
        assert meta.author == "forgequant"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self._make_metadata(name="")

    def test_non_lowercase_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must be lowercase"):
            self._make_metadata(name="TestBlock")

    def test_invalid_chars_in_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain only"):
            self._make_metadata(name="test-block")

    def test_empty_display_name_raises(self) -> None:
        with pytest.raises(ValueError, match="display_name cannot be empty"):
            self._make_metadata(display_name="")

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description cannot be empty"):
            self._make_metadata(description="")

    def test_duplicate_parameter_names_raises(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
            ParameterSpec(name="period", param_type="int", default=20),
        )
        with pytest.raises(ValueError, match="duplicate parameter names"):
            self._make_metadata(parameters=params)

    def test_get_parameter_found(self) -> None:
        params = (ParameterSpec(name="period", param_type="int", default=14),)
        meta = self._make_metadata(parameters=params)
        p = meta.get_parameter("period")
        assert p is not None
        assert p.name == "period"

    def test_get_parameter_not_found(self) -> None:
        meta = self._make_metadata()
        assert meta.get_parameter("nonexistent") is None

    def test_get_defaults(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
            ParameterSpec(name="multiplier", param_type="float", default=2.0),
        )
        meta = self._make_metadata(parameters=params)
        defaults = meta.get_defaults()
        assert defaults == {"period": 14, "multiplier": 2.0}

    def test_validate_params_defaults(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
        )
        meta = self._make_metadata(parameters=params)
        result = meta.validate_params({})
        assert result == {"period": 14}

    def test_validate_params_override(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500),
        )
        meta = self._make_metadata(parameters=params)
        result = meta.validate_params({"period": 20})
        assert result == {"period": 20}

    def test_validate_params_unknown_raises(self) -> None:
        meta = self._make_metadata()
        with pytest.raises(ValueError, match="unknown parameters"):
            meta.validate_params({"bogus": 42})

    def test_validate_params_invalid_value_raises(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500),
        )
        meta = self._make_metadata(parameters=params)
        with pytest.raises(ValueError, match="below minimum"):
            meta.validate_params({"period": 1})
```

---

### `tests/unit/test_base_block.py`

```python
"""Tests for forgequant.blocks.base.BaseBlock."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


class TestBaseBlockSubclassing:
    """Tests for BaseBlock's __init_subclass__ validation."""

    def test_concrete_without_metadata_raises(self) -> None:
        """A concrete subclass without metadata should raise TypeError."""
        with pytest.raises(TypeError, match="must define a 'metadata'"):

            class BadBlock(BaseBlock):
                def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
                    return data["close"]

            # Force the check by trying to use the class
            _ = BadBlock

    def test_concrete_with_metadata_succeeds(self, sample_block_class: Any) -> None:
        """A properly defined concrete subclass should work fine."""
        MyBlock = sample_block_class("valid_block")
        instance = MyBlock()
        assert instance.metadata.name == "valid_block"


class TestBaseBlockExecute:
    """Tests for the execute() public API."""

    def test_execute_with_defaults(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """Execute with no params should use defaults and return a result."""
        MyBlock = sample_block_class("exec_test")
        instance = MyBlock()
        result = instance.execute(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_execute_validates_ohlcv(self, sample_block_class: Any) -> None:
        """Execute should reject an empty DataFrame."""
        MyBlock = sample_block_class("empty_test")
        instance = MyBlock()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            instance.execute(empty_df)

    def test_execute_validates_params(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """Execute should reject unknown parameters."""
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
        )
        MyBlock = sample_block_class("param_test", parameters=params)
        instance = MyBlock()
        with pytest.raises(BlockValidationError):
            instance.execute(sample_ohlcv, {"unknown_param": 42})

    def test_execute_fills_defaults(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """Missing params should be filled with defaults."""
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
        )

        def compute_fn(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            # Return the period value as verification
            return pd.Series([params["period"]] * len(data), index=data.index)

        MyBlock = sample_block_class(
            "default_test",
            parameters=params,
            compute_fn=compute_fn,
        )
        instance = MyBlock()
        result = instance.execute(sample_ohlcv)
        assert result.iloc[0] == 14

    def test_execute_wraps_exception(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """Exceptions in compute() should be wrapped in BlockComputeError."""

        def bad_compute(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            raise ZeroDivisionError("boom")

        MyBlock = sample_block_class("error_test", compute_fn=bad_compute)
        instance = MyBlock()
        with pytest.raises(BlockComputeError, match="boom"):
            instance.execute(sample_ohlcv)

    def test_execute_preserves_block_compute_error(
        self, sample_ohlcv: pd.DataFrame, sample_block_class: Any
    ) -> None:
        """BlockComputeError raised in compute() should not be double-wrapped."""

        def deliberate_error(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            raise BlockComputeError("deliberate", "intentional failure")

        MyBlock = sample_block_class("preserve_test", compute_fn=deliberate_error)
        instance = MyBlock()
        with pytest.raises(BlockComputeError, match="intentional failure"):
            instance.execute(sample_ohlcv)

    def test_execute_normalizes_columns(self, sample_block_class: Any) -> None:
        """Uppercase column names in input should be lowered automatically."""
        import numpy as np

        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "Open": np.ones(10),
                "High": np.ones(10) * 1.1,
                "Low": np.ones(10) * 0.9,
                "Close": np.ones(10),
                "Volume": np.ones(10) * 100,
            },
            index=dates,
        )

        MyBlock = sample_block_class("case_test")
        instance = MyBlock()
        result = instance.execute(df)
        assert isinstance(result, pd.Series)


class TestBaseBlockRepr:
    """Tests for __repr__ and __str__."""

    def test_repr(self, sample_block_class: Any) -> None:
        MyBlock = sample_block_class("repr_test")
        instance = MyBlock()
        r = repr(instance)
        assert "repr_test" in r
        assert "indicator" in r

    def test_str(self, sample_block_class: Any) -> None:
        MyBlock = sample_block_class("str_test")
        instance = MyBlock()
        s = str(instance)
        assert "Str Test" in s
        assert "indicator" in s
```

---

### `tests/unit/test_registry.py`

```python
"""Tests for forgequant.blocks.registry.BlockRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockNotFoundError, BlockRegistrationError
from forgequant.core.types import BlockCategory


class TestBlockRegistrySingleton:
    """Tests for the singleton behavior."""

    def test_singleton(self) -> None:
        r1 = BlockRegistry()
        r2 = BlockRegistry()
        assert r1 is r2

    def test_len_empty(self, clean_registry: BlockRegistry) -> None:
        assert len(clean_registry) == 0

    def test_contains_empty(self, clean_registry: BlockRegistry) -> None:
        assert "nonexistent" not in clean_registry

    def test_repr(self, clean_registry: BlockRegistry) -> None:
        r = repr(clean_registry)
        assert "BlockRegistry" in r
        assert "blocks=0" in r


class TestBlockRegistration:
    """Tests for registering blocks."""

    def test_register_decorator(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("decorator_test")
        BlockRegistry.register(MyBlock)
        assert "decorator_test" in clean_registry
        assert clean_registry.count() == 1

    def test_register_class_method(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("method_test")
        clean_registry.register_class(MyBlock)
        assert "method_test" in clean_registry

    def test_register_duplicate_raises(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("dupe_test")
        clean_registry.register_class(MyBlock)
        MyBlock2 = sample_block_class("dupe_test")
        with pytest.raises(BlockRegistrationError, match="Already registered"):
            clean_registry.register_class(MyBlock2)

    def test_register_non_baseblock_raises(self, clean_registry: BlockRegistry) -> None:
        with pytest.raises(BlockRegistrationError, match="subclass of BaseBlock"):
            clean_registry.register_class(str)  # type: ignore[arg-type]

    def test_register_without_metadata_raises(self, clean_registry: BlockRegistry) -> None:
        class BareBlock(BaseBlock):
            pass  # No metadata, no compute

        with pytest.raises((BlockRegistrationError, TypeError)):
            clean_registry.register_class(BareBlock)


class TestBlockLookup:
    """Tests for get() and get_or_raise()."""

    def test_get_found(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("lookup_test")
        clean_registry.register_class(MyBlock)
        result = clean_registry.get("lookup_test")
        assert result is MyBlock

    def test_get_not_found(self, clean_registry: BlockRegistry) -> None:
        assert clean_registry.get("ghost") is None

    def test_get_or_raise_found(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("raise_test")
        clean_registry.register_class(MyBlock)
        result = clean_registry.get_or_raise("raise_test")
        assert result is MyBlock

    def test_get_or_raise_not_found(self, clean_registry: BlockRegistry) -> None:
        with pytest.raises(BlockNotFoundError, match="ghost"):
            clean_registry.get_or_raise("ghost")

    def test_instantiate(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("inst_test")
        clean_registry.register_class(MyBlock)
        instance = clean_registry.instantiate("inst_test")
        assert isinstance(instance, BaseBlock)
        assert instance.metadata.name == "inst_test"


class TestBlockFiltering:
    """Tests for list_by_category() and search()."""

    def test_list_by_category(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        Ind1 = sample_block_class("ind_alpha", BlockCategory.INDICATOR)
        Ind2 = sample_block_class("ind_beta", BlockCategory.INDICATOR)
        Flt1 = sample_block_class("flt_one", BlockCategory.FILTER)

        clean_registry.register_class(Ind1)
        clean_registry.register_class(Ind2)
        clean_registry.register_class(Flt1)

        indicators = clean_registry.list_by_category(BlockCategory.INDICATOR)
        assert len(indicators) == 2
        assert indicators[0].metadata.name == "ind_alpha"
        assert indicators[1].metadata.name == "ind_beta"

        filters = clean_registry.list_by_category(BlockCategory.FILTER)
        assert len(filters) == 1

        exits = clean_registry.list_by_category(BlockCategory.EXIT_RULE)
        assert len(exits) == 0

    def test_search_by_name(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("ema_indicator")
        clean_registry.register_class(MyBlock)
        results = clean_registry.search("ema")
        assert len(results) == 1
        assert results[0].metadata.name == "ema_indicator"

    def test_search_case_insensitive(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("bollinger_bands")
        clean_registry.register_class(MyBlock)
        results = clean_registry.search("BOLLINGER")
        assert len(results) == 1

    def test_search_empty_query(self, clean_registry: BlockRegistry) -> None:
        assert clean_registry.search("") == []
        assert clean_registry.search("   ") == []

    def test_search_no_match(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        MyBlock = sample_block_class("rsi_indicator")
        clean_registry.register_class(MyBlock)
        results = clean_registry.search("ichimoku")
        assert len(results) == 0


class TestBlockIteration:
    """Tests for all_blocks(), all_names(), count()."""

    def test_all_blocks_sorted(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        B1 = sample_block_class("zebra")
        B2 = sample_block_class("alpha")
        clean_registry.register_class(B1)
        clean_registry.register_class(B2)

        names = [cls.metadata.name for cls in clean_registry.all_blocks()]
        assert names == ["alpha", "zebra"]

    def test_all_names(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        B1 = sample_block_class("bravo")
        B2 = sample_block_class("alpha")
        clean_registry.register_class(B1)
        clean_registry.register_class(B2)
        assert clean_registry.all_names() == ["alpha", "bravo"]

    def test_count(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        assert clean_registry.count() == 0
        B1 = sample_block_class("one")
        clean_registry.register_class(B1)
        assert clean_registry.count() == 1

    def test_count_by_category(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        I1 = sample_block_class("i1", BlockCategory.INDICATOR)
        I2 = sample_block_class("i2", BlockCategory.INDICATOR)
        F1 = sample_block_class("f1", BlockCategory.FILTER)
        clean_registry.register_class(I1)
        clean_registry.register_class(I2)
        clean_registry.register_class(F1)

        counts = clean_registry.count_by_category()
        assert counts[BlockCategory.INDICATOR] == 2
        assert counts[BlockCategory.FILTER] == 1
        assert BlockCategory.EXIT_RULE not in counts


class TestBlockRegistryMaintenance:
    """Tests for clear() and unregister()."""

    def test_clear(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        B1 = sample_block_class("clear_me")
        clean_registry.register_class(B1)
        assert clean_registry.count() == 1
        clean_registry.clear()
        assert clean_registry.count() == 0

    def test_unregister_existing(
        self, clean_registry: BlockRegistry, sample_block_class: Any
    ) -> None:
        B1 = sample_block_class("removable")
        clean_registry.register_class(B1)
        result = clean_registry.unregister("removable")
        assert result is True
        assert "removable" not in clean_registry

    def test_unregister_nonexistent(self, clean_registry: BlockRegistry) -> None:
        result = clean_registry.unregister("ghost")
        assert result is False
```

---

## 1.21 How to Verify Phase 1

Open a terminal in the project root and run:

```bash
# 1. Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install the project in editable mode with dev dependencies
pip install -e ".[dev]"

# 3. Run the full test suite
pytest -v

# 4. Run type checking
mypy src/forgequant/core/ src/forgequant/blocks/

# 5. Run linting
ruff check src/ tests/
ruff format --check src/ tests/
```

**Expected pytest output:** All tests in `tests/unit/` should pass — approximately **60+ tests** across 7 test modules with 100% of Phase 1 code covered.

---

## What Phase 1 Delivers

| Component | File | Status |
|-----------|------|--------|
| Project metadata & deps | `pyproject.toml` | ✅ Complete |
| Environment config | `.env.example`, `core/config.py` | ✅ Complete |
| Structured logging | `core/logging.py` | ✅ Complete |
| Exception hierarchy | `core/exceptions.py` | ✅ Complete |
| Shared types & enums | `core/types.py` | ✅ Complete |
| OHLCV validation | `core/types.py::validate_ohlcv` | ✅ Complete |
| Parameter specification | `blocks/metadata.py::ParameterSpec` | ✅ Complete |
| Block metadata | `blocks/metadata.py::BlockMetadata` | ✅ Complete |
| Abstract base block | `blocks/base.py::BaseBlock` | ✅ Complete |
| Singleton registry | `blocks/registry.py::BlockRegistry` | ✅ Complete |
| Test suite (7 files) | `tests/unit/*` | ✅ Complete |
| Test fixtures | `tests/conftest.py` | ✅ Complete |

---

**Ready for Phase 2** — say the word and I'll write all 8 indicator blocks (EMA, RSI, MACD, ADX, ATR, Bollinger Bands, Ichimoku, Stochastic) with full implementations and complete test coverage for each.
