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
            pad_event_to=40,
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
