"""
ForgeQuant Structured Logging.

Uses structlog for JSON-formatted, timestamped, context-rich logs.
All modules should import the logger from here.

Usage:
    from core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Strategy generated", strategy_name="MomentumBreakout", blocks=5)
"""

import sys
import structlog
from core.config import settings


def _configure_logging() -> None:
    """Configure structlog with appropriate processors based on log level."""

    log_level = settings.log_level.upper()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(indent=2),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


# Run configuration on import
_configure_logging()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger instance.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A structlog BoundLogger instance with the module name bound.
    """
    return structlog.get_logger(module=name)
