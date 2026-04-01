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

    def test_returns_bound_logger(self, capsys: object) -> None:
        configure_logging(log_level="DEBUG", log_format=LogFormat.CONSOLE)
        get_logger.cache_clear()
        structlog.reset_defaults()
        logger = get_logger("test_module")
        # Verify it can log without raising
        logger.info("hello", key="value")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_returns_root_logger_when_none(self, capsys: object) -> None:
        configure_logging(log_level="DEBUG", log_format=LogFormat.CONSOLE)
        get_logger.cache_clear()
        structlog.reset_defaults()
        logger = get_logger(None)
        # Verify it can log without raising
        logger.info("root_event")
        captured = capsys.readouterr()
        assert "root_event" in captured.out

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
