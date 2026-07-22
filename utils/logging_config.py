"""Centralized logging utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings

_CONFIGURED: bool = False


def setup_logging(
    name: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a named logger with file and console handlers.

    Args:
        name: Logger name. Defaults to ``ai_platform``.
        level: Optional logging level override (e.g. ``DEBUG``).

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        ValueError: If an invalid log level is provided.
        OSError: If the log file cannot be created.
    """
    global _CONFIGURED

    settings = get_settings()
    logger_name = name or "ai_platform"
    logger = logging.getLogger(logger_name)

    try:
        log_level = getattr(logging, (level or settings.log_level).upper())
    except AttributeError as exc:
        raise ValueError(f"Invalid log level: {level}") from exc

    logger.setLevel(log_level)

    if not _CONFIGURED:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        log_file: Path = settings.logs_dir / "platform.log"
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        except OSError as exc:
            raise OSError(f"Cannot open log file {log_file}: {exc}") from exc

        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        root = logging.getLogger("ai_platform")
        root.setLevel(log_level)
        root.handlers.clear()
        root.addHandler(console_handler)
        root.addHandler(file_handler)
        root.propagate = False
        _CONFIGURED = True

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the platform root logger.

    Args:
        name: Short module / agent name.

    Returns:
        logging.Logger: Child logger.
    """
    setup_logging()
    return logging.getLogger(f"ai_platform.{name}")
