"""
Structured Logging with RotatingFileHandler and Console StreamHandler.
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from backend.app.config import settings

_logger_configured = False


def setup_logging() -> logging.Logger:
    """Configures application logger with rotating file and stdout handlers."""
    global _logger_configured

    logger = logging.getLogger("sentiment_analyzer")

    if _logger_configured:
        return logger

    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False

    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Stream Handler (Container / Terminal output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Rotating File Handler
    try:
        log_file_path = os.path.abspath(settings.LOG_FILE_PATH)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize rotating file handler at '{settings.LOG_FILE_PATH}': {e}")

    _logger_configured = True
    logger.info(f"Logging initialized at level {settings.LOG_LEVEL} (File: {settings.LOG_FILE_PATH})")
    return logger


def get_logger(name: str = "sentiment_analyzer") -> logging.Logger:
    """Convenience getter for configured logger."""
    if not _logger_configured:
        return setup_logging()
    return logging.getLogger(name)
