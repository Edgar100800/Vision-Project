"""
Logging utilities for the store monitoring system.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_event(
    logger: logging.Logger,
    event_type: str,
    person_id: int,
    timestamp: Optional[datetime] = None,
    additional_info: Optional[dict] = None
):
    """
    Log an event with structured information.

    Args:
        logger: Logger instance
        event_type: Type of event (entrance, exit, queue_enter, etc.)
        person_id: Unique person identifier
        timestamp: Event timestamp (defaults to now)
        additional_info: Additional information dictionary
    """
    if timestamp is None:
        timestamp = datetime.now()

    event_info = {
        "event": event_type,
        "person_id": person_id,
        "timestamp": timestamp.isoformat(),
    }

    if additional_info:
        event_info.update(additional_info)

    logger.info(f"Event: {event_info}")


# Create default logger
default_logger = setup_logger("store_monitor", "logs/system.log")
