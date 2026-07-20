"""Logging setup: one timestamped log file per run under ``_crop_aerial_logs/``,
plus a console stream. Uses the standard :mod:`logging` module (not bare prints)
so downstream tooling can filter/parse pipeline logs.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_LOGGER_NAME = "crop_aerial_pipeline"


def setup_logger(logs_dir: Path, run_id: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or return, if already configured) the pipeline's shared logger.

    A fresh call with a new ``run_id`` adds a new file handler pointed at
    ``logs_dir/<run_id>.log`` without disturbing handlers from a previous run in
    the same process (relevant for notebooks, where the module stays imported
    across cells/re-runs).
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = logs_dir / f"{run_id}.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Only attach one console handler total, even across multiple setup_logger()
    # calls in the same process -- avoids duplicated console output in notebooks.
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console_handler:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    logger.info("Logging to %s", log_path)
    return logger


def get_logger() -> logging.Logger:
    """Fetch the pipeline logger without configuring it (falls back to a bare
    stdlib logger with no handlers if ``setup_logger`` hasn't run yet -- Python's
    logging module still works, it just won't have written a file yet)."""
    return logging.getLogger(_LOGGER_NAME)


def new_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")
