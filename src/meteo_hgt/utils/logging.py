"""Minimal logging helpers."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "meteo_hgt", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
