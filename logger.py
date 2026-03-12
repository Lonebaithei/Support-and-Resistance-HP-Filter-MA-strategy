"""
==============================================================================
utils/logger.py  —  Structured Logging
==============================================================================
"""

import os
import logging
from datetime import datetime


def setup_logger(name:      str = "HPSRBot",
                 log_dir:   str = "logs",
                 log_level: str = "INFO") -> logging.Logger:
    """
    Configure a logger that writes to both console and a daily rotating file.
    """
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"bot_{datetime.now().strftime('%Y%m%d')}.log"
    )

    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
