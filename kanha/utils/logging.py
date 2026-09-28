"""
kanha/utils/logging.py
Logging utilities and banner printer.
"""

import logging
import sys


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Returns a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def print_banner():
    """Prints the KANHA ASCII banner."""
    banner = r"""
    ╔═══════════════════════════════════════════╗
    ║   KANHA AI                                ║
    ║   Knowledge-Augmented Neural Heuristic    ║
    ║   Assistant                               ║
    ╚═══════════════════════════════════════════╝
    """
    print(banner)
