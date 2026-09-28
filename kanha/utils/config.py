"""
kanha/utils/config.py
YAML configuration loader.

Loads config.yaml from project root and exposes it as a
dot-accessible namespace (cfg.model.dim, cfg.training.lr, etc.)
"""

import os
import yaml
from types import SimpleNamespace


def _to_namespace(d):
    """Recursively converts a dict to a dot-accessible SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d


def _find_config():
    """Walks up from this file to find config.yaml in the project root."""
    # Try common locations
    candidates = [
        os.path.join(os.getcwd(), "config.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "config.yaml not found. Make sure you're running from the project root."
    )


def load_config(path=None):
    """Loads config.yaml and returns a dot-accessible namespace."""
    if path is None:
        path = _find_config()
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


# Module-level singleton — import as: from kanha.utils.config import cfg
cfg = load_config()
