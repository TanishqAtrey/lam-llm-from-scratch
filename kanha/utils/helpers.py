"""
kanha/utils/helpers.py
Common utility functions.
"""

import os
import torch


def get_device():
    """Returns the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model, trainable_only=True):
    """Counts model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path):
    """Creates directory if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)
