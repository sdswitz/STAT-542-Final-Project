from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Choose the best available training device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_device_type(device: torch.device) -> str:
    """Return the device type string expected by torch.autocast."""
    return "cuda" if device.type == "cuda" else "cpu"
