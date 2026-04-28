from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a standard project checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model weights and return the full checkpoint dictionary."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
