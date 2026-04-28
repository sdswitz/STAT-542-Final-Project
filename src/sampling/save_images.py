from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from src.data.transforms import generated_to_display_range


def save_sample_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """Save a generated image grid to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display_images = generated_to_display_range(images)
    grid = make_grid(display_images, nrow=nrow)
    save_image(grid, path)
