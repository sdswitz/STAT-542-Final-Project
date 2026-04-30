from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from src.data.transforms import generated_to_display_range


def save_image_batch(
    images: torch.Tensor,
    output_dir: str | Path,
    *,
    start_index: int = 0,
    filename_width: int = 6,
) -> None:
    """Save a batch of generated images as individual PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    display_images = generated_to_display_range(images).detach().cpu()

    for offset, image in enumerate(display_images):
        image_index = start_index + offset
        save_image(image, output_dir / f"{image_index:0{filename_width}d}.png")


def save_sample_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """Save a generated image grid to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display_images = generated_to_display_range(images)
    grid = make_grid(display_images, nrow=nrow)
    save_image(grid, path)
