from __future__ import annotations

import torch


def compute_basic_image_summaries(images: torch.Tensor) -> dict[str, float]:
    """Compute simple generated-image summaries in normalized image space."""
    with torch.no_grad():
        return {
            "image_mean": float(images.mean().cpu()),
            "image_std": float(images.std().cpu()),
            "image_min": float(images.min().cpu()),
            "image_max": float(images.max().cpu()),
        }
