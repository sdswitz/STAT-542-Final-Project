from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_image_paths(path: str | Path) -> list[Path]:
    image_dir = Path(path)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {image_dir}")

    paths = sorted(item for item in image_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"Image directory contains no supported image files: {image_dir}")
    return paths


def load_flat_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).flatten()


def load_image_matrix(paths: list[Path]) -> torch.Tensor:
    tensors = [load_flat_image(path) for path in paths]
    first_shape = tensors[0].shape
    for path, tensor in zip(paths, tensors):
        if tensor.shape != first_shape:
            raise ValueError(f"Image shape mismatch in {path}: expected {first_shape}, got {tensor.shape}")
    return torch.stack(tensors, dim=0)


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    jsonable = {}
    for key, value in metrics.items():
        if isinstance(value, Path):
            value = str(value)
        elif hasattr(value, "item"):
            value = value.item()
        jsonable[key] = value
    return jsonable


def compute_memorization_metrics(
    *,
    fake_dir: str | Path,
    train_dir: str | Path,
    output_path: str | Path | None = None,
    threshold: float = 1.0 / 3.0,
    batch_size: int = 256,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Compute nearest-neighbor ratio memorization metrics.

    A generated sample is counted as memorized when its L2 distance to the nearest
    training image is less than threshold times its L2 distance to the second
    nearest training image.
    """
    fake_paths = list_image_paths(fake_dir)
    train_paths = list_image_paths(train_dir)
    if len(train_paths) < 2:
        raise ValueError("At least two training images are required for nearest/second-nearest ratios.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    reference = load_image_matrix(train_paths).to(device)
    ratios = []
    nearest_distances = []
    second_nearest_distances = []

    for start in range(0, len(fake_paths), batch_size):
        batch_paths = fake_paths[start : start + batch_size]
        batch = load_image_matrix(batch_paths).to(device)
        if batch.shape[1] != reference.shape[1]:
            raise ValueError(
                f"Generated/reference image dimensionality mismatch: {batch.shape[1]} vs {reference.shape[1]}"
            )

        distances = torch.cdist(batch, reference, p=2)
        top2 = torch.topk(distances, k=2, dim=1, largest=False).values
        nearest = top2[:, 0]
        second = top2[:, 1].clamp_min(1e-12)
        ratio = nearest / second

        ratios.append(ratio.detach().cpu())
        nearest_distances.append(nearest.detach().cpu())
        second_nearest_distances.append(second.detach().cpu())

    all_ratios = torch.cat(ratios)
    all_nearest = torch.cat(nearest_distances)
    all_second = torch.cat(second_nearest_distances)
    memorized = all_ratios < threshold
    memorized_count = int(memorized.sum().item())
    generated_count = len(fake_paths)

    metrics = {
        "memorization_fraction": memorized_count / generated_count,
        "memorized_count": memorized_count,
        "generated_count": generated_count,
        "reference_count": len(train_paths),
        "mean_ratio": float(all_ratios.mean().item()),
        "median_ratio": float(all_ratios.median().item()),
        "mean_nearest_distance": float(all_nearest.mean().item()),
        "mean_second_nearest_distance": float(all_second.mean().item()),
        "threshold": threshold,
        "criterion": "l2_nearest_ratio",
        "memorized_rule": "nearest_l2 / second_nearest_l2 < threshold",
        "fake_dir": str(fake_dir),
        "train_dir": str(train_dir),
    }
    metrics = _jsonable_metrics(metrics)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    return metrics
