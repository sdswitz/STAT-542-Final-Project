from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _count_images(path: Path) -> int:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in extensions)


def _validate_image_dir(path: str | Path, label: str) -> Path:
    image_dir = Path(path)
    if not image_dir.exists():
        raise FileNotFoundError(f"{label} image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {image_dir}")
    image_count = _count_images(image_dir)
    if image_count == 0:
        raise ValueError(f"{label} image directory contains no supported image files: {image_dir}")
    return image_dir


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    jsonable = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        jsonable[key] = value
    return jsonable


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Save a metrics dictionary as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable_metrics(metrics), handle, indent=2)


def compute_torch_fidelity_metrics(
    *,
    fake_dir: str | Path,
    real_dir: str | Path,
    output_path: str | Path | None = None,
    cuda: bool | None = None,
    isc: bool = True,
    fid: bool = False,
    kid: bool = True,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    rng_seed: int = 2020,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute torch-fidelity metrics between generated and real image folders.

    Conventions:
    - fake_dir is passed as input1, the generated/evaluated sample set.
    - real_dir is passed as input2, the real/reference sample set.
    - ISC is computed on input1 only.
    - FID and KID are computed between input1 and input2.
    """
    fake_dir = _validate_image_dir(fake_dir, "fake")
    real_dir = _validate_image_dir(real_dir, "real")

    if cuda is None:
        import torch

        cuda = torch.cuda.is_available()

    try:
        from torch_fidelity import calculate_metrics
    except ImportError as exc:
        raise ImportError(
            "torch-fidelity is required for evaluation. Install it with `pip install torch-fidelity`."
        ) from exc

    metrics = calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        cuda=cuda,
        isc=isc,
        fid=fid,
        kid=kid,
        kid_subsets=kid_subsets,
        kid_subset_size=kid_subset_size,
        rng_seed=rng_seed,
        verbose=verbose,
        **kwargs,
    )
    metrics = _jsonable_metrics(metrics)

    if output_path is not None:
        save_metrics(metrics, output_path)

    return metrics


def compute_kid_for_run(
    *,
    fake_dir: str | Path,
    real_dir: str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute KID only for a generated image folder against a real image folder."""
    return compute_torch_fidelity_metrics(
        fake_dir=fake_dir,
        real_dir=real_dir,
        output_path=output_path,
        isc=False,
        fid=False,
        kid=True,
        **kwargs,
    )


def compute_fid_for_run(
    *,
    fake_dir: str | Path,
    real_dir: str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute FID only for a generated image folder against a real image folder."""
    return compute_torch_fidelity_metrics(
        fake_dir=fake_dir,
        real_dir=real_dir,
        output_path=output_path,
        isc=False,
        fid=True,
        kid=False,
        **kwargs,
    )
