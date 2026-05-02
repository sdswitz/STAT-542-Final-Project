from __future__ import annotations

import json
import time
import gc
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
    batch_size: int | None = None,
    save_cpu_ram: bool = False,
    isc: bool = True,
    fid: bool = True,
    kid: bool = True,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    rng_seed: int = 2020,
    verbose: bool = True,
    timing_breakdown: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute torch-fidelity metrics between generated and real image folders.

    Conventions:
    - fake_dir is passed as input1, the generated/evaluated sample set.
    - real_dir is passed as input2, the real/reference sample set.
    - ISC is computed on input1 only.
    - FID and KID are computed between input1 and input2.

    torch-fidelity defaults batch_size to 64; on shared GPUs a smaller batch_size
    can avoid OOM during feature extraction even for small image folders.
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

    fidelity_kwargs: dict[str, Any] = dict(kwargs)
    if batch_size is not None:
        fidelity_kwargs.pop("batch_size", None)
        fidelity_kwargs["batch_size"] = batch_size
    if save_cpu_ram:
        fidelity_kwargs["save_cpu_ram"] = True

    def _run_one(*, isc: bool, fid: bool, kid: bool) -> tuple[dict[str, Any], float]:
        start = time.time()
        if verbose:
            print(
                f"[torch-fidelity] computing (isc={isc}, fid={fid}, kid={kid}, cuda={cuda})...",
                flush=True,
            )
        try:
            out = calculate_metrics(
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
                **fidelity_kwargs,
            )
        finally:
            # torch-fidelity can retain CUDA memory across repeated calls in the
            # same process (especially with timing_breakdown=True). Do a best-effort
            # cleanup between metrics to avoid OOM on shared GPUs.
            gc.collect()
            if cuda:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
        elapsed = time.time() - start
        if verbose:
            print(f"[torch-fidelity] done in {elapsed:.1f}s", flush=True)
        return _jsonable_metrics(out), float(elapsed)

    if not timing_breakdown:
        metrics, total_s = _run_one(isc=isc, fid=fid, kid=kid)
        if verbose:
            metrics["timing_seconds"] = {"total": total_s}
    else:
        metrics: dict[str, Any] = {}
        timing: dict[str, float] = {}

        # Run each requested metric in isolation so you can attribute time.
        # Note: each call may re-read images and/or re-extract features.
        if isc:
            isc_metrics, s = _run_one(isc=True, fid=False, kid=False)
            metrics.update(isc_metrics)
            timing["isc"] = s
        if fid:
            fid_metrics, s = _run_one(isc=False, fid=True, kid=False)
            metrics.update(fid_metrics)
            timing["fid"] = s
        if kid:
            kid_metrics, s = _run_one(isc=False, fid=False, kid=True)
            metrics.update(kid_metrics)
            timing["kid"] = s

        timing["sum"] = float(sum(timing.values()))
        metrics["timing_seconds"] = timing

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
