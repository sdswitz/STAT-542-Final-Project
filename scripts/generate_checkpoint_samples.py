from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.device import get_device
from src.core.seeding import seed_everything
from src.sampling.save_images import save_image_batch, save_sample_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG samples from an arbitrary saved checkpoint.")
    parser.add_argument("--model-type", choices=("ddpm", "flow", "flow_matching"), required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Fallback config path when the checkpoint does not contain a config dictionary.",
    )
    parser.add_argument("--preview-samples", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNGs in the output directory.")
    return parser.parse_args()


def canonical_model_type(model_type: str) -> str:
    if model_type == "flow":
        return "flow_matching"
    return model_type


def torch_load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must load to a dictionary: {path}")
    return checkpoint


def config_from_checkpoint(checkpoint: dict[str, Any], config_path: str | Path | None) -> dict[str, Any]:
    config = checkpoint.get("config")
    if isinstance(config, dict):
        return config
    if config_path is None:
        raise ValueError("Checkpoint does not contain a config; pass --config to build the model.")
    return load_config(config_path)


def build_model_and_scheduler(config: dict[str, Any], model_type: str, device: torch.device):
    if model_type == "ddpm":
        from src.models.ddpm.model import build_ddpm_components

        model, scheduler = build_ddpm_components(config)
        return model.to(device), scheduler

    if model_type == "flow_matching":
        from src.models.flow_matching.model import build_flow_matching_components

        model = build_flow_matching_components(config)
        return model.to(device), None

    raise ValueError(f"Unsupported model type: {model_type}")


@torch.no_grad()
def generate_batch(
    *,
    model_type: str,
    model: torch.nn.Module,
    scheduler,
    config: dict[str, Any],
    batch_size: int,
    image_shape: tuple[int, int, int],
    device: torch.device,
    num_steps: int,
    seed: int,
) -> torch.Tensor:
    if model_type == "ddpm":
        from src.models.ddpm.sampler import sample_ddpm

        return sample_ddpm(
            model,
            scheduler,
            num_samples=batch_size,
            image_shape=image_shape,
            device=device,
            num_steps=num_steps,
            seed=seed,
        )

    if model_type == "flow_matching":
        from src.models.flow_matching.sampler import sample_flow_matching

        return sample_flow_matching(
            model,
            num_samples=batch_size,
            image_shape=image_shape,
            device=device,
            num_steps=num_steps,
            time_embedding_scale=config["flow_matching"]["time_embedding_scale"],
            seed=seed,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def existing_png_count(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return len(list(output_dir.glob("*.png")))


def clear_pngs(output_dir: Path) -> None:
    for path in output_dir.glob("*.png"):
        path.unlink()


def checkpoint_step(checkpoint: dict[str, Any], checkpoint_path: Path) -> int | None:
    step = checkpoint.get("step")
    if isinstance(step, int):
        return step
    stem = checkpoint_path.stem
    if stem.startswith("step_"):
        try:
            return int(stem.removeprefix("step_"))
        except ValueError:
            return None
    return None


def write_metadata(
    output_dir: Path,
    *,
    model_type: str,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    num_samples: int,
    batch_size: int,
    num_steps: int,
    seed: int,
    device: torch.device,
    preview_grid: Path | None,
) -> None:
    metadata = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step(checkpoint, checkpoint_path),
        "num_samples": num_samples,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "seed": seed,
        "device": str(device),
        "experiment_name": config.get("experiment", {}).get("name"),
        "data_percent": config.get("dataset", {}).get("train_percent"),
        "image_size": config["dataset"]["image_size"],
        "channels": config["dataset"]["channels"],
        "format": "png",
        "pixel_range": "uint8 RGB saved from generated tensors clipped to [-1, 1]",
        "preview_grid": str(preview_grid) if preview_grid is not None else None,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def generate_checkpoint_samples(
    *,
    model_type: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 10_000,
    batch_size: int = 256,
    num_steps: int = 100,
    seed: int = 0,
    config_path: str | Path | None = None,
    preview_samples: int = 64,
    force: bool = False,
) -> Path:
    model_type = canonical_model_type(model_type)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    existing = existing_png_count(output_dir)
    if existing > 0 and not force:
        raise FileExistsError(f"{output_dir} already contains {existing} PNG files; pass --force to overwrite.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if existing > 0 and force:
        clear_pngs(output_dir)

    seed_everything(seed)
    device = get_device()
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    config = config_from_checkpoint(checkpoint, config_path)
    model, scheduler = build_model_and_scheduler(config, model_type, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_shape = (
        config["dataset"]["channels"],
        config["dataset"]["image_size"],
        config["dataset"]["image_size"],
    )
    filename_width = max(6, len(str(num_samples - 1)))
    preview_batches = []

    generated = 0
    progress = tqdm(total=num_samples, desc=f"generate {model_type}")
    while generated < num_samples:
        current_batch_size = min(batch_size, num_samples - generated)
        samples = generate_batch(
            model_type=model_type,
            model=model,
            scheduler=scheduler,
            config=config,
            batch_size=current_batch_size,
            image_shape=image_shape,
            device=device,
            num_steps=num_steps,
            seed=seed + generated,
        )
        save_image_batch(samples, output_dir, start_index=generated, filename_width=filename_width)

        if preview_samples > 0 and sum(batch.shape[0] for batch in preview_batches) < preview_samples:
            preview_batches.append(samples.detach().cpu())

        generated += current_batch_size
        progress.update(current_batch_size)
    progress.close()

    preview_grid = None
    if preview_samples > 0 and preview_batches:
        preview = torch.cat(preview_batches, dim=0)[:preview_samples]
        preview_grid = output_dir.parent / f"{output_dir.name}_preview_grid.png"
        save_sample_grid(preview, preview_grid)

    write_metadata(
        output_dir,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config=config,
        num_samples=num_samples,
        batch_size=batch_size,
        num_steps=num_steps,
        seed=seed,
        device=device,
        preview_grid=preview_grid,
    )
    return output_dir


def main() -> None:
    args = parse_args()
    generate_checkpoint_samples(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        seed=args.seed,
        config_path=args.config,
        preview_samples=args.preview_samples,
        force=args.force,
    )


if __name__ == "__main__":
    main()
