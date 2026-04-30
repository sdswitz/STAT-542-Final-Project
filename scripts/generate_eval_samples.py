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

from src.core.checkpointing import load_model_checkpoint
from src.core.config import load_config
from src.core.device import get_device
from src.core.seeding import seed_everything
from src.models.ddpm.model import build_ddpm_components
from src.models.ddpm.sampler import sample_ddpm
from src.models.flow_matching.model import build_flow_matching_components
from src.models.flow_matching.sampler import sample_flow_matching
from src.sampling.save_images import save_image_batch, save_sample_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate individual PNG samples for evaluation metrics.",
    )
    parser.add_argument(
        "--model-type",
        choices=("ddpm", "flow_matching"),
        required=True,
        help="Which trained model family to sample from.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment config used for model construction.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where individual PNG samples will be written.",
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=64,
        help="Number of generated images to include in preview_grid.png. Use 0 to disable.",
    )
    return parser.parse_args()


def existing_pngs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.png"))


def build_model_and_sampler(config: dict[str, Any], model_type: str, checkpoint: str, device: torch.device):
    if model_type == "ddpm":
        model, scheduler = build_ddpm_components(config)
        load_model_checkpoint(checkpoint, model, map_location=device)
        model = model.to(device)
        return model, scheduler

    if model_type == "flow_matching":
        model = build_flow_matching_components(config)
        load_model_checkpoint(checkpoint, model, map_location=device)
        model = model.to(device)
        return model, None

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


def write_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
    num_steps: int,
    batch_size: int,
    seed: int,
) -> None:
    metadata = {
        "model_type": args.model_type,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "seed": seed,
        "device": str(device),
        "experiment_name": config["experiment"]["name"],
        "image_size": config["dataset"]["image_size"],
        "channels": config["dataset"]["channels"],
        "format": "png",
        "pixel_range": "uint8 RGB saved from generated tensors clipped to [-1, 1]",
        "preview_grid": str(output_dir.parent / f"{output_dir.name}_preview_grid.png"),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pngs = existing_pngs(output_dir)
    if pngs:
        raise FileExistsError(
            f"{output_dir} already contains {len(pngs)} PNG files. "
            "Use a new output directory to avoid mixing sample sets."
        )

    seed = config["experiment"]["seed"] if args.seed is None else args.seed
    batch_size = args.batch_size or config["sampling"]["batch_size"]
    num_steps = args.num_steps or config["sampling"]["num_steps"]
    filename_width = max(6, len(str(args.num_samples - 1)))

    seed_everything(seed)
    device = get_device()
    model, scheduler = build_model_and_sampler(config, args.model_type, args.checkpoint, device)

    image_shape = (
        config["dataset"]["channels"],
        config["dataset"]["image_size"],
        config["dataset"]["image_size"],
    )

    preview_batches = []
    generated = 0
    progress = tqdm(total=args.num_samples, desc=f"generate {args.model_type}")
    while generated < args.num_samples:
        current_batch_size = min(batch_size, args.num_samples - generated)
        samples = generate_batch(
            model_type=args.model_type,
            model=model,
            scheduler=scheduler,
            config=config,
            batch_size=current_batch_size,
            image_shape=image_shape,
            device=device,
            num_steps=num_steps,
            seed=seed + generated,
        )
        save_image_batch(
            samples,
            output_dir,
            start_index=generated,
            filename_width=filename_width,
        )

        if args.preview_samples > 0 and sum(batch.shape[0] for batch in preview_batches) < args.preview_samples:
            preview_batches.append(samples.detach().cpu())

        generated += current_batch_size
        progress.update(current_batch_size)

    progress.close()
    write_metadata(
        output_dir,
        args=args,
        config=config,
        device=device,
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
    )

    if args.preview_samples > 0 and preview_batches:
        preview = torch.cat(preview_batches, dim=0)[: args.preview_samples]
        save_sample_grid(preview, output_dir.parent / f"{output_dir.name}_preview_grid.png")


if __name__ == "__main__":
    main()
