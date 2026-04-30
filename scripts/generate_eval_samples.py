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
from src.sampling.save_images import save_image_batch, save_sample_grid


EVAL_NUM_SAMPLES = 50_000
EVAL_BATCH_SIZE = 256
EVAL_NUM_STEPS = 100
PREVIEW_SAMPLES = 64
FINAL_CHECKPOINT_NAME = "step_00100000.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate individual PNG samples for evaluation metrics.",
    )
    parser.add_argument(
        "model_type",
        choices=("ddpm", "flow"),
        help="Which trained model family to sample from.",
    )
    parser.add_argument(
        "seed",
        type=int,
        help="Training seed for the run to sample, e.g. 0 or 542.",
    )
    return parser.parse_args()


def canonical_model_type(model_type: str) -> str:
    if model_type == "flow":
        return "flow_matching"
    return model_type


def run_name(model_type: str, seed: int) -> str:
    prefix = "flow" if model_type == "flow_matching" else model_type
    return f"{prefix}_cifar10_seed{seed}"


def config_path_for(model_type: str) -> Path:
    prefix = "flow" if model_type == "flow_matching" else model_type
    return PROJECT_ROOT / "configs" / "experiments" / f"{prefix}_cifar10.yaml"


def checkpoint_path_for(model_type: str, seed: int) -> Path:
    name = run_name(model_type, seed)
    return PROJECT_ROOT / "outputs" / "runs" / name / "checkpoints" / FINAL_CHECKPOINT_NAME


def output_dir_for(model_type: str, seed: int) -> Path:
    name = run_name(model_type, seed)
    return PROJECT_ROOT / "outputs" / "eval" / "samples" / name


def load_run_config(model_type: str, seed: int) -> tuple[dict[str, Any], Path]:
    config_path = config_path_for(model_type)
    config = load_config(config_path)
    name = run_name(model_type, seed)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = name
    config["experiment"]["output_dir"] = str(PROJECT_ROOT / "outputs" / "runs" / name)
    return config, config_path


def existing_pngs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.png"))


def build_model_and_sampler(config: dict[str, Any], model_type: str, checkpoint: str, device: torch.device):
    if model_type == "ddpm":
        from src.models.ddpm.model import build_ddpm_components

        model, scheduler = build_ddpm_components(config)
        load_model_checkpoint(checkpoint, model, map_location=device)
        model = model.to(device)
        return model, scheduler

    if model_type == "flow_matching":
        from src.models.flow_matching.model import build_flow_matching_components

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


def write_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    seed: int,
) -> None:
    metadata = {
        "model_type": canonical_model_type(args.model_type),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "num_samples": EVAL_NUM_SAMPLES,
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
    model_type = canonical_model_type(args.model_type)
    config, config_path = load_run_config(model_type, args.seed)
    checkpoint_path = checkpoint_path_for(model_type, args.seed)
    output_dir = output_dir_for(model_type, args.seed)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint does not exist: {checkpoint_path}")

    pngs = existing_pngs(output_dir)
    if pngs:
        raise FileExistsError(
            f"{output_dir} already contains {len(pngs)} PNG files. "
            "Use a new output directory to avoid mixing sample sets."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    batch_size = EVAL_BATCH_SIZE
    num_steps = EVAL_NUM_STEPS
    filename_width = max(6, len(str(EVAL_NUM_SAMPLES - 1)))

    seed_everything(seed)
    device = get_device()
    model, scheduler = build_model_and_sampler(config, model_type, str(checkpoint_path), device)

    print(f"model_type: {model_type}")
    print(f"seed: {seed}")
    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output_dir: {output_dir}")
    print(f"num_samples: {EVAL_NUM_SAMPLES}")
    print(f"batch_size: {batch_size}")
    print(f"num_steps: {num_steps}")

    image_shape = (
        config["dataset"]["channels"],
        config["dataset"]["image_size"],
        config["dataset"]["image_size"],
    )

    preview_batches = []
    generated = 0
    progress = tqdm(total=EVAL_NUM_SAMPLES, desc=f"generate {model_type}")
    while generated < EVAL_NUM_SAMPLES:
        current_batch_size = min(batch_size, EVAL_NUM_SAMPLES - generated)
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
        save_image_batch(
            samples,
            output_dir,
            start_index=generated,
            filename_width=filename_width,
        )

        if PREVIEW_SAMPLES > 0 and sum(batch.shape[0] for batch in preview_batches) < PREVIEW_SAMPLES:
            preview_batches.append(samples.detach().cpu())

        generated += current_batch_size
        progress.update(current_batch_size)

    progress.close()
    write_metadata(
        output_dir,
        args=args,
        config=config,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
    )

    if PREVIEW_SAMPLES > 0 and preview_batches:
        preview = torch.cat(preview_batches, dim=0)[:PREVIEW_SAMPLES]
        save_sample_grid(preview, output_dir.parent / f"{output_dir.name}_preview_grid.png")


if __name__ == "__main__":
    main()
