from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.checkpointing import load_model_checkpoint
from src.core.config import load_config
from src.core.device import get_device
from src.core.seeding import seed_everything
from src.models.ddpm.model import build_ddpm_components
from src.models.ddpm.sampler import sample_ddpm
from src.sampling.save_images import save_sample_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained DDPM checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/ddpm_cifar10.yaml",
        help="Path to the experiment config used for model construction.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint created by scripts/train_ddpm.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ddpm_samples.png",
        help="Where to save the sample grid.",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config["experiment"]["seed"] if args.seed is None else args.seed
    seed_everything(seed)

    device = get_device()
    model, scheduler = build_ddpm_components(config)
    load_model_checkpoint(args.checkpoint, model, map_location=device)
    model = model.to(device)

    image_shape = (
        config["dataset"]["channels"],
        config["dataset"]["image_size"],
        config["dataset"]["image_size"],
    )
    samples = sample_ddpm(
        model,
        scheduler,
        num_samples=args.num_samples or config["sampling"]["num_samples"],
        image_shape=image_shape,
        device=device,
        num_steps=args.num_steps or config["sampling"]["num_steps"],
        seed=seed,
    )
    save_sample_grid(samples, Path(args.output))


if __name__ == "__main__":
    main()
