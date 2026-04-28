from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.checkpointing import save_checkpoint
from src.core.config import load_config, prepare_output_dir
from src.core.device import autocast_device_type, get_device
from src.core.seeding import seed_everything
from src.core.wandb_logging import init_wandb, wandb_log
from src.data.dataloaders import build_dataloaders
from src.evaluation.image_summaries import compute_basic_image_summaries
from src.models.flow_matching.model import build_flow_matching_components
from src.models.flow_matching.objective import FlowMatchingObjective
from src.models.flow_matching.sampler import sample_flow_matching
from src.sampling.save_images import save_sample_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a flow matching baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/flow_cifar10.yaml",
        help="Path to a YAML experiment config.",
    )
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training_config = config["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )


def build_objective(config: dict[str, Any]) -> FlowMatchingObjective:
    flow_config = config["flow_matching"]
    return FlowMatchingObjective(
        time_embedding_scale=flow_config["time_embedding_scale"],
        min_time=flow_config["min_time"],
        max_time=flow_config["max_time"],
    )


def validate_flow_matching(
    model: torch.nn.Module,
    objective: FlowMatchingObjective,
    val_loader,
    *,
    device: torch.device,
    max_batches: int = 10,
) -> dict[str, float]:
    """Compute a lightweight validation objective loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            images = batch[0].to(device)
            loss_dict = objective.loss(model, (images,), batch_idx)
            losses.append(float(loss_dict["loss"].cpu()))
    model.train()
    return {"val/loss": sum(losses) / max(len(losses), 1)}


def log_custom_validation_metrics(*args, **kwargs) -> dict[str, float]:
    """Add project-specific validation metrics here.

    Suggested additions:
    - KID on fixed generated sample counts
    - loss by time bucket
    - vector-field direction diagnostics
    - real-vs-generated marginal summaries
    """
    raise NotImplementedError("Add custom flow validation metrics after the baseline trains.")


def train(config: dict[str, Any]) -> None:
    seed_everything(config["experiment"]["seed"])
    output_dir = prepare_output_dir(config)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"

    device = get_device()
    train_loader, val_loader = build_dataloaders(config, device=device)
    model = build_flow_matching_components(config).to(device)
    objective = build_objective(config)
    optimizer = build_optimizer(model, config)

    use_amp = config["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    run = init_wandb(config)

    step = 0
    model.train()
    progress = tqdm(total=config["training"]["num_steps"], desc=config["experiment"]["name"])
    train_iter = iter(train_loader)

    while step < config["training"]["num_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch[0].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=autocast_device_type(device),
            enabled=use_amp,
        ):
            loss_dict = objective.loss(model, (images,), step)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()

        step += 1
        progress.update(1)

        if step % config["training"]["log_every"] == 0:
            metrics = {
                "train/loss": float(loss.detach().cpu()),
                "train/mean_time": float(loss_dict["diagnostics"]["mean_time"].cpu()),
                "train/target_velocity_norm": float(loss_dict["diagnostics"]["target_velocity_norm"].cpu()),
                "train/prediction_norm": float(loss_dict["diagnostics"]["prediction_norm"].cpu()),
            }
            progress.set_postfix(loss=f"{metrics['train/loss']:.4f}")
            wandb_log(run, metrics, step=step)

        if step % config["training"]["validate_every"] == 0:
            metrics = validate_flow_matching(model, objective, val_loader, device=device)
            wandb_log(run, metrics, step=step)
            print(metrics)

        if step % config["training"]["sample_every"] == 0:
            image_shape = (
                config["dataset"]["channels"],
                config["dataset"]["image_size"],
                config["dataset"]["image_size"],
            )
            samples = sample_flow_matching(
                model,
                num_samples=config["sampling"]["num_samples"],
                image_shape=image_shape,
                device=device,
                num_steps=config["sampling"]["num_steps"],
                time_embedding_scale=config["flow_matching"]["time_embedding_scale"],
                seed=config["experiment"]["seed"] + step,
            )
            save_sample_grid(samples, sample_dir / f"step_{step:08d}.png")
            wandb_log(run, {f"samples/{k}": v for k, v in compute_basic_image_summaries(samples).items()}, step=step)

        if step % config["training"]["checkpoint_every"] == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{step:08d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
            )

    save_checkpoint(
        checkpoint_dir / "final.pt",
        model=model,
        optimizer=optimizer,
        step=step,
        config=config,
    )
    progress.close()
    if run is not None:
        run.finish()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
