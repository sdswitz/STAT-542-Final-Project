from __future__ import annotations

from typing import Any


def init_wandb(config: dict[str, Any]):
    """Initialize W&B if enabled; otherwise return None."""
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    import wandb

    return wandb.init(
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
        name=config["experiment"]["name"],
        tags=wandb_config.get("tags"),
        config=config,
    )


def wandb_log(run, metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics only when a W&B run exists."""
    if run is not None:
        run.log(metrics, step=step)
