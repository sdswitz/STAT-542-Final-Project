from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.datasets import build_datasets


def build_dataloaders(config: dict[str, Any], device: torch.device | None = None):
    """Build train and validation dataloaders."""
    train_dataset, val_dataset = build_datasets(config)
    dataset_config = config["dataset"]
    training_config = config["training"]
    pin_memory = dataset_config.get("pin_memory", False)
    if device is not None and device.type != "cuda":
        pin_memory = False

    loader_kwargs = {
        "batch_size": training_config["batch_size"],
        "num_workers": dataset_config.get("num_workers", 0),
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader
