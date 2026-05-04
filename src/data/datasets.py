from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Subset
from torchvision import datasets

from src.core.data_fraction import random_subset_indices, stratified_subset_indices
from src.data.transforms import image_transform


def _stratified_subset_indices(targets: list[int], train_percent: float, seed: int) -> list[int]:
    return stratified_subset_indices(targets, train_percent, seed)


def _random_subset_indices(num_items: int, train_percent: float, seed: int) -> list[int]:
    return random_subset_indices(num_items, train_percent, seed)


def apply_train_subset(train_dataset, config: dict[str, Any]):
    """Apply a deterministic training subset when dataset.train_percent < 100."""
    dataset_config = config["dataset"]
    train_percent = float(dataset_config.get("train_percent", 100.0))
    if not 0.0 < train_percent <= 100.0:
        raise ValueError(f"dataset.train_percent must be in (0, 100], got {train_percent}")

    if train_percent >= 100.0:
        return train_dataset

    subset_seed = int(dataset_config.get("subset_seed", config["experiment"]["seed"]))
    targets = getattr(train_dataset, "targets", None)
    if targets is not None:
        indices = _stratified_subset_indices(targets, train_percent, subset_seed)
    else:
        indices = _random_subset_indices(len(train_dataset), train_percent, subset_seed)

    return Subset(train_dataset, indices)


def build_datasets(config: dict[str, Any]):
    """Build train and validation datasets from config."""
    dataset_config = config["dataset"]
    name = dataset_config["name"].lower()
    data_dir = Path(dataset_config.get("data_dir", "data"))
    transform = image_transform(dataset_config["image_size"])

    if name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        val_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        train_dataset = apply_train_subset(train_dataset, config)
        return train_dataset, val_dataset

    raise ValueError(f"Unsupported dataset: {dataset_config['name']}")
