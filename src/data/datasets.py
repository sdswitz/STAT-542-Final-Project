from __future__ import annotations

from pathlib import Path
from typing import Any

from torchvision import datasets

from src.data.transforms import image_transform


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
        return train_dataset, val_dataset

    raise ValueError(f"Unsupported dataset: {dataset_config['name']}")
