from __future__ import annotations

from typing import Any

import torch


def validate_data_percent(data_percent: float) -> float:
    data_percent = float(data_percent)
    if not 0.0 < data_percent <= 100.0:
        raise ValueError(f"data percent must be in (0, 100], got {data_percent}")
    return data_percent


def data_percent_tag(data_percent: float) -> str:
    value = validate_data_percent(data_percent)
    if value.is_integer():
        return f"pct{int(value)}"
    return f"pct{str(value).replace('.', 'p')}"


def apply_data_percent_override(config: dict[str, Any], data_percent: float | None) -> dict[str, Any]:
    """Override dataset percentage and isolate outputs for that run."""
    if data_percent is None:
        return config

    value = validate_data_percent(data_percent)
    tag = data_percent_tag(value)
    config["dataset"]["train_percent"] = value

    base_name = config["experiment"]["name"]
    config["experiment"]["name"] = f"{base_name}_{tag}"
    config["experiment"]["output_dir"] = f"{config['experiment']['output_dir']}_{tag}"

    wandb_config = config.get("wandb")
    if isinstance(wandb_config, dict):
        tags = list(wandb_config.get("tags") or [])
        if tag not in tags:
            tags.append(tag)
        wandb_config["tags"] = tags

    return config


def stratified_subset_indices(targets: list[int], train_percent: float, seed: int) -> list[int]:
    """Return deterministic class-balanced subset indices for labeled datasets."""
    value = validate_data_percent(train_percent)
    generator = torch.Generator().manual_seed(seed)
    targets_tensor = torch.as_tensor(targets)
    indices = []

    for class_id in sorted(targets_tensor.unique().tolist()):
        class_indices = torch.where(targets_tensor == class_id)[0]
        keep_count = max(1, round(len(class_indices) * value / 100.0))
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        indices.extend(shuffled[:keep_count].tolist())

    return sorted(indices)


def random_subset_indices(num_items: int, train_percent: float, seed: int) -> list[int]:
    """Return deterministic random subset indices for unlabeled datasets."""
    value = validate_data_percent(train_percent)
    generator = torch.Generator().manual_seed(seed)
    keep_count = max(1, round(num_items * value / 100.0))
    return sorted(torch.randperm(num_items, generator=generator)[:keep_count].tolist())
