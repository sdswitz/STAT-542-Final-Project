from __future__ import annotations

from typing import Any

import torch
from diffusers import UNet2DModel


class DiffusersUNetWrapper(torch.nn.Module):
    """Project wrapper that makes Diffusers UNet2DModel return a tensor."""

    def __init__(self, unet: UNet2DModel):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(x, t).sample


def build_diffusers_unet(config: dict[str, Any]) -> DiffusersUNetWrapper:
    """Build a low-resolution U-Net denoiser from Diffusers config."""
    model_config = config["model"]
    dataset_config = config["dataset"]

    unet = UNet2DModel(
        sample_size=dataset_config["image_size"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        layers_per_block=model_config["layers_per_block"],
        block_out_channels=tuple(model_config["block_out_channels"]),
        down_block_types=tuple(model_config["down_block_types"]),
        up_block_types=tuple(model_config["up_block_types"]),
    )
    return DiffusersUNetWrapper(unet)
