from __future__ import annotations

from typing import Any

from src.architectures.unet_diffusers import build_diffusers_unet


def build_flow_matching_components(config: dict[str, Any]):
    """Build the flow matching vector-field model."""
    model = build_diffusers_unet(config)
    return model
