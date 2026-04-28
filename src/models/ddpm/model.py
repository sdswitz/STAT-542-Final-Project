from __future__ import annotations

from typing import Any

from diffusers import DDPMScheduler

from src.architectures.unet_diffusers import build_diffusers_unet


def build_ddpm_components(config: dict[str, Any]):
    """Build the DDPM denoiser and Diffusers scheduler."""
    model = build_diffusers_unet(config)
    diffusion_config = config["diffusion"]
    scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_config["num_train_timesteps"],
        beta_schedule=diffusion_config["beta_schedule"],
        prediction_type=diffusion_config["prediction_type"],
    )
    return model, scheduler
