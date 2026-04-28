from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class DDPMObjective:
    """Standard epsilon-prediction DDPM objective."""

    def __init__(self, scheduler, num_train_timesteps: int):
        self.scheduler = scheduler
        self.num_train_timesteps = num_train_timesteps

    def loss(self, model: torch.nn.Module, batch, batch_idx: int) -> dict[str, Any]:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
            dtype=torch.long,
        )
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        pred_noise = model(noisy_images, timesteps)
        loss = F.mse_loss(pred_noise, noise)
        return {
            "loss": loss,
            "loss_main": loss.detach(),
            "diagnostics": {
                "mean_timestep": timesteps.float().mean().detach(),
                "noise_norm": noise.flatten(1).norm(dim=1).mean().detach(),
                "prediction_norm": pred_noise.flatten(1).norm(dim=1).mean().detach(),
            },
        }


def compute_custom_ddpm_diagnostics(loss_dict: dict[str, Any]) -> dict[str, Any]:
    """Add project-specific DDPM diagnostics here.

    Examples:
    - loss by timestep bucket
    - prediction/noise cosine similarity
    - per-channel noise prediction error
    """
    raise NotImplementedError("Add custom DDPM diagnostics for the project here.")
