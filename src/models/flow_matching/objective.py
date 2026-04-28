from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class FlowMatchingObjective:
    """Linear Gaussian-to-data flow matching objective.

    The interpolation is x_t = (1 - t) x_0 + t epsilon, where t=0 is data and
    t=1 is Gaussian noise. The target vector field is dx_t / dt = epsilon - x_0.
    """

    def __init__(
        self,
        *,
        time_embedding_scale: float = 1000.0,
        min_time: float = 0.0,
        max_time: float = 1.0,
    ):
        self.time_embedding_scale = time_embedding_scale
        self.min_time = min_time
        self.max_time = max_time

    def loss(self, model: torch.nn.Module, batch, batch_idx: int) -> dict[str, Any]:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        noise = torch.randn_like(images)
        t = torch.empty((images.shape[0],), device=images.device).uniform_(
            self.min_time,
            self.max_time,
        )
        t_view = t.view(-1, 1, 1, 1)
        interpolated = (1.0 - t_view) * images + t_view * noise
        target_velocity = noise - images

        model_t = t * self.time_embedding_scale
        pred_velocity = model(interpolated, model_t)
        loss = F.mse_loss(pred_velocity, target_velocity)

        return {
            "loss": loss,
            "loss_main": loss.detach(),
            "diagnostics": {
                "mean_time": t.mean().detach(),
                "target_velocity_norm": target_velocity.flatten(1).norm(dim=1).mean().detach(),
                "prediction_norm": pred_velocity.flatten(1).norm(dim=1).mean().detach(),
            },
        }


def compute_custom_flow_diagnostics(loss_dict: dict[str, Any]) -> dict[str, Any]:
    """Add project-specific flow matching diagnostics here.

    Examples:
    - loss by time bucket
    - vector-field cosine similarity
    - generated summary drift by Euler step count
    """
    raise NotImplementedError("Add custom flow matching diagnostics for the project here.")
