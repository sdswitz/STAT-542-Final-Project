from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def sample_flow_matching(
    model: torch.nn.Module,
    *,
    num_samples: int,
    image_shape: tuple[int, int, int],
    device: torch.device,
    num_steps: int,
    time_embedding_scale: float = 1000.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample by backward Euler integration from noise at t=1 to data at t=0."""
    model.eval()
    channels, height, width = image_shape
    generator = None
    if seed is not None:
        if device.type in {"cuda", "cpu"}:
            generator = torch.Generator(device=device.type).manual_seed(seed)
        else:
            torch.manual_seed(seed)

    x = torch.randn(
        (num_samples, channels, height, width),
        generator=generator,
        device=device,
    )

    times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for current_t, next_t in zip(times[:-1], times[1:]):
        step_size = current_t - next_t
        t_batch = torch.full(
            (num_samples,),
            float(current_t * time_embedding_scale),
            device=device,
        )
        pred_velocity = model(x, t_batch)
        x = x - step_size * pred_velocity

    return x.clamp(-1, 1)


def custom_flow_solver(config: dict[str, Any]) -> Any:
    """Define a custom ODE solver for flow matching here if needed."""
    raise NotImplementedError("Add a custom flow solver only after the Euler baseline works.")
