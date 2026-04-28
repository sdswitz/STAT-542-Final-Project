from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    scheduler,
    *,
    num_samples: int,
    image_shape: tuple[int, int, int],
    device: torch.device,
    num_steps: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample images from a DDPM denoiser with a Diffusers scheduler."""
    model.eval()
    channels, height, width = image_shape
    generator = None
    if seed is not None:
        if device.type in {"cuda", "cpu"}:
            generator = torch.Generator(device=device.type).manual_seed(seed)
        else:
            torch.manual_seed(seed)

    images = torch.randn(
        (num_samples, channels, height, width),
        generator=generator,
        device=device,
    )

    scheduler.set_timesteps(num_steps, device=device)
    for timestep in scheduler.timesteps:
        model_input = scheduler.scale_model_input(images, timestep)
        timestep_batch = torch.full(
            (num_samples,),
            int(timestep),
            device=device,
            dtype=torch.long,
        )
        pred_noise = model(model_input, timestep_batch)
        images = scheduler.step(pred_noise, timestep, images).prev_sample

    return images.clamp(-1, 1)


def custom_sampling_schedule(config: dict[str, Any]) -> list[int]:
    """Define a custom DDPM sampling schedule here if needed."""
    raise NotImplementedError("Add a custom sampling schedule only if the default scheduler is not enough.")
