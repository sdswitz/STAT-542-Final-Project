from __future__ import annotations

from torchvision import transforms


def image_transform(image_size: int):
    """Resize images and normalize pixels from [0, 1] to [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def generated_to_display_range(images):
    """Convert generated tensors from [-1, 1] to [0, 1] for saving."""
    return ((images.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
