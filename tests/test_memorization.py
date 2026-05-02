from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.memorization import compute_memorization_metrics


def write_rgb(path: Path, value: int) -> None:
    array = np.full((4, 4, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_exact_copy_is_marked_memorized(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    fake_dir = tmp_path / "fake"
    train_dir.mkdir()
    fake_dir.mkdir()

    write_rgb(train_dir / "000000.png", 0)
    write_rgb(train_dir / "000001.png", 255)
    write_rgb(fake_dir / "000000.png", 0)
    write_rgb(fake_dir / "000001.png", 128)

    metrics = compute_memorization_metrics(
        fake_dir=fake_dir,
        train_dir=train_dir,
        threshold=1.0 / 3.0,
        batch_size=1,
        device="cpu",
    )

    assert metrics["generated_count"] == 2
    assert metrics["reference_count"] == 2
    assert metrics["memorized_count"] == 1
    assert metrics["memorization_fraction"] == 0.5
