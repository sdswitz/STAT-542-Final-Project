from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_torch_fidelity_metrics


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test torch-fidelity timing on a tiny subset of images."
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        required=True,
        help="Directory of generated images (flat folder).",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default="outputs/eval/reference/cifar10_test_32",
        help="Directory of reference images (flat folder).",
    )
    parser.add_argument("--n", type=int, default=10, help="How many images to copy from each folder.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for torch-fidelity.")
    parser.add_argument(
        "--save-cpu-ram",
        action="store_true",
        help="Pass torch-fidelity save_cpu_ram=True.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="torch-fidelity dataloader batch size (smaller avoids GPU OOM).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="torch-fidelity dataloader workers.",
    )
    parser.add_argument(
        "--no-timing-breakdown",
        action="store_true",
        help="Use a single torch-fidelity call (only timing_seconds.total).",
    )
    return parser.parse_args()


def _list_images(root: Path) -> list[Path]:
    paths = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not paths:
        raise ValueError(f"No images with extensions {sorted(IMG_EXTS)} in {root}")
    return paths


def _copy_subset(src_dir: Path, dst_dir: Path, n: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = _list_images(src_dir)[:n]
    if len(paths) < n:
        raise ValueError(f"Need {n} images in {src_dir}, found {len(paths)}")
    for p in paths:
        shutil.copy2(p, dst_dir / p.name)


def main() -> None:
    args = parse_args()
    fake_src = Path(args.fake_dir)
    real_src = Path(args.real_dir)

    with tempfile.TemporaryDirectory(prefix="smoke_eval_") as tmp:
        tmp_root = Path(tmp)
        fake_small = tmp_root / "fake"
        real_small = tmp_root / "real"
        _copy_subset(fake_src, fake_small, args.n)
        _copy_subset(real_src, real_small, args.n)

        metrics = compute_torch_fidelity_metrics(
            fake_dir=fake_small,
            real_dir=real_small,
            output_path=None,
            cuda=False if args.cpu else None,
            batch_size=args.batch_size,
            save_cpu_ram=args.save_cpu_ram,
            isc=True,
            fid=True,
            kid=True,
            kid_subsets=2,
            kid_subset_size=min(4, args.n),
            verbose=True,
            timing_breakdown=not args.no_timing_breakdown,
            num_workers=args.num_workers,
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
