from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_fraction import data_percent_tag, stratified_subset_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 reference images as PNGs.")
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="CIFAR-10 split to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where PNG reference images will be written.",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--train-percent",
        type=float,
        default=None,
        help="For --split train, export the deterministic class-balanced subset used during training.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Seed used for deterministic train subset selection.",
    )
    parser.add_argument(
        "--memorization-root",
        type=str,
        default=None,
        help="Optional root for paper-style memorization references.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing non-empty output directory.")
    return parser.parse_args()


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


def default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)

    if args.memorization_root is not None:
        root = Path(args.memorization_root) / "reference"
        if args.split == "test":
            return root / f"cifar10_test_{args.image_size}"
        if args.train_percent is None:
            return root / f"cifar10_train_full_seed{args.subset_seed}"
        return root / f"cifar10_train_{data_percent_tag(args.train_percent)}_seed{args.subset_seed}"

    if args.split == "test":
        return Path(f"outputs/eval/reference/cifar10_test_{args.image_size}")
    if args.train_percent is None:
        return Path(f"outputs/eval/reference/cifar10_train_{args.image_size}")
    return Path(f"outputs/eval/reference/cifar10_train_{data_percent_tag(args.train_percent)}_seed{args.subset_seed}")


def selected_indices_for_targets(targets: list[int], train_percent: float | None, subset_seed: int) -> list[int] | None:
    if train_percent is None:
        return None
    return stratified_subset_indices(targets, train_percent, subset_seed)


def clear_pngs(output_dir: Path) -> None:
    for path in output_dir.glob("*.png"):
        path.unlink()


def write_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    image_count: int,
    selected_count: int,
) -> None:
    metadata = {
        "dataset": "cifar10",
        "split": args.split,
        "data_dir": args.data_dir,
        "image_size": args.image_size,
        "train_percent": args.train_percent,
        "subset_seed": args.subset_seed,
        "image_count": image_count,
        "selected_count": selected_count,
        "format": "png",
        "pixel_range": "float image saved from torchvision tensor in [0, 1]",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    if args.train_percent is not None and args.split != "train":
        raise ValueError("--train-percent can only be used with --split train")

    output_dir = default_output_dir(args)
    existing = count_pngs(output_dir)
    if existing > 0 and not args.force:
        print(f"Reference folder already has {existing} PNGs: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if existing > 0 and args.force:
        clear_pngs(output_dir)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=(args.split == "train"),
        download=True,
        transform=transform,
    )

    indices = selected_indices_for_targets(
        list(getattr(dataset, "targets", [])),
        args.train_percent,
        args.subset_seed,
    )
    if indices is None:
        indices = list(range(len(dataset)))

    for output_index, dataset_index in enumerate(tqdm(indices, desc=f"export cifar10-{args.split}")):
        image, _ = dataset[dataset_index]
        index = output_index
        save_image(image, output_dir / f"{index:06d}.png")

    written = count_pngs(output_dir)
    write_metadata(output_dir, args=args, image_count=len(dataset), selected_count=len(indices))
    print(f"Wrote {written} PNGs to {output_dir}")


if __name__ == "__main__":
    main()
