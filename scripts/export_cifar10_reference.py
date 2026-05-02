from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm


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
        default="outputs/eval/reference/cifar10_test_32",
        help="Directory where PNG reference images will be written.",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing non-empty output directory.")
    return parser.parse_args()


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    existing = count_pngs(output_dir)
    if existing > 0 and not args.force:
        print(f"Reference folder already has {existing} PNGs: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
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

    for index, (image, _) in enumerate(tqdm(dataset, desc=f"export cifar10-{args.split}")):
        save_image(image, output_dir / f"{index:06d}.png")

    print(f"Wrote {count_pngs(output_dir)} PNGs to {output_dir}")


if __name__ == "__main__":
    main()
