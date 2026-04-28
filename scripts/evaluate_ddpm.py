from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPM run.")
    parser.add_argument("--run-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """Implement the project-wide KID/FID evaluation protocol here.

    This is intentionally blank for now because the statistical protocol should
    be chosen before implementation:
    - fixed real reference split
    - generated sample count
    - feature extractor/package
    - bootstrap or repeated-subset uncertainty
    """
    raise NotImplementedError("Implement DDPM evaluation after choosing the KID/FID protocol.")


if __name__ == "__main__":
    main()
