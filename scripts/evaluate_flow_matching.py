from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_torch_fidelity_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained flow matching run.")
    parser.add_argument(
        "--fake-dir",
        type=str,
        required=True,
        help="Directory containing generated PNGs.",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default="outputs/eval/reference/cifar10_test_32",
        help="Directory containing real/reference PNGs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save metrics JSON. Defaults to outputs/eval/metrics/<fake-dir-name>_torch_fidelity.json.",
    )
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true", help="Force torch-fidelity to run on CPU.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="torch-fidelity feature-extraction batch size (lower if you hit CUDA OOM).",
    )
    parser.add_argument(
        "--save-cpu-ram",
        action="store_true",
        help="Pass torch-fidelity save_cpu_ram=True (trades speed for lower host RAM).",
    )
    parser.add_argument(
        "--timing-breakdown",
        action="store_true",
        help="Run ISC/FID/KID as separate calls and record timing per metric.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fake_dir = Path(args.fake_dir)
    output = (
        Path(args.output)
        if args.output is not None
        else Path("outputs") / "eval" / "metrics" / f"{fake_dir.name}_torch_fidelity.json"
    )

    metrics = compute_torch_fidelity_metrics(
        fake_dir=fake_dir,
        real_dir=args.real_dir,
        output_path=output,
        cuda=False if args.cpu else None,
        batch_size=args.batch_size,
        save_cpu_ram=args.save_cpu_ram,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        timing_breakdown=args.timing_breakdown,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
