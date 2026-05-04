from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_checkpoint_samples import (
    checkpoint_step,
    config_from_checkpoint,
    generate_checkpoint_samples,
    torch_load_checkpoint,
)
from src.core.data_fraction import data_percent_tag
from src.evaluation.memorization import compute_memorization_metrics
from src.evaluation.metrics import compute_torch_fidelity_metrics


AGGREGATE_COLUMNS = [
    "model_type",
    "run_id",
    "data_percent",
    "n_train",
    "checkpoint_path",
    "step",
    "batch_size",
    "exposure",
    "num_samples",
    "sampling_steps",
    "memorization_fraction",
    "memorized_count",
    "fid_test",
    "kid_test",
    "fid_train",
    "kid_train",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 subset memorization evaluation from a manifest.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument(
        "--model-type",
        choices=("ddpm", "flow", "flow_matching"),
        action="append",
        default=None,
        help="Only evaluate this model type. Can be passed multiple times.",
    )
    parser.add_argument("--seed", type=int, action="append", default=None, help="Only evaluate this seed.")
    parser.add_argument(
        "--data-percent",
        type=float,
        action="append",
        default=None,
        help="Only evaluate this CIFAR-10 training percentage.",
    )
    parser.add_argument("--run-id", action="append", default=None, help="Only evaluate this run_id.")
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--memorization-batch-size", type=int, default=256)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true", help="Force torch-fidelity and memorization distance to CPU.")
    parser.add_argument("--skip-fidelity", action="store_true")
    parser.add_argument("--limit-checkpoints", type=int, default=None)
    parser.add_argument("--force-samples", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    parser.add_argument("--force-references", action="store_true")
    parser.add_argument(
        "--skip-missing-checkpoints",
        action="store_true",
        help="Skip selected runs whose checkpoint paths are not present on this machine.",
    )
    return parser.parse_args()


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            manifest = yaml.safe_load(handle)
        else:
            manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must load to a dictionary: {path}")
    if not isinstance(manifest.get("runs", []), list):
        raise ValueError("Manifest 'runs' must be a list when present.")
    if not isinstance(manifest.get("run_matrix", []), list):
        raise ValueError("Manifest 'run_matrix' must be a list when present.")
    if not manifest.get("runs") and not manifest.get("run_matrix"):
        raise ValueError("Manifest must contain either a 'runs' list or a 'run_matrix' list.")
    return manifest


def canonical_model_type(model_type: str) -> str:
    if model_type == "flow":
        return "flow_matching"
    return model_type


def percent_value(data_percent: float) -> str:
    value = float(data_percent)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def template_context(*, model_type: str, seed: int, subset_seed: int, data_percent: float) -> dict[str, Any]:
    return {
        "model": model_type,
        "model_type": model_type,
        "canonical_model_type": canonical_model_type(model_type),
        "seed": seed,
        "subset_seed": subset_seed,
        "data_percent": data_percent,
        "pct": percent_value(data_percent),
        "pct_tag": data_percent_tag(data_percent),
    }


def format_template(value: str, context: dict[str, Any]) -> str:
    return value.format(**context)


def normalize_runs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    runs = [dict(run) for run in manifest.get("runs", [])]

    for matrix in manifest.get("run_matrix", []):
        if "model_type" not in matrix:
            raise ValueError(f"Each run_matrix entry must define model_type: {matrix}")

        seeds = matrix.get("seeds", [matrix.get("seed", 0)])
        data_percents = matrix.get("data_percents", matrix.get("data_percent"))
        if data_percents is None:
            raise ValueError(f"Each run_matrix entry must define data_percents or data_percent: {matrix}")
        if not isinstance(seeds, list):
            seeds = [seeds]
        if not isinstance(data_percents, list):
            data_percents = [data_percents]

        for seed in seeds:
            for data_percent in data_percents:
                seed = int(seed)
                data_percent = float(data_percent)
                subset_seed = int(matrix.get("subset_seed", seed))
                context = template_context(
                    model_type=matrix["model_type"],
                    seed=seed,
                    subset_seed=subset_seed,
                    data_percent=data_percent,
                )
                run = {
                    key: value
                    for key, value in matrix.items()
                    if key
                    not in {
                        "seeds",
                        "seed",
                        "data_percents",
                        "data_percent",
                        "run_id_template",
                        "run_dir_template",
                        "checkpoint_glob_template",
                    }
                }
                run["seed"] = seed
                run["subset_seed"] = subset_seed
                run["data_percent"] = data_percent
                run["run_id"] = format_template(
                    matrix.get("run_id_template", "{model_type}_{pct_tag}_seed{seed}"),
                    context,
                )
                if "run_dir_template" in matrix:
                    run["run_dir"] = format_template(matrix["run_dir_template"], context)
                if "checkpoint_glob_template" in matrix:
                    run["checkpoint_glob"] = format_template(matrix["checkpoint_glob_template"], context)
                runs.append(run)

    return runs


def list_from_selection(selection: dict[str, Any], key: str) -> list[Any] | None:
    value = selection.get(key)
    if value in (None, [], ""):
        return None
    if isinstance(value, list):
        return value
    return [value]


def merged_filters(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, set[Any] | None]:
    selection = manifest.get("selection") or {}
    if not isinstance(selection, dict):
        raise ValueError("Manifest 'selection' must be a dictionary when present.")

    model_types = args.model_type if args.model_type is not None else list_from_selection(selection, "model_types")
    seeds = args.seed if args.seed is not None else list_from_selection(selection, "seeds")
    data_percents = (
        args.data_percent if args.data_percent is not None else list_from_selection(selection, "data_percents")
    )
    run_ids = args.run_id if args.run_id is not None else list_from_selection(selection, "run_ids")

    return {
        "model_types": {canonical_model_type(str(model_type)) for model_type in model_types} if model_types else None,
        "seeds": {int(seed) for seed in seeds} if seeds else None,
        "data_percents": {float(data_percent) for data_percent in data_percents} if data_percents else None,
        "run_ids": {str(run_id) for run_id in run_ids} if run_ids else None,
    }


def run_matches_filters(run: dict[str, Any], filters: dict[str, set[Any] | None]) -> bool:
    if filters["model_types"] is not None and canonical_model_type(str(run["model_type"])) not in filters["model_types"]:
        return False
    if filters["seeds"] is not None and int(run.get("seed", run.get("subset_seed", 0))) not in filters["seeds"]:
        return False
    if filters["data_percents"] is not None and float(run["data_percent"]) not in filters["data_percents"]:
        return False
    if filters["run_ids"] is not None and str(run["run_id"]) not in filters["run_ids"]:
        return False
    return True


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


def run_export_reference(
    *,
    split: str,
    output_root: Path,
    data_percent: float | None,
    subset_seed: int,
    image_size: int,
    force: bool,
) -> Path:
    if split == "test":
        output_dir = output_root / "reference" / f"cifar10_test_{image_size}"
        expected = 10_000
    else:
        if data_percent is None:
            raise ValueError("data_percent is required for train reference export.")
        output_dir = output_root / "reference" / f"cifar10_train_{data_percent_tag(data_percent)}_seed{subset_seed}"
        expected = round(50_000 * float(data_percent) / 100.0)

    existing = count_pngs(output_dir)
    if existing == expected and existing > 0:
        return output_dir
    if existing > 0 and not force:
        raise ValueError(f"{output_dir} has {existing} PNGs, expected {expected}; rerun with --force-references.")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_cifar10_reference.py"),
        "--split",
        split,
        "--memorization-root",
        str(output_root),
        "--image-size",
        str(image_size),
        "--subset-seed",
        str(subset_seed),
    ]
    if data_percent is not None:
        command.extend(["--train-percent", str(data_percent)])
    if force:
        command.append("--force")
    subprocess.run(command, check=True)

    written = count_pngs(output_dir)
    if written != expected:
        raise ValueError(f"Reference export wrote {written} PNGs to {output_dir}, expected {expected}.")
    return output_dir


def parse_step_from_path(path: Path) -> int:
    stem = path.stem
    if stem.startswith("step_"):
        try:
            return int(stem.removeprefix("step_"))
        except ValueError:
            return -1
    return -1


def discover_checkpoints(run: dict[str, Any], limit: int | None) -> list[Path]:
    if "checkpoint_paths" in run:
        paths = [Path(path) for path in run["checkpoint_paths"]]
    elif "checkpoint_glob" in run:
        paths = [Path(path) for path in glob.glob(str(run["checkpoint_glob"]))]
    else:
        run_dir = Path(run["run_dir"])
        paths = sorted((run_dir / "checkpoints").glob("step_*.pt"))
        paths.extend(sorted(run_dir.glob("step_*.pt")))

    paths = sorted(set(paths), key=lambda path: (parse_step_from_path(path), str(path)))
    if not paths:
        raise FileNotFoundError(f"No step_*.pt checkpoints found for run: {run.get('run_id', run)}")
    if limit is not None:
        paths = paths[:limit]
    return paths


def extract_metric(metrics: dict[str, Any], candidates: list[str]) -> float | str:
    for key in candidates:
        if key in metrics:
            value = metrics[key]
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
    for key, value in metrics.items():
        lowered = key.lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
    return ""


def compute_fidelity_pair(
    *,
    fake_dir: Path,
    real_dir: Path,
    cuda: bool | None,
    kid_subsets: int,
    kid_subset_size: int,
) -> tuple[dict[str, Any], float | str, float | str]:
    metrics = compute_torch_fidelity_metrics(
        fake_dir=fake_dir,
        real_dir=real_dir,
        cuda=cuda,
        isc=False,
        fid=True,
        kid=True,
        kid_subsets=kid_subsets,
        kid_subset_size=kid_subset_size,
        verbose=False,
    )
    fid = extract_metric(metrics, ["frechet_inception_distance", "fid"])
    kid = extract_metric(metrics, ["kernel_inception_distance_mean", "kid_mean", "kid"])
    return metrics, fid, kid


def load_or_compute_checkpoint_metrics(
    *,
    run: dict[str, Any],
    checkpoint_path: Path,
    output_root: Path,
    test_reference_dir: Path,
    train_reference_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_type = canonical_model_type(run["model_type"])
    run_id = safe_name(run["run_id"])
    data_percent = float(run["data_percent"])
    subset_seed = int(run.get("subset_seed", run.get("seed", 0)))
    run_tag = f"{model_type}_{data_percent_tag(data_percent)}_{run_id}"

    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    config = config_from_checkpoint(checkpoint, run.get("config"))
    step = checkpoint_step(checkpoint, checkpoint_path)
    if step is None:
        step = parse_step_from_path(checkpoint_path)
    if step < 0:
        raise ValueError(f"Could not determine checkpoint step for {checkpoint_path}")

    step_tag = f"step_{step:08d}"
    sample_dir = output_root / "samples" / run_tag / step_tag
    metrics_path = output_root / "metrics" / f"{run_tag}_{step_tag}.json"

    if metrics_path.exists() and not args.force_metrics:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)["row"]

    existing_samples = count_pngs(sample_dir)
    if existing_samples == args.num_samples:
        pass
    elif existing_samples > 0 and not args.force_samples:
        raise ValueError(f"{sample_dir} has {existing_samples} PNGs, expected {args.num_samples}; use --force-samples.")
    else:
        generate_checkpoint_samples(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            output_dir=sample_dir,
            num_samples=args.num_samples,
            batch_size=args.sample_batch_size,
            num_steps=args.sampling_steps,
            seed=int(run.get("sample_seed", run.get("seed", 0))) + step,
            config_path=run.get("config"),
            force=args.force_samples,
        )

    memorization = compute_memorization_metrics(
        fake_dir=sample_dir,
        train_dir=train_reference_dir,
        threshold=float(run.get("memorization_threshold", 1.0 / 3.0)),
        batch_size=args.memorization_batch_size,
        device="cpu" if args.cpu else None,
    )

    fidelity_test: dict[str, Any] = {}
    fidelity_train: dict[str, Any] = {}
    fid_test: float | str = ""
    kid_test: float | str = ""
    fid_train: float | str = ""
    kid_train: float | str = ""
    if not args.skip_fidelity:
        cuda = False if args.cpu else None
        fidelity_test, fid_test, kid_test = compute_fidelity_pair(
            fake_dir=sample_dir,
            real_dir=test_reference_dir,
            cuda=cuda,
            kid_subsets=args.kid_subsets,
            kid_subset_size=args.kid_subset_size,
        )
        fidelity_train, fid_train, kid_train = compute_fidelity_pair(
            fake_dir=sample_dir,
            real_dir=train_reference_dir,
            cuda=cuda,
            kid_subsets=args.kid_subsets,
            kid_subset_size=args.kid_subset_size,
        )

    n_train = count_pngs(train_reference_dir)
    batch_size = int(run.get("training_batch_size", config["training"]["batch_size"]))
    exposure = step * batch_size / n_train
    row = {
        "model_type": model_type,
        "run_id": run_id,
        "data_percent": data_percent,
        "n_train": n_train,
        "checkpoint_path": str(checkpoint_path),
        "step": step,
        "batch_size": batch_size,
        "exposure": exposure,
        "num_samples": args.num_samples,
        "sampling_steps": args.sampling_steps,
        "memorization_fraction": memorization["memorization_fraction"],
        "memorized_count": memorization["memorized_count"],
        "fid_test": fid_test,
        "kid_test": kid_test,
        "fid_train": fid_train,
        "kid_train": kid_train,
    }

    payload = {
        "row": row,
        "run": run,
        "checkpoint": str(checkpoint_path),
        "sample_dir": str(sample_dir),
        "test_reference_dir": str(test_reference_dir),
        "train_reference_dir": str(train_reference_dir),
        "subset_seed": subset_seed,
        "memorization": memorization,
        "fidelity_test": fidelity_test,
        "fidelity_train": fidelity_train,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return row


def write_aggregate(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["model_type"], float(row["data_percent"]), row["run_id"], int(row["step"])))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in AGGREGATE_COLUMNS})


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_root = Path(args.output_root or manifest.get("output_root", "outputs/eval/memorization"))
    output_root.mkdir(parents=True, exist_ok=True)

    filters = merged_filters(manifest, args)
    runs = [run for run in normalize_runs(manifest) if run_matches_filters(run, filters)]
    if not runs:
        raise ValueError("No manifest runs matched the selected model/seed/data-percent filters.")

    rows = []
    skipped = []
    for run in runs:
        if "run_id" not in run or "model_type" not in run or "data_percent" not in run:
            raise ValueError(f"Each run must define run_id, model_type, and data_percent: {run}")

        try:
            checkpoint_paths = discover_checkpoints(run, args.limit_checkpoints)
        except FileNotFoundError as exc:
            if args.skip_missing_checkpoints:
                skipped.append(str(exc))
                continue
            raise

        subset_seed = int(run.get("subset_seed", run.get("seed", 0)))
        image_size = int(run.get("image_size", 32))
        test_reference_dir = run_export_reference(
            split="test",
            output_root=output_root,
            data_percent=None,
            subset_seed=subset_seed,
            image_size=image_size,
            force=args.force_references,
        )
        train_reference_dir = run_export_reference(
            split="train",
            output_root=output_root,
            data_percent=float(run["data_percent"]),
            subset_seed=subset_seed,
            image_size=image_size,
            force=args.force_references,
        )

        for checkpoint_path in checkpoint_paths:
            row = load_or_compute_checkpoint_metrics(
                run=run,
                checkpoint_path=checkpoint_path,
                output_root=output_root,
                test_reference_dir=test_reference_dir,
                train_reference_dir=train_reference_dir,
                args=args,
            )
            rows.append(row)

    if not rows:
        message = "No checkpoint metrics were written."
        if skipped:
            message += " All selected runs were skipped because checkpoints were missing."
        raise ValueError(message)

    aggregate_path = output_root / "metrics" / "aggregate_metrics.csv"
    write_aggregate(rows, aggregate_path)
    print(f"Wrote aggregate metrics to {aggregate_path}")
    if skipped:
        print("Skipped missing checkpoint runs:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
