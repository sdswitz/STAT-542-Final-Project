from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot memorization experiment aggregate metrics.")
    parser.add_argument(
        "--aggregate",
        type=str,
        default="outputs/eval/memorization/metrics/aggregate_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval/memorization/plots",
    )
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in aggregate CSV: {path}")

    numeric_columns = {
        "data_percent",
        "n_train",
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
    }
    for row in rows:
        for column in numeric_columns:
            value = row.get(column, "")
            if value == "":
                row[column] = None
            elif column in {"n_train", "step", "batch_size", "num_samples", "sampling_steps", "memorized_count"}:
                row[column] = int(float(value))
            else:
                row[column] = float(value)
    return rows


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = f"{row['model_type']} pct{row['data_percent']:g} {row['run_id']}"
        groups.setdefault(label, []).append(row)
    for label in groups:
        groups[label] = sorted(groups[label], key=lambda row: row["step"])
    return groups


def plot_single_metric(
    *,
    rows: list[dict[str, Any]],
    x_column: str,
    y_column: str,
    y_label: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    groups = group_rows(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in groups.items():
        points = [(row[x_column], row[y_column]) for row in group if row.get(y_column) is not None]
        if not points:
            continue
        x_values, y_values = zip(*points)
        ax.plot(x_values, y_values, marker="o", label=label)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_fid_kid(*, rows: list[dict[str, Any]], x_column: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    groups = group_rows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    metric_specs = [
        ("fid_test", "fid_train", "FID"),
        ("kid_test", "kid_train", "KID"),
    ]

    for ax, (test_column, train_column, label) in zip(axes, metric_specs):
        for group_label, group in groups.items():
            test_points = [(row[x_column], row[test_column]) for row in group if row.get(test_column) is not None]
            train_points = [(row[x_column], row[train_column]) for row in group if row.get(train_column) is not None]
            if test_points:
                x_values, y_values = zip(*test_points)
                ax.plot(x_values, y_values, marker="o", label=f"{group_label} test")
            if train_points:
                x_values, y_values = zip(*train_points)
                ax.plot(x_values, y_values, marker="x", linestyle="--", label=f"{group_label} train")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel(x_column)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.aggregate)
    output_dir = Path(args.output_dir)
    mpl_config_dir = output_dir / ".matplotlib-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    plot_single_metric(
        rows=rows,
        x_column="step",
        y_column="memorization_fraction",
        y_label="Memorization fraction",
        output_path=output_dir / "memorization_vs_step.png",
    )
    plot_single_metric(
        rows=rows,
        x_column="exposure",
        y_column="memorization_fraction",
        y_label="Memorization fraction",
        output_path=output_dir / "memorization_vs_exposure.png",
    )
    plot_fid_kid(rows=rows, x_column="step", output_path=output_dir / "fid_kid_vs_step.png")
    plot_fid_kid(rows=rows, x_column="exposure", output_path=output_dir / "fid_kid_vs_exposure.png")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
