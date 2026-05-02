from __future__ import annotations

from types import SimpleNamespace

from scripts.run_memorization_evaluation import merged_filters, normalize_runs, run_matches_filters


def test_run_matrix_expands_model_seed_and_percent_combinations() -> None:
    manifest = {
        "run_matrix": [
            {
                "model_type": "ddpm",
                "seeds": [0, 542],
                "data_percents": [10, 25],
                "run_id_template": "ddpm_{pct_tag}_seed{seed}",
                "run_dir_template": "outputs/runs/ddpm_cifar10_seed{seed}_{pct_tag}",
                "config": "configs/experiments/ddpm_cifar10.yaml",
            }
        ]
    }

    runs = normalize_runs(manifest)

    assert [run["run_id"] for run in runs] == [
        "ddpm_pct10_seed0",
        "ddpm_pct25_seed0",
        "ddpm_pct10_seed542",
        "ddpm_pct25_seed542",
    ]
    assert runs[2]["run_dir"] == "outputs/runs/ddpm_cifar10_seed542_pct10"


def test_cli_filters_select_only_requested_model_and_seed() -> None:
    manifest = {
        "selection": {"model_types": [], "seeds": [], "data_percents": []},
        "run_matrix": [
            {"model_type": "ddpm", "seeds": [0, 542], "data_percents": [10]},
            {"model_type": "flow", "seeds": [0, 542], "data_percents": [10]},
        ],
    }
    args = SimpleNamespace(model_type=["flow"], seed=[542], data_percent=None, run_id=None)

    filters = merged_filters(manifest, args)
    selected = [run for run in normalize_runs(manifest) if run_matches_filters(run, filters)]

    assert len(selected) == 1
    assert selected[0]["model_type"] == "flow"
    assert selected[0]["seed"] == 542


def test_manifest_selection_is_used_when_cli_filter_is_absent() -> None:
    manifest = {
        "selection": {"model_types": ["ddpm"], "seeds": [0], "data_percents": [25]},
        "run_matrix": [
            {"model_type": "ddpm", "seeds": [0, 542], "data_percents": [10, 25]},
            {"model_type": "flow", "seeds": [0], "data_percents": [25]},
        ],
    }
    args = SimpleNamespace(model_type=None, seed=None, data_percent=None, run_id=None)

    filters = merged_filters(manifest, args)
    selected = [run for run in normalize_runs(manifest) if run_matches_filters(run, filters)]

    assert len(selected) == 1
    assert selected[0]["model_type"] == "ddpm"
    assert selected[0]["seed"] == 0
    assert selected[0]["data_percent"] == 25.0
