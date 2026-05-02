from __future__ import annotations

from collections import Counter

from src.core.data_fraction import stratified_subset_indices


def cifar10_like_targets() -> list[int]:
    targets = []
    for class_id in range(10):
        targets.extend([class_id] * 5000)
    return targets


def test_cifar10_subset_counts_match_experiment_percents() -> None:
    targets = cifar10_like_targets()

    expected_counts = {
        10.0: 5000,
        25.0: 12500,
        50.0: 25000,
    }

    for percent, expected_count in expected_counts.items():
        indices = stratified_subset_indices(targets, percent, seed=0)
        selected_targets = [targets[index] for index in indices]
        class_counts = Counter(selected_targets)

        assert len(indices) == expected_count
        assert set(class_counts) == set(range(10))
        assert len(set(class_counts.values())) == 1
