# CIFAR-10 Subset Memorization Experiment Plan

## Summary

Implement a paper-aligned evaluation pipeline for existing DDPM and flow-matching CIFAR-10 subset checkpoints trained on 10%, 25%, and 50% of the train set.

Each 10k-step checkpoint should be evaluated with:

- 10,000 generated samples per checkpoint.
- Nearest-neighbor memorization ratio using `dist_1 / dist_2 < 1/3`.
- FID/KID against the CIFAR-10 test set and against the exact deterministic train subset used by that run.
- Plots over training step and effective exposure, where `exposure = step * batch_size / n_train`.

The final writeup should frame the experiment as a CIFAR-10 subset adaptation of the memorization papers, not as a direct reproduction.

## Reference Papers

- `data/papers/memorization-papers/2310.02664v2.pdf`: primary comparison, because it evaluates CIFAR-10 subset memorization with 10k generated samples and the nearest/second-nearest distance ratio criterion.
- `data/papers/memorization-papers/87_Diffusion_Probabilistic_Mod.pdf`: supports the same CIFAR-10 subset and ratio-criterion framing, especially the model-capacity interpretation.
- `data/papers/memorization-papers/23738_Why_Diffusion_Models_Don.pdf`: useful for the training-dynamics interpretation, especially memorization emerging later than the sample-quality plateau.

## Implementation

- Add generic checkpoint sampling so evaluation can target any saved `step_*.pt` checkpoint for DDPM or flow matching.
- Export CIFAR-10 references under the memorization output root:

```text
outputs/eval/memorization/reference/
  cifar10_test_32/
  cifar10_train_pct10_seed<subset_seed>/
  cifar10_train_pct25_seed<subset_seed>/
  cifar10_train_pct50_seed<subset_seed>/
```

- Compute memorization using PNG-space L2 nearest and second-nearest training-image distances.
- Drive the full experiment from a manifest containing run IDs, model types, data percentages, subset seeds, and run directories.
- Save per-checkpoint JSON metrics and an aggregate CSV under `outputs/eval/memorization/metrics/`.
- Save plots under `outputs/eval/memorization/plots/`.

## Required Aggregate Columns

The aggregate CSV must include:

```text
model_type, run_id, data_percent, n_train, checkpoint_path, step,
batch_size, exposure, num_samples, sampling_steps, memorization_fraction,
memorized_count, fid_test, kid_test, fid_train, kid_train
```

## Output Layout

```text
outputs/eval/memorization/
  reference/
    cifar10_test_32/
    cifar10_train_pct10_seed<subset_seed>/
    cifar10_train_pct25_seed<subset_seed>/
    cifar10_train_pct50_seed<subset_seed>/
  samples/
    <model>_pct<percent>_<run_id>/step_XXXXXXXX/
  metrics/
    <model>_pct<percent>_<run_id>_step_XXXXXXXX.json
    aggregate_metrics.csv
  plots/
    memorization_vs_step.png
    memorization_vs_exposure.png
    fid_kid_vs_step.png
    fid_kid_vs_exposure.png
```

## Test Plan

- Unit test the memorization metric on tiny synthetic image folders where exact copies are marked memorized and unrelated images are not.
- Unit test that deterministic CIFAR-10-style train subsets produce exactly 5,000, 12,500, and 25,000 indices for 10%, 25%, and 50%.
- Smoke-test the full pipeline on one checkpoint with 32 generated samples and reduced KID settings before running the 10k-sample evaluation.
- Verify that generated sample metadata records the source checkpoint, model type, data percent, seed, sample count, and sampling steps.
- Manually inspect a small grid of memorized generated samples beside their nearest train neighbors for the highest-memorization checkpoint, if any.
