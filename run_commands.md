## Commands to start training a model

### DDPM:
trial run just to make sure it's working:
`python scripts/train_ddpm.py --config configs/experiments/ddpm_cifar10_smoke.yaml`

full run:
`python scripts/train_ddpm.py --config configs/experiments/ddpm_cifar10.yaml`

full run on a dataset fraction:
`python scripts/train_ddpm.py --config configs/experiments/ddpm_cifar10.yaml --data-percent 10`

### Flow matching:
trial run just to make sure it's working:
`python scripts/train_flow_matching.py --config configs/experiments/flow_cifar10_smoke.yaml`

full run:
`python scripts/train_flow_matching.py --config configs/experiments/flow_cifar10.yaml`

full run on a dataset fraction:
`python scripts/train_flow_matching.py --config configs/experiments/flow_cifar10.yaml --data-percent 10`

### Dataset fraction runs:
Use `--data-percent` to train on a deterministic, class-balanced subset of CIFAR-10. Good first values:
`10`, `25`, `50`, `100`

For example, DDPM with 10% of the training data writes to:
`outputs/runs/ddpm_cifar10_seed0_pct10/`

Flow matching with 25% of the training data writes to:
`outputs/runs/flow_cifar10_seed0_pct25/`

## Commands to generate samples for evaluation

These commands generate individual PNG files for KID/FID-style evaluation. The script takes only the model type and training seed, then derives the config, checkpoint, and output paths automatically.

The script expects the final checkpoint to be named:
`step_00100000.pt`

### DDPM:
generate eval samples for seed 0:
`python scripts/generate_eval_samples.py ddpm 0`

generate eval samples for another seed:
`python scripts/generate_eval_samples.py ddpm 542`

### Flow matching:
generate eval samples for seed 0:
`python scripts/generate_eval_samples.py flow 0`

generate eval samples for another seed:
`python scripts/generate_eval_samples.py flow 542`

### Derived paths:
DDPM seed 0 checkpoint:
`outputs/runs/ddpm_cifar10_seed0/checkpoints/step_00100000.pt`

Flow seed 0 checkpoint:
`outputs/runs/flow_cifar10_seed0/checkpoints/step_00100000.pt`

Generated DDPM seed 0 samples:
`outputs/eval/samples/ddpm_cifar10_seed0/`

Generated flow seed 0 samples:
`outputs/eval/samples/flow_cifar10_seed0/`

By default, the eval generation script writes 50,000 PNGs with batch size 256 and 100 sampling steps. It also writes `metadata.json` inside the sample folder and a preview grid next to the sample folder.



## Evaluation Process:

First, rerun `pip install -r requirements.txt` to make sure torch_fidelity is installed (only need to do this once)

Then, create a folder with the actual CIFAR-10 images (only need to do this once):
`python scripts/export_cifar10_reference.py`

The default real/reference folder is:
`outputs/eval/reference/cifar10_test_32`

Then the following should work:

### DDPM:
evaluate DDPM seed 0 samples:
`python scripts/evaluate_ddpm.py --fake-dir outputs/eval/samples/ddpm_cifar10_seed0`

evaluate DDPM seed 542 samples:
`python scripts/evaluate_ddpm.py --fake-dir outputs/eval/samples/ddpm_cifar10_seed542`

### Flow matching:
evaluate flow seed 0 samples:
`python scripts/evaluate_flow_matching.py --fake-dir outputs/eval/samples/flow_cifar10_seed0`

evaluate flow seed 542 samples:
`python scripts/evaluate_flow_matching.py --fake-dir outputs/eval/samples/flow_cifar10_seed542`



The output metrics JSON defaults to:
`outputs/eval/metrics/<sample-folder-name>_torch_fidelity.json`

For quick validation on a small sample folder, reduce KID subset settings:
`python scripts/evaluate_ddpm.py --fake-dir outputs/eval/samples/ddpm_cifar10_seed0 --kid-subsets 10 --kid-subset-size 100`
