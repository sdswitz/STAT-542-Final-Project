## Commands to start training a model

### DDPM:
trial run just to make sure it's working:
`python scripts/train_ddpm.py --config configs/experiments/ddpm_cifar10_smoke.yaml`

full run:
`python scripts/train_ddpm.py --config configs/experiments/ddpm_cifar10.yaml`

### Flow matching:
trial run just to make sure it's working:
`python scripts/train_flow_matching.py --config configs/experiments/flow_cifar10_smoke.yaml`

full run:
`python scripts/train_flow_matching.py --config configs/experiments/flow_cifar10.yaml`

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
