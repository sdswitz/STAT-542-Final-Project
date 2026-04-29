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