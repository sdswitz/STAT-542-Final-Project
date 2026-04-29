# STAT 542 Spring 26 Final Project
This is the code for our final project on diffusion models in STAT 542 at UIUC.

As of 4/29, the plan is to:
- run our own DDPM model on CIFAR-10 (in progress)
- run our own flow matching model on CIFAR-10 (todo)
    - the config for the flow matching model is in the config folder
- (maybe) run our own diffusion-transformer model on CIFAR-10
    - no config exists for this yet

## TODO

- [ ] Smoke-test DDPM training on the compute cluster.
- [ ] Start the full DDPM CIFAR-10 run.
- [ ] Smoke-test flow matching training on the compute cluster.
- [ ] Start the full flow matching CIFAR-10 run.
- [ ] Enable Weights & Biases tracking in the experiment configs:
    - set `wandb.enabled: true`
    - set `wandb.project: stat542-generative-comparison`
    - set `wandb.entity` if using a team account, otherwise leave it as `null`
- [ ] Log in to W&B on the cluster with `wandb login` or set `WANDB_API_KEY` in the shell.
- [ ] Add local JSONL or CSV logging for training and validation metrics under `outputs/runs/<run_name>/metrics/`.
- [ ] Decide whether to log generated image grids and checkpoints as W&B artifacts.
- [ ] Implement the shared KID/FID evaluation protocol.
- [ ] Write the more in-depth evaluation functions for final model comparison.
- [ ] Add the Tiny DiT baseline after DDPM and flow matching are stable.
