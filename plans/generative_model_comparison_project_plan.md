# Project Plan: Modular Small-Scale Generative Model Comparison Framework

## 1. Project Goal

Build a **scalable, modular research codebase** for training and evaluating small versions of several generative model families on the same datasets under controlled conditions.

The initial model families are:

1. **DDPM**
2. **Diffusion Transformer**
3. **Flow Matching**
4. **Consistency Models**

The purpose is not to achieve state-of-the-art image generation, but to enable **apples-to-apples statistical comparisons** across model types.

Primary comparison goals include:

- Distributional fidelity
- Mode coverage
- Generalization behavior
- Bias-variance behavior
- Consistency of generated marginal summaries
- Training loss behavior
- Validation loss behavior
- Sampling quality under controlled settings

The codebase should be designed for both:

- **Human researchers**, who need clarity, reproducibility, and extensibility.
- **Coding agents**, such as Codex or Claude Code, which need explicit file organization, module boundaries, and implementation instructions.

---

## 1.1 Minimum Viable Project Scope

The full framework vision is intentionally broad. The first complete deliverable should use a narrower scope that produces statistically meaningful results without requiring every model family to be implemented immediately.

Initial MVP:

1. Shared config, dataset loading, seeding, device handling, output directories, checkpointing, and logging.
2. DDPM with a U-Net backbone trains successfully on CIFAR-10 at 32x32.
3. Sampling and KID evaluation work end-to-end for the DDPM baseline.
4. Flow Matching with the same U-Net backbone trains and samples successfully.
5. Tiny DiT trained under the DDPM noise-prediction objective trains and samples successfully.
6. Final comparison uses 3 random seeds after the full single-seed pipeline is stable.

Initial model set:

```text
1. DDPM objective + U-Net backbone
2. Flow Matching objective + same U-Net backbone
3. DDPM objective + Tiny DiT backbone
```

Initial primary metric:

```text
KID
```

Initial secondary metrics:

```text
FID
NFE / sampling time
validation loss
loss by timestep or noise bucket
channel means and variances
brightness and contrast
edge density
```

Deferred until after the MVP:

```text
Consistency models
64x64 experiments
precision-recall metrics
5-seed runs
larger datasets
larger architectures
```

This MVP gives two clean comparisons:

```text
Objective comparison:
DDPM-U-Net vs. FlowMatching-U-Net

Architecture comparison:
DDPM-U-Net vs. DDPM-TinyDiT
```

---

## 2. Design Principles

### 2.1 Modularity

Each major concept should live in its own module:

- Model architectures
- Training objectives
- Samplers
- Datasets
- Metrics
- Experiment configuration
- Logging
- Visualization

The code should avoid mixing model architecture, loss computation, sampling logic, and evaluation logic in a single file.

---

### 2.2 Organized by Model Type

Each model family should have its own folder, even when it reuses shared components.

For example, DDPM, flow matching, and consistency models may all use the same U-Net architecture, but their objectives and samplers should remain separate.

This makes it easy to inspect and modify one model type without accidentally affecting another.

---

### 2.3 Shared Infrastructure

All model types should use the same shared infrastructure where possible:

- Same dataset loaders
- Same train-validation split
- Same image preprocessing
- Same optimizer defaults
- Same logging format
- Same Weights & Biases logging conventions
- Same checkpoint format
- Same metric computation
- Same evaluation scripts
- Same random seed handling

This is necessary for fair comparison.

---

### 2.4 Scalability

The initial project should work on small datasets and small models, but the structure should support scaling up.

The framework should allow changes in:

- Dataset size
- Image resolution
- Model size
- Number of training steps
- Number of sampling steps
- Number of evaluation samples
- Hardware setup
- Number of random seeds

without requiring major rewrites.

---

### 2.5 Minimal Duplication

Common code should live in shared modules.

Model-specific code should only include what is genuinely specific to that model family.

For example:

- DDPM-specific noise prediction loss belongs in `models/ddpm/`.
- Generic U-Net wrappers belong in `architectures/`.
- Generic image metrics belong in `evaluation/`.

---

## 3. Recommended Repository Structure

```text
generative-comparison/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── datasets/
│   │   ├── cifar10.yaml
│   │   ├── flowers.yaml
│   │   └── imagefolder.yaml
│   │
│   ├── models/
│   │   ├── ddpm_unet.yaml
│   │   ├── dit_tiny.yaml
│   │   ├── flow_unet.yaml
│   │   └── consistency_unet.yaml
│   │
│   └── experiments/
│       ├── ddpm_cifar10.yaml
│       ├── dit_cifar10.yaml
│       ├── flow_cifar10.yaml
│       └── consistency_cifar10.yaml
│
├── src/
│   ├── main.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── registry.py
│   │   ├── seeding.py
│   │   ├── checkpointing.py
│   │   ├── logging.py
│   │   └── device.py
│   │
│   ├── data/
│   │   ├── datasets.py
│   │   ├── transforms.py
│   │   ├── dataloaders.py
│   │   └── splits.py
│   │
│   ├── architectures/
│   │   ├── unet_diffusers.py
│   │   ├── tiny_dit.py
│   │   ├── embeddings.py
│   │   └── blocks.py
│   │
│   ├── models/
│   │   ├── ddpm/
│   │   │   ├── model.py
│   │   │   ├── objective.py
│   │   │   ├── sampler.py
│   │   │   └── trainer.py
│   │   │
│   │   ├── dit/
│   │   │   ├── model.py
│   │   │   ├── objective.py
│   │   │   ├── sampler.py
│   │   │   └── trainer.py
│   │   │
│   │   ├── flow_matching/
│   │   │   ├── model.py
│   │   │   ├── objective.py
│   │   │   ├── sampler.py
│   │   │   └── trainer.py
│   │   │
│   │   └── consistency/
│   │       ├── model.py
│   │       ├── objective.py
│   │       ├── sampler.py
│   │       └── trainer.py
│   │
│   ├── training/
│   │   ├── base_trainer.py
│   │   ├── optimizers.py
│   │   ├── schedulers.py
│   │   ├── ema.py
│   │   └── loop.py
│   │
│   ├── sampling/
│   │   ├── sample.py
│   │   ├── save_images.py
│   │   └── grids.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── fid_kid.py
│   │   ├── precision_recall.py
│   │   ├── summary_statistics.py
│   │   ├── nearest_neighbors.py
│   │   └── evaluate.py
│   │
│   └── visualization/
│       ├── loss_curves.py
│       ├── metric_curves.py
│       ├── generated_grids.py
│       └── summary_plots.py
│
├── scripts/
│   ├── train.py
│   ├── sample.py
│   ├── evaluate.py
│   ├── compare_runs.py
│   └── make_report.py
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── metric_sanity_checks.ipynb
│   └── comparison_plots.ipynb
│
├── outputs/
│   ├── checkpoints/
│   ├── samples/
│   ├── metrics/
│   ├── logs/
│   └── reports/
│
└── tests/
    ├── test_data.py
    ├── test_architectures.py
    ├── test_objectives.py
    ├── test_samplers.py
    └── test_metrics.py
```

---

## 4. Model Family Responsibilities

### 4.1 DDPM

The DDPM module should implement the standard denoising diffusion baseline.

Suggested location:

```text
src/models/ddpm/
```

Responsibilities:

- Define the DDPM model wrapper.
- Use a U-Net denoiser.
- Implement the noise prediction objective.
- Implement or wrap the DDPM sampler.
- Log loss by timestep.
- Support training and validation under the shared training loop.

The DDPM objective should train:

```math
\epsilon_\theta(x_t, t) \approx \epsilon
```

with loss:

```math
\mathcal{L}_{DDPM}
=
\mathbb{E}_{x_0, \epsilon, t}
\left[
\|\epsilon - \epsilon_\theta(x_t, t)\|^2
\right].
```

Recommended architecture source:

```python
from diffusers import UNet2DModel, DDPMScheduler
```

Expected amount of custom architecture code:

```text
Low
```

Most DDPM work should be objective, training, and evaluation logic.

---

### 4.2 Diffusion Transformer

The DiT module should implement a small transformer-based diffusion model.

Suggested location:

```text
src/models/dit/
```

Responsibilities:

- Define a small DiT model.
- Patchify images or latent images.
- Add positional embeddings.
- Add timestep embeddings.
- Use transformer blocks to predict denoising targets.
- Support the same DDPM-style objective initially.
- Allow parameter count matching against the U-Net baseline.

The first DiT comparison should use the same objective as DDPM:

```math
\epsilon_\theta(x_t, t) \approx \epsilon
```

This isolates the architectural difference:

```math
\text{U-Net backbone}
\quad \text{vs.} \quad
\text{Transformer backbone}
```

Recommended implementation approach:

- Either adapt a small DiT from `facebookresearch/DiT`.
- Or write a compact `TinyDiT` in `src/architectures/tiny_dit.py`.

Expected amount of custom architecture code:

```text
Moderate
```

The DiT architecture is likely the main model architecture that needs custom work.

---

### 4.3 Flow Matching

The flow matching module should implement a vector-field model trained to map between noise and data.

Suggested location:

```text
src/models/flow_matching/
```

Responsibilities:

- Reuse the shared U-Net architecture.
- Implement flow matching interpolation.
- Implement the flow matching vector-field loss.
- Implement Euler or ODE-based sampling.
- Support validation loss and summary statistics.

A simple setup is:

```math
x_t = (1-t)x_0 + t\epsilon
```

where:

```math
x_0 \sim p_{\text{data}}, \qquad \epsilon \sim \mathcal{N}(0,I)
```

and the model learns:

```math
v_\theta(x_t, t) \approx \epsilon - x_0.
```

Loss:

```math
\mathcal{L}_{FM}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\|v_\theta(x_t,t)-(\epsilon-x_0)\|^2
\right].
```

Recommended architecture source:

```python
from diffusers import UNet2DModel
```

Expected amount of custom architecture code:

```text
Low
```

Expected amount of custom objective and sampler code:

```text
Moderate
```

---

### 4.4 Consistency Models

The consistency model module should implement a small consistency model trained from scratch or through distillation.

Suggested location:

```text
src/models/consistency/
```

Responsibilities:

- Reuse the shared U-Net architecture.
- Implement consistency training or consistency distillation.
- Implement the noise-level sampling schedule.
- Implement EMA or target network logic.
- Implement one-step and few-step sampling.
- Support validation consistency loss.

A consistency model should learn a function that maps noisy samples at different noise levels to a consistent prediction.

Conceptually:

```math
f_\theta(x_t, t) \approx f_\theta(x_s, s)
```

for appropriately related noisy samples `x_t` and `x_s`.

Recommended architecture source:

```python
from diffusers import UNet2DModel
```

Recommended reference source:

```text
openai/consistency_models
```

Expected amount of custom architecture code:

```text
Low
```

Expected amount of custom training logic:

```text
Moderate to high
```

This is likely the trickiest objective to implement correctly.

---

## 5. Architecture Layer

The `src/architectures/` folder should contain reusable neural network components.

### 5.1 U-Net Wrapper

File:

```text
src/architectures/unet_diffusers.py
```

Purpose:

- Provide a thin project-specific wrapper around Diffusers `UNet2DModel`.
- Normalize input and output conventions.
- Make U-Net usable across DDPM, flow matching, and consistency models.

Suggested responsibilities:

```python
class DiffusersUNetWrapper:
    def forward(self, x, t):
        ...
```

This wrapper should hide Diffusers-specific return formats.

For example, Diffusers models often return objects with `.sample`. The wrapper should return a tensor directly.

---

### 5.2 Tiny DiT

File:

```text
src/architectures/tiny_dit.py
```

Purpose:

- Implement a small transformer denoiser for low-resolution images.
- Keep the architecture readable.
- Make parameter count configurable.
- Match the same input-output interface as the U-Net.

Required features:

- Patch embedding
- Positional embedding
- Timestep embedding
- Transformer blocks
- Final projection
- Unpatchifying output back to image shape

Suggested interface:

```python
class TinyDiT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        ...

    def forward(self, x, t):
        ...
```

The output shape should match the input shape:

```python
output.shape == x.shape
```

This is important so that DDPM and DiT can share objective code where possible.

---

## 6. Objective Layer

Each model family should expose a function or class that computes the training loss.

Suggested interface:

```python
class Objective:
    def loss(self, model, batch, batch_idx):
        ...
```

or:

```python
def compute_loss(model, batch, config):
    ...
```

Each objective should return a dictionary:

```python
{
    "loss": loss,
    "loss_main": loss_main,
    "loss_aux": optional_auxiliary_loss,
    "diagnostics": {
        ...
    }
}
```

The dictionary format allows logging of model-specific diagnostics while keeping the training loop generic.

---

### 6.1 DDPM Objective

File:

```text
src/models/ddpm/objective.py
```

Responsibilities:

- Sample timestep `t`.
- Sample Gaussian noise `epsilon`.
- Construct noisy image `x_t`.
- Predict noise.
- Compute MSE loss.

Return diagnostics:

- Mean timestep
- Loss by timestep bucket
- Noise norm
- Prediction norm

---

### 6.2 DiT Objective

File:

```text
src/models/dit/objective.py
```

Initial implementation:

- Use the same DDPM noise prediction objective.
- Keep this intentionally close to DDPM.

Possible later extensions:

- `x_0`-prediction
- `v`-prediction
- Class conditioning
- Latent-space DiT

---

### 6.3 Flow Matching Objective

File:

```text
src/models/flow_matching/objective.py
```

Responsibilities:

- Sample `t ~ U(0,1)`.
- Sample noise `epsilon`.
- Construct interpolation `x_t`.
- Compute target vector field.
- Train model to predict vector field.

Return diagnostics:

- Mean `t`
- Vector field norm
- Prediction norm
- Loss by time bucket

---

### 6.4 Consistency Objective

File:

```text
src/models/consistency/objective.py
```

Responsibilities:

- Sample two noise levels.
- Construct paired noisy samples.
- Apply consistency model to both.
- Use EMA or target network where needed.
- Compute consistency loss.
- Apply boundary conditions if using skip-out parameterization.

Return diagnostics:

- Noise level distribution
- Consistency gap
- Prediction norm
- Target norm

---

## 7. Sampler Layer

Sampling should be separate from training.

Suggested location:

```text
src/models/{model_type}/sampler.py
```

All samplers should expose a common interface:

```python
class Sampler:
    def sample(
        self,
        model,
        num_samples: int,
        image_shape: tuple,
        device: torch.device,
        num_steps: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        ...
```

The output should always be:

```python
Tensor[num_samples, channels, height, width]
```

with values in a consistent range, preferably:

```text
[-1, 1]
```

---

### 7.1 DDPM Sampler

Can use Diffusers `DDPMScheduler`.

Responsibilities:

- Start from Gaussian noise.
- Iteratively denoise.
- Save intermediate states optionally.

---

### 7.2 DiT Sampler

Can initially use the same DDPM sampling logic as the DDPM model.

This keeps the architecture comparison clean.

---

### 7.3 Flow Matching Sampler

Implement a simple Euler solver.

Basic sampling direction:

- Start from noise.
- Integrate learned vector field toward data.

The exact time direction should be documented clearly in the code.

---

### 7.4 Consistency Sampler

Implement:

- One-step sampling
- Few-step sampling

The sampler should support configurable noise levels.

---

## 8. Training System

The training system should be mostly model-agnostic.

Suggested files:

```text
src/training/base_trainer.py
src/training/loop.py
src/training/ema.py
src/training/optimizers.py
src/training/schedulers.py
```

### 8.1 Base Trainer

The base trainer should handle:

- Device placement
- Mixed precision
- Gradient accumulation
- Optimizer steps
- Learning-rate scheduling
- EMA updates
- Checkpointing
- Logging
- Weights & Biases run initialization and metric logging
- Validation
- Periodic sampling

The base trainer should not contain model-specific loss logic.

Instead, it should call:

```python
loss_dict = objective.loss(model, batch, batch_idx)
```

---

### 8.2 Model-Specific Trainers

Each model family may have a small trainer wrapper:

```text
src/models/ddpm/trainer.py
src/models/dit/trainer.py
src/models/flow_matching/trainer.py
src/models/consistency/trainer.py
```

These should only contain model-specific training details.

For example:

- Consistency model trainer may need target network updates.
- DDPM trainer may not need anything beyond the base trainer.
- Flow matching trainer may define time-sampling behavior.
- DiT trainer may add transformer-specific initialization or parameter grouping.

---

## 9. Configuration System

Use YAML configs to define experiments.

A config should specify:

```yaml
experiment:
  name: ddpm_cifar10_seed0
  seed: 0
  output_dir: outputs/runs/ddpm_cifar10_seed0

dataset:
  name: cifar10
  image_size: 32
  channels: 3
  train_split: train
  val_split: test
  normalize: true

model:
  type: ddpm
  architecture: unet
  hidden_channels: 128

training:
  batch_size: 128
  num_steps: 100000
  learning_rate: 0.0002
  weight_decay: 0.0
  grad_clip_norm: 1.0
  ema_decay: 0.999

sampling:
  num_steps: 100
  num_samples: 5000

evaluation:
  metrics:
    - fid
    - kid
    - precision_recall
    - summary_statistics
  eval_every: 5000
```

The config should make it easy to run the same experiment across model types.

Example:

```bash
python scripts/train.py --config configs/experiments/ddpm_cifar10.yaml
python scripts/train.py --config configs/experiments/dit_cifar10.yaml
python scripts/train.py --config configs/experiments/flow_cifar10.yaml
python scripts/train.py --config configs/experiments/consistency_cifar10.yaml
```

---

## 10. Evaluation System

Evaluation should be shared across all model types.

Suggested location:

```text
src/evaluation/
```

Metrics should include:

### 10.1 Distributional Fidelity

- FID
- KID
- MMD if useful

### 10.2 Mode Coverage

- Precision and recall for generative models
- Class frequency comparison if labels are available
- Cluster coverage in embedding space

### 10.3 Generalization

- Training loss
- Validation loss
- Train-validation gap
- Loss by timestep or noise level

### 10.4 Bias-Variance Behavior

Run multiple seeds:

```text
seed = 0, 1, 2, 3, 4
```

Then compare:

- Mean metric value
- Standard deviation across seeds
- Variability of generated summary statistics
- Variability of loss curves

### 10.5 Marginal Summary Consistency

Compare real and generated samples using summary statistics.

Possible summaries:

- Channel means
- Channel variances
- Brightness distribution
- Contrast distribution
- Edge density
- Color histograms
- Embedding means
- Embedding covariances
- Class proportions if labels are available

The evaluation module should save results as structured files:

```text
outputs/metrics/{run_name}/metrics.json
outputs/metrics/{run_name}/summary_statistics.csv
outputs/metrics/{run_name}/loss_by_timestep.csv
```

---

## 11. Experiment Naming and Output Organization

Each experiment should produce an isolated output directory.

Recommended structure:

```text
outputs/runs/
  ddpm_cifar10_seed0/
    config.yaml
    checkpoints/
    samples/
    metrics/
    logs/

  dit_cifar10_seed0/
    config.yaml
    checkpoints/
    samples/
    metrics/
    logs/

  flow_cifar10_seed0/
    config.yaml
    checkpoints/
    samples/
    metrics/
    logs/

  consistency_cifar10_seed0/
    config.yaml
    checkpoints/
    samples/
    metrics/
    logs/
```

Each run directory should contain the exact config used for that run.

This is essential for reproducibility.

---

## 12. Implementation Order

### Phase 1: Shared Infrastructure

Implement:

1. Config loading
2. Dataset loading
3. U-Net wrapper
4. Base training loop
5. Checkpointing
6. Logging
7. Sampling image saving

Target outcome:

```text
One DDPM model trains successfully on CIFAR-10 or an image folder dataset.
```

---

### Phase 2: DDPM Baseline

Implement:

1. DDPM objective
2. DDPM sampler
3. DDPM validation loss
4. Sample generation
5. Basic metric evaluation

Target outcome:

```text
A working DDPM baseline with saved samples, checkpoints, and loss curves.
```

---

### Phase 3: Flow Matching

Implement:

1. Flow matching objective
2. Flow matching Euler sampler
3. Flow validation loss
4. Shared U-Net support

Target outcome:

```text
A flow matching model using the same U-Net backbone as DDPM.
```

---

### Phase 4: Tiny DiT

Implement:

1. Tiny DiT architecture
2. DiT DDPM objective
3. DiT sampler using DDPM-style denoising
4. Parameter count matching utility

Target outcome:

```text
A transformer-based diffusion model trained under the same objective as DDPM.
```

---

### Phase 5: Consistency Model

Implement:

1. Consistency objective
2. Noise-level schedule
3. EMA or target network
4. One-step sampler
5. Few-step sampler

Target outcome:

```text
A small consistency model trained and evaluated under the shared framework.
```

---

### Phase 6: Evaluation and Reporting

Implement:

1. FID and KID
2. Precision-recall
3. Summary statistics
4. Loss by timestep/noise level
5. Cross-seed comparison
6. Report generation

Target outcome:

```text
A reproducible comparison table and plots across DDPM, DiT, flow matching, and consistency models.
```

---

## 13. Coding Agent Instructions

Coding agents should follow these rules when modifying the project.

### 13.1 Do Not Mix Model Families

Do not place DDPM, flow matching, DiT, and consistency model logic in the same file unless the code is genuinely shared.

Use:

```text
src/models/ddpm/
src/models/dit/
src/models/flow_matching/
src/models/consistency/
```

---

### 13.2 Keep Architecture Separate from Objective

Architecture files should define neural networks.

Objective files should define losses.

Sampler files should define generation procedures.

Do not put training loss code inside architecture classes unless absolutely necessary.

---

### 13.3 Preserve Common Interfaces

All model architectures should support:

```python
model(x, t)
```

and return a tensor with the same shape as `x`.

All objective modules should return a dictionary with at least:

```python
{
    "loss": loss
}
```

All samplers should return generated images as tensors.

---

### 13.4 Prefer Thin Wrappers Around External Libraries

Use Diffusers where possible, but wrap external components behind project-specific interfaces.

Good:

```python
class DiffusersUNetWrapper(nn.Module):
    ...
```

Bad:

```python
# Directly using Diffusers-specific return types throughout the training loop
```

The rest of the codebase should not need to know Diffusers internals.

---

### 13.5 Avoid Hidden Experimental Differences

When adding or modifying a model type, make sure that shared experimental settings remain shared.

Do not silently change:

- Image normalization
- Batch size
- Optimizer
- Learning rate
- EMA behavior
- Number of training steps
- Evaluation sample count
- Train-validation split
- Data augmentation

unless the config explicitly requests it.

---

### 13.6 Add Tests for Every New Objective

Each objective should have a simple shape and gradient test.

For example:

```python
def test_ddpm_objective_backward():
    ...
```

The test should verify:

- Loss is scalar
- Loss is finite
- Backward pass works
- Model output shape is correct

---

## 14. Recommended First Comparison

The first complete experiment should be:

```text
Dataset: CIFAR-10
Resolution: 32x32
Channels: 3
Training steps: modest, for example 50,000 to 100,000
Seeds: 3 initially
Models:
  1. DDPM-U-Net
  2. DiT-DDPM
  3. FlowMatching-U-Net
  4. Consistency-U-Net
```

Primary metrics:

- Validation objective loss
- FID
- KID
- Precision-recall
- Loss by timestep/noise level
- Generated summary statistics
- Cross-seed variance

This gives one clean architecture comparison:

```math
\text{DDPM-U-Net} \quad \text{vs.} \quad \text{DiT-DDPM}
```

and one clean objective comparison:

```math
\text{DDPM-U-Net}
\quad \text{vs.} \quad
\text{FlowMatching-U-Net}
\quad \text{vs.} \quad
\text{Consistency-U-Net}
```

---

## 15. Summary

The intended project structure is:

```text
one shared training/evaluation framework
+
one reusable U-Net
+
one small DiT
+
separate model-family folders
+
separate objectives and samplers
```

The main implementation burden is not writing four full model architectures.

The realistic burden is:

```text
1. Wrap Diffusers U-Net.
2. Write or adapt a small DiT.
3. Implement DDPM, flow matching, and consistency objectives.
4. Implement samplers for each model family.
5. Build shared evaluation tools.
```

This setup should remain simple enough for small-scale experiments while being modular enough to scale into a serious statistical comparison framework.
