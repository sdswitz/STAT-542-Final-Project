## Project Abstract

We will compare different styles of diffusion models for the task of image generation, specifically DDPM, Stable Diffusion, Transformer-based diffusion, Flow Matching and consistency models. We will look for pretrained models with these architectures and then compare them across statistical metrics such as Frechet and Kernel Inception Distance, computational cost, and bias/variance tradeoffs.

## Desired Evaluation Metrics

- Robustness & Stability: Test how these models react to noise and hyperparameter tunings (any perturbations to the data and model)
- Inception scores: how sharpness and diversity within the model are satisfied and calculated
    - FID (Frechet Inception Distance): evaluate quality and diversity of generated models by 
    - KID (Kernel Inception Distance): computes the Maximum Mean Discrepancy (MMD) between real and generated Inception features. Performs better in smaller samples and is unbiased, compared to FID.
- Inference Speed / Number of Function Evaluations (NFE): compare the computational cost and time to generate a single image 
- Analyze Bias and Variance of models with experiments:
    - Bias: train models on different training set sizes and monitor KID
    - Variance: train models across different seeds and compute variance of metrics like KID and FID
- Consistency under marginal summaries of the images
    - Real vs generated means of image characteristics like pixel values, texture, class frequency


## Papers and models for final project

### DDPM-style diffusion:
https://huggingface.co/google/ddpm-cifar10-32
https://arxiv.org/pdf/2006.11239

### Stable Diffusion:
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
https://arxiv.org/pdf/2112.10752

## DiT
https://huggingface.co/facebook/DiT-XL-2-256
https://arxiv.org/pdf/2212.09748

### SiT
https://scalable-interpolant.github.io/
https://github.com/willisma/SiT
https://arxiv.org/pdf/2401.08740