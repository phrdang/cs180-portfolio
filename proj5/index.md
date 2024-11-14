<!-- Mathjax Support -->
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<!-- TODO upscale output images -->

# Project 5

1. Table of Contents
{:toc}

## Project 5A: The Power of Diffusion Models!

### Part 0: Setup

In the first part of the project, I played around with an already trained stable diffusion model, [DeepFloyd IF](https://huggingface.co/docs/diffusers/api/pipelines/deepfloyd_if).

To try out the diffusion model, I tried the following 3 prompts with the following `num_inference_steps` values. A higher number of steps resulted in higher quality output. I used a random seed of `180`.

20 steps:

| A man wearing a hat | A rocket ship | An oil painting of a snowy mountain village |
| :--- | :--- | :--- |
| ![A man wearing a hat](assets/a/0/20-steps/a%20man%20wearing%20a%20hat.png) | ![A rocket ship](assets/a/0/20-steps/a%20rocket%20ship.png) | ![An oil painting of a snowy mountain village](assets/a/0/20-steps/an%20oil%20painting%20of%20a%20snowy%20mountain%20village.png) |

100 steps:

| A man wearing a hat | A rocket ship | An oil painting of a snowy mountain village |
| :--- | :--- | :--- |
| ![A man wearing a hat](assets/a/0/100-steps/a%20man%20wearing%20a%20hat.png) | ![A rocket ship](assets/a/0/100-steps/a%20rocket%20ship.png) | ![An oil painting of a snowy mountain village](assets/a/0/100-steps/an%20oil%20painting%20of%20a%20snowy%20mountain%20village.png) |

### Part 1: Sampling Loops

#### 1.1 Implementing the Forward Process

First, I implemented a function `forward` which takes in an original image `im` and time step `t`. A higher `t` means more noise was added. Mathematically, this is represented as follows:

$$
x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \epsilon
$$

where:

- $$x_t$$ is the image at time step $$t$$ ($$x_0$$ is the original image)
- $$\bar{\alpha_t}$$ is a noise coefficient chosen by the people who trained DeepFloyd
- $$\epsilon \sim \mathcal{N}(0, 1)$$ is Gaussian noise

| Original | `t = 250` | `t = 500` | `t = 750` |
| :--- | :--- | :--- | :--- |
| ![original campanile image](assets/a/1/1/campanile.png) | ![noisy campanile image at t = 250](assets/a/1/1/campanile-250.png) | ![noisy campanile image at t = 500](assets/a/1/1/campanile-500.png) | ![noisy campanile image at t = 750](assets/a/1/1/campanile-750.png) |

#### 1.2 Classical Denoising

Since stable diffusion is essentially trying to denoise an image step by step, first, I tried a classical denoising technique to attempt to recover the original image: [Gaussian blur](https://pytorch.org/vision/0.16/generated/torchvision.transforms.functional.gaussian_blur.html). As expected, the denoised output does not work very well. We get rid of the noise, but we also get rid of the "signal" (e.g. the actual original image).

| t | Noisy | Denoised |
| :--- | :--- | :--- |
| 250 | ![noisy campanile image at t = 250](assets/a/1/1/campanile-250.png) | ![denoised campanile image at t = 250](assets/a/1/2/denoised-250.png) |
| 500 | ![noisy campanile image at t = 500](assets/a/1/1/campanile-500.png) | ![denoised campanile image at t = 500](assets/a/1/2/denoised-500.png) |
| 750 | ![noisy campanile image at t = 750](assets/a/1/1/campanile-750.png) | ![denoised campanile image at t = 750](assets/a/1/2/denoised-750.png) |

#### 1.3 One-Step Denoising

Next I used a pretrained diffusion model (a UNet trained on a large dataset of pairs of $$x_0$$ and $$x_t$$ images) to denoise the noisy images. The UNet uses estimates of the noise at a particular timestep to denoise the image. The output is noticeably better than using Gaussian blur, but at higher `t` values (such as `t = 750`) the denoised output tower starts to look quite different from the original Campanile tower.

For reference, here is the original image again:

![original campanile image](assets/a/1/1/campanile.png)

| t | Noisy | Denoised |
| :--- | :--- | :--- |
| 250 | ![noisy campanile image at t = 250](assets/a/1/1/campanile-250.png) | ![denoised campanile image at t = 250](assets/a/1/3/denoised-250.png) |
| 500 | ![noisy campanile image at t = 500](assets/a/1/1/campanile-500.png) | ![denoised campanile image at t = 500](assets/a/1/3/denoised-500.png) |
| 750 | ![noisy campanile image at t = 750](assets/a/1/1/campanile-750.png) | ![denoised campanile image at t = 750](assets/a/1/3/denoised-750.png) |

#### 1.4 Iterative Denoising

To solve the issue mentioned in 1.3, we can use iterative denoising with a stride of 30 time steps, starting at `t = 990` to `t = 0`. I implemented the function `iterative_denoise` which uses the following formula:

$$
x_{t'} = \frac{\sqrt{\bar{\alpha_{t'}}} \beta_t}{1 - \bar{\alpha_t}} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha_{t'}})}{1 - \bar{\alpha_t}} x_t + v_{\sigma}
$$

where:

- $$x_t$$ is the same as in 1.1
- $$x_{t'}$$ is the noisy image at time step $$t'$$ where $$t' < t$$ (e.g. a slightly less noisy image)
- $$\bar{\alpha_t}$$ is the same as in 1.1
- $$\alpha_t$$ = $$\frac{\bar{\alpha_t}}{\bar{\alpha_{t'}}}$$
- $$\beta_t$$ = $$1 - \alpha_t$$
- $$x_0$$ is the same as in 1.1
- $$v_\sigma$$ is random noise

See also: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) by Jonathan Ho, Ajay Jain, and Pieter Abbeel.

The following shows the results of iterative denoising for every 5 loops:

| `t = 690` | `t = 540` | `t = 390` | `t = 240` | `t = 90` |
| :--- | :--- | :--- | :--- | :--- |
| ![iterative denoise output t = 690](assets/a/1/4/iter_denoise_10.png) | ![iterative denoise output t = 540](assets/a/1/4/iter_denoise_15.png) | ![iterative denoise output t = 390](assets/a/1/4/iter_denoise_20.png) | ![iterative denoise output t = 240](assets/a/1/4/iter_denoise_25.png) | ![iterative denoise output t = 90](assets/a/1/4/iter_denoise_30.png) |

Here are the other outputs for reference:

| Original | Gaussian Blur Denoised | One-Step Denoised | Iterative Denoised |
| :--- | :--- | :--- | :--- |
| ![original campanile image](assets/a/1/1/campanile.png) | ![gaussian blur denoised](assets/a/1/4/blur_filtered.png) | ![one-step denoised](assets/a/1/4/clean_one_step.png) | ![iterative denoised](assets/a/1/4/iter_denoise_final.png) |

We can observe that the iteratively denoised image produces the best result.

#### 1.5 Diffusion Model Sampling

Stable diffusion works by converting noise to completely denoised images. Thus, I can use my `iterative_denoise` function from 1.4 to generate images from completely noisy images (e.g. an image with pixels from a [Gaussian distribution](https://pytorch.org/docs/stable/generated/torch.randn.html)). Here are 5 generated images:

| Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| :--- | :--- | :--- | :--- | :--- |
| ![sample 1](assets/a/1/5/sample1.png) | ![sample 2](assets/a/1/5/sample2.png) | ![sample 3](assets/a/1/5/sample3.png) | ![sample 4](assets/a/1/5/sample4.png) | ![sample 5](assets/a/1/5/sample5.png) |

#### 1.6 Classifier-Free Guidance (CFG)

Some of the images in 1.5 are non-sensical, so to improve image output quality (at the expense of image diversity), I used a technique called [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) developed by Jonathan Ho and Tim Salimans.

In CFG, we compute a conditional and unconditional noise estimate $$\epsilon_c$$ and $$\epsilon_u$$, respectively. Then we let our new noise estimate be:

$$
\epsilon = \epsilon_u + \gamma (\epsilon_c - \epsilon_u)
$$

where $$\gamma$$ controls the strength of the CFG. When $$\gamma \gt 1$$, we get higher quality images, such as the 5 random samples below:

| Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| :--- | :--- | :--- | :--- | :--- |
| ![sample 1](assets/a/1/6/sample1.png) | ![sample 2](assets/a/1/6/sample2.png) | ![sample 3](assets/a/1/6/sample3.png) | ![sample 4](assets/a/1/6/sample4.png) | ![sample 5](assets/a/1/6/sample5.png) |

#### 1.7 Image-to-image Translation

Below are the results of taking 3 original images, adding varying levels of noise, and then iteratively denoising the noisy images with CFG. `i_start` denotes where we begin the iterative denoising process (e.g. lower value means we denoised with a few iterations and higher value means we denoised with a lot of iterations).

In addition to the given test image of the Campanile, I chose an image of the [Golden Gate Bridge](assets/a/1/7/0/golden-gate/golden-gate-bridge.jpg) and the [Eiffel Tower](assets/a/1/7/0/eiffel/eiffel.jpg) that were resized to $$64 \times 64$$.

| `i_start = 1` | `i_start = 3` | `i_start = 5` | `i_start = 7` | `i_start = 10` | `i_start = 20` | Original |
| ![denoised campanile with i_start = 1](assets/a/1/7/0/campanile/sample1.png) | ![denoised campanile with i_start = 3](assets/a/1/7/0/campanile/sample3.png) | ![denoised campanile with i_start = 5](assets/a/1/7/0/campanile/sample5.png) | ![denoised campanile with i_start = 7](assets/a/1/7/0/campanile/sample7.png) | ![denoised campanile with i_start = 10](assets/a/1/7/0/campanile/sample10.png) | ![denoised campanile with i_start = 20](assets/a/1/7/0/campanile/sample20.png) | ![original campanile image](assets/a/1/1/campanile.png) |
| ![denoised golden gate bridge with i_start = 1](assets/a/1/7/0/golden-gate/golden-gate-bridge1.png) | ![denoised golden gate bridge with i_start = 3](assets/a/1/7/0/golden-gate/golden-gate-bridge3.png) | ![denoised golden gate gate bridge with i_start = 5](assets/a/1/7/0/golden-gate/golden-gate-bridge5.png) | ![denoised golden gate bridge with i_start = 7](assets/a/1/7/0/golden-gate/golden-gate-bridge7.png) | ![denoised golden gate bridge with i_start = 10](assets/a/1/7/0/golden-gate/golden-gate-bridge10.png) | ![denoised golden gate bridge with i_start = 20](assets/a/1/7/0/golden-gate/golden-gate-bridge20.png) | ![original golden gate bridge image](assets/a/1/7/0/golden-gate/golden-gate-bridge-resized.png) |
| ![denoised Eiffel Tower with i_start = 1](assets/a/1/7/0/eiffel/eiffel1.png) | ![denoised Eiffel Tower with i_start = 3](assets/a/1/7/0/eiffel/eiffel3.png) | ![denoised Eiffel Tower with i_start = 5](assets/a/1/7/0/eiffel/eiffel5.png) | ![denoised Eiffel Tower with i_start = 7](assets/a/1/7/0/eiffel/eiffel7.png) | ![denoised Eiffel Tower with i_start = 10](assets/a/1/7/0/eiffel/eiffel10.png) | ![denoised Eiffel Tower with i_start = 20](assets/a/1/7/0/eiffel/eiffel20.png) | ![original Eiffel Tower image](assets/a/1/7/0/eiffel/eiffel-resized.png) |

See also: [SDEdit algorithm](https://sde-image-editing.github.io/) by Meng et al. (2022).

##### 1.7.1 Editing Hand-Drawn and Web Images

In construction

##### 1.7.2 Inpainting

In construction

##### 1.7.3 Text-Conditional Image-to-image Translation

In construction

#### 1.8 Visual Anagrams

In construction

#### 1.9 Hybrid Images

In construction

### Part 2: Bells & Whistles

In construction

## Project 5B: Diffusion Models from Scratch!

In construction

### Part 1: Training a Single-Step Denoising U-Net

In construction

### Part 2: Training a DDPM Denoising U-Net

In construction

### Part 3: Bells & Whistles

In construction
