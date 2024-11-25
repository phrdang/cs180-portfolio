<!-- Mathjax Support -->
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<style>
.rotate180 {
    -webkit-transform: rotate(180deg);
    -moz-transform: rotate(180deg);
    -o-transform: rotate(180deg);
    -ms-transform: rotate(180deg);
    transform: rotate(180deg);
}
</style>

# Project 5

Fun With Diffusion Models! - [Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj5/index.html)

1. Table of Contents
{:toc}

## Project 5A: The Power of Diffusion Models!

[Part A Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj5/parta.html)

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
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ![denoised campanile with i_start = 1](assets/a/1/7/0/campanile/sample1.png) | ![denoised campanile with i_start = 3](assets/a/1/7/0/campanile/sample3.png) | ![denoised campanile with i_start = 5](assets/a/1/7/0/campanile/sample5.png) | ![denoised campanile with i_start = 7](assets/a/1/7/0/campanile/sample7.png) | ![denoised campanile with i_start = 10](assets/a/1/7/0/campanile/sample10.png) | ![denoised campanile with i_start = 20](assets/a/1/7/0/campanile/sample20.png) | ![original campanile image](assets/a/1/1/campanile.png) |
| ![denoised golden gate bridge with i_start = 1](assets/a/1/7/0/golden-gate/golden-gate-bridge1.png) | ![denoised golden gate bridge with i_start = 3](assets/a/1/7/0/golden-gate/golden-gate-bridge3.png) | ![denoised golden gate gate bridge with i_start = 5](assets/a/1/7/0/golden-gate/golden-gate-bridge5.png) | ![denoised golden gate bridge with i_start = 7](assets/a/1/7/0/golden-gate/golden-gate-bridge7.png) | ![denoised golden gate bridge with i_start = 10](assets/a/1/7/0/golden-gate/golden-gate-bridge10.png) | ![denoised golden gate bridge with i_start = 20](assets/a/1/7/0/golden-gate/golden-gate-bridge20.png) | ![original golden gate bridge image](assets/a/1/7/0/golden-gate/golden-gate-bridge-resized.png) |
| ![denoised Eiffel Tower with i_start = 1](assets/a/1/7/0/eiffel/eiffel1.png) | ![denoised Eiffel Tower with i_start = 3](assets/a/1/7/0/eiffel/eiffel3.png) | ![denoised Eiffel Tower with i_start = 5](assets/a/1/7/0/eiffel/eiffel5.png) | ![denoised Eiffel Tower with i_start = 7](assets/a/1/7/0/eiffel/eiffel7.png) | ![denoised Eiffel Tower with i_start = 10](assets/a/1/7/0/eiffel/eiffel10.png) | ![denoised Eiffel Tower with i_start = 20](assets/a/1/7/0/eiffel/eiffel20.png) | ![original Eiffel Tower image](assets/a/1/7/0/eiffel/eiffel-resized.png) |

See also: [SDEdit algorithm](https://sde-image-editing.github.io/) by Meng et al. (2022).

##### 1.7.1 Editing Hand-Drawn and Web Images

In this subpart, I drew 2 images and found 1 image from the web and utilized the diffusion model to edit the images using the same method as above.

| `i_start = 1` | `i_start = 3` | `i_start = 5` | `i_start = 7` | `i_start = 10` | `i_start = 20` | Original |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ![drawn tree 1](assets/a/1/7/1/drawn-tree/drawn-tree1.png) | ![drawn tree 3](assets/a/1/7/1/drawn-tree/drawn-tree3.png) | ![drawn tree 5](assets/a/1/7/1/drawn-tree/drawn-tree5.png) | ![drawn tree 7](assets/a/1/7/1/drawn-tree/drawn-tree7.png) | ![drawn tree 10](assets/a/1/7/1/drawn-tree/drawn-tree10.png) | ![drawn tree 20](assets/a/1/7/1/drawn-tree/drawn-tree20.png) | ![drawn tree original](assets/a/1/7/1/drawn-tree/drawn-tree.png) |
| ![drawn tulip 1](assets/a/1/7/1/tulip/tulip1.png) | ![drawn tulip 3](assets/a/1/7/1/tulip/tulip3.png) | ![drawn tulip 5](assets/a/1/7/1/tulip/tulip5.png) | ![drawn tulip 7](assets/a/1/7/1/tulip/tulip7.png) | ![drawn tulip 1](assets/a/1/7/1/tulip/tulip10.png) | ![drawn tulip 20](assets/a/1/7/1/tulip/tulip20.png) |![drawn tulip original](assets/a/1/7/1/tulip/tulip.png) |
| ![tree 1](assets/a/1/7/1/tree/tree1.png) | ![tree 3](assets/a/1/7/1/tree/tree3.png) | ![tree 5](assets/a/1/7/1/tree/tree5.png) | ![tree 7](assets/a/1/7/1/tree/tree7.png) | ![tree 10](assets/a/1/7/1/tree/tree10.png) | ![tree 20](assets/a/1/7/1/tree/tree20.png) | ![tree original](assets/a/1/7/1/tree/tree.png) |


##### 1.7.2 Inpainting

Next I implemented the inpainting part of this [paper](https://arxiv.org/abs/2201.09865), RePaint: Inpainting using Denoising Diffusion Probabilistic Models by Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. (Using this [paper](http://graphics.cs.cmu.edu/projects/scene-completion/), Scene Completion Using Millions of Photographs by James Hays and Alexei Efros, for inspiration on how to "complete" the scene.)

Given an image, $$x_{orig}$$, and binary mask $$m$$, I created a new image with the same content as the original where $$m = 0$$, but new content where $$m = 1$$. To do this, we run the diffusion denoising loop and only keep the part where the mask is 1:

$$
x_t \leftarrow m x_t + (1 - m) \text{forward}(x_{orig}, t)
$$

| Campanile with new top | Cat with dog head | Oski with dog head |
| :--- | :--- | :--- |
| ![inpainted campanile](assets/a/1/7/2/campanile.png) | ![inpainted cat](assets/a/1/7/2/cat-inpainted.png) | ![inpainted oski](assets/a/1/7/2/oski.png) |

##### 1.7.3 Text-Conditional Image-to-image Translation

In this part, I changed the text prompt from `"a high quality image"` to `"a rocket ship"`, obtaining the results below with different levels of noise for the Campanile image and some images from the web.

| `i_start = 1` | `i_start = 3` | `i_start = 5` | `i_start = 7` | `i_start = 10` | `i_start = 20` | Original |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ![campanile 1](assets/a/1/7/3/campanile/sample1.png) | ![campanile 3](assets/a/1/7/3/campanile/sample3.png) | ![campanile 5](assets/a/1/7/3/campanile/sample5.png) | ![campanile 1](assets/a/1/7/3/campanile/sample7.png) | ![campanile 10](assets/a/1/7/3/campanile/sample10.png) | ![campanile 20](assets/a/1/7/3/campanile/sample20.png) | ![campanile original](assets/a/1/1/campanile.png) |
| ![guitar 1](assets/a/1/7/3/guitar/sample1.png) | ![guitar 3](assets/a/1/7/3/guitar/sample3.png) | ![guitar 5](assets/a/1/7/3/guitar/sample5.png) | ![guitar 7](assets/a/1/7/3/guitar/sample7.png) | ![guitar 10](assets/a/1/7/3/guitar/sample10.png) | ![guitar 20](assets/a/1/7/3/guitar/sample20.png) | ![guitar original](assets/a/1/7/3/guitar/guitar.png) |
| ![waterfall 1](assets/a/1/7/3/waterfall/sample1.png) | ![waterfall 3](assets/a/1/7/3/waterfall/sample3.png) | ![waterfall 5](assets/a/1/7/3/waterfall/sample5.png) | ![waterfall 7](assets/a/1/7/3/waterfall/sample7.png) | ![waterfall 10](assets/a/1/7/3/waterfall/sample10.png) | ![waterfall 20](assets/a/1/7/3/waterfall/sample20.png) | ![waterfall 1](assets/a/1/7/3/waterfall/waterfall.png) |


#### 1.8 Visual Anagrams

In this part, I implemented Visual Anagrams from [this paper](https://dangeng.github.io/visual_anagrams/), Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models by Daniel Geng, Inbum Park, and Andrew Owens. To do this, we denoise like before but we alternate between 2 different text prompts and flip the image upside-down. That way, the final image right-side up looks like the first prompt, and upside-down looks like the second prompt.

The full algorithm is as follows:

$$
\epsilon_1 = \text{UNet}(x_t, t, p_1)
$$

$$
\epsilon_2 = \text{flip}(\text{UNet}(\text{flip}(x_t), t, p_2))
$$

$$
\epsilon = (\epsilon_1 + \epsilon_2) / 2
$$

where `flip` is a function that flips the image and `p_i` are the 2 text prompts. Our final noise estimate is $$\epsilon$$, which is the average of the 2 individual noise estimates.

<table>
    <tr>
        <th>Prompt 1</th>
        <th>Image</th>
        <th>Prompt 2</th>
        <th>Flipped Image</th>
    </tr>
    <tr>
        <td>an oil painting of an old man</td>
        <td><img src="assets/a/1/8/campfire_man.png"></td>
        <td>an oil painting of people around a campfire</td>
        <td><img src="assets/a/1/8/campfire_man.png" class="rotate180"></td>
    </tr>
    <tr>
        <td>an oil painting of a snowy mountain village</td>
        <td><img src="assets/a/1/8/mountain_coast.png"></td>
        <td>a photo of the amalfi cost*</td>
        <td><img src="assets/a/1/8/mountain_coast.png" class="rotate180"></td>
    </tr>
    <tr>
        <td>a lithograph of waterfalls</td>
        <td><img src="assets/a/1/8/waterfall_skull.png"></td>
        <td>a lithograph of a skull</td>
        <td><img src="assets/a/1/8/waterfall_skull.png" class="rotate180"></td>
    </tr>
</table>

\*This should say "coast" but that's how the text embeddings were loaded in the starter code provided by staff :)

#### 1.9 Hybrid Images

In this part, I implemented this [paper](https://arxiv.org/abs/2404.11615), Factorized Diffusion: Perceptual Illusions by Noise Decomposition by Daniel Geng, Inbum Park, and Andrew Owens. Specifically, I used the diffusion model to create hybrid images like in [Project 2](/proj2/#part-22-hybrid-images).

Similar to the visual anagrams, we just have to adjust our noise estimate as follows:

$$
\epsilon_1 = \text{UNet}(x_t, t, p_1)
$$

$$
\epsilon_2 = \text{UNet}(x_t, t, p_2)
$$

$$
\epsilon = f_{\text{lowpass}}(\epsilon_1) + f_{\text{highpass}}(\epsilon_2)
$$

where $$f_{\text{lowpass}}$$ is a low pass filter and $$f_{\text{highpass}}$$ is a high pass filter. I used the suggested Gaussian blur of kernel size 33 and $$\sigma = 2$$.

| Prompt 1 | Prompt 2 | Hybrid Image |
| :--- | :--- | :--- |
| a lithograph of waterfalls | a lithograph of a skull | ![man in hat waterfall](assets/a/1/9/waterfall_skull.png) |
| an oil painting of people around a campfire | a lithograph of a skull | ![campfire skull](assets/a/1/9/campfire_skull.png) |
| a man wearing a hat | a lithograph of waterfalls | ![man in hat waterfall](assets/a/1/9/man_in_hat_waterfall.png) |

## Project 5B: Diffusion Models from Scratch!

[Part B Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj5/partb.html)

In this part of the project, I wrote a stable diffusion model from scratch trained on the [MNIST dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

### Part 1: Training a Single-Step Denoising UNet

#### 1.1: Implementing the UNet

Given a noisy image $$z$$, I trained a denoiser $$D_\theta$$ that maps $$z$$ to a clean image $$x$$ by optimizing L2 loss:

$$
L = \mathbb{E}_{z, x}||D_\theta(z) - x ||^2
$$

I did this by implementing a UNet, from this [paper](https://arxiv.org/abs/1505.04597): U-Net: Convolutional Networks for Biomedical Image Segmentation by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

#### 1.2 Using the UNet to Train a Denoiser

To begin, I visualized how an image can become more and more noisy by implementing a function `add_noise` which takes a clean image `x` and adds Gaussian noise:

$$
z = x + \sigma \epsilon
$$

where $$\sigma = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]$$, $$\epsilon \sim \mathcal{N}(0, I)$$, and normalized $$x \in [0, 1)$$.

<div style="text-align: center">
<img src="assets/b/1/2/digits_with_noise.png" alt="varying levels of noise with noise">
</div>

##### 1.2.1: Training

Next, I began training my unconditioned UNet to be able to denoise an image in 1 step by training on pairs of clean and noisy images where $$\sigma = 0.5$$. I used the following hyperparameters, as suggested on the spec:

- Batch size: 256
- Epochs: 5
- Hidden dimension `D`: 128
- Optimizer: [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- Learning rate: `1e-4`

Here is my training loss graph:

<div style="text-align: center">
<img src="assets/b/1/2/training_loss.png" alt="unconditioned unet training loss graph">
</div>

Here are the results of the denoiser at epoch 1 and epoch 5:

<div style="text-align: center">
<img src="assets/b/1/2/epoch1.png" alt="results after epoch 1">
</div>

<div style="text-align: center">
<img src="assets/b/1/2/epoch5.png" alt="results after epoch 5">
</div>

##### 1.2.2: Out-of-Distribution Testing

To see if the denoiser can generalize to other levels of noise, I kept the image the same and varied $$\sigma$$. The denoised images can be seen below, and are not bad but also not great (this is to be fixed later in the project).

<div style="text-align: center">
<img src="assets/b/1/2/testing.png" alt="out of distribution testing results">
</div>

### Part 2: Training a DDPM Denoising U-Net

In this part, I implemented DDPM from this [paper](https://arxiv.org/abs/2006.11239), Denoising Diffusion Probabilistic Models by Jonathan Ho, Ajay Jain, and Pieter Abbeel.

Instead of estimating the denoised image, we can train a UNet to estimate the noise that was added to the image. This changes the loss function we're trying to minimize accordingly:

$$
L = \mathbb{E}_{z, x}||\epsilon_\theta(z) - \epsilon||^2
$$

where $$\epsilon_\theta$$ is a UNet trained to predict noise. Additionally, to produce better results we do iterative denoising instead of one-step denoising. Like in Part A of the project, we use the following formula to compute iteratively noisier images in the forward pass:

$$
x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \epsilon
$$

where:

- $$x_t$$ is the image at time step $$t$$ ($$x_0$$ is the original image)
- $$\bar{\alpha_t}$$ is a noise coefficient computed accordingly:
    - $$\beta_t$$ is a list of numbers of length 300 (`t = 299` to `t = 0`) such that $$\beta_0 = 0.0001$$ and $$\beta_T = 0.02$$ and all other elements are evenly spaced between the two
    - $$\alpha_t = 1 - \beta_t$$
    - $$\bar{\alpha_t} = \Pi_{s = 1}^{t} \alpha_s$$
- $$\epsilon \sim \mathcal{N}(0, 1)$$ is Gaussian noise

#### 2.1: Adding Time Conditioning to UNet

We can add time conditioning to the UNet by modifying the loss function again:

$$
L = \mathbb{E}_{z, x}||\epsilon_\theta(x_t, t) - \epsilon||^2
$$

and adding an `FCBlock` to the unconditioned UNet made up of some `nn.Linear` and `nn.GELU` elements (see the project spec for more details).

#### 2.2: Training the UNet

I implemented this algorithm:

![algorithm 1](assets/b/2/2/algorithm1.png)

to perform the forward pass of the UNet. I trained the network with the following hyperparameters, as suggested:

- Batch size: 128
- Epochs: 20
- Hidden dimension `D`: 64
- Optimizer: [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- Learning rate: `1e-3`
- Learning rate decay scheduler: [ExponentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html) with `gamma = 0.1 ** (1.0 / num_epochs)`

Below is my training loss curve:

<div style="text-align: center">
<img src="assets/b/2/2/training_loss.png" alt="training loss curve">
</div>

#### 2.3: Sampling from the UNet

I then implemented this algorithm:

![algorithm 2](assets/b/2/3/algorithm2.png)

to generate/sample 10 digits from the UNet at epoch = 5:

<div style="text-align: center">
<img src="assets/b/2/3/epoch5.png" alt="epoch 5 sampling results">
</div>

and at epoch 20:

<div style="text-align: center">
<img src="assets/b/2/3/epoch20.png" alt="epoch 20 sampling results">
</div>

One thing I had to be very careful about was setting the seed to generate the starting $$x_t$$ (I used the current epoch index) and setting the seed to generate $$z$$ (I used the current $$t$$ value).

#### 2.4: Adding Class-Conditioning to UNet

To be able to generate a particular digit (rather than any digit), I implemented class conditioning by adding 2 more `FCBlock`s in a similar way as before (see the spec for details). Additionally, I dropped out (e.g. turned that image's one-hot-encoded vector to all zeros in the batch) the class label for 10% of images in the batch. I also used Classifier-Free Guidance to estimate noise from both the unconditioned and conditioned UNet.

I implemented this algorithm:

![algorithm 3](assets/b/2/4/algorithm3.png)

And had this training loss curve with the same hyperparameters as before:

<div style="text-align: center">
<img src="assets/b/2/4/training_loss.png" alt="training loss curve">
</div>

#### 2.5: Sampling from the Class-Conditioned UNet

I then implemented this algorithm:

![algorithm 4](assets/b/2/5/algorithm4.png)

to generate/sample digits 0-9 in that order at epoch 5:

<div style="text-align: center">
<img src="assets/b/2/5/epoch5.png" alt="epoch 5 sampling results">
</div>

and at epoch 20:

<div style="text-align: center">
<img src="assets/b/2/5/epoch20.png" alt="epoch 20 sampling results">
</div>

Once again, I had to be very careful about was setting the seed to generate the starting $$x_t$$ (I used `epoch + j` where `epoch` is the epoch index and `j` is an index from `[0, 3]` since we have to sample digits 0-9 at least 4 times) and setting the seed to generate $$z$$ (I used the current $$t$$ value).

### Part 3: Bells & Whistles

#### Sampling Gifs

I created gifs of the sampling process from `t = 299` to `t = 0` for the time conditioned UNet at epoch 5 (first row) and epoch 20 (second row):

<table>
    <tr>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame0.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame1.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame2.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame3.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame4.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame5.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame6.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame7.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame8.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch4_gifs/time_cond_epoch4_frame9.gif"></td>
    </tr>
    <tr>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame0.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame1.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame2.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame3.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame4.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame5.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame6.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame7.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame8.gif"></td>
        <td><img src="assets/b/bw/time_cond_epoch19_gifs/time_cond_epoch19_frame9.gif"></td>
    </tr>
</table>

I did the same for the class conditioned UNet:

TODO
