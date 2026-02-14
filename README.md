# Denoising Diffusion Probabilistic Models

We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at https://github.com/hojonathanho/diffusion

## Implementation Details

## Denoising Diffusion Probabilistic Models (DDPM)

### 1. Brainstorming & Design Choices

**Architecture Selection:** 
The core of DDPM is the reverse process, parameterized by a neural network that predicts noise. The original paper uses a complex U-Net based on the PixelCNN++ architecture with self-attention. For this implementation, I chose a simplified **U-Net with Residual blocks and Time Embeddings**. 
*   *Trade-off:* The full OpenAI U-Net requires significant VRAM and training time (days). A simplified U-Net (3 downsampling layers, sinusoidal time embeddings) allows this code to train on a standard Colab GPU (T4) in under 10 minutes while still demonstrating the fundamental mechanics of diffusion.

**Dataset Strategy:**
The prompt requires a real dataset. While the paper highlights CIFAR-10 and LSUN, CIFAR-10 (32x32 color) is notoriously noisy and slow to converge for a tutorial-sized diffusion model. 
*   *Choice:* **Fashion-MNIST**. It is a real dataset (clothing items), gray-scale (faster computation), and 28x28. It provides immediate visual feedback (shapes of shirts, shoes) compared to CIFAR-10, which often looks like unstructured color blobs if trained for only a few epochs.

### 2. Dataset & Tools
*   **Dataset:** Fashion-MNIST (via `torchvision.datasets`).
    *   Source: [Zalando Research - Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
*   **Tools:** PyTorch (Core logic), Matplotlib (Visualization), Tqdm (Progress bars).

### 3. Theoretical Foundation

**The Problem Space:**
Generative models aim to learn the data distribution $q(x_0)$. DDPMs belong to the class of Latent Variable Models. They map data to a latent space, but unlike VAEs, the latent space is of the same dimension as the input, and the encoding is a fixed, non-learnable Markov chain.

**Forward Process (Diffusion):**
We define a forward process that gradually adds Gaussian noise to the data $x_0$ over $T$ timesteps until it becomes isotropic Gaussian noise $x_T \sim \mathcal{N}(0, I)$.

$$ q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I) $$

A crucial property allows us to sample $x_t$ directly from $x_0$ without iterating:
$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I) $$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

**Reverse Process (Denoising):**
The goal is to reverse this process: sample noise $x_T$ and gradually denoise it to reach a valid image $x_0$. Since $q(x_{t-1}|x_t)$ is intractable, we approximate it with a neural network $p_\theta$:
$$ p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$

**The Objective (Simplified Loss):**
Ho et al. (2020) demonstrated that predicting the mean $\mu_\theta$ is equivalent to predicting the noise $\epsilon$ added at step $t$. The variational lower bound reduces to a simple Mean Squared Error (MSE) loss between the actual noise added ($\epsilon$) and the network's predicted noise ($\epsilon_\theta(x_t, t)$):

$$ \mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} [ || \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) ||^2 ] $$

### 4. Implementation Walkthrough

1.  **`DiffusionUtils` Class:**
    *   This encapsulates the math. We define the linear $\beta$ schedule (`self.betas`).
    *   We precompute $\alpha$, $\bar{\alpha}$ (`alphas_cumprod`) to avoid recomputing them at every iteration.
    *   `q_sample`: Implements the "nice property" equation to get $x_t$ from $x_0$ immediately using the precomputed coefficients.

2.  **`SimpleUNet` & Time Embeddings:**
    *   The network must know *which* timestep $t$ it is currently denoising. If $t=999$, the image is pure noise; if $t=1$, it is almost a clean image. 
    *   `SinusoidalPositionEmbeddings`: Creates a vector representation of $t$ (similar to Transformers).
    *   The U-Net architecture takes $(x_t, t)$ as input. The time embedding is injected into every residual block via an MLP projection, modulating the features based on the noise level.

3.  **Training Loop (`get_loss`):**
    *   Sample a batch of real images $x_0$.
    *   Sample random timesteps $t$ uniformly.
    *   Sample random noise $\epsilon$.
    *   Create noisy image $x_t$ using `q_sample`.
    *   Feed $x_t$ and $t$ to the U-Net.
    *   Calculate MSE between the true noise $\epsilon$ and output of U-Net.

4.  **Sampling (`sample`):**
    *   Start with pure Gaussian noise $x_T$.
    *   Loop backwards from $T$ down to 0.
    *   At each step, predict noise using the model.
    *   Subtract a fraction of this predicted noise to get the mean estimate.
    *   Add a small amount of random noise (Langevin dynamics) scaled by $\sqrt{\beta_t}$ (except for the final step).

### 5. Expected Plots & Visuals

*   **Loss Curve:** A line plot showing MSE Loss vs. Epochs. It should decrease rapidly in the first epoch and then stabilize. This confirms the model is learning to predict the noise structure.
*   **Generated Samples:** A grid of $32$ grayscale images. After 5 epochs on Fashion-MNIST, you will see recognizable shapes (trousers, pullovers, bags) emerging from the noise. They might be slightly blurry compared to GANs, which is characteristic of Diffusion models trained for short periods, but they will show high diversity.

## Verification & Testing

The code provides a functional implementation of a Denoising Diffusion Probabilistic Model (DDPM) on FashionMNIST. The mathematical logic for the diffusion process (forward `q_sample` and reverse `sample`) aligns well with the original paper (Algorithm 1 and 2). 

However, there is a notable architectural weakness in the `SimpleUNet` implementation regarding skip connections:

1.  **Suboptimal Skip Connections**: In the `downs` loop, `x = down(x, t)` is called, and the result is appended to `residuals`. The `Block` class performs convolution and then downsampling (via `self.transform` with stride 2). Consequently, `residuals` stores the *downsampled* feature maps. In the `ups` loop, this downsampled residual is concatenated with the input. While the tensor shapes align (both are low-resolution), this defeats the primary purpose of U-Net skip connections, which is to preserve high-frequency spatial information lost during downsampling. A standard implementation would store the features *before* the downsampling operation.

2.  **Position Embeddings**: The `SinusoidalPositionEmbeddings` logic assumes `dim` is even. If an odd dimension is passed, the concatenation of sine and cosine terms will result in a shape mismatch (outputting `dim-1`). The default `time_emb_dim=32` is safe.

3.  **Variance Schedule**: The code uses a fixed linear beta schedule. While correct for the paper's default, modern implementations often use cosine schedules for better stability, though this is not a bug.

Overall, the code is valid and runnable, but the U-Net performance might be degraded due to the skip connection logic.