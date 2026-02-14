!pip install matplotlib torch torchvision tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
class Config:
    dataset_name = "FashionMNIST"  # Using FashionMNIST as a proxy for speed/viz
    img_size = 28
    channels = 1
    timesteps = 1000
    batch_size = 128
    lr = 2e-4
    epochs = 5  # Kept low for demonstration purposes. Increase for better results.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./ddpm_results"

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
print(f"Using device: {config.device}")

# ==========================================
# 2. Dataset Loading
# ==========================================
def get_dataloader(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ])
    
    if config.dataset_name == "FashionMNIST":
        dataset = datasets.FashionMNIST(
            root="./data", 
            train=True, 
            download=True, 
            transform=transform
        )
    else:
        raise ValueError("Dataset not supported for this demo")
        
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

# ==========================================
# 3. Diffusion Logic (The Math)
# ==========================================
class DiffusionUtils:
    def __init__(self, timesteps, device):
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule (linear)
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, timesteps).to(device)
        
        # Pre-calculate alpha terms
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: x_0 -> x_t"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps t and reshape to [batch, 1, 1, 1]"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu() if a.device.type == 'cpu' else t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# ==========================================
# 4. Neural Network (U-Net with Time Embedding)
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time Embedding Injection
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    def __init__(self, img_channels=1, down_channels=[64, 128, 256], time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(img_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels)-1)
        ])
        
        # Upsample
        self.ups = nn.ModuleList([
            Block(down_channels[i+1], down_channels[i], time_emb_dim, up=True)
            for i in range(len(down_channels)-1, 0, -1)
        ])
        
        self.output = nn.Conv2d(down_channels[0], img_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residuals = []
        
        # Down
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
            
        # Up
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
            
        return self.output(x)

# ==========================================
# 5. Training Logic
# ==========================================
def get_loss(model, x_0, t, diffusion_utils):
    noise = torch.randn_like(x_0)
    x_t = diffusion_utils.q_sample(x_0, t, noise)
    predicted_noise = model(x_t, t)
    return F.mse_loss(noise, predicted_noise)

@torch.no_grad()
def sample(model, diffusion_utils, image_size, batch_size=16, channels=3):
    img = torch.randn((batch_size, channels, image_size, image_size), device=diffusion_utils.device)
    
    for i in tqdm(reversed(range(0, diffusion_utils.timesteps)), desc='Sampling', total=diffusion_utils.timesteps):
        t = torch.full((batch_size,), i, device=diffusion_utils.device, dtype=torch.long)
        predicted_noise = model(img, t)
        
        # Algorithm 2 Line 4:
        alpha = diffusion_utils.alphas[t][:, None, None, None]
        alpha_hat = diffusion_utils.alphas_cumprod[t][:, None, None, None]
        beta = diffusion_utils.betas[t][:, None, None, None]
        
        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)
            
        img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
    img = (img.clamp(-1, 1) + 1) / 2 # Normalize to [0, 1]
    return img

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    # Setup
    dataloader = get_dataloader(config)
    model = SimpleUNet(img_channels=config.channels).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    diffusion = DiffusionUtils(config.timesteps, config.device)
    
    losses = []
    
    # Training
    print("Starting Training...")
    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(dataloader)
        epoch_loss = 0
        for step, (images, _) in enumerate(pbar):
            images = images.to(config.device)
            t = torch.randint(0, config.timesteps, (images.shape[0],), device=config.device).long()
            
            loss = get_loss(model, images, t, diffusion)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{config.epochs} | Average Loss: {avg_loss:.4f}")

    # Visualization of Loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('DDPM Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Sampling
    print("Sampling new images...")
    model.eval()
    samples = sample(model, diffusion, config.img_size, batch_size=32, channels=config.channels)
    
    # Visualization of Samples
    grid_img = make_grid(samples, nrow=8).cpu().permute(1, 2, 0)
    plt.figure(figsize=(12, 6))
    plt.imshow(grid_img)
    plt.title("Generated Fashion-MNIST Samples")
    plt.axis('off')
    plt.show()
