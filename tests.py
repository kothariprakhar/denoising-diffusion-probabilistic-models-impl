import unittest
import torch
import torch.nn as nn
import numpy as np

# Assuming the reviewed code structure is available in the namespace or imported.
# If running standalone, one would paste the classes here or import them.
# For this test suite, we assume classes (Config, DiffusionUtils, SimpleUNet, get_loss, sample) are defined.

class TestDDPM(unittest.TestCase):
    def setUp(self):
        self.device = "cpu" # Use CPU for testing to avoid CUDA requirement in CI
        self.config = Config()
        self.config.device = self.device
        self.config.timesteps = 20 # Lower timesteps for faster testing
        self.img_size = 28
        self.channels = 1

    def test_diffusion_utils_shapes(self):
        """Verify shapes of pre-calculated diffusion variance schedules."""
        diff = DiffusionUtils(self.config.timesteps, self.device)
        self.assertEqual(diff.betas.shape[0], self.config.timesteps)
        self.assertEqual(diff.alphas_cumprod.shape[0], self.config.timesteps)
        self.assertEqual(diff.sqrt_alphas_cumprod.shape[0], self.config.timesteps)

    def test_q_sample_shape(self):
        """Test forward diffusion process output shapes."""
        diff = DiffusionUtils(self.config.timesteps, self.device)
        x_0 = torch.randn(4, 1, 28, 28)
        t = torch.randint(0, self.config.timesteps, (4,))
        x_t = diff.q_sample(x_0, t)
        self.assertEqual(x_t.shape, x_0.shape)

    def test_model_forward(self):
        """Test U-Net forward pass for shape consistency."""
        model = SimpleUNet(img_channels=1, time_emb_dim=32).to(self.device)
        x = torch.randn(2, 1, 28, 28).to(self.device)
        t = torch.randint(0, self.config.timesteps, (2,)).to(self.device)
        output = model(x, t)
        self.assertEqual(output.shape, (2, 1, 28, 28))

    def test_loss_calculation_and_backward(self):
        """Test if loss is scalar and gradients can be computed."""
        model = SimpleUNet(img_channels=1).to(self.device)
        diff = DiffusionUtils(self.config.timesteps, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x_0 = torch.randn(2, 1, 28, 28).to(self.device)
        t = torch.randint(0, self.config.timesteps, (2,)).to(self.device)
        
        loss = get_loss(model, x_0, t, diff)
        
        # Check loss is scalar
        self.assertEqual(loss.ndim, 0)
        
        # Check backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if weights updated (rudimentary check or just ensure no crash)
        for param in model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())
                break

    def test_sampling_integration(self):
        """Test the sampling loop to ensure it produces images in correct range."""
        model = SimpleUNet(img_channels=1).to(self.device)
        model.eval()
        diff = DiffusionUtils(timesteps=5, device=self.device) # Very short diffusion for test
        
        with torch.no_grad():
            # Run sampling
            samples = sample(model, diff, image_size=16, batch_size=2, channels=1)
        
        self.assertEqual(samples.shape, (2, 1, 16, 16))
        # Check normalization range [0, 1]
        self.assertTrue(samples.max() <= 1.0 + 1e-5)
        self.assertTrue(samples.min() >= 0.0 - 1e-5)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
