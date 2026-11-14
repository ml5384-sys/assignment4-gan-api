import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    """Fully connected network for 28x28 grayscale images."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Simple CNN: two conv blocks followed by a classifier."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EnhancedCNN(nn.Module):
    """CNN with BatchNorm and Dropout for better generalization."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- Module 6: GAN (MNIST) ----------------

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),     # 1 x 28 x 28
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),   # raw logit
        )

    def forward(self, x):
        return self.net(x)


class GAN(nn.Module):
    """Wrapper that holds both the generator and the discriminator."""
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()


# ---------------- Assignment 4: Diffusion model (CIFAR-10) ----------------

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (Theory Q1):
        freq_i   = exp(-ln(max_period) * i / (dim/2)),  i = 0 .. dim/2 - 1
        emb(t)   = [sin(t * freq_0), ..., sin(t * freq_{dim/2-1}),
                    cos(t * freq_0), ..., cos(t * freq_{dim/2-1})]
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    """Residual conv block; the time embedding is added after the first convolution."""
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """
    Small UNet that predicts the noise eps added to x_t.
    32x32 -> (stride-2 conv) 16x16 -> (stride-2 conv) 8x8 bottleneck -> upsample back to 32x32,
    with skip connections between matching resolutions.
    """
    def __init__(self, in_ch=3, base=64, t_dim=128):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 2
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(t_dim), nn.Linear(t_dim, t_dim * 2), nn.SiLU(), nn.Linear(t_dim * 2, t_dim),
        )
        self.init_conv = nn.Conv2d(in_ch, c1, 3, padding=1)
        # encoder
        self.down1 = ResBlock(c1, c1, t_dim)                       # 32x32
        self.pool1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)     # -> 16x16
        self.down2 = ResBlock(c1, c2, t_dim)                       # 16x16
        self.pool2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)     # -> 8x8
        # bottleneck
        self.mid1 = ResBlock(c2, c3, t_dim)
        self.mid2 = ResBlock(c3, c3, t_dim)
        # decoder
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)   # -> 16x16
        self.dec2 = ResBlock(c2 * 2, c2, t_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)   # -> 32x32
        self.dec1 = ResBlock(c1 * 2, c1, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, in_ch, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        s1 = self.down1(x, t_emb)
        s2 = self.down2(self.pool1(s1), t_emb)
        h = self.mid2(self.mid1(self.pool2(s2), t_emb), t_emb)
        h = self.dec2(torch.cat([self.up2(h), s2], dim=1), t_emb)
        h = self.dec1(torch.cat([self.up1(h), s1], dim=1), t_emb)
        return self.out(h)


class DiffusionModel(nn.Module):
    """DDPM: linear beta schedule + a UNet noise predictor."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, image_size=32, channels=3, base=64):
        super().__init__()
        self.timesteps = timesteps
        self.image_size = image_size
        self.channels = channels
        self.unet = UNet(in_ch=channels, base=base)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps."""
        ab = self.alpha_bars[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def forward(self, x_t, t):
        return self.unet(x_t, t)   # predicted noise


# ---------------- Assignment 4: Energy-based model (CIFAR-10) ----------------

class EnergyModel(nn.Module):
    """
    CNN that maps an image x to a scalar energy E(x).
    Low energy = looks like real data, high energy = unlikely.
    Swish activations keep the energy surface smooth, which Langevin sampling needs.
    """
    def __init__(self, in_ch=3, hidden=32):
        super().__init__()
        c = hidden
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 5, stride=2, padding=2), nn.SiLU(),       # 16x16
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.SiLU(),       # 8x8
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.SiLU(),   # 4x4
            nn.Conv2d(c * 4, c * 4, 3, stride=2, padding=1), nn.SiLU(),   # 2x2
            nn.Flatten(),
            nn.Linear(c * 4 * 4, c * 4), nn.SiLU(),
            nn.Linear(c * 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # energy, shape (B,)


def get_model(model_name):
    """Return a model by name: one of FCNN, CNN, EnhancedCNN, GAN, Diffusion, EBM."""
    if model_name == "FCNN":
        model = FCNN()
    elif model_name == "CNN":
        model = CNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    elif model_name == "GAN":
        model = GAN(z_dim=100)
    elif model_name == "Diffusion":
        model = DiffusionModel()
    elif model_name == "EBM":
        model = EnergyModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model
