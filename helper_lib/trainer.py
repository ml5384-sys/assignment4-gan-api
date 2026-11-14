import random

import torch
import torch.nn.functional as F

from helper_lib.generator import langevin_sample


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Standard supervised training loop for classifiers."""
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  loss={running_loss/len(data_loader):.4f}")
    return model


def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """
    Train a GAN. `optimizer` is a dict {"generator": opt_G, "discriminator": opt_D}.
    """
    G, D = model.generator, model.discriminator
    opt_G, opt_D = optimizer["generator"], optimizer["discriminator"]
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss_D, total_loss_G = 0.0, 0.0
        for real, _ in data_loader:
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            z = torch.randn(b, model.z_dim, device=device)
            fake = G(z)
            loss_D = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            loss_G = criterion(D(fake), ones)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
        n = len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}  loss_D={total_loss_D/n:.4f}  loss_G={total_loss_G/n:.4f}")
    return model


# ---------------- Assignment 4: Diffusion ----------------

def train_diffusion(model, data_loader, optimizer, device="cpu", epochs=10, max_batches=None):
    """
    DDPM training (Ho et al., 2020):
      1. sample a clean image x_0, a random timestep t ~ U{0..T-1} and noise eps ~ N(0, I)
      2. build x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps
      3. the UNet predicts eps from (x_t, t); loss = MSE(eps_pred, eps)
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for i, (x0, _) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            x0 = x0.to(device)
            if random.random() < 0.5:                 # horizontal flip augmentation
                x0 = x0.flip(-1)
            t = torch.randint(0, model.timesteps, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            x_t = model.q_sample(x0, t, noise)
            loss = F.mse_loss(model(x_t, t), noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
            if n % 100 == 0:
                print(f"  epoch {epoch+1} batch {n}  loss={total/n:.4f}", flush=True)
        print(f"Epoch {epoch+1}/{epochs}  loss={total/max(n, 1):.4f}", flush=True)
    return model


# ---------------- Assignment 4: Energy-based model ----------------

class SampleBuffer:
    """Replay buffer of past Langevin samples (persistent contrastive divergence)."""
    def __init__(self, size=8192, shape=(3, 32, 32)):
        self.size = size
        self.shape = shape
        self.examples = [torch.rand(1, *shape) * 2 - 1 for _ in range(size)]

    def sample(self, model, batch_size, device, steps=60, step_size=10.0, noise_std=0.005):
        n_new = sum(random.random() < 0.05 for _ in range(batch_size))   # 5% fresh noise
        new = torch.rand(n_new, *self.shape) * 2 - 1
        old = torch.cat(random.choices(self.examples, k=batch_size - n_new), dim=0)
        x = torch.cat([new, old], dim=0).to(device)
        x = langevin_sample(model, x, steps=steps, step_size=step_size, noise_std=noise_std)
        self.examples = list(x.cpu().chunk(batch_size, dim=0)) + self.examples
        self.examples = self.examples[: self.size]
        return x


def train_ebm(model, data_loader, optimizer, device="cpu", epochs=10, alpha=0.1,
              steps=60, step_size=10.0, noise_std=0.005, max_batches=None):
    """
    Contrastive divergence:
      loss = E(real).mean() - E(fake).mean()          # push real energy down, fake energy up
             + alpha * (E(real)^2 + E(fake)^2).mean() # keeps energies near 0 (regularizer)
    Fake samples come from Langevin dynamics on the inputs (see langevin_sample), and are
    detached, so this loss only updates the network weights.
    """
    model.to(device)
    buffer = SampleBuffer()
    for epoch in range(epochs):
        model.train()
        total_cd, n = 0.0, 0
        for i, (real, _) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            real = real.to(device)
            real = (real + torch.randn_like(real) * noise_std).clamp(-1, 1)
            fake = buffer.sample(model, real.size(0), device, steps, step_size, noise_std)

            e_real, e_fake = model(torch.cat([real, fake], dim=0)).chunk(2, dim=0)
            cd_loss = e_real.mean() - e_fake.mean()
            reg_loss = alpha * (e_real ** 2 + e_fake ** 2).mean()
            loss = cd_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_cd += cd_loss.item()
            n += 1
            if n % 50 == 0:
                print(f"  epoch {epoch+1} batch {n}  cd={total_cd/n:.4f}  "
                      f"E_real={e_real.mean().item():.3f} E_fake={e_fake.mean().item():.3f}", flush=True)
        print(f"Epoch {epoch+1}/{epochs}  cd_loss={total_cd/max(n, 1):.4f}", flush=True)
    return model
