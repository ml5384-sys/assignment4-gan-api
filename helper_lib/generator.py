import math

import torch


def _show_grid(imgs, num_samples):
    import matplotlib.pyplot as plt
    import numpy as np

    ncols = min(num_samples, 5)
    nrows = math.ceil(num_samples / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < num_samples:
            img = imgs[i].permute(1, 2, 0).numpy()
            ax.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
    plt.tight_layout()
    plt.show()


# ---------------- Module 6: GAN ----------------

@torch.no_grad()
def sample_gan(model, device="cpu", num_samples=10):
    """Return GAN samples as a (N, 1, 28, 28) tensor in [0, 1]."""
    model.to(device).eval()
    z = torch.randn(num_samples, model.z_dim, device=device)
    return (model.generator(z).cpu() * 0.5 + 0.5).clamp(0, 1)


def generate_samples(model, device="cpu", num_samples=10):
    """Sample random latent vectors, generate images, and plot them on a grid."""
    _show_grid(sample_gan(model, device, num_samples), num_samples)


# ---------------- Assignment 4: Diffusion ----------------

@torch.no_grad()
def sample_diffusion(model, device="cpu", num_samples=10, steps=None, seed=None):
    """
    Reverse diffusion starting from pure Gaussian noise x_T.
    steps=None (or >= T) -> full DDPM ancestral sampling over all T steps.
    steps < T            -> deterministic DDIM sampling on an evenly spaced subset of timesteps
                            (same trained network, much faster - handy for the API).
    Returns a (N, 3, 32, 32) tensor in [0, 1].
    """
    model.to(device).eval()
    g = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    shape = (num_samples, model.channels, model.image_size, model.image_size)
    x = torch.randn(shape, generator=g).to(device)
    T = model.timesteps

    if steps is None or steps >= T:   # DDPM
        for t in reversed(range(T)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)
            alpha, ab, beta = model.alphas[t], model.alpha_bars[t], model.betas[t]
            mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if t > 0:
                noise = torch.randn(shape, generator=g).to(device)
                x = mean + beta.sqrt() * noise
            else:
                x = mean
    else:                              # DDIM (eta = 0)
        ts = torch.linspace(T - 1, 0, steps).long().tolist()
        for i, t in enumerate(ts):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)
            ab = model.alpha_bars[t]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            ab_prev = model.alpha_bars[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return (x.clamp(-1, 1).cpu() + 1) / 2


def generate_diffusion_samples(model, device="cpu", num_samples=10, steps=None):
    _show_grid(sample_diffusion(model, device, num_samples, steps), num_samples)


# ---------------- Assignment 4: Energy-based model ----------------

def langevin_sample(model, x, steps=60, step_size=10.0, noise_std=0.005, grad_clip=0.03):
    """
    Langevin dynamics: move the *input images* downhill on the energy surface.
        x <- x - step_size * dE/dx + noise
    Here the gradient is taken with respect to x, NOT the network weights, so:
      * the weights are frozen (requires_grad=False) so no gradient is accumulated for them,
      * x is detached from any previous graph and marked requires_grad=True,
      * torch.autograd.grad(E.sum(), x) gives dE/dx for every image in the batch.
    """
    was_training = model.training
    model.eval()
    param_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    x = x.detach().clone().requires_grad_(True)
    for _ in range(steps):
        with torch.no_grad():                               # 1) add noise (not part of the graph)
            x.add_(torch.randn_like(x) * noise_std).clamp_(-1, 1)
        energy = model(x)                                   # 2) E(x), graph built w.r.t. x only
        grad_x, = torch.autograd.grad(energy.sum(), x)      # 3) dE/dx
        with torch.no_grad():                               # 4) gradient *descent* on the input
            x.add_(-step_size * grad_x.clamp(-grad_clip, grad_clip)).clamp_(-1, 1)

    for p, flag in zip(model.parameters(), param_flags):   # restore the weights' flags
        p.requires_grad_(flag)
    model.train(was_training)
    return x.detach()


def sample_ebm(model, device="cpu", num_samples=10, steps=256, step_size=10.0, noise_std=0.005, seed=None):
    """Start from uniform noise in [-1, 1] and run Langevin dynamics. Returns (N, 3, 32, 32) in [0, 1]."""
    model.to(device)
    g = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    x = (torch.rand(num_samples, 3, 32, 32, generator=g) * 2 - 1).to(device)
    x = langevin_sample(model, x, steps=steps, step_size=step_size, noise_std=noise_std)
    return (x.cpu() + 1) / 2


def generate_ebm_samples(model, device="cpu", num_samples=10, steps=256):
    _show_grid(sample_ebm(model, device, num_samples, steps), num_samples)
