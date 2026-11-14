# Assignment 4 – Diffusion & Energy-Based Generators API

This builds on the Module 6 class activity (a `helper_lib` package with a GAN) by adding two CIFAR-10 image generators and serving all three through FastAPI in Docker:

| Model | Dataset | Endpoint |
|---|---|---|
| GAN (Module 6) | MNIST | `GET /generate/gan` |
| Diffusion model (DDPM with a UNet and a sinusoidal time embedding) | CIFAR-10 | `GET /generate/diffusion` |
| Energy-based model (CNN energy, Langevin dynamics sampling) | CIFAR-10 | `GET /generate/ebm` |

The trained weights are committed in `checkpoints/`, so the server works without any training.
The theory answers (Parts 2 and 3) are in [`ANSWERS.md`](ANSWERS.md).

## Project structure

```
helper_lib/
  data_loader.py   get_data_loader (MNIST), get_cifar10_loader (CIFAR-10, [-1, 1])
  model.py         FCNN, CNN, EnhancedCNN, GAN  +  SinusoidalEmbedding, UNet, DiffusionModel, EnergyModel
                   get_model("GAN" | "Diffusion" | "EBM" | ...)
  trainer.py       train_model, train_gan  +  train_diffusion, train_ebm (contrastive divergence + replay buffer)
  generator.py     generate_samples/sample_gan  +  sample_diffusion (DDPM/DDIM), langevin_sample, sample_ebm
  evaluator.py, utils.py
app/main.py        FastAPI server
train.py           trains a model and writes checkpoints/*.pth
checkpoints/       gan_mnist.pth, diffusion_cifar10.pth, ebm_cifar10.pth
Dockerfile
```

## Run with Docker

```bash
docker build -t assignment4-gan-api .
```

```bash
docker run --rm -p 8000:8000 assignment4-gan-api
```

Then open http://localhost:8000/docs for the interactive Swagger UI, or:

```bash
curl "http://localhost:8000/generate/diffusion?num_samples=16&steps=50" -o diffusion.png
```

```bash
curl "http://localhost:8000/generate/ebm?num_samples=16&steps=256" -o ebm.png
```

```bash
curl "http://localhost:8000/generate/gan?num_samples=16" -o gan.png
```

## Endpoints

| Endpoint | Parameters | Returns |
|---|---|---|
| `GET /` | – | list of endpoints |
| `GET /health` | – | `{"status": "ok"}` |
| `GET /models` | – | which checkpoints were loaded |
| `GET /generate/gan` | `num_samples` (1–64), `format` (`png`/`json`) | grid of MNIST digits |
| `GET /generate/diffusion` | `num_samples`, `steps` (2–1000; `<1000` = DDIM, `1000` = full DDPM), `seed`, `format` | grid of CIFAR-10-like images |
| `GET /generate/ebm` | `num_samples`, `steps` (Langevin steps), `step_size`, `noise`, `seed`, `format` | grid of images + their energies (JSON) |

`format=png` (default) returns an `image/png` grid; `format=json` returns each image as base64 PNG.

## Run without Docker

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.main:app --reload
```

## Training

```bash
python train.py gan --epochs 5
```

```bash
python train.py diffusion --epochs 20
```

```bash
python train.py ebm --epochs 5
```

`get_cifar10_loader` downloads CIFAR-10 with torchvision; if that host is unreachable it falls back to the Hugging Face copy (`uoft-cs/cifar10`).

## Model details

**Diffusion (DDPM).** Linear β schedule from 1e-4 to 0.02 over T = 1000 steps. The forward process is $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$. The UNet works at 32×32 → 16×16 → 8×8 (bottleneck) and back, with skip connections and residual blocks. A sinusoidal embedding of `t` goes through an MLP into every block. The UNet predicts ε, and the loss is `MSE(ε_pred, ε)`. Sampling supports full DDPM ancestral sampling and fast deterministic DDIM.

**Energy-based model.** A CNN with Swish activations maps an image to a scalar energy $E(x)$. Training uses contrastive divergence, `E(real).mean() - E(fake).mean() + α·(E(real)² + E(fake)²).mean()`. The negative samples come from Langevin dynamics, which runs gradient descent **on the input image**: `x ← x − η·∂E/∂x + noise`. During sampling the network weights are frozen (`requires_grad_(False)`) and the image is marked `requires_grad_(True)`, so `torch.autograd.grad(E.sum(), x)` returns ∂E/∂x only. A replay buffer of past samples keeps the Markov chains running across iterations.
