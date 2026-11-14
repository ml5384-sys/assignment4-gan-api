"""
FastAPI server exposing the image generators from helper_lib:

    GET /generate/gan        MNIST digits from the Module 6 GAN
    GET /generate/diffusion  CIFAR-10 images from the DDPM diffusion model
    GET /generate/ebm        CIFAR-10 images from the energy-based model (Langevin sampling)

Each generator endpoint returns a PNG grid by default, or JSON with base64 images when
?format=json is passed.
"""
import base64
import io
import os
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from helper_lib.generator import sample_diffusion, sample_ebm, sample_gan
from helper_lib.model import get_model

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "GAN": "gan_mnist.pth",
    "Diffusion": "diffusion_cifar10.pth",
    "EBM": "ebm_cifar10.pth",
}

app = FastAPI(
    title="Generative Models API",
    description="GAN (MNIST), Diffusion and Energy-Based (CIFAR-10) image generators.",
    version="2.0",
)

MODELS = {}
LOADED = {}


def _load(name):
    model = get_model(name)
    path = os.path.join(CHECKPOINT_DIR, CHECKPOINTS[name])
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        LOADED[name] = True
    else:
        print(f"WARNING: {path} not found - {name} will use untrained weights")
        LOADED[name] = False
    return model.to(DEVICE).eval()


@app.on_event("startup")
def load_models():
    for name in CHECKPOINTS:
        MODELS[name] = _load(name)


def _respond(imgs, fmt, nrow, meta):
    grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=1.0)
    pil = to_pil_image(grid)
    if fmt == "png":
        pil = pil.resize((pil.width * 4, pil.height * 4), resample=0)   # upscale for viewing
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    def b64(img):
        buf = io.BytesIO()
        to_pil_image(img).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({**meta, "num_samples": len(imgs), "images_base64_png": [b64(i) for i in imgs]})


@app.get("/")
def root():
    return {
        "message": "Generative Models API",
        "endpoints": ["/health", "/models", "/generate/gan", "/generate/diffusion", "/generate/ebm", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/models")
def models():
    return {name: {"checkpoint": CHECKPOINTS[name], "trained_weights_loaded": LOADED.get(name, False)}
            for name in CHECKPOINTS}


@app.get("/generate/gan")
def generate_gan(
    num_samples: int = Query(16, ge=1, le=64),
    format: Literal["png", "json"] = "png",
):
    imgs = sample_gan(MODELS["GAN"], DEVICE, num_samples)
    return _respond(imgs, format, nrow=min(num_samples, 8), meta={"model": "GAN", "dataset": "MNIST"})


@app.get("/generate/diffusion")
def generate_diffusion(
    num_samples: int = Query(16, ge=1, le=64),
    steps: int = Query(50, ge=2, le=1000, description="< 1000 uses DDIM, 1000 uses full DDPM sampling"),
    seed: int | None = None,
    format: Literal["png", "json"] = "png",
):
    imgs = sample_diffusion(MODELS["Diffusion"], DEVICE, num_samples, steps=steps, seed=seed)
    sampler = "DDPM" if steps >= MODELS["Diffusion"].timesteps else "DDIM"
    return _respond(imgs, format, nrow=min(num_samples, 8),
                    meta={"model": "Diffusion", "dataset": "CIFAR-10", "sampler": sampler, "steps": steps})


@app.get("/generate/ebm")
def generate_ebm(
    num_samples: int = Query(16, ge=1, le=64),
    steps: int = Query(256, ge=1, le=2000, description="number of Langevin dynamics steps"),
    step_size: float = Query(10.0, gt=0),
    noise: float = Query(0.005, ge=0),
    seed: int | None = None,
    format: Literal["png", "json"] = "png",
):
    if steps * num_samples > 2000 * 64:
        raise HTTPException(400, "request too large")
    imgs = sample_ebm(MODELS["EBM"], DEVICE, num_samples, steps=steps, step_size=step_size,
                      noise_std=noise, seed=seed)
    with torch.no_grad():
        energies = MODELS["EBM"]((imgs * 2 - 1).to(DEVICE)).cpu().tolist()
    return _respond(imgs, format, nrow=min(num_samples, 8),
                    meta={"model": "EBM", "dataset": "CIFAR-10", "langevin_steps": steps, "energies": energies})
