"""
Train the models served by the API and save their weights in checkpoints/.

    python train.py gan        --epochs 5     # MNIST GAN (Module 6)
    python train.py diffusion  --epochs 20    # CIFAR-10 DDPM
    python train.py ebm        --epochs 5     # CIFAR-10 energy-based model
"""
import argparse
import os

import torch
import torch.nn as nn

from helper_lib.data_loader import get_cifar10_loader, get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion, train_ebm, train_gan
from helper_lib.utils import save_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["gan", "diffusion", "ebm"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (quick runs)")
    p.add_argument("--out-dir", default="checkpoints")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.model == "gan":
        loader = get_data_loader("data", batch_size=args.batch_size, train=True)
        model = get_model("GAN")
        opt = {
            "generator": torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
        }
        train_gan(model, loader, nn.BCEWithLogitsLoss(), opt, device=device, epochs=args.epochs)
        save_model(model, os.path.join(args.out_dir, "gan_mnist.pth"))

    elif args.model == "diffusion":
        loader = get_cifar10_loader("data", batch_size=args.batch_size, train=True)
        model = get_model("Diffusion")
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)
        train_diffusion(model, loader, opt, device=device, epochs=args.epochs, max_batches=args.max_batches)
        save_model(model, os.path.join(args.out_dir, "diffusion_cifar10.pth"))

    else:
        loader = get_cifar10_loader("data", batch_size=args.batch_size, train=True)
        model = get_model("EBM")
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
        train_ebm(model, loader, opt, device=device, epochs=args.epochs, max_batches=args.max_batches)
        save_model(model, os.path.join(args.out_dir, "ebm_cifar10.pth"))


if __name__ == "__main__":
    main()
