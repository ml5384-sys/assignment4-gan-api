import io
import os
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def get_data_loader(data_dir="data", batch_size=32, train=True):
    """Create a DataLoader for MNIST, with pixels normalized to [-1, 1]. (Module 4/6)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # [0, 1] -> [-1, 1]
    ])
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


# ---------------- Assignment 4: CIFAR-10 ----------------

_HF_URL = ("https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
           "plain_text/{split}-00000-of-00001.parquet")


def _load_cifar10_from_hf(data_dir, train):
    """Fallback when the official CIFAR-10 host is unreachable: the same images from Hugging Face."""
    import pandas as pd
    from PIL import Image

    split = "train" if train else "test"
    cache = os.path.join(data_dir, f"cifar10_{split}.npz")
    if not os.path.exists(cache):
        os.makedirs(data_dir, exist_ok=True)
        parquet = os.path.join(data_dir, f"cifar10_{split}.parquet")
        if not os.path.exists(parquet):
            urllib.request.urlretrieve(_HF_URL.format(split=split), parquet)
        df = pd.read_parquet(parquet)
        images = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
        np.savez(cache, images=images, labels=df["label"].to_numpy())
    arr = np.load(cache)
    return arr["images"], arr["labels"]


def get_cifar10_loader(data_dir="data", batch_size=128, train=True):
    """CIFAR-10 DataLoader with images as (3, 32, 32) tensors normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),   # [0, 1] -> [-1, 1]
    ])
    try:
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    except Exception as err:   # e.g. no network access to www.cs.toronto.edu
        print(f"torchvision download failed ({err}); using the Hugging Face copy of CIFAR-10")
        images, labels = _load_cifar10_from_hf(data_dir, train)
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 127.5 - 1.0   # [-1, 1]
        dataset = TensorDataset(x, torch.from_numpy(labels).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train)
