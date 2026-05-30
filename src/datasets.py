from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataset_loaders(
    dataset_name: str,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 7777,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """Build train, validation, and test loaders for a supported dataset."""
    dataset_name = dataset_name.lower()
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == "fashionmnist":
        dataset_class = datasets.FashionMNIST
        image_channels = 1
        image_size = 28
        num_classes = 10
    elif dataset_name == "cifar10":
        dataset_class = datasets.CIFAR10
        image_channels = 3
        image_size = 32
        num_classes = 10
    elif dataset_name == "cifar100":
        dataset_class = datasets.CIFAR100
        image_channels = 3
        image_size = 32
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_full = dataset_class(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = dataset_class(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_full,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        image_channels,
        image_size,
        num_classes,
    )
