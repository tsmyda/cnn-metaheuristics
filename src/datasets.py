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
    dataset_name = dataset_name.lower()

    if dataset_name == "fashionmnist":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_full = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=train_transform,
        )
        test_set = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=test_transform,
        )

        image_channels = 1
        image_size = 28
        num_classes = 10

    elif dataset_name == "cifar10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_full = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=train_transform,
        )
        test_set = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=test_transform,
        )

        image_channels = 3
        image_size = 32
        num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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

    return train_loader, val_loader, test_loader, image_channels, image_size, num_classes