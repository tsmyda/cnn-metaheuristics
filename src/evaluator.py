import time
from typing import Dict, Any

import torch
from torch.optim import Adam

from src.datasets import get_dataset_loaders
from src.model import TunableCNN
from src.train import train_one_epoch, evaluate
from src.utils import count_parameters, set_seed


def evaluate_config(
    config: Dict[str, Any],
    dataset_name: str,
    epochs: int,
    device: str,
    seed: int = 42,
    val_split: float = 0.1,
    num_workers: int = 2,
) -> Dict[str, Any]:
    set_seed(seed)
    start_time = time.time()

    train_loader, val_loader, test_loader, image_channels, image_size, num_classes = get_dataset_loaders(
        dataset_name=dataset_name,
        batch_size=config["batch_size"],
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
    )

    model = TunableCNN(
        image_channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        num_blocks=config["num_blocks"],
        filters_1=config["filters_1"],
        filters_2=config["filters_2"],
        filters_3=config["filters_3"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        dense_units=config["dense_units"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    best_val_acc = 0.0
    best_val_loss = float("inf")

    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    test_loss, test_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time
    num_params = count_parameters(model)

    return {
        "val_accuracy": best_val_acc,
        "val_loss": best_val_loss,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "time_sec": elapsed,
        "num_params": num_params,
    }