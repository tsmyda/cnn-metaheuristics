from typing import Dict, Any
from pathlib import Path

import torch
from torch.optim import Adam

from src.datasets import get_dataset_loaders
from src.model import TunableCNN
from src.train import train_one_epoch, evaluate
from src.utils import count_parameters, set_torch_seed


def retrain_best_config(
    config: Dict[str, Any],
    dataset_name: str,
    epochs: int,
    device: str,
    seed: int = 7777,
    save_path: str = "results/best_model.pt",
) -> Dict[str, Any]:
    set_torch_seed(seed)

    train_loader, val_loader, test_loader, image_channels, image_size, num_classes = get_dataset_loaders(
        dataset_name=dataset_name,
        batch_size=config["batch_size"],
        val_split=0.1,
        num_workers=2,
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

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(
            f"[RETRAIN] epoch={epoch:02d}/{epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"best_val_acc={best_val_acc:.4f}"
        )

    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device)

    return {
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "num_params": count_parameters(model),
        "history": history,
    }