import time
import copy
from typing import Dict, Any

import torch
from torch.optim import Adam, SGD, AdamW

from src.datasets import get_dataset_loaders
from src.model import TunableCNN
from src.train import train_one_epoch, evaluate
from src.utils import count_parameters, set_torch_seed


def evaluate_config(
    config: Dict[str, Any],
    dataset_name: str,
    epochs: int,
    device: str,
    seed: int = 7777,
    val_split: float = 0.1,
    num_workers: int = 2,
) -> Dict[str, Any]:
    set_torch_seed(seed)
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
        use_batch_norm=int(config.get("use_batch_norm", 1)),
        dense_units=config["dense_units"],
    ).to(device)

    # respect optimizer and weight_decay from config (if present)
    opt_name = config.get("optimizer", "adam")
    weight_decay = float(config.get("weight_decay", 0.0))

    if opt_name == "sgd":
        optimizer = SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_state = None

    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # If we saved the best model during validation, load it for final test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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