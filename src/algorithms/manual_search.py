from typing import Any, Dict, Tuple

import pandas as pd

from src.evaluator import evaluate_config


MANUAL_CONFIGS = [
    {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_blocks": 2,
        "filters_1": 32,
        "filters_2": 64,
        "filters_3": 128,
        "kernel_size": 3,
        "dropout": 0.25,
        "dense_units": 128,
    },
    {
        "learning_rate": 5e-4,
        "batch_size": 128,
        "num_blocks": 2,
        "filters_1": 32,
        "filters_2": 64,
        "filters_3": 128,
        "kernel_size": 3,
        "dropout": 0.30,
        "dense_units": 256,
    },
    {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_blocks": 3,
        "filters_1": 32,
        "filters_2": 64,
        "filters_3": 128,
        "kernel_size": 3,
        "dropout": 0.35,
        "dense_units": 128,
    },
    {
        "learning_rate": 2e-4,
        "batch_size": 128,
        "num_blocks": 3,
        "filters_1": 64,
        "filters_2": 128,
        "filters_3": 256,
        "kernel_size": 3,
        "dropout": 0.40,
        "dense_units": 256,
    },
    {
        "learning_rate": 8e-4,
        "batch_size": 32,
        "num_blocks": 1,
        "filters_1": 32,
        "filters_2": 64,
        "filters_3": 128,
        "kernel_size": 5,
        "dropout": 0.20,
        "dense_units": 128,
    },
]


def run_manual_search(
    dataset_name: str,
    epochs: int,
    device: str,
    seed: int = 7777,
) -> Tuple[Dict[str, Any] | None, pd.DataFrame]:
    results = []

    best_score = -1.0
    best_config = None

    for iteration, config in enumerate(MANUAL_CONFIGS, start=1):
        metrics = evaluate_config(
            config=config,
            dataset_name=dataset_name,
            epochs=epochs,
            device=device,
            seed=seed,
        )

        row = {
            "method": "manual_search",
            "iteration": iteration,
            **config,
            **metrics,
        }
        results.append(row)

        if metrics["val_accuracy"] > best_score:
            best_score = metrics["val_accuracy"]
            best_config = dict(config)

        print(
            f"[MANUAL] iter={iteration:02d}/{len(MANUAL_CONFIGS)} | "
            f"val_acc={metrics['val_accuracy']:.4f} | "
            f"time={metrics['time_sec']:.1f}s"
        )

    df = pd.DataFrame(results)
    return best_config, df