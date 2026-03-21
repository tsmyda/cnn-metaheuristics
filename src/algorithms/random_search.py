from typing import Any, Dict, Tuple

import pandas as pd

from src.evaluator import evaluate_config
from src.search_space import sample_config


def run_random_search(
    dataset_name: str,
    budget: int,
    epochs: int,
    device: str,
    seed: int = 7777,
) -> Tuple[Dict[str, Any] | None, pd.DataFrame]:
    results = []

    best_score = -1.0
    best_config = None

    for iteration in range(1, budget + 1):
        config = sample_config()
        metrics = evaluate_config(
            config=config,
            dataset_name=dataset_name,
            epochs=epochs,
            device=device,
            seed=seed,
        )

        row = {
            "method": "random_search",
            "iteration": iteration,
            **config,
            **metrics,
        }
        results.append(row)

        if metrics["val_accuracy"] > best_score:
            best_score = metrics["val_accuracy"]
            best_config = dict(config)

        print(
            f"[RS] iter={iteration:02d}/{budget} | "
            f"val_acc={metrics['val_accuracy']:.4f} | "
            f"time={metrics['time_sec']:.1f}s"
        )

    df = pd.DataFrame(results)
    return best_config, df