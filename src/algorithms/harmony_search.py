import copy
import random
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.evaluator import evaluate_config
from src.search_space import repair_config, sample_config


RESAMPLED_KEYS = (
    "batch_size",
    "filters_1",
    "filters_2",
    "filters_3",
    "kernel_size",
    "dense_units",
)


def random_neighbor(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one local perturbation to a sampled configuration."""
    new_config = copy.deepcopy(config)
    key = random.choice(list(new_config.keys()))

    # Continuous values are nudged, while discrete values are resampled from a
    # fresh configuration to stay consistent with the global search space.
    if key == "learning_rate":
        factor = random.uniform(0.5, 1.5)
        new_config[key] = new_config[key] * factor
    elif key == "dropout":
        delta = random.uniform(-0.1, 0.1)
        new_config[key] = new_config[key] + delta
    elif key == "num_blocks":
        new_config[key] = new_config[key] + random.choice([-1, 1])
    elif key in RESAMPLED_KEYS:
        fresh = sample_config()
        new_config[key] = fresh[key]

    return repair_config(new_config)


def improvise_harmony(
    harmony_memory: List[Dict[str, Any]],
    hmcr: float,
    par: float,
) -> Dict[str, Any]:
    """Sample one harmony from memory and optional local pitch adjustment."""
    base_random = sample_config()
    new_config = {}

    for key in base_random:
        if random.random() < hmcr and harmony_memory:
            source = random.choice(harmony_memory)
            new_config[key] = source[key]
        else:
            new_config[key] = base_random[key]

    if random.random() < par:
        new_config = random_neighbor(new_config)

    return repair_config(new_config)


def run_harmony_search(
    dataset_name: str,
    harmony_memory_size: int,
    iterations: int,
    epochs: int,
    device: str,
    seed: int = 7777,
    hmcr: float = 0.9,
    par: float = 0.3,
) -> Tuple[Dict[str, Any] | None, pd.DataFrame]:
    """Run harmony search for CNN hyperparameter tuning."""
    random.seed(seed)

    results = []
    eval_counter = 0

    harmony_memory = []
    harmony_scores = []

    best_score = -1.0
    best_config = None

    for idx in range(harmony_memory_size):
        config = repair_config(sample_config())
        metrics = evaluate_config(
            config=config,
            dataset_name=dataset_name,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        score = metrics["val_accuracy"]
        eval_counter += 1

        harmony_memory.append(config)
        harmony_scores.append(score)

        row = {
            "method": "harmony_search",
            "iteration": eval_counter,
            "hs_phase": "init",
            "memory_index": idx + 1,
            **config,
            **metrics,
        }
        results.append(row)

        if score > best_score:
            best_score = score
            best_config = copy.deepcopy(config)

        print(
            f"[HS] {idx + 1:02d}/{harmony_memory_size} | "
            f"val_acc={score:.4f} | best={best_score:.4f}"
        )

    for it in range(1, iterations + 1):
        new_config = improvise_harmony(
            harmony_memory=harmony_memory,
            hmcr=hmcr,
            par=par,
        )
        metrics = evaluate_config(
            config=new_config,
            dataset_name=dataset_name,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        score = metrics["val_accuracy"]
        eval_counter += 1

        row = {
            "method": "harmony_search",
            "iteration": eval_counter,
            "hs_phase": "main",
            "hs_iteration": it,
            **new_config,
            **metrics,
        }
        results.append(row)

        worst_idx = min(range(len(harmony_scores)), key=lambda i: harmony_scores[i])
        if score > harmony_scores[worst_idx]:
            # Harmony memory is updated only when the new sample improves the
            # current worst stored solution.
            harmony_memory[worst_idx] = copy.deepcopy(new_config)
            harmony_scores[worst_idx] = score

        if score > best_score:
            best_score = score
            best_config = copy.deepcopy(new_config)

        print(
            f"[HS] iter={it:02d}/{iterations} | "
            f"val_acc={score:.4f} | best={best_score:.4f}"
        )

    df = pd.DataFrame(results)
    return best_config, df
