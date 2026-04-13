from typing import Any, Dict, List, Tuple
import copy
import random

import pandas as pd

from src.evaluator import evaluate_config
from src.search_space import sample_config, repair_config


def random_neighbor(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Small local modification of a configuration.
    For categorical parameters we resample one value.
    For continuous parameters we slightly perturb.
    """
    new_config = copy.deepcopy(config)
    key = random.choice(list(new_config.keys()))

    if key == "learning_rate":
        factor = random.uniform(0.5, 1.5)
        new_config[key] = new_config[key] * factor

    elif key == "dropout":
        delta = random.uniform(-0.1, 0.1)
        new_config[key] = new_config[key] + delta

    elif key == "num_blocks":
        new_config[key] = new_config[key] + random.choice([-1, 1])

    elif key in ["batch_size", "filters_1", "filters_2", "filters_3", "kernel_size", "dense_units"]:
        fresh = sample_config()
        new_config[key] = fresh[key]

    return repair_config(new_config)


def improvise_harmony(
    harmony_memory: List[Dict[str, Any]],
    hmcr: float,
    par: float,
) -> Dict[str, Any]:
    """
    Create a new harmony:
    - with probability HMCR choose values from memory
    - otherwise sample random value
    - with probability PAR apply small pitch adjustment
    """
    base_random = sample_config()
    new_config = {}

    keys = list(base_random.keys())

    for key in keys:
        if random.random() < hmcr and len(harmony_memory) > 0:
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
    """
    Harmony Search for CNN hyperparameter tuning.

    Total budget:
        harmony_memory_size + iterations
    """
    random.seed(seed)

    results = []
    eval_counter = 0

    harmony_memory = []
    harmony_scores = []

    best_score = -1.0
    best_config = None

    # Initialize harmony memory
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
            f"[HS] {idx+1:02d}/{harmony_memory_size} | "
            f"val_acc={score:.4f} | best={best_score:.4f}"
        )

    # Main loop
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