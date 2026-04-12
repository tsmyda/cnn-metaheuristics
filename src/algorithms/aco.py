from typing import Any, Dict, List, Tuple
import copy
import math
import random

import pandas as pd

from src.evaluator import evaluate_config
from src.search_space import repair_config


DISCRETE_SPACE = {
    "batch_size": [32, 64, 128, 256],
    "num_blocks": [1, 2, 3],
    "filters_1": [16, 32, 64],
    "filters_2": [32, 64, 128],
    "filters_3": [64, 128, 256],
    "kernel_size": [3, 5],
    "dense_units": [64, 128, 256],
}

CONTINUOUS_SPACE = {
    "learning_rate": (1e-4, 1e-2),
    "dropout": (0.0, 0.5),
}


def init_pheromones() -> Dict[str, Dict[Any, float]]:
    pheromones = {}
    for key, values in DISCRETE_SPACE.items():
        pheromones[key] = {value: 1.0 for value in values}
    return pheromones


def sample_from_pheromones(prob_dict: Dict[Any, float]) -> Any:
    values = list(prob_dict.keys())
    weights = list(prob_dict.values())
    return random.choices(values, weights=weights, k=1)[0]


def construct_solution(
    pheromones: Dict[str, Dict[Any, float]],
) -> Dict[str, Any]:
    config = {}

    for key in DISCRETE_SPACE:
        config[key] = sample_from_pheromones(pheromones[key])

    lr_low, lr_high = CONTINUOUS_SPACE["learning_rate"]
    dr_low, dr_high = CONTINUOUS_SPACE["dropout"]

    config["learning_rate"] = 10 ** random.uniform(math.log10(lr_low), math.log10(lr_high))
    config["dropout"] = random.uniform(dr_low, dr_high)

    return repair_config(config)


def evaporate(
    pheromones: Dict[str, Dict[Any, float]],
    evaporation_rate: float,
) -> None:
    for key in pheromones:
        for value in pheromones[key]:
            pheromones[key][value] *= (1.0 - evaporation_rate)
            pheromones[key][value] = max(pheromones[key][value], 1e-6)


def deposit(
    pheromones: Dict[str, Dict[Any, float]],
    config: Dict[str, Any],
    score: float,
    q: float,
) -> None:
    for key in DISCRETE_SPACE:
        pheromones[key][config[key]] += q * score


def run_aco(
    dataset_name: str,
    ants: int,
    iterations: int,
    epochs: int,
    device: str,
    seed: int = 7777,
    evaporation_rate: float = 0.2,
    q: float = 1.0,
    top_k_deposit: int = 2,
) -> Tuple[Dict[str, Any] | None, pd.DataFrame]:
    """
    Ant Colony Optimization for CNN hyperparameter tuning.

    Total budget:
        ants * iterations
    """
    random.seed(seed)

    pheromones = init_pheromones()
    results = []

    best_score = -1.0
    best_config = None
    eval_counter = 0

    for iteration in range(1, iterations + 1):
        iter_solutions: List[Tuple[Dict[str, Any], float]] = []

        for ant_idx in range(1, ants + 1):
            config = construct_solution(pheromones)

            metrics = evaluate_config(
                config=config,
                dataset_name=dataset_name,
                epochs=epochs,
                device=device,
                seed=seed,
            )

            score = metrics["val_accuracy"]
            eval_counter += 1

            row = {
                "method": "aco",
                "iteration": eval_counter,
                "aco_iteration": iteration,
                "ant": ant_idx,
                **config,
                **metrics,
            }
            results.append(row)
            iter_solutions.append((config, score))

            if score > best_score:
                best_score = score
                best_config = copy.deepcopy(config)

            print(
                f"[ACO] iter={iteration:02d}/{iterations} | "
                f"ant={ant_idx:02d}/{ants} | "
                f"val_acc={score:.4f} | best={best_score:.4f}"
            )

        evaporate(pheromones, evaporation_rate)

        iter_solutions.sort(key=lambda x: x[1], reverse=True)
        for config, score in iter_solutions[:top_k_deposit]:
            deposit(pheromones, config, score, q=q)

    df = pd.DataFrame(results)
    return best_config, df