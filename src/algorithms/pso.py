from typing import Any, Dict, Tuple
import copy
import random

import pandas as pd

from src.evaluator import evaluate_config
from src.search_space import repair_config


BOUNDS = {
    "learning_rate": (1e-4, 1e-2),
    "batch_size": (32, 256),
    "num_blocks": (1, 3),
    "filters_1": (16, 64),
    "filters_2": (32, 128),
    "filters_3": (64, 256),
    "kernel_size": (3, 5),
    "dropout": (0.0, 0.5),
    "dense_units": (64, 256),
}

KEYS = list(BOUNDS.keys())


def random_particle() -> Tuple[Dict[str, float], Dict[str, float]]:
    pos = {}
    vel = {}

    for key, (low, high) in BOUNDS.items():
        pos[key] = random.uniform(low, high)
        vel[key] = random.uniform(-(high - low) * 0.1, (high - low) * 0.1)

    return pos, vel


def clamp_position(pos: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for key, value in pos.items():
        low, high = BOUNDS[key]
        out[key] = max(low, min(high, value))
    return out


def run_pso(
    dataset_name: str,
    swarm_size: int,
    iterations: int,
    epochs: int,
    device: str,
    seed: int = 7777,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    PSO dla strojenia hiperparametrów CNN.

    Budżet ewaluacji:
        swarm_size * iterations
    """

    random.seed(seed)

    swarm = []
    for _ in range(swarm_size):
        pos, vel = random_particle()
        swarm.append({
            "pos": pos,
            "vel": vel,
            "best_pos": copy.deepcopy(pos),
            "best_score": -1.0,
        })

    global_best_pos = None
    global_best_score = -1.0

    results = []
    eval_counter = 0

    for iteration in range(1, iterations + 1):
        for particle_idx, particle in enumerate(swarm, start=1):
            config = repair_config(particle["pos"])

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
                "method": "pso",
                "iteration": eval_counter,
                "pso_iteration": iteration,
                "particle": particle_idx,
                **config,
                **metrics,
            }
            results.append(row)

            if score > particle["best_score"]:
                particle["best_score"] = score
                particle["best_pos"] = copy.deepcopy(particle["pos"])

            if score > global_best_score:
                global_best_score = score
                global_best_pos = copy.deepcopy(particle["pos"])

            print(
                f"[PSO] iter={iteration:02d}/{iterations} | "
                f"particle={particle_idx:02d}/{swarm_size} | "
                f"val_acc={score:.4f} | "
                f"best={global_best_score:.4f}"
            )

        for particle in swarm:
            for key in KEYS:
                r1 = random.random()
                r2 = random.random()

                cognitive = c1 * r1 * (particle["best_pos"][key] - particle["pos"][key])
                if global_best_pos is None:
                    raise RuntimeError("global_best_pos is None after evaluation.")
                social = c2 * r2 * (global_best_pos[key] - particle["pos"][key])

                particle["vel"][key] = w * particle["vel"][key] + cognitive + social
                particle["pos"][key] = particle["pos"][key] + particle["vel"][key]

            particle["pos"] = clamp_position(particle["pos"])
    
    if global_best_pos is None:
        raise RuntimeError("PSO finished without best position.")

    best_config = repair_config(global_best_pos)
    df = pd.DataFrame(results)

    return best_config, df