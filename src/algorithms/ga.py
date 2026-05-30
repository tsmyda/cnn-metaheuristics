import copy
import random
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.evaluator import evaluate_config
from src.search_space import repair_config, sample_config


def crossover(parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
    """Create a child by choosing each hyperparameter from one parent."""
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child


def mutate(config: Dict[str, Any], mutation_prob: float = 0.2) -> Dict[str, Any]:
    """Randomly resample selected genes and repair the child afterward."""
    child = copy.deepcopy(config)
    random_config = sample_config()

    for key in child:
        if random.random() < mutation_prob:
            child[key] = random_config[key]

    return repair_config(child)


def tournament_selection(
    population_with_scores: List[Tuple[Dict[str, Any], float]],
    tournament_size: int = 3,
) -> Dict[str, Any]:
    """Select one parent from a random mini-tournament."""
    participants = random.sample(
        population_with_scores,
        k=min(tournament_size, len(population_with_scores)),
    )
    participants.sort(key=lambda x: x[1], reverse=True)
    return copy.deepcopy(participants[0][0])


def run_ga(
    dataset_name: str,
    population_size: int,
    generations: int,
    epochs: int,
    device: str,
    seed: int = 7777,
    mutation_prob: float = 0.2,
    elite_size: int = 1,
    tournament_size: int = 3,
) -> Tuple[Dict[str, Any] | None, pd.DataFrame]:
    """Run a genetic algorithm for CNN hyperparameter search."""
    random.seed(seed)

    population = [sample_config() for _ in range(population_size)]

    results = []
    best_score = -1.0
    best_config = None
    eval_counter = 0

    for generation in range(1, generations + 1):
        population_with_scores = []

        for individual_idx, individual in enumerate(population, start=1):
            config = repair_config(individual)
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
                "method": "ga",
                "iteration": eval_counter,
                "generation": generation,
                "individual_in_generation": individual_idx,
                **config,
                **metrics,
            }
            results.append(row)
            population_with_scores.append((config, score))

            if score > best_score:
                best_score = score
                best_config = copy.deepcopy(config)

            print(
                f"[GA] gen={generation:02d}/{generations} | "
                f"ind={individual_idx:02d}/{population_size} | "
                f"val_acc={score:.4f} | "
                f"best={best_score:.4f}"
            )

        population_with_scores.sort(key=lambda x: x[1], reverse=True)
        elites = [
            copy.deepcopy(individual)
            for individual, _ in population_with_scores[:elite_size]
        ]

        new_population = elites[:]
        while len(new_population) < population_size:
            parent1 = tournament_selection(population_with_scores, tournament_size)
            parent2 = tournament_selection(population_with_scores, tournament_size)

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_prob=mutation_prob)
            child = repair_config(child)
            new_population.append(child)

        population = new_population[:population_size]

    df = pd.DataFrame(results)
    return best_config, df
