from typing import List

import pandas as pd

from src.algorithms import (
    run_manual_search,
    run_random_search,
    run_ga,
    run_pso,
    run_aco,
    run_harmony_search,
)


def run_all_methods(
    dataset_name: str,
    epochs: int,
    device: str,
    seed: int,
    random_budget: int = 10,
    ga_population: int = 5,
    ga_generations: int = 4,
    pso_swarm: int = 5,
    pso_iterations: int = 4,
    aco_ants: int = 5,
    aco_iterations: int = 4,
    hs_memory_size: int = 5,
    hs_iterations: int = 15,
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    _, df_manual = run_manual_search(
        dataset_name=dataset_name,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    dfs.append(df_manual)

    _, df_random = run_random_search(
        dataset_name=dataset_name,
        budget=random_budget,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    dfs.append(df_random)

    _, df_ga = run_ga(
        dataset_name=dataset_name,
        population_size=ga_population,
        generations=ga_generations,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    dfs.append(df_ga)

    _, df_pso = run_pso(
        dataset_name=dataset_name,
        swarm_size=pso_swarm,
        iterations=pso_iterations,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    dfs.append(df_pso)

    _, df_aco = run_aco(
        dataset_name=dataset_name,
        ants=aco_ants,
        iterations=aco_iterations,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    dfs.append(df_aco)

    _, df_hs = run_harmony_search(
        dataset_name=dataset_name,
        harmony_memory_size=hs_memory_size,
        iterations=hs_iterations,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    dfs.append(df_hs)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["dataset"] = dataset_name
    df_all["seed"] = seed

    return df_all