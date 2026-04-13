import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from InquirerPy import inquirer

from src.algorithms import (
    run_manual_search,
    run_random_search,
    run_ga,
    run_pso,
    run_aco,
    run_harmony_search,
)

from src.plots import plot_best_so_far
from src.report_tables import save_method_summary, save_best_configs, save_time_to_best, plot_time_to_best
from src.utils import ensure_dir, set_seed


def run_all_methods(
    dataset_name: str,
    epochs: int,
    device: str,
    seed: int,
    selected_methods: list[str],
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
    method_set = set(selected_methods)
    dfs: list[pd.DataFrame] = []

    if "manual_search" in method_set:
        _, df_manual = run_manual_search(
            dataset_name=dataset_name,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        dfs.append(df_manual)

    if "random_search" in method_set:
        _, df_random = run_random_search(
            dataset_name=dataset_name,
            budget=random_budget,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        dfs.append(df_random)

    if "ga" in method_set:
        _, df_ga = run_ga(
            dataset_name=dataset_name,
            population_size=ga_population,
            generations=ga_generations,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        dfs.append(df_ga)

    if "pso" in method_set:
        _, df_pso = run_pso(
            dataset_name=dataset_name,
            swarm_size=pso_swarm,
            iterations=pso_iterations,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        dfs.append(df_pso)

    if "aco" in method_set:
        _, df_aco = run_aco(
            dataset_name=dataset_name,
            ants=aco_ants,
            iterations=aco_iterations,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        dfs.append(df_aco)

    if "harmony_search" in method_set:
        _, df_hs = run_harmony_search(
            dataset_name=dataset_name,
            harmony_memory_size=hs_memory_size,
            iterations=hs_iterations,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        dfs.append(df_hs)

    if not dfs:
        raise ValueError("No methods selected.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["dataset"] = dataset_name
    df_all["seed"] = seed
    return df_all


def prompt_selection() -> tuple[str, list[str]]:
    dataset_name = inquirer.select(
        message="Choose dataset:",
        choices=["FashionMNIST", "CIFAR10", "CIFAR100"],
        default="FashionMNIST",
    ).execute()

    selected_methods = inquirer.checkbox(
        message="Select methods to run:",
        choices=[
            {"name": "Manual Search", "value": "manual_search"},
            {"name": "Random Search", "value": "random_search"},
            {"name": "Genetic Algorithm", "value": "ga"},
            {"name": "Particle Swarm Optimization", "value": "pso"},
            {"name": "Ant Colony Optimization", "value": "aco"},
            {"name": "Harmony Search", "value": "harmony_search"},
        ],
    ).execute()

    return dataset_name, selected_methods

def main():
    dataset_name, selected_methods = prompt_selection()

    set_seed(7777)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 5
    seed = 7777

    ensure_dir("results")
    ensure_dir("results/tables")
    ensure_dir("results/figures")

    df = run_all_methods(
        dataset_name=dataset_name,
        epochs=epochs,
        device=device,
        seed=seed,
        selected_methods=selected_methods,
        random_budget=10,
        ga_population=5,
        ga_generations=4,
        pso_swarm=5,
        pso_iterations=4,
        aco_ants=5,
        aco_iterations=4,
        hs_memory_size=5,
        hs_iterations=15,
    )

    out_csv = "results/tables/all_methods_results.csv"
    df.to_csv(out_csv, index=False)

    plot_best_so_far(
        csv_path=out_csv,
        output_path="results/figures/best_so_far_all_methods.png",
    )

    save_method_summary(df, "results/tables/method_summary.csv")
    save_best_configs(df, "results/tables/best_configs.csv")
    save_time_to_best(df, "results/tables/time_to_best.csv")
    plot_time_to_best(df, "results/figures/time_to_best_all_methods.png")

    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    main()