import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.experiment import run_all_methods
from src.plots import plot_best_so_far
from src.report_tables import save_method_summary, save_best_configs, save_time_to_best
from src.utils import ensure_dir, set_seed


def main():
    set_seed(7777)

    dataset_name = "FashionMNIST"
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

    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()