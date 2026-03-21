import torch
import pandas as pd

from src.algorithms.manual_search import run_manual_search
from src.algorithms.random_search import run_random_search
from src.plots import plot_best_so_far, save_summary_table
from src.utils import ensure_dir, set_seed


def main():
    set_seed(42)

    dataset_name = "FashionMNIST"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 5

    random_budget = 5

    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs per evaluation: {epochs}")
    print(f"Random budget: {random_budget}")

    ensure_dir("results")
    ensure_dir("results/tables")
    ensure_dir("results/figures")

    best_manual, df_manual = run_manual_search(
        dataset_name=dataset_name,
        epochs=epochs,
        device=device,
        seed=42,
    )

    best_random, df_random = run_random_search(
        dataset_name=dataset_name,
        budget=random_budget,
        epochs=epochs,
        device=device,
        seed=42,
    )

    df_all = pd.concat([df_manual, df_random], ignore_index=True)

    csv_path = "results/tables/baselines_comparison.csv"
    fig_path = "results/figures/best_so_far_baselines.png"
    summary_path = "results/tables/baselines_summary.csv"

    df_all.to_csv(csv_path, index=False)
    plot_best_so_far(csv_path, fig_path)
    save_summary_table(csv_path, summary_path)

    print("\n=== BEST MANUAL CONFIG ===")
    print(best_manual)

    print("\n=== BEST RANDOM CONFIG ===")
    print(best_random)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()