from pathlib import Path

import pandas as pd
import torch

from src.algorithms.random_search import run_random_search
from src.utils import ensure_dir, set_seed


def main():
    set_seed(7777)

    dataset_name = "FashionMNIST"
    budget = 10
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Budget: {budget}")
    print(f"Epochs per evaluation: {epochs}")

    ensure_dir("results")
    ensure_dir("results/tables")

    best_config, df = run_random_search(
        dataset_name=dataset_name,
        budget=budget,
        epochs=epochs,
        device=device,
        seed=7777,
    )

    out_csv = "results/tables/random_search_results.csv"
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("method")[["val_accuracy", "test_accuracy", "time_sec", "num_params"]]
        .agg(["max", "mean"])
    )

    print("\n=== SUMMARY ===")
    print(summary)

    print("\n=== BEST CONFIG ===")
    print(best_config)

    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()