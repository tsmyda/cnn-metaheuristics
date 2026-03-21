from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_best_so_far(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for method, group in df.groupby("method"):
        group = group.sort_values("iteration").copy()
        group["best_so_far"] = group["val_accuracy"].cummax()
        plt.plot(group["iteration"], group["best_so_far"], label=method)

    plt.xlabel("Liczba ewaluacji")
    plt.ylabel("Best validation accuracy")
    plt.title("Porównanie metod strojenia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_summary_table(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)

    summary = (
        df.groupby("method")[["val_accuracy", "test_accuracy", "time_sec", "num_params"]]
        .agg(["max", "mean"])
        .round(4)
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)