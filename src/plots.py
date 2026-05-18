from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

def plot_best_so_far(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for method, group in df.groupby("method"):
        group = group.sort_values("iteration").copy()
        group["best_so_far"] = group["val_accuracy"].cummax()

        ax.step(
            group["iteration"],
            group["best_so_far"],
            where="post",
            marker="o",
            label=method,
        )

    ax.set_xlabel("Liczba ewaluacji")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Porównanie metod strojenia")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_time_to_best(df: pd.DataFrame, output_path: str) -> None:

    plt.figure(figsize=(10, 6))
    plt.bar(df["method"], df["time_of_best_sec"], color="skyblue")
    plt.xlabel("Method")
    plt.ylabel("Time to Best (sec)")
    plt.title("Time to Best Validation Accuracy by Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_time_to_best_combined(
    dataset_to_df: dict[str, pd.DataFrame],
    output_path: str,
) -> None:

    method_order = [
        "manual_search",
        "random_search",
        "ga",
        "pso",
        "aco",
        "harmony_search",
    ]

    pretty_names = {
        "manual_search": "Manual",
        "random_search": "Random",
        "ga": "GA",
        "pso": "PSO",
        "aco": "ACO",
        "harmony_search": "HS",
    }

    n_datasets = len(dataset_to_df)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), sharey=True)

    if n_datasets == 1:
        axes = [axes]

    for ax, (dataset_name, df) in zip(axes, dataset_to_df.items()):
        plot_df = df.copy()

        plot_df["method"] = pd.Categorical(
            plot_df["method"],
            categories=method_order,
            ordered=True,
        )
        plot_df = plot_df.sort_values("method")

        x_labels = [pretty_names[m] for m in plot_df["method"]]
        y_values = plot_df["time_of_best_sec"]

        ax.bar(x_labels, y_values)
        ax.set_title(dataset_name)
        ax.set_xlabel("Metoda")
        ax.tick_params(axis="x", rotation=35)

    axes[0].set_ylabel("Czas do najlepszego wyniku [s]")

    fig.suptitle("Czas do osiągnięcia najlepszej dokładności walidacyjnej", fontsize=12)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
def plot_hparam_vs_accuracy(
    csv_path: str,
    param_name: str,
    output_path: str,
    log_x: bool = False,
) -> None:
    df = pd.read_csv(csv_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(df[param_name], df["val_accuracy"], alpha=0.7)

    if log_x:
        plt.xscale("log")

    plt.xlabel(param_name)
    plt.ylabel("Validation accuracy")
    plt.title(f"{param_name} vs validation accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_hyperparam_metric_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    hyperparams = [
        "learning_rate",
        "batch_size",
        "num_blocks",
        "filters_1",
        "filters_2",
        "filters_3",
        "kernel_size",
        "dropout",
        "dense_units",
    ]
    metrics = ["val_accuracy", "test_accuracy", "time_sec", "num_params"]

    available_hparams = [col for col in hyperparams if col in df.columns]
    available_metrics = [col for col in metrics if col in df.columns]

    if not available_hparams or not available_metrics:
        return

    corr = df[available_hparams + available_metrics].corr(numeric_only=True)
    corr_block = corr.loc[available_hparams, available_metrics]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    im = plt.imshow(corr_block.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    plt.xticks(range(len(available_metrics)), available_metrics, rotation=30, ha="right")
    plt.yticks(range(len(available_hparams)), available_hparams)
    plt.title("Hyperparameter Correlation with Result Metrics")

    for row_idx in range(corr_block.shape[0]):
        for col_idx in range(corr_block.shape[1]):
            value = corr_block.iat[row_idx, col_idx]
            plt.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im)
    cbar.set_label("Pearson correlation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_hyperparam_metric_correlation_heatmaps_by_method(
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    if "method" not in df.columns:
        return

    hyperparams = [
        "learning_rate",
        "batch_size",
        "num_blocks",
        "filters_1",
        "filters_2",
        "filters_3",
        "kernel_size",
        "dropout",
        "dense_units",
    ]
    metrics = ["val_accuracy", "test_accuracy", "time_sec", "num_params"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for method, group in df.groupby("method"):
        available_hparams = [col for col in hyperparams if col in group.columns]
        available_metrics = [col for col in metrics if col in group.columns]

        if not available_hparams or not available_metrics:
            continue

        numeric_group = group[available_hparams + available_metrics].dropna()
        if numeric_group.shape[0] < 2:
            continue

        corr = numeric_group.corr(numeric_only=True)
        corr_block = corr.loc[available_hparams, available_metrics]

        plt.figure(figsize=(9, 6))
        im = plt.imshow(corr_block.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        plt.xticks(range(len(available_metrics)), available_metrics, rotation=30, ha="right")
        plt.yticks(range(len(available_hparams)), available_hparams)
        plt.title(f"Hyperparameter Correlation with Metrics - {method}")

        for row_idx in range(corr_block.shape[0]):
            for col_idx in range(corr_block.shape[1]):
                value = corr_block.iat[row_idx, col_idx]
                plt.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)

        cbar = plt.colorbar(im)
        cbar.set_label("Pearson correlation")

        output_path = Path(output_dir) / f"hyperparam_metric_correlation_{method}.png"
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