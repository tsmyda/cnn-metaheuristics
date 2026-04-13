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