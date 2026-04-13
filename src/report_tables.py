from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def save_method_summary(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    summary = (
        df.groupby("method")[["val_accuracy", "test_accuracy", "time_sec", "num_params"]]
        .agg(["max", "mean", "std"])
        .round(4)
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    return summary


def save_best_configs(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    idx = df.groupby("method")["val_accuracy"].idxmax()
    best_df = df.loc[idx].sort_values("val_accuracy", ascending=False).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(output_path, index=False)
    return best_df


def save_time_to_best(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    rows = []

    for method, group in df.groupby("method"):
        group = group.sort_values("iteration").copy()
        best_idx = group["val_accuracy"].idxmax()
        best_row = group.loc[best_idx]

        rows.append({
            "method": method,
            "best_val_accuracy": best_row["val_accuracy"],
            "test_accuracy_at_best": best_row["test_accuracy"],
            "iteration_of_best": best_row["iteration"],
            "time_of_best_sec": group.loc[group.index <= best_idx, "time_sec"].sum(),
            "total_time_sec": group["time_sec"].sum(),
        })

    out = pd.DataFrame(rows).sort_values("best_val_accuracy", ascending=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out

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
    