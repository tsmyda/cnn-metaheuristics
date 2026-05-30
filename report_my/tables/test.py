import pandas as pd

df = pd.read_csv("all_methods_all_datasets.csv")

summary = (
    df.groupby(["dataset", "method"])
      .agg(
          val_mean=("val_accuracy", "mean"),
          val_std=("val_accuracy", "std"),
          val_max=("val_accuracy", "max"),
          test_mean=("test_accuracy", "mean"),
          test_std=("test_accuracy", "std"),
          test_max=("test_accuracy", "max"),
      )
      .reset_index()
)

print(summary)