import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.evaluator import evaluate_config
from src.search_space import sample_config


def main():
    dataset_name = "FashionMNIST"
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 7777

    runs = 3
    times = []

    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs per evaluation: {epochs}")
    print(f"Running {runs} measurements...")

    for i in range(runs):
        config = sample_config()
        metrics = evaluate_config(
            config=config,
            dataset_name=dataset_name,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        t = metrics.get("time_sec", None)
        print(f"Run {i+1}: {t:.2f}s")
        times.append(t)

    avg = sum(times) / len(times)
    budget = max(1, int(round(10 * 3600 / avg)))

    print(f"Avg time per evaluation: {avg:.2f}s")
    print(f"Suggested budget for ~10 hours: {budget} evaluations (~{(budget*avg)/3600:.2f} hours)")


if __name__ == '__main__':
    main()
