# CNN Metaheuristics

This project compares several hyperparameter search strategies for a tunable CNN
on FashionMNIST, CIFAR-10, and CIFAR-100. The study covers manual search, random
search, GA, PSO, ACO, and Harmony Search.

## Project Layout

- `src/`: core library code for datasets, model, training, algorithms, evaluation, and plots.
- `scripts/`: runnable experiment entry points.
- `notebooks/`: interactive analysis and experimentation.
- `report/`: the final report.
- `notebooks/results/`: generated figures and tables from the notebook.
- `data/`: dataset cache and data files.

## Reproducing The Experiment

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook or experiment scripts from the project root.

3. Generated tables and figures will appear under `notebooks/results/`.

## Run Scripts

From the project root:

```bash
python scripts/run_all_methods.py
```

Then choose supported dataset and heuristics.
