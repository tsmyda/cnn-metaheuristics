import math
import random
from typing import Any, Dict


SEARCH_SPACE = {
    "learning_rate": ("log_float", 1e-4, 1e-2),
    "batch_size": ("categorical", [32, 64, 128, 256]),
    "num_blocks": ("int", 1, 3),
    "filters_1": ("categorical", [16, 32, 64]),
    "filters_2": ("categorical", [32, 64, 128]),
    "filters_3": ("categorical", [64, 128, 256]),
    "kernel_size": ("categorical", [3, 5]),
    "dropout": ("float", 0.0, 0.5),
    "dense_units": ("categorical", [64, 128, 256]),
    "optimizer": ("categorical", ["adam", "sgd", "adamw"]),
    "weight_decay": ("log_float", 1e-6, 1e-3),
    "use_batch_norm": ("categorical", [0, 1]),
}


def sample_config() -> Dict[str, Any]:
    config = {}

    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]

        if kind == "int":
            _, low, high = spec
            config[name] = random.randint(low, high)

        elif kind == "float":
            _, low, high = spec
            config[name] = random.uniform(low, high)

        elif kind == "log_float":
            _, low, high = spec
            config[name] = 10 ** random.uniform(math.log10(low), math.log10(high))

        elif kind == "categorical":
            _, values = spec
            config[name] = random.choice(values)

        else:
            raise ValueError(f"Unknown search space kind: {kind}")

    return config


def repair_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config)

    cfg.setdefault("learning_rate", 1e-3)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("num_blocks", 2)
    cfg.setdefault("filters_1", 32)
    cfg.setdefault("filters_2", 64)
    cfg.setdefault("filters_3", 128)
    cfg.setdefault("kernel_size", 3)
    cfg.setdefault("dropout", 0.25)
    cfg.setdefault("dense_units", 128)
    cfg.setdefault("optimizer", "adam")
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("use_batch_norm", 1)

    cfg["learning_rate"] = float(max(1e-4, min(1e-2, cfg["learning_rate"])))
    cfg["dropout"] = float(max(0.0, min(0.5, cfg["dropout"])))
    cfg["num_blocks"] = int(max(1, min(3, round(cfg["num_blocks"]))))

    batch_choices = [32, 64, 128, 256]
    kernel_choices = [3, 5]
    filters_1_choices = [16, 32, 64]
    filters_2_choices = [32, 64, 128]
    filters_3_choices = [64, 128, 256]
    dense_choices = [64, 128, 256]
    optimizer_choices = ["adam", "sgd", "adamw"]
    weight_decay_choices = [1e-6, 1e-5, 1e-4, 1e-3]
    use_batch_norm_choices = [0, 1]

    cfg["batch_size"] = min(batch_choices, key=lambda x: abs(x - int(cfg["batch_size"])))
    cfg["kernel_size"] = min(kernel_choices, key=lambda x: abs(x - int(cfg["kernel_size"])))
    cfg["filters_1"] = min(filters_1_choices, key=lambda x: abs(x - int(cfg["filters_1"])))
    cfg["filters_2"] = min(filters_2_choices, key=lambda x: abs(x - int(cfg["filters_2"])))
    cfg["filters_3"] = min(filters_3_choices, key=lambda x: abs(x - int(cfg["filters_3"])))
    cfg["dense_units"] = min(dense_choices, key=lambda x: abs(x - int(cfg["dense_units"])))
    optimizer_value = cfg.get("optimizer")
    cfg["optimizer"] = optimizer_value if optimizer_value in optimizer_choices else "adam"
    cfg["weight_decay"] = min(weight_decay_choices, key=lambda x: abs(x - cfg["weight_decay"]))
    cfg["use_batch_norm"] = min(use_batch_norm_choices, key=lambda x: abs(x - cfg["use_batch_norm"]))


    return cfg