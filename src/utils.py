import os
import random

import numpy as np
import torch


def set_seed(seed: int = 7777) -> None:
    """Set Python, NumPy, and Torch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_torch_seed(seed: int = 7777) -> None:
    """Set NumPy and Torch seeds used by model evaluation code."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a Torch model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
