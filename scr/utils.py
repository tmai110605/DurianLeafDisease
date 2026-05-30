import torch
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config_device="cuda"):
    if config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")