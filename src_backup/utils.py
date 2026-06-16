import torch
import random
import numpy as np
import torch
from thop import profile, clever_format


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_flops(model, input_size=(1, 3, 224, 224), device="cuda"):
    model.eval()

    dummy_input = torch.randn(*input_size).to(device)

    flops, params = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )

    flops_readable, params_readable = clever_format([flops, params], "%.3f")

    return flops, params, flops_readable, params_readable

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config_device="cuda"):
    if config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")