import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


def format_parameter_count(n: int) -> str:
    for unit, div in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)
