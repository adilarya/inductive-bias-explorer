from typing import Dict, Any

import torch
import torch.nn as nn

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def model_param_summary(model: nn.Module) -> Dict[str, Any]:
    return {
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model), 
        "num_buffers": sum(b.numel() for b in model.buffers()),
    }
    
def assert_param_budget(
    model_a: nn.Module,
    model_b: nn.Module,
    tolerance: float = 0.10,
) -> None:
    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative.")
    
    pa = count_trainable_params(model_a)
    pb = count_trainable_params(model_b)
    
    if pa == 0 or pb == 0:
        raise ValueError("One of the models has zero trainable parameters.")
    
    ratio = pa / pb if pa >= pb else pb / pa
    if ratio > (1 + tolerance):
        raise ValueError(
            f"Parameter budget exceeded: Model A has {pa} params, "
            f"Model B has {pb} params, ratio {ratio:.4f} exceeds tolerance {tolerance:.4f}."
        )