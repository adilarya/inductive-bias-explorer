from typing import Any

import torch.nn as nn

from .mlp import MLPClassifier
from .cnn_tiny import TinyCNN 

def build_model(model_name: str, cfg) -> nn.Module:
    name = model_name.lower().strip()
    
    if hasattr(cfg, "model_rules") and hasattr(cfg.model_rules, "batch_norm"):
        if bool(cfg.model_rules.batch_norm):
            raise ValueError("cfg.model_rules.batch_norm is True but this project phase freezes BN off.")
        
    if name in ("mlp", "mlp_classifier"):
        return MLPClassifier(num_classes=10)
    
    if name in ("cnn_tiny", "tinycnn", "tiny_cnn"):
        return TinyCNN(num_classes=10)
    
    raise ValueError(f"Unknown model_name '{model_name}'. Available models: 'mlp', 'cnn_tiny'.")