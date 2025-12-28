from typing import Any

import torch.nn as nn

from .mlp import MLPClassifier
from .cnn_tiny import TinyCNN 
from .cnn_1x1 import CNN1x1

def build_model(model_name: str, cfg) -> nn.Module:
    name = model_name.lower().strip()
    
    if hasattr(cfg, "model_rules") and hasattr(cfg.model_rules, "batch_norm"):
        if bool(cfg.model_rules.batch_norm):
            raise ValueError("cfg.model_rules.batch_norm is True but this project phase freezes BN off.")
        
    if name in ("mlp", "mlp_classifier"):
        h = int(getattr(cfg.models.mlp, "hidden_dim", 512))
        L = int(getattr(cfg.models.mlp, "num_hidden_layers", 2))
        drop = float(getattr(cfg.models.mlp, "dropout", 0.0))
        return MLPClassifier(num_classes=10, hidden_dim=h, num_hidden_layers=L, dropout=drop)
    
    if name in ("cnn_tiny", "tinycnn", "tiny_cnn"):
        return TinyCNN(num_classes=10)
    
    if name in ("cnn_1x1", "cnn1x1", "tiny_cnn_1x1"):
        return CNN1x1(num_classes=10)
    
    raise ValueError(f"Unknown model_name '{model_name}'. Available models: 'mlp', 'cnn_tiny'.")