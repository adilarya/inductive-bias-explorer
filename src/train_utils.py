from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

@torch.no_grad()
def evaluate(model: nn.Module, 
             loader,
             device: torch.device,
             criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    
    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("[EVALUATE] Loss became NaN/Inf during training.")
        
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += y.size(0)
        
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
    }
    
def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("[TRAIN] Loss became NaN/Inf during training.")
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += y.size(0)
        
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
    }