from dataclasses import dataclass
from typing import Optional

import torch

@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.ndim != 2:
        raise ValueError(f"Logits must be (B, C). Got shape {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"Targets must be (B,). Got shape {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"Batch size mismatch between logits and targets: {logits.shape[0]} vs {targets.shape[0]}")
    
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

@dataclass
class AverageMeter:
    name: str
    total: float = 0.0
    count: int = 0
    
    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
        
    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)
        
    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

def to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)