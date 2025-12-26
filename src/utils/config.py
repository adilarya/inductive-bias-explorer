from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from .io import load_yaml

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = self._wrap(v)
        
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(f"'Config' object has no attribute '{item}'") from e
    
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)
        
    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))
        
    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [
                    vv.to_dict() if isinstance(vv, Config) else vv for vv in v
                ]
            else:
                out[k] = v
        return out
    
def load_config(path: str) -> Config:
    data = load_yaml(path)
    return Config(data)

def require_keys(cfg: Config, keys: Iterable[str]) -> None:
    for key in keys:
        _ = get_by_path(cfg, key)
        
def get_by_path(cfg: Config, path: str) -> Any:
    cur: Any = cfg
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: '{path}' (at '{part}')")
        cur = cur[part]
    return cur

def validate_minimal(cfg: Config) -> None:
    require_keys(
        cfg, 
        keys=[
            "dataset.name",
            "dataset.root",
            "runs.dir",
            "runs.name_format",
            "split.train",
            "split.val",
            "split.test",
            "split.val_split_seed",
            "optimizer.name",
            "training.epochs",
            "training.batch_size",
            "lr_schedule.type",
            "lr_schedule.base_lr",
            "augmentations.train",
            "augmentations.eval",
            "seeds.dev",
            "seeds.final",
            "parameter_budget.rule",
            "parameter_budget.tolerance",
            "logging.log_every_epoch",
        ],
    )
    
    if cfg.split.train + cfg.split.val != 50000 and cfg.dataset.name.lower() == "cifar10":
        raise ValueError(
            f"For CIFAR-10, train and val splits must sum to 50000. Got train={cfg.split.train}, val={cfg.split.val}."
        )
        
    if cfg.training.epochs <= 0:
        raise ValueError("Number of training epochs must be positive.")
    
    if cfg.training.batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    
    tol = float(cfg.parameter_budget.tolerance)
    if tol < 0:
        raise ValueError("Parameter budget tolerance must be non-negative.")