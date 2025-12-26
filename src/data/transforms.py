from typing import Any, Dict, List

from torchvision import transforms

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def _get_normalize(name: str) -> transforms.Normalize:
    name = name.lower()
    if name == "cifar10":
        return transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD)
    raise ValueError(f"Unknown normalize preset: {name}")

def _build_one(op: Dict[str, Any]) -> transforms.Transform:
    if len(op) != 1:
        raise ValueError(f"Each augmentation entry must have exactly one key. Got: {op}")
    
    (name, params), = op.items()
    name_l = name.lower()
    
    if name_l == "random_crop":
        size = int(params.get("size", 32))
        padding = int(params.get("padding", 0))
        return transforms.RandomCrop(size=size, padding=padding)
    
    if name_l == "horizontal_flip":
        p = float(params.get("p", 0.5))
        return transforms.RandomHorizontalFlip(p=p)
    
    if name_l == "normalize":
        if isinstance(params, str):
            preset = params
        elif isinstance(params, dict):
            preset = params.get("preset", "cifar10")
        else:
            raise ValueError(f"Invalid parameters for normalize: {params}")
        return _get_normalize(preset)
    raise ValueError(f"Unknown augmentation operation: {name}")

def build_transforms(cfg, split: str) -> transforms.Compose:
    split = split.lower()
    
    if split not in ("train", "eval"):
        raise ValueError("Split must be 'train' or 'eval'")
    
    ops_list: List[Dict[str, Any]] = cfg.augmentations[split]
    tfs = []
    
    for op in ops_list:
        if "normalize" in op:
            if not any(isinstance(x, transforms.ToTensor) for x in tfs):
                tfs.append(transforms.ToTensor())
            tfs.append(_build_one(op))
        else:
            tfs.append(_build_one(op))
            
    if not any(isinstance(x, transforms.ToTensor) for x in tfs):
        tfs.append(transforms.ToTensor())
    
    return transforms.Compose(tfs)