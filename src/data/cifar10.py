from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from .transforms import build_transforms

_CIFAR10_NUM_TRAIN = 50000
_CIFAR10_NUM_TEST = 10000

def _stratified_split_indices(
    labels: np.ndarray,
    train_size: int,
    val_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if train_size + val_size != len(labels):
        raise ValueError("train_size + val_size must equal the total number of samples")
    
    rng = np.random.default_rng(seed)
    
    labels = labels.astype(int)
    classes = np.unique(labels)
    
    train_idx_parts = []
    val_idx_parts = []
    
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        
        n_c = len(idx_c)
        val_c = int(round(val_size * (n_c / len(labels))))
        
        val_c = max(0, min(val_c, n_c))
        train_c = n_c - val_c
        
        val_idx_parts.append(idx_c[:val_c])
        train_idx_parts.append(idx_c[val_c:])
    
    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    
    def _fix_sizes(train_idx, val_idx):
        rng_local = np.random.default_rng(seed + 999)
        if len(val_idx) > val_size:
            extra = len(val_idx) - val_size
            rng_local.shuffle(val_idx)
            move = val_idx[:extra]
            keep = val_idx[extra:]
            val_idx = keep
            train_idx = np.concatenate([train_idx, move])
        elif len(val_idx) < val_size:
            need = val_size - len(val_idx)
            rng_local.shuffle(train_idx)
            move = train_idx[:need]
            keep = train_idx[need:]
            train_idx = keep
            val_idx = np.concatenate([val_idx, move])
        return train_idx, val_idx
    
    train_idx, val_idx = _fix_sizes(train_idx, val_idx)
    
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    
    assert len(train_idx) == train_size
    assert len(val_idx) == val_size
    
    return train_idx, val_idx

def _random_split_indices(
    n: int, 
    train_size: int,
    val_size: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if train_size + val_size != n:
        raise ValueError("train_size + val_size must equal n")
    
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    return train_idx, val_idx

def get_cifar10_datasets(cfg): 
    if cfg.dataset.name.lower() != "cifar10":
        raise ValueError("This function only supports CIFAR-10 dataset")
    
    train_tf = build_transforms(cfg, split="train")
    eval_tf = build_transforms(cfg, split="eval")
    
    base_train = CIFAR10(
        root=cfg.dataset.root,
        train=True,
        download=True,
        transform=train_tf
    )
    
    base_train_eval = CIFAR10(
        root=cfg.dataset.root,
        train=True,
        download=False,
        transform=eval_tf
    )
    
    test_ds = CIFAR10(
        root=cfg.dataset.root,
        train=False,
        download=True,
        transform=eval_tf
    )
    
    if len(base_train) != _CIFAR10_NUM_TRAIN:
        raise ValueError("Unexpected number of training samples in CIFAR-10")
    if len(test_ds) != _CIFAR10_NUM_TEST:
        raise ValueError("Unexpected number of test samples in CIFAR-10")
    
    train_size = int(cfg.split.train)
    val_size = int(cfg.split.val)
    seed = int(cfg.split.val_split_seed)
    stratified = bool(cfg.split.stratified)
    
    labels = np.array(base_train.targets)
    
    if stratified:
        train_idx, val_idx = _stratified_split_indices(
            labels=labels,
            train_size=train_size,
            val_size=val_size,
            seed=seed
        )
    else:
        train_idx, val_idx = _random_split_indices(
            n=len(base_train),
            train_size=train_size,
            val_size=val_size,
            seed=seed
        )
        
    train_ds = Subset(base_train, indices=train_idx.tolist())
    val_ds = Subset(base_train_eval, indices=val_idx.tolist())
    
    split_info = {
        "dataset": "cifar10",
        "train_size": train_size,
        "val_size": val_size,
        "test_size": len(test_ds),
        "val_split_seed": seed,
        "stratified": stratified,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    
    return train_ds, val_ds, test_ds, split_info