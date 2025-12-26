from typing import Tuple, Any, Dict

from torch.utils.data import DataLoader

from src.utils.seed import seed_worker, make_torch_generator
from .cifar10 import get_cifar10_datasets

def get_loaders(cfg, seed: int):
    train_ds, val_ds, test_ds, split_info = get_cifar10_datasets(cfg)
    
    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.dataset.num_workers)
    pin_memory = bool(cfg.dataset.pin_memory)
    
    g = make_torch_generator(seed)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )
    
    return train_loader, val_loader, test_loader, split_info