import argparse

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils.config import load_config, validate_minimal
from src.utils.seed import set_seed
from src.utils.io import make_run_dir, save_yaml, save_json, append_jsonl
from src.utils.params import count_trainable_params
from src.data.loader import get_loaders
from src.models.build import build_model
from src.train_utils import train_one_epoch, evaluate

def build_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    if cfg.optimizer.name.lower() != "sgd":
        raise ValueError(f"Only SGD is supported right now. Got: {cfg.optimizer.name}")
    
    return optim.SGD(
        model.parameters(),
        lr=float(cfg.lr_schedule.base_lr),
        momentum=float(cfg.optimizer.momentum),
        nesterov=bool(cfg.optimizer.nesterov),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    
def build_scheduler(cfg, optimizer: optim.Optimizer):
    t = cfg.lr_schedule.type.lower()
    epochs = int(cfg.training.epochs)
    
    if t == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    if t == "step":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    
    raise ValueError(f"Unknown lr_schedule type: {cfg.lr_schedule.type}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--model", type=str, required=True, help="e.g., mlp or cnn_tiny")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--timestamped_run_dir", action="store_true", help="Create a timestamped subfolder.")
    return p.parse_args()

def main():
    args = parse_args()
    
    cfg = load_config(args.config)
    validate_minimal(cfg)
    
    set_seed(args.seed, deterministic=True)
    
    device = torch.device(args.device)
    
    train_loader, val_loader, test_loader, split_info = get_loaders(cfg, seed=args.seed)
    
    model = build_model(args.model, cfg).to(device)
    n_params = count_trainable_params(model)
    
    run_dir = make_run_dir(
        runs_dir=cfg.runs.dir,
        name_format=cfg.runs.name_format,
        model=args.model,
        seed=args.seed,
        add_timestamp=bool(args.timestamped_run_dir),
    )
    
    save_yaml(cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg), run_dir / "config.yaml")
    save_json(split_info, run_dir / "split_indices.json")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    
    metrics_path = run_dir / "metrics.jsonl"
    
    best_val_acc = -1.0
    best_epoch = -1
    
    epochs = int(cfg.training.epochs)
    
    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        train_metrics = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, device, criterion)
        
        scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]
        
        record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "model": args.model,
            "seed": args.seed,
            "trainable_params": n_params,
        }
        append_jsonl(record, metrics_path)
        
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
    
    test_metrics = evaluate(model, test_loader, device, criterion)
    
    final = {
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "epochs": epochs,
        "trainable_params": n_params,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_test_acc": test_metrics["acc"],
        "final_test_loss": test_metrics["loss"],
    }
    save_json(final, run_dir / "final.json")
    
    print(f"[DONE] Run saved to: {run_dir}")
    print(final)
    
if __name__ == "__main__":
    main()