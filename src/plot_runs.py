import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    
    return rows

def find_latest_run_dir(run_root: Path) -> Path:
    if (run_root / "metrics.jsonl").exists():
        return run_root
    
    subdirs = [p for p in run_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectories found in {run_root}")
    
    subdirs_sorted = sorted(subdirs, key=lambda p: p.name)
    latest = subdirs_sorted[-1]
    if not (latest / "metrics.jsonl").exists():
        raise FileNotFoundError(f"metrics.jsonl not found in latest dir: {latest}")
    
    return latest

def plot_learning_curves(run_dir: Path, out_dir: Path) -> None:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    final = read_json(run_dir / "final.json")
    
    epochs = [r["epoch"] for r in rows]
    
    train_acc = [r["train_acc"] for r in rows]
    val_acc = [r["val_acc"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    
    model = final.get("model", run_dir.parent.name)
    seed = final.get("seed", "unknown")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{model} (seed={seed}) accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model}_seed{seed}_acc.png")
    plt.close()
    
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{model} (seed={seed}) loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model}_seed{seed}_loss.png")
    plt.close()
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--run_roots", nargs="+", required=True, help="Run roots like runs/mlp-0 runs/cnn_tiny-0 (will auto-pick latest timestamped subdir).")
    p.add_argument("--out_dir", type=str, default="runs/plots")
    args = p.parse_args()
    
    out_dir = Path(args.out_dir)
    for rr in args.run_roots:
        run_root = Path(rr)
        run_dir = find_latest_run_dir(run_root)
        plot_learning_curves(run_dir, out_dir)
        print(f"[saved plots] {run_dir} -> {out_dir}")
        
if __name__ == "__main__":
    main()