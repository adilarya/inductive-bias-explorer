import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PathLike = Union[str, os.PathLike, Path] 

def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_run_name(name_format: str, model: str, seed: int) -> str:
    return name_format.format(model=model, seed=seed)

def make_run_dir(
    runs_dir: PathLike,
    name_format: str,
    model: str,
    seed: int, 
    add_timestamp: bool = True,
) -> Path:
    base = ensure_dir(runs_dir)
    run_name = format_run_name(name_format, model=model, seed=seed)
    run_root = ensure_dir(base / run_name)
    
    if add_timestamp:
        run_path = ensure_dir(run_root / timestamp())
    else:
        run_path = run_root
        
    return run_path

def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj

def save_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, sort_keys=True, default=_to_serializable)
        
def load_json(path: PathLike) -> Any:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)
    
def save_yaml(obj: Any, path: PathLike) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open('w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False)
        
def load_yaml(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {p} must be a mapping/dict at top level.")
    return data

def append_jsonl(record: Dict[str, Any], path: PathLike) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=_to_serializable) + '\n')