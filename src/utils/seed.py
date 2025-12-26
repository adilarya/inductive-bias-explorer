import os
import random
from typing import Optional

import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True) -> None:
    if seed is None:
        raise ValueError("Seed must be an int, not None.")
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g