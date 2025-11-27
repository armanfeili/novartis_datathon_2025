import os
import random
import numpy as np
import torch
import logging
import time
import yaml
from pathlib import Path
from contextlib import contextmanager

def set_seed(seed: int = 42):
    """Set random seed across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device() -> torch.device:
    """Get CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_path: str = None, level=logging.INFO):
    """Configure logging to console and optional file."""
    handlers = [logging.StreamHandler()]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    # Remove existing handlers to avoid duplicates
    logging.getLogger().handlers = []
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

@contextmanager
def timer(name: str):
    """Context manager to time a block of code."""
    t0 = time.time()
    logging.info(f"[{name}] starting...")
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def resolve_path(path_str: str, drive_base: str = None) -> Path:
    """Resolve path, handling potential Google Drive prefixes."""
    if drive_base and "${drive.base_path}" in path_str:
        return Path(path_str.replace("${drive.base_path}", drive_base))
    return Path(path_str)
