"""
Utility functions for Novartis Datathon 2025.

Core helpers: seeding, logging, timing, configuration loading.
"""

import os
import random
import logging
import time
import yaml
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Any, Optional

import numpy as np

# Optional torch import for reproducibility
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy, and optionally torch for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device() -> Optional[Any]:
    """
    Get CUDA device if available, else CPU.
    
    Returns:
        torch.device if torch is available, else None
    """
    if HAS_TORCH:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return None


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for console and optional file output.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    # Map string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Create handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


@contextmanager
def timer(name: str) -> Generator:
    """
    Context manager to time code blocks.
    
    Args:
        name: Name of the operation being timed
        
    Usage:
        with timer("Feature engineering"):
            features = make_features(panel, scenario="scenario1")
    """
    logger = logging.getLogger(__name__)
    t0 = time.time()
    logger.info(f"[{name}] starting...")
    try:
        yield
    finally:
        elapsed = time.time() - t0
        logger.info(f"[{name}] done in {elapsed:.3f}s")


def load_config(path: str) -> dict:
    """
    Load YAML configuration file.
    
    Args:
        path: Path to YAML config file (relative or absolute)
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(path)
    
    # Try relative to project root if not found
    if not config_path.exists():
        project_root = get_project_root()
        config_path = project_root / path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_path(config: dict, key: str) -> Path:
    """
    Resolve paths from nested config keys.
    
    Args:
        config: Configuration dictionary
        key: Dot-separated key path (e.g., 'paths.raw_dir')
        
    Returns:
        Path object
        
    Example:
        >>> config = load_config('configs/data.yaml')
        >>> raw_dir = get_path(config, 'paths.raw_dir')
    """
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"Key '{key}' not found in config")
    
    return Path(value)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root (parent of src/)
    """
    return Path(__file__).parent.parent


def is_colab() -> bool:
    """
    Check if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise
    """
    return Path("/content/drive").exists()


def resolve_path(path_str: str, drive_base: str = None) -> Path:
    """
    Resolve path, handling potential Google Drive prefixes.
    
    Args:
        path_str: Path string, may contain ${drive.base_path}
        drive_base: Base path for Google Drive
        
    Returns:
        Resolved Path object
    """
    if drive_base and "${drive.base_path}" in path_str:
        return Path(path_str.replace("${drive.base_path}", drive_base))
    return Path(path_str)
