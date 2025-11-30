"""
Utility functions for Novartis Datathon 2025.

Core helpers: seeding, logging, timing, configuration loading,
GPU detection, memory management, progress tracking, and performance optimization.
"""

import gc
import os
import random
import logging
import time
import yaml
import sys
import platform
import psutil
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Any, Optional, Dict, List, Tuple, Callable
from functools import wraps

import numpy as np
import pandas as pd

# Optional torch import for reproducibility
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional tqdm import for progress bars
try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as tqdm_auto
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)


# =============================================================================
# Section 11.1: Environment Detection
# =============================================================================

def is_colab() -> bool:
    """
    Check if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise
    """
    return 'google.colab' in sys.modules or Path("/content/drive").exists()


def is_jupyter() -> bool:
    """
    Check if running in a Jupyter notebook environment.
    
    Returns:
        True if running in Jupyter, False otherwise
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except (NameError, AttributeError, ModuleNotFoundError, ImportError):
        return False


def get_platform_info() -> Dict[str, Any]:
    """
    Get comprehensive platform information.
    
    Returns:
        Dictionary with platform details
    """
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'is_colab': is_colab(),
        'is_jupyter': is_jupyter(),
    }
    
    # Memory info
    try:
        mem = psutil.virtual_memory()
        info['total_ram_gb'] = round(mem.total / (1024**3), 2)
        info['available_ram_gb'] = round(mem.available / (1024**3), 2)
    except Exception:
        info['total_ram_gb'] = None
        info['available_ram_gb'] = None
    
    # CPU info
    try:
        info['cpu_count'] = os.cpu_count()
    except Exception:
        info['cpu_count'] = None
    
    return info


# =============================================================================
# Section 11.1: GPU Detection and Utilization
# =============================================================================

def get_device() -> Optional[Any]:
    """
    Get the best available device (CUDA GPU, MPS, or CPU).
    
    Returns:
        torch.device if torch is available, else None
    """
    if not HAS_TORCH:
        return None
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed GPU information.
    
    Returns:
        Dictionary with GPU details, empty dict if no GPU
    """
    info = {
        'gpu_available': False,
        'device_name': None,
        'total_memory_gb': None,
        'memory_allocated_gb': None,
        'memory_cached_gb': None,
        'cuda_version': None,
        'mps_available': False,
    }
    
    if not HAS_TORCH:
        return info
    
    if torch.cuda.is_available():
        info['gpu_available'] = True
        info['device_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['total_memory_gb'] = round(props.total_memory / (1024**3), 2)
        info['memory_allocated_gb'] = round(torch.cuda.memory_allocated(0) / (1024**3), 3)
        info['memory_cached_gb'] = round(torch.cuda.memory_reserved(0) / (1024**3), 3)
        info['cuda_version'] = torch.version.cuda
        info['compute_capability'] = f"{props.major}.{props.minor}"
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['mps_available'] = True
        if not info['gpu_available']:
            info['gpu_available'] = True
            info['device_name'] = "Apple Silicon (MPS)"
    
    return info


def print_gpu_info() -> None:
    """Print formatted GPU information."""
    info = get_gpu_info()
    
    if info['gpu_available']:
        print(f"🚀 GPU Available: {info['device_name']}")
        if info['total_memory_gb']:
            print(f"   Total Memory: {info['total_memory_gb']:.1f} GB")
            print(f"   Allocated: {info['memory_allocated_gb']:.3f} GB")
            print(f"   Cached: {info['memory_cached_gb']:.3f} GB")
        if info['cuda_version']:
            print(f"   CUDA Version: {info['cuda_version']}")
        if info['mps_available']:
            print(f"   MPS Backend: Available")
    else:
        print("❌ No GPU available - using CPU")


def enable_gpu_for_catboost() -> Dict[str, Any]:
    """
    Get CatBoost GPU configuration if GPU is available.
    
    Returns:
        Dictionary with CatBoost GPU parameters
    """
    gpu_info = get_gpu_info()
    
    if gpu_info['gpu_available'] and gpu_info.get('cuda_version'):
        return {
            'task_type': 'GPU',
            'devices': '0',
        }
    return {}


def enable_gpu_for_xgboost() -> Dict[str, Any]:
    """
    Get XGBoost GPU configuration if GPU is available.
    
    Returns:
        Dictionary with XGBoost GPU parameters
    """
    gpu_info = get_gpu_info()
    
    if gpu_info['gpu_available'] and gpu_info.get('cuda_version'):
        return {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
        }
    return {}


def enable_gpu_for_lightgbm() -> Dict[str, Any]:
    """
    Get LightGBM GPU configuration if GPU is available.
    
    Returns:
        Dictionary with LightGBM GPU parameters
    """
    gpu_info = get_gpu_info()
    
    if gpu_info['gpu_available'] and gpu_info.get('cuda_version'):
        return {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        }
    return {}


# =============================================================================
# Section 11.1: Memory Management
# =============================================================================

def clear_memory() -> Dict[str, float]:
    """
    Clear memory by running garbage collection and clearing GPU cache.
    
    Returns:
        Dictionary with memory freed information
    """
    result = {'gc_collected': 0, 'gpu_cache_cleared': False}
    
    # Python garbage collection
    result['gc_collected'] = gc.collect()
    
    # Clear GPU cache
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        result['gpu_cache_cleared'] = True
    
    logger.debug(f"Memory cleared: GC collected {result['gc_collected']} objects")
    return result


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in GB
    """
    info = {}
    
    try:
        process = psutil.Process()
        info['process_rss_gb'] = round(process.memory_info().rss / (1024**3), 3)
        info['process_vms_gb'] = round(process.memory_info().vms / (1024**3), 3)
        
        mem = psutil.virtual_memory()
        info['system_total_gb'] = round(mem.total / (1024**3), 2)
        info['system_available_gb'] = round(mem.available / (1024**3), 2)
        info['system_used_percent'] = mem.percent
    except Exception as e:
        logger.warning(f"Failed to get memory info: {e}")
    
    if HAS_TORCH and torch.cuda.is_available():
        info['gpu_allocated_gb'] = round(torch.cuda.memory_allocated(0) / (1024**3), 3)
        info['gpu_cached_gb'] = round(torch.cuda.memory_reserved(0) / (1024**3), 3)
    
    return info


def log_memory_usage(prefix: str = "") -> None:
    """
    Log current memory usage.
    
    Args:
        prefix: Optional prefix for log message
    """
    mem = get_memory_usage()
    prefix_str = f"[{prefix}] " if prefix else ""
    
    logger.info(
        f"{prefix_str}Memory: Process={mem.get('process_rss_gb', 'N/A')} GB, "
        f"System={mem.get('system_used_percent', 'N/A')}% used"
    )
    
    if 'gpu_allocated_gb' in mem:
        logger.info(f"{prefix_str}GPU: Allocated={mem['gpu_allocated_gb']} GB, Cached={mem['gpu_cached_gb']} GB")


@contextmanager
def memory_monitor(name: str, log_before: bool = True, log_after: bool = True, 
                   clear_after: bool = False) -> Generator:
    """
    Context manager to monitor memory usage of a code block.
    
    Args:
        name: Name of the operation being monitored
        log_before: Log memory before operation
        log_after: Log memory after operation
        clear_after: Run garbage collection after operation
    """
    if log_before:
        log_memory_usage(f"{name} - Before")
    
    try:
        yield
    finally:
        if clear_after:
            clear_memory()
        if log_after:
            log_memory_usage(f"{name} - After")


def optimize_dataframe_memory(df: pd.DataFrame, 
                              deep_copy: bool = False,
                              categorical_threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimize DataFrame memory usage by downcasting types.
    
    Args:
        df: Input DataFrame
        deep_copy: Whether to create a deep copy
        categorical_threshold: Convert object columns to category if unique ratio < threshold
        
    Returns:
        Tuple of (optimized DataFrame, optimization stats)
    """
    if deep_copy:
        df = df.copy()
    
    stats = {
        'original_memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'columns_optimized': 0,
    }
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimize integers
        if col_type in ['int64', 'int32']:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)
            stats['columns_optimized'] += 1
        
        # Optimize floats (carefully - preserve precision for important columns)
        elif col_type == 'float64':
            # Only downcast if precision loss is acceptable
            df[col] = pd.to_numeric(df[col], downcast='float')
            stats['columns_optimized'] += 1
        
        # Convert low-cardinality objects to category
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df)
            if num_total > 0 and (num_unique / num_total) < categorical_threshold:
                df[col] = df[col].astype('category')
                stats['columns_optimized'] += 1
    
    stats['optimized_memory_mb'] = df.memory_usage(deep=True).sum() / (1024**2)
    stats['memory_reduction_percent'] = round(
        (1 - stats['optimized_memory_mb'] / stats['original_memory_mb']) * 100, 1
    ) if stats['original_memory_mb'] > 0 else 0
    
    return df, stats


# =============================================================================
# Section 11.1: Progress Bars
# =============================================================================

def get_progress_bar(iterable: Any = None, total: int = None, desc: str = None,
                     disable: bool = False, **kwargs) -> Any:
    """
    Get a progress bar wrapper (tqdm if available, else passthrough).
    
    Args:
        iterable: Iterable to wrap
        total: Total number of iterations
        desc: Description to display
        disable: If True, disable progress bar
        **kwargs: Additional arguments passed to tqdm
        
    Returns:
        tqdm progress bar or iterable
    """
    if not HAS_TQDM or disable:
        return iterable if iterable is not None else range(total or 0)
    
    # Use auto tqdm for notebook compatibility
    return tqdm_auto(iterable, total=total, desc=desc, **kwargs)


def progress_wrapper(func: Callable) -> Callable:
    """
    Decorator to add progress tracking to a function.
    
    The decorated function should yield progress updates as (current, total, message) tuples.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        show_progress = kwargs.pop('show_progress', True)
        
        if not HAS_TQDM or not show_progress:
            # Just consume the generator
            result = None
            for item in func(*args, **kwargs):
                if isinstance(item, tuple) and len(item) >= 2:
                    result = item[-1] if len(item) > 2 else None
                else:
                    result = item
            return result
        
        pbar = None
        result = None
        
        for item in func(*args, **kwargs):
            if isinstance(item, tuple) and len(item) >= 2:
                current, total = item[:2]
                message = item[2] if len(item) > 2 else None
                
                if pbar is None:
                    pbar = tqdm_auto(total=total)
                
                pbar.n = current
                pbar.refresh()
                
                if message:
                    pbar.set_description(str(message))
                
                result = item[-1] if len(item) > 2 else None
            else:
                result = item
        
        if pbar:
            pbar.close()
        
        return result
    
    return wrapper


# =============================================================================
# Core Functions (existing)
# =============================================================================

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
    # Handle both string and int level inputs
    if isinstance(level, int):
        log_level = level
    else:
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


# =============================================================================
# Section 11.3: Performance Optimization & Caching
# =============================================================================

def lazy_load_dataframe(path: Path, 
                        use_cache: bool = True,
                        optimize_memory: bool = True,
                        **read_kwargs) -> pd.DataFrame:
    """
    Lazily load a DataFrame with optional memory optimization.
    
    Args:
        path: Path to the data file (CSV or Parquet)
        use_cache: If True and parquet cache exists, use it
        optimize_memory: If True, optimize memory after loading
        **read_kwargs: Additional arguments passed to pandas read function
        
    Returns:
        Loaded DataFrame
    """
    path = Path(path)
    cache_path = path.with_suffix('.parquet') if path.suffix == '.csv' else None
    
    # Try to load from parquet cache
    if use_cache and cache_path and cache_path.exists():
        logger.info(f"Loading from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        # Load from original file
        if path.suffix == '.csv':
            df = pd.read_csv(path, **read_kwargs)
        elif path.suffix in ['.parquet', '.pq']:
            df = pd.read_parquet(path, **read_kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Optionally save to parquet cache
        if use_cache and cache_path and path.suffix == '.csv':
            try:
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached to: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache: {e}")
    
    # Optimize memory
    if optimize_memory:
        df, stats = optimize_dataframe_memory(df)
        logger.debug(f"Memory optimized: {stats['memory_reduction_percent']}% reduction")
    
    return df


def chunked_apply(df: pd.DataFrame, 
                  func: Callable, 
                  chunk_size: int = 10000,
                  show_progress: bool = True,
                  **func_kwargs) -> pd.DataFrame:
    """
    Apply a function to a DataFrame in chunks to manage memory.
    
    Args:
        df: Input DataFrame
        func: Function to apply (should accept DataFrame and return DataFrame)
        chunk_size: Size of each chunk
        show_progress: Show progress bar
        **func_kwargs: Additional arguments passed to func
        
    Returns:
        Concatenated result DataFrame
    """
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    chunks = []
    
    iterator = range(0, len(df), chunk_size)
    if show_progress and HAS_TQDM:
        iterator = tqdm_auto(iterator, total=n_chunks, desc="Processing chunks")
    
    for start in iterator:
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end].copy()
        result_chunk = func(chunk, **func_kwargs)
        chunks.append(result_chunk)
        
        # Clear memory after each chunk
        del chunk
        gc.collect()
    
    return pd.concat(chunks, ignore_index=True)


def get_optimal_n_jobs(memory_per_job_gb: float = 1.0, 
                       max_jobs: int = None) -> int:
    """
    Calculate optimal number of parallel jobs based on available memory.
    
    Args:
        memory_per_job_gb: Estimated memory usage per job in GB
        max_jobs: Maximum number of jobs (defaults to CPU count)
        
    Returns:
        Optimal number of jobs
    """
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        cpu_count = os.cpu_count() or 1
        
        # Calculate based on memory
        mem_based = max(1, int(available_gb / memory_per_job_gb))
        
        # Cap at CPU count
        result = min(mem_based, cpu_count)
        
        # Apply max_jobs limit if specified
        if max_jobs:
            result = min(result, max_jobs)
        
        return result
    except Exception:
        return 1


# =============================================================================
# Section 11.2: Environment Verification
# =============================================================================

def verify_environment(raise_on_error: bool = False) -> Dict[str, Any]:
    """
    Verify that the environment has all required dependencies.
    
    Args:
        raise_on_error: If True, raise exception on missing dependencies
        
    Returns:
        Dictionary with verification results
    """
    results = {
        'python_version': sys.version_info[:2],
        'python_ok': sys.version_info >= (3, 8),
        'packages': {},
        'all_ok': True,
    }
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('pyyaml', 'yaml'),
        ('tqdm', 'tqdm'),
    ]
    
    optional_packages = [
        ('torch', 'torch'),
        ('catboost', 'catboost'),
        ('lightgbm', 'lightgbm'),
        ('xgboost', 'xgboost'),
        ('pyarrow', 'pyarrow'),
        ('optuna', 'optuna'),
        ('mlflow', 'mlflow'),
        ('wandb', 'wandb'),
    ]
    
    # Check required packages
    for name, import_name in required_packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            results['packages'][name] = {'installed': True, 'version': version, 'required': True}
        except Exception:  # Catch all exceptions including OSError from native libs
            results['packages'][name] = {'installed': False, 'required': True}
            results['all_ok'] = False
    
    # Check optional packages
    for name, import_name in optional_packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            results['packages'][name] = {'installed': True, 'version': version, 'required': False}
        except Exception:  # Catch all exceptions including OSError from native libs
            results['packages'][name] = {'installed': False, 'required': False}
    
    if raise_on_error and not results['all_ok']:
        missing = [k for k, v in results['packages'].items() if v['required'] and not v['installed']]
        raise ImportError(f"Missing required packages: {missing}")
    
    return results


def print_environment_info() -> None:
    """Print comprehensive environment information."""
    print("=" * 60)
    print("🔧 Environment Information")
    print("=" * 60)
    
    # Platform
    platform_info = get_platform_info()
    print(f"\n📋 Platform:")
    print(f"   Python: {platform_info['python_version'].split()[0]}")
    print(f"   OS: {platform_info['platform']}")
    print(f"   Machine: {platform_info['machine']}")
    print(f"   CPUs: {platform_info['cpu_count']}")
    print(f"   RAM: {platform_info['total_ram_gb']} GB total, {platform_info['available_ram_gb']} GB available")
    
    if platform_info['is_colab']:
        print(f"   🌐 Running in Google Colab")
    if platform_info['is_jupyter']:
        print(f"   📓 Running in Jupyter Notebook")
    
    # GPU
    print(f"\n🖥️  GPU:")
    print_gpu_info()
    
    # Packages
    env_check = verify_environment()
    print(f"\n📦 Packages:")
    for name, info in env_check['packages'].items():
        status = "✅" if info['installed'] else "❌"
        required = "(required)" if info['required'] else "(optional)"
        version = info.get('version', 'N/A') if info['installed'] else 'not installed'
        print(f"   {status} {name}: {version} {required}")
    
    print("=" * 60)


# =============================================================================
# Section 11: Colab-specific Utilities
# =============================================================================

def mount_google_drive(force_remount: bool = False) -> Optional[Path]:
    """
    Mount Google Drive in Colab environment.
    
    Args:
        force_remount: If True, force remount even if already mounted
        
    Returns:
        Path to mounted drive, or None if not in Colab
    """
    if not is_colab():
        logger.warning("Not running in Colab - Drive mount not available")
        return None
    
    drive_path = Path("/content/drive/MyDrive")
    
    if drive_path.exists() and not force_remount:
        logger.info("Google Drive already mounted")
        return drive_path
    
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=force_remount)
        logger.info("Google Drive mounted successfully")
        return drive_path
    except Exception as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        return None


def sync_to_drive() -> bool:
    """
    Sync local changes to Google Drive (Colab only).
    
    Returns:
        True if sync successful, False otherwise
    """
    if not is_colab():
        logger.warning("Not running in Colab - sync not available")
        return False
    
    try:
        from google.colab import drive
        drive.flush_and_unmount()
        drive.mount('/content/drive')
        logger.info("Synced to Google Drive")
        return True
    except Exception as e:
        logger.error(f"Failed to sync: {e}")
        return False


def download_file(filepath: str) -> bool:
    """
    Download a file to local machine (Colab only).
    
    Args:
        filepath: Path to file to download
        
    Returns:
        True if download initiated, False otherwise
    """
    if not is_colab():
        logger.info(f"Not in Colab - file available at: {filepath}")
        return False
    
    try:
        from google.colab import files
        files.download(filepath)
        logger.info(f"Download initiated: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return False


def upload_files(target_dir: str = None) -> Dict[str, bytes]:
    """
    Upload files from local machine (Colab only).
    
    Args:
        target_dir: Optional directory to save uploaded files
        
    Returns:
        Dictionary of filename -> file content
    """
    if not is_colab():
        logger.warning("Not running in Colab - upload not available")
        return {}
    
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if target_dir:
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            for filename, content in uploaded.items():
                (target_path / filename).write_bytes(content)
                logger.info(f"Saved: {target_path / filename}")
        
        return uploaded
    except Exception as e:
        logger.error(f"Failed to upload: {e}")
        return {}


# =============================================================================
# Section 12: Competition Strategy Utilities
# =============================================================================

class SubmissionTracker:
    """
    Track all submissions with scores and notes for leaderboard management.
    
    This class helps:
    - Track all submissions with timestamps, scores, and notes
    - Analyze score variance between local CV and LB
    - Identify potential overfitting to leaderboard
    - Save submissions for final selection
    
    Example usage:
        tracker = SubmissionTracker(log_path='submissions/submission_tracker.json')
        tracker.log_submission(
            submission_path='submissions/v1/submission.csv',
            cv_score=0.15,
            lb_score=0.18,
            model_info={'type': 'catboost', 'params': {...}},
            notes='First CatBoost submission'
        )
        analysis = tracker.analyze_cv_lb_variance()
    """
    
    def __init__(self, log_path: str = 'submissions/submission_tracker.json'):
        """
        Initialize submission tracker.
        
        Args:
            log_path: Path to JSON file for tracking submissions
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.submissions: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self) -> None:
        """Load existing submissions from log file."""
        import json
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    self.submissions = json.load(f)
                logger.info(f"Loaded {len(self.submissions)} submissions from {self.log_path}")
            except Exception as e:
                logger.warning(f"Could not load submissions: {e}")
                self.submissions = []
    
    def _save(self) -> None:
        """Save submissions to log file."""
        import json
        with open(self.log_path, 'w') as f:
            json.dump(self.submissions, f, indent=2, default=str)
    
    def log_submission(
        self,
        submission_path: str,
        cv_score: Optional[float] = None,
        lb_score: Optional[float] = None,
        scenario: Optional[int] = None,
        model_info: Optional[Dict[str, Any]] = None,
        notes: str = ''
    ) -> Dict[str, Any]:
        """
        Log a submission with scores and notes.
        
        Args:
            submission_path: Path to submission file
            cv_score: Local cross-validation score
            lb_score: Leaderboard score (can be updated later)
            scenario: Scenario number (1 or 2)
            model_info: Model configuration and parameters
            notes: Free-form notes about this submission
            
        Returns:
            The submission record
        """
        from datetime import datetime
        
        record = {
            'id': len(self.submissions) + 1,
            'timestamp': datetime.now().isoformat(),
            'submission_path': str(submission_path),
            'cv_score': cv_score,
            'lb_score': lb_score,
            'scenario': scenario,
            'model_info': model_info or {},
            'notes': notes,
            'cv_lb_gap': None
        }
        
        if cv_score is not None and lb_score is not None:
            record['cv_lb_gap'] = lb_score - cv_score
        
        self.submissions.append(record)
        self._save()
        
        logger.info(f"Logged submission #{record['id']}: CV={cv_score}, LB={lb_score}")
        return record
    
    def update_lb_score(self, submission_id: int, lb_score: float) -> None:
        """
        Update leaderboard score for a submission.
        
        Args:
            submission_id: ID of submission to update
            lb_score: Leaderboard score to set
        """
        for sub in self.submissions:
            if sub['id'] == submission_id:
                sub['lb_score'] = lb_score
                if sub['cv_score'] is not None:
                    sub['cv_lb_gap'] = lb_score - sub['cv_score']
                self._save()
                logger.info(f"Updated submission #{submission_id} LB score: {lb_score}")
                return
        
        logger.warning(f"Submission #{submission_id} not found")
    
    def analyze_cv_lb_variance(self) -> Dict[str, Any]:
        """
        Analyze score variance between local CV and leaderboard.
        
        Returns:
            Dictionary with variance analysis
        """
        submissions_with_both = [
            s for s in self.submissions 
            if s['cv_score'] is not None and s['lb_score'] is not None
        ]
        
        if len(submissions_with_both) == 0:
            return {
                'n_submissions': 0,
                'mean_gap': None,
                'std_gap': None,
                'correlation': None,
                'overfitting_warning': False
            }
        
        cv_scores = [s['cv_score'] for s in submissions_with_both]
        lb_scores = [s['lb_score'] for s in submissions_with_both]
        gaps = [s['cv_lb_gap'] for s in submissions_with_both]
        
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps) if len(gaps) > 1 else 0.0
        
        # Compute correlation between CV and LB
        if len(cv_scores) > 1:
            correlation = np.corrcoef(cv_scores, lb_scores)[0, 1]
        else:
            correlation = None
        
        # Check for overfitting: LB consistently worse than CV
        overfitting_warning = mean_gap > 0.05 and len(gaps) >= 3
        
        analysis = {
            'n_submissions': len(submissions_with_both),
            'mean_gap': float(mean_gap),
            'std_gap': float(std_gap),
            'correlation': float(correlation) if correlation is not None else None,
            'overfitting_warning': overfitting_warning,
            'best_cv': min(cv_scores),
            'best_lb': min(lb_scores),
            'most_recent_gap': gaps[-1] if gaps else None
        }
        
        if overfitting_warning:
            logger.warning(
                f"Potential overfitting detected! Mean CV-LB gap: {mean_gap:.4f}. "
                f"Consider using simpler models or more regularization."
            )
        
        return analysis
    
    def get_best_submission(self, by: str = 'lb_score') -> Optional[Dict[str, Any]]:
        """
        Get the best submission by specified metric.
        
        Args:
            by: 'lb_score' or 'cv_score'
            
        Returns:
            Best submission record, or None if no valid submissions
        """
        valid = [s for s in self.submissions if s.get(by) is not None]
        if not valid:
            return None
        return min(valid, key=lambda x: x[by])
    
    def get_submissions_df(self) -> pd.DataFrame:
        """
        Get all submissions as a DataFrame for analysis.
        
        Returns:
            DataFrame with all submissions
        """
        if not self.submissions:
            return pd.DataFrame()
        return pd.DataFrame(self.submissions)
    
    def identify_overfitting_submissions(self, gap_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Identify submissions that may be overfitting (large CV-LB gap).
        
        Args:
            gap_threshold: Maximum acceptable CV-LB gap
            
        Returns:
            List of potentially overfitting submissions
        """
        return [
            s for s in self.submissions
            if s['cv_lb_gap'] is not None and s['cv_lb_gap'] > gap_threshold
        ]


class TimeAllocationTracker:
    """
    Track time allocation across different competition phases.
    
    Recommended allocation:
    - EDA: 20%
    - Feature Engineering: 25%
    - Modeling: 30%
    - Tuning/Ensemble: 15%
    - Documentation: 10%
    
    Example usage:
        tracker = TimeAllocationTracker()
        tracker.start_phase('eda')
        # ... do EDA work ...
        tracker.end_phase()
        summary = tracker.get_summary()
    """
    
    RECOMMENDED_ALLOCATION = {
        'eda': 0.20,
        'feature_engineering': 0.25,
        'modeling': 0.30,
        'tuning': 0.15,
        'documentation': 0.10
    }
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize time tracker.
        
        Args:
            log_path: Optional path to persist time logs
        """
        self.log_path = Path(log_path) if log_path else None
        self.time_logs: List[Dict[str, Any]] = []
        self.current_phase: Optional[str] = None
        self.phase_start: Optional[float] = None
        
        if self.log_path and self.log_path.exists():
            self._load()
    
    def _load(self) -> None:
        """Load existing time logs."""
        import json
        try:
            with open(self.log_path, 'r') as f:
                self.time_logs = json.load(f)
        except Exception:
            self.time_logs = []
    
    def _save(self) -> None:
        """Save time logs to file."""
        if self.log_path:
            import json
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w') as f:
                json.dump(self.time_logs, f, indent=2, default=str)
    
    def start_phase(self, phase: str) -> None:
        """
        Start tracking time for a phase.
        
        Args:
            phase: Phase name (eda, feature_engineering, modeling, tuning, documentation)
        """
        if self.current_phase is not None:
            self.end_phase()
        
        self.current_phase = phase.lower()
        self.phase_start = time.time()
        logger.info(f"Started phase: {phase}")
    
    def end_phase(self) -> float:
        """
        End tracking for current phase.
        
        Returns:
            Duration in hours
        """
        if self.current_phase is None or self.phase_start is None:
            return 0.0
        
        duration_hours = (time.time() - self.phase_start) / 3600
        
        from datetime import datetime
        self.time_logs.append({
            'phase': self.current_phase,
            'duration_hours': duration_hours,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Ended phase: {self.current_phase}, duration: {duration_hours:.2f} hours")
        
        self.current_phase = None
        self.phase_start = None
        self._save()
        
        return duration_hours
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of time allocation.
        
        Returns:
            Dictionary with time allocation summary
        """
        phase_totals: Dict[str, float] = {}
        for log in self.time_logs:
            phase = log['phase']
            phase_totals[phase] = phase_totals.get(phase, 0) + log['duration_hours']
        
        total_hours = sum(phase_totals.values())
        
        # Calculate actual percentages
        actual_pct = {}
        for phase, hours in phase_totals.items():
            actual_pct[phase] = hours / total_hours if total_hours > 0 else 0
        
        # Compare with recommended
        deviations = {}
        for phase, recommended in self.RECOMMENDED_ALLOCATION.items():
            actual = actual_pct.get(phase, 0)
            deviations[phase] = {
                'recommended': recommended,
                'actual': actual,
                'deviation': actual - recommended
            }
        
        return {
            'total_hours': total_hours,
            'phase_hours': phase_totals,
            'phase_percentages': actual_pct,
            'deviations': deviations
        }


def create_backup_submission(
    submission_path: str,
    backup_dir: str = 'submissions/backups',
    model_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Create a backup of submission and optionally the model.
    
    Args:
        submission_path: Path to submission CSV
        backup_dir: Directory for backups
        model_path: Optional path to model file
        
    Returns:
        Dictionary with paths to backup files
    """
    import shutil
    from datetime import datetime
    
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result = {}
    
    # Backup submission
    submission_src = Path(submission_path)
    if submission_src.exists():
        submission_dst = backup_path / f"submission_{timestamp}.csv"
        shutil.copy2(submission_src, submission_dst)
        result['submission'] = str(submission_dst)
        logger.info(f"Backed up submission to {submission_dst}")
    
    # Backup model if provided
    if model_path:
        model_src = Path(model_path)
        if model_src.exists():
            model_dst = backup_path / f"model_{timestamp}{model_src.suffix}"
            shutil.copy2(model_src, model_dst)
            result['model'] = str(model_dst)
            logger.info(f"Backed up model to {model_dst}")
    
    return result


def validate_submission_completeness(
    submission_df: pd.DataFrame,
    template_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate that submission covers all required series and months.
    
    Args:
        submission_df: Generated submission
        template_df: Official template
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_complete': True,
        'missing_rows': 0,
        'extra_rows': 0,
        'missing_series': [],
        'missing_months': []
    }
    
    # Key columns
    key_cols = ['country', 'brand_name', 'months_postgx']
    
    # Create key tuples
    submission_keys = set(submission_df[key_cols].apply(tuple, axis=1))
    template_keys = set(template_df[key_cols].apply(tuple, axis=1))
    
    # Check for missing
    missing = template_keys - submission_keys
    extra = submission_keys - template_keys
    
    results['missing_rows'] = len(missing)
    results['extra_rows'] = len(extra)
    
    if missing:
        results['is_complete'] = False
        # Identify missing series
        missing_series = set((k[0], k[1]) for k in missing)
        results['missing_series'] = list(missing_series)[:10]  # Limit to 10
        # Identify missing months
        missing_months = set(k[2] for k in missing)
        results['missing_months'] = sorted(missing_months)
        
        logger.warning(f"Submission incomplete: {len(missing)} rows missing")
    
    if extra:
        logger.warning(f"Submission has {len(extra)} extra rows not in template")
    
    return results


def check_prediction_sanity(
    predictions_df: pd.DataFrame,
    volume_col: str = 'volume'
) -> Dict[str, Any]:
    """
    Perform sanity checks on predictions.
    
    Checks:
    - No negative values
    - No NaN/Inf values
    - Reasonable range (not all zeros, not extreme values)
    - Reasonable erosion pattern (volume should generally decrease)
    
    Args:
        predictions_df: DataFrame with predictions
        volume_col: Name of volume column
        
    Returns:
        Dictionary with sanity check results
    """
    volume = predictions_df[volume_col]
    
    results = {
        'is_sane': True,
        'issues': [],
        'statistics': {
            'mean': float(volume.mean()),
            'std': float(volume.std()),
            'min': float(volume.min()),
            'max': float(volume.max()),
            'pct_zero': float((volume == 0).mean() * 100),
            'pct_negative': float((volume < 0).mean() * 100)
        }
    }
    
    # Check for negative values
    n_negative = (volume < 0).sum()
    if n_negative > 0:
        results['is_sane'] = False
        results['issues'].append(f"{n_negative} negative predictions")
    
    # Check for NaN/Inf
    n_nan = volume.isna().sum()
    n_inf = np.isinf(volume).sum()
    if n_nan > 0:
        results['is_sane'] = False
        results['issues'].append(f"{n_nan} NaN predictions")
    if n_inf > 0:
        results['is_sane'] = False
        results['issues'].append(f"{n_inf} Inf predictions")
    
    # Check for all zeros
    pct_zero = (volume == 0).mean()
    if pct_zero > 0.5:
        results['issues'].append(f"Warning: {pct_zero*100:.1f}% of predictions are zero")
    
    # Check for extreme values
    pct_extreme = (volume > volume.quantile(0.99) * 10).mean()
    if pct_extreme > 0.01:
        results['issues'].append(f"Warning: {pct_extreme*100:.1f}% extreme predictions")
    
    # Check erosion pattern (volume should decrease over time)
    if 'months_postgx' in predictions_df.columns:
        avg_by_month = predictions_df.groupby('months_postgx')[volume_col].mean()
        if len(avg_by_month) > 6:
            early_avg = avg_by_month.iloc[:6].mean()
            late_avg = avg_by_month.iloc[-6:].mean()
            if late_avg > early_avg * 1.5:
                results['issues'].append(
                    f"Warning: Late months have higher volume ({late_avg:.0f}) "
                    f"than early months ({early_avg:.0f})"
                )
    
    return results


def generate_final_submission_report(
    submission_path: str,
    model_paths: Dict[str, str],
    cv_scores: Dict[str, float],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a final submission report with all details.
    
    Args:
        submission_path: Path to final submission
        model_paths: Dictionary of scenario -> model path
        cv_scores: Dictionary of scenario -> CV score
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    from datetime import datetime
    
    lines = [
        "=" * 60,
        "FINAL SUBMISSION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "SUBMISSION FILE",
        f"  Path: {submission_path}",
    ]
    
    # Check submission exists and get stats
    submission_file = Path(submission_path)
    if submission_file.exists():
        submission_df = pd.read_csv(submission_file)
        lines.extend([
            f"  Rows: {len(submission_df)}",
            f"  Volume range: [{submission_df['volume'].min():.2f}, {submission_df['volume'].max():.2f}]",
            f"  Volume mean: {submission_df['volume'].mean():.2f}",
        ])
    else:
        lines.append("  WARNING: Submission file not found!")
    
    lines.extend(["", "MODELS"])
    for scenario, path in model_paths.items():
        exists = "✓" if Path(path).exists() else "✗"
        lines.append(f"  Scenario {scenario}: {path} [{exists}]")
    
    lines.extend(["", "CV SCORES"])
    for scenario, score in cv_scores.items():
        lines.append(f"  Scenario {scenario}: {score:.4f}")
    
    lines.extend([
        "",
        "CHECKLIST",
        "  [ ] Submission format validated",
        "  [ ] All series predicted",
        "  [ ] Predictions are reasonable",
        "  [ ] Code and models backed up",
        "",
        "=" * 60
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved final report to {output_path}")
    
    return report


class FinalWeekPlaybook:
    """
    Manage final week execution with frozen configuration.
    
    Helps teams:
    - Define a frozen best config 48 hours before deadline
    - Generate multiple submission variants
    - Track submission verification
    
    Example usage:
        playbook = FinalWeekPlaybook()
        playbook.freeze_config(config_dict)
        variants = playbook.generate_submission_variants()
    """
    
    def __init__(self, playbook_path: str = 'artifacts/final_week_playbook.json'):
        """
        Initialize playbook.
        
        Args:
            playbook_path: Path to save playbook state
        """
        self.playbook_path = Path(playbook_path)
        self.frozen_config: Optional[Dict[str, Any]] = None
        self.freeze_timestamp: Optional[str] = None
        self.submissions_generated: List[Dict[str, Any]] = []
        self.verifications: List[Dict[str, Any]] = []
        
        self._load()
    
    def _load(self) -> None:
        """Load playbook state."""
        import json
        if self.playbook_path.exists():
            try:
                with open(self.playbook_path, 'r') as f:
                    data = json.load(f)
                self.frozen_config = data.get('frozen_config')
                self.freeze_timestamp = data.get('freeze_timestamp')
                self.submissions_generated = data.get('submissions_generated', [])
                self.verifications = data.get('verifications', [])
            except Exception:
                pass
    
    def _save(self) -> None:
        """Save playbook state."""
        import json
        self.playbook_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.playbook_path, 'w') as f:
            json.dump({
                'frozen_config': self.frozen_config,
                'freeze_timestamp': self.freeze_timestamp,
                'submissions_generated': self.submissions_generated,
                'verifications': self.verifications
            }, f, indent=2, default=str)
    
    def freeze_config(
        self,
        model_type: str,
        model_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        cv_scheme: str,
        ensemble_weights: Optional[Dict[str, float]] = None,
        notes: str = ''
    ) -> None:
        """
        Freeze the best configuration 48 hours before deadline.
        
        Args:
            model_type: Type of model (catboost, lightgbm, etc.)
            model_config: Model hyperparameters
            feature_config: Feature engineering settings
            cv_scheme: Cross-validation scheme used
            ensemble_weights: Optional ensemble weights
            notes: Additional notes
        """
        from datetime import datetime
        
        self.frozen_config = {
            'model_type': model_type,
            'model_config': model_config,
            'feature_config': feature_config,
            'cv_scheme': cv_scheme,
            'ensemble_weights': ensemble_weights,
            'notes': notes
        }
        self.freeze_timestamp = datetime.now().isoformat()
        self._save()
        
        logger.info(f"Configuration frozen at {self.freeze_timestamp}")
        logger.info("IMPORTANT: No major changes allowed after this point!")
    
    def is_config_frozen(self) -> bool:
        """Check if configuration has been frozen."""
        return self.frozen_config is not None
    
    def get_frozen_config(self) -> Optional[Dict[str, Any]]:
        """Get the frozen configuration."""
        return self.frozen_config
    
    def suggest_submission_variants(self) -> List[Dict[str, str]]:
        """
        Suggest submission variants to generate.
        
        Returns:
            List of suggested variants
        """
        return [
            {
                'name': 'best_cv',
                'description': 'Model with best CV score (frozen config)',
                'modifications': 'None - use frozen config exactly'
            },
            {
                'name': 'underfitted',
                'description': 'Slightly simpler model for robustness',
                'modifications': 'Reduce iterations/depth by 20%'
            },
            {
                'name': 'overfitted',
                'description': 'Slightly more complex model',
                'modifications': 'Increase iterations/depth by 20%'
            },
            {
                'name': 'bucket1_focus',
                'description': 'Higher weight on bucket 1 (high erosion)',
                'modifications': 'Increase bucket 1 sample weight by 50%'
            },
            {
                'name': 'multi_seed',
                'description': 'Ensemble of models with different seeds',
                'modifications': 'Train with seeds [42, 123, 456, 789, 1234]'
            }
        ]
    
    def log_submission_generated(
        self,
        variant_name: str,
        submission_path: str,
        cv_score: float,
        notes: str = ''
    ) -> None:
        """
        Log that a submission variant was generated.
        
        Args:
            variant_name: Name of variant
            submission_path: Path to submission file
            cv_score: CV score for this variant
            notes: Additional notes
        """
        from datetime import datetime
        
        self.submissions_generated.append({
            'variant': variant_name,
            'path': str(submission_path),
            'cv_score': cv_score,
            'timestamp': datetime.now().isoformat(),
            'notes': notes
        })
        self._save()
        logger.info(f"Logged submission variant: {variant_name}")
    
    def log_verification(
        self,
        submission_path: str,
        format_ok: bool,
        metric_score: Optional[float] = None,
        issues: Optional[List[str]] = None
    ) -> None:
        """
        Log verification of a submission.
        
        Args:
            submission_path: Path to verified submission
            format_ok: Whether format validation passed
            metric_score: Official metric score if computed
            issues: List of issues found
        """
        from datetime import datetime
        
        self.verifications.append({
            'path': str(submission_path),
            'format_ok': format_ok,
            'metric_score': metric_score,
            'issues': issues or [],
            'timestamp': datetime.now().isoformat()
        })
        self._save()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get playbook summary.
        
        Returns:
            Dictionary with playbook status
        """
        return {
            'is_frozen': self.is_config_frozen(),
            'freeze_timestamp': self.freeze_timestamp,
            'n_submissions_generated': len(self.submissions_generated),
            'n_verified': len([v for v in self.verifications if v['format_ok']]),
            'submissions': self.submissions_generated,
            'verifications': self.verifications
        }


def verify_external_data_compliance(project_root: str = '.') -> Dict[str, Any]:
    """
    Verify that no prohibited external data is used.
    
    Checks:
    - Only official competition data in data/raw/
    - No external API calls in code
    - No hardcoded external URLs
    
    Args:
        project_root: Root directory of project
        
    Returns:
        Dictionary with compliance check results
    """
    root = Path(project_root)
    
    results = {
        'is_compliant': True,
        'issues': [],
        'data_sources': []
    }
    
    # Check data directories
    raw_dir = root / 'data' / 'raw'
    if raw_dir.exists():
        for item in raw_dir.iterdir():
            if item.is_dir():
                results['data_sources'].append({
                    'path': str(item),
                    'type': 'directory',
                    'files': len(list(item.glob('*')))
                })
            else:
                results['data_sources'].append({
                    'path': str(item),
                    'type': 'file'
                })
    
    # Check for external data in code (simple heuristic)
    suspicious_patterns = [
        'requests.get',
        'urllib.request',
        'http://',
        'https://api.',
        'kaggle.com',
        'huggingface.co',
        'drive.google.com'
    ]
    
    src_dir = root / 'src'
    if src_dir.exists():
        for py_file in src_dir.glob('**/*.py'):
            try:
                content = py_file.read_text()
                for pattern in suspicious_patterns:
                    if pattern in content:
                        # Skip if it's in a comment or docstring context
                        if f'# {pattern}' not in content and f"'{pattern}" not in content:
                            results['issues'].append(
                                f"Potential external data access in {py_file}: {pattern}"
                            )
            except Exception:
                pass
    
    if results['issues']:
        results['is_compliant'] = False
        logger.warning(f"External data compliance issues found: {len(results['issues'])}")
    else:
        logger.info("External data compliance check passed")
    
    return results


def run_pre_submission_checklist(
    submission_path: str,
    template_path: str,
    model_s1_path: Optional[str] = None,
    model_s2_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete pre-submission checklist.
    
    Checks:
    1. Submission format matches template
    2. All series are predicted
    3. Predictions are reasonable
    4. Models are saved
    
    Args:
        submission_path: Path to submission file
        template_path: Path to template file
        model_s1_path: Optional path to S1 model
        model_s2_path: Optional path to S2 model
        
    Returns:
        Dictionary with checklist results
    """
    results = {
        'all_passed': True,
        'checks': {}
    }
    
    submission_file = Path(submission_path)
    template_file = Path(template_path)
    
    # Check 1: Submission file exists
    if not submission_file.exists():
        results['checks']['file_exists'] = {'passed': False, 'message': 'Submission file not found'}
        results['all_passed'] = False
        return results
    results['checks']['file_exists'] = {'passed': True}
    
    # Load files
    try:
        submission_df = pd.read_csv(submission_file)
        template_df = pd.read_csv(template_file)
    except Exception as e:
        results['checks']['file_readable'] = {'passed': False, 'message': str(e)}
        results['all_passed'] = False
        return results
    results['checks']['file_readable'] = {'passed': True}
    
    # Check 2: Column structure
    expected_cols = list(template_df.columns)
    actual_cols = list(submission_df.columns)
    cols_match = expected_cols == actual_cols
    results['checks']['columns_match'] = {
        'passed': cols_match,
        'expected': expected_cols,
        'actual': actual_cols
    }
    if not cols_match:
        results['all_passed'] = False
    
    # Check 3: All series predicted
    completeness = validate_submission_completeness(submission_df, template_df)
    results['checks']['completeness'] = {
        'passed': completeness['is_complete'],
        'missing_rows': completeness['missing_rows'],
        'extra_rows': completeness['extra_rows']
    }
    if not completeness['is_complete']:
        results['all_passed'] = False
    
    # Check 4: Sanity check
    sanity = check_prediction_sanity(submission_df)
    results['checks']['sanity'] = {
        'passed': sanity['is_sane'],
        'issues': sanity['issues'],
        'statistics': sanity['statistics']
    }
    if not sanity['is_sane']:
        results['all_passed'] = False
    
    # Check 5: Models saved
    if model_s1_path:
        s1_exists = Path(model_s1_path).exists()
        results['checks']['model_s1_saved'] = {'passed': s1_exists}
        if not s1_exists:
            results['all_passed'] = False
    
    if model_s2_path:
        s2_exists = Path(model_s2_path).exists()
        results['checks']['model_s2_saved'] = {'passed': s2_exists}
        if not s2_exists:
            results['all_passed'] = False
    
    # Summary
    passed_count = sum(1 for c in results['checks'].values() if c.get('passed', False))
    total_count = len(results['checks'])
    
    logger.info(f"Pre-submission checklist: {passed_count}/{total_count} checks passed")
    
    if results['all_passed']:
        logger.info("✓ All pre-submission checks passed!")
    else:
        logger.warning("✗ Some pre-submission checks failed - review before submitting")
    
    return results
