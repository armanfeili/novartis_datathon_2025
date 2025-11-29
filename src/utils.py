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
    except (NameError, AttributeError):
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

