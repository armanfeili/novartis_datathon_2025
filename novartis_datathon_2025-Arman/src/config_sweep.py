"""
Configuration sweep utilities for hyperparameter grid search.

This module provides functionality to:
1. Define named configurations (sweep_configs) for explicit hyperparameter sets
2. Expand grid-based sweeps into Cartesian products
3. Generate sweep runs with merged hyperparameters
4. Track active config IDs for model/submission naming
5. Log and aggregate sweep results

Supports both:
- Explicit config lists (SWEEP_CONFIGS / sweep_configs in YAML)
- Grid-based expansion (SWEEP_GRID / sweep.grid in YAML)

Example usage:
    from src.config_sweep import SweepEngine
    
    engine = SweepEngine.from_config('configs/model_xgb.yaml')
    runs = engine.generate_sweep_runs()
    
    for run in runs:
        model = build_model(run['params'])
        metrics = train_and_evaluate(model)
        engine.log_result(run['config_id'], metrics)
    
    best = engine.get_best_config()
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from itertools import product
from pathlib import Path
from datetime import datetime
import json
import csv
import logging

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# GLOBAL STATE: ACTIVE CONFIG ID
# =============================================================================
_ACTIVE_CONFIG_ID: Optional[str] = None
_ACTIVE_MODEL_TYPE: Optional[str] = None


def get_active_config_id() -> Optional[str]:
    """Get the currently active configuration ID."""
    return _ACTIVE_CONFIG_ID


def set_active_config_id(config_id: Optional[str]) -> None:
    """Set the currently active configuration ID."""
    global _ACTIVE_CONFIG_ID
    _ACTIVE_CONFIG_ID = config_id
    logger.debug(f"Active config ID set to: {config_id}")


def get_active_model_type() -> Optional[str]:
    """Get the currently active model type."""
    return _ACTIVE_MODEL_TYPE


def set_active_model_type(model_type: Optional[str]) -> None:
    """Set the currently active model type."""
    global _ACTIVE_MODEL_TYPE
    _ACTIVE_MODEL_TYPE = model_type


def get_model_filename(
    base_name: str,
    config_id: Optional[str] = None,
    model_type: Optional[str] = None,
    extension: str = ".joblib"
) -> str:
    """
    Generate a model filename that includes config_id.
    
    Args:
        base_name: Base filename (e.g., "model" or "fold_0")
        config_id: Config ID to include (uses active if None)
        model_type: Model type to include (uses active if None)
        extension: File extension
        
    Returns:
        Filename like "model_xgboost_low_lr.joblib"
    """
    config_id = config_id or get_active_config_id() or "default"
    model_type = model_type or get_active_model_type() or "model"
    safe_config_id = config_id.replace("/", "-").replace(" ", "_")
    safe_model_type = model_type.replace("/", "-").replace(" ", "_")
    return f"{base_name}_{safe_model_type}_{safe_config_id}{extension}"


def get_submission_filename(
    config_id: Optional[str] = None,
    model_type: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """Generate a submission filename that includes config_id."""
    config_id = config_id or get_active_config_id() or "default"
    model_type = model_type or get_active_model_type() or "model"
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    safe_config_id = config_id.replace("/", "-").replace(" ", "_")
    safe_model_type = model_type.replace("/", "-").replace(" ", "_")
    return f"submission_{safe_model_type}_{safe_config_id}_{timestamp}.csv"


def deep_merge(base: Dict, overrides: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


# =============================================================================
# SWEEP ENGINE CLASS
# =============================================================================

class SweepEngine:
    """Unified sweep engine for all models."""
    
    def __init__(
        self,
        base_params: Dict[str, Any],
        sweep_configs: Optional[List[Dict]] = None,
        sweep_grid: Optional[Dict[str, List]] = None,
        model_name: Optional[str] = None,
        selection_metric: str = "official_metric",
        mode: str = "configs"
    ):
        self.base_params = base_params
        self.sweep_configs = sweep_configs or []
        self.sweep_grid = sweep_grid or {}
        self.model_name = model_name or "unknown"
        self.selection_metric = selection_metric
        self.mode = mode
        self.results: List[Dict] = []
    
    @classmethod
    def from_config(cls, config_path: str) -> "SweepEngine":
        """Create a SweepEngine from a YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)
    
    @classmethod
    def from_dict(cls, config: Dict) -> "SweepEngine":
        """Create a SweepEngine from a config dictionary."""
        model_name = config.get("model", {}).get("name", "unknown")
        base_params = config.get("params", {})
        sweep_config = config.get("sweep", {})
        sweep_configs = config.get("sweep_configs", [])
        sweep_grid = sweep_config.get("grid", {})
        selection_metric = sweep_config.get("selection_metric", "official_metric")
        mode = sweep_config.get("mode", "configs")
        return cls(base_params, sweep_configs, sweep_grid, model_name, selection_metric, mode)
    
    def get_config_by_id(self, config_id: str) -> Dict:
        """Get a specific named configuration by its ID."""
        if config_id in (None, "default", "base"):
            return deepcopy(self.base_params)
        for cfg in self.sweep_configs:
            if cfg.get("id") == config_id:
                return deep_merge(self.base_params, cfg.get("params", {}))
        raise ValueError(f"Config ID '{config_id}' not found")
    
    def apply_config(self, config_id: str) -> Dict:
        """Apply a named config and return the merged params."""
        set_active_config_id(config_id)
        set_active_model_type(self.model_name)
        return self.get_config_by_id(config_id)
    
    def generate_sweep_runs(self) -> List[Dict]:
        """Generate all sweep runs based on mode."""
        if self.mode == "grid":
            return self._generate_grid_runs()
        return self._generate_configs_runs()
    
    def _generate_configs_runs(self) -> List[Dict]:
        """Generate runs from explicit sweep_configs list."""
        if not self.sweep_configs:
            return [{"config_id": "default", "description": "Base configuration",
                     "params": deepcopy(self.base_params), "index": 0, "total": 1}]
        runs = []
        total = len(self.sweep_configs)
        for idx, cfg in enumerate(self.sweep_configs):
            config_id = cfg.get("id", f"config_{idx}")
            runs.append({
                "config_id": config_id,
                "description": cfg.get("description", ""),
                "params": deep_merge(self.base_params, cfg.get("params", {})),
                "overrides": cfg.get("params", {}),
                "index": idx,
                "total": total
            })
        return runs
    
    def _generate_grid_runs(self) -> List[Dict]:
        """Generate runs from Cartesian product of sweep.grid."""
        if not self.sweep_grid:
            return [{"config_id": "default", "params": deepcopy(self.base_params),
                     "index": 0, "total": 1}]
        param_names = list(self.sweep_grid.keys())
        param_values = [self.sweep_grid[name] for name in param_names]
        combinations = list(product(*param_values))
        total = len(combinations)
        runs = []
        for idx, combo in enumerate(combinations):
            overrides = dict(zip(param_names, combo))
            merged = deepcopy(self.base_params)
            merged.update(overrides)
            config_id = self._generate_grid_config_id(param_names, combo)
            runs.append({
                "config_id": config_id,
                "description": f"Grid: {overrides}",
                "params": merged,
                "overrides": overrides,
                "index": idx,
                "total": total
            })
        return runs
    
    def _generate_grid_config_id(self, param_names: List[str], values: Tuple) -> str:
        abbrevs = {"learning_rate": "lr", "max_depth": "d", "depth": "d",
                   "num_leaves": "nl", "min_data_in_leaf": "mdl", "reg_lambda": "lam"}
        parts = []
        for name, value in zip(param_names, values):
            abbr = abbrevs.get(name, name[:3])
            if isinstance(value, float):
                val_str = f"{value:.3g}".replace(".", "p") if value >= 0.01 else f"{value:.0e}"
            else:
                val_str = str(value)
            parts.append(f"{abbr}{val_str}")
        return "_".join(parts)
    
    def log_result(self, config_id: str, metrics: Dict[str, float],
                   params: Optional[Dict] = None, training_time: Optional[float] = None,
                   scenario: Optional[int] = None, seed: Optional[int] = None) -> None:
        """Log the result of a sweep run."""
        result = {"model_name": self.model_name, "config_id": config_id,
                  "params_json": json.dumps(params) if params else None,
                  "training_time": training_time, "scenario": scenario,
                  "seed": seed, "timestamp": datetime.now().isoformat()}
        result.update(metrics)
        self.results.append(result)
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"[{config_id}] {metric_str}")
    
    def get_best_config(self, higher_is_better: bool = True) -> Optional[Dict]:
        """Get the best configuration based on selection_metric."""
        valid = [r for r in self.results if self.selection_metric in r]
        if not valid:
            return None
        return sorted(valid, key=lambda x: x[self.selection_metric], reverse=higher_is_better)[0]
    
    def save_results(self, output_path: str, format: str = "csv") -> None:
        """Save sweep results to a file."""
        if not self.results:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        else:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
                writer.writeheader()
                writer.writerows(self.results)
    
    def print_summary(self, top_n: int = 5, higher_is_better: bool = True) -> None:
        """Print a summary of sweep results."""
        if not self.results:
            print("No results")
            return
        valid = [r for r in self.results if self.selection_metric in r]
        if not valid:
            return
        sorted_results = sorted(valid, key=lambda x: float(x[self.selection_metric]),
                                reverse=higher_is_better)
        print(f"\n{'='*60}\nSWEEP SUMMARY: {self.model_name}\n{'='*60}")
        print(f"Total runs: {len(self.results)}")
        for i, r in enumerate(sorted_results[:top_n]):
            print(f"  {i+1}. {r.get('config_id')}: {self.selection_metric}={r.get(self.selection_metric)}")
        best = sorted_results[0]
        print(f"\nBEST: {best.get('config_id')} ({self.selection_metric}={best.get(self.selection_metric)})")


# =============================================================================
# SWEEP RESULTS LOGGER (Persistent)
# =============================================================================

class SweepResultsLogger:
    """Central logging for sweep results across multiple runs."""
    
    def __init__(self, output_dir: str = "reports", filename: str = "sweep_results.csv"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename
    
    def log(self, model_name: str, config_id: str, metrics: Dict[str, float],
            params: Optional[Dict] = None, scenario: Optional[int] = None,
            training_time: Optional[float] = None, seed: Optional[int] = None) -> None:
        """Log a single result to the results file."""
        row = {"timestamp": datetime.now().isoformat(), "model_name": model_name,
               "config_id": config_id, "scenario": scenario, "training_time": training_time,
               "seed": seed, "params_json": json.dumps(params) if params else None}
        row.update(metrics)
        file_exists = self.filepath.exists()
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
    def get_best_per_model(self, metric: str = "official_metric",
                           higher_is_better: bool = True) -> Dict[str, Dict]:
        """Get the best configuration for each model."""
        if not self.filepath.exists():
            return {}
        with open(self.filepath, 'r') as f:
            rows = list(csv.DictReader(f))
        by_model = {}
        for row in rows:
            model = row.get("model_name")
            by_model.setdefault(model, []).append(row)
        return {m: sorted([r for r in rs if r.get(metric)],
                          key=lambda x: float(x[metric]), reverse=higher_is_better)[0]
                for m, rs in by_model.items() if any(r.get(metric) for r in rs)}
    
    def print_summary(self) -> None:
        """Print a summary of all sweep results."""
        best = self.get_best_per_model()
        if not best:
            print("No sweep results found")
            return
        print(f"\n{'='*60}\nSWEEP RESULTS SUMMARY\n{'='*60}")
        for model, b in sorted(best.items()):
            print(f"  {model}: {b.get('config_id')} (official_metric={b.get('official_metric')})")


# =============================================================================
# BACKWARD COMPATIBLE FUNCTIONS
# =============================================================================

SWEEPABLE_KEYS = {
    "params.depth", "params.learning_rate", "params.l2_leaf_reg",
    "params.min_data_in_leaf", "params.num_leaves", "params.max_depth",
    "params.feature_fraction", "params.bagging_fraction", "params.n_estimators",
    "params.subsample", "params.colsample_bytree", "params.reg_alpha", "params.reg_lambda",
}


def detect_list_axes(config: Dict, keys: Optional[List[str]] = None) -> List[str]:
    """
    Detect which config keys contain list values (for sweep expansion).
    
    Args:
        config: Configuration dictionary
        keys: Optional list of keys to check (defaults to SWEEPABLE_KEYS)
        
    Returns:
        List of dotted paths that contain list values
    """
    keys = keys or list(SWEEPABLE_KEYS)
    list_axes = []
    for path in keys:
        value = get_nested_value(config, path)
        if isinstance(value, list) and len(value) > 1:
            list_axes.append(path)
    return list_axes


def get_nested_value(d: Dict, path: str, default: Any = None) -> Any:
    """Get a value from a nested dict using a dotted path."""
    keys = path.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def set_nested_value(d: Dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted path."""
    keys = path.split(".")
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def has_nested_key(d: Dict, path: str) -> bool:
    """Check if a nested path exists in a dictionary."""
    return get_nested_value(d, path) is not None


def get_sweep_axes(config: Dict) -> List[str]:
    """Get the list of sweep axes from a config (legacy support)."""
    sweep_config = config.get("sweep", {})
    if not sweep_config.get("enabled", True):
        return []
    axes = sweep_config.get("axes", sweep_config.get("grid", {}))
    return list(axes.keys()) if isinstance(axes, dict) else axes


def expand_sweep(config: Dict, sweep_axes: Optional[List[str]] = None,
                 include_metadata: bool = True) -> List[Dict]:
    """Expand a config into multiple configs (legacy + new style support).
    
    Supports:
    1. New-style sweep_configs (explicit configs)
    2. Legacy sweep.grid (grid dict in sweep section)
    3. Direct list values in params (when sweep_axes specifies paths)
    """
    # New-style sweep_configs
    if config.get("sweep_configs"):
        engine = SweepEngine.from_dict(config)
        runs = engine.generate_sweep_runs()
        expanded = []
        for run in runs:
            cfg = deepcopy(config)
            cfg["params"] = run["params"]
            if include_metadata:
                cfg["_sweep_metadata"] = {
                    "axes": run.get("overrides", {}),
                    "index": run["index"],
                    "total": run["total"],
                    "is_sweep": run["total"] > 1,
                    "config_id": run["config_id"]
                }
            expanded.append(cfg)
        return expanded
    
    # If sweep_axes is provided, look for list values at those paths
    if sweep_axes:
        list_axes = {}
        for path in sweep_axes:
            value = get_nested_value(config, path)
            if isinstance(value, list) and len(value) > 1:
                list_axes[path] = value
        
        if list_axes:
            axis_names = list(list_axes.keys())
            combinations = list(product(*[list_axes[n] for n in axis_names]))
            total = len(combinations)
            
            expanded = []
            for idx, combo in enumerate(combinations):
                resolved = deepcopy(config)
                axes_values = {}
                for path, value in zip(axis_names, combo):
                    set_nested_value(resolved, path, value)
                    # Use full path for axes_values to maintain backward compatibility
                    axes_values[path] = value
                if include_metadata:
                    # Use short key for config_id
                    config_id = "_".join(f"{p.split('.')[-1]}{v}" for p, v in axes_values.items())
                    resolved["_sweep_metadata"] = {
                        "axes": axes_values, "index": idx, "total": total,
                        "is_sweep": True, "config_id": config_id
                    }
                expanded.append(resolved)
            return expanded
    
    # Legacy grid-based expansion from sweep.grid
    sweep_config = config.get("sweep", {})
    grid = sweep_config.get("grid", sweep_config.get("axes", {}))
    if not grid or not isinstance(grid, dict):
        result = deepcopy(config)
        if include_metadata:
            result["_sweep_metadata"] = {"axes": {}, "index": 0, "total": 1,
                                          "is_sweep": False, "config_id": "default"}
        return [result]
    
    list_axes = {k: v for k, v in grid.items() if isinstance(v, list)}
    if not list_axes:
        result = deepcopy(config)
        if include_metadata:
            result["_sweep_metadata"] = {"axes": {}, "index": 0, "total": 1,
                                          "is_sweep": False, "config_id": "default"}
        return [result]
    
    axis_names = list(list_axes.keys())
    combinations = list(product(*[list_axes[n] for n in axis_names]))
    total = len(combinations)
    
    expanded = []
    for idx, combo in enumerate(combinations):
        resolved = deepcopy(config)
        axes_values = dict(zip(axis_names, combo))
        if "params" in resolved:
            resolved["params"].update(axes_values)
        if include_metadata:
            config_id = "_".join(f"{k}{v}" for k, v in axes_values.items())
            resolved["_sweep_metadata"] = {
                "axes": axes_values, "index": idx, "total": total,
                "is_sweep": True, "config_id": config_id
            }
        expanded.append(resolved)
    return expanded


def generate_sweep_suffix(sweep_metadata: Dict) -> str:
    """Generate a filesystem-safe suffix from sweep metadata."""
    axes = sweep_metadata.get("axes", {})
    if not axes:
        return sweep_metadata.get("config_id", "")
    abbrevs = {"learning_rate": "lr", "l2_leaf_reg": "l2", "max_depth": "d",
               "depth": "d", "num_leaves": "nl", "min_data_in_leaf": "mdl", "reg_lambda": "lam"}
    parts = []
    for path, value in sorted(axes.items()):
        key = abbrevs.get(path.split(".")[-1], path.split(".")[-1])
        if isinstance(value, float):
            val_str = f"{value:.3g}".replace(".", "p") if value >= 0.01 else str(value)
        else:
            val_str = str(value)
        parts.append(f"{key}{val_str}")
    return "_".join(parts)


def build_sweep_run_id(base_run_id: str, sweep_metadata: Optional[Dict] = None) -> str:
    """Build a run ID that includes sweep parameters."""
    if sweep_metadata is None:
        return base_run_id
    config_id = sweep_metadata.get("config_id")
    if config_id and config_id != "default":
        return f"{base_run_id}_{config_id}"
    if not sweep_metadata.get("is_sweep", False):
        return base_run_id
    suffix = generate_sweep_suffix(sweep_metadata)
    if suffix:
        return f"{base_run_id}_{suffix}"
    idx = sweep_metadata.get("index", 0)
    total = sweep_metadata.get("total", 1)
    return f"{base_run_id}_run{idx+1}of{total}"


def log_sweep_combination(sweep_metadata: Dict, logger_instance=None) -> None:
    """Log the current sweep combination clearly."""
    log = logger_instance or logger
    if not sweep_metadata.get("is_sweep", False):
        return
    idx = sweep_metadata.get("index", 0) + 1
    total = sweep_metadata.get("total", 1)
    config_id = sweep_metadata.get("config_id", "unknown")
    axes = sweep_metadata.get("axes", {})
    axes_str = ", ".join(f"{k}={v}" for k, v in sorted(axes.items()))
    log.info(f"[SWEEP] Config {idx}/{total}: {config_id}")
    if axes_str:
        log.info(f"        Params: {axes_str}")


def get_sweep_columns(sweep_metadata: Dict) -> Dict[str, Any]:
    """Get sweep parameters as flat columns for metrics output."""
    result = {"sweep_index": sweep_metadata.get("index", 0),
              "sweep_total": sweep_metadata.get("total", 1),
              "is_sweep": sweep_metadata.get("is_sweep", False),
              "config_id": sweep_metadata.get("config_id", "default")}
    for path, value in sweep_metadata.get("axes", {}).items():
        result[f"sweep_{path.replace('.', '_')}"] = value
    return result


# =============================================================================
# STANDALONE HELPER FUNCTIONS (for backward compatibility)
# =============================================================================

def get_config_by_id(config: Dict, config_id: str) -> Optional[Dict]:
    """
    Get a named configuration by ID from config dict.
    
    Looks up config_id in 'sweep_configs' or 'named_configs' list and returns the config dict.
    
    Args:
        config: Full config dictionary containing 'sweep_configs' list
        config_id: ID of the configuration to retrieve
        
    Returns:
        Configuration dict with id, description, params, or None if not found
    """
    # Try sweep_configs first (new style), then named_configs (alternate name)
    named_configs = config.get('sweep_configs', config.get('named_configs', []))
    for cfg in named_configs:
        if cfg.get('id') == config_id:
            return cfg
    return None


def apply_config_overrides(base_params: Dict, overrides: Dict) -> Dict:
    """
    Merge base parameters with config-specific overrides.
    
    Uses deep merge to combine nested dictionaries.
    
    Args:
        base_params: Base parameter dictionary
        overrides: Parameter overrides from named config
        
    Returns:
        Merged parameter dictionary
    """
    return deep_merge(base_params, overrides)


def get_active_config(config: Dict) -> Optional[Dict]:
    """
    Get the currently active configuration from config dict.
    
    Checks active_config_id to find which named config is active.
    
    Args:
        config: Full config dictionary
        
    Returns:
        Active configuration dict or None
    """
    active_id = config.get('active_config_id')
    if not active_id or active_id == 'null':
        return None
    return get_config_by_id(config, active_id)


def generate_sweep_runs(
    config: Dict, 
    mode: str = 'both'
) -> List[Dict]:
    """
    Generate all sweep runs from a config dictionary.
    
    Args:
        config: Model configuration dictionary with sweep_configs and/or sweep.grid
        mode: 'explicit' for sweep_configs only, 'grid' for sweep.grid only, 'both' for all
        
    Returns:
        List of dicts with 'config_id', 'params', 'source', etc.
    """
    runs = []
    base_params = config.get('params', {})
    
    # Generate from sweep_configs (explicit mode)
    if mode in ('explicit', 'both'):
        sweep_configs = config.get('sweep_configs', config.get('named_configs', []))
        for cfg in sweep_configs:
            config_id = cfg.get('id', 'unnamed')
            params = deep_merge(base_params, cfg.get('params', {}))
            runs.append({
                'config_id': config_id,
                'description': cfg.get('description', ''),
                'params': params,
                'overrides': cfg.get('params', {}),
                'source': 'sweep_config'
            })
    
    # Generate from sweep.grid (grid mode)
    if mode in ('grid', 'both'):
        sweep_section = config.get('sweep', {})
        sweep_grid = sweep_section.get('grid', config.get('sweep_grid', {}))
        if sweep_grid:
            param_names = list(sweep_grid.keys())
            param_values = [sweep_grid[name] if isinstance(sweep_grid[name], list) else [sweep_grid[name]] 
                           for name in param_names]
            combinations = list(product(*param_values))
            
            for combo in combinations:
                overrides = dict(zip(param_names, combo))
                params = deepcopy(base_params)
                params.update(overrides)
                
                # Generate config_id from parameters
                config_id = _generate_grid_config_id(param_names, combo)
                runs.append({
                    'config_id': config_id,
                    'description': f"Grid: {overrides}",
                    'params': params,
                    'overrides': overrides,
                    'source': 'sweep_grid'
                })
    
    return runs


def _generate_grid_config_id(param_names: List[str], values: Tuple) -> str:
    """Generate a config ID from grid parameter values."""
    abbrevs = {
        "learning_rate": "lr", "max_depth": "d", "depth": "d",
        "num_leaves": "nl", "min_data_in_leaf": "mdl", "reg_lambda": "lam",
        "reg_alpha": "alpha", "subsample": "ss", "colsample_bytree": "cs",
        "n_estimators": "est"
    }
    parts = []
    for name, value in zip(param_names, values):
        abbr = abbrevs.get(name, name[:3])
        if isinstance(value, float):
            val_str = f"{value:.3g}".replace(".", "p") if value >= 0.01 else f"{value:.0e}"
        else:
            val_str = str(value)
        parts.append(f"{abbr}{val_str}")
    return "_".join(parts)


class SweepResultsLogger:
    """Logger for tracking sweep experiment results."""
    
    def __init__(self, output_path: str):
        """
        Initialize the sweep results logger.
        
        Args:
            output_path: Path to output CSV file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
        self._fieldnames: Optional[List[str]] = None
    
    def log_result(
        self,
        model_name: str,
        config_id: str,
        params: Dict,
        metrics: Dict,
        training_time: Optional[float] = None,
        scenario: Optional[int] = None
    ) -> None:
        """
        Log a single sweep result.
        
        Args:
            model_name: Name of the model (e.g., 'xgboost')
            config_id: Configuration ID
            params: Hyperparameters used
            metrics: Resulting metrics
            training_time: Training time in seconds
            scenario: Scenario number (1 or 2)
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'config_id': config_id,
            'scenario': scenario,
            'training_time': training_time,
            'params_json': json.dumps(params),
        }
        
        # Add all metrics
        for key, value in metrics.items():
            if key not in ('run_id', 'sweep_params', 'config_id'):
                result[f'metric_{key}' if not key.startswith('metric_') else key] = value
        
        self.results.append(result)
        self._write_result(result)
    
    def _write_result(self, result: Dict) -> None:
        """Write a single result to the CSV file."""
        if self._fieldnames is None:
            self._fieldnames = list(result.keys())
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(result)
        else:
            # Check for new columns
            new_keys = [k for k in result.keys() if k not in self._fieldnames]
            if new_keys:
                self._fieldnames.extend(new_keys)
                # Rewrite file with new columns
                with open(self.output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                    writer.writeheader()
                    for r in self.results:
                        writer.writerow(r)
            else:
                with open(self.output_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                    writer.writerow(result)
    
    def get_results_df(self) -> "pd.DataFrame":
        """Get results as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.results)
    
    def summarize(self, metric: str = 'metric_official_metric') -> Dict:
        """
        Summarize sweep results.
        
        Args:
            metric: Metric to optimize (lower is better)
            
        Returns:
            Dict with best_per_model, best_overall
        """
        if not self.results:
            return {'best_per_model': {}, 'best_overall': None}
        
        import pandas as pd
        df = self.get_results_df()
        
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in results")
            return {'best_per_model': {}, 'best_overall': None}
        
        # Best per model
        best_per_model = {}
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            best_idx = model_df[metric].idxmin()
            best_per_model[model] = model_df.loc[best_idx].to_dict()
        
        # Best overall
        best_idx = df[metric].idxmin()
        best_overall = df.loc[best_idx].to_dict()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SWEEP RESULTS SUMMARY")
        logger.info("="*60)
        for model, result in best_per_model.items():
            logger.info(f"\nBest {model}:")
            logger.info(f"  Config: {result.get('config_id')}")
            logger.info(f"  {metric}: {result.get(metric):.4f}")
        
        logger.info(f"\nOverall Best:")
        logger.info(f"  Model: {best_overall.get('model_name')}")
        logger.info(f"  Config: {best_overall.get('config_id')}")
        logger.info(f"  {metric}: {best_overall.get(metric):.4f}")
        logger.info("="*60 + "\n")
        
        return {'best_per_model': best_per_model, 'best_overall': best_overall}
