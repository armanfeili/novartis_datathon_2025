"""
Configuration sweep utilities for hyperparameter grid search.

This module provides functionality to:
1. Detect list-valued parameters in configs (sweep axes)
2. Expand configs into all Cartesian product combinations
3. Track sweep metadata for logging and analysis

Only explicitly whitelisted keys are treated as sweep axes to avoid
confusing them with existing semantic lists (like pre_entry.windows).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from itertools import product
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SWEEPABLE KEYS REGISTRY
# =============================================================================
# These are the only keys that can be treated as sweep axes.
# Other lists (like hidden_layers, windows, etc.) are semantic lists
# that should NOT be expanded into multiple runs.

SWEEPABLE_KEYS = {
    # CatBoost parameters
    "params.depth",
    "params.learning_rate",
    "params.l2_leaf_reg",
    "params.min_data_in_leaf",
    "params.bagging_temperature",
    "params.random_strength",
    "params.iterations",
    "params.early_stopping_rounds",
    
    # LightGBM parameters
    "params.num_leaves",
    "params.max_depth",
    "params.feature_fraction",
    "params.bagging_fraction",
    "params.n_estimators",
    
    # XGBoost parameters
    "params.subsample",
    "params.colsample_bytree",
    "params.min_child_weight",
    "params.eta",
    "params.reg_alpha",
    "params.reg_lambda",
    
    # Neural network parameters
    "training.learning_rate",
    "training.weight_decay",
    "training.batch_size",
    "training.epochs",
    "architecture.mlp.dropout",
    
    # Linear model parameters
    "ridge.alpha",
    "lasso.alpha",
    "elasticnet.alpha",
    "elasticnet.l1_ratio",
    
    # Run-level parameters (validation, sample weights)
    "validation.val_fraction",
    "sample_weights.scenario1.months_0_5",
    "sample_weights.scenario1.months_6_11",
    "sample_weights.scenario1.months_12_23",
    "sample_weights.scenario2.months_6_11",
    "sample_weights.scenario2.months_12_23",
    "sample_weights.bucket_weights.bucket1",
    "sample_weights.bucket_weights.bucket2",
    
    # Feature toggles
    "interactions.enabled",
    "target_encoding.enabled",
}


# =============================================================================
# NESTED DICT UTILITIES
# =============================================================================

def get_nested_value(d: Dict, path: str, default: Any = None) -> Any:
    """
    Get a value from a nested dict using a dotted path.
    
    Args:
        d: Dictionary to traverse
        path: Dotted path like "params.depth" or "training.learning_rate"
        default: Value to return if path not found
        
    Returns:
        Value at path, or default if not found
        
    Examples:
        >>> get_nested_value({"params": {"depth": 6}}, "params.depth")
        6
        >>> get_nested_value({"a": {"b": {"c": 1}}}, "a.b.c")
        1
    """
    keys = path.split(".")
    current = d
    
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    
    return current


def set_nested_value(d: Dict, path: str, value: Any) -> None:
    """
    Set a value in a nested dict using a dotted path.
    Creates intermediate dicts if they don't exist.
    
    Args:
        d: Dictionary to modify (in-place)
        path: Dotted path like "params.depth"
        value: Value to set
        
    Examples:
        >>> d = {"params": {"depth": 6}}
        >>> set_nested_value(d, "params.depth", 8)
        >>> d["params"]["depth"]
        8
    """
    keys = path.split(".")
    current = d
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def has_nested_key(d: Dict, path: str) -> bool:
    """
    Check if a nested path exists in a dictionary.
    
    Args:
        d: Dictionary to check
        path: Dotted path to check
        
    Returns:
        True if path exists, False otherwise
    """
    keys = path.split(".")
    current = d
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    
    return True


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

def get_sweep_axes(config: Dict) -> List[str]:
    """
    Get the list of sweep axes from a config.
    
    Reads from config["sweep"]["axes"] if present and sweep is enabled.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of dotted paths that are sweep axes
    """
    sweep_config = config.get("sweep", {})
    
    # Check if sweep is enabled (default: True if axes are specified)
    enabled = sweep_config.get("enabled", True)
    if not enabled:
        return []
    
    axes = sweep_config.get("axes", [])
    
    # Validate that all axes are in the allowed set
    valid_axes = []
    for axis in axes:
        if axis in SWEEPABLE_KEYS:
            valid_axes.append(axis)
        else:
            logger.warning(
                f"Sweep axis '{axis}' is not in SWEEPABLE_KEYS and will be ignored. "
                f"Add it to SWEEPABLE_KEYS in src/config_sweep.py if you want to sweep over it."
            )
    
    return valid_axes


def detect_list_axes(config: Dict, sweep_axes: List[str]) -> Dict[str, List[Any]]:
    """
    Detect which sweep axes currently have list values.
    
    Args:
        config: Configuration dictionary
        sweep_axes: List of dotted paths to check
        
    Returns:
        Dict mapping path -> list of values (only for paths with list values)
    """
    list_axes = {}
    
    for axis in sweep_axes:
        value = get_nested_value(config, axis)
        
        if value is None:
            continue
            
        if isinstance(value, list):
            # This is a sweep axis with multiple values
            list_axes[axis] = value
        # Scalar values are not included - they don't need expansion
    
    return list_axes


# =============================================================================
# SWEEP EXPANSION
# =============================================================================

def expand_sweep(
    config: Dict,
    sweep_axes: Optional[List[str]] = None,
    include_metadata: bool = True
) -> List[Dict]:
    """
    Expand a config with list-valued sweep axes into multiple scalar configs.
    
    Given a config where some sweep axes have list values, produces the
    Cartesian product of all combinations, each as a separate config.
    
    Args:
        config: Configuration dictionary (may have list values for sweep axes)
        sweep_axes: List of dotted paths to treat as sweep axes.
                   If None, reads from config["sweep"]["axes"].
        include_metadata: If True, add _sweep_metadata to each expanded config
        
    Returns:
        List of resolved configs, each with scalar values for all sweep axes.
        If no axes have list values, returns [config] (single config).
        
    Examples:
        >>> config = {"params": {"depth": [4, 6], "lr": 0.03}}
        >>> expanded = expand_sweep(config, ["params.depth", "params.lr"])
        >>> len(expanded)
        2
        >>> expanded[0]["params"]["depth"]
        4
        >>> expanded[1]["params"]["depth"]
        6
    """
    # Get sweep axes from config if not provided
    if sweep_axes is None:
        sweep_axes = get_sweep_axes(config)
    
    # If no sweep axes or sweep is disabled, return config as-is
    if not sweep_axes:
        if include_metadata:
            result = deepcopy(config)
            result["_sweep_metadata"] = {
                "axes": {},
                "index": 0,
                "total": 1,
                "is_sweep": False
            }
            return [result]
        return [config]
    
    # Detect which axes have list values
    list_axes = detect_list_axes(config, sweep_axes)
    
    # If no axes have lists, return config as-is
    if not list_axes:
        if include_metadata:
            result = deepcopy(config)
            result["_sweep_metadata"] = {
                "axes": {},
                "index": 0,
                "total": 1,
                "is_sweep": False
            }
            return [result]
        return [config]
    
    # Generate Cartesian product of all list-valued axes
    axis_names = list(list_axes.keys())
    axis_values = [list_axes[name] for name in axis_names]
    combinations = list(product(*axis_values))
    
    total = len(combinations)
    logger.info(f"Expanding sweep: {len(list_axes)} axes, {total} combinations")
    for axis, values in list_axes.items():
        logger.info(f"  {axis}: {values}")
    
    # Create a config for each combination
    expanded_configs = []
    
    for idx, combo in enumerate(combinations):
        # Deep copy the original config
        resolved = deepcopy(config)
        
        # Set each axis to its scalar value for this combination
        axes_values = {}
        for axis_name, value in zip(axis_names, combo):
            set_nested_value(resolved, axis_name, value)
            axes_values[axis_name] = value
        
        # Add sweep metadata
        if include_metadata:
            resolved["_sweep_metadata"] = {
                "axes": axes_values,
                "index": idx,
                "total": total,
                "is_sweep": True
            }
        
        expanded_configs.append(resolved)
    
    return expanded_configs


def combine_sweep_axes(*configs: Dict) -> List[str]:
    """
    Combine sweep axes from multiple configs.
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Combined list of unique sweep axes from all configs
    """
    all_axes = []
    seen = set()
    
    for config in configs:
        axes = get_sweep_axes(config)
        for axis in axes:
            if axis not in seen:
                all_axes.append(axis)
                seen.add(axis)
    
    return all_axes


def expand_combined_sweep(
    model_config: Dict,
    run_config: Optional[Dict] = None,
    include_metadata: bool = True
) -> List[Tuple[Dict, Dict]]:
    """
    Expand sweep for both model and run configs, returning combined combinations.
    
    This handles the case where both model_config and run_config have sweep axes,
    producing the full Cartesian product across both.
    
    Args:
        model_config: Model configuration (e.g., model_cat.yaml)
        run_config: Run configuration (e.g., run_defaults.yaml), optional
        include_metadata: If True, add _sweep_metadata to expanded configs
        
    Returns:
        List of (model_config, run_config) tuples, one for each combination
    """
    if run_config is None:
        # Only expand model config
        expanded_models = expand_sweep(model_config, include_metadata=include_metadata)
        return [(m, None) for m in expanded_models]
    
    # Get axes from both configs
    model_axes = get_sweep_axes(model_config)
    run_axes = get_sweep_axes(run_config)
    
    # Expand each separately first
    expanded_models = expand_sweep(model_config, model_axes, include_metadata=False)
    expanded_runs = expand_sweep(run_config, run_axes, include_metadata=False)
    
    # Compute full Cartesian product
    total = len(expanded_models) * len(expanded_runs)
    
    results = []
    idx = 0
    
    for model_cfg in expanded_models:
        for run_cfg in expanded_runs:
            # Deep copy to avoid mutations
            m = deepcopy(model_cfg)
            r = deepcopy(run_cfg)
            
            if include_metadata:
                # Combine axes from both configs
                combined_axes = {}
                
                # Get model axes
                model_list_axes = detect_list_axes(model_config, model_axes)
                for axis in model_list_axes:
                    combined_axes[f"model.{axis}"] = get_nested_value(m, axis)
                
                # Get run axes
                run_list_axes = detect_list_axes(run_config, run_axes)
                for axis in run_list_axes:
                    combined_axes[f"run.{axis}"] = get_nested_value(r, axis)
                
                metadata = {
                    "axes": combined_axes,
                    "index": idx,
                    "total": total,
                    "is_sweep": total > 1
                }
                
                m["_sweep_metadata"] = metadata
                r["_sweep_metadata"] = metadata
            
            results.append((m, r))
            idx += 1
    
    return results


# =============================================================================
# RUN ID GENERATION
# =============================================================================

def generate_sweep_suffix(sweep_metadata: Dict) -> str:
    """
    Generate a filesystem-safe suffix from sweep metadata.
    
    Args:
        sweep_metadata: The _sweep_metadata dict from an expanded config
        
    Returns:
        String suffix like "depth-6_lr-0.03" or empty string if no axes
    """
    axes = sweep_metadata.get("axes", {})
    
    if not axes:
        return ""
    
    parts = []
    for path, value in sorted(axes.items()):
        # Simplify path: params.depth -> depth, params.learning_rate -> lr
        key = path.split(".")[-1]
        
        # Common abbreviations
        abbreviations = {
            "learning_rate": "lr",
            "l2_leaf_reg": "l2",
            "bagging_temperature": "bt",
            "random_strength": "rs",
            "min_data_in_leaf": "mdl",
            "num_leaves": "nl",
            "feature_fraction": "ff",
            "bagging_fraction": "bf",
            "colsample_bytree": "csb",
            "early_stopping_rounds": "es",
            "val_fraction": "vf",
        }
        key = abbreviations.get(key, key)
        
        # Format value
        if isinstance(value, float):
            if value < 0.01:
                val_str = f"{value:.0e}".replace("-", "m")
            elif value == int(value):
                val_str = str(int(value))
            else:
                val_str = f"{value:.3g}".replace(".", "p")
        else:
            val_str = str(value)
        
        # Make filesystem-safe
        val_str = val_str.replace("/", "-").replace(" ", "")
        
        parts.append(f"{key}{val_str}")
    
    return "_".join(parts)


def build_sweep_run_id(
    base_run_id: str,
    sweep_metadata: Optional[Dict] = None
) -> str:
    """
    Build a run ID that includes sweep parameters.
    
    Args:
        base_run_id: Base run ID (e.g., "2025-01-15_10-30_catboost_scenario1")
        sweep_metadata: Sweep metadata dict, if present
        
    Returns:
        Run ID with sweep suffix appended if applicable
    """
    if sweep_metadata is None or not sweep_metadata.get("is_sweep", False):
        return base_run_id
    
    suffix = generate_sweep_suffix(sweep_metadata)
    
    if suffix:
        return f"{base_run_id}_{suffix}"
    
    # Fallback: use index
    idx = sweep_metadata.get("index", 0)
    total = sweep_metadata.get("total", 1)
    return f"{base_run_id}_run{idx+1}of{total}"


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def log_sweep_combination(sweep_metadata: Dict, logger_instance=None) -> None:
    """
    Log the current sweep combination clearly.
    
    Args:
        sweep_metadata: The _sweep_metadata dict
        logger_instance: Logger to use (defaults to module logger)
    """
    log = logger_instance or logger
    
    if not sweep_metadata.get("is_sweep", False):
        return
    
    idx = sweep_metadata.get("index", 0) + 1
    total = sweep_metadata.get("total", 1)
    axes = sweep_metadata.get("axes", {})
    
    axes_str = ", ".join(f"{k}={v}" for k, v in sorted(axes.items()))
    
    log.info(f"[SWEEP] Combination {idx}/{total}: {axes_str}")


def get_sweep_columns(sweep_metadata: Dict) -> Dict[str, Any]:
    """
    Get sweep parameters as flat columns for metrics output.
    
    Args:
        sweep_metadata: The _sweep_metadata dict
        
    Returns:
        Dict with flattened sweep parameters for CSV/JSON output
    """
    result = {
        "sweep_index": sweep_metadata.get("index", 0),
        "sweep_total": sweep_metadata.get("total", 1),
        "is_sweep": sweep_metadata.get("is_sweep", False),
    }
    
    # Add each axis as a separate column
    axes = sweep_metadata.get("axes", {})
    for path, value in axes.items():
        # Use dotted path as column name, replacing dots with underscores
        col_name = f"sweep_{path.replace('.', '_')}"
        result[col_name] = value
    
    return result
