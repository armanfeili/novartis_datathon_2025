"""
Model training pipeline for Novartis Datathon 2025.

Unified training with sample weights aligned to official metric.
Handles feature/target/meta separation to prevent leakage.

Section 5 Implementation:
- 5.1: Core training with CLI interface
- 5.2: CLI consistency with --help documentation
- 5.3: Experiment metadata logging (git hash, config hash)
- 5.4: Sample weights refinement (time/bucket weights, metric-aligned)
- 5.5: Hyperparameter optimization with Optuna integration
- 5.7: Training workflow improvements (checkpointing, memory profiling)
"""

import argparse
import hashlib
import logging
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List, Union, Callable

import numpy as np
import pandas as pd
import yaml

# Optional experiment tracking imports
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from .utils import setup_logging, load_config, set_seed, timer, get_project_root
from .data import (
    load_raw_data, prepare_base_panel, compute_pre_entry_stats, 
    handle_missing_values, META_COLS, ID_COLS, TIME_COL,
    get_panel, verify_no_future_leakage
)
from .features import make_features, select_training_rows, _normalize_scenario, get_features, prune_features_by_importance
from .validation import create_validation_split, get_fold_series
from .evaluate import (
    compute_metric1, compute_metric2, create_aux_file,
    make_metric_record, save_metric_records,
    METRIC_NAME_S1, METRIC_NAME_S2, METRIC_NAME_RMSE, METRIC_NAME_MAE
)
from .config_sweep import (
    expand_sweep, get_sweep_axes, build_sweep_run_id,
    get_config_by_id, apply_config_overrides, get_active_config,
    generate_sweep_runs, SweepResultsLogger, get_model_filename, 
    get_submission_filename, deep_merge
)

logger = logging.getLogger(__name__)

# Re-export META_COLS from data.py for backward compatibility
# These columns are NEVER used as model features
# Canonical definition is in src/data.py
TARGET_COL = 'y_norm'

# ==============================================================================
# EXPERIMENT TRACKING (Section 5.1)
# ==============================================================================

class ExperimentTracker:
    """
    Unified experiment tracking interface supporting MLflow and W&B.
    
    Provides a consistent API for logging metrics, parameters, and artifacts
    across different tracking backends. Falls back gracefully when backends
    are not available.
    
    Example usage:
        tracker = ExperimentTracker(backend='mlflow', experiment_name='my_exp')
        tracker.start_run(run_name='catboost_s1')
        tracker.log_params({'lr': 0.03, 'depth': 6})
        tracker.log_metrics({'rmse': 0.15, 'mae': 0.12})
        tracker.log_artifact('/path/to/model.bin')
        tracker.end_run()
    """
    
    def __init__(
        self,
        backend: Optional[str] = None,
        experiment_name: str = 'novartis-datathon-2025',
        tracking_uri: Optional[str] = None,
        project_name: Optional[str] = None,  # For W&B
        enabled: bool = True
    ):
        """
        Initialize experiment tracker.
        
        Args:
            backend: 'mlflow', 'wandb', or None (disabled)
            experiment_name: Name for MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to local ./mlruns)
            project_name: W&B project name (defaults to experiment_name)
            enabled: Whether tracking is enabled
        """
        self.backend = backend.lower() if backend else None
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.project_name = project_name or experiment_name
        self.enabled = enabled
        self._run_active = False
        self._run_id = None
        
        if not self.enabled or self.backend is None:
            logger.info("Experiment tracking disabled")
            return
        
        if self.backend == 'mlflow':
            if not MLFLOW_AVAILABLE:
                logger.warning("MLflow not installed. Tracking disabled.")
                self.enabled = False
                return
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking enabled for experiment: {experiment_name}")
            
        elif self.backend == 'wandb':
            if not WANDB_AVAILABLE:
                logger.warning("W&B not installed. Tracking disabled.")
                self.enabled = False
                return
            logger.info(f"W&B tracking enabled for project: {self.project_name}")
        else:
            logger.warning(f"Unknown backend: {self.backend}. Tracking disabled.")
            self.enabled = False
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Start a new tracking run.
        
        Args:
            run_name: Name for this run
            tags: Dictionary of tags
            config: Configuration dictionary (logged as params in W&B)
            
        Returns:
            Run ID if available
        """
        if not self.enabled:
            return None
        
        if self.backend == 'mlflow':
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self._run_id = run.info.run_id
            self._run_active = True
            logger.info(f"Started MLflow run: {run_name} (ID: {self._run_id})")
            return self._run_id
            
        elif self.backend == 'wandb':
            wandb.init(
                project=self.project_name,
                name=run_name,
                tags=list(tags.values()) if tags else None,
                config=config
            )
            self._run_id = wandb.run.id if wandb.run else None
            self._run_active = True
            logger.info(f"Started W&B run: {run_name}")
            return self._run_id
        
        return None
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters/hyperparameters."""
        if not self.enabled or not self._run_active:
            return
        
        if self.backend == 'mlflow':
            # MLflow has param length limits, so truncate if needed
            for key, value in params.items():
                str_value = str(value)[:500]  # MLflow limit
                mlflow.log_param(key, str_value)
                
        elif self.backend == 'wandb':
            wandb.config.update(params)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics."""
        if not self.enabled or not self._run_active:
            return
        
        if self.backend == 'mlflow':
            mlflow.log_metrics(metrics, step=step)
            
        elif self.backend == 'wandb':
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log a file artifact."""
        if not self.enabled or not self._run_active:
            return
        
        if not Path(artifact_path).exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return
        
        if self.backend == 'mlflow':
            mlflow.log_artifact(artifact_path)
            
        elif self.backend == 'wandb':
            artifact = wandb.Artifact(
                name=artifact_name or Path(artifact_path).stem,
                type='model'
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
    
    def log_figure(self, fig, name: str):
        """Log a matplotlib figure."""
        if not self.enabled or not self._run_active:
            return
        
        if self.backend == 'mlflow':
            mlflow.log_figure(fig, f"{name}.png")
            
        elif self.backend == 'wandb':
            wandb.log({name: wandb.Image(fig)})
    
    def end_run(self, status: str = 'FINISHED'):
        """End the current run."""
        if not self.enabled or not self._run_active:
            return
        
        if self.backend == 'mlflow':
            mlflow.end_run(status=status)
            
        elif self.backend == 'wandb':
            wandb.finish()
        
        self._run_active = False
        logger.info(f"Ended tracking run (status: {status})")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = 'FAILED' if exc_type else 'FINISHED'
        self.end_run(status=status)
        return False  # Don't suppress exceptions


def setup_experiment_tracking(
    run_config: dict,
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[ExperimentTracker]:
    """
    Setup experiment tracking from run_config with optional auto-start.
    
    Section 13.1: Provides consistent experiment tracking setup across all
    training pipelines with standard run naming and tagging.
    
    Args:
        run_config: Run configuration dictionary
        run_name: Name for the run (pattern: YYYY-mm-dd_HH-MM_<model>_scenario<1|2>[_suffix])
        tags: Optional dictionary of tags (e.g., config_hash, git_commit)
        config: Optional config dict to log as parameters
        
    Returns:
        ExperimentTracker instance or None if tracking is disabled
        
    Example:
        tracker = setup_experiment_tracking(
            run_config,
            run_name='2025-01-15_14-30_catboost_scenario1',
            tags={'config_hash': 'abc123', 'git_commit': 'def456'},
            config={'model': 'catboost', 'scenario': 1}
        )
        if tracker:
            with tracker:
                # ... training code ...
                tracker.log_metrics({'official_metric': 0.15})
    """
    tracking_config = run_config.get('experiment_tracking', {})
    
    if not tracking_config.get('enabled', False):
        return None
    
    backend = tracking_config.get('backend')
    experiment_name = tracking_config.get('experiment_name', 'novartis-datathon-2025')
    tracking_uri = tracking_config.get('tracking_uri')
    project_name = tracking_config.get('project_name')
    
    tracker = ExperimentTracker(
        backend=backend,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        project_name=project_name,
        enabled=True
    )
    
    return tracker


def generate_run_name(
    model_type: str,
    scenario: int,
    suffix: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """
    Generate a consistent run name following the project convention.
    
    Section 13.1: Standard run naming pattern for all experiments.
    
    Pattern: YYYY-mm-dd_HH-MM_<model_type>_scenario<1|2>[_suffix]
    
    Args:
        model_type: Type of model (catboost, lightgbm, xgboost, etc.)
        scenario: Scenario number (1 or 2)
        suffix: Optional suffix to append (e.g., 'cv', 'hpo', 'sweep')
        timestamp: Optional timestamp string (defaults to current time)
        
    Returns:
        Formatted run name string
        
    Example:
        >>> generate_run_name('catboost', 1)
        '2025-01-15_14-30_catboost_scenario1'
        >>> generate_run_name('catboost', 2, suffix='cv5')
        '2025-01-15_14-30_catboost_scenario2_cv5'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    run_name = f"{timestamp}_{model_type}_scenario{scenario}"
    
    if suffix:
        run_name = f"{run_name}_{suffix}"
    
    return run_name


def get_standard_tags(
    run_config: Dict[str, Any],
    model_config: Optional[Dict[str, Any]] = None,
    config_hash: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate standard tags for experiment tracking.
    
    Section 13.1: Consistent tagging across all tracked experiments.
    
    Args:
        run_config: Run configuration dictionary
        model_config: Model configuration dictionary
        config_hash: Pre-computed config hash
        
    Returns:
        Dictionary of standard tags
    """
    tags = {
        'git_commit': get_git_commit_hash() or 'unknown',
        'seed': str(run_config.get('reproducibility', {}).get('seed', 42)),
    }
    
    if config_hash:
        tags['config_hash'] = config_hash
    
    if model_config:
        active_config = model_config.get('_active_config_id')
        if active_config:
            tags['config_id'] = active_config
    
    return tags


def validate_training_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scenario: int,
    min_train_rows: int = 100,
    min_val_rows: int = 20,
    min_train_series: int = 3,
    min_val_series: int = 1
) -> None:
    """
    Validate training and validation data before model training.
    
    Section 25.1: Safety guards to detect empty or insufficient data
    with clear error messages.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        scenario: Scenario number for context in error messages
        min_train_rows: Minimum required training rows
        min_val_rows: Minimum required validation rows
        min_train_series: Minimum required unique training series
        min_val_series: Minimum required unique validation series
        
    Raises:
        ValueError: If data is empty or below minimum thresholds
        
    Example:
        validate_training_data(train_df, val_df, scenario=1)
    """
    # Check for empty data
    if train_df is None or len(train_df) == 0:
        raise ValueError(
            f"Scenario {scenario}: Training data is empty. "
            "Check data loading and filtering steps."
        )
    
    if val_df is None or len(val_df) == 0:
        raise ValueError(
            f"Scenario {scenario}: Validation data is empty. "
            "Check validation split configuration (val_fraction may be too small)."
        )
    
    # Check minimum rows
    if len(train_df) < min_train_rows:
        raise ValueError(
            f"Scenario {scenario}: Training data has only {len(train_df)} rows "
            f"(minimum: {min_train_rows}). Consider using more data or reducing val_fraction."
        )
    
    if len(val_df) < min_val_rows:
        raise ValueError(
            f"Scenario {scenario}: Validation data has only {len(val_df)} rows "
            f"(minimum: {min_val_rows}). Consider increasing val_fraction."
        )
    
    # Check minimum series
    id_cols = ['country', 'brand_name']
    if all(col in train_df.columns for col in id_cols):
        n_train_series = train_df[id_cols].drop_duplicates().shape[0]
        n_val_series = val_df[id_cols].drop_duplicates().shape[0]
        
        if n_train_series < min_train_series:
            raise ValueError(
                f"Scenario {scenario}: Training data has only {n_train_series} unique series "
                f"(minimum: {min_train_series}). Need more diverse training data."
            )
        
        if n_val_series < min_val_series:
            raise ValueError(
                f"Scenario {scenario}: Validation data has only {n_val_series} unique series "
                f"(minimum: {min_val_series}). Validation may not be representative."
            )
        
        # Log data statistics
        logger.info(f"Scenario {scenario} data validation passed:")
        logger.info(f"  Training: {len(train_df)} rows, {n_train_series} series")
        logger.info(f"  Validation: {len(val_df)} rows, {n_val_series} series")


def ensure_reproducibility(
    run_config: Dict[str, Any],
    artifacts_dir: Optional[Path] = None
) -> str:
    """
    Ensure reproducibility by setting seed and creating config snapshot.
    
    Section 25.2: Consistent reproducibility setup across all entrypoints.
    
    Args:
        run_config: Run configuration dictionary
        artifacts_dir: Optional directory to save reproducibility info
        
    Returns:
        The seed value that was set
        
    Side Effects:
        - Calls set_seed() with the configured seed
        - If artifacts_dir provided, saves reproducibility_info.json
    """
    seed = run_config.get('reproducibility', {}).get('seed', 42)
    set_seed(seed)
    
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        repro_info = {
            'seed': seed,
            'git_commit': get_git_commit_hash(),
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
        }
        
        with open(artifacts_dir / 'reproducibility_info.json', 'w') as f:
            json.dump(repro_info, f, indent=2)
    
    logger.debug(f"Reproducibility configured with seed={seed}")
    return str(seed)


# ==============================================================================
# CHECKPOINT SAVING (Section 5.1)
# ==============================================================================

class TrainingCheckpoint:
    """
    Checkpoint manager for saving and resuming training.
    
    Supports saving/loading:
    - Model state
    - Training state (epoch, step, best score)
    - Optimizer state (if applicable)
    - Sample weights
    - Configuration
    
    Example usage:
        checkpoint = TrainingCheckpoint(checkpoint_dir='artifacts/checkpoints')
        
        # Save during training
        checkpoint.save(
            model=model,
            epoch=10,
            step=1000,
            metrics={'rmse': 0.15},
            config=config_dict
        )
        
        # Resume training
        state = checkpoint.load_latest()
        model = state['model']
        start_epoch = state['epoch'] + 1
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_best_n: int = 3,
        metric_name: str = 'official_metric',
        minimize: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
            metric_name: Metric to track for best model
            minimize: If True, lower metric is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.metric_name = metric_name
        self.minimize = minimize
        self._checkpoint_history: List[Dict] = []
    
    def save(
        self,
        model: Any,
        epoch: int,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None,
        sample_weights: Optional[np.ndarray] = None,
        additional_state: Optional[Dict] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save a training checkpoint.
        
        Args:
            model: Model instance (must have .save() method)
            epoch: Current epoch number
            step: Current step number
            metrics: Dictionary of current metrics
            config: Training configuration
            sample_weights: Sample weights array
            additional_state: Any additional state to save
            is_best: If True, also save as best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch:04d}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / "model.bin"
        model.save(str(model_path))
        
        # Save training state
        state = {
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {},
            'timestamp': timestamp,
            'checkpoint_name': checkpoint_name
        }
        
        if config:
            state['config'] = config
        
        if sample_weights is not None:
            weights_path = checkpoint_path / "sample_weights.npy"
            np.save(weights_path, sample_weights)
            state['sample_weights_path'] = str(weights_path)
        
        if additional_state:
            state['additional'] = additional_state
        
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Update history
        self._checkpoint_history.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics or {},
            'timestamp': timestamp
        })
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save as best if requested
        if is_best:
            best_path = self.checkpoint_dir / "best"
            if best_path.exists():
                shutil.rmtree(best_path)
            shutil.copytree(checkpoint_path, best_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        model_class: Optional[type] = None,
        model_config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model_class: Model class to instantiate
            model_config: Configuration for model instantiation
            
        Returns:
            Dictionary with loaded state
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load training state
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        # Load model
        model_path = checkpoint_path / "model.bin"
        if model_class is not None:
            model = model_class(model_config or {})
            model.load(str(model_path))
            state['model'] = model
        else:
            state['model_path'] = str(model_path)
        
        # Load sample weights if available
        if 'sample_weights_path' in state:
            weights_path = Path(state['sample_weights_path'])
            if weights_path.exists():
                state['sample_weights'] = np.load(weights_path)
        
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        return state
    
    def load_latest(
        self,
        model_class: Optional[type] = None,
        model_config: Optional[dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not checkpoints:
            logger.warning("No checkpoints found")
            return None
        
        return self.load(checkpoints[0], model_class, model_config)
    
    def load_best(
        self,
        model_class: Optional[type] = None,
        model_config: Optional[dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best"
        
        if not best_path.exists():
            logger.warning("No best checkpoint found")
            return None
        
        return self.load(best_path, model_class, model_config)
    
    def get_checkpoint_history(self) -> List[Dict]:
        """Get list of all saved checkpoints."""
        return self._checkpoint_history.copy()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only best N by metric."""
        if self.keep_best_n <= 0:
            return
        
        # Sort by metric
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*"),
            key=lambda p: self._get_checkpoint_metric(p),
            reverse=not self.minimize
        )
        
        # Keep best N + best
        to_keep = set()
        for cp in checkpoints[:self.keep_best_n]:
            to_keep.add(cp.name)
        
        # Remove old checkpoints
        for cp in checkpoints[self.keep_best_n:]:
            if cp.name not in to_keep and cp.name != "best":
                shutil.rmtree(cp)
                logger.debug(f"Removed old checkpoint: {cp}")
    
    def _get_checkpoint_metric(self, checkpoint_path: Path) -> float:
        """Get metric value from checkpoint for sorting."""
        state_path = checkpoint_path / "training_state.json"
        if not state_path.exists():
            return float('inf') if self.minimize else float('-inf')
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            return state.get('metrics', {}).get(self.metric_name, 
                float('inf') if self.minimize else float('-inf'))
        except Exception:
            return float('inf') if self.minimize else float('-inf')
    
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*"))
        return len(checkpoints) > 0 or (self.checkpoint_dir / "best").exists()
    
    def get_best_metric(self) -> Optional[float]:
        """Get the best metric value from all checkpoints."""
        best_path = self.checkpoint_dir / "best"
        if best_path.exists():
            return self._get_checkpoint_metric(best_path)
        return None
    
    def save_scenario_checkpoint(
        self,
        model: Any,
        scenario: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
        model_type: str = 'catboost'
    ) -> Path:
        """
        Save a checkpoint for a specific scenario.
        
        Section 14.2: Per-scenario checkpointing with consistent naming.
        
        Args:
            model: Trained model instance
            scenario: Scenario number (1 or 2)
            metrics: Metrics dictionary including 'official_metric'
            config: Optional configuration dictionary
            model_type: Type of model for naming
            
        Returns:
            Path to saved checkpoint directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"scenario{scenario}_{model_type}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / f"model_scenario{scenario}.bin"
        model.save(str(model_path))
        
        # Save state
        state = {
            'scenario': scenario,
            'model_type': model_type,
            'metrics': metrics,
            'timestamp': timestamp,
            'checkpoint_name': checkpoint_name
        }
        
        if config:
            state['config'] = config
        
        state_path = checkpoint_path / "scenario_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Check if this is the best for this scenario
        official_metric = metrics.get('official_metric', float('inf'))
        best_scenario_path = self.checkpoint_dir / f"best_scenario{scenario}"
        
        should_save_best = True
        if best_scenario_path.exists():
            prev_best = self._get_checkpoint_metric(best_scenario_path)
            if self.minimize:
                should_save_best = official_metric < prev_best
            else:
                should_save_best = official_metric > prev_best
        
        if should_save_best:
            if best_scenario_path.exists():
                shutil.rmtree(best_scenario_path)
            shutil.copytree(checkpoint_path, best_scenario_path)
            logger.info(f"New best for scenario {scenario}: {official_metric:.4f}")
        
        logger.info(f"Saved scenario {scenario} checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_scenario_checkpoint(
        self,
        scenario: int,
        model_class: Optional[type] = None,
        model_config: Optional[dict] = None,
        load_best: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for a specific scenario.
        
        Section 14.3: Resume from scenario-specific checkpoint.
        
        Args:
            scenario: Scenario number (1 or 2)
            model_class: Model class to instantiate
            model_config: Configuration for model instantiation
            load_best: If True, load best checkpoint; otherwise load latest
            
        Returns:
            Dictionary with loaded state or None if not found
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / f"best_scenario{scenario}"
        else:
            # Find latest checkpoint for this scenario
            pattern = f"scenario{scenario}_*"
            checkpoints = sorted(
                self.checkpoint_dir.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            checkpoint_path = checkpoints[0] if checkpoints else None
        
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.info(f"No checkpoint found for scenario {scenario}")
            return None
        
        # Load state
        state_path = checkpoint_path / "scenario_state.json"
        if not state_path.exists():
            logger.warning(f"Checkpoint state not found: {state_path}")
            return None
        
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        # Load model
        model_path = checkpoint_path / f"model_scenario{scenario}.bin"
        if model_class is not None and model_path.exists():
            model = model_class(model_config or {})
            model.load(str(model_path))
            state['model'] = model
        else:
            state['model_path'] = str(model_path)
        
        logger.info(f"Loaded scenario {scenario} checkpoint from: {checkpoint_path}")
        return state
    
    def cleanup_scenario_checkpoints(self, scenario: int, keep_best: bool = True):
        """
        Clean up old checkpoints for a specific scenario.
        
        Section 14.4: Cleanup logic that preserves best checkpoints.
        
        Args:
            scenario: Scenario number to clean up
            keep_best: If True, keep the best_scenarioN directory
        """
        pattern = f"scenario{scenario}_*"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        for cp in checkpoints:
            if cp.is_dir() and cp.name != f"best_scenario{scenario}":
                try:
                    shutil.rmtree(cp)
                    logger.debug(f"Removed old scenario {scenario} checkpoint: {cp}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {cp}: {e}")
        
        if not keep_best:
            best_path = self.checkpoint_dir / f"best_scenario{scenario}"
            if best_path.exists():
                try:
                    shutil.rmtree(best_path)
                    logger.debug(f"Removed best scenario {scenario} checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to remove best checkpoint: {e}")


def load_checkpoint_if_requested(
    checkpoint_dir: Union[str, Path],
    scenario: int,
    resume_checkpoint: bool = False,
    model_class: Optional[type] = None,
    model_config: Optional[dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint if resume is requested.
    
    Section 14.3: Helper function for resume from checkpoint capability.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        scenario: Scenario number to resume
        resume_checkpoint: If True, attempt to load checkpoint
        model_class: Model class for instantiation
        model_config: Configuration for model instantiation
        
    Returns:
        Loaded checkpoint state or None if not resuming
    """
    if not resume_checkpoint:
        return None
    
    checkpoint_mgr = TrainingCheckpoint(checkpoint_dir=checkpoint_dir)
    return checkpoint_mgr.load_scenario_checkpoint(
        scenario=scenario,
        model_class=model_class,
        model_config=model_config,
        load_best=True
    )


# ==============================================================================
# WEIGHT TRANSFORMATIONS (Section 5.4)
# ==============================================================================

def transform_weights(
    weights: pd.Series,
    transformation: str = 'identity',
    clip_min: float = 0.01,
    clip_max: float = 100.0
) -> pd.Series:
    """
    Apply transformation to sample weights.
    
    Available transformations:
    - 'identity': No transformation
    - 'sqrt': Square root transformation (reduces extreme values)
    - 'log': Log transformation (stronger reduction of extremes)
    - 'softmax': Softmax normalization (sums to 1)
    - 'rank': Rank-based transformation (uniform spread)
    
    Args:
        weights: Input sample weights
        transformation: Type of transformation to apply
        clip_min: Minimum weight value after transformation
        clip_max: Maximum weight value after transformation
        
    Returns:
        Transformed weights
    """
    if transformation == 'identity':
        transformed = weights.copy()
        
    elif transformation == 'sqrt':
        # Square root reduces extreme values
        transformed = np.sqrt(weights)
        
    elif transformation == 'log':
        # Log transformation (add small epsilon to avoid log(0))
        transformed = np.log1p(weights)
        
    elif transformation == 'softmax':
        # Softmax normalization
        exp_w = np.exp(weights - weights.max())  # Subtract max for numerical stability
        transformed = exp_w / exp_w.sum()
        
    elif transformation == 'rank':
        # Rank-based transformation - creates uniform spread
        ranks = weights.rank(method='average')
        transformed = ranks / len(ranks)
        
    else:
        raise ValueError(f"Unknown weight transformation: {transformation}")
    
    # Clip and normalize
    transformed = transformed.clip(lower=clip_min, upper=clip_max)
    
    # Normalize so weights sum to len(weights)
    transformed = transformed * len(transformed) / (transformed.sum() + 1e-8)
    
    logger.debug(f"Weight transformation '{transformation}': "
                f"min={transformed.min():.4f}, max={transformed.max():.4f}, "
                f"mean={transformed.mean():.4f}")
    
    return transformed


def validate_weights_correlation(
    weights: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    meta_df: pd.DataFrame,
    scenario: int,
    n_bootstrap: int = 100
) -> Dict[str, Any]:
    """
    Validate that sample weights correlate with metric improvement.
    
    Computes metrics with different weight configurations to verify
    that the chosen weights actually improve the target metric.
    
    Args:
        weights: Sample weights to validate
        y_true: True target values
        y_pred: Predicted values
        meta_df: Metadata DataFrame (with bucket, months_postgx)
        scenario: Scenario number (1 or 2)
        n_bootstrap: Number of bootstrap samples for confidence intervals
        
    Returns:
        Dictionary with validation results
    """
    from .evaluate import compute_metric1, compute_metric2
    
    results = {
        'scenario': scenario,
        'n_samples': len(weights),
        'weight_stats': {
            'min': float(weights.min()),
            'max': float(weights.max()),
            'mean': float(weights.mean()),
            'std': float(weights.std()),
        }
    }
    
    # Compute weighted vs unweighted RMSE
    weighted_rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights))
    unweighted_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    results['weighted_rmse'] = weighted_rmse
    results['unweighted_rmse'] = unweighted_rmse
    results['rmse_ratio'] = weighted_rmse / (unweighted_rmse + 1e-8)
    
    # Compute per-bucket weighted errors
    if 'bucket' in meta_df.columns:
        for bucket in [1, 2]:
            mask = meta_df['bucket'] == bucket
            if mask.sum() > 0:
                bucket_rmse = np.sqrt(np.average(
                    (y_true[mask] - y_pred[mask]) ** 2,
                    weights=weights[mask]
                ))
                results[f'bucket{bucket}_weighted_rmse'] = bucket_rmse
    
    # Compute per-time-window weighted errors
    if 'months_postgx' in meta_df.columns:
        months = meta_df['months_postgx']
        windows = [
            ('0_5', (months >= 0) & (months <= 5)),
            ('6_11', (months >= 6) & (months <= 11)),
            ('12_23', (months >= 12) & (months <= 23)),
        ]
        
        for name, mask in windows:
            if mask.sum() > 0:
                window_rmse = np.sqrt(np.average(
                    (y_true[mask] - y_pred[mask]) ** 2,
                    weights=weights[mask]
                ))
                results[f'window_{name}_weighted_rmse'] = window_rmse
    
    # Weight efficiency: correlation between weight and absolute error
    abs_errors = np.abs(y_true - y_pred)
    weight_error_corr = np.corrcoef(weights, abs_errors)[0, 1]
    results['weight_error_correlation'] = weight_error_corr
    
    # Higher correlation means weights focus on high-error regions
    # Negative correlation means weights focus on low-error regions (desired)
    results['interpretation'] = (
        'Good' if weight_error_corr < 0 else 
        'Neutral' if abs(weight_error_corr) < 0.1 else 
        'Review weights'
    )
    
    logger.info(f"Weight validation: RMSE ratio={results['rmse_ratio']:.4f}, "
               f"weight-error correlation={weight_error_corr:.4f}")
    
    return results


# ==============================================================================
# HYPERPARAMETER OPTIMIZATION (Section 5.5)
# ==============================================================================

def get_default_search_space(model_type: str) -> Dict[str, List]:
    """
    Get default search space for hyperparameter optimization.
    
    Section 16.1.2: Default search spaces for supported model types.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary with parameter ranges [min, max]
    """
    search_spaces = {
        'catboost': {
            'depth': [4, 8],
            'learning_rate': [0.01, 0.1],
            'l2_leaf_reg': [1.0, 10.0],
            'min_data_in_leaf': [10, 50],
            'random_strength': [0.0, 5.0],
            'bagging_temperature': [0.0, 5.0],
        },
        'lightgbm': {
            'num_leaves': [15, 63],
            'learning_rate': [0.01, 0.1],
            'min_data_in_leaf': [10, 50],
            'feature_fraction': [0.6, 1.0],
            'bagging_fraction': [0.6, 1.0],
        },
        'xgboost': {
            'max_depth': [4, 8],
            'learning_rate': [0.01, 0.1],
            'min_child_weight': [1, 10],
            'colsample_bytree': [0.6, 1.0],
            'subsample': [0.6, 1.0],
        }
    }
    return search_spaces.get(model_type, {})


def merge_search_space(
    user_search_space: Optional[Dict[str, List]],
    model_type: str
) -> Dict[str, List]:
    """
    Merge user-provided search space with defaults.
    
    Section 16.1.2: User can override only some ranges, rest use defaults.
    
    Args:
        user_search_space: User-provided search space (optional)
        model_type: Type of model
        
    Returns:
        Merged search space dictionary
    """
    default_space = get_default_search_space(model_type)
    
    if user_search_space is None:
        return default_space
    
    # Merge: user overrides defaults
    merged = default_space.copy()
    merged.update(user_search_space)
    
    return merged


def create_optuna_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario: int,
    model_type: str = 'catboost',
    search_space: Optional[Dict] = None,
    run_config: Optional[Dict] = None,
    disable_tracking: bool = True
) -> Callable:
    """
    Create an Optuna objective function for hyperparameter optimization.
    
    Section 16.1.1: Uses train_scenario_model with model_config={'params': params}.
    Experiment tracking is disabled by default to avoid excessive logging.
    
    Args:
        X_train, y_train, meta_train: Training data
        X_val, y_val, meta_val: Validation data
        scenario: Scenario number
        model_type: Type of model to tune
        search_space: Custom search space dict (merged with defaults)
        run_config: Run configuration
        disable_tracking: If True, disable experiment tracking inside objective
        
    Returns:
        Objective function for Optuna
    """
    # Merge user search space with defaults
    merged_search_space = merge_search_space(search_space, model_type)
    
    # Create run_config copy without tracking if needed
    hpo_run_config = run_config.copy() if run_config else {}
    if disable_tracking:
        hpo_run_config['experiment_tracking'] = {'enabled': False}
    
    def objective(trial: 'optuna.Trial') -> float:
        """Optuna objective function."""
        # Define hyperparameters based on model type
        if model_type == 'catboost':
            params = {
                'depth': trial.suggest_int('depth', 
                    merged_search_space.get('depth', [4, 8])[0],
                    merged_search_space.get('depth', [4, 8])[1]
                ),
                'learning_rate': trial.suggest_float('learning_rate',
                    merged_search_space.get('learning_rate', [0.01, 0.1])[0],
                    merged_search_space.get('learning_rate', [0.01, 0.1])[1],
                    log=True
                ),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg',
                    merged_search_space.get('l2_leaf_reg', [1.0, 10.0])[0],
                    merged_search_space.get('l2_leaf_reg', [1.0, 10.0])[1],
                    log=True
                ),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                    merged_search_space.get('min_data_in_leaf', [10, 50])[0],
                    merged_search_space.get('min_data_in_leaf', [10, 50])[1]
                ),
                'random_strength': trial.suggest_float('random_strength',
                    merged_search_space.get('random_strength', [0.0, 5.0])[0],
                    merged_search_space.get('random_strength', [0.0, 5.0])[1]
                ),
                'bagging_temperature': trial.suggest_float('bagging_temperature',
                    merged_search_space.get('bagging_temperature', [0.0, 5.0])[0],
                    merged_search_space.get('bagging_temperature', [0.0, 5.0])[1]
                ),
            }
            
        elif model_type == 'lightgbm':
            params = {
                'num_leaves': trial.suggest_int('num_leaves',
                    merged_search_space.get('num_leaves', [15, 63])[0],
                    merged_search_space.get('num_leaves', [15, 63])[1]
                ),
                'learning_rate': trial.suggest_float('learning_rate',
                    merged_search_space.get('learning_rate', [0.01, 0.1])[0],
                    merged_search_space.get('learning_rate', [0.01, 0.1])[1],
                    log=True
                ),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                    merged_search_space.get('min_data_in_leaf', [10, 50])[0],
                    merged_search_space.get('min_data_in_leaf', [10, 50])[1]
                ),
                'feature_fraction': trial.suggest_float('feature_fraction',
                    merged_search_space.get('feature_fraction', [0.6, 1.0])[0],
                    merged_search_space.get('feature_fraction', [0.6, 1.0])[1]
                ),
                'bagging_fraction': trial.suggest_float('bagging_fraction',
                    merged_search_space.get('bagging_fraction', [0.6, 1.0])[0],
                    merged_search_space.get('bagging_fraction', [0.6, 1.0])[1]
                ),
            }
            
        elif model_type == 'xgboost':
            params = {
                'max_depth': trial.suggest_int('max_depth',
                    merged_search_space.get('max_depth', [4, 8])[0],
                    merged_search_space.get('max_depth', [4, 8])[1]
                ),
                'learning_rate': trial.suggest_float('learning_rate',
                    merged_search_space.get('learning_rate', [0.01, 0.1])[0],
                    merged_search_space.get('learning_rate', [0.01, 0.1])[1],
                    log=True
                ),
                'min_child_weight': trial.suggest_int('min_child_weight',
                    merged_search_space.get('min_child_weight', [1, 10])[0],
                    merged_search_space.get('min_child_weight', [1, 10])[1]
                ),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                    merged_search_space.get('colsample_bytree', [0.6, 1.0])[0],
                    merged_search_space.get('colsample_bytree', [0.6, 1.0])[1]
                ),
                'subsample': trial.suggest_float('subsample',
                    merged_search_space.get('subsample', [0.6, 1.0])[0],
                    merged_search_space.get('subsample', [0.6, 1.0])[1]
                ),
            }
        else:
            raise ValueError(f"HPO not supported for model type: {model_type}")
        
        # Train model with suggested parameters
        try:
            model, metrics = train_scenario_model(
                X_train, y_train, meta_train,
                X_val, y_val, meta_val,
                scenario=scenario,
                model_type=model_type,
                model_config={'params': params},
                run_config=hpo_run_config
            )
            
            # Return official metric (to minimize)
            metric_value = metrics.get('official_metric', np.inf)
            
            # Report intermediate value for pruning
            trial.report(metric_value, step=0)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return metric_value
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return np.inf
    
    return objective


def run_hyperparameter_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario: int,
    model_type: str = 'catboost',
    n_trials: int = 100,
    timeout: Optional[int] = 3600,
    search_space: Optional[Dict] = None,
    run_config: Optional[Dict] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    artifacts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization using Optuna.
    
    Section 16.2: HPO runner with configurable n_trials, timeout, and search space.
    Results are saved to artifacts_dir for later use.
    
    Args:
        X_train, y_train, meta_train: Training data
        X_val, y_val, meta_val: Validation data  
        scenario: Scenario number
        model_type: Type of model to tune
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds (None for no limit)
        search_space: Custom search space dict (merged with defaults)
        run_config: Run configuration
        study_name: Name for Optuna study
        storage: Storage URL for distributed optimization
        artifacts_dir: Directory to save HPO results
        
    Returns:
        Dictionary with best parameters and optimization history
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for hyperparameter optimization. "
            "Install with: pip install optuna"
        )
    
    # Merge user search space with defaults
    merged_search_space = merge_search_space(search_space, model_type)
    
    # Create study
    study_name = study_name or f"{model_type}_scenario{scenario}_{datetime.now():%Y%m%d_%H%M%S}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    )
    
    # Create objective with merged search space
    objective = create_optuna_objective(
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        scenario=scenario,
        model_type=model_type,
        search_space=merged_search_space,
        run_config=run_config
    )
    
    logger.info(f"Starting HPO with {n_trials} trials, timeout={timeout}s")
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # Collect results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'study_name': study_name,
        'model_type': model_type,
        'scenario': scenario,
        'search_space': search_space,
    }
    
    # Save results if artifacts_dir provided
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best params
        best_params_path = artifacts_dir / 'best_params.yaml'
        with open(best_params_path, 'w') as f:
            yaml.dump(results['best_params'], f)
        
        # Save full results
        results_path = artifacts_dir / 'hpo_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save trial history
        history = []
        for trial in study.trials:
            history.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
            })
        
        history_path = artifacts_dir / 'trial_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"HPO results saved to {artifacts_dir}")
    
    logger.info(f"HPO complete. Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return results


def is_hpo_enabled(run_config: Dict[str, Any]) -> bool:
    """
    Check if HPO is enabled in run configuration.
    
    Section 16.2.3: Simple boolean flag for HPO enablement.
    
    Checks for `hpo.enabled` in run_config.
    
    Args:
        run_config: Run configuration dictionary
        
    Returns:
        True if HPO is enabled
    """
    hpo_config = run_config.get('hpo', {})
    return hpo_config.get('enabled', False)


def get_hpo_settings(run_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get HPO settings from run configuration.
    
    Section 16.2.3: Extract HPO settings with defaults.
    
    Args:
        run_config: Run configuration dictionary
        
    Returns:
        Dictionary with n_trials, timeout, and other HPO settings
    """
    hpo_config = run_config.get('hpo', {})
    return {
        'enabled': hpo_config.get('enabled', False),
        'n_trials': hpo_config.get('n_trials', 50),
        'timeout': hpo_config.get('timeout', 3600),
        'search_space': hpo_config.get('search_space'),
    }


def propagate_hpo_best_params(
    model_config: Dict[str, Any],
    hpo_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Propagate best HPO parameters into model configuration.
    
    Section 16.2.2: After HPO, merge best params back into model_config.
    
    Args:
        model_config: Original model configuration
        hpo_results: Results from run_hyperparameter_optimization
        
    Returns:
        Updated model configuration with best params
    """
    if not hpo_results or 'best_params' not in hpo_results:
        return model_config
    
    updated_config = model_config.copy()
    
    # Merge best params into params section
    if 'params' not in updated_config:
        updated_config['params'] = {}
    
    updated_config['params'].update(hpo_results['best_params'])
    
    # Add metadata about HPO source
    updated_config['_hpo_applied'] = True
    updated_config['_hpo_best_value'] = hpo_results.get('best_value')
    updated_config['_hpo_n_trials'] = hpo_results.get('n_trials')
    
    logger.info(f"Applied HPO best params to model config: {hpo_results['best_params']}")
    
    return updated_config


# ==============================================================================
# MEMORY PROFILING (Section 5.7)
# ==============================================================================

class MemoryProfiler:
    """
    Memory profiler for tracking memory usage during training.
    
    Uses tracemalloc for detailed memory tracking.
    
    Example usage:
        profiler = MemoryProfiler()
        profiler.start()
        
        # ... training code ...
        profiler.snapshot("after_data_load")
        
        # ... more training ...
        profiler.snapshot("after_training")
        
        profiler.stop()
        report = profiler.get_report()
    """
    
    def __init__(self, enabled: bool = True):
        """Initialize memory profiler."""
        self.enabled = enabled
        self._snapshots: Dict[str, tracemalloc.Snapshot] = {}
        self._started = False
    
    def start(self):
        """Start memory tracking."""
        if not self.enabled:
            return
        
        tracemalloc.start()
        self._started = True
        self.snapshot("start")
        logger.debug("Memory profiling started")
    
    def stop(self):
        """Stop memory tracking."""
        if not self.enabled or not self._started:
            return
        
        self.snapshot("end")
        tracemalloc.stop()
        self._started = False
        logger.debug("Memory profiling stopped")
    
    def snapshot(self, name: str):
        """Take a memory snapshot."""
        if not self.enabled or not self._started:
            return
        
        self._snapshots[name] = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        logger.debug(f"Memory snapshot '{name}': current={current/1024**2:.1f}MB, "
                    f"peak={peak/1024**2:.1f}MB")
    
    def get_report(self) -> Dict[str, Any]:
        """Generate memory usage report."""
        if not self._snapshots:
            return {'enabled': False}
        
        report = {
            'enabled': True,
            'snapshots': {},
            'peak_memory_mb': 0,
        }
        
        # Get peak memory
        if 'end' in self._snapshots:
            snapshot = self._snapshots['end']
            stats = snapshot.statistics('lineno')
            total = sum(stat.size for stat in stats)
            report['peak_memory_mb'] = total / 1024 ** 2
        
        # Compare snapshots
        if 'start' in self._snapshots and 'end' in self._snapshots:
            start_snap = self._snapshots['start']
            end_snap = self._snapshots['end']
            
            top_stats = end_snap.compare_to(start_snap, 'lineno')
            report['top_memory_growth'] = [
                {
                    'file': str(stat.traceback),
                    'size_diff_mb': stat.size_diff / 1024 ** 2,
                    'count_diff': stat.count_diff
                }
                for stat in top_stats[:10]
            ]
        
        return report
    
    def log_current(self):
        """Log current memory usage."""
        if not self.enabled or not self._started:
            return
        
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"Memory: current={current/1024**2:.1f}MB, peak={peak/1024**2:.1f}MB")
    
    def save_report(self, artifacts_dir: Union[str, Path]) -> Optional[Path]:
        """
        Save memory profiling report to artifacts directory.
        
        Section 17.2: Save memory_report.json to artifacts_dir.
        
        Args:
            artifacts_dir: Directory to save the report
            
        Returns:
            Path to saved report or None if profiling disabled
        """
        if not self.enabled:
            return None
        
        report = self.get_report()
        if not report.get('enabled'):
            return None
        
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = artifacts_dir / 'memory_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Memory report saved to {report_path}")
        
        # Log top memory growth sites
        if 'top_memory_growth' in report:
            logger.debug("Top memory growth sites:")
            for item in report['top_memory_growth'][:5]:
                logger.debug(f"  {item['file']}: {item['size_diff_mb']:.2f}MB")
        
        return report_path
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def create_memory_profiler(
    run_config: Dict[str, Any],
    cli_enabled: Optional[bool] = None
) -> MemoryProfiler:
    """
    Create MemoryProfiler based on configuration.
    
    Section 17.1: Optional memory profiling controlled by config or CLI flag.
    
    Args:
        run_config: Run configuration dictionary
        cli_enabled: Override from CLI flag (takes precedence)
        
    Returns:
        MemoryProfiler instance (enabled or disabled based on config)
    """
    # CLI flag takes precedence
    if cli_enabled is not None:
        enabled = cli_enabled
    else:
        # Check run_config for profiling settings
        profiling_config = run_config.get('profiling', {})
        enabled = profiling_config.get('memory', False)
    
    profiler = MemoryProfiler(enabled=enabled)
    
    if enabled:
        logger.info("Memory profiling enabled")
    
    return profiler


# ==============================================================================
# PARALLEL TRAINING (Section 5.7)
# ==============================================================================

def train_scenario_parallel(
    scenario_configs: List[Dict[str, Any]],
    max_workers: int = 2
) -> List[Dict[str, Any]]:
    """
    Train multiple scenarios in parallel.
    
    WARNING: This creates separate processes, so GPU memory is not shared.
    Use with caution on memory-limited systems.
    
    Args:
        scenario_configs: List of configuration dicts, each with:
            - scenario: 1 or 2
            - model_type: Model type string
            - model_config_path: Path to model config
            - run_config_path: Path to run config
            - data_config_path: Path to data config
            - features_config_path: Path to features config
            - run_name: Name for this run
        max_workers: Maximum parallel workers
        
    Returns:
        List of results from each training run
    """
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for config in scenario_configs:
            future = executor.submit(
                run_experiment,
                scenario=config['scenario'],
                model_type=config.get('model_type', 'catboost'),
                model_config_path=config.get('model_config_path'),
                run_config_path=config.get('run_config_path', 'configs/run_defaults.yaml'),
                data_config_path=config.get('data_config_path', 'configs/data.yaml'),
                features_config_path=config.get('features_config_path', 'configs/features.yaml'),
                run_name=config.get('run_name')
            )
            futures[future] = config
        
        for future in as_completed(futures):
            config = futures[future]
            try:
                model, metrics = future.result()
                results.append({
                    'scenario': config['scenario'],
                    'model_type': config.get('model_type'),
                    'metrics': metrics,
                    'status': 'success'
                })
                logger.info(f"Completed training for scenario {config['scenario']}")
            except Exception as e:
                logger.error(f"Training failed for scenario {config['scenario']}: {e}")
                results.append({
                    'scenario': config['scenario'],
                    'model_type': config.get('model_type'),
                    'error': str(e),
                    'status': 'failed'
                })
    
    return results


def run_full_training_pipeline(
    run_config_path: str = 'configs/run_defaults.yaml',
    data_config_path: str = 'configs/data.yaml',
    features_config_path: str = 'configs/features.yaml',
    model_config_path: Optional[str] = None,
    model_type: str = 'catboost',
    run_cv: bool = False,
    n_folds: int = 5,
    parallel: bool = False,
    run_hpo: bool = False,
    hpo_trials: int = 50,
    enable_tracking: bool = False,
    enable_checkpoints: bool = True,
    enable_profiling: bool = False,
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the full training pipeline for both scenarios.
    
    This is a comprehensive training function that can:
    - Train both scenarios (sequential or parallel)
    - Run cross-validation
    - Perform hyperparameter optimization
    - Track experiments
    - Save checkpoints
    - Profile memory usage
    
    Args:
        run_config_path: Path to run config
        data_config_path: Path to data config
        features_config_path: Path to features config
        model_config_path: Path to model config
        model_type: Model type to train
        run_cv: Whether to run cross-validation
        n_folds: Number of CV folds
        parallel: Train scenarios in parallel
        run_hpo: Run hyperparameter optimization first
        hpo_trials: Number of HPO trials
        enable_tracking: Enable experiment tracking
        enable_checkpoints: Enable checkpoint saving
        enable_profiling: Enable memory profiling
        run_name: Custom run name
        
    Returns:
        Dictionary with all results
    """
    # Load configs
    run_config = load_config(run_config_path)
    data_config = load_config(data_config_path)
    features_config = load_config(features_config_path) if features_config_path else {}
    model_config = load_config(model_config_path) if model_config_path else {}
    
    # Setup
    seed = run_config['reproducibility']['seed']
    set_seed(seed)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = run_name or f"{timestamp}_{model_type}_full"
    
    artifacts_dir = get_project_root() / run_config['paths']['artifacts_dir'] / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_file=str(artifacts_dir / "train_full.log"))
    logger.info(f"Starting full training pipeline: {run_name}")
    
    # Initialize profiler
    profiler = MemoryProfiler(enabled=enable_profiling)
    profiler.start()
    
    # Initialize experiment tracking
    tracker = None
    if enable_tracking:
        tracker = setup_experiment_tracking(run_config, run_name)
        if tracker:
            tracker.start_run(run_name=run_name, config={
                'run_config': run_config,
                'model_config': model_config,
                'model_type': model_type
            })
    
    # Initialize checkpoint manager
    checkpoint_mgr = None
    if enable_checkpoints:
        checkpoint_mgr = TrainingCheckpoint(
            checkpoint_dir=artifacts_dir / 'checkpoints',
            keep_best_n=3
        )
    
    results = {
        'run_name': run_name,
        'timestamp': timestamp,
        'scenarios': {},
        'hpo_results': None,
        'memory_report': None
    }
    
    profiler.snapshot("after_setup")
    
    try:
        # Run HPO if requested
        if run_hpo:
            logger.info("Running hyperparameter optimization...")
            # Load data once for HPO
            from .data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values
            
            train_data = load_raw_data(data_config, split='train')
            panel = prepare_base_panel(
                train_data['volume'],
                train_data['generics'],
                train_data['medicine_info']
            )
            panel = handle_missing_values(panel)
            panel = compute_pre_entry_stats(panel, is_train=True)
            
            profiler.snapshot("after_data_load")
            
            # Run HPO for each scenario
            for scenario in [1, 2]:
                from .features import make_features, select_training_rows
                
                panel_features = make_features(panel.copy(), scenario=scenario, mode='train', config=features_config)
                train_rows = select_training_rows(panel_features, scenario=scenario)
                
                train_df, val_df = create_validation_split(
                    train_rows,
                    val_fraction=run_config['validation']['val_fraction'],
                    stratify_by=run_config['validation']['stratify_by'],
                    random_state=seed
                )
                
                X_train, y_train, meta_train = split_features_target_meta(train_df)
                X_val, y_val, meta_val = split_features_target_meta(val_df)
                
                hpo_results = run_hyperparameter_optimization(
                    X_train, y_train, meta_train,
                    X_val, y_val, meta_val,
                    scenario=scenario,
                    model_type=model_type,
                    n_trials=hpo_trials,
                    artifacts_dir=artifacts_dir / f'hpo_scenario{scenario}'
                )
                
                results['hpo_results'] = results.get('hpo_results', {})
                results['hpo_results'][scenario] = hpo_results
                
                # Use best params
                model_config['params'] = hpo_results['best_params']
            
            profiler.snapshot("after_hpo")
        
        # Train scenarios
        scenario_configs = [
            {
                'scenario': 1,
                'model_type': model_type,
                'model_config_path': model_config_path,
                'run_config_path': run_config_path,
                'data_config_path': data_config_path,
                'features_config_path': features_config_path,
                'run_name': f"{run_name}_s1"
            },
            {
                'scenario': 2,
                'model_type': model_type,
                'model_config_path': model_config_path,
                'run_config_path': run_config_path,
                'data_config_path': data_config_path,
                'features_config_path': features_config_path,
                'run_name': f"{run_name}_s2"
            }
        ]
        
        if parallel and len(scenario_configs) > 1:
            logger.info("Training scenarios in parallel...")
            scenario_results = train_scenario_parallel(scenario_configs, max_workers=2)
        else:
            logger.info("Training scenarios sequentially...")
            scenario_results = []
            for config in scenario_configs:
                model, metrics = run_experiment(
                    scenario=config['scenario'],
                    model_type=config['model_type'],
                    model_config_path=config['model_config_path'],
                    run_config_path=config['run_config_path'],
                    data_config_path=config['data_config_path'],
                    features_config_path=config['features_config_path'],
                    run_name=config['run_name']
                )
                scenario_results.append({
                    'scenario': config['scenario'],
                    'metrics': metrics,
                    'status': 'success'
                })
                
                # Save checkpoint
                if checkpoint_mgr:
                    is_best = True  # First model of each scenario is "best" by default
                    checkpoint_mgr.save(
                        model=model,
                        epoch=0,
                        metrics=metrics,
                        config={'scenario': config['scenario'], 'model_type': model_type},
                        is_best=is_best
                    )
                
                # Log to tracker
                if tracker:
                    tracker.log_metrics({
                        f"s{config['scenario']}_official_metric": metrics.get('official_metric', np.nan),
                        f"s{config['scenario']}_rmse": metrics.get('rmse_norm', np.nan)
                    })
        
        for res in scenario_results:
            results['scenarios'][res['scenario']] = res
        
        profiler.snapshot("after_training")
        
    finally:
        # Stop profiler
        profiler.stop()
        results['memory_report'] = profiler.get_report()
        
        # End tracking
        if tracker:
            tracker.end_run()
    
    # Save final results
    results_path = artifacts_dir / 'full_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Full training pipeline complete. Results saved to {artifacts_dir}")
    
    return results


def compute_config_hash(configs: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of configuration dictionaries.
    
    Used for reproducibility tracking - same configs = same hash.
    
    Args:
        configs: Dictionary of config name -> config dict
        
    Returns:
        8-character hex hash string
    """
    # Sort keys for deterministic ordering
    config_str = json.dumps(configs, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def save_config_snapshot(
    artifacts_dir: Path,
    run_config_path: Optional[str] = None,
    data_config_path: Optional[str] = None,
    features_config_path: Optional[str] = None,
    model_config_path: Optional[str] = None
) -> str:
    """
    Save exact copies of all config files used for a run.
    
    Creates a 'configs/' subdirectory in artifacts_dir with copies
    of all config files, plus computes a hash for reproducibility tracking.
    
    Args:
        artifacts_dir: Directory to save config copies
        run_config_path: Path to run config YAML
        data_config_path: Path to data config YAML
        features_config_path: Path to features config YAML
        model_config_path: Path to model config YAML
        
    Returns:
        Config hash string
    """
    config_dir = artifacts_dir / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    configs = {}
    config_paths = {
        'run_config': run_config_path,
        'data_config': data_config_path,
        'features_config': features_config_path,
        'model_config': model_config_path,
    }
    
    for name, path in config_paths.items():
        if path and Path(path).exists():
            # Copy the file
            src_path = Path(path)
            dst_path = config_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            
            # Load for hash computation
            with open(src_path, 'r') as f:
                configs[name] = yaml.safe_load(f)
    
    # Compute and save hash
    config_hash = compute_config_hash(configs)
    hash_file = config_dir / 'config_hash.txt'
    with open(hash_file, 'w') as f:
        f.write(f"{config_hash}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Files:\n")
        for name, path in config_paths.items():
            if path:
                f.write(f"  {name}: {path}\n")
    
    logger.info(f"Config snapshot saved to {config_dir} (hash: {config_hash})")
    return config_hash


def compute_metric_aligned_weights(
    meta_df: pd.DataFrame,
    scenario: int,
    avg_vol_col: str = 'avg_vol_12m'
) -> pd.Series:
    """
    Compute sample weights that exactly align with the official metric formula.
    
    This derives per-row weights from the official metric equations:
    
    **Scenario 1 (Metric 1) formula:**
    PE = 0.2 * sum(|actual-pred|)/(24*avg_vol)     [monthly component, all months]
      + 0.5 * |sum(actual)-sum(pred)|/(6*avg_vol)  [months 0-5]
      + 0.2 * |sum(actual)-sum(pred)|/(6*avg_vol)  [months 6-11]
      + 0.1 * |sum(actual)-sum(pred)|/(12*avg_vol) [months 12-23]
    
    Bucket weighting: bucket1 weight = 2/n1, bucket2 weight = 1/n2
    
    **Scenario 2 (Metric 2) formula:**
    PE = 0.2 * sum(|actual-pred|)/(18*avg_vol)     [monthly component, months 6-23]
      + 0.5 * |sum(actual)-sum(pred)|/(6*avg_vol)  [months 6-11]
      + 0.3 * |sum(actual)-sum(pred)|/(12*avg_vol) [months 12-23]
    
    For training, we approximate the window-sum components with per-row weights
    based on the relative importance of each window.
    
    Args:
        meta_df: DataFrame with months_postgx, bucket, and avg_vol columns
        scenario: 1 or 2
        avg_vol_col: Column name for average volume (default: 'avg_vol_12m')
        
    Returns:
        Series of sample weights aligned with meta_df index
    """
    scenario = _normalize_scenario(scenario)
    
    months = meta_df['months_postgx']
    
    # Get avg_vol for normalization (higher avg_vol = lower weight contribution)
    avg_vol = meta_df[avg_vol_col].clip(lower=1.0)  # Prevent division by zero
    inv_avg_vol = 1.0 / avg_vol
    
    # Normalize inv_avg_vol to have mean 1
    inv_avg_vol = inv_avg_vol / inv_avg_vol.mean()
    
    if scenario == 1:
        # Metric 1 weights derived from formula
        # Monthly component (0.2 weight, divided by 24 months): 0.2/24 = 0.00833 per month
        # Window 0-5 (0.5 weight, 6 months): 0.5/6 = 0.0833 per month
        # Window 6-11 (0.2 weight, 6 months): 0.2/6 = 0.0333 per month
        # Window 12-23 (0.1 weight, 12 months): 0.1/12 = 0.00833 per month
        
        # Combined per-month weights (monthly + window contribution)
        # Months 0-5: 0.00833 (monthly) + 0.0833 (window) = 0.0917
        # Months 6-11: 0.00833 (monthly) + 0.0333 (window) = 0.0417
        # Months 12-23: 0.00833 (monthly) + 0.00833 (window) = 0.0167
        
        base_weights = np.where(months <= 5, 0.0917,
                       np.where((months >= 6) & (months <= 11), 0.0417,
                       np.where(months >= 12, 0.0167, 0.0)))
    else:
        # Metric 2 weights derived from formula
        # Monthly component (0.2 weight, divided by 18 months): 0.2/18 = 0.0111 per month
        # Window 6-11 (0.5 weight, 6 months): 0.5/6 = 0.0833 per month
        # Window 12-23 (0.3 weight, 12 months): 0.3/12 = 0.025 per month
        
        # Combined per-month weights
        # Months 6-11: 0.0111 (monthly) + 0.0833 (window) = 0.0944
        # Months 12-23: 0.0111 (monthly) + 0.025 (window) = 0.0361
        
        base_weights = np.where((months >= 6) & (months <= 11), 0.0944,
                       np.where(months >= 12, 0.0361, 0.0))
    
    weights = pd.Series(base_weights, index=meta_df.index)
    
    # Apply inverse avg_vol weighting (smaller series have higher metric contribution)
    weights = weights * inv_avg_vol
    
    # Apply bucket weighting (bucket 1 has 2x weight)
    if 'bucket' in meta_df.columns:
        # Estimate n1, n2 for proper bucket weighting
        bucket_counts = meta_df.groupby(['country', 'brand_name'])['bucket'].first().value_counts()
        n1 = bucket_counts.get(1, 1)
        n2 = bucket_counts.get(2, 1)
        
        # Bucket 1 weight: 2/n1, Bucket 2 weight: 1/n2
        # Normalize so average weight ≈ 1
        total_weight = 2 + 1  # 2 for bucket 1, 1 for bucket 2
        bucket1_w = (2 / n1) * (n1 + n2) / total_weight if n1 > 0 else 1.0
        bucket2_w = (1 / n2) * (n1 + n2) / total_weight if n2 > 0 else 1.0
        
        bucket_weight = meta_df['bucket'].map({1: bucket1_w, 2: bucket2_w}).fillna(1.0)
        weights = weights * bucket_weight
    
    # Normalize so weights sum to len(weights) (helps with loss scale)
    weights = weights * len(weights) / (weights.sum() + 1e-8)
    
    # Ensure no zero weights
    weights = weights.clip(lower=0.01)
    
    logger.debug(f"Metric-aligned weights - min: {weights.min():.4f}, max: {weights.max():.4f}, "
                f"mean: {weights.mean():.4f}")
    
    return weights


def split_features_target_meta(
    df: pd.DataFrame,
    validate_target: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Separate pure features from target and meta columns.
    
    This GUARANTEES bucket/y_norm never leak into model features.
    
    Section 18.1.1: META_COLS always excluded from features.
    Section 18.1.2: Symmetric with get_feature_matrix_and_meta for inference.
    
    Args:
        df: DataFrame with features, target, and meta columns
        validate_target: If True, raises error if TARGET_COL is missing
        
    Returns:
        X: Pure features for model (excludes all META_COLS)
        y: Target (y_norm)
        meta: Meta columns for weights, grouping, metrics
        
    Raises:
        ValueError: If TARGET_COL is missing and validate_target=True
        ValueError: If META_COLS leak into features
    """
    # 18.1.2: Check that TARGET_COL exists (training mode requires it)
    if validate_target and TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL '{TARGET_COL}' not found in DataFrame. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    
    # Identify feature columns (everything except meta)
    # 18.1.1: Use META_COLS consistently to guarantee no leakage
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
    # Split
    X = df[feature_cols].copy()
    
    # Handle target (may be missing for inference mode)
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].copy()
    else:
        y = pd.Series(dtype=float)
    
    # Meta columns that exist in the dataframe
    meta_cols_present = [c for c in META_COLS if c in df.columns]
    meta = df[meta_cols_present].copy()
    
    # Log
    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Meta: {len(meta_cols_present)} columns: {meta_cols_present}")
    
    # 18.1.1: Validate no leakage - raise immediately if any META_COLS in features
    leaked = set(X.columns) & set(META_COLS)
    if leaked:
        raise ValueError(
            f"LEAKAGE DETECTED! Meta columns in features: {leaked}. "
            f"Check that META_COLS definition matches data.yaml columns.meta_cols."
        )
    
    # Additional validation: check for suspicious column patterns
    suspicious_patterns = ['_target', '_bucket', '_y_norm', 'future_', 'label_']
    suspicious_cols = [c for c in X.columns if any(p in c.lower() for p in suspicious_patterns)]
    if suspicious_cols:
        logger.warning(
            f"Suspicious columns detected in features (may indicate leakage): {suspicious_cols[:10]}"
        )
    
    return X, y, meta


def get_feature_matrix_and_meta(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For INFERENCE: separate features from meta (no target).
    
    CRITICAL: Uses META_COLS same as training's split_features_target_meta
    to ensure feature_cols match the model exactly!
    
    Section 18.1.2: Symmetric with split_features_target_meta for training.
    
    Args:
        df: DataFrame with features and meta columns
        
    Returns:
        X: Pure features for model
        meta: Meta columns including avg_vol_12m for inverse transform
    """
    # Use META_COLS to exclude same columns as training
    # This ensures feature columns match the trained model exactly
    # Also exclude raw categorical columns that are only used for encoding (not as direct features)
    raw_categoricals_to_exclude = ['ther_area', 'main_package']  # These are encoded, not used directly
    
    feature_cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if col in raw_categoricals_to_exclude:
            continue  # Exclude raw categoricals - only encoded versions should be features
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
        elif df[col].dtype.name == 'category':
            # Only include categorical columns that are encoded versions (have '_encoded' suffix)
            # Raw categoricals like 'ther_area' should be excluded - they're encoded elsewhere
            if col.endswith('_encoded'):
                feature_cols.append(col)
            # For other categoricals, exclude them (they should be encoded first)
            # This prevents issues with CatBoost expecting them in cat_features list
    
    X = df[feature_cols].copy()
    
    # Ensure all categorical-like features are integers (for CatBoost compatibility)
    # CatBoost requires categorical features to be integers, not floats
    categorical_patterns = ['_encoded', '_bin']  # Patterns that indicate categorical features
    
    for col in X.columns:
        is_categorical = (
            col.endswith('_encoded') or
            col.endswith('_bin') or
            X[col].dtype.name == 'category'
        )
        
        if is_categorical:
            if X[col].dtype.name == 'category':
                # Category dtype: convert to codes
                X[col] = X[col].cat.codes.astype(int)
                X[col] = X[col].replace(-1, 0).astype(int)  # Replace missing with 0
            elif X[col].dtype in ['float64', 'float32']:
                # Float: convert to int (e.g., 0.0 -> 0)
                X[col] = X[col].fillna(0).astype(int)
            elif pd.api.types.is_integer_dtype(X[col]):
                # Already integer: ensure no NaN
                X[col] = X[col].fillna(0).astype(int)
    
    # Meta columns that exist
    meta_cols_present = [c for c in META_COLS if c in df.columns]
    meta = df[meta_cols_present].copy()
    
    # 18.1.2: Log for debugging symmetry issues
    logger.debug(f"Inference features: {len(feature_cols)} columns")
    logger.debug(f"Inference meta: {len(meta_cols_present)} columns")
    
    # Validate no meta columns in features
    leaked = set(X.columns) & set(META_COLS)
    if leaked:
        raise ValueError(f"LEAKAGE DETECTED in inference! Meta columns in features: {leaked}")
    
    return X, meta


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if in a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=get_project_root(),
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
    except Exception:
        pass
    return None


def get_experiment_metadata(
    scenario: int,
    model_type: str,
    run_config: dict,
    data_config: dict,
    model_config: dict,
    panel_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Collect experiment metadata for logging.
    
    Args:
        scenario: Scenario number (1 or 2)
        model_type: Type of model being trained
        run_config: Run configuration dictionary
        data_config: Data configuration dictionary
        model_config: Model configuration dictionary
        panel_df: Full panel DataFrame
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        
    Returns:
        Dictionary with experiment metadata
    """
    # Get unique series counts
    n_total_series = panel_df[['country', 'brand_name']].drop_duplicates().shape[0]
    n_train_series = train_df[['country', 'brand_name']].drop_duplicates().shape[0]
    n_val_series = val_df[['country', 'brand_name']].drop_duplicates().shape[0] if val_df is not None else 0
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash(),
        'scenario': scenario,
        'model_type': model_type,
        'random_seed': run_config.get('reproducibility', {}).get('seed', 42),
        'dataset': {
            'total_rows': len(panel_df),
            'total_series': n_total_series,
            'train_rows': len(train_df),
            'train_series': n_train_series,
            'val_rows': len(val_df) if val_df is not None else 0,
            'val_series': n_val_series,
        },
        'validation': {
            'val_fraction': run_config.get('validation', {}).get('val_fraction', 0.2),
            'stratify_by': run_config.get('validation', {}).get('stratify_by', 'bucket'),
            'split_level': run_config.get('validation', {}).get('split_level', 'series'),
        },
        'configs': {
            'run_config': run_config,
            'data_config': data_config,
            'model_config': model_config,
        }
    }
    
    return metadata


def compute_sample_weights(
    meta_df: pd.DataFrame, 
    scenario,
    config: Optional[dict] = None,
    use_metric_aligned: bool = False,
    avg_vol_col: str = 'avg_vol_12m',
    weight_transform: str = 'identity'
) -> pd.Series:
    """
    Compute sample weights that approximate official metric weighting.
    
    Default weights (can be overridden via config):
    
    Scenario 1:
        - Months 0-5: weight 3.0 (highest priority)
        - Months 6-11: weight 1.5
        - Months 12-23: weight 1.0
    
    Scenario 2:
        - Months 6-11: weight 2.5 (highest priority)
        - Months 12-23: weight 1.0
    
    Bucket weights:
        - Bucket 1: multiply by 2.0
        - Bucket 2: multiply by 1.0
    
    Args:
        meta_df: DataFrame with months_postgx and bucket columns
        scenario: 1, 2, "scenario1", or "scenario2"
        config: Optional config dict with 'sample_weights' section
        use_metric_aligned: If True, use exact metric-aligned weights
        avg_vol_col: Column name for average volume (used when use_metric_aligned=True)
        weight_transform: Transformation to apply: 'identity', 'sqrt', 'log', 'softmax', 'rank'
        
    Returns:
        Series of sample weights aligned with meta_df index
    """
    scenario = _normalize_scenario(scenario)
    
    # Use metric-aligned weights if requested
    if use_metric_aligned:
        weights = compute_metric_aligned_weights(
            meta_df=meta_df,
            scenario=scenario,
            avg_vol_col=avg_vol_col
        )
        return transform_weights(weights, weight_transform)
    
    weights = pd.Series(1.0, index=meta_df.index)
    
    # Get weights from config or use defaults
    if config and 'sample_weights' in config:
        sw_config = config['sample_weights']
        if scenario == 1:
            s1_config = sw_config.get('scenario1', {})
            w_0_5 = s1_config.get('months_0_5', 3.0)
            w_6_11 = s1_config.get('months_6_11', 1.5)
            w_12_23 = s1_config.get('months_12_23', 1.0)
        else:
            s2_config = sw_config.get('scenario2', {})
            w_6_11 = s2_config.get('months_6_11', 2.5)
            w_12_23 = s2_config.get('months_12_23', 1.0)
            w_0_5 = 1.0  # Not used in S2
        
        bucket_weights = sw_config.get('bucket_weights', {})
        bucket1_w = bucket_weights.get('bucket1', 2.0)
        bucket2_w = bucket_weights.get('bucket2', 1.0)
    else:
        # Default weights
        if scenario == 1:
            w_0_5, w_6_11, w_12_23 = 3.0, 1.5, 1.0
        else:
            w_0_5, w_6_11, w_12_23 = 1.0, 2.5, 1.0
        bucket1_w, bucket2_w = 2.0, 1.0
    
    # Time-based weights
    months = meta_df['months_postgx']
    
    if scenario == 1:
        # Phase 1A: 50% months 0-5, 20% months 6-11, 10% months 12-23
        weights = np.where(months <= 5, w_0_5, weights)
        weights = np.where((months >= 6) & (months <= 11), w_6_11, weights)
        weights = np.where(months >= 12, w_12_23, weights)
    elif scenario == 2:
        # Phase 1B: 50% months 6-11, 30% months 12-23
        weights = np.where((months >= 6) & (months <= 11), w_6_11, weights)
        weights = np.where(months >= 12, w_12_23, weights)
    
    weights = pd.Series(weights, index=meta_df.index)
    
    # Bucket weights
    if 'bucket' in meta_df.columns:
        bucket_weight = meta_df['bucket'].map({1: bucket1_w, 2: bucket2_w}).fillna(1.0)
        weights = weights * bucket_weight
    
    # Normalize so weights sum to len(weights) (optional, helps with loss scale)
    weights = weights * len(weights) / weights.sum()
    
    # Apply transformation
    weights = transform_weights(weights, weight_transform)
    
    logger.info(f"Sample weights - min: {weights.min():.2f}, max: {weights.max():.2f}, mean: {weights.mean():.2f}")
    
    return weights


# ==============================================================================
# BONUS EXPERIMENTS: B10 - Target Transform
# ==============================================================================

def transform_target(y: np.ndarray, transform_config: dict) -> Tuple[np.ndarray, dict]:
    """
    Transform target variable (B10: Target Transform).
    
    Args:
        y: Target values (normalized volume)
        transform_config: Transform configuration dict
        
    Returns:
        (transformed_y, transform_params) where transform_params can be used for inverse
    """
    transform_type = transform_config.get('type', 'none')
    epsilon = float(transform_config.get('epsilon', 1e-6))  # Ensure float type
    
    if transform_type == 'none':
        return y.copy(), {'type': 'none'}
    elif transform_type == 'log1p':
        y_transformed = np.log1p(np.maximum(y, 0) + epsilon)
        return y_transformed, {'type': 'log1p', 'epsilon': epsilon}
    elif transform_type == 'power':
        power_exp = float(transform_config.get('power_exponent', 0.5))  # Ensure float type
        y_transformed = np.power(np.maximum(y, 0) + epsilon, power_exp)
        return y_transformed, {'type': 'power', 'exponent': power_exp, 'epsilon': epsilon}
    else:
        raise ValueError(f"Unknown target transform type: {transform_type}")


def inverse_transform_target(y_transformed: np.ndarray, transform_params: dict) -> np.ndarray:
    """
    Inverse transform target variable.
    
    Args:
        y_transformed: Transformed target values
        transform_params: Parameters from transform_target
        
    Returns:
        Original scale target values
    """
    transform_type = transform_params.get('type', 'none')
    epsilon = transform_params.get('epsilon', 1e-6)
    
    if transform_type == 'none':
        return y_transformed.copy()
    elif transform_type == 'log1p':
        y_original = np.expm1(y_transformed) - epsilon
        return np.maximum(y_original, 0.0)
    elif transform_type == 'power':
        power_exp = transform_params.get('exponent', 0.5)
        y_original = np.power(np.maximum(y_transformed, 0), 1.0 / power_exp) - epsilon
        return np.maximum(y_original, 0.0)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def train_scenario_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario,
    model_type: str = 'catboost',
    model_config: Optional[dict] = None,
    run_config: Optional[dict] = None,
    run_id: Optional[str] = None,
    metrics_dir: Optional[Path] = None,
    fold_idx: Optional[int] = None
) -> Tuple[Any, Dict]:
    """
    Train model for specific scenario with early stopping.
    
    Uses sample weights from META to align with official metric.
    Optionally saves unified metric records for training/validation.
    
    Section 18.2: Standard model interface with consistent metric computation.
    
    Args:
        X_train, y_train, meta_train: Training data
        X_val, y_val, meta_val: Validation data
        scenario: 1, 2, "scenario1", or "scenario2"
        model_type: 'catboost', 'lightgbm', 'xgboost', 'linear'
        model_config: Model configuration dict
        run_config: Run configuration dict (for sample weights)
        run_id: Optional run ID for metrics logging
        metrics_dir: Optional directory to save metrics
        fold_idx: Optional fold index for CV logging
        
    Returns:
        (trained_model, metrics_dict)
        
    Note:
        All model wrappers must support:
        - fit(X_train, y_train, X_val=None, y_val=None, sample_weight=None)
        - predict(X)
    """
    scenario = _normalize_scenario(scenario)
    
    # 18.2.2: Validate that avg_vol_12m is present in meta_val (required for metric computation)
    if 'avg_vol_12m' not in meta_val.columns:
        raise ValueError(
            "meta_val must contain 'avg_vol_12m' for official metric computation. "
            f"Available columns: {list(meta_val.columns)}"
        )
    
    # BONUS: B10 - Apply target transform if configured
    transform_config = run_config.get('target_transform', {}) if run_config else {}
    transform_type = transform_config.get('type', 'none')
    
    y_train_original = y_train.copy()
    y_val_original = y_val.copy()
    transform_params = None
    
    if transform_type != 'none':
        y_train_transformed, transform_params = transform_target(y_train.values, transform_config)
        y_val_transformed, _ = transform_target(y_val.values, transform_config)
        y_train = pd.Series(y_train_transformed, index=y_train.index)
        y_val = pd.Series(y_val_transformed, index=y_val.index)
        logger.info(f"Applied target transform: {transform_type}")
    
    # Import model class
    model = _get_model(model_type, model_config)
    
    # Store transform params in model if it supports it (for inference)
    if hasattr(model, '_transform_params'):
        model._transform_params = transform_params
    
    # Compute sample weights (using run_config if available)
    sample_weights = compute_sample_weights(meta_train, scenario, config=run_config)
    
    # Track training time
    train_start = time.time()
    
    # Prepare feature matrices - add meta columns for hybrid models
    X_train_for_model = X_train.copy()
    X_val_for_model = X_val.copy()
    
    if model_type.lower().startswith('hybrid'):
        # Hybrid models need months_postgx and avg_vol_12m in features
        if 'months_postgx' not in X_train_for_model.columns:
            X_train_for_model['months_postgx'] = meta_train['months_postgx'].values
        if 'avg_vol_12m' not in X_train_for_model.columns:
            X_train_for_model['avg_vol_12m'] = meta_train['avg_vol_12m'].values
        if 'months_postgx' not in X_val_for_model.columns:
            X_val_for_model['months_postgx'] = meta_val['months_postgx'].values
        if 'avg_vol_12m' not in X_val_for_model.columns:
            X_val_for_model['avg_vol_12m'] = meta_val['avg_vol_12m'].values
    
    if model_type.lower() in ('arihow', 'arima_hw', 'arima_holtwinters'):
        # ARIHOW models need country, brand_name, months_postgx, avg_vol_12m in features
        for col in ['country', 'brand_name', 'months_postgx', 'avg_vol_12m']:
            if col not in X_train_for_model.columns:
                X_train_for_model[col] = meta_train[col].values
            if col not in X_val_for_model.columns:
                X_val_for_model[col] = meta_val[col].values
    
    # 18.2.1: Standard model interface - all models must support fit/predict
    with timer(f"Train {model_type} for scenario {scenario}"):
        model.fit(
            X_train_for_model, y_train,
            X_val=X_val_for_model, y_val=y_val,
            sample_weight=sample_weights
        )
    
    train_time = time.time() - train_start
    
    # Compute validation metrics
    val_preds_norm_transformed = model.predict(X_val_for_model)
    
    # BONUS: B10 - Inverse transform predictions if transform was applied
    if transform_params is not None:
        val_preds_norm = inverse_transform_target(val_preds_norm_transformed, transform_params)
    else:
        val_preds_norm = val_preds_norm_transformed
    
    # Denormalize predictions for metric calculation
    avg_vol_val = meta_val['avg_vol_12m'].values
    val_preds_volume = val_preds_norm * avg_vol_val
    val_actual_volume = y_val_original.values * avg_vol_val
    
    # Build DataFrames for official metric
    df_pred = meta_val[['country', 'brand_name', 'months_postgx']].copy()
    df_pred['volume'] = val_preds_volume
    
    df_actual = meta_val[['country', 'brand_name', 'months_postgx']].copy()
    df_actual['volume'] = val_actual_volume
    
    # 18.2.2: Create aux file correctly for compute_metric1/2
    # Ensure bucket column exists
    if 'bucket' not in meta_val.columns:
        logger.warning("'bucket' not in meta_val, official metric may fail")
        val_with_bucket = meta_val[['country', 'brand_name', 'avg_vol_12m']].drop_duplicates()
        val_with_bucket = val_with_bucket.rename(columns={'avg_vol_12m': 'avg_vol'})
        val_with_bucket['bucket'] = 1  # Default bucket
    else:
        val_with_bucket = meta_val[['country', 'brand_name', 'avg_vol_12m', 'bucket']].drop_duplicates()
        val_with_bucket = val_with_bucket.rename(columns={'avg_vol_12m': 'avg_vol'})
    
    # Compute official metric
    try:
        if scenario == 1:
            official_metric = compute_metric1(df_actual, df_pred, val_with_bucket)
        else:
            official_metric = compute_metric2(df_actual, df_pred, val_with_bucket)
    except Exception as e:
        logger.warning(f"Could not compute official metric: {e}")
        official_metric = np.nan
    
    # Compute additional metrics
    rmse = np.sqrt(np.mean((val_preds_norm - y_val.values) ** 2))
    mae = np.mean(np.abs(val_preds_norm - y_val.values))
    
    # Compute R² score
    ss_res = np.sum((val_preds_norm - y_val.values) ** 2)
    ss_tot = np.sum((y_val.values - np.mean(y_val.values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    metrics = {
        'official_metric': official_metric,
        'rmse_norm': rmse,
        'mae_norm': mae,
        'r2_norm': r2,
        'scenario': scenario,
        'model_type': model_type,
        'train_time_seconds': train_time,
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_features': len(X_train.columns),
    }
    
    # 18.2.3: Save unified metric records if metrics_dir is provided
    if metrics_dir is not None:
        metrics_dir = Path(metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / 'metrics.csv'
        
        # Determine phase and split based on fold_idx
        phase = 'cv' if fold_idx is not None else 'train'
        split_name = f'fold_{fold_idx}' if fold_idx is not None else 'val'
        step = fold_idx if fold_idx is not None else 'final'
        
        # Create metric records
        official_metric_name = METRIC_NAME_S1 if scenario == 1 else METRIC_NAME_S2
        records = [
            make_metric_record(
                phase=phase, split=split_name, scenario=scenario,
                model_name=model_type, metric_name=official_metric_name,
                value=official_metric, run_id=run_id, step=step
            ),
            make_metric_record(
                phase=phase, split=split_name, scenario=scenario,
                model_name=model_type, metric_name=METRIC_NAME_RMSE,
                value=rmse, run_id=run_id, step=step
            ),
            make_metric_record(
                phase=phase, split=split_name, scenario=scenario,
                model_name=model_type, metric_name=METRIC_NAME_MAE,
                value=mae, run_id=run_id, step=step
            ),
        ]
        
        save_metric_records(records, metrics_path, append=True)
        logger.debug(f"Saved {len(records)} metric records to {metrics_path}")
    
    logger.info(f"Validation metrics: Official={official_metric:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    logger.info(f"Training time: {train_time:.2f} seconds")
    
    return model, metrics


def _get_model(model_type: str, config: Optional[dict] = None):
    """
    Get model instance by type.
    
    Supported model types:
    - 'catboost': CatBoost gradient boosting
    - 'lightgbm', 'lgbm': LightGBM gradient boosting
    - 'xgboost', 'xgb': XGBoost gradient boosting
    - 'linear', 'ridge', 'lasso', 'elasticnet', 'huber': Linear models
    - 'nn', 'neural', 'mlp': Neural network
    - 'baseline_global_mean', 'global_mean': Global mean baseline
    - 'baseline_flat', 'flat': Flat baseline (no erosion)
    - 'baseline_trend', 'trend': Trend extrapolation baseline
    - 'baseline_historical', 'historical_curve', 'knn_curve': Historical curve baseline
    - 'averaging', 'averaging_ensemble': Simple averaging ensemble
    - 'weighted', 'weighted_averaging', 'weighted_ensemble': Weighted averaging ensemble
    - 'stacking', 'stacking_ensemble': Stacking ensemble
    - 'blending', 'blending_ensemble': Blending ensemble
    - 'kg_gcn_lstm', 'gcn_lstm': Knowledge Graph GCN + LSTM model
    - 'cnn_lstm': CNN-LSTM model (Li et al. 2024)
    - 'lstm', 'lstm_only': Pure LSTM model (ablation)
    
    Args:
        model_type: Type of model to create
        config: Model configuration dictionary
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model type is not recognized
    """
    model_type_lower = model_type.lower()
    config = config or {}
    
    # Tree boosters
    if model_type_lower in ('catboost', 'cat'):
        from .models.cat_model import CatBoostModel
        return CatBoostModel(config)
    elif model_type_lower in ('lightgbm', 'lgbm'):
        from .models.lgbm_model import LGBMModel
        return LGBMModel(config)
    elif model_type_lower in ('xgboost', 'xgb'):
        from .models.xgb_model import XGBModel
        return XGBModel(config)
    
    # Linear models
    elif model_type_lower in ('linear', 'ridge', 'lasso', 'elasticnet', 'huber'):
        from .models.linear import LinearModel
        return LinearModel(config)
    
    # Neural network
    elif model_type_lower in ('nn', 'neural', 'mlp'):
        from .models.nn import NNModel
        return NNModel(config)
    
    # Baselines
    elif model_type_lower in ('baseline_global_mean', 'global_mean'):
        from .models.linear import GlobalMeanBaseline
        return GlobalMeanBaseline(config)
    elif model_type_lower in ('baseline_flat', 'flat'):
        from .models.linear import FlatBaseline
        return FlatBaseline(config)
    elif model_type_lower in ('baseline_trend', 'trend'):
        from .models.linear import TrendBaseline
        return TrendBaseline(config)
    elif model_type_lower in ('baseline_historical', 'historical_curve', 'knn_curve'):
        from .models.linear import HistoricalCurveBaseline
        return HistoricalCurveBaseline(config)
    
    # Ensemble models
    elif model_type_lower in ('averaging', 'averaging_ensemble'):
        from .models.ensemble import AveragingEnsemble
        return AveragingEnsemble(config)
    elif model_type_lower in ('weighted', 'weighted_averaging', 'weighted_ensemble'):
        from .models.ensemble import WeightedAveragingEnsemble
        return WeightedAveragingEnsemble(config)
    elif model_type_lower in ('stacking', 'stacking_ensemble'):
        from .models.ensemble import StackingEnsemble
        return StackingEnsemble(config)
    elif model_type_lower in ('blending', 'blending_ensemble'):
        from .models.ensemble import BlendingEnsemble
        return BlendingEnsemble(config)
    
    # Hybrid Physics + ML model (all variants use wrapper)
    elif model_type_lower in ('hybrid_lgbm', 'hybrid_lightgbm'):
        from .models.hybrid_physics_ml import HybridPhysicsMLWrapper
        wrapper_config = {
            'decay_rate': config.get('physics', {}).get('decay_rate', 0.05),
            'ml_model_type': 'lightgbm',
            'clip_min': config.get('physics', {}).get('clip_min', 0.0),
            'clip_max': config.get('physics', {}).get('clip_max', 2.0),
            'early_stopping_rounds': config.get('ml_model', {}).get('early_stopping_rounds', 50),
        }
        ml_params = config.get('ml_model', {}).get('lightgbm', {})
        if ml_params:
            wrapper_config['ml_params'] = ml_params
        return HybridPhysicsMLWrapper(wrapper_config)
    elif model_type_lower in ('hybrid_xgb', 'hybrid_xgboost'):
        from .models.hybrid_physics_ml import HybridPhysicsMLWrapper
        wrapper_config = {
            'decay_rate': config.get('physics', {}).get('decay_rate', 0.05),
            'ml_model_type': 'xgboost',
            'clip_min': config.get('physics', {}).get('clip_min', 0.0),
            'clip_max': config.get('physics', {}).get('clip_max', 2.0),
            'early_stopping_rounds': config.get('ml_model', {}).get('early_stopping_rounds', 50),
        }
        ml_params = config.get('ml_model', {}).get('xgboost', {})
        if ml_params:
            wrapper_config['ml_params'] = ml_params
        return HybridPhysicsMLWrapper(wrapper_config)
    elif model_type_lower in ('hybrid_cat', 'hybrid_catboost'):
        from .models.hybrid_physics_ml import HybridPhysicsMLWrapper
        wrapper_config = {
            'decay_rate': config.get('physics', {}).get('decay_rate', 0.05),
            'ml_model_type': 'catboost',
            'clip_min': config.get('physics', {}).get('clip_min', 0.0),
            'clip_max': config.get('physics', {}).get('clip_max', 2.0),
            'early_stopping_rounds': config.get('ml_model', {}).get('early_stopping_rounds', 50),
        }
        ml_params = config.get('ml_model', {}).get('catboost', {})
        if ml_params:
            wrapper_config['ml_params'] = ml_params
        return HybridPhysicsMLWrapper(wrapper_config)
    elif model_type_lower == 'hybrid':
        from .models.hybrid_physics_ml import HybridPhysicsMLWrapper
        # Build config for wrapper
        wrapper_config = {
            'decay_rate': config.get('physics', {}).get('decay_rate', 0.05),
            'ml_model_type': config.get('ml_model', {}).get('type', 'catboost'),  # Default to catboost (works on Apple Silicon)
            'clip_min': config.get('physics', {}).get('clip_min', 0.0),
            'clip_max': config.get('physics', {}).get('clip_max', 2.0),
            'early_stopping_rounds': config.get('ml_model', {}).get('early_stopping_rounds', 50),
        }
        ml_type = wrapper_config['ml_model_type']
        ml_params = config.get('ml_model', {}).get(ml_type, {})
        if ml_params:
            wrapper_config['ml_params'] = ml_params
        return HybridPhysicsMLWrapper(wrapper_config)
    
    # ARIMA + Holt-Winters hybrid model (using wrapper for standard interface)
    elif model_type_lower in ('arihow', 'arima_hw', 'arima_holtwinters'):
        from .models.arihow import ARIHOWWrapper
        # Build config for wrapper
        wrapper_config = {
            'arima': config.get('arima', {}),
            'holt_winters': config.get('holt_winters', {}),
            'blend_weight': config.get('blend_weight', 0.5),
            'min_history_months': config.get('min_history_months', 3),
            'weight_window': config.get('weight_window', 12),
            'suppress_warnings': config.get('suppress_warnings', True)
        }
        return ARIHOWWrapper(wrapper_config)
    
    # KG-GCN-LSTM model (from KG-GCN-LSTM paper)
    elif model_type_lower in ('kg_gcn_lstm', 'kggcnlstm', 'gcn_lstm'):
        from .models.kg_gcn_lstm import KGGCNLSTMModel
        return KGGCNLSTMModel(config)
    
    # CNN-LSTM model (from Li et al. 2024)
    elif model_type_lower in ('cnn_lstm', 'cnnlstm'):
        from .models.cnn_lstm import CNNLSTMModel
        return CNNLSTMModel(config)
    
    # LSTM-only model (for ablation comparison)
    elif model_type_lower in ('lstm', 'lstm_only'):
        from .models.cnn_lstm import LSTMModel
        return LSTMModel(config)
    
    else:
        available = [
            'catboost', 'lightgbm', 'xgboost', 'linear', 'nn',
            'global_mean', 'flat', 'trend', 'historical_curve',
            'averaging', 'weighted', 'stacking', 'blending',
            'hybrid', 'hybrid_lgbm', 'hybrid_xgb', 'arihow',
            'kg_gcn_lstm', 'cnn_lstm', 'lstm'
        ]
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")


def train_hybrid_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario,
    model_config: Optional[dict] = None,
    run_config: Optional[dict] = None,
    run_id: Optional[str] = None,
    metrics_dir: Optional[Path] = None
) -> Tuple[Any, Dict]:
    """
    Train a hybrid physics + ML model for a specific scenario.
    
    The hybrid model requires additional inputs:
    - avg_vol: Pre-LOE average volume (from meta_train/meta_val)
    - months_postgx: Months since generic entry
    
    Args:
        X_train, y_train, meta_train: Training data
        X_val, y_val, meta_val: Validation data
        scenario: 1 or 2
        model_config: Hybrid model configuration
        run_config: Run configuration for sample weights
        run_id: Optional run ID for tracking
        metrics_dir: Optional directory for metrics
        
    Returns:
        (trained_model, metrics_dict)
    """
    from .models.hybrid_physics_ml import HybridPhysicsMLModel
    
    scenario = _normalize_scenario(scenario)
    model_config = model_config or {}
    
    # Extract hybrid-specific config
    decay_rate = model_config.get('physics', {}).get('decay_rate', 0.05)
    ml_type = model_config.get('ml_model', {}).get('type', 'lightgbm')
    ml_params = model_config.get('ml_model', {}).get(ml_type, {})
    
    # Create model
    model = HybridPhysicsMLModel(
        ml_model_type=ml_type,
        decay_rate=decay_rate,
        params=ml_params if ml_params else None
    )
    
    # Extract avg_vol and months from metadata
    avg_vol_train = meta_train['avg_vol_12m'].values
    months_train = meta_train['months_postgx'].values
    avg_vol_val = meta_val['avg_vol_12m'].values
    months_val = meta_val['months_postgx'].values
    
    # Compute sample weights
    sample_weights = compute_sample_weights(meta_train, scenario, config=run_config)
    
    # Train
    train_start = time.time()
    
    early_stopping_rounds = model_config.get('training', {}).get('early_stopping_rounds', 50)
    
    model.fit(
        X_train, y_train.values if isinstance(y_train, pd.Series) else y_train,
        avg_vol_train=avg_vol_train,
        months_train=months_train,
        X_val=X_val,
        y_val=y_val.values if isinstance(y_val, pd.Series) else y_val,
        avg_vol_val=avg_vol_val,
        months_val=months_val,
        sample_weight_train=sample_weights.values if isinstance(sample_weights, pd.Series) else sample_weights,
        early_stopping_rounds=early_stopping_rounds
    )
    
    train_time = time.time() - train_start
    
    # Compute validation metrics
    val_preds_norm = model.predict(X_val, avg_vol_val, months_val)
    
    # Denormalize for official metric
    val_preds_volume = val_preds_norm * avg_vol_val
    val_actual_volume = y_val.values * avg_vol_val
    
    # Build DataFrames for official metric
    df_pred = meta_val[['country', 'brand_name', 'months_postgx']].copy()
    df_pred['volume'] = val_preds_volume
    
    df_actual = meta_val[['country', 'brand_name', 'months_postgx']].copy()
    df_actual['volume'] = val_actual_volume
    
    # Create aux file from validation data
    val_with_bucket = meta_val[['country', 'brand_name', 'avg_vol_12m', 'bucket']].drop_duplicates()
    val_with_bucket = val_with_bucket.rename(columns={'avg_vol_12m': 'avg_vol'})
    
    # Compute official metric
    try:
        if scenario == 1:
            official_metric = compute_metric1(df_actual, df_pred, val_with_bucket)
        else:
            official_metric = compute_metric2(df_actual, df_pred, val_with_bucket)
    except Exception as e:
        logger.warning(f"Could not compute official metric: {e}")
        official_metric = np.nan
    
    # Additional metrics
    rmse = np.sqrt(np.mean((val_preds_norm - y_val.values) ** 2))
    mae = np.mean(np.abs(val_preds_norm - y_val.values))
    
    # Get physics vs ML contribution stats
    train_stats = model.get_training_stats()
    
    metrics = {
        'official_metric': official_metric,
        'rmse_norm': rmse,
        'mae_norm': mae,
        'scenario': scenario,
        'model_type': f'hybrid_{ml_type}',
        'train_time_seconds': train_time,
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_features': len(X_train.columns),
        'decay_rate': decay_rate,
        'physics_rmse': train_stats.get('physics_rmse', np.nan),
        'residual_std': train_stats.get('residual_std', np.nan),
    }
    
    logger.info(f"Hybrid model metrics: Official={official_metric:.4f}, RMSE={rmse:.4f}")
    logger.info(f"Physics baseline RMSE: {train_stats.get('physics_rmse', 'N/A')}")
    
    return model, metrics


def run_cross_validation(
    panel_features: pd.DataFrame,
    scenario,
    model_type: str = 'catboost',
    model_config: Optional[dict] = None,
    run_config: Optional[dict] = None,
    n_folds: int = 5,
    save_oof: bool = True,
    artifacts_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    metrics_dir: Optional[Path] = None,
    tracker: Optional['ExperimentTracker'] = None
) -> Tuple[List[Any], Dict, pd.DataFrame]:
    """
    Run K-fold cross-validation at series level with optional experiment tracking.
    
    Section 13.1/13.3: Integrates ExperimentTracker for logging per-fold and
    aggregated CV metrics.
    
    Args:
        panel_features: DataFrame with features already built
        scenario: 1 or 2
        model_type: Model type to train
        model_config: Model configuration dict
        run_config: Run configuration dict
        n_folds: Number of folds
        save_oof: Whether to save out-of-fold predictions
        artifacts_dir: Directory to save artifacts
        run_id: Optional run ID for metrics logging
        metrics_dir: Optional directory to save unified metric records
        tracker: Optional ExperimentTracker for logging CV metrics
        
    Returns:
        (list of models, aggregated metrics dict, OOF predictions DataFrame)
    """
    scenario = _normalize_scenario(scenario)
    seed = run_config.get('reproducibility', {}).get('seed', 42) if run_config else 42
    
    # Get folds
    folds = get_fold_series(panel_features, n_folds=n_folds, random_state=seed)
    
    fold_metrics = []
    models = []
    oof_predictions = []
    
    logger.info(f"Starting {n_folds}-fold cross-validation for scenario {scenario}")
    
    # Log CV configuration to tracker if available
    if tracker and tracker._run_active:
        tracker.log_params({
            'cv_n_folds': n_folds,
            'cv_scenario': scenario,
            'cv_model_type': model_type,
        })
    
    for fold_idx, (train_df, val_df) in enumerate(folds):
        logger.info(f"=== Fold {fold_idx + 1}/{n_folds} ===")
        
        # Split features/target/meta
        X_train, y_train, meta_train = split_features_target_meta(train_df)
        X_val, y_val, meta_val = split_features_target_meta(val_df)
        
        # Train model (with unified logging if metrics_dir provided)
        model, metrics = train_scenario_model(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=scenario,
            model_type=model_type,
            model_config=model_config,
            run_config=run_config,
            run_id=run_id,
            metrics_dir=metrics_dir,
            fold_idx=fold_idx
        )
        
        metrics['fold'] = fold_idx + 1
        fold_metrics.append(metrics)
        models.append(model)
        
        # Log per-fold metrics to tracker
        if tracker and tracker._run_active:
            tracker.log_metrics({
                f'fold{fold_idx+1}_official_metric': metrics['official_metric'],
                f'fold{fold_idx+1}_rmse_norm': metrics['rmse_norm'],
                f'fold{fold_idx+1}_mae_norm': metrics['mae_norm'],
            }, step=fold_idx)
        
        # Collect OOF predictions
        if save_oof:
            val_preds = model.predict(X_val)
            oof_df = meta_val[['country', 'brand_name', 'months_postgx']].copy()
            oof_df['y_true'] = y_val.values
            oof_df['y_pred'] = val_preds
            oof_df['fold'] = fold_idx + 1
            oof_predictions.append(oof_df)
        
        logger.info(f"Fold {fold_idx + 1} - Official metric: {metrics['official_metric']:.4f}")
    
    # Aggregate metrics
    official_scores = [m['official_metric'] for m in fold_metrics if not np.isnan(m['official_metric'])]
    rmse_scores = [m['rmse_norm'] for m in fold_metrics]
    mae_scores = [m['mae_norm'] for m in fold_metrics]
    
    agg_metrics = {
        'cv_official_mean': np.mean(official_scores) if official_scores else np.nan,
        'cv_official_std': np.std(official_scores) if official_scores else np.nan,
        'cv_rmse_mean': np.mean(rmse_scores),
        'cv_rmse_std': np.std(rmse_scores),
        'cv_mae_mean': np.mean(mae_scores),
        'cv_mae_std': np.std(mae_scores),
        'n_folds': n_folds,
        'fold_metrics': fold_metrics,
        'scenario': scenario,
        'model_type': model_type,
    }
    
    # Log aggregated CV metrics to tracker
    if tracker and tracker._run_active:
        tracker.log_metrics({
            'cv_official_mean': agg_metrics['cv_official_mean'],
            'cv_official_std': agg_metrics['cv_official_std'],
            'cv_rmse_mean': agg_metrics['cv_rmse_mean'],
            'cv_rmse_std': agg_metrics['cv_rmse_std'],
            'cv_mae_mean': agg_metrics['cv_mae_mean'],
            'cv_mae_std': agg_metrics['cv_mae_std'],
        })
    
    # Save aggregated CV metrics using unified logging
    if metrics_dir is not None:
        metrics_dir = Path(metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / 'metrics.csv'
        
        official_metric_name = METRIC_NAME_S1 if scenario == 1 else METRIC_NAME_S2
        agg_records = [
            make_metric_record(
                phase='cv', split='cv_agg', scenario=scenario,
                model_name=model_type, metric_name=f'{official_metric_name}_mean',
                value=agg_metrics['cv_official_mean'], run_id=run_id, step='cv_agg'
            ),
            make_metric_record(
                phase='cv', split='cv_agg', scenario=scenario,
                model_name=model_type, metric_name=f'{official_metric_name}_std',
                value=agg_metrics['cv_official_std'], run_id=run_id, step='cv_agg'
            ),
            make_metric_record(
                phase='cv', split='cv_agg', scenario=scenario,
                model_name=model_type, metric_name=f'{METRIC_NAME_RMSE}_mean',
                value=agg_metrics['cv_rmse_mean'], run_id=run_id, step='cv_agg'
            ),
            make_metric_record(
                phase='cv', split='cv_agg', scenario=scenario,
                model_name=model_type, metric_name=f'{METRIC_NAME_RMSE}_std',
                value=agg_metrics['cv_rmse_std'], run_id=run_id, step='cv_agg'
            ),
            make_metric_record(
                phase='cv', split='cv_agg', scenario=scenario,
                model_name=model_type, metric_name=f'{METRIC_NAME_MAE}_mean',
                value=agg_metrics['cv_mae_mean'], run_id=run_id, step='cv_agg'
            ),
            make_metric_record(
                phase='cv', split='cv_agg', scenario=scenario,
                model_name=model_type, metric_name=f'{METRIC_NAME_MAE}_std',
                value=agg_metrics['cv_mae_std'], run_id=run_id, step='cv_agg'
            ),
        ]
        save_metric_records(agg_records, metrics_path, append=True)
        logger.debug(f"Saved {len(agg_records)} CV aggregate metric records to {metrics_path}")
    
    logger.info(f"CV Complete - Official: {agg_metrics['cv_official_mean']:.4f} ± {agg_metrics['cv_official_std']:.4f}")
    logger.info(f"CV Complete - RMSE: {agg_metrics['cv_rmse_mean']:.4f} ± {agg_metrics['cv_rmse_std']:.4f}")
    
    # Combine OOF predictions
    oof_df = pd.concat(oof_predictions, ignore_index=True) if oof_predictions else pd.DataFrame()
    
    # Save artifacts if directory provided
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CV metrics
        with open(artifacts_dir / "cv_metrics.json", "w") as f:
            json.dump(agg_metrics, f, indent=2, default=str)
        
        # Save OOF predictions
        if len(oof_df) > 0:
            oof_df.to_csv(artifacts_dir / "oof_predictions.csv", index=False)
            logger.info(f"OOF predictions saved to {artifacts_dir / 'oof_predictions.csv'}")
        
        # Save each fold's model
        for fold_idx, model in enumerate(models):
            model_path = artifacts_dir / f"model_fold{fold_idx + 1}.bin"
            model.save(str(model_path))
    
    return models, agg_metrics, oof_df


def run_experiment(
    scenario,
    model_type: str = 'catboost',
    model_config_path: Optional[str] = None,
    model_config: Optional[Dict] = None,
    run_config_path: str = 'configs/run_defaults.yaml',
    data_config_path: str = 'configs/data.yaml',
    features_config_path: str = 'configs/features.yaml',
    run_name: Optional[str] = None,
    use_cached_features: bool = True,
    force_rebuild: bool = False
) -> Tuple[Any, Dict]:
    """
    Run a full experiment: load data, train model, evaluate.
    
    Uses get_features() for cached feature loading when use_cached_features=True.
    
    Args:
        scenario: 1, 2, "scenario1", or "scenario2"
        model_type: Model type to train
        model_config_path: Path to model config
        model_config: Model config dict (takes precedence over path)
        run_config_path: Path to run defaults config
        data_config_path: Path to data config
        features_config_path: Path to features config
        run_name: Optional custom run name
        use_cached_features: If True, use get_features() with caching
        force_rebuild: If True, rebuild features even if cached
        
    Returns:
        (trained_model, metrics_dict)
    """
    scenario = _normalize_scenario(scenario)
    # Load configs
    run_config = load_config(run_config_path)
    data_config = load_config(data_config_path)
    features_config = load_config(features_config_path) if features_config_path else {}
    
    # Load model config from dict or path
    if model_config is None:
        model_config = load_config(model_config_path) if model_config_path else {}
    
    # Setup
    set_seed(run_config['reproducibility']['seed'])
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = run_name or f"{timestamp}_{model_type}_scenario{scenario}"
    
    # Setup artifacts directory
    artifacts_dir = get_project_root() / run_config['paths']['artifacts_dir'] / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_file=str(artifacts_dir / "train.log"))
    logger.info(f"Starting experiment: {run_id}")
    
    # Save exact config file copies and compute hash for reproducibility
    config_hash = save_config_snapshot(
        artifacts_dir=artifacts_dir,
        run_config_path=run_config_path,
        data_config_path=data_config_path,
        features_config_path=features_config_path,
        model_config_path=model_config_path
    )
    logger.info(f"Config hash: {config_hash}")
    
    # Also save a merged config snapshot with scenario/model info for quick reference
    config_dict = {
        'run_config': run_config,
        'data_config': data_config,
        'features_config': features_config,
        'model_config': model_config,
        'scenario': scenario,
        'model_type': model_type,
        'config_hash': config_hash
    }
    with open(artifacts_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config_dict, f)
    
    # Load and prepare data using cached features if enabled
    if use_cached_features:
        # BONUS: G6 - Check if augmentation is enabled
        # If augmentation is enabled, we need to rebuild features (can't use cache)
        augmentation_config = run_config.get('augmentation', {})
        use_augmentation = augmentation_config.get('enabled', False)
        
        # If augmentation is enabled, we need to rebuild features to apply augmentation
        # Otherwise, use cached features for speed
        force_rebuild_features = force_rebuild or use_augmentation
        
        with timer("Load features (cached)"):
            # Pass run_config to features_config so augmentation can be applied
            if use_augmentation and features_config is not None:
                # Merge augmentation config into features_config for get_features
                features_config_with_aug = features_config.copy()
                features_config_with_aug['run_config'] = run_config
            else:
                features_config_with_aug = features_config
            
            X_full, y_full, meta_full = get_features(
                split='train',
                scenario=scenario,
                mode='train',
                data_config=data_config,
                features_config=features_config_with_aug,
                use_cache=not force_rebuild_features,
                force_rebuild=force_rebuild_features
            )
            # Combine for validation split
            # Remove columns from meta that already exist in X to avoid duplicates
            meta_cols_to_add = [c for c in meta_full.columns if c not in X_full.columns]
            train_rows = pd.concat([X_full, meta_full[meta_cols_to_add]], axis=1)
            train_rows['y_norm'] = y_full
            
            # Get panel for metadata (load cached)
            panel = get_panel('train', data_config, use_cache=True, force_rebuild=force_rebuild)
    else:
        # Legacy path: build features manually
        with timer("Load and prepare data"):
            train_data = load_raw_data(data_config, split='train')
            
            panel = prepare_base_panel(
                train_data['volume'],
                train_data['generics'],
                train_data['medicine_info']
            )
            panel = handle_missing_values(panel)
            panel = compute_pre_entry_stats(panel, is_train=True)
            
            # BONUS: G6 - Apply data augmentation if enabled
            augmentation_config = run_config.get('augmentation', {})
            if augmentation_config.get('enabled', False):
                from .data import augment_panel
                logger.info("Applying data augmentation to training panel")
                panel = augment_panel(panel, run_config)
        
        with timer("Feature engineering"):
            panel_features = make_features(panel, scenario=scenario, mode='train', config=features_config)
            train_rows = select_training_rows(panel_features, scenario=scenario)
    
    # Create validation split
    train_df, val_df = create_validation_split(
        train_rows,
        val_fraction=run_config['validation']['val_fraction'],
        stratify_by=run_config['validation']['stratify_by'],
        random_state=run_config['reproducibility']['seed']
    )
    
    # Split features/target/meta
    X_train, y_train, meta_train = split_features_target_meta(train_df)
    X_val, y_val, meta_val = split_features_target_meta(val_df)
    
    # Save experiment metadata
    metadata = get_experiment_metadata(
        scenario=scenario,
        model_type=model_type,
        run_config=run_config,
        data_config=data_config,
        model_config=model_config,
        panel_df=panel,
        train_df=train_df,
        val_df=val_df
    )
    metadata['config_hash'] = config_hash  # Add config hash for reproducibility
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # BONUS: B8 - Multi-seed training
    multi_seed_config = run_config.get('multi_seed', {})
    if multi_seed_config.get('enabled', False):
        logger.info("Multi-seed experiment enabled")
        seed_results = run_multi_seed_experiment(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=scenario,
            model_type=model_type,
            model_config=model_config,
            run_config=run_config,
            artifacts_dir=artifacts_dir
        )
        # Use best seed model
        best_seed = seed_results.loc[seed_results['official_metric'].idxmin(), 'seed']
        logger.info(f"Best seed: {best_seed}")
        # Load best model
        best_model_path = artifacts_dir / f'seed_{best_seed}' / 'model.bin'
        if best_model_path.exists():
            from .models.cat_model import CatBoostModel
            model = CatBoostModel(model_config)
            model.load(str(best_model_path))
            metrics = {'official_metric': seed_results.loc[seed_results['seed'] == best_seed, 'official_metric'].iloc[0]}
        else:
            # Fall back to regular training
            model, metrics = train_scenario_model(
                X_train, y_train, meta_train,
                X_val, y_val, meta_val,
                scenario=scenario,
                model_type=model_type,
                model_config=model_config,
                run_config=run_config
            )
    # BONUS: B2 - Bucket specialization
    elif run_config.get('bucket_specialization', {}).get('enabled', False):
        logger.info("Bucket specialization enabled")
        bucket_models = train_bucket_specialized_models(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=scenario,
            model_type=model_type,
            model_config=model_config,
            run_config=run_config,
            artifacts_dir=artifacts_dir
        )
        # For compatibility, use bucket 1 model as primary
        if 1 in bucket_models:
            model, metrics = bucket_models[1]
        else:
            # Fall back to regular training
            model, metrics = train_scenario_model(
                X_train, y_train, meta_train,
                X_val, y_val, meta_val,
                scenario=scenario,
                model_type=model_type,
                model_config=model_config,
                run_config=run_config
            )
    else:
        # Regular training
        model, metrics = train_scenario_model(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=scenario,
            model_type=model_type,
            model_config=model_config,
            run_config=run_config
        )
    
    # BONUS: B3 - Fit calibration on validation set
    calibration_config = run_config.get('calibration', {})
    if calibration_config.get('enabled', False):
        logger.info("Fitting calibration parameters")
        # Build validation DataFrame with predictions
        # Note: model.predict() already returns inverse-transformed predictions if target transform was used
        # Prepare X_val for prediction (add meta columns if needed for hybrid/arihow models)
        X_val_for_calib = X_val.copy()
        if model_type.lower().startswith('hybrid'):
            if 'months_postgx' not in X_val_for_calib.columns:
                X_val_for_calib['months_postgx'] = meta_val['months_postgx'].values
            if 'avg_vol_12m' not in X_val_for_calib.columns:
                X_val_for_calib['avg_vol_12m'] = meta_val['avg_vol_12m'].values
        elif model_type.lower() in ('arihow', 'arima_hw', 'arima_holtwinters'):
            for col in ['country', 'brand_name', 'months_postgx', 'avg_vol_12m']:
                if col not in X_val_for_calib.columns:
                    X_val_for_calib[col] = meta_val[col].values
        
        val_preds_norm = model.predict(X_val_for_calib)
        avg_vol_val = meta_val['avg_vol_12m'].values
        val_preds_volume = val_preds_norm * avg_vol_val
        # Get actual volumes - model was trained on y_norm, so y_val is already in normalized space
        val_actual_volume = y_val.values * avg_vol_val
        
        df_val_calib = meta_val[['country', 'brand_name', 'months_postgx']].copy()
        df_val_calib['scenario'] = scenario
        df_val_calib['bucket'] = meta_val.get('bucket', 1)
        df_val_calib['volume_true'] = val_actual_volume
        df_val_calib['volume_pred'] = val_preds_volume
        
        calibration_params = fit_grouped_calibration(
            df_val_calib,
            run_config,
            artifacts_dir=artifacts_dir
        )
        logger.info(f"Calibration parameters fitted for {len(calibration_params)} groups")
    
    # BONUS: B6 - Fit bias corrections
    bias_config = run_config.get('bias_correction', {})
    if bias_config.get('enabled', False):
        logger.info("Fitting bias corrections")
        # Prepare X_val for prediction (add meta columns if needed for hybrid/arihow models)
        X_val_for_bias = X_val.copy()
        if model_type.lower().startswith('hybrid'):
            if 'months_postgx' not in X_val_for_bias.columns:
                X_val_for_bias['months_postgx'] = meta_val['months_postgx'].values
            if 'avg_vol_12m' not in X_val_for_bias.columns:
                X_val_for_bias['avg_vol_12m'] = meta_val['avg_vol_12m'].values
        elif model_type.lower() in ('arihow', 'arima_hw', 'arima_holtwinters'):
            for col in ['country', 'brand_name', 'months_postgx', 'avg_vol_12m']:
                if col not in X_val_for_bias.columns:
                    X_val_for_bias[col] = meta_val[col].values
        
        val_preds_norm = model.predict(X_val_for_bias)
        avg_vol_val = meta_val['avg_vol_12m'].values
        val_preds_volume = val_preds_norm * avg_vol_val
        # Get actual volumes - model was trained on y_norm, so y_val is already in normalized space
        val_actual_volume = y_val.values * avg_vol_val
        
        df_val_bias = meta_val.copy()
        df_val_bias['volume_true'] = val_actual_volume
        df_val_bias['volume_pred'] = val_preds_volume
        
        # Merge group columns (e.g., ther_area) from panel if needed
        group_cols = bias_config.get('group_cols', ['ther_area', 'country'])
        missing_group_cols = [col for col in group_cols if col not in df_val_bias.columns]
        if missing_group_cols and 'panel' in locals():
            # Try to merge from panel
            panel_group_cols = [col for col in missing_group_cols if col in panel.columns]
            if panel_group_cols:
                # Merge by country and brand_name
                panel_subset = panel[['country', 'brand_name'] + panel_group_cols].drop_duplicates()
                df_val_bias = df_val_bias.merge(
                    panel_subset,
                    on=['country', 'brand_name'],
                    how='left'
                )
                logger.info(f"Merged group columns from panel: {panel_group_cols}")
        
        # Filter to only use group columns that exist
        available_group_cols = [col for col in group_cols if col in df_val_bias.columns]
        if not available_group_cols:
            logger.warning(f"None of the requested group columns {group_cols} are available. Skipping bias correction.")
        else:
            # Temporarily update config to use only available columns
            bias_config_updated = bias_config.copy()
            bias_config_updated['group_cols'] = available_group_cols
            run_config_updated = run_config.copy()
            run_config_updated['bias_correction'] = bias_config_updated
            
            bias_corrections = fit_bias_corrections(
                df_val_bias,
                run_config_updated,
                artifacts_dir=artifacts_dir
            )
            logger.info(f"Bias corrections fitted for {len(bias_corrections)} groups")
    
    # BONUS: B7 - Feature pruning (if enabled, save importance for later use)
    pruning_config = features_config.get('feature_pruning', {})
    if pruning_config.get('enabled', False) and hasattr(model, 'get_feature_importance'):
        logger.info("Extracting feature importance for pruning")
        importance = model.get_feature_importance()
        if len(importance) > 0:
            # Convert to dict format
            feature_importance_dict = dict(zip(importance['feature'], importance['importance']))
            # Prune features (for future training runs)
            X_train_pruned, dropped_features = prune_features_by_importance(
                X_train,
                feature_importance_dict,
                pruning_config,
                artifacts_dir=artifacts_dir
            )
            logger.info(f"Feature pruning: {len(dropped_features)} features dropped")
    
    # Save model
    model_path = artifacts_dir / f"model_{scenario}.bin"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # BONUS: B10 - Save target transform parameters if used
    transform_config = run_config.get('target_transform', {}) if run_config else {}
    transform_type = transform_config.get('type', 'none')
    if transform_type != 'none':
        # Build transform params from config to save for inference
        transform_params = {
            'type': transform_type,
            'epsilon': float(transform_config.get('epsilon', 1e-6))
        }
        if transform_type == 'power':
            transform_params['exponent'] = float(transform_config.get('power_exponent', 0.5))
        transform_path = artifacts_dir / f"target_transform_params_{scenario}.json"
        import json as json_module
        with open(transform_path, 'w') as f:
            json_module.dump(transform_params, f, indent=2)
        logger.info(f"Target transform parameters saved to {transform_path}")
    
    # Save metrics
    import json as json_module
    with open(artifacts_dir / "metrics.json", "w") as f:
        json_module.dump(metrics, f, indent=2)
    
    # Save feature importance
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
        if len(importance) > 0:
            importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    
    logger.info(f"Experiment {run_id} completed. Artifacts saved to {artifacts_dir}")
    
    return model, metrics


# =============================================================================
# SECTION 8.2: MODEL EXPERIMENTS - Compare Models & Loss Functions
# =============================================================================

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_train: pd.DataFrame,
    meta_val: pd.DataFrame,
    scenario: int,
    model_configs: Dict[str, Dict],
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_val: Optional[pd.Series] = None,
    artifacts_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compare multiple model types on the same train/validation split.
    
    Section 21.1: Uses official metric for comparison and saves results to CSV.
    
    This function trains and evaluates different model configurations
    to identify the best performing approach.
    
    Args:
        X_train, y_train: Training features and target
        X_val, y_val: Validation features and target
        meta_train, meta_val: Metadata for metric computation
        scenario: 1 or 2
        model_configs: Dict mapping model name to configuration dict
                      Each config should have 'model_type' and optional 'params'
        sample_weight_train: Training sample weights
        sample_weight_val: Validation sample weights (for weighted metrics)
        artifacts_dir: Optional directory to save comparison results
        
    Returns:
        DataFrame with comparison results sorted by official metric
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    results = []
    
    for model_name, config in model_configs.items():
        model_type = config.get('model_type', model_name)
        model_params = config.get('params', {})
        
        logger.info(f"Training model: {model_name} ({model_type})")
        
        try:
            start_time = time.time()
            
            # Get model instance
            model = _get_model(model_type, model_params)
            
            # Train
            model.fit(
                X_train, y_train, X_val, y_val,
                sample_weight=sample_weight_train
            )
            
            train_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            # Compute official metric if possible
            official_metric = np.nan
            try:
                pred_volume = y_pred * meta_val['avg_vol_12m'].values
                actual_volume = y_val.values * meta_val['avg_vol_12m'].values
                
                pred_df = meta_val[['country', 'brand_name', 'months_postgx']].copy()
                pred_df['volume'] = pred_volume
                
                actual_df = pred_df[['country', 'brand_name', 'months_postgx']].copy()
                actual_df['volume'] = actual_volume
                
                aux_df = create_aux_file(meta_val, y_val)
                
                if scenario == 1:
                    official_metric = compute_metric1(actual_df, pred_df, aux_df)
                else:
                    official_metric = compute_metric2(actual_df, pred_df, aux_df)
            except Exception as e:
                logger.warning(f"Could not compute official metric for {model_name}: {e}")
            
            results.append({
                'model_name': model_name,
                'model_type': model_type,
                'official_metric': official_metric,
                'rmse': rmse,
                'mae': mae,
                'train_time_s': train_time,
                'n_features': len(X_train.columns),
                'status': 'success'
            })
            
            logger.info(f"  {model_name}: Official={official_metric:.4f}, RMSE={rmse:.4f}, "
                       f"MAE={mae:.4f}, Time={train_time:.1f}s")
            
        except Exception as e:
            logger.error(f"  {model_name} failed: {e}")
            results.append({
                'model_name': model_name,
                'model_type': model_type,
                'official_metric': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'train_time_s': np.nan,
                'n_features': len(X_train.columns),
                'status': f'failed: {str(e)[:50]}'
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('official_metric', ascending=True)
    
    logger.info(f"\nModel Comparison Results:")
    logger.info(f"\n{results_df.to_string()}")
    
    # Section 21.1.3: Save comparison tables to CSV
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = artifacts_dir / f'model_comparison_scenario{scenario}.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Model comparison saved to {csv_path}")
        
        # Also save as markdown
        md_path = artifacts_dir / f'model_comparison_scenario{scenario}.md'
        with open(md_path, 'w') as f:
            f.write(f"# Model Comparison - Scenario {scenario}\n\n")
            f.write(f"**Best Model**: {results_df.iloc[0]['model_name']}\n")
            f.write(f"**Best Official Metric**: {results_df.iloc[0]['official_metric']:.4f}\n\n")
            f.write(results_df.to_markdown(index=False))
    
    return results_df


def test_loss_functions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_train: pd.DataFrame,
    meta_val: pd.DataFrame,
    scenario: int,
    model_type: str = 'catboost',
    loss_functions: Optional[List[str]] = None,
    sample_weight_train: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Test different loss functions for a given model type.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        meta_train, meta_val: Metadata
        scenario: 1 or 2
        model_type: Base model type
        loss_functions: List of loss functions to test.
                       For CatBoost: 'RMSE', 'MAE', 'Quantile', 'Huber'
        sample_weight_train: Training sample weights
        
    Returns:
        DataFrame with results for each loss function
    """
    if loss_functions is None:
        if model_type.lower() in ('catboost', 'cat'):
            loss_functions = ['RMSE', 'MAE', 'Quantile:alpha=0.5', 'Huber:delta=1.0']
        elif model_type.lower() in ('lightgbm', 'lgbm'):
            loss_functions = ['regression', 'regression_l1', 'huber', 'quantile']
        elif model_type.lower() in ('xgboost', 'xgb'):
            loss_functions = ['reg:squarederror', 'reg:absoluteerror', 'reg:pseudohubererror']
        else:
            loss_functions = ['mse']  # Default for linear models
    
    # Build configs for each loss function
    model_configs = {}
    for loss_fn in loss_functions:
        config_name = f'{model_type}_{loss_fn.replace(":", "_").replace("=", "_")}'
        
        if model_type.lower() in ('catboost', 'cat'):
            model_configs[config_name] = {
                'model_type': model_type,
                'params': {'loss_function': loss_fn}
            }
        elif model_type.lower() in ('lightgbm', 'lgbm'):
            model_configs[config_name] = {
                'model_type': model_type,
                'params': {'objective': loss_fn}
            }
        elif model_type.lower() in ('xgboost', 'xgb'):
            model_configs[config_name] = {
                'model_type': model_type,
                'params': {'objective': loss_fn}
            }
        else:
            model_configs[config_name] = {
                'model_type': model_type,
                'params': {}
            }
    
    return compare_models(
        X_train, y_train, X_val, y_val,
        meta_train, meta_val, scenario,
        model_configs, sample_weight_train
    )


def run_model_experiments(
    panel_features: pd.DataFrame,
    scenario: int,
    run_config: dict,
    experiment_type: str = 'compare_models',
    custom_configs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Run comprehensive model experiments.
    
    Experiment types:
    - 'compare_models': Compare all model types
    - 'loss_functions': Compare loss functions for hero model
    - 'learning_rates': Compare different learning rates
    - 'ensemble_configs': Compare ensemble configurations
    
    Args:
        panel_features: Feature DataFrame
        scenario: 1 or 2
        run_config: Run configuration
        experiment_type: Type of experiment
        custom_configs: Custom model configurations
        
    Returns:
        Dictionary with experiment results
    """
    seed = run_config.get('reproducibility', {}).get('seed', 42)
    val_fraction = run_config.get('validation', {}).get('val_fraction', 0.2)
    
    # Get training rows
    from .features import select_training_rows
    train_rows = select_training_rows(panel_features, scenario=scenario)
    
    # Split
    train_df, val_df = create_validation_split(
        train_rows,
        val_fraction=val_fraction,
        stratify_by=run_config.get('validation', {}).get('stratify_by'),
        random_state=seed
    )
    
    X_train, y_train, meta_train = split_features_target_meta(train_df)
    X_val, y_val, meta_val = split_features_target_meta(val_df)
    
    # Compute sample weights
    sample_weight_train = compute_sample_weights(meta_train, scenario, run_config)
    
    results = {'experiment_type': experiment_type, 'scenario': scenario}
    
    if experiment_type == 'compare_models':
        # Default model configs for comparison
        if custom_configs is None:
            custom_configs = {
                'catboost_default': {'model_type': 'catboost', 'params': {}},
                'lightgbm_default': {'model_type': 'lightgbm', 'params': {}},
                'xgboost_default': {'model_type': 'xgboost', 'params': {}},
                'ridge': {'model_type': 'linear', 'params': {'model': {'type': 'ridge'}}},
                'global_mean': {'model_type': 'global_mean', 'params': {}},
                'flat': {'model_type': 'flat', 'params': {}},
            }
        
        results['comparison'] = compare_models(
            X_train, y_train, X_val, y_val,
            meta_train, meta_val, scenario,
            custom_configs, sample_weight_train
        )
        
    elif experiment_type == 'loss_functions':
        results['loss_functions'] = test_loss_functions(
            X_train, y_train, X_val, y_val,
            meta_train, meta_val, scenario,
            model_type=custom_configs.get('model_type', 'catboost') if custom_configs else 'catboost',
            sample_weight_train=sample_weight_train
        )
        
    elif experiment_type == 'learning_rates':
        learning_rates = [0.01, 0.03, 0.05, 0.1, 0.2]
        configs = {}
        for lr in learning_rates:
            configs[f'catboost_lr{lr}'] = {
                'model_type': 'catboost',
                'params': {'learning_rate': lr}
            }
        
        results['learning_rates'] = compare_models(
            X_train, y_train, X_val, y_val,
            meta_train, meta_val, scenario,
            configs, sample_weight_train
        )
    
    return results


# =============================================================================
# SECTION 8.4: POST-PROCESSING - Ensemble Weight Optimization on Validation
# =============================================================================

def optimize_ensemble_weights_on_validation(
    models: List[Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario: int,
    optimization_metric: str = 'official',
    n_restarts: int = 5
) -> Tuple[np.ndarray, float]:
    """
    Optimize ensemble weights on validation data using the official metric.
    
    Uses multiple restarts to avoid local minima.
    
    Args:
        models: List of trained models
        X_val: Validation features
        y_val: Validation target
        meta_val: Validation metadata (needs avg_vol_12m for inverse transform)
        scenario: 1 or 2
        optimization_metric: 'official', 'rmse', or 'mae'
        n_restarts: Number of random restarts for optimization
        
    Returns:
        Tuple of (optimal_weights, best_metric_value)
    """
    from scipy.optimize import minimize
    
    n_models = len(models)
    
    # Get predictions from all models
    all_preds = np.array([model.predict(X_val) for model in models])  # (n_models, n_samples)
    
    y_val_arr = y_val.values
    avg_vol = meta_val['avg_vol_12m'].values
    
    def compute_ensemble_metric(weights):
        """Compute metric for given weights."""
        weights = weights / weights.sum()  # Ensure sum to 1
        ensemble_pred = np.dot(weights, all_preds)
        
        if optimization_metric == 'rmse':
            return np.sqrt(np.mean((ensemble_pred - y_val_arr) ** 2))
        elif optimization_metric == 'mae':
            return np.mean(np.abs(ensemble_pred - y_val_arr))
        elif optimization_metric == 'official':
            try:
                pred_volume = ensemble_pred * avg_vol
                actual_volume = y_val_arr * avg_vol
                
                pred_df = meta_val[['country', 'brand_name', 'months_postgx']].copy()
                pred_df['volume'] = pred_volume
                
                actual_df = pred_df[['country', 'brand_name', 'months_postgx']].copy()
                actual_df['volume'] = actual_volume
                
                aux_df = create_aux_file(meta_val, y_val)
                
                if scenario == 1:
                    return compute_metric1(actual_df, pred_df, aux_df)
                else:
                    return compute_metric2(actual_df, pred_df, aux_df)
            except Exception:
                # Fallback to RMSE
                return np.sqrt(np.mean((ensemble_pred - y_val_arr) ** 2))
        else:
            return np.sqrt(np.mean((ensemble_pred - y_val_arr) ** 2))
    
    best_weights = None
    best_metric = float('inf')
    
    for restart in range(n_restarts):
        # Random initial weights
        if restart == 0:
            w0 = np.ones(n_models) / n_models  # Uniform start
        else:
            w0 = np.random.dirichlet(np.ones(n_models))
        
        # Optimize
        result = minimize(
            compute_ensemble_metric,
            w0,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(n_models)],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        if result.success and result.fun < best_metric:
            best_metric = result.fun
            best_weights = result.x / result.x.sum()
    
    if best_weights is None:
        best_weights = np.ones(n_models) / n_models
        best_metric = compute_ensemble_metric(best_weights)
    
    logger.info(f"Optimized ensemble weights: {best_weights.round(4)}")
    logger.info(f"Best {optimization_metric} metric: {best_metric:.4f}")
    
    return best_weights, best_metric


# =============================================================================
# SECTION 11: Optional Multi-Model Ensemble Training
# =============================================================================

def train_multi_model_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario: int,
    model_types: List[str] = ['catboost', 'linear'],
    model_configs: Optional[Dict[str, dict]] = None,
    run_config: Optional[dict] = None,
    optimize_weights: bool = True,
    optimization_metric: str = 'official',
    hero_model: str = 'catboost',
    ensemble_improvement_threshold: float = 0.0,
    n_restarts: int = 5,
    artifacts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Train multiple model types on the same data and build an optimized ensemble.
    
    Section 11: Optional Ensemble Layer
    
    This function implements a clean path to:
    1. Train CatBoost, Linear, and/or other models on the same folds
    2. Collect per-model validation predictions
    3. Optimize ensemble weights using the official metric
    4. Enable ensemble only if it improves over the hero model alone
    
    The ensemble is evaluated against the hero model (typically CatBoost).
    If the ensemble doesn't improve over the hero by at least the threshold,
    the function recommends using the hero model alone.
    
    Args:
        X_train: Training features
        y_train: Training target
        meta_train: Training metadata
        X_val: Validation features
        y_val: Validation target
        meta_val: Validation metadata
        scenario: 1 or 2
        model_types: List of model types to train (e.g., ['catboost', 'linear', 'hybrid'])
        model_configs: Dict mapping model_type to config override
        run_config: Run configuration (for sample weights)
        optimize_weights: If True, optimize ensemble weights on validation
        optimization_metric: 'official', 'rmse', or 'mae'
        hero_model: Model type to compare against (default: 'catboost')
        ensemble_improvement_threshold: Minimum improvement over hero to enable ensemble
        n_restarts: Number of restarts for weight optimization
        artifacts_dir: Optional directory to save results
        
    Returns:
        Dictionary with:
        - 'models': Dict[model_type, trained_model]
        - 'per_model_metrics': Dict[model_type, metrics_dict]
        - 'per_model_predictions': Dict[model_type, val_predictions]
        - 'ensemble_weights': np.ndarray (optimized weights)
        - 'ensemble_metric': float (ensemble's official metric)
        - 'hero_metric': float (hero model's official metric)
        - 'improvement': float (ensemble - hero, lower is better)
        - 'use_ensemble': bool (True if ensemble improves over hero)
        - 'recommendation': str (human-readable recommendation)
    """
    scenario = _normalize_scenario(scenario)
    model_configs = model_configs or {}
    
    logger.info(f"=== Multi-Model Ensemble Training: Scenario {scenario} ===")
    logger.info(f"Model types: {model_types}")
    logger.info(f"Hero model: {hero_model}")
    
    # Ensure hero model is in the list
    if hero_model not in model_types:
        model_types = [hero_model] + list(model_types)
        logger.info(f"Added hero model to list: {model_types}")
    
    # Train each model type
    trained_models = {}
    per_model_metrics = {}
    per_model_predictions = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type}...")
        
        config = model_configs.get(model_type, {})
        
        try:
            model, metrics = train_scenario_model(
                X_train=X_train,
                y_train=y_train,
                meta_train=meta_train,
                X_val=X_val,
                y_val=y_val,
                meta_val=meta_val,
                scenario=scenario,
                model_type=model_type,
                model_config=config,
                run_config=run_config
            )
            
            # Store model and metrics
            trained_models[model_type] = model
            per_model_metrics[model_type] = metrics
            
            # Get validation predictions
            val_preds = model.predict(X_val)
            per_model_predictions[model_type] = val_preds
            
            logger.info(f"  {model_type}: official={metrics.get('official_metric', np.nan):.4f}, "
                       f"rmse={metrics.get('rmse_norm', np.nan):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            continue
    
    if len(trained_models) == 0:
        raise ValueError("No models were successfully trained")
    
    if len(trained_models) == 1:
        # Only one model trained - no ensemble possible
        model_type = list(trained_models.keys())[0]
        metrics = per_model_metrics[model_type]
        
        result = {
            'models': trained_models,
            'per_model_metrics': per_model_metrics,
            'per_model_predictions': per_model_predictions,
            'ensemble_weights': np.array([1.0]),
            'ensemble_metric': metrics.get('official_metric', np.nan),
            'hero_metric': metrics.get('official_metric', np.nan),
            'improvement': 0.0,
            'use_ensemble': False,
            'recommendation': f"Only {model_type} trained - no ensemble possible"
        }
        return result
    
    # Get hero model metric
    hero_metric = per_model_metrics.get(hero_model, {}).get('official_metric', np.nan)
    
    # Optimize ensemble weights
    if optimize_weights:
        logger.info("Optimizing ensemble weights on validation...")
        
        models_list = [trained_models[mt] for mt in model_types if mt in trained_models]
        
        optimal_weights, ensemble_metric = optimize_ensemble_weights_on_validation(
            models=models_list,
            X_val=X_val,
            y_val=y_val,
            meta_val=meta_val,
            scenario=scenario,
            optimization_metric=optimization_metric,
            n_restarts=n_restarts
        )
    else:
        # Equal weights
        n_models = len(trained_models)
        optimal_weights = np.ones(n_models) / n_models
        
        # Compute ensemble metric with equal weights
        all_preds = np.array([per_model_predictions[mt] for mt in model_types if mt in trained_models])
        ensemble_preds = np.dot(optimal_weights, all_preds)
        
        # Compute official metric for ensemble
        avg_vol = meta_val['avg_vol_12m'].values
        pred_volume = ensemble_preds * avg_vol
        actual_volume = y_val.values * avg_vol
        
        pred_df = meta_val[['country', 'brand_name', 'months_postgx']].copy()
        pred_df['volume'] = pred_volume
        
        actual_df = pred_df[['country', 'brand_name', 'months_postgx']].copy()
        actual_df['volume'] = actual_volume
        
        aux_df = create_aux_file(meta_val, y_val)
        
        try:
            if scenario == 1:
                ensemble_metric = compute_metric1(actual_df, pred_df, aux_df)
            else:
                ensemble_metric = compute_metric2(actual_df, pred_df, aux_df)
        except Exception as e:
            logger.warning(f"Could not compute ensemble official metric: {e}")
            ensemble_metric = np.nan
    
    # Compute improvement (lower is better for official metric)
    improvement = hero_metric - ensemble_metric  # positive = ensemble is better
    
    # Determine if ensemble should be used
    use_ensemble = improvement > ensemble_improvement_threshold and not np.isnan(improvement)
    
    # Generate recommendation
    if np.isnan(hero_metric) or np.isnan(ensemble_metric):
        recommendation = "Could not compute metrics - cannot recommend"
    elif use_ensemble:
        recommendation = (
            f"USE ENSEMBLE: Improves over {hero_model} by {improvement:.4f} "
            f"({hero_metric:.4f} -> {ensemble_metric:.4f})"
        )
    else:
        recommendation = (
            f"USE {hero_model.upper()} ALONE: Ensemble improvement ({improvement:.4f}) "
            f"below threshold ({ensemble_improvement_threshold:.4f})"
        )
    
    logger.info(f"\n{'='*60}")
    logger.info("ENSEMBLE TRAINING RESULTS:")
    logger.info(f"  Hero model ({hero_model}): {hero_metric:.4f}")
    logger.info(f"  Ensemble: {ensemble_metric:.4f}")
    logger.info(f"  Improvement: {improvement:.4f}")
    logger.info(f"  Weights: {dict(zip([mt for mt in model_types if mt in trained_models], optimal_weights.round(4)))}")
    logger.info(f"  Recommendation: {recommendation}")
    logger.info('='*60)
    
    result = {
        'models': trained_models,
        'per_model_metrics': per_model_metrics,
        'per_model_predictions': per_model_predictions,
        'ensemble_weights': optimal_weights,
        'ensemble_metric': ensemble_metric,
        'hero_metric': hero_metric,
        'improvement': improvement,
        'use_ensemble': use_ensemble,
        'recommendation': recommendation
    }
    
    # Save results to artifacts if provided
    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            'scenario': scenario,
            'model_types': model_types,
            'hero_model': hero_model,
            'hero_metric': float(hero_metric) if not np.isnan(hero_metric) else None,
            'ensemble_metric': float(ensemble_metric) if not np.isnan(ensemble_metric) else None,
            'improvement': float(improvement) if not np.isnan(improvement) else None,
            'use_ensemble': use_ensemble,
            'weights': {
                mt: float(w) for mt, w in zip(
                    [mt for mt in model_types if mt in trained_models], 
                    optimal_weights
                )
            },
            'per_model_metrics': {
                mt: {k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else v 
                     for k, v in metrics.items()}
                for mt, metrics in per_model_metrics.items()
            },
            'recommendation': recommendation
        }
        
        import json
        with open(artifacts_dir / 'ensemble_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved ensemble summary to {artifacts_dir / 'ensemble_summary.json'}")
    
    return result


def train_xgb_lgbm_ensemble(
    scenario: int,
    xgb_config_path: Optional[str] = None,
    lgbm_config_path: Optional[str] = None,
    run_config_path: str = 'configs/run_defaults.yaml',
    data_config_path: str = 'configs/data.yaml',
    features_config_path: str = 'configs/features.yaml',
    ensemble_method: str = 'weighted',
    optimize_weights: bool = True,
    use_official_metric: bool = True,
    run_name: Optional[str] = None,
    use_cached_features: bool = True,
    enable_tracking: bool = True
) -> Dict[str, Any]:
    """
    Train an XGBoost + LightGBM ensemble with optional weight optimization.
    
    Section 13.1/21.2: Integrates ExperimentTracker for logging individual
    and ensemble metrics. Uses official metric for weight optimization.
    
    This function trains both XGBoost and LightGBM models and combines them
    into an ensemble. Weights can be:
    1. Equal (simple averaging)
    2. Optimized to minimize MSE on validation
    3. Optimized to minimize official metric on validation (recommended)
    
    Args:
        scenario: 1 or 2
        xgb_config_path: Path to XGBoost config
        lgbm_config_path: Path to LightGBM config
        run_config_path: Path to run defaults config
        data_config_path: Path to data config
        features_config_path: Path to features config
        ensemble_method: 'averaging' or 'weighted' (default: 'weighted')
        optimize_weights: If True and method='weighted', optimize ensemble weights
        use_official_metric: If True, optimize weights using official metric
        run_name: Optional run name
        use_cached_features: If True, use cached feature loading
        enable_tracking: If True, use ExperimentTracker (if configured)
        
    Returns:
        Dictionary with:
            - 'ensemble': Fitted ensemble model
            - 'xgb_model': Fitted XGBoost model
            - 'lgbm_model': Fitted LightGBM model
            - 'weights': Final ensemble weights [xgb_weight, lgbm_weight]
            - 'metrics': Dict with individual and ensemble metrics
            - 'artifacts_dir': Path to saved artifacts
    """
    from .models.ensemble import AveragingEnsemble, WeightedAveragingEnsemble
    
    scenario = _normalize_scenario(scenario)
    
    # Load configs
    run_config = load_config(run_config_path)
    data_config = load_config(data_config_path)
    features_config = load_config(features_config_path) if features_config_path else {}
    xgb_config = load_config(xgb_config_path) if xgb_config_path else {}
    lgbm_config = load_config(lgbm_config_path) if lgbm_config_path else {}
    
    # Setup
    seed = run_config['reproducibility']['seed']
    set_seed(seed)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = run_name or generate_run_name('xgb_lgbm_ensemble', scenario, timestamp=timestamp)
    
    artifacts_dir = get_project_root() / run_config['paths']['artifacts_dir'] / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_file=str(artifacts_dir / "ensemble.log"))
    logger.info(f"Training XGB+LGBM ensemble: {run_name}")
    
    # Setup experiment tracking (Section 13.1)
    tracker = None
    if enable_tracking:
        tracker = setup_experiment_tracking(run_config, run_name)
        if tracker:
            standard_tags = get_standard_tags(run_config)
            tracker.start_run(
                run_name=run_name,
                tags=standard_tags,
                config={
                    'scenario': scenario,
                    'ensemble_method': ensemble_method,
                    'optimize_weights': optimize_weights,
                    'use_official_metric': use_official_metric,
                }
            )
            # Log parameters (Section 13.2)
            tracker.log_params({
                'scenario': scenario,
                'ensemble_method': ensemble_method,
                'optimize_weights': optimize_weights,
                'use_official_metric': use_official_metric,
                'seed': seed,
            })
    
    try:
        # Load data
        logger.info("Loading features...")
        if use_cached_features:
            X_full, y_full, meta_full = get_features(
                split='train',
                scenario=scenario,
                mode='train',
                data_config=data_config,
                features_config=features_config,
                use_cache=True
            )
            full_df = pd.concat([X_full, meta_full], axis=1)
            full_df['y_norm'] = y_full
        else:
            train_data = load_raw_data(data_config, split='train')
            panel = prepare_base_panel(
                train_data['volume'],
                train_data['generics'],
                train_data['medicine_info']
            )
            panel = handle_missing_values(panel)
            panel = compute_pre_entry_stats(panel, is_train=True)
            panel_features = make_features(panel, scenario=scenario, mode='train', config=features_config)
            full_df = select_training_rows(panel_features, scenario=scenario)
        
        # Create train/val split
        train_df, val_df = create_validation_split(
            full_df,
            val_fraction=run_config['validation']['val_fraction'],
            stratify_by=run_config['validation']['stratify_by'],
            random_state=seed
        )
        
        X_train, y_train, meta_train = split_features_target_meta(train_df)
        X_val, y_val, meta_val = split_features_target_meta(val_df)
        
        # Train XGBoost
        logger.info("Training XGBoost model...")
        xgb_model, xgb_metrics = train_scenario_model(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=scenario,
            model_type='xgboost',
            model_config=xgb_config,
            run_config=run_config
        )
        xgb_model.save(str(artifacts_dir / "xgb_model.bin"))
        logger.info(f"XGBoost official_metric: {xgb_metrics.get('official_metric', np.nan):.4f}")
        
        # Train LightGBM
        logger.info("Training LightGBM model...")
        lgbm_model, lgbm_metrics = train_scenario_model(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=scenario,
            model_type='lightgbm',
            model_config=lgbm_config,
            run_config=run_config
        )
        lgbm_model.save(str(artifacts_dir / "lgbm_model.bin"))
        logger.info(f"LightGBM official_metric: {lgbm_metrics.get('official_metric', np.nan):.4f}")
        
        # Create ensemble
        models = [xgb_model, lgbm_model]
        model_names = ['xgboost', 'lightgbm']
        
        if ensemble_method == 'weighted' and optimize_weights:
            logger.info("Optimizing ensemble weights...")
            
            # Get validation predictions
            xgb_preds = xgb_model.predict(X_val)
            lgbm_preds = lgbm_model.predict(X_val)
            preds_list = [xgb_preds, lgbm_preds]
            
            if use_official_metric:
                # Optimize using official metric
                weights, best_metric = optimize_ensemble_weights_on_validation(
                    models=[xgb_model, lgbm_model],
                    X_val=X_val,
                    y_val=y_val,
                    meta_val=meta_val,
                    scenario=scenario,
                    optimization_metric='official'
                )
                logger.info(f"Optimized weights (official metric): XGB={weights[0]:.3f}, LGBM={weights[1]:.3f}")
            else:
                # Optimize using MSE
                from scipy.optimize import minimize
                
                def ensemble_mse(w):
                    w = w / w.sum()
                    pred = w[0] * xgb_preds + w[1] * lgbm_preds
                    return np.mean((pred - y_val.values) ** 2)
                
                result = minimize(
                    ensemble_mse,
                    [0.5, 0.5],
                    method='SLSQP',
                    bounds=[(0, 1), (0, 1)],
                    constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
                )
                weights = result.x / result.x.sum()
                logger.info(f"Optimized weights (MSE): XGB={weights[0]:.3f}, LGBM={weights[1]:.3f}")
            
            # Create weighted ensemble
            ensemble = WeightedAveragingEnsemble({
                'models': models,
                'weights': list(weights),
                'clip_predictions': True
            })
            ensemble.feature_names = list(X_train.columns)
        else:
            # Simple averaging
            weights = np.array([0.5, 0.5])
            ensemble = AveragingEnsemble({
                'models': models,
                'clip_predictions': True
            })
            ensemble.feature_names = list(X_train.columns)
            logger.info("Using equal weights: XGB=0.5, LGBM=0.5")
        
        # Compute ensemble metrics on validation
        logger.info("Computing ensemble validation metrics...")
        ensemble_preds = ensemble.predict(X_val)
        
        # Denormalize for official metric
        avg_vol_val = meta_val['avg_vol_12m'].values
        ensemble_preds_volume = ensemble_preds * avg_vol_val
        val_actual_volume = y_val.values * avg_vol_val
        
        # Build DataFrames for official metric
        df_pred = meta_val[['country', 'brand_name', 'months_postgx']].copy()
        df_pred['volume'] = ensemble_preds_volume
        
        df_actual = meta_val[['country', 'brand_name', 'months_postgx']].copy()
        df_actual['volume'] = val_actual_volume
        
        val_with_bucket = meta_val[['country', 'brand_name', 'avg_vol_12m', 'bucket']].drop_duplicates()
        val_with_bucket = val_with_bucket.rename(columns={'avg_vol_12m': 'avg_vol'})
        
        try:
            if scenario == 1:
                ensemble_official = compute_metric1(df_actual, df_pred, val_with_bucket)
            else:
                ensemble_official = compute_metric2(df_actual, df_pred, val_with_bucket)
        except Exception as e:
            logger.warning(f"Could not compute official metric for ensemble: {e}")
            ensemble_official = np.nan
        
        ensemble_rmse = np.sqrt(np.mean((ensemble_preds - y_val.values) ** 2))
        
        ensemble_metrics = {
            'official_metric': ensemble_official,
            'rmse_norm': ensemble_rmse,
            'weights': {'xgboost': weights[0], 'lightgbm': weights[1]}
        }
        
        logger.info(f"Ensemble official_metric: {ensemble_official:.4f}")
        logger.info(f"Ensemble RMSE: {ensemble_rmse:.4f}")
        
        # Compare models
        logger.info("\n=== Model Comparison ===")
        logger.info(f"XGBoost:   official_metric={xgb_metrics.get('official_metric', np.nan):.4f}")
        logger.info(f"LightGBM:  official_metric={lgbm_metrics.get('official_metric', np.nan):.4f}")
        logger.info(f"Ensemble:  official_metric={ensemble_official:.4f}")
        
        # Log metrics to tracker (Section 13.3)
        if tracker and tracker._run_active:
            tracker.log_metrics({
                'xgb_official_metric': xgb_metrics.get('official_metric', np.nan),
                'xgb_rmse_norm': xgb_metrics.get('rmse_norm', np.nan),
                'lgbm_official_metric': lgbm_metrics.get('official_metric', np.nan),
                'lgbm_rmse_norm': lgbm_metrics.get('rmse_norm', np.nan),
                'ensemble_official_metric': ensemble_official,
                'ensemble_rmse_norm': ensemble_rmse,
                'weight_xgboost': float(weights[0]),
                'weight_lightgbm': float(weights[1]),
            })
        
        # Save ensemble
        ensemble.save(str(artifacts_dir / "ensemble.bin"))
        
        # Save results
        results = {
            'scenario': scenario,
            'ensemble_method': ensemble_method,
            'weights': {'xgboost': float(weights[0]), 'lightgbm': float(weights[1])},
            'xgb_metrics': xgb_metrics,
            'lgbm_metrics': lgbm_metrics,
            'ensemble_metrics': ensemble_metrics,
            'artifacts_dir': str(artifacts_dir)
        }
        
        with open(artifacts_dir / "ensemble_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # End experiment tracking (Section 13.4)
        if tracker:
            tracker.end_run(status='FINISHED')
        
        return {
            'ensemble': ensemble,
            'xgb_model': xgb_model,
            'lgbm_model': lgbm_model,
            'weights': weights,
            'metrics': {
                'xgboost': xgb_metrics,
                'lightgbm': lgbm_metrics,
                'ensemble': ensemble_metrics
            },
            'artifacts_dir': str(artifacts_dir)
        }
    
    except Exception as e:
        # End tracking with failure status
        if tracker:
            tracker.end_run(status='FAILED')
        raise

# =============================================================================
# SECTION: CONFIG SWEEP - Train Multiple Hyperparameter Configurations
# =============================================================================

def run_sweep_experiments(
    scenario: int,
    model_type: str = 'catboost',
    model_config_path: Optional[str] = None,
    model_config: Optional[Dict] = None,
    run_config_path: str = 'configs/run_defaults.yaml',
    data_config_path: str = 'configs/data.yaml',
    features_config_path: str = 'configs/features.yaml',
    base_run_name: Optional[str] = None,
    use_cached_features: bool = True,
    force_rebuild: bool = False,
    collect_summary: bool = True,
    quick_sweep: bool = False,
    use_named_configs: bool = True,
    use_sweep_grid: bool = False
) -> Dict[str, Any]:
    """
    Run a sweep of experiments based on named configs or sweep grid.
    
    Supports two sweep modes:
    1. Named configs: Pre-defined parameter sets in 'named_configs' list
    2. Sweep grid: Cartesian product of 'sweep_grid' parameter lists
    
    Args:
        scenario: 1 or 2
        model_type: Model type to train
        model_config_path: Path to model config file
        model_config: Model config dict (takes precedence over path)
        run_config_path: Path to run defaults config
        data_config_path: Path to data config
        features_config_path: Path to features config
        base_run_name: Base name for runs (each run appends sweep params)
        use_cached_features: If True, use cached feature loading
        force_rebuild: If True, rebuild features even if cached
        collect_summary: If True, aggregate all metrics into summary
        quick_sweep: If True, only run first 3 configs
        use_named_configs: If True, iterate over named_configs list
        use_sweep_grid: If True, use sweep_grid cartesian product
        
    Returns:
        Dictionary with:
            - 'n_runs': Number of runs executed
            - 'runs': List of (run_id, metrics) tuples
            - 'best_run': Run ID with best official_metric
            - 'best_metrics': Metrics dict for best run
            - 'sweep_metadata': List of sweep parameter combos
            - 'summary_df': DataFrame with all runs (if collect_summary=True)
    """
    scenario = _normalize_scenario(scenario)
    
    # Load configs
    run_config = load_config(run_config_path)
    data_config = load_config(data_config_path)
    features_config = load_config(features_config_path) if features_config_path else {}
    
    # Load model config from path or use provided dict
    if model_config is None:
        model_config = load_config(model_config_path) if model_config_path else {}
    
    # Generate sweep runs using the new sweep engine
    sweep_runs = generate_sweep_runs(
        model_config, 
        mode='explicit' if use_named_configs else 'grid' if use_sweep_grid else 'both'
    )
    
    # Check if we have any configs to sweep
    if not sweep_runs:
        # Fallback: Check for old-style sweep config
        sweep_config = model_config.get('sweep', {})
        sweep_enabled = sweep_config.get('enabled', False)
        
        if not sweep_enabled:
            # No sweep - run single experiment
            logger.info("Sweep not enabled in config, running single experiment")
            model, metrics = run_experiment(
                scenario=scenario,
                model_type=model_type,
                model_config_path=model_config_path,
                run_config_path=run_config_path,
                data_config_path=data_config_path,
                features_config_path=features_config_path,
                run_name=base_run_name,
                use_cached_features=use_cached_features,
                force_rebuild=force_rebuild
            )
            return {
                'n_runs': 1,
                'runs': [(base_run_name, metrics)],
                'best_run': base_run_name,
                'best_metrics': metrics,
                'sweep_metadata': [{}],
                'summary_df': pd.DataFrame([metrics]) if collect_summary else None
            }
        
        # Legacy sweep mode fallback - try old-style sweep axes
        sweep_axes = sweep_config.get('axes', [])
        if not sweep_axes:
            sweep_axes = get_sweep_axes(model_config, model_type)
        
        expanded_configs = expand_sweep(model_config, sweep_axes)
        n_runs = len(expanded_configs)
        
        if n_runs <= 1:
            logger.info("No sweep configurations found, running single experiment")
            model, metrics = run_experiment(
                scenario=scenario,
                model_type=model_type,
                model_config_path=model_config_path,
                run_config_path=run_config_path,
                data_config_path=data_config_path,
                features_config_path=features_config_path,
                run_name=base_run_name,
                use_cached_features=use_cached_features,
                force_rebuild=force_rebuild
            )
            return {
                'n_runs': 1,
                'runs': [(base_run_name, metrics)],
                'best_run': base_run_name,
                'best_metrics': metrics,
                'sweep_metadata': [{}],
                'summary_df': pd.DataFrame([metrics]) if collect_summary else None
            }
        
        # Convert legacy configs to new format
        sweep_runs = []
        for cfg in expanded_configs:
            meta = cfg.pop('_sweep_metadata', {})
            sweep_runs.append({
                'config_id': f"sweep_{len(sweep_runs)+1}",
                'params': meta.get('axes', {}),
                'full_config': cfg,
                'source': 'legacy_sweep'
            })
    
    # Apply quick sweep filter (first 3 configs)
    if quick_sweep and len(sweep_runs) > 3:
        logger.info(f"Quick sweep: limiting from {len(sweep_runs)} to 3 configs")
        sweep_runs = sweep_runs[:3]
    
    n_runs = len(sweep_runs)
    logger.info(f"Config sweep: {n_runs} configurations to evaluate")
    for i, run_info in enumerate(sweep_runs[:5], 1):  # Show first 5
        logger.info(f"  {i}. {run_info.get('config_id', 'unnamed')}: {run_info.get('params', {})}")
    if n_runs > 5:
        logger.info(f"  ... and {n_runs - 5} more configs")
    
    # Setup
    set_seed(run_config['reproducibility']['seed'])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_run_name = base_run_name or f"{timestamp}_{model_type}_scenario{scenario}"
    
    # Pre-load data once for efficiency
    logger.info("Pre-loading data for sweep...")
    if use_cached_features:
        X_full, y_full, meta_full = get_features(
            split='train',
            scenario=scenario,
            mode='train',
            data_config=data_config,
            features_config=features_config,
            use_cache=True,
            force_rebuild=force_rebuild
        )
        train_rows = pd.concat([X_full, meta_full], axis=1)
        train_rows['y_norm'] = y_full
        panel = get_panel('train', data_config, use_cache=True, force_rebuild=force_rebuild)
    else:
        train_data = load_raw_data(data_config, split='train')
        panel = prepare_base_panel(
            train_data['volume'],
            train_data['generics'],
            train_data['medicine_info']
        )
        panel = handle_missing_values(panel)
        panel = compute_pre_entry_stats(panel, is_train=True)
        panel_features = make_features(panel, scenario=scenario, mode='train', config=features_config)
        train_rows = select_training_rows(panel_features, scenario=scenario)
    
    # Create train/val split once
    train_df, val_df = create_validation_split(
        train_rows,
        val_fraction=run_config['validation']['val_fraction'],
        stratify_by=run_config['validation']['stratify_by'],
        random_state=run_config['reproducibility']['seed']
    )
    
    X_train, y_train, meta_train = split_features_target_meta(train_df)
    X_val, y_val, meta_val = split_features_target_meta(val_df)
    
    # Run sweep
    results = {
        'n_runs': n_runs,
        'runs': [],
        'best_run': None,
        'best_metrics': None,
        'sweep_metadata': [],
        'summary_df': None
    }
    
    best_official = float('inf')
    all_metrics = []
    
    for run_idx, run_info in enumerate(sweep_runs, 1):
        config_id = run_info.get('config_id', f'config_{run_idx}')
        run_params = run_info.get('params', {})
        source = run_info.get('source', 'named_config')
        
        # Build resolved config by merging base params with run-specific params
        base_params = model_config.get('params', {})
        resolved_params = apply_config_overrides(base_params, run_params)
        
        # Create full resolved config
        resolved_config = model_config.copy()
        resolved_config['params'] = resolved_params
        resolved_config['_active_config_id'] = config_id
        
        # Build sweep metadata for compatibility
        sweep_meta = {
            'config_id': config_id,
            'axes': run_params,
            'source': source
        }
        results['sweep_metadata'].append(sweep_meta)
        
        # Generate unique run ID using config_id
        run_id = f"{base_run_name}_{config_id}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Sweep run {run_idx}/{n_runs}: {run_id}")
        logger.info(f"Config ID: {config_id}")
        logger.info(f"Parameters: {run_params}")
        logger.info(f"{'='*60}")
        
        try:
            # Setup artifacts directory for this run
            artifacts_dir = get_project_root() / run_config['paths']['artifacts_dir'] / run_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save this run's config
            with open(artifacts_dir / "config_snapshot.yaml", "w") as f:
                yaml.dump({
                    'model_config': resolved_config,
                    'scenario': scenario,
                    'model_type': model_type,
                    'config_id': config_id,
                    'sweep_metadata': sweep_meta
                }, f)
            
            # Train model
            model, metrics = train_scenario_model(
                X_train, y_train, meta_train,
                X_val, y_val, meta_val,
                scenario=scenario,
                model_type=model_type,
                model_config=resolved_config,
                run_config=run_config
            )
            
            # Add sweep info to metrics
            metrics['run_id'] = run_id
            metrics['config_id'] = config_id
            metrics['sweep_params'] = sweep_meta
            
            # Save model using naming convention
            model_filename = get_model_filename(config_id, model_type, scenario)
            model_path = artifacts_dir / model_filename
            model.save(str(model_path))
            
            with open(artifacts_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Save feature importance
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                if len(importance) > 0:
                    importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'git_commit': get_git_commit_hash(),
                'scenario': scenario,
                'model_type': model_type,
                'config_id': config_id,
                'sweep_metadata': sweep_meta,
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
            }
            with open(artifacts_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            results['runs'].append((run_id, metrics))
            all_metrics.append(metrics)
            
            # Track best
            official = metrics.get('official_metric', float('inf'))
            if official < best_official:
                best_official = official
                results['best_run'] = run_id
                results['best_metrics'] = metrics
            
            logger.info(f"Run {run_idx} complete: official_metric={official:.4f}")
            
        except Exception as e:
            logger.error(f"Run {run_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            results['runs'].append((run_id, {'error': str(e), 'run_id': run_id, 'config_id': config_id, 'sweep_params': sweep_meta}))
            all_metrics.append({'error': str(e), 'run_id': run_id, 'config_id': config_id, 'official_metric': float('inf')})
    
    # Create summary DataFrame
    if collect_summary and all_metrics:
        # Flatten sweep_params into columns
        summary_rows = []
        for m in all_metrics:
            row = {k: v for k, v in m.items() if k != 'sweep_params'}
            if 'sweep_params' in m and isinstance(m['sweep_params'], dict):
                for k, v in m['sweep_params'].items():
                    # Convert dotted path to column name
                    col_name = k.replace('.', '_')
                    row[col_name] = v
            summary_rows.append(row)
        
        results['summary_df'] = pd.DataFrame(summary_rows)
        
        # Sort by official metric
        if 'official_metric' in results['summary_df'].columns:
            results['summary_df'] = results['summary_df'].sort_values('official_metric')
        
        # Save summary
        summary_dir = get_project_root() / run_config['paths']['artifacts_dir'] / f"{base_run_name}_sweep_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        results['summary_df'].to_csv(summary_dir / "sweep_results.csv", index=False)
        
        # Save as markdown table too
        with open(summary_dir / "sweep_results.md", "w") as f:
            f.write(f"# Sweep Results: {base_run_name}\n\n")
            f.write(f"- **Scenario**: {scenario}\n")
            f.write(f"- **Model**: {model_type}\n")
            f.write(f"- **Total Runs**: {n_runs}\n")
            f.write(f"- **Sweep Axes**: {sweep_axes}\n\n")
            f.write("## Results (sorted by official_metric)\n\n")
            f.write(results['summary_df'].to_markdown(index=False))
            f.write(f"\n\n## Best Run\n\n")
            f.write(f"- **Run ID**: {results['best_run']}\n")
            if results['best_metrics']:
                f.write(f"- **Official Metric**: {results['best_metrics'].get('official_metric', 'N/A'):.4f}\n")
                f.write(f"- **RMSE**: {results['best_metrics'].get('rmse_norm', 'N/A'):.4f}\n")
        
        logger.info(f"Sweep summary saved to {summary_dir}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SWEEP COMPLETE: {n_runs} runs")
    logger.info(f"Best run: {results['best_run']}")
    if results['best_metrics']:
        logger.info(f"Best official_metric: {results['best_metrics'].get('official_metric', 'N/A'):.4f}")
    logger.info(f"{'='*60}")
    
    return results


# =============================================================================
# SECTION: CONFIG SWEEP WITH K-FOLD CV
# =============================================================================

def run_sweep_with_cv(
    scenario: int,
    model_type: str = 'xgboost',
    model_config_path: Optional[str] = None,
    run_config_path: str = 'configs/run_defaults.yaml',
    data_config_path: str = 'configs/data.yaml',
    features_config_path: str = 'configs/features.yaml',
    base_run_name: Optional[str] = None,
    n_folds: int = 3,
    use_cached_features: bool = True,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Run a sweep of experiments with K-fold CV for robust hyperparameter selection.
    
    For each hyperparameter configuration in the sweep:
    1. Train on K-1 folds, validate on 1 fold
    2. Compute official_metric for each fold
    3. Average official_metric across folds (with std for confidence)
    4. Select best config based on mean official_metric
    
    This provides more robust hyperparameter selection than a single hold-out split.
    
    Args:
        scenario: 1 or 2
        model_type: Model type to train (xgboost, lightgbm, catboost)
        model_config_path: Path to model config with potential list values
        run_config_path: Path to run defaults config
        data_config_path: Path to data config
        features_config_path: Path to features config
        base_run_name: Base name for runs
        n_folds: Number of CV folds (default 3 for speed/robustness tradeoff)
        use_cached_features: If True, use cached feature loading
        force_rebuild: If True, rebuild features even if cached
        
    Returns:
        Dictionary with:
            - 'n_configs': Number of configurations evaluated
            - 'n_folds': Number of CV folds
            - 'results': List of per-config results with fold details
            - 'best_config': Best hyperparameter configuration
            - 'best_mean_metric': Mean official_metric for best config
            - 'best_std_metric': Std of official_metric for best config
            - 'summary_df': DataFrame with all results
    """
    from .validation import get_fold_series, aggregate_cv_scores
    
    scenario = _normalize_scenario(scenario)
    
    # Load configs
    run_config = load_config(run_config_path)
    data_config = load_config(data_config_path)
    features_config = load_config(features_config_path) if features_config_path else {}
    model_config = load_config(model_config_path) if model_config_path else {}
    
    # Setup
    seed = run_config['reproducibility']['seed']
    set_seed(seed)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_run_name = base_run_name or f"{timestamp}_{model_type}_scenario{scenario}_cv{n_folds}"
    
    # Check if sweep is enabled
    sweep_config = model_config.get('sweep', {})
    sweep_enabled = sweep_config.get('enabled', False)
    
    if not sweep_enabled:
        logger.warning("Sweep not enabled in config. Set sweep.enabled: true")
        return {'error': 'Sweep not enabled in config'}
    
    # Expand sweep configurations
    sweep_axes = sweep_config.get('axes', [])
    expanded_configs = expand_sweep(model_config, sweep_axes)
    n_configs = len(expanded_configs)
    
    logger.info(f"CV Sweep: {n_configs} configs x {n_folds} folds = {n_configs * n_folds} total runs")
    logger.info(f"Sweep axes: {sweep_axes}")
    
    # Load data once
    logger.info("Loading data for CV sweep...")
    if use_cached_features:
        X_full, y_full, meta_full = get_features(
            split='train',
            scenario=scenario,
            mode='train',
            data_config=data_config,
            features_config=features_config,
            use_cache=True,
            force_rebuild=force_rebuild
        )
        full_df = pd.concat([X_full, meta_full], axis=1)
        full_df['y_norm'] = y_full
        panel = get_panel('train', data_config, use_cache=True, force_rebuild=force_rebuild)
    else:
        train_data = load_raw_data(data_config, split='train')
        panel = prepare_base_panel(
            train_data['volume'],
            train_data['generics'],
            train_data['medicine_info']
        )
        panel = handle_missing_values(panel)
        panel = compute_pre_entry_stats(panel, is_train=True)
        panel_features = make_features(panel, scenario=scenario, mode='train', config=features_config)
        full_df = select_training_rows(panel_features, scenario=scenario)
    
    # Generate K-fold splits (series-level stratified)
    logger.info(f"Creating {n_folds} stratified folds at series level...")
    folds = get_fold_series(
        full_df,
        n_folds=n_folds,
        stratify_by='bucket',
        random_state=seed
    )
    
    # Log fold sizes
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        n_train_series = train_fold[['country', 'brand_name']].drop_duplicates().shape[0]
        n_val_series = val_fold[['country', 'brand_name']].drop_duplicates().shape[0]
        logger.info(f"  Fold {fold_idx+1}: {n_train_series} train series, {n_val_series} val series")
    
    # Setup artifacts directory
    artifacts_base = get_project_root() / run_config['paths']['artifacts_dir'] / f"{base_run_name}_sweep_cv"
    artifacts_base.mkdir(parents=True, exist_ok=True)
    
    # Run sweep with CV
    all_results = []
    best_mean_metric = float('inf')
    best_config = None
    best_config_idx = -1
    
    for config_idx, resolved_config in enumerate(expanded_configs, 1):
        sweep_meta = resolved_config.pop('_sweep_metadata', {})
        axes = sweep_meta.get('axes', {})
        config_suffix = "_".join(f"{k.split('.')[-1]}{v}" for k, v in sorted(axes.items()))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {config_idx}/{n_configs}: {axes}")
        logger.info(f"{'='*60}")
        
        # Train and evaluate on each fold
        fold_metrics = []
        
        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            logger.info(f"  Fold {fold_idx+1}/{n_folds}...")
            
            try:
                # Split features/target/meta
                X_train, y_train, meta_train = split_features_target_meta(train_fold)
                X_val, y_val, meta_val = split_features_target_meta(val_fold)
                
                # Train model
                model, metrics = train_scenario_model(
                    X_train, y_train, meta_train,
                    X_val, y_val, meta_val,
                    scenario=scenario,
                    model_type=model_type,
                    model_config=resolved_config,
                    run_config=run_config
                )
                
                fold_metrics.append({
                    'fold': fold_idx,
                    'official_metric': metrics.get('official_metric', np.nan),
                    'rmse_norm': metrics.get('rmse_norm', np.nan),
                    'mae_norm': metrics.get('mae_norm', np.nan)
                })
                
                logger.info(f"    Fold {fold_idx+1} official_metric: {metrics.get('official_metric', np.nan):.4f}")
                
            except Exception as e:
                logger.error(f"  Fold {fold_idx+1} failed: {e}")
                fold_metrics.append({
                    'fold': fold_idx,
                    'official_metric': np.nan,
                    'rmse_norm': np.nan,
                    'mae_norm': np.nan,
                    'error': str(e)
                })
        
        # Aggregate fold metrics
        agg = aggregate_cv_scores(fold_metrics, ['official_metric', 'rmse_norm', 'mae_norm'])
        
        mean_metric = agg.get('official_metric', {}).get('mean', np.nan)
        std_metric = agg.get('official_metric', {}).get('std', np.nan)
        
        config_result = {
            'config_idx': config_idx,
            'config_suffix': config_suffix,
            'sweep_params': axes,
            'n_folds': n_folds,
            'fold_metrics': fold_metrics,
            'mean_official_metric': mean_metric,
            'std_official_metric': std_metric,
            'ci_lower': agg.get('official_metric', {}).get('ci_lower', np.nan),
            'ci_upper': agg.get('official_metric', {}).get('ci_upper', np.nan),
            'mean_rmse': agg.get('rmse_norm', {}).get('mean', np.nan),
            'mean_mae': agg.get('mae_norm', {}).get('mean', np.nan)
        }
        all_results.append(config_result)
        
        logger.info(f"  Mean official_metric: {mean_metric:.4f} ± {std_metric:.4f}")
        
        # Track best
        if not np.isnan(mean_metric) and mean_metric < best_mean_metric:
            best_mean_metric = mean_metric
            best_config = axes
            best_config_idx = config_idx
    
    # Create summary DataFrame
    summary_rows = []
    for r in all_results:
        row = {
            'config_idx': r['config_idx'],
            'mean_official_metric': r['mean_official_metric'],
            'std_official_metric': r['std_official_metric'],
            'ci_lower': r['ci_lower'],
            'ci_upper': r['ci_upper'],
            'mean_rmse': r['mean_rmse'],
            'mean_mae': r['mean_mae']
        }
        # Add sweep params as columns
        for k, v in r['sweep_params'].items():
            col_name = k.replace('params.', '')
            row[col_name] = v
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('mean_official_metric')
    
    # Save results
    summary_df.to_csv(artifacts_base / "cv_sweep_results.csv", index=False)
    
    with open(artifacts_base / "cv_sweep_full.json", "w") as f:
        json.dump({
            'scenario': scenario,
            'model_type': model_type,
            'n_configs': n_configs,
            'n_folds': n_folds,
            'sweep_axes': sweep_axes,
            'best_config': best_config,
            'best_mean_metric': best_mean_metric,
            'results': all_results
        }, f, indent=2, default=str)
    
    # Save markdown report
    with open(artifacts_base / "cv_sweep_report.md", "w") as f:
        f.write(f"# CV Sweep Results: {base_run_name}\n\n")
        f.write(f"- **Scenario**: {scenario}\n")
        f.write(f"- **Model**: {model_type}\n")
        f.write(f"- **Configurations**: {n_configs}\n")
        f.write(f"- **CV Folds**: {n_folds}\n")
        f.write(f"- **Sweep Axes**: {sweep_axes}\n\n")
        f.write("## Results (sorted by mean official_metric)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write(f"\n\n## Best Configuration\n\n")
        f.write(f"- **Config**: {best_config}\n")
        f.write(f"- **Mean Official Metric**: {best_mean_metric:.4f}\n")
        f.write(f"- **Std Official Metric**: {all_results[best_config_idx-1]['std_official_metric']:.4f}\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CV SWEEP COMPLETE")
    logger.info(f"Best config: {best_config}")
    logger.info(f"Best mean official_metric: {best_mean_metric:.4f}")
    logger.info(f"Results saved to: {artifacts_base}")
    logger.info(f"{'='*60}")
    
    return {
        'n_configs': n_configs,
        'n_folds': n_folds,
        'results': all_results,
        'best_config': best_config,
        'best_mean_metric': best_mean_metric,
        'best_std_metric': all_results[best_config_idx-1]['std_official_metric'] if best_config_idx > 0 else np.nan,
        'summary_df': summary_df,
        'artifacts_dir': str(artifacts_base)
    }


def main():
    """CLI entry point for training models.
    
    Examples:
        # Train a single model with hold-out validation
        python -m src.train --scenario 1 --model catboost
        
        # Train with cross-validation
        python -m src.train --scenario 1 --model catboost --cv --n-folds 5
        
        # Run hyperparameter optimization
        python -m src.train --scenario 1 --model catboost --hpo --hpo-trials 50
        
        # Run config sweep (train all combinations of list values)
        python -m src.train --scenario 1 --model catboost --sweep \\
            --model-config configs/model_cat.yaml
        
        # Train both scenarios with full pipeline
        python -m src.train --full-pipeline --model catboost
        
        # Use custom configs
        python -m src.train --scenario 2 --model lightgbm \\
            --model-config configs/model_lgbm.yaml \\
            --run-config configs/run_defaults.yaml
    """
    parser = argparse.ArgumentParser(
        description="Train models for Novartis Datathon 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.train --scenario 1 --model catboost
  python -m src.train --scenario 1 --model catboost --cv --n-folds 5
  python -m src.train --scenario 1 --model catboost --hpo --hpo-trials 50
  python -m src.train --scenario 1 --model catboost --sweep --model-config configs/model_cat.yaml
  python -m src.train --full-pipeline --model catboost --parallel
  python -m src.train --scenario 2 --model lightgbm --model-config configs/model_lgbm.yaml
        """
    )
    
    # Core arguments
    parser.add_argument('--scenario', type=int, default=None,
                        choices=[1, 2],
                        help="Forecasting scenario: 1 (no actuals) or 2 (6 months actuals)")
    parser.add_argument('--model', type=str, default='catboost',
                        choices=['catboost', 'lightgbm', 'xgboost', 'linear', 
                                'nn', 'historical_curve', 'global_mean', 'flat', 'trend',
                                'baseline_global_mean', 'baseline_flat', 'hybrid', 'arihow'],
                        help="Model type to train (default: catboost)")
    
    # Config ID selection (new sweep schema)
    parser.add_argument('--config-id', type=str, default=None,
                        help="Named configuration ID from sweep_configs (e.g., 'low_lr', 'deep'). "
                             "Overrides active_config_id in model config.")
    parser.add_argument('--all-models', action='store_true',
                        help="Run for all enabled models (xgboost, lightgbm, catboost)")
    parser.add_argument('--quick-sweep', action='store_true',
                        help="Run quick sweep with fewer configs (first 3 sweep_configs only)")
    
    # Config paths
    parser.add_argument('--model-config', type=str, default=None,
                        help="Path to model config YAML (e.g., configs/model_cat.yaml)")
    parser.add_argument('--run-config', type=str, default='configs/run_defaults.yaml',
                        help="Path to run defaults YAML (default: configs/run_defaults.yaml)")
    parser.add_argument('--data-config', type=str, default='configs/data.yaml',
                        help="Path to data config YAML (default: configs/data.yaml)")
    parser.add_argument('--features-config', type=str, default='configs/features.yaml',
                        help="Path to features config YAML (default: configs/features.yaml)")
    
    # Run options
    parser.add_argument('--run-name', type=str, default=None,
                        help="Custom run name for artifacts directory")
    parser.add_argument('--cv', action='store_true',
                        help="Run cross-validation instead of single train/val split")
    parser.add_argument('--n-folds', type=int, default=5,
                        help="Number of CV folds (default: 5, only used with --cv)")
    
    # Data options
    parser.add_argument('--force-rebuild', action='store_true',
                        help="Force rebuild of cached panels and features")
    parser.add_argument('--no-cache', action='store_true',
                        help="Disable feature caching (build features from scratch)")
    
    # HPO options (Section 5.5)
    parser.add_argument('--hpo', action='store_true',
                        help="Run hyperparameter optimization using Optuna")
    parser.add_argument('--hpo-trials', type=int, default=100,
                        help="Number of HPO trials (default: 100)")
    parser.add_argument('--hpo-timeout', type=int, default=3600,
                        help="HPO timeout in seconds (default: 3600)")
    
    # Config sweep options
    parser.add_argument('--sweep', action='store_true',
                        help="Run config sweep: iterate over all sweep_configs defined in model config. "
                             "Use with --all-models to sweep all models.")
    parser.add_argument('--sweep-cv', action='store_true',
                        help="Run config sweep with K-fold CV for robust hyperparameter selection. "
                             "Uses --n-folds for number of CV folds (default 3).")
    parser.add_argument('--ensemble', action='store_true',
                        help="Train XGBoost + LightGBM ensemble with optimized weights.")
    
    # Full pipeline options (Section 5.7)
    parser.add_argument('--full-pipeline', action='store_true',
                        help="Run full training pipeline for both scenarios")
    parser.add_argument('--parallel', action='store_true',
                        help="Train scenarios in parallel (use with --full-pipeline)")
    
    # Tracking and profiling (Section 5.1, 5.7)
    parser.add_argument('--enable-tracking', action='store_true',
                        help="Enable experiment tracking (MLflow/W&B)")
    parser.add_argument('--tracking-backend', type=str, default='mlflow',
                        choices=['mlflow', 'wandb'],
                        help="Experiment tracking backend (default: mlflow)")
    parser.add_argument('--enable-checkpoints', action='store_true',
                        help="Enable checkpoint saving during training")
    parser.add_argument('--enable-profiling', action='store_true',
                        help="Enable memory profiling during training")
    
    # Weight options (Section 5.4)
    parser.add_argument('--weight-transform', type=str, default='identity',
                        choices=['identity', 'sqrt', 'log', 'softmax', 'rank'],
                        help="Transformation to apply to sample weights (default: identity)")
    parser.add_argument('--metric-aligned-weights', action='store_true',
                        help="Use exact metric-aligned sample weights")
    
    args = parser.parse_args()
    
    # Handle full pipeline mode
    if args.full_pipeline:
        results = run_full_training_pipeline(
            run_config_path=args.run_config,
            data_config_path=args.data_config,
            features_config_path=args.features_config,
            model_config_path=args.model_config,
            model_type=args.model,
            run_cv=args.cv,
            n_folds=args.n_folds,
            parallel=args.parallel,
            run_hpo=args.hpo,
            hpo_trials=args.hpo_trials,
            enable_tracking=args.enable_tracking,
            enable_checkpoints=args.enable_checkpoints,
            enable_profiling=args.enable_profiling,
            run_name=args.run_name
        )
        logger.info(f"Full pipeline complete. Results: {json.dumps(results, indent=2, default=str)[:500]}...")
        return
    
    # Require scenario for single-scenario training
    if args.scenario is None:
        parser.error("--scenario is required unless using --full-pipeline")
    
    # Handle HPO mode
    if args.hpo:
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna is required for HPO. Install with: pip install optuna")
            sys.exit(1)
        
        # Load configs and data
        run_config = load_config(args.run_config)
        data_config = load_config(args.data_config)
        model_config = load_config(args.model_config) if args.model_config else {}
        features_config = load_config(args.features_config) if args.features_config else {}
        
        set_seed(run_config['reproducibility']['seed'])
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_id = args.run_name or f"{timestamp}_{args.model}_scenario{args.scenario}_hpo"
        
        artifacts_dir = get_project_root() / run_config['paths']['artifacts_dir'] / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        setup_logging(log_file=str(artifacts_dir / "hpo.log"))
        logger.info(f"Starting HPO: {run_id}")
        
        # Load data
        from .data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values
        from .features import make_features, select_training_rows
        
        with timer("Load and prepare data"):
            train_data = load_raw_data(data_config, split='train')
            panel = prepare_base_panel(
                train_data['volume'],
                train_data['generics'],
                train_data['medicine_info']
            )
            panel = handle_missing_values(panel)
            panel = compute_pre_entry_stats(panel, is_train=True)
        
        with timer("Feature engineering"):
            panel_features = make_features(panel, scenario=args.scenario, mode='train', config=features_config)
            train_rows = select_training_rows(panel_features, scenario=args.scenario)
        
        # Create validation split
        train_df, val_df = create_validation_split(
            train_rows,
            val_fraction=run_config['validation']['val_fraction'],
            stratify_by=run_config['validation']['stratify_by'],
            random_state=run_config['reproducibility']['seed']
        )
        
        X_train, y_train, meta_train = split_features_target_meta(train_df)
        X_val, y_val, meta_val = split_features_target_meta(val_df)
        
        # Get search space from model config
        search_space = model_config.get('tuning', {}).get('search_space', {})
        
        # Run HPO
        hpo_results = run_hyperparameter_optimization(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=args.scenario,
            model_type=args.model,
            n_trials=args.hpo_trials,
            timeout=args.hpo_timeout,
            search_space=search_space,
            run_config=run_config,
            artifacts_dir=artifacts_dir
        )
        
        logger.info(f"HPO complete. Best params: {hpo_results['best_params']}")
        logger.info(f"Best metric: {hpo_results['best_value']:.4f}")
        return
    
    # Handle running specific config by ID
    if args.config_id:
        # Load config and find named config
        model_config = load_config(args.model_config) if args.model_config else {}
        named_config = get_config_by_id(model_config, args.config_id)
        
        if named_config is None:
            # List available configs
            available_ids = [c.get('id', 'unnamed') for c in model_config.get('named_configs', [])]
            logger.error(f"Config ID '{args.config_id}' not found. Available: {available_ids}")
            return
        
        # Apply config params to base params
        base_params = model_config.get('params', {})
        merged_params = apply_config_overrides(base_params, named_config.get('params', {}))
        
        # Create merged config for training
        resolved_config = model_config.copy()
        resolved_config['params'] = merged_params
        resolved_config['_active_config_id'] = args.config_id
        
        # Save resolved config for artifact tracking
        run_name = args.run_name or f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{args.model}_{args.config_id}_scenario{args.scenario}"
        
        logger.info(f"Running named config: {args.config_id}")
        logger.info(f"Description: {named_config.get('description', 'N/A')}")
        logger.info(f"Merged params: {merged_params}")
        
        model, metrics = run_experiment(
            scenario=args.scenario,
            model_type=args.model,
            model_config=resolved_config,
            run_config_path=args.run_config,
            data_config_path=args.data_config,
            features_config_path=args.features_config,
            run_name=run_name,
            use_cached_features=not args.no_cache,
            force_rebuild=args.force_rebuild
        )
        logger.info(f"Config '{args.config_id}' complete. Official metric: {metrics.get('official_metric', 'N/A'):.4f}")
        return
    
    # Handle all-models sweep
    if args.all_models:
        enabled_models = ['xgboost', 'lightgbm']  # Add catboost if needed
        results_file = get_project_root() / 'artifacts' / f"sweep_results_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
        sweep_logger = SweepResultsLogger(str(results_file))
        
        for model_type in enabled_models:
            model_config_map = {
                'xgboost': 'configs/model_xgb.yaml',
                'lightgbm': 'configs/model_lgbm.yaml',
                'catboost': 'configs/model_cat.yaml'
            }
            config_path = model_config_map.get(model_type)
            if not config_path:
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"Running sweep for model: {model_type}")
            logger.info(f"{'='*60}")
            
            results = run_sweep_experiments(
                scenario=args.scenario,
                model_type=model_type,
                model_config_path=config_path,
                run_config_path=args.run_config,
                data_config_path=args.data_config,
                features_config_path=args.features_config,
                base_run_name=args.run_name,
                use_cached_features=not args.no_cache,
                force_rebuild=args.force_rebuild,
                collect_summary=True,
                quick_sweep=args.quick_sweep
            )
            
            # Log to sweep results
            for run_id, metrics in results.get('runs', []):
                if 'error' not in metrics:
                    sweep_logger.log_result(
                        model_name=model_type,
                        config_id=metrics.get('sweep_params', {}).get('config_id', 'unknown'),
                        params=metrics.get('sweep_params', {}).get('axes', {}),
                        metrics=metrics
                    )
        
        # Print summary
        sweep_logger.summarize()
        logger.info(f"All-models sweep complete. Results saved to {results_file}")
        return
    
    # Handle sweep mode
    if args.sweep:
        results = run_sweep_experiments(
            scenario=args.scenario,
            model_type=args.model,
            model_config_path=args.model_config,
            run_config_path=args.run_config,
            data_config_path=args.data_config,
            features_config_path=args.features_config,
            base_run_name=args.run_name,
            use_cached_features=not args.no_cache,
            force_rebuild=args.force_rebuild,
            collect_summary=True,
            quick_sweep=args.quick_sweep
        )
        logger.info(f"Sweep complete. {results['n_runs']} runs executed.")
        logger.info(f"Best run: {results['best_run']}")
        if results['best_metrics']:
            logger.info(f"Best official_metric: {results['best_metrics'].get('official_metric', 'N/A'):.4f}")
        return
    
    # Handle sweep with CV mode
    if args.sweep_cv:
        results = run_sweep_with_cv(
            scenario=args.scenario,
            model_type=args.model,
            model_config_path=args.model_config,
            run_config_path=args.run_config,
            data_config_path=args.data_config,
            features_config_path=args.features_config,
            base_run_name=args.run_name,
            n_folds=args.n_folds,
            use_cached_features=not args.no_cache,
            force_rebuild=args.force_rebuild
        )
        logger.info(f"CV Sweep complete. {results['n_configs']} configs evaluated over {results['n_folds']} folds.")
        logger.info(f"Best config: {results['best_config']}")
        logger.info(f"Best mean official_metric: {results['best_mean_metric']:.4f} ± {results['best_std_metric']:.4f}")
        return
    
    # Handle ensemble mode
    if args.ensemble:
        results = train_xgb_lgbm_ensemble(
            scenario=args.scenario,
            xgb_config_path='configs/model_xgb.yaml',
            lgbm_config_path='configs/model_lgbm.yaml',
            run_config_path=args.run_config,
            data_config_path=args.data_config,
            features_config_path=args.features_config,
            ensemble_method='weighted',
            optimize_weights=True,
            use_official_metric=True,
            run_name=args.run_name,
            use_cached_features=not args.no_cache
        )
        logger.info(f"Ensemble complete. Metrics: {results['metrics']['ensemble']}")
        return
    
    if args.cv:
        # Cross-validation mode
        run_config = load_config(args.run_config)
        data_config = load_config(args.data_config)
        model_config = load_config(args.model_config) if args.model_config else {}
        
        # Setup
        set_seed(run_config['reproducibility']['seed'])
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_id = args.run_name or f"{timestamp}_{args.model}_scenario{args.scenario}_cv{args.n_folds}"
        
        artifacts_dir = get_project_root() / run_config['paths']['artifacts_dir'] / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        setup_logging(log_file=str(artifacts_dir / "train.log"))
        logger.info(f"Starting CV experiment: {run_id}")
        
        # Load and prepare data
        from .data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values
        from .features import make_features, select_training_rows
        
        with timer("Load and prepare data"):
            train_data = load_raw_data(data_config, split='train')
            panel = prepare_base_panel(
                train_data['volume'],
                train_data['generics'],
                train_data['medicine_info']
            )
            panel = handle_missing_values(panel)
            panel = compute_pre_entry_stats(panel, is_train=True)
        
        with timer("Feature engineering"):
            panel_features = make_features(panel, scenario=args.scenario, mode='train')
            train_rows = select_training_rows(panel_features, scenario=args.scenario)
        
        # Run CV
        models, cv_metrics, oof_df = run_cross_validation(
            train_rows,
            scenario=args.scenario,
            model_type=args.model,
            model_config=model_config,
            run_config=run_config,
            n_folds=args.n_folds,
            save_oof=True,
            artifacts_dir=artifacts_dir
        )
        
        logger.info(f"CV experiment {run_id} completed. Artifacts saved to {artifacts_dir}")
    else:
        # Single train/val split
        run_experiment(
            scenario=args.scenario,
            model_type=args.model,
            model_config_path=args.model_config,
            run_config_path=args.run_config,
            data_config_path=args.data_config,
            features_config_path=args.features_config,
            run_name=args.run_name,
            use_cached_features=not args.no_cache,
            force_rebuild=args.force_rebuild
        )


# ==============================================================================
# BONUS EXPERIMENTS: B2 - Bucket Specialization
# ==============================================================================

def train_bucket_specialized_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario: int,
    model_type: str = 'catboost',
    model_config: Optional[dict] = None,
    run_config: Optional[dict] = None,
    artifacts_dir: Optional[Path] = None
) -> Dict[int, Tuple[Any, Dict]]:
    """
    Train separate models for each bucket (B2: Bucket Specialization).
    
    Args:
        X_train, y_train, meta_train: Training data (must have 'bucket' column)
        X_val, y_val, meta_val: Validation data (must have 'bucket' column)
        scenario: 1 or 2
        model_type: Model type to use ('catboost', 'lightgbm', etc.)
        model_config: Model configuration
        run_config: Run configuration
        artifacts_dir: Directory to save bucket-specific models
        
    Returns:
        Dictionary mapping bucket -> (model, metrics_dict)
    """
    if 'bucket' not in meta_train.columns:
        raise ValueError("meta_train must contain 'bucket' column for bucket specialization")
    
    bucket_config = run_config.get('bucket_specialization', {}) if run_config else {}
    buckets = bucket_config.get('buckets', [1, 2])
    base_model_type = bucket_config.get('base_model_type', 'catboost')
    
    bucket_models = {}
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
    
    for bucket in buckets:
        logger.info(f"Training bucket {bucket} specialized model...")
        
        # Filter training data by bucket
        train_mask = meta_train['bucket'] == bucket
        val_mask = meta_val['bucket'] == bucket
        
        if train_mask.sum() == 0:
            logger.warning(f"No training samples for bucket {bucket}, skipping")
            continue
        
        X_train_bk = X_train[train_mask].copy()
        y_train_bk = y_train[train_mask].copy()
        meta_train_bk = meta_train[train_mask].copy()
        
        X_val_bk = X_val[val_mask].copy()
        y_val_bk = y_val[val_mask].copy()
        meta_val_bk = meta_val[val_mask].copy()
        
        logger.info(f"Bucket {bucket}: {len(X_train_bk)} train, {len(X_val_bk)} val samples")
        
        # Train model for this bucket
        model, metrics = train_scenario_model(
            X_train_bk, y_train_bk, meta_train_bk,
            X_val_bk, y_val_bk, meta_val_bk,
            scenario=scenario,
            model_type=base_model_type,
            model_config=model_config,
            run_config=run_config
        )
        
        # Save bucket-specific model
        if artifacts_dir:
            bucket_dir = artifacts_dir / f'bucket{bucket}_{base_model_type}'
            bucket_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(bucket_dir / 'model.bin'))
            
            # Save metrics
            metrics_path = bucket_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        bucket_models[bucket] = (model, metrics)
        logger.info(f"Bucket {bucket} model: Official={metrics.get('official_metric', np.nan):.4f}")
    
    return bucket_models


# ==============================================================================
# BONUS EXPERIMENTS: B3 - Post-hoc Calibration
# ==============================================================================

def fit_grouped_calibration(
    df_val: pd.DataFrame,
    config: dict,
    artifacts_dir: Optional[Path] = None
) -> Dict[Tuple, Dict[str, float]]:
    """
    Fit calibration parameters for grouped predictions (B3: Calibration).
    
    Groups predictions by (scenario, bucket, time_window) and fits linear
    calibration: volume_true = a * volume_pred + b
    
    Args:
        df_val: Validation DataFrame with columns:
            - scenario, bucket, months_postgx, volume_true, volume_pred
        config: Calibration configuration dict
        artifacts_dir: Directory to save calibration parameters
        
    Returns:
        Dictionary mapping (scenario, bucket, window_id) -> {'a': slope, 'b': intercept}
    """
    calibration_config = config.get('calibration', {})
    grouping = calibration_config.get('grouping', ['scenario', 'bucket', 'time_window'])
    method = calibration_config.get('method', 'linear')
    
    time_windows_s1 = calibration_config.get('time_windows_s1', [[0, 5], [6, 11], [12, 23]])
    time_windows_s2 = calibration_config.get('time_windows_s2', [[6, 11], [12, 17], [18, 23]])
    
    calibration_params = {}
    
    # Assign time windows
    df_val = df_val.copy()
    df_val['time_window'] = None
    
    for scenario in [1, 2]:
        windows = time_windows_s1 if scenario == 1 else time_windows_s2
        mask = df_val['scenario'] == scenario
        
        for window_id, (start, end) in enumerate(windows):
            window_mask = mask & (df_val['months_postgx'] >= start) & (df_val['months_postgx'] <= end)
            df_val.loc[window_mask, 'time_window'] = window_id
    
    # Fit calibration per group
    for (scenario, bucket, time_window), group_df in df_val.groupby(['scenario', 'bucket', 'time_window']):
        if len(group_df) < 5:  # Need minimum samples
            logger.warning(f"Insufficient samples for calibration group ({scenario}, {bucket}, {time_window})")
            continue
        
        volume_true = group_df['volume_true'].values
        volume_pred = group_df['volume_pred'].values
        
        if method == 'linear':
            # Linear regression: volume_true = a * volume_pred + b
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(volume_pred.reshape(-1, 1), volume_true)
            a = reg.coef_[0]
            b = reg.intercept_
        elif method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            reg = IsotonicRegression(out_of_bounds='clip')
            reg.fit(volume_pred, volume_true)
            # For isotonic, we'll store the model itself
            a = 1.0  # Placeholder
            b = 0.0  # Placeholder
            calibration_params[(scenario, bucket, time_window)] = {
                'method': 'isotonic',
                'model': reg,
                'a': a,
                'b': b
            }
            continue
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        calibration_params[(scenario, bucket, time_window)] = {
            'method': method,
            'a': float(a),
            'b': float(b),
            'n_samples': len(group_df)
        }
        
        logger.debug(f"Calibration ({scenario}, {bucket}, {time_window}): a={a:.4f}, b={b:.4f}, n={len(group_df)}")
    
    # Save calibration parameters
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format (skip isotonic models)
        params_json = {}
        for key, params in calibration_params.items():
            if params['method'] != 'isotonic':
                params_json[str(key)] = params
            else:
                # For isotonic, save metadata only
                params_json[str(key)] = {
                    'method': 'isotonic',
                    'n_samples': params['n_samples']
                }
        
        calib_path = artifacts_dir / 'calibration_params.json'
        with open(calib_path, 'w') as f:
            json.dump(params_json, f, indent=2)
        logger.info(f"Saved calibration parameters to {calib_path}")
    
    return calibration_params


# ==============================================================================
# BONUS EXPERIMENTS: B6 - Group-Level Bias Correction
# ==============================================================================

def fit_bias_corrections(
    df_val: pd.DataFrame,
    config: dict,
    artifacts_dir: Optional[Path] = None
) -> Dict[Tuple, float]:
    """
    Fit bias corrections per group (B6: Bias Correction).
    
    Computes mean error per group (e.g., ther_area, country) and stores
    as additive corrections.
    
    Args:
        df_val: Validation DataFrame with columns:
            - group columns (e.g., ther_area, country)
            - volume_true, volume_pred
        config: Bias correction configuration
        artifacts_dir: Directory to save bias corrections
        
    Returns:
        Dictionary mapping group tuple -> bias value
    """
    bias_config = config.get('bias_correction', {})
    group_cols = bias_config.get('group_cols', ['ther_area', 'country'])
    method = bias_config.get('method', 'mean_error')
    min_samples = bias_config.get('min_samples_per_group', 5)
    
    # Filter to only use group columns that exist in df_val
    available_group_cols = [col for col in group_cols if col in df_val.columns]
    if not available_group_cols:
        logger.warning(f"None of the requested group columns {group_cols} are available in validation data. Skipping bias correction.")
        return {}
    
    if len(available_group_cols) < len(group_cols):
        logger.warning(f"Some group columns missing: {set(group_cols) - set(available_group_cols)}. Using only: {available_group_cols}")
    
    # Compute errors
    df_val = df_val.copy()
    df_val['error'] = df_val['volume_true'] - df_val['volume_pred']
    
    bias_corrections = {}
    
    # Compute bias per group (using only available columns)
    for group_key, group_df in df_val.groupby(available_group_cols):
        if len(group_df) < min_samples:
            continue
        
        if method == 'mean_error':
            bias = group_df['error'].mean()
        elif method == 'median_error':
            bias = group_df['error'].median()
        else:
            raise ValueError(f"Unknown bias correction method: {method}")
        
        # Convert group_key to tuple if it's not already
        if isinstance(group_key, tuple):
            key = group_key
        else:
            key = (group_key,)
        
        bias_corrections[key] = float(bias)
        logger.debug(f"Bias correction {key}: {bias:.4f} (n={len(group_df)})")
    
    # Save bias corrections
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        # Convert tuple keys to strings for JSON serialization
        bias_json = {}
        for k, v in bias_corrections.items():
            # Handle both tuple and single-value keys
            if isinstance(k, tuple):
                key_str = str(k)
            else:
                key_str = str(k)
            bias_json[key_str] = float(v)
        
        bias_path = artifacts_dir / 'bias_corrections.json'
        import json as json_module
        with open(bias_path, 'w') as f:
            json_module.dump(bias_json, f, indent=2)
        logger.info(f"Saved bias corrections to {bias_path}")
    
    return bias_corrections


# ==============================================================================
# BONUS EXPERIMENTS: B8 - Multi-Seed Training
# ==============================================================================

def run_multi_seed_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario: int,
    model_type: str = 'catboost',
    model_config: Optional[dict] = None,
    run_config: Optional[dict] = None,
    artifacts_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Train model with multiple random seeds and compare results (B8: Multi-Seed).
    
    Args:
        X_train, y_train, meta_train: Training data
        X_val, y_val, meta_val: Validation data
        scenario: 1 or 2
        model_type: Model type
        model_config: Model configuration
        run_config: Run configuration
        artifacts_dir: Directory to save seed-specific models
        
    Returns:
        DataFrame with one row per seed: seed, official_metric, rmse, mae, model_path
    """
    multi_seed_config = run_config.get('multi_seed', {}) if run_config else {}
    seeds = multi_seed_config.get('seeds', [42, 2025, 1337])
    
    results = []
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
    
    for seed in seeds:
        logger.info(f"Training with seed {seed}...")
        
        # Override seed in model config
        seed_model_config = model_config.copy() if model_config else {}
        if 'params' not in seed_model_config:
            seed_model_config['params'] = {}
        seed_model_config['params']['random_seed'] = seed
        
        # Override seed in run config
        seed_run_config = run_config.copy() if run_config else {}
        if 'reproducibility' not in seed_run_config:
            seed_run_config['reproducibility'] = {}
        seed_run_config['reproducibility']['seed'] = seed
        
        # Set seed
        set_seed(seed)
        
        # Train model
        model, metrics = train_scenario_model(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=scenario,
            model_type=model_type,
            model_config=seed_model_config,
            run_config=seed_run_config
        )
        
        # Save model
        model_path = None
        if artifacts_dir:
            seed_dir = artifacts_dir / f'seed_{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            model_path = seed_dir / 'model.bin'
            model.save(str(model_path))
        
        results.append({
            'seed': seed,
            'official_metric': metrics.get('official_metric', np.nan),
            'rmse_norm': metrics.get('rmse_norm', np.nan),
            'mae_norm': metrics.get('mae_norm', np.nan),
            'model_path': str(model_path) if model_path else None
        })
        
        logger.info(f"Seed {seed}: Official={metrics.get('official_metric', np.nan):.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Save summary
    if artifacts_dir:
        summary_path = artifacts_dir / 'multi_seed_summary.csv'
        results_df.to_csv(summary_path, index=False)
        logger.info(f"Saved multi-seed summary to {summary_path}")
        
        # Log statistics
        logger.info(f"Multi-seed results:")
        logger.info(f"  Mean official metric: {results_df['official_metric'].mean():.4f}")
        logger.info(f"  Std official metric: {results_df['official_metric'].std():.4f}")
        logger.info(f"  Best seed: {results_df.loc[results_df['official_metric'].idxmin(), 'seed']}")
    
    return results_df


# ==============================================================================
# BONUS EXPERIMENTS: B5 - Residual Model Training
# ==============================================================================

def train_residual_model(
    df_residual: pd.DataFrame,
    features_config: Optional[dict] = None,
    residual_config: Optional[dict] = None,
    artifacts_dir: Optional[Path] = None
) -> Tuple[Any, Dict]:
    """
    Train residual model on high-risk segments (B5: Residual Model).
    
    Args:
        df_residual: DataFrame with columns:
            - country, brand_name, months_postgx, bucket, scenario
            - volume_true, volume_pred (from hero model)
            - All feature columns
        features_config: Feature configuration
        residual_config: Residual model configuration
        artifacts_dir: Directory to save residual model
        
    Returns:
        (trained_residual_model, metrics_dict)
    """
    residual_config = residual_config or {}
    model_type = residual_config.get('model_type', 'catboost')
    target_buckets = residual_config.get('target_buckets', [1])
    target_windows_s1 = residual_config.get('target_windows_s1', [[0, 5], [6, 11]])
    target_windows_s2 = residual_config.get('target_windows_s2', [[6, 11]])
    
    # Compute residual
    df_residual = df_residual.copy()
    df_residual['residual'] = df_residual['volume_true'] - df_residual['volume_pred']
    
    # Filter to target buckets and windows
    mask = df_residual['bucket'].isin(target_buckets)
    
    # Filter by time windows per scenario
    window_mask = pd.Series(False, index=df_residual.index)
    for scenario in [1, 2]:
        windows = target_windows_s1 if scenario == 1 else target_windows_s2
        scenario_mask = df_residual['scenario'] == scenario
        
        for start, end in windows:
            window_mask |= scenario_mask & (df_residual['months_postgx'] >= start) & (df_residual['months_postgx'] <= end)
    
    mask &= window_mask
    
    if mask.sum() == 0:
        raise ValueError("No samples match residual model criteria")
    
    df_residual_filtered = df_residual[mask].copy()
    logger.info(f"Residual model training: {len(df_residual_filtered)} samples")
    
    # Split features and target
    feature_cols = [c for c in df_residual_filtered.columns 
                    if c not in ['country', 'brand_name', 'months_postgx', 'bucket', 
                                'scenario', 'volume_true', 'volume_pred', 'residual']]
    
    X_residual = df_residual_filtered[feature_cols]
    y_residual = df_residual_filtered['residual']
    
    # Simple train/val split (80/20)
    from sklearn.model_selection import train_test_split
    X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(
        X_residual, y_residual, test_size=0.2, random_state=42
    )
    
    # Train residual model
    model = _get_model(model_type, residual_config.get('model_config', {}))
    model.fit(X_train_res, y_train_res, X_val=X_val_res, y_val=y_val_res)
    
    # Evaluate
    val_pred_residual = model.predict(X_val_res)
    rmse = np.sqrt(np.mean((val_pred_residual - y_val_res.values) ** 2))
    mae = np.mean(np.abs(val_pred_residual - y_val_res.values))
    
    metrics = {
        'rmse_residual': rmse,
        'mae_residual': mae,
        'n_train': len(X_train_res),
        'n_val': len(X_val_res)
    }
    
    # Save model
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(artifacts_dir / 'residual_model.bin'))
        
        metrics_path = artifacts_dir / 'residual_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    logger.info(f"Residual model: RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return model, metrics


if __name__ == "__main__":
    main()
