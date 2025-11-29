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
from .features import make_features, select_training_rows, _normalize_scenario, get_features
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
    run_name: str
) -> Optional[ExperimentTracker]:
    """
    Setup experiment tracking from run_config.
    
    Args:
        run_config: Run configuration dictionary
        run_name: Name for the run
        
    Returns:
        ExperimentTracker instance or None
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
    run_config: Optional[Dict] = None
) -> Callable:
    """
    Create an Optuna objective function for hyperparameter optimization.
    
    Args:
        X_train, y_train, meta_train: Training data
        X_val, y_val, meta_val: Validation data
        scenario: Scenario number
        model_type: Type of model to tune
        search_space: Custom search space dict
        run_config: Run configuration
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial: 'optuna.Trial') -> float:
        """Optuna objective function."""
        # Define hyperparameters based on model type
        if model_type == 'catboost':
            params = {
                'depth': trial.suggest_int('depth', 
                    search_space.get('depth', [4, 8])[0],
                    search_space.get('depth', [4, 8])[1]
                ),
                'learning_rate': trial.suggest_float('learning_rate',
                    search_space.get('learning_rate', [0.01, 0.1])[0],
                    search_space.get('learning_rate', [0.01, 0.1])[1],
                    log=True
                ),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg',
                    search_space.get('l2_leaf_reg', [1.0, 10.0])[0],
                    search_space.get('l2_leaf_reg', [1.0, 10.0])[1],
                    log=True
                ),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                    search_space.get('min_data_in_leaf', [10, 50])[0],
                    search_space.get('min_data_in_leaf', [10, 50])[1]
                ),
                'random_strength': trial.suggest_float('random_strength',
                    search_space.get('random_strength', [0.0, 5.0])[0],
                    search_space.get('random_strength', [0.0, 5.0])[1]
                ),
                'bagging_temperature': trial.suggest_float('bagging_temperature',
                    search_space.get('bagging_temperature', [0.0, 5.0])[0],
                    search_space.get('bagging_temperature', [0.0, 5.0])[1]
                ),
            }
            
        elif model_type == 'lightgbm':
            params = {
                'num_leaves': trial.suggest_int('num_leaves',
                    search_space.get('num_leaves', [15, 63])[0],
                    search_space.get('num_leaves', [15, 63])[1]
                ),
                'learning_rate': trial.suggest_float('learning_rate',
                    search_space.get('learning_rate', [0.01, 0.1])[0],
                    search_space.get('learning_rate', [0.01, 0.1])[1],
                    log=True
                ),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',
                    search_space.get('min_data_in_leaf', [10, 50])[0],
                    search_space.get('min_data_in_leaf', [10, 50])[1]
                ),
                'feature_fraction': trial.suggest_float('feature_fraction',
                    search_space.get('feature_fraction', [0.6, 1.0])[0],
                    search_space.get('feature_fraction', [0.6, 1.0])[1]
                ),
                'bagging_fraction': trial.suggest_float('bagging_fraction',
                    search_space.get('bagging_fraction', [0.6, 1.0])[0],
                    search_space.get('bagging_fraction', [0.6, 1.0])[1]
                ),
            }
            
        elif model_type == 'xgboost':
            params = {
                'max_depth': trial.suggest_int('max_depth',
                    search_space.get('max_depth', [4, 8])[0],
                    search_space.get('max_depth', [4, 8])[1]
                ),
                'learning_rate': trial.suggest_float('learning_rate',
                    search_space.get('learning_rate', [0.01, 0.1])[0],
                    search_space.get('learning_rate', [0.01, 0.1])[1],
                    log=True
                ),
                'min_child_weight': trial.suggest_int('min_child_weight',
                    search_space.get('min_child_weight', [1, 10])[0],
                    search_space.get('min_child_weight', [1, 10])[1]
                ),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                    search_space.get('colsample_bytree', [0.6, 1.0])[0],
                    search_space.get('colsample_bytree', [0.6, 1.0])[1]
                ),
                'subsample': trial.suggest_float('subsample',
                    search_space.get('subsample', [0.6, 1.0])[0],
                    search_space.get('subsample', [0.6, 1.0])[1]
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
                run_config=run_config
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
    
    Args:
        X_train, y_train, meta_train: Training data
        X_val, y_val, meta_val: Validation data  
        scenario: Scenario number
        model_type: Type of model to tune
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds (None for no limit)
        search_space: Custom search space dict (uses defaults if None)
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
    
    # Default search spaces
    default_search_spaces = {
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
    
    search_space = search_space or default_search_spaces.get(model_type, {})
    
    # Create study
    study_name = study_name or f"{model_type}_scenario{scenario}_{datetime.now():%Y%m%d_%H%M%S}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    )
    
    # Create objective
    objective = create_optuna_objective(
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        scenario=scenario,
        model_type=model_type,
        search_space=search_space,
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
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Separate pure features from target and meta columns.
    
    This GUARANTEES bucket/y_norm never leak into model features.
    
    Args:
        df: DataFrame with features, target, and meta columns
        
    Returns:
        X: Pure features for model (excludes all META_COLS)
        y: Target (y_norm)
        meta: Meta columns for weights, grouping, metrics
    """
    # Identify feature columns (everything except meta)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
    # Split
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    
    # Meta columns that exist in the dataframe
    meta_cols_present = [c for c in META_COLS if c in df.columns]
    meta = df[meta_cols_present].copy()
    
    # Log
    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Meta: {len(meta_cols_present)} columns")
    
    # Validate no leakage
    leaked = set(X.columns) & set(META_COLS)
    if leaked:
        raise ValueError(f"LEAKAGE DETECTED! Meta columns in features: {leaked}")
    
    return X, y, meta


def get_feature_matrix_and_meta(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For INFERENCE: separate features from meta (no target).
    
    CRITICAL: Uses META_COLS same as training's split_features_target_meta
    to ensure feature_cols match the model exactly!
    
    Args:
        df: DataFrame with features and meta columns
        
    Returns:
        X: Pure features for model
        meta: Meta columns including avg_vol_12m for inverse transform
    """
    # Use META_COLS to exclude same columns as training
    # This ensures feature columns match the trained model exactly
    feature_cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
        elif df[col].dtype.name == 'category':
            feature_cols.append(col)
    
    X = df[feature_cols].copy()
    
    # Meta columns that exist
    meta_cols_present = [c for c in META_COLS if c in df.columns]
    meta = df[meta_cols_present].copy()
    
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
    """
    scenario = _normalize_scenario(scenario)
    # Import model class
    model = _get_model(model_type, model_config)
    
    # Compute sample weights (using run_config if available)
    sample_weights = compute_sample_weights(meta_train, scenario, config=run_config)
    
    # Track training time
    train_start = time.time()
    
    with timer(f"Train {model_type} for scenario {scenario}"):
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            sample_weight=sample_weights
        )
    
    train_time = time.time() - train_start
    
    # Compute validation metrics
    val_preds_norm = model.predict(X_val)
    
    # Denormalize predictions for metric calculation
    avg_vol_val = meta_val['avg_vol_12m'].values
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
    
    # Compute additional metrics
    rmse = np.sqrt(np.mean((val_preds_norm - y_val.values) ** 2))
    mae = np.mean(np.abs(val_preds_norm - y_val.values))
    
    metrics = {
        'official_metric': official_metric,
        'rmse_norm': rmse,
        'mae_norm': mae,
        'scenario': scenario,
        'model_type': model_type,
        'train_time_seconds': train_time,
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_features': len(X_train.columns),
    }
    
    # Save unified metric records if metrics_dir is provided
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
    
    # Hybrid Physics + ML model
    elif model_type_lower in ('hybrid_lgbm', 'hybrid_lightgbm'):
        from .models.hybrid_physics_ml import HybridPhysicsMLModel
        decay_rate = config.get('physics', {}).get('decay_rate', 0.05)
        ml_params = config.get('ml_model', {}).get('lightgbm', {})
        return HybridPhysicsMLModel(
            ml_model_type='lightgbm',
            decay_rate=decay_rate,
            params=ml_params if ml_params else None
        )
    elif model_type_lower in ('hybrid_xgb', 'hybrid_xgboost'):
        from .models.hybrid_physics_ml import HybridPhysicsMLModel
        decay_rate = config.get('physics', {}).get('decay_rate', 0.05)
        ml_params = config.get('ml_model', {}).get('xgboost', {})
        return HybridPhysicsMLModel(
            ml_model_type='xgboost',
            decay_rate=decay_rate,
            params=ml_params if ml_params else None
        )
    elif model_type_lower == 'hybrid':
        from .models.hybrid_physics_ml import HybridPhysicsMLModel
        decay_rate = config.get('physics', {}).get('decay_rate', 0.05)
        ml_type = config.get('ml_model', {}).get('type', 'lightgbm')
        ml_params = config.get('ml_model', {}).get(ml_type, {})
        return HybridPhysicsMLModel(
            ml_model_type=ml_type,
            decay_rate=decay_rate,
            params=ml_params if ml_params else None
        )
    
    # ARIMA + Holt-Winters hybrid model
    elif model_type_lower in ('arihow', 'arima_hw', 'arima_holtwinters'):
        from .models.arihow import ARIHOWModel
        arima_params = config.get('arima', {})
        hw_params = config.get('holt_winters', {})
        
        # Map config to model parameters
        arima_order = tuple(arima_params.get('order', [1, 1, 1]))
        seasonal_order = tuple(arima_params.get('seasonal_order', [0, 0, 0, 0])) if arima_params.get('seasonal', False) else (0, 0, 0, 0)
        
        return ARIHOWModel(
            arima_order=arima_order,
            seasonal_order=seasonal_order,
            hw_trend=hw_params.get('trend', 'add'),
            hw_seasonal=hw_params.get('seasonal', None),
            hw_seasonal_periods=hw_params.get('seasonal_periods', 12),
            weight_window=config.get('weight_window', 12),
            suppress_warnings=config.get('suppress_warnings', True)
        )
    
    else:
        available = [
            'catboost', 'lightgbm', 'xgboost', 'linear', 'nn',
            'global_mean', 'flat', 'trend', 'historical_curve',
            'averaging', 'weighted', 'stacking', 'blending',
            'hybrid', 'hybrid_lgbm', 'hybrid_xgb', 'arihow'
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
    metrics_dir: Optional[Path] = None
) -> Tuple[List[Any], Dict, pd.DataFrame]:
    """
    Run K-fold cross-validation at series level.
    
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
        with timer("Load features (cached)"):
            X_full, y_full, meta_full = get_features(
                split='train',
                scenario=scenario,
                mode='train',
                data_config=data_config,
                features_config=features_config,
                use_cache=True,
                force_rebuild=force_rebuild
            )
            # Combine for validation split
            train_rows = pd.concat([X_full, meta_full], axis=1)
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
    
    # Train model
    model, metrics = train_scenario_model(
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        scenario=scenario,
        model_type=model_type,
        model_config=model_config,
        run_config=run_config
    )
    
    # Save model
    model_path = artifacts_dir / f"model_{scenario}.bin"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
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
    sample_weight_val: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compare multiple model types on the same train/validation split.
    
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
    use_cached_features: bool = True
) -> Dict[str, Any]:
    """
    Train an XGBoost + LightGBM ensemble with optional weight optimization.
    
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
    run_name = run_name or f"{timestamp}_xgb_lgbm_ensemble_scenario{scenario}"
    
    artifacts_dir = get_project_root() / run_config['paths']['artifacts_dir'] / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_file=str(artifacts_dir / "ensemble.log"))
    logger.info(f"Training XGB+LGBM ensemble: {run_name}")
    
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


if __name__ == "__main__":
    main()
