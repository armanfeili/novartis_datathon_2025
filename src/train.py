"""
Model training pipeline for Novartis Datathon 2025.

Unified training with sample weights aligned to official metric.
Handles feature/target/meta separation to prevent leakage.
"""

import argparse
import logging
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yaml

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

logger = logging.getLogger(__name__)

# Re-export META_COLS from data.py for backward compatibility
# These columns are NEVER used as model features
# Canonical definition is in src/data.py
TARGET_COL = 'y_norm'


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
    
    CRITICAL: feature_cols must match training exactly!
    
    Args:
        df: DataFrame with features and meta columns
        
    Returns:
        X: Pure features for model
        meta: Meta columns including avg_vol_12m for inverse transform
    """
    # Identify feature columns (everything except meta)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
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
    config: Optional[dict] = None
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
        
    Returns:
        Series of sample weights aligned with meta_df index
    """
    scenario = _normalize_scenario(scenario)
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
    """Get model instance by type."""
    if model_type == 'catboost':
        from .models.cat_model import CatBoostModel
        return CatBoostModel(config or {})
    elif model_type == 'lightgbm':
        from .models.lgbm_model import LGBMModel
        return LGBMModel(config or {})
    elif model_type == 'xgboost':
        from .models.xgb_model import XGBModel
        return XGBModel(config or {})
    elif model_type == 'linear':
        from .models.linear import LinearModel
        return LinearModel(config or {})
    elif model_type == 'baseline_global_mean':
        from .models.linear import GlobalMeanBaseline
        return GlobalMeanBaseline(config or {})
    elif model_type == 'baseline_flat':
        from .models.linear import FlatBaseline
        return FlatBaseline(config or {})
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
    
    # Save config snapshot
    with open(artifacts_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump({
            'run_config': run_config,
            'data_config': data_config,
            'features_config': features_config,
            'model_config': model_config,
            'scenario': scenario,
            'model_type': model_type
        }, f)
    
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


def main():
    """CLI entry point for training models.
    
    Examples:
        # Train a single model with hold-out validation
        python -m src.train --scenario 1 --model catboost
        
        # Train with cross-validation
        python -m src.train --scenario 1 --model catboost --cv --n-folds 5
        
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
  python -m src.train --scenario 2 --model lightgbm --model-config configs/model_lgbm.yaml
        """
    )
    parser.add_argument('--scenario', type=int, required=True,
                        choices=[1, 2],
                        help="Forecasting scenario: 1 (no actuals) or 2 (6 months actuals)")
    parser.add_argument('--model', type=str, default='catboost',
                        choices=['catboost', 'lightgbm', 'xgboost', 'linear', 
                                'baseline_global_mean', 'baseline_flat'],
                        help="Model type to train (default: catboost)")
    parser.add_argument('--model-config', type=str, default=None,
                        help="Path to model config YAML (e.g., configs/model_cat.yaml)")
    parser.add_argument('--run-config', type=str, default='configs/run_defaults.yaml',
                        help="Path to run defaults YAML (default: configs/run_defaults.yaml)")
    parser.add_argument('--data-config', type=str, default='configs/data.yaml',
                        help="Path to data config YAML (default: configs/data.yaml)")
    parser.add_argument('--features-config', type=str, default='configs/features.yaml',
                        help="Path to features config YAML (default: configs/features.yaml)")
    parser.add_argument('--run-name', type=str, default=None,
                        help="Custom run name for artifacts directory")
    parser.add_argument('--cv', action='store_true',
                        help="Run cross-validation instead of single train/val split")
    parser.add_argument('--n-folds', type=int, default=5,
                        help="Number of CV folds (default: 5, only used with --cv)")
    parser.add_argument('--force-rebuild', action='store_true',
                        help="Force rebuild of cached panels and features")
    parser.add_argument('--no-cache', action='store_true',
                        help="Disable feature caching (build features from scratch)")
    
    args = parser.parse_args()
    
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
