"""
Model training pipeline for Novartis Datathon 2025.

Unified training with sample weights aligned to official metric.
Handles feature/target/meta separation to prevent leakage.
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml

from .utils import setup_logging, load_config, set_seed, timer, get_project_root
from .data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values
from .features import make_features, select_training_rows
from .validation import create_validation_split
from .evaluate import compute_metric1, compute_metric2, create_aux_file

logger = logging.getLogger(__name__)

# Define column groups to prevent leakage
# These columns are NEVER used as model features
META_COLS = ['country', 'brand_name', 'months_postgx', 'bucket', 'avg_vol_12m', 'y_norm', 'volume', 'mean_erosion', 'month']
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


def compute_sample_weights(meta_df: pd.DataFrame, scenario: str) -> pd.Series:
    """
    Compute sample weights that approximate official metric weighting.
    
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
        scenario: "scenario1" or "scenario2"
        
    Returns:
        Series of sample weights aligned with meta_df index
    """
    weights = pd.Series(1.0, index=meta_df.index)
    
    # Time-based weights
    months = meta_df['months_postgx']
    
    if scenario == 'scenario1':
        # Phase 1A: 50% months 0-5, 20% months 6-11, 10% months 12-23
        weights = np.where(months <= 5, 3.0, weights)
        weights = np.where((months >= 6) & (months <= 11), 1.5, weights)
        weights = np.where(months >= 12, 1.0, weights)
    elif scenario == 'scenario2':
        # Phase 1B: 50% months 6-11, 30% months 12-23
        weights = np.where((months >= 6) & (months <= 11), 2.5, weights)
        weights = np.where(months >= 12, 1.0, weights)
    
    weights = pd.Series(weights, index=meta_df.index)
    
    # Bucket weights
    if 'bucket' in meta_df.columns:
        bucket_weight = meta_df['bucket'].map({1: 2.0, 2: 1.0}).fillna(1.0)
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
    scenario: str,
    model_type: str = 'catboost',
    config: Optional[dict] = None
) -> Tuple[Any, Dict]:
    """
    Train model for specific scenario with early stopping.
    
    Uses sample weights from META to align with official metric.
    
    Args:
        X_train, y_train, meta_train: Training data
        X_val, y_val, meta_val: Validation data
        scenario: "scenario1" or "scenario2"
        model_type: 'catboost', 'lightgbm', 'xgboost', 'linear'
        config: Model configuration dict
        
    Returns:
        (trained_model, metrics_dict)
    """
    # Import model class
    model = _get_model(model_type, config)
    
    # Compute sample weights
    sample_weights = compute_sample_weights(meta_train, scenario)
    
    with timer(f"Train {model_type} for {scenario}"):
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            sample_weight=sample_weights
        )
    
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
        if scenario == 'scenario1':
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
    }
    
    logger.info(f"Validation metrics: Official={official_metric:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
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


def run_experiment(
    scenario: str,
    model_type: str = 'catboost',
    model_config_path: Optional[str] = None,
    run_config_path: str = 'configs/run_defaults.yaml',
    data_config_path: str = 'configs/data.yaml',
    run_name: Optional[str] = None
) -> Tuple[Any, Dict]:
    """
    Run a full experiment: load data, train model, evaluate.
    
    Args:
        scenario: "scenario1" or "scenario2"
        model_type: Model type to train
        model_config_path: Path to model config
        run_config_path: Path to run defaults config
        data_config_path: Path to data config
        run_name: Optional custom run name
        
    Returns:
        (trained_model, metrics_dict)
    """
    # Load configs
    run_config = load_config(run_config_path)
    data_config = load_config(data_config_path)
    model_config = load_config(model_config_path) if model_config_path else {}
    
    # Setup
    set_seed(run_config['reproducibility']['seed'])
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = run_name or f"{timestamp}_{model_type}_{scenario}"
    
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
            'model_config': model_config,
            'scenario': scenario,
            'model_type': model_type
        }, f)
    
    # Load and prepare data
    with timer("Load and prepare data"):
        train_data = load_raw_data(data_config, split='train')
        
        panel = prepare_base_panel(
            train_data['volume'],
            train_data['generics'],
            train_data['medicine_info']
        )
        panel = handle_missing_values(panel)
        panel = compute_pre_entry_stats(panel, is_train=True)
    
    # Build features
    with timer("Feature engineering"):
        panel_features = make_features(panel, scenario=scenario, mode='train')
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
    
    # Train model
    model, metrics = train_scenario_model(
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        scenario=scenario,
        model_type=model_type,
        config=model_config
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
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train models for Novartis Datathon 2025")
    parser.add_argument('--scenario', type=str, required=True,
                        choices=['scenario1', 'scenario2'],
                        help="Forecasting scenario")
    parser.add_argument('--model', type=str, default='catboost',
                        choices=['catboost', 'lightgbm', 'xgboost', 'linear', 
                                'baseline_global_mean', 'baseline_flat'],
                        help="Model type to train")
    parser.add_argument('--model-config', type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument('--run-config', type=str, default='configs/run_defaults.yaml',
                        help="Path to run defaults YAML")
    parser.add_argument('--data-config', type=str, default='configs/data.yaml',
                        help="Path to data config YAML")
    parser.add_argument('--run-name', type=str, default=None,
                        help="Custom run name")
    
    args = parser.parse_args()
    
    run_experiment(
        scenario=args.scenario,
        model_type=args.model,
        model_config_path=args.model_config,
        run_config_path=args.run_config,
        data_config_path=args.data_config,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()
