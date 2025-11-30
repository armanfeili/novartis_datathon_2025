#!/usr/bin/env python
"""
Generate Out-of-Fold predictions for Bayesian stacking.

This script trains each base model with GroupKFold CV and saves OOF predictions.
Run this once before training the Bayesian stacker.

Usage:
    python tools/run_oof_for_stacking.py --scenario 1
    python tools/run_oof_for_stacking.py --scenario 2
    python tools/run_oof_for_stacking.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

from src.data import load_panel_data, get_feature_columns
from src.stacking.bayesian_stacking import (
    compute_sample_weights_vectorized,
    generate_oof_predictions
)
from src.models import CatBoostModel, LGBMModel, XGBModel
from src.models.hybrid_physics_ml import HybridPhysicsMLModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model name to class mapping
MODEL_CLASSES = {
    'catboost': CatBoostModel,
    'lgbm': LGBMModel,
    'xgb': XGBModel,
    # 'hybrid': HybridPhysicsMLModel,  # Requires special handling
}


def load_stacking_config() -> dict:
    """Load stacking configuration."""
    config_path = Path(__file__).parent.parent / 'configs' / 'stacking.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_config(model_name: str) -> dict:
    """Load model configuration based on model name."""
    # Parse model name (e.g., 'catboost_s1' -> 'catboost', scenario 1)
    parts = model_name.rsplit('_', 1)
    model_type = parts[0] if len(parts) > 1 else model_name
    
    # Map to config file
    config_map = {
        'catboost': 'model_cat.yaml',
        'lgbm': 'model_lgbm.yaml',
        'xgb': 'model_xgb.yaml',
        'hybrid': 'model_hybrid.yaml',
    }
    
    config_file = config_map.get(model_type, f'model_{model_type}.yaml')
    config_path = Path(__file__).parent.parent / 'configs' / config_file
    
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    return {}


def get_model_class(model_name: str):
    """Get model class based on model name."""
    for key, cls in MODEL_CLASSES.items():
        if key in model_name.lower():
            return cls
    
    raise ValueError(f"Unknown model type in name: {model_name}")


def generate_oof_for_model(
    model_name: str,
    scenario: int,
    df_train: pd.DataFrame,
    feature_cols: list,
    config: dict,
    output_dir: Path
) -> pd.DataFrame:
    """Generate OOF predictions for a single model."""
    logger.info(f"Generating OOF for {model_name} (scenario {scenario})")
    
    # Get model class and config
    model_class = get_model_class(model_name)
    model_config = load_model_config(model_name)
    
    # Filter by scenario
    if scenario == 1:
        df = df_train[df_train['months_postgx'].between(0, 23)].copy()
    else:
        df = df_train[df_train['months_postgx'].between(6, 23)].copy()
    
    # Create group identifiers for GroupKFold
    df['_group'] = df['country'] + '_' + df['brand_name']
    groups = df.groupby('_group').ngroup()
    
    # OOF settings
    n_folds = config.get('oof', {}).get('n_folds', 5)
    
    # Initialize OOF predictions
    oof_preds = np.full(len(df), np.nan)
    
    # GroupKFold CV
    gkf = GroupKFold(n_splits=n_folds)
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        logger.info(f"  Fold {fold_idx + 1}/{n_folds}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        X_train = train_df[feature_cols]
        y_train = train_df['y_norm']
        X_val = val_df[feature_cols]
        y_val = val_df['y_norm']
        
        # Compute sample weights
        train_weights = compute_sample_weights_vectorized(
            train_df['months_postgx'].values,
            train_df['bucket'].values,
            scenario
        )
        
        # Train model
        model = model_class(model_config)
        model.fit(
            X_train, y_train, 
            X_val, y_val,
            sample_weight=pd.Series(train_weights, index=train_df.index)
        )
        
        # Predict on validation fold
        oof_preds[val_idx] = model.predict(X_val)
    
    # Build result DataFrame
    result = df[['country', 'brand_name', 'months_postgx', 'bucket']].copy()
    result['y_true'] = df['y_norm'].values
    result['y_pred'] = oof_preds
    
    # Save
    output_path = output_dir / f'{model_name}_scenario{scenario}.parquet'
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info(f"  Saved to {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate OOF predictions for stacking')
    parser.add_argument('--scenario', type=int, choices=[1, 2], 
                       help='Scenario to generate OOF for')
    parser.add_argument('--all', action='store_true',
                       help='Generate OOF for all scenarios')
    parser.add_argument('--models', nargs='+',
                       help='Specific models to run (default: all from config)')
    args = parser.parse_args()
    
    if not args.scenario and not args.all:
        parser.error('Either --scenario or --all must be specified')
    
    # Load configuration
    config = load_stacking_config()
    
    # Load training data
    logger.info("Loading training data...")
    df_train = load_panel_data(train=True)
    feature_cols = get_feature_columns(df_train)
    
    logger.info(f"Training data: {len(df_train)} rows, {len(feature_cols)} features")
    
    # Determine scenarios to process
    scenarios = [args.scenario] if args.scenario else [1, 2]
    
    # Output directory
    output_dir = Path(config.get('oof', {}).get('output_dir', 'artifacts/oof'))
    
    # Process each scenario
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Scenario {scenario}")
        logger.info(f"{'='*60}")
        
        # Get models for this scenario
        model_names = config['stacking'][f'scenario{scenario}_models']
        
        if args.models:
            model_names = [m for m in model_names if m in args.models]
        
        for model_name in model_names:
            try:
                generate_oof_for_model(
                    model_name, scenario, df_train, feature_cols, config, output_dir
                )
            except Exception as e:
                logger.error(f"Failed to generate OOF for {model_name}: {e}")
                continue
    
    logger.info("\nOOF generation complete!")


if __name__ == '__main__':
    main()
