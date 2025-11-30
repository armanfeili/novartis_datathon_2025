#!/usr/bin/env python
"""
Run Hierarchical Bayesian Decay Model as Extra Base Model.

This script:
1. Fits the hierarchical decay model with Empirical Bayes priors
2. Generates OOF predictions
3. Adds predictions to the meta-dataset
4. Re-trains the stacker with the enhanced meta-dataset

Usage:
    python tools/run_hierarchical_decay.py --scenario 1 2
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stacking import (
    HierarchicalBayesianDecay,
    generate_oof_for_hierarchical_decay,
    add_hierarchical_decay_to_meta_dataset,
    train_stacker_for_scenario,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load stacking configuration."""
    config_path = Path(__file__).parent.parent / 'configs' / 'stacking.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_training_data(scenario: int) -> pd.DataFrame:
    """Load training data for the specified scenario."""
    # Try multiple possible locations
    possible_paths = [
        Path('data/processed/train_featured.parquet'),
        Path('data/processed/train.parquet'),
        Path('artifacts/data/train_featured.parquet'),
    ]
    
    for path in possible_paths:
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"Loaded training data from {path}: {len(df)} rows")
            
            # Filter by scenario
            if scenario == 1:
                df = df[df['months_postgx'].between(0, 23)]
            else:
                df = df[df['months_postgx'].between(6, 23)]
            
            return df
    
    raise FileNotFoundError("Training data not found. Please run feature engineering first.")


def run_hierarchical_decay_for_scenario(scenario: int, config: dict):
    """Run hierarchical decay model for a scenario."""
    logger.info(f"=" * 60)
    logger.info(f"Running Hierarchical Bayesian Decay for Scenario {scenario}")
    logger.info(f"=" * 60)
    
    decay_config = config.get('hierarchical_decay', {})
    prior_config = decay_config.get('prior', {})
    
    # Load training data
    df_train = load_training_data(scenario)
    
    # Determine target column
    target_col = 'y_norm' if 'y_norm' in df_train.columns else 'volume_norm'
    
    # Create decay model
    decay_kwargs = {
        'prior_a_mean': prior_config.get('a_mean', 1.0),
        'prior_a_std': prior_config.get('a_std', 0.5),
        'prior_b_mean': prior_config.get('b_mean', 0.05),
        'prior_b_std': prior_config.get('b_std', 0.02),
        'prior_c_mean': prior_config.get('c_mean', 0.3),
        'prior_c_std': prior_config.get('c_std', 0.2),
        'use_hierarchical_priors': decay_config.get('use_hierarchical_priors', True),
        'shrinkage_strength': decay_config.get('shrinkage_strength', 0.3),
    }
    
    # Generate OOF predictions
    oof_dir = Path(config.get('oof', {}).get('output_dir', 'artifacts/oof'))
    oof_path = oof_dir / f'bayes_decay_scenario{scenario}.parquet'
    
    n_folds = config.get('oof', {}).get('n_folds', 5)
    
    df_oof = generate_oof_for_hierarchical_decay(
        df_train=df_train,
        scenario=scenario,
        target_col=target_col,
        n_folds=n_folds,
        save_path=str(oof_path),
        **decay_kwargs
    )
    
    logger.info(f"Generated OOF predictions: {len(df_oof)} rows")
    logger.info(f"OOF saved to {oof_path}")
    
    # Also fit full model and save
    model = HierarchicalBayesianDecay(**decay_kwargs)
    model.fit(df_train, target_col=target_col)
    
    model_dir = Path(config.get('bayesian_stacker', {}).get('output_dir', 'artifacts/stacking'))
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / f'hierarchical_decay_scenario{scenario}.joblib'))
    
    logger.info(f"Saved full decay model to {model_dir}")
    
    # Load existing meta-dataset and enhance
    meta_dir = Path(config.get('meta_dataset', {}).get('output_dir', 'artifacts/stacking/meta'))
    meta_path = meta_dir / f'meta_scenario{scenario}.parquet'
    
    if meta_path.exists():
        df_meta = pd.read_parquet(meta_path)
        
        # Add decay predictions
        df_meta_enhanced = add_hierarchical_decay_to_meta_dataset(
            df_meta=df_meta,
            df_train=df_train,
            scenario=scenario,
            target_col=target_col,
            **decay_kwargs
        )
        
        # Save enhanced meta-dataset
        enhanced_path = meta_dir / f'meta_scenario{scenario}_with_decay.parquet'
        df_meta_enhanced.to_parquet(enhanced_path, index=False)
        logger.info(f"Enhanced meta-dataset saved to {enhanced_path}")
        
        # Re-train stacker with enhanced meta-dataset
        stacker_dir = Path(config.get('bayesian_stacker', {}).get('output_dir', 'artifacts/stacking'))
        
        stacker = train_stacker_for_scenario(
            df_meta=df_meta_enhanced,
            scenario=scenario,
            alpha=None,  # Uniform prior
            regularization_strength=config.get('bayesian_stacker', {}).get('regularization_strength', 1.0),
            save_dir=str(stacker_dir)
        )
        
        logger.info(f"Re-trained stacker with hierarchical decay model")
        logger.info(f"Final weights: {stacker.get_weights_dict()}")
        
    else:
        logger.warning(f"Meta-dataset not found at {meta_path}. "
                      f"Run train_bayesian_stacker.py first, then re-run this script.")
    
    return df_oof


def main():
    parser = argparse.ArgumentParser(description='Run Hierarchical Bayesian Decay Model')
    parser.add_argument(
        '--scenario', '-s',
        type=int,
        nargs='+',
        default=[1, 2],
        choices=[1, 2],
        help='Scenario(s) to run (default: both)'
    )
    args = parser.parse_args()
    
    config = load_config()
    
    if not config.get('hierarchical_decay', {}).get('enabled', False):
        logger.warning("Hierarchical decay is disabled in config. Enable it first.")
        logger.info("Set 'hierarchical_decay.enabled: true' in configs/stacking.yaml")
        return
    
    for scenario in args.scenario:
        run_hierarchical_decay_for_scenario(scenario, config)
    
    logger.info("=" * 60)
    logger.info("Hierarchical Bayesian Decay completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
