#!/usr/bin/env python
"""
Train Bayesian Stacker for ensemble predictions.

This script builds the meta-dataset from OOF predictions and trains
the Bayesian stacker with Dirichlet priors.

Usage:
    python tools/train_bayesian_stacker.py --scenario 1
    python tools/train_bayesian_stacker.py --scenario 2
    python tools/train_bayesian_stacker.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import pandas as pd

from src.stacking.bayesian_stacking import (
    BayesianStacker,
    build_meta_dataset_for_scenario,
    train_stacker_for_scenario,
    compute_sample_weights_vectorized
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_stacking_config() -> dict:
    """Load stacking configuration."""
    config_path = Path(__file__).parent.parent / 'configs' / 'stacking.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_oof_file_dict(scenario: int, config: dict) -> dict:
    """Build dictionary of model name -> OOF file path."""
    model_names = config['stacking'][f'scenario{scenario}_models']
    oof_dir = Path(config.get('oof', {}).get('output_dir', 'artifacts/oof'))
    
    oof_files = {}
    for model_name in model_names:
        oof_path = oof_dir / f'{model_name}_scenario{scenario}.parquet'
        if oof_path.exists():
            oof_files[model_name] = str(oof_path)
        else:
            logger.warning(f"OOF file not found: {oof_path}")
    
    return oof_files


def evaluate_stacker(
    stacker: BayesianStacker,
    df_meta: pd.DataFrame,
    pred_cols: list
) -> dict:
    """Evaluate stacker performance vs individual models."""
    X = df_meta[pred_cols].values
    y = df_meta['y_true'].values
    sample_weight = df_meta['sample_weight'].values
    
    # Ensemble predictions
    y_ensemble = stacker.predict(X)
    
    # Weighted MSE for ensemble
    ensemble_wmse = np.sum(sample_weight * (y - y_ensemble) ** 2) / np.sum(sample_weight)
    
    # Weighted MSE for each individual model
    model_wmses = {}
    for i, col in enumerate(pred_cols):
        y_model = X[:, i]
        wmse = np.sum(sample_weight * (y - y_model) ** 2) / np.sum(sample_weight)
        model_wmses[col.replace('pred_', '')] = wmse
    
    # RMSE versions
    ensemble_wrmse = np.sqrt(ensemble_wmse)
    model_wrmses = {k: np.sqrt(v) for k, v in model_wmses.items()}
    
    return {
        'ensemble_wrmse': ensemble_wrmse,
        'model_wrmses': model_wrmses,
        'best_single_model': min(model_wrmses, key=model_wrmses.get),
        'best_single_wrmse': min(model_wrmses.values()),
        'improvement': min(model_wrmses.values()) - ensemble_wrmse
    }


def main():
    parser = argparse.ArgumentParser(description='Train Bayesian stacker')
    parser.add_argument('--scenario', type=int, choices=[1, 2],
                       help='Scenario to train stacker for')
    parser.add_argument('--all', action='store_true',
                       help='Train stackers for all scenarios')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Dirichlet prior concentration parameter')
    parser.add_argument('--reg-strength', type=float, default=1.0,
                       help='Regularization strength')
    args = parser.parse_args()
    
    if not args.scenario and not args.all:
        parser.error('Either --scenario or --all must be specified')
    
    # Load configuration
    config = load_stacking_config()
    
    # Determine scenarios to process
    scenarios = [args.scenario] if args.scenario else [1, 2]
    
    # Output directories
    meta_dir = Path(config.get('meta_dataset', {}).get('output_dir', 'artifacts/stacking/meta'))
    stacker_dir = Path(config.get('bayesian_stacker', {}).get('output_dir', 'artifacts/stacking'))
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Bayesian Stacker for Scenario {scenario}")
        logger.info(f"{'='*60}")
        
        # Build OOF file dictionary
        oof_files = build_oof_file_dict(scenario, config)
        
        if len(oof_files) < 2:
            logger.error(f"Need at least 2 models to stack. Found: {list(oof_files.keys())}")
            continue
        
        logger.info(f"Found OOF files for {len(oof_files)} models: {list(oof_files.keys())}")
        
        # Build meta-dataset
        logger.info("Building meta-dataset...")
        meta_path = meta_dir / f'meta_scenario{scenario}.parquet'
        df_meta = build_meta_dataset_for_scenario(
            oof_files, 
            scenario,
            save_path=str(meta_path)
        )
        
        # Get prediction columns
        pred_cols = [c for c in df_meta.columns if c.startswith('pred_')]
        n_models = len(pred_cols)
        
        logger.info(f"Meta-dataset: {len(df_meta)} rows, {n_models} models")
        
        # Train stacker
        logger.info("Training Bayesian stacker...")
        alpha = np.ones(n_models) * args.alpha
        
        stacker = train_stacker_for_scenario(
            df_meta,
            scenario,
            alpha=alpha,
            regularization_strength=args.reg_strength,
            save_dir=str(stacker_dir)
        )
        
        # Evaluate
        logger.info("Evaluating stacker...")
        eval_results = evaluate_stacker(stacker, df_meta, pred_cols)
        
        logger.info(f"\n{'='*40}")
        logger.info("RESULTS")
        logger.info(f"{'='*40}")
        logger.info(f"Ensemble weighted RMSE: {eval_results['ensemble_wrmse']:.6f}")
        logger.info(f"Best single model: {eval_results['best_single_model']} "
                   f"(RMSE: {eval_results['best_single_wrmse']:.6f})")
        logger.info(f"Improvement: {eval_results['improvement']:.6f}")
        
        logger.info("\nIndividual model weighted RMSEs:")
        for model, rmse in sorted(eval_results['model_wrmses'].items(), key=lambda x: x[1]):
            logger.info(f"  {model}: {rmse:.6f}")
        
        logger.info("\nLearned weights:")
        for name, weight in stacker.get_weights_dict().items():
            logger.info(f"  {name}: {weight:.4f}")
    
    logger.info("\nBayesian stacker training complete!")


if __name__ == '__main__':
    main()
