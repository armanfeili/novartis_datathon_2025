#!/usr/bin/env python
"""
Evaluate Models with Official Metric.

This script evaluates all base models and the ensemble using the
official competition metric (compute_metric1 / compute_metric2).

Usage:
    python tools/evaluate_stacking.py --scenario 1 2
    python tools/evaluate_stacking.py --scenario 1 --include-decay
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
    BayesianStacker,
    evaluate_oof_predictions,
    compare_models_on_oof,
    evaluate_ensemble_vs_single_models,
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


def load_aux_data() -> pd.DataFrame:
    """Load auxiliary data for metric computation."""
    possible_paths = [
        Path('data/processed/aux.parquet'),
        Path('data/processed/df_aux.parquet'),
        Path('artifacts/data/aux.parquet'),
        Path('data/raw/train_aux.csv'),
    ]
    
    for path in possible_paths:
        if path.exists():
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            logger.info(f"Loaded aux data from {path}: {len(df)} series")
            return df
    
    # Try to create from training data
    train_paths = [
        Path('data/processed/train_featured.parquet'),
        Path('data/processed/train.parquet'),
    ]
    
    for path in train_paths:
        if path.exists():
            df = pd.read_parquet(path)
            if 'avg_vol_12m' in df.columns and 'bucket' in df.columns:
                aux = df[['country', 'brand_name', 'avg_vol_12m', 'bucket']].drop_duplicates()
                aux = aux.rename(columns={'avg_vol_12m': 'avg_vol'})
                logger.info(f"Created aux data from {path}: {len(aux)} series")
                return aux
    
    raise FileNotFoundError("Auxiliary data not found")


def get_oof_files(scenario: int, config: dict, include_decay: bool = True) -> dict:
    """Get OOF file paths for all models."""
    model_list = config['stacking'].get(f'scenario{scenario}_models', [])
    oof_dir = Path(config.get('oof', {}).get('output_dir', 'artifacts/oof'))
    
    oof_files = {}
    for model_name in model_list:
        oof_path = oof_dir / f'{model_name}_scenario{scenario}.parquet'
        if oof_path.exists():
            oof_files[model_name] = str(oof_path)
        else:
            logger.warning(f"OOF not found: {oof_path}")
    
    # Add hierarchical decay if enabled
    if include_decay and config.get('hierarchical_decay', {}).get('enabled', False):
        decay_path = oof_dir / f'bayes_decay_scenario{scenario}.parquet'
        if decay_path.exists():
            oof_files['bayes_decay'] = str(decay_path)
    
    return oof_files


def evaluate_for_scenario(scenario: int, config: dict, include_decay: bool = True):
    """Evaluate all models for a scenario."""
    logger.info(f"{'='*60}")
    logger.info(f"EVALUATION - Scenario {scenario}")
    logger.info(f"{'='*60}")
    
    # Load aux data
    df_aux = load_aux_data()
    
    # Get OOF files
    oof_files = get_oof_files(scenario, config, include_decay)
    
    if not oof_files:
        logger.error("No OOF files found. Run run_oof_for_stacking.py first.")
        return None
    
    logger.info(f"Found {len(oof_files)} models to evaluate:")
    for name, path in oof_files.items():
        logger.info(f"  - {name}: {path}")
    
    # Compare individual models
    logger.info("\n--- Individual Model Comparison ---")
    df_comparison = compare_models_on_oof(oof_files, df_aux, scenario)
    
    # Load meta-dataset and stacker if available
    stacker_dir = Path(config.get('bayesian_stacker', {}).get('output_dir', 'artifacts/stacking'))
    stacker_path = stacker_dir / f'stacker_scenario{scenario}.joblib'
    
    meta_dir = Path(config.get('meta_dataset', {}).get('output_dir', 'artifacts/stacking/meta'))
    
    # Try enhanced meta-dataset first
    meta_path = meta_dir / f'meta_scenario{scenario}_with_decay.parquet'
    if not meta_path.exists():
        meta_path = meta_dir / f'meta_scenario{scenario}.parquet'
    
    if stacker_path.exists() and meta_path.exists():
        logger.info("\n--- Ensemble vs Single Model Comparison ---")
        
        stacker = BayesianStacker.load(str(stacker_path))
        df_meta = pd.read_parquet(meta_path)
        
        ensemble_results = evaluate_ensemble_vs_single_models(
            stacker, df_meta, df_aux, scenario
        )
        
        return {
            'individual_comparison': df_comparison,
            'ensemble_results': ensemble_results
        }
    else:
        logger.warning("Stacker or meta-dataset not found. Run train_bayesian_stacker.py first.")
        return {
            'individual_comparison': df_comparison
        }


def print_summary(results: dict):
    """Print evaluation summary."""
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    for scenario, result in results.items():
        if result is None:
            continue
            
        logger.info(f"\nScenario {scenario}:")
        
        if 'individual_comparison' in result:
            df = result['individual_comparison']
            metric_col = f'metric{scenario}_official'
            
            if metric_col in df.columns:
                best_model = df.iloc[0]['model']
                best_score = df.iloc[0][metric_col]
                logger.info(f"  Best Single Model: {best_model} ({best_score:.6f})")
        
        if 'ensemble_results' in result:
            ens = result['ensemble_results']
            if ens.get('ensemble_metric'):
                logger.info(f"  Ensemble Score: {ens['ensemble_metric']:.6f}")
                logger.info(f"  Improvement: {ens['improvement_pct']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stacking Models with Official Metric')
    parser.add_argument(
        '--scenario', '-s',
        type=int,
        nargs='+',
        default=[1, 2],
        choices=[1, 2],
        help='Scenario(s) to evaluate (default: both)'
    )
    parser.add_argument(
        '--include-decay',
        action='store_true',
        default=True,
        help='Include hierarchical decay model (default: True)'
    )
    parser.add_argument(
        '--no-decay',
        action='store_true',
        help='Exclude hierarchical decay model'
    )
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save results CSV'
    )
    args = parser.parse_args()
    
    include_decay = not args.no_decay
    
    config = load_config()
    
    all_results = {}
    
    for scenario in args.scenario:
        try:
            result = evaluate_for_scenario(scenario, config, include_decay)
            all_results[scenario] = result
        except Exception as e:
            logger.error(f"Error evaluating scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
            all_results[scenario] = None
    
    print_summary(all_results)
    
    # Save results if requested
    if args.save_results:
        all_dfs = []
        for scenario, result in all_results.items():
            if result and 'individual_comparison' in result:
                df = result['individual_comparison'].copy()
                df['scenario'] = scenario
                all_dfs.append(df)
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(args.save_results, index=False)
            logger.info(f"Results saved to {args.save_results}")


if __name__ == '__main__':
    main()
