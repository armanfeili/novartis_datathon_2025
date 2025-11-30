#!/usr/bin/env python
"""
Generate Ensemble Submissions using trained Bayesian Stacker.

This script applies the trained Bayesian stacker to test predictions
and generates final submission files.

Usage:
    python tools/generate_ensemble_submission.py --scenario 1
    python tools/generate_ensemble_submission.py --scenario 2
    python tools/generate_ensemble_submission.py --all
    python tools/generate_ensemble_submission.py --all --diversify
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
    apply_ensemble_to_test,
    generate_diversified_submissions,
    build_test_meta_dataset
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


def build_test_pred_file_dict(scenario: int, config: dict) -> dict:
    """Build dictionary of model name -> test prediction file path."""
    model_names = config['stacking'][f'scenario{scenario}_models']
    test_pred_dir = Path('artifacts/test_preds')
    
    test_files = {}
    for model_name in model_names:
        # Try different naming conventions
        possible_paths = [
            test_pred_dir / f'{model_name}_scenario{scenario}.parquet',
            test_pred_dir / f'{model_name}_s{scenario}.parquet',
            test_pred_dir / f'{model_name}.parquet',
        ]
        
        for path in possible_paths:
            if path.exists():
                test_files[model_name] = str(path)
                break
        else:
            logger.warning(f"Test prediction file not found for: {model_name}")
    
    return test_files


def validate_submission(df: pd.DataFrame, scenario: int) -> bool:
    """Validate submission format and content."""
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    
    # Check columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        return False
    
    # Check months range
    if scenario == 1:
        expected_months = set(range(0, 24))
    else:
        expected_months = set(range(6, 24))
    
    actual_months = set(df['months_postgx'].unique())
    if not expected_months.issubset(actual_months):
        missing_months = expected_months - actual_months
        logger.warning(f"Missing months: {missing_months}")
    
    # Check for negatives
    n_negative = (df['volume'] < 0).sum()
    if n_negative > 0:
        logger.warning(f"Found {n_negative} negative predictions (clipped to 0)")
    
    # Check for NaNs
    n_nan = df['volume'].isna().sum()
    if n_nan > 0:
        logger.error(f"Found {n_nan} NaN predictions")
        return False
    
    # Check value range
    min_val = df['volume'].min()
    max_val = df['volume'].max()
    logger.info(f"Prediction range: [{min_val:.4f}, {max_val:.4f}]")
    
    if max_val > 2.0:
        logger.warning(f"Max prediction ({max_val:.4f}) > 2.0")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate ensemble submissions')
    parser.add_argument('--scenario', type=int, choices=[1, 2],
                       help='Scenario to generate submission for')
    parser.add_argument('--all', action='store_true',
                       help='Generate submissions for all scenarios')
    parser.add_argument('--diversify', action='store_true',
                       help='Generate diversified submission variants')
    parser.add_argument('--n-variants', type=int, default=5,
                       help='Number of diversified variants')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for submissions')
    args = parser.parse_args()
    
    if not args.scenario and not args.all:
        parser.error('Either --scenario or --all must be specified')
    
    # Load configuration
    config = load_stacking_config()
    
    # Determine scenarios
    scenarios = [args.scenario] if args.scenario else [1, 2]
    
    # Output directory
    output_dir = Path(args.output_dir or config.get('diversification', {}).get(
        'output_dir', 'submissions/ensemble'
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stacker_dir = Path(config.get('bayesian_stacker', {}).get(
        'output_dir', 'artifacts/stacking'
    ))
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating Ensemble Submission for Scenario {scenario}")
        logger.info(f"{'='*60}")
        
        # Load trained stacker
        stacker_path = stacker_dir / f'stacker_scenario{scenario}.joblib'
        if not stacker_path.exists():
            logger.error(f"Stacker not found: {stacker_path}")
            logger.error("Run train_bayesian_stacker.py first")
            continue
        
        stacker = BayesianStacker.load(str(stacker_path))
        logger.info(f"Loaded stacker with weights: {stacker.get_weights_dict()}")
        
        # Build test prediction file dictionary
        test_files = build_test_pred_file_dict(scenario, config)
        
        if len(test_files) < 2:
            logger.error(f"Need at least 2 test prediction files. Found: {list(test_files.keys())}")
            continue
        
        logger.info(f"Found test predictions for: {list(test_files.keys())}")
        
        if args.diversify:
            # Generate diversified submissions
            logger.info(f"Generating {args.n_variants} diversified submissions...")
            
            submissions = generate_diversified_submissions(
                stacker,
                test_files,
                scenario,
                n_variants=args.n_variants,
                noise_std=config.get('diversification', {}).get('noise_std', 0.1),
                save_dir=str(output_dir)
            )
            
            for i, df in enumerate(submissions):
                variant_name = 'base' if i == 0 else f'variant_{i}'
                logger.info(f"\nVariant: {variant_name}")
                validate_submission(df, scenario)
        
        else:
            # Generate single ensemble submission
            submission_path = output_dir / f'ensemble_scenario{scenario}.csv'
            
            df_submission = apply_ensemble_to_test(
                stacker,
                test_files,
                scenario,
                save_path=str(submission_path)
            )
            
            logger.info(f"\nValidating submission...")
            is_valid = validate_submission(df_submission, scenario)
            
            if is_valid:
                logger.info(f"✓ Valid submission saved to: {submission_path}")
            else:
                logger.error("✗ Submission validation failed")
    
    logger.info("\nEnsemble submission generation complete!")


if __name__ == '__main__':
    main()
