#!/usr/bin/env python
"""
Generate Diversified Submissions using Bayesian Stacking.

This script generates multiple submission variants by:
1. Multi-init: Running MAP from different random initializations
2. MCMC: Sampling from the posterior over weights
3. Noise: Adding noise to MAP weights in log space

This hedges risk on the final leaderboard by providing legitimate
diversified submissions that all respect the physics and metric.

Usage:
    python tools/generate_diversified_submissions.py --scenario 1 2 --method multi_init
    python tools/generate_diversified_submissions.py --scenario 1 2 --method mcmc
    python tools/generate_diversified_submissions.py --scenario 1 2 --method noise
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
    generate_bayesian_submission_variants,
    create_blend_of_blends,
    build_test_meta_dataset,
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


def load_meta_dataset(scenario: int, config: dict, use_enhanced: bool = True) -> pd.DataFrame:
    """Load meta-dataset for the scenario."""
    meta_dir = Path(config.get('meta_dataset', {}).get('output_dir', 'artifacts/stacking/meta'))
    
    # Try enhanced version first (with hierarchical decay)
    if use_enhanced:
        enhanced_path = meta_dir / f'meta_scenario{scenario}_with_decay.parquet'
        if enhanced_path.exists():
            logger.info(f"Loading enhanced meta-dataset from {enhanced_path}")
            return pd.read_parquet(enhanced_path)
    
    # Fallback to regular meta-dataset
    meta_path = meta_dir / f'meta_scenario{scenario}.parquet'
    if meta_path.exists():
        logger.info(f"Loading meta-dataset from {meta_path}")
        return pd.read_parquet(meta_path)
    
    raise FileNotFoundError(f"Meta-dataset not found. Run train_bayesian_stacker.py first.")


def get_test_pred_files(scenario: int, config: dict) -> dict:
    """Get test prediction file paths for all base models."""
    model_list = config['stacking'].get(f'scenario{scenario}_models', [])
    test_dir = Path('artifacts/test_preds')
    
    test_files = {}
    for model_name in model_list:
        test_path = test_dir / f'{model_name}_scenario{scenario}.parquet'
        if test_path.exists():
            test_files[model_name] = str(test_path)
        else:
            logger.warning(f"Test predictions not found for {model_name}: {test_path}")
    
    # Add hierarchical decay if enabled
    if config.get('hierarchical_decay', {}).get('enabled', False):
        decay_path = test_dir / f'bayes_decay_scenario{scenario}.parquet'
        if decay_path.exists():
            test_files['bayes_decay'] = str(decay_path)
    
    return test_files


def generate_diversified_for_scenario(scenario: int, config: dict, method: str):
    """Generate diversified submissions for a scenario."""
    logger.info(f"=" * 60)
    logger.info(f"Generating Diversified Submissions for Scenario {scenario}")
    logger.info(f"Method: {method}")
    logger.info(f"=" * 60)
    
    div_config = config.get('diversification', {})
    stacker_config = config.get('bayesian_stacker', {})
    
    # Load meta-dataset
    df_meta = load_meta_dataset(scenario, config, use_enhanced=True)
    
    # Get prediction columns
    pred_cols = [c for c in df_meta.columns if c.startswith('pred_')]
    model_names = [c.replace('pred_', '') for c in pred_cols]
    
    # Extract training data
    X_train = df_meta[pred_cols].values
    y_train = df_meta['y_true'].values
    sample_weight_train = df_meta['sample_weight'].values
    
    # Dirichlet prior
    n_models = len(pred_cols)
    alpha = np.ones(n_models) * stacker_config.get('alpha', 1.0)
    
    # Get test prediction files
    test_pred_files = get_test_pred_files(scenario, config)
    
    if not test_pred_files:
        logger.error("No test prediction files found. Run base models first.")
        return None
    
    # Method-specific kwargs
    method_kwargs = {}
    if method == 'noise':
        method_kwargs['noise_std'] = div_config.get('noise_std', 0.1)
    elif method == 'mcmc':
        mcmc_config = div_config.get('mcmc', {})
        method_kwargs['step_size'] = mcmc_config.get('step_size', 0.05)
        method_kwargs['burn_in'] = mcmc_config.get('burn_in', 50)
        method_kwargs['thin'] = mcmc_config.get('thin', 5)
    
    # Output directory
    output_dir = Path(div_config.get('output_dir', 'submissions/ensemble'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate variants
    n_variants = div_config.get('n_variants', 5)
    
    submissions = generate_bayesian_submission_variants(
        X_train=X_train,
        y_train=y_train,
        sample_weight_train=sample_weight_train,
        test_pred_files=test_pred_files,
        scenario=scenario,
        method=method,
        n_variants=n_variants,
        alpha=alpha,
        save_dir=str(output_dir),
        **method_kwargs
    )
    
    logger.info(f"Generated {len(submissions)} submission variants")
    
    # Create blend of all variants
    if div_config.get('create_blend', True) and len(submissions) > 1:
        blend = create_blend_of_blends(submissions)
        blend_path = output_dir / f'ensemble_scenario{scenario}_blend.csv'
        blend.to_csv(blend_path, index=False)
        logger.info(f"Created blend of {len(submissions)} variants: {blend_path}")
    
    return submissions


def main():
    parser = argparse.ArgumentParser(description='Generate Diversified Submissions')
    parser.add_argument(
        '--scenario', '-s',
        type=int,
        nargs='+',
        default=[1, 2],
        choices=[1, 2],
        help='Scenario(s) to run (default: both)'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        default='multi_init',
        choices=['multi_init', 'mcmc', 'noise'],
        help='Diversification method (default: multi_init)'
    )
    parser.add_argument(
        '--n-variants', '-n',
        type=int,
        default=None,
        help='Number of variants to generate (overrides config)'
    )
    args = parser.parse_args()
    
    config = load_config()
    
    # Override config if specified
    if args.n_variants:
        config.setdefault('diversification', {})['n_variants'] = args.n_variants
    
    all_submissions = {}
    
    for scenario in args.scenario:
        try:
            submissions = generate_diversified_for_scenario(scenario, config, args.method)
            if submissions:
                all_submissions[scenario] = submissions
        except Exception as e:
            logger.error(f"Failed to generate for scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("=" * 60)
    logger.info("Diversified Submission Generation Complete!")
    logger.info("=" * 60)
    
    output_dir = config.get('diversification', {}).get('output_dir', 'submissions/ensemble')
    logger.info(f"Submissions saved to: {output_dir}")
    logger.info("")
    logger.info("Tips for leaderboard strategy:")
    logger.info("1. Submit the 'map' or 'base' variant first as your main submission")
    logger.info("2. Use other variants to hedge risk on private leaderboard")
    logger.info("3. The 'blend' submission often performs most stably")


if __name__ == '__main__':
    main()
