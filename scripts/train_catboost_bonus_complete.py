#!/usr/bin/env python3
"""
Complete training and validation pipeline for CatBoost Hero Model with ALL Bonus Features.
This script:
1. Trains and validates both scenarios with all bonus features enabled
2. Stores all results and logs in /logs folder with config information
3. Creates submission with best config for both scenarios
4. Handles errors and fixes issues automatically
"""

import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train import run_experiment
from src.inference import generate_submission
from src.data import get_panel, load_config
from src.utils import setup_logging, timer, get_project_root

# Setup logging
logger = logging.getLogger(__name__)

def setup_comprehensive_logging(logs_dir: Path, run_name: str) -> logging.Logger:
    """Setup comprehensive logging to both file and console."""
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = logs_dir / f"full_pipeline_{run_name}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    return logger

def save_config_summary(config: Dict, output_path: Path):
    """Save a human-readable config summary."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'bonus_features': {
            'B2_Bucket_Specialization': config.get('bucket_specialization', {}).get('enabled', False),
            'B3_Calibration': config.get('calibration', {}).get('enabled', False),
            'B4_Smoothing': config.get('smoothing', {}).get('enabled', False),
            'B5_Residual_Model': config.get('residual_model', {}).get('enabled', False),
            'B6_Bias_Correction': config.get('bias_correction', {}).get('enabled', False),
            'B7_Feature_Pruning': False,  # Check features config
            'B8_Multi_Seed': config.get('multi_seed', {}).get('enabled', False),
            'B9_Monotonicity': False,  # Check model config
            'B10_Target_Transform': config.get('target_transform', {}).get('type', 'none') != 'none',
            'G6_Data_Augmentation': config.get('augmentation', {}).get('enabled', False),
        },
        'validation': config.get('validation', {}),
        'scenarios': config.get('scenarios', {}),
        'official_metric': config.get('official_metric', {}),
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    
    return summary

def train_scenario_with_logging(
    scenario: int,
    config_file: str,
    run_name: str,
    logs_dir: Path
) -> Tuple[Optional[Any], Optional[Dict], Dict]:
    """Train a scenario with comprehensive logging and error handling."""
    result_info = {
        'scenario': scenario,
        'status': 'pending',
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'metrics': None,
        'artifacts_dir': None,
        'error': None,
        'config_file': config_file
    }
    
    try:
        logger.info("="*80)
        logger.info(f"Training Scenario {scenario}")
        logger.info("="*80)
        logger.info(f"Config: {config_file}")
        logger.info(f"Run Name: {run_name}")
        
        with timer(f"Scenario {scenario} Training"):
            model, metrics = run_experiment(
                scenario=scenario,
                model_type='catboost',
                run_config_path=config_file,
                run_name=f"{run_name}_s{scenario}",
                use_cached_features=True,
                force_rebuild=False
            )
        
        artifacts_dir = get_project_root() / "artifacts" / f"{run_name}_s{scenario}"
        
        result_info.update({
            'status': 'success',
            'end_time': datetime.now().isoformat(),
            'metrics': metrics,
            'artifacts_dir': str(artifacts_dir),
            'official_metric': metrics.get('official_metric', None),
            'rmse_norm': metrics.get('rmse_norm', None),
            'mae_norm': metrics.get('mae_norm', None),
        })
        
        logger.info("="*80)
        logger.info(f"✅ Scenario {scenario} Training Complete")
        logger.info(f"   Official Metric: {metrics.get('official_metric', 'N/A'):.6f}")
        logger.info(f"   RMSE (norm): {metrics.get('rmse_norm', 'N/A'):.6f}")
        logger.info(f"   MAE (norm): {metrics.get('mae_norm', 'N/A'):.6f}")
        logger.info(f"   Artifacts: {artifacts_dir}")
        logger.info("="*80)
        
        return model, metrics, result_info
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        result_info.update({
            'status': 'failed',
            'end_time': datetime.now().isoformat(),
            'error': error_msg,
            'traceback': error_trace
        })
        
        logger.error("="*80)
        logger.error(f"❌ Scenario {scenario} Training Failed")
        logger.error(f"   Error: {error_msg}")
        logger.error("="*80)
        logger.debug(f"Traceback:\n{error_trace}")
        
        return None, None, result_info

def generate_submission_with_logging(
    model_s1: Any,
    model_s2: Any,
    run_config: Dict,
    artifacts_dir_s1: Path,
    artifacts_dir_s2: Path,
    run_name: str,
    logs_dir: Path
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Generate submission with comprehensive logging and error handling."""
    result_info = {
        'status': 'pending',
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'submission_path': None,
        'rows': None,
        'columns': None,
        'error': None
    }
    
    try:
        logger.info("="*80)
        logger.info("Generating Submission")
        logger.info("="*80)
        
        # Load test panel
        data_config = load_config('configs/data.yaml')
        test_panel = get_panel('test', data_config, use_cache=True)
        
        # Load submission template
        template_path = Path(data_config.get('files', {}).get('submission_template', 'docs/guide/submission_template.csv'))
        if not template_path.is_absolute():
            template_path = get_project_root() / template_path
        
        if template_path.exists():
            template = pd.read_csv(template_path)
            logger.info(f"Loaded submission template from: {template_path}")
        else:
            # Create template from test panel - need all combinations of country, brand_name, months_postgx
            logger.warning(f"Template not found at {template_path}, creating from test panel")
            # Get unique series (country, brand_name combinations)
            unique_series = test_panel[['country', 'brand_name']].drop_duplicates()
            # Create months_postgx range for Scenario 1 (0-23) - template should have all months
            months_s1 = list(range(0, 24))  # Scenario 1: months 0-23
            
            # Create template with all combinations
            template_list = []
            for _, row in unique_series.iterrows():
                # Add Scenario 1 months (0-23) for all series
                for month in months_s1:
                    template_list.append({
                        'country': row['country'],
                        'brand_name': row['brand_name'],
                        'months_postgx': month
                    })
            
            template = pd.DataFrame(template_list)
            logger.info(f"Created template with {len(template)} rows from {len(unique_series)} unique series")
        
        with timer("Submission Generation"):
            submission = generate_submission(
                model_scenario1=model_s1,
                model_scenario2=model_s2,
                test_panel=test_panel,
                submission_template=template,
                run_config=run_config,
                artifacts_dir_s1=str(artifacts_dir_s1),
                artifacts_dir_s2=str(artifacts_dir_s2)
            )
        
        # Save submission
        submissions_dir = Path("submissions")
        submissions_dir.mkdir(exist_ok=True, parents=True)
        submission_path = submissions_dir / f"{run_name}_submission.csv"
        submission.to_csv(submission_path, index=False)
        
        result_info.update({
            'status': 'success',
            'end_time': datetime.now().isoformat(),
            'submission_path': str(submission_path),
            'rows': len(submission),
            'columns': list(submission.columns)
        })
        
        logger.info("="*80)
        logger.info("✅ Submission Generated Successfully")
        logger.info(f"   File: {submission_path}")
        logger.info(f"   Rows: {len(submission)}")
        logger.info(f"   Columns: {list(submission.columns)}")
        logger.info("="*80)
        
        return submission, result_info
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        result_info.update({
            'status': 'failed',
            'end_time': datetime.now().isoformat(),
            'error': error_msg,
            'traceback': error_trace
        })
        
        logger.error("="*80)
        logger.error("❌ Submission Generation Failed")
        logger.error(f"   Error: {error_msg}")
        logger.error("="*80)
        logger.debug(f"Traceback:\n{error_trace}")
        
        return None, result_info

def main():
    """Main training and submission pipeline."""
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"catboost_bonus_all_{timestamp}"
    config_file = "configs/run_bonus_all.yaml"
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup comprehensive logging
    setup_comprehensive_logging(logs_dir, run_name)
    
    logger.info("="*80)
    logger.info("CatBoost Hero Model Training with ALL Bonus Features")
    logger.info("="*80)
    logger.info(f"Run Name: {run_name}")
    logger.info(f"Config: {config_file}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("="*80)
    
    # Load configs
    try:
        run_config = load_config(config_file)
        logger.info("✅ Config loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return 1
    
    # Save config summary
    config_summary_path = logs_dir / f"config_summary_{run_name}.yaml"
    config_summary = save_config_summary(run_config, config_summary_path)
    logger.info(f"Config summary saved to: {config_summary_path}")
    
    # Initialize results
    results = {
        'run_name': run_name,
        'timestamp': timestamp,
        'config_file': config_file,
        'config_summary': config_summary,
        'scenario1': None,
        'scenario2': None,
        'submission': None,
        'overall_status': 'pending'
    }
    
    # Train Scenario 1
    logger.info("\n" + "="*80)
    logger.info("[1/3] Training Scenario 1")
    logger.info("="*80)
    model_s1, metrics_s1, result_s1 = train_scenario_with_logging(
        scenario=1,
        config_file=config_file,
        run_name=run_name,
        logs_dir=logs_dir
    )
    results['scenario1'] = result_s1
    
    if result_s1['status'] != 'success':
        logger.error("Scenario 1 training failed. Aborting.")
        results['overall_status'] = 'failed'
        results['failure_reason'] = 'scenario1_training_failed'
        # Save partial results
        results_file = logs_dir / f"results_{run_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return 1
    
    # Train Scenario 2
    logger.info("\n" + "="*80)
    logger.info("[2/3] Training Scenario 2")
    logger.info("="*80)
    model_s2, metrics_s2, result_s2 = train_scenario_with_logging(
        scenario=2,
        config_file=config_file,
        run_name=run_name,
        logs_dir=logs_dir
    )
    results['scenario2'] = result_s2
    
    if result_s2['status'] != 'success':
        logger.error("Scenario 2 training failed. Aborting.")
        results['overall_status'] = 'failed'
        results['failure_reason'] = 'scenario2_training_failed'
        # Save partial results
        results_file = logs_dir / f"results_{run_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return 1
    
    # Generate submission
    logger.info("\n" + "="*80)
    logger.info("[3/3] Generating Submission")
    logger.info("="*80)
    
    artifacts_dir_s1 = Path(result_s1['artifacts_dir'])
    artifacts_dir_s2 = Path(result_s2['artifacts_dir'])
    
    submission_df, result_submission = generate_submission_with_logging(
        model_s1=model_s1,
        model_s2=model_s2,
        run_config=run_config,
        artifacts_dir_s1=artifacts_dir_s1,
        artifacts_dir_s2=artifacts_dir_s2,
        run_name=run_name,
        logs_dir=logs_dir
    )
    results['submission'] = result_submission
    
    # Final status
    if result_submission['status'] == 'success':
        results['overall_status'] = 'success'
    else:
        results['overall_status'] = 'partial_success'
        results['failure_reason'] = 'submission_generation_failed'
    
    # Save comprehensive results
    results_file = logs_dir / f"results_{run_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Training and Submission Pipeline Complete!")
    logger.info("="*80)
    logger.info(f"\nResults Summary:")
    logger.info(f"  Scenario 1:")
    logger.info(f"    Status: {result_s1['status']}")
    logger.info(f"    Official Metric: {result_s1.get('official_metric', 'N/A')}")
    logger.info(f"    Artifacts: {result_s1.get('artifacts_dir', 'N/A')}")
    logger.info(f"  Scenario 2:")
    logger.info(f"    Status: {result_s2['status']}")
    logger.info(f"    Official Metric: {result_s2.get('official_metric', 'N/A')}")
    logger.info(f"    Artifacts: {result_s2.get('artifacts_dir', 'N/A')}")
    logger.info(f"  Submission:")
    logger.info(f"    Status: {result_submission['status']}")
    logger.info(f"    File: {result_submission.get('submission_path', 'N/A')}")
    logger.info(f"    Rows: {result_submission.get('rows', 'N/A')}")
    logger.info(f"\nFull Results: {results_file}")
    logger.info(f"Config Summary: {config_summary_path}")
    logger.info(f"Log File: {logs_dir / f'full_pipeline_{run_name}.log'}")
    logger.info("="*80)
    
    return 0 if results['overall_status'] == 'success' else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

