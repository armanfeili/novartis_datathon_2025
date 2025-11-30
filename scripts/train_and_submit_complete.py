#!/usr/bin/env python3
"""
Complete training and submission pipeline with all bonus features.
This script trains both scenarios and generates a submission.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train import run_experiment
from src.inference import generate_submission
from src.data import get_panel, load_config
from src.models.cat_model import CatBoostModel
from src.utils import setup_logging, timer

def main():
    setup_logging()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"catboost_bonus_all_{timestamp}"
    config_file = "configs/run_bonus_all.yaml"
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("CatBoost Training with ALL Bonus Features Enabled")
    print("="*70)
    print(f"Run Name: {run_name}")
    print(f"Config: {config_file}")
    print(f"Timestamp: {timestamp}")
    print("="*70)
    
    results = {
        'run_name': run_name,
        'timestamp': timestamp,
        'config_file': config_file,
        'enabled_features': [
            'B2: Bucket Specialization',
            'B3: Post-hoc Calibration',
            'B4: Temporal Smoothing',
            'B6: Bias Correction',
            'B8: Multi-Seed Training',
            'B10: Target Transform (log1p)',
            'G6: Data Augmentation'
        ]
    }
    
    # Load configs
    run_config = load_config(config_file)
    data_config = load_config('configs/data.yaml')
    
    # Train Scenario 1
    print("\n[1/3] Training Scenario 1...")
    print("-" * 70)
    try:
        with timer("Scenario 1 Training"):
            model_s1, metrics_s1 = run_experiment(
                scenario=1,
                model_type='catboost',
                run_config_path=config_file,
                run_name=f"{run_name}_s1",
                use_cached_features=True,
                force_rebuild=False
            )
        
        s1_artifacts = Path("artifacts") / f"{run_name}_s1"
        results['scenario1'] = {
            'status': 'success',
            'metrics': metrics_s1,
            'artifacts_dir': str(s1_artifacts),
            'official_metric': metrics_s1.get('official_metric', 'N/A')
        }
        
        # Find model path
        model_paths = list(s1_artifacts.glob("model*.bin"))
        if not model_paths:
            # Check for bucket models
            bucket_dirs = list(s1_artifacts.glob("bucket*_cat_model"))
            if bucket_dirs:
                model_paths = [list(bd.glob("model.bin"))[0] for bd in bucket_dirs if list(bd.glob("model.bin"))]
        
        if model_paths:
            results['scenario1']['model_path'] = str(model_paths[0])
        
        print(f"✅ Scenario 1 completed - Official Metric: {metrics_s1.get('official_metric', 'N/A'):.6f}")
        
    except Exception as e:
        results['scenario1'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ Scenario 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train Scenario 2
    print("\n[2/3] Training Scenario 2...")
    print("-" * 70)
    try:
        with timer("Scenario 2 Training"):
            model_s2, metrics_s2 = run_experiment(
                scenario=2,
                model_type='catboost',
                run_config_path=config_file,
                run_name=f"{run_name}_s2",
                use_cached_features=True,
                force_rebuild=False
            )
        
        s2_artifacts = Path("artifacts") / f"{run_name}_s2"
        results['scenario2'] = {
            'status': 'success',
            'metrics': metrics_s2,
            'artifacts_dir': str(s2_artifacts),
            'official_metric': metrics_s2.get('official_metric', 'N/A')
        }
        
        # Find model path
        model_paths = list(s2_artifacts.glob("model*.bin"))
        if not model_paths:
            # Check for bucket models
            bucket_dirs = list(s2_artifacts.glob("bucket*_cat_model"))
            if bucket_dirs:
                model_paths = [list(bd.glob("model.bin"))[0] for bd in bucket_dirs if list(bd.glob("model.bin"))]
        
        if model_paths:
            results['scenario2']['model_path'] = str(model_paths[0])
        
        print(f"✅ Scenario 2 completed - Official Metric: {metrics_s2.get('official_metric', 'N/A'):.6f}")
        
    except Exception as e:
        results['scenario2'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ Scenario 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate submission
    print("\n[3/3] Generating submission...")
    print("-" * 70)
    try:
        # Load test panel
        test_panel = get_panel('test', data_config, use_cache=True)
        
        # Load submission template
        template_path = Path('data/raw/TEST/submission_template.csv')
        if template_path.exists():
            template = pd.read_csv(template_path)
        else:
            # Create template from test panel
            template = pd.DataFrame({
                'country': test_panel['country'].unique(),
                'brand_name': test_panel['brand_name'].unique()
            })
        
        # Generate submission
        with timer("Submission Generation"):
            submission = generate_submission(
                model_scenario1=model_s1,
                model_scenario2=model_s2,
                test_panel=test_panel,
                submission_template=template,
                run_config=run_config,
                artifacts_dir_s1=str(s1_artifacts),
                artifacts_dir_s2=str(s2_artifacts)
            )
        
        # Save submission
        submissions_dir = Path("submissions")
        submissions_dir.mkdir(exist_ok=True)
        submission_path = submissions_dir / f"{run_name}_submission.csv"
        submission.to_csv(submission_path, index=False)
        
        results['submission'] = {
            'status': 'success',
            'file': str(submission_path),
            'rows': len(submission),
            'columns': list(submission.columns)
        }
        
        print(f"✅ Submission saved to: {submission_path}")
        print(f"   Rows: {len(submission)}, Columns: {list(submission.columns)}")
        
    except Exception as e:
        results['submission'] = {'status': 'failed', 'error': str(e)}
        print(f"❌ Submission generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results summary
    results_file = logs_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("Training and Submission Complete!")
    print("="*70)
    print(f"\nResults Summary:")
    print(f"  Scenario 1 Metric: {results['scenario1'].get('official_metric', 'N/A')}")
    print(f"  Scenario 2 Metric: {results['scenario2'].get('official_metric', 'N/A')}")
    print(f"  Submission: {results['submission'].get('file', 'N/A')}")
    print(f"\nFull results: {results_file}")
    print(f"Artifacts: artifacts/{run_name}_*")
    print("="*70)

if __name__ == "__main__":
    main()

