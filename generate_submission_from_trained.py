#!/usr/bin/env python3
"""
Generate submission from already-trained models.
This script loads the best trained models and generates a submission file.
"""

import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import generate_submission
from src.data import get_panel, load_config
from src.models.cat_model import CatBoostModel
from src.utils import setup_logging, get_project_root

def main():
    setup_logging()
    
    # Use the latest trained models
    run_name = "catboost_bonus_all_20251130_102233"
    
    artifacts_dir_s1 = get_project_root() / "artifacts" / f"{run_name}_s1"
    artifacts_dir_s2 = get_project_root() / "artifacts" / f"{run_name}_s2"
    
    model_path_s1 = artifacts_dir_s1 / "model_1.bin.cbm"
    model_path_s2 = artifacts_dir_s2 / "model_2.bin.cbm"
    
    if not model_path_s1.exists():
        print(f"ERROR: Model not found: {model_path_s1}")
        return 1
    
    if not model_path_s2.exists():
        print(f"ERROR: Model not found: {model_path_s2}")
        return 1
    
    print("="*80)
    print("Generating Submission from Trained Models")
    print("="*80)
    print(f"Scenario 1 Model: {model_path_s1}")
    print(f"Scenario 2 Model: {model_path_s2}")
    print("="*80)
    
    # Load models using our wrapper (which has the fixes)
    print("\nLoading models...")
    model_s1 = CatBoostModel.load(str(model_path_s1))
    model_s2 = CatBoostModel.load(str(model_path_s2))
    print("✅ Models loaded")
    
    # Load configs
    data_config = load_config('configs/data.yaml')
    run_config = load_config('configs/run_bonus_all.yaml')
    
    # Load test panel
    print("\nLoading test data...")
    test_panel = get_panel('test', data_config, use_cache=True)
    print(f"✅ Test panel loaded: {len(test_panel)} rows")
    
    # Load submission template
    template_path = Path(data_config.get('files', {}).get('submission_template', 'docs/guide/submission_template.csv'))
    if not template_path.is_absolute():
        template_path = get_project_root() / template_path
    
    if template_path.exists():
        template = pd.read_csv(template_path)
        print(f"✅ Template loaded: {len(template)} rows")
    else:
        # Create template from test panel
        print("⚠️  Template not found, creating from test panel...")
        unique_series = test_panel[['country', 'brand_name']].drop_duplicates()
        months_s1 = list(range(0, 24))
        template_list = []
        for _, row in unique_series.iterrows():
            for month in months_s1:
                template_list.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month
                })
        template = pd.DataFrame(template_list)
        print(f"✅ Template created: {len(template)} rows")
    
    # Generate submission
    print("\nGenerating submission...")
    try:
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
        
        print("="*80)
        print("✅ Submission Generated Successfully!")
        print("="*80)
        print(f"File: {submission_path}")
        print(f"Rows: {len(submission)}")
        print(f"Columns: {list(submission.columns)}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error generating submission: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import pandas as pd
    exit_code = main()
    sys.exit(exit_code)

