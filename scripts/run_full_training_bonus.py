#!/usr/bin/env python3
"""
Complete training and submission pipeline with all bonus features enabled.
Trains both scenarios, generates submissions, and stores all logs/results.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json

def run_command(cmd, log_file, timeout=3600):
    """Run a command and log output."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")
    print(f"{'='*60}\n")
    
    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False
            )
            
            f.write(result.stdout)
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            f.write(f"\n\nTIMEOUT after {timeout} seconds\n")
            return False
        except Exception as e:
            f.write(f"\n\nERROR: {e}\n")
            return False

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"catboost_bonus_all_{timestamp}"
    config_file = "configs/run_bonus_all.yaml"
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("CatBoost Training with ALL Bonus Features")
    print("="*60)
    print(f"Run Name: {run_name}")
    print(f"Config: {config_file}")
    print(f"Timestamp: {timestamp}")
    print("="*60)
    
    results = {
        'run_name': run_name,
        'timestamp': timestamp,
        'config_file': config_file,
        'scenario1': {'status': 'pending'},
        'scenario2': {'status': 'pending'},
        'submission': {'status': 'pending'}
    }
    
    # Train Scenario 1
    print("\n[1/3] Training Scenario 1...")
    s1_log = logs_dir / f"training_s1_{timestamp}.log"
    s1_success = run_command(
        [
            sys.executable, "-m", "src.train",
            "--scenario", "1",
            "--model", "catboost",
            "--run-config", config_file,
            "--run-name", f"{run_name}_s1"
        ],
        s1_log,
        timeout=3600
    )
    
    if s1_success:
        results['scenario1']['status'] = 'success'
        results['scenario1']['log_file'] = str(s1_log)
        s1_artifacts = f"artifacts/{run_name}_s1"
        results['scenario1']['artifacts_dir'] = s1_artifacts
        
        # Find model file
        artifacts_path = Path(s1_artifacts)
        model_files = list(artifacts_path.glob("model*.bin"))
        if model_files:
            results['scenario1']['model_path'] = str(model_files[0])
        print(f"✅ Scenario 1 training completed")
    else:
        results['scenario1']['status'] = 'failed'
        print(f"❌ Scenario 1 training failed - check {s1_log}")
        return
    
    # Train Scenario 2
    print("\n[2/3] Training Scenario 2...")
    s2_log = logs_dir / f"training_s2_{timestamp}.log"
    s2_success = run_command(
        [
            sys.executable, "-m", "src.train",
            "--scenario", "2",
            "--model", "catboost",
            "--run-config", config_file,
            "--run-name", f"{run_name}_s2"
        ],
        s2_log,
        timeout=3600
    )
    
    if s2_success:
        results['scenario2']['status'] = 'success'
        results['scenario2']['log_file'] = str(s2_log)
        s2_artifacts = f"artifacts/{run_name}_s2"
        results['scenario2']['artifacts_dir'] = s2_artifacts
        
        # Find model file
        artifacts_path = Path(s2_artifacts)
        model_files = list(artifacts_path.glob("model*.bin"))
        if model_files:
            results['scenario2']['model_path'] = str(model_files[0])
        print(f"✅ Scenario 2 training completed")
    else:
        results['scenario2']['status'] = 'failed'
        print(f"❌ Scenario 2 training failed - check {s2_log}")
        return
    
    # Generate submission
    print("\n[3/3] Generating submission...")
    if 'model_path' in results['scenario1'] and 'model_path' in results['scenario2']:
        submission_log = logs_dir / f"inference_{timestamp}.log"
        
        # Use generate_submission function directly
        submission_success = run_command(
            [
                sys.executable, "-c",
                f"""
import sys
sys.path.insert(0, '.')
from src.inference import generate_submission
from src.data import get_panel, load_config
from src.models.cat_model import CatBoostModel
from pathlib import Path
import pandas as pd

# Load configs
data_config = load_config('configs/data.yaml')
run_config = load_config('{config_file}')

# Load test panel
test_panel = get_panel('test', data_config, use_cache=True)

# Load submission template
template_path = Path('data/raw/TEST/submission_template.csv')
if not template_path.exists():
    # Create template from test panel
    template = pd.DataFrame({
        'country': test_panel['country'].unique(),
        'brand_name': test_panel['brand_name'].unique()
    })
else:
    template = pd.read_csv(template_path)

# Load models
model_s1 = CatBoostModel({{}})
model_s1.load('{results['scenario1']['model_path']}')

model_s2 = CatBoostModel({{}})
model_s2.load('{results['scenario2']['model_path']}')

# Generate submission
submission = generate_submission(
    model_scenario1=model_s1,
    model_scenario2=model_s2,
    test_panel=test_panel,
    submission_template=template,
    run_config=run_config,
    artifacts_dir_s1='{results['scenario1']['artifacts_dir']}',
    artifacts_dir_s2='{results['scenario2']['artifacts_dir']}'
)

# Save submission
output_dir = Path('submissions')
output_dir.mkdir(exist_ok=True)
submission_path = output_dir / f'{run_name}_submission.csv'
submission.to_csv(submission_path, index=False)
print(f'Submission saved to: {{submission_path}}')
                """
            ],
            submission_log,
            timeout=600
        )
        
        if submission_success:
            results['submission']['status'] = 'success'
            results['submission']['log_file'] = str(submission_log)
            results['submission']['file'] = f"submissions/{run_name}_submission.csv"
            print(f"✅ Submission generated")
        else:
            results['submission']['status'] = 'failed'
            print(f"❌ Submission generation failed - check {submission_log}")
    
    # Save results summary
    results_file = logs_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Training and Submission Complete!")
    print("="*60)
    print(f"Results summary: {results_file}")
    print(f"Logs directory: {logs_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

