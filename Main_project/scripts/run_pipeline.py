# =============================================================================
# File: scripts/run_pipeline.py
# Description: Master pipeline script - runs all steps based on config.py toggles
#
# 🚀 USAGE:
#    python scripts/run_pipeline.py
#
# All settings are read from src/config.py:
#    - RUN_SCENARIO: Which scenario(s) to run (1, 2, or [1, 2])
#    - TEST_MODE: Fast testing with 50 brands vs full training
#    - RUN_EDA: Run EDA visualization
#    - RUN_TRAINING: Train models
#    - RUN_SUBMISSION: Generate submission files
#    - RUN_VALIDATION: Validate submissions
#    - MODELS_ENABLED: Which models to train
#    - SUBMISSION_MODEL: Which model to use for submissions
#
# =============================================================================

import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *


def print_config_summary():
    """Print current configuration summary."""
    print("=" * 70)
    print("📋 PIPELINE CONFIGURATION")
    print("=" * 70)
    
    print(f"\n🎮 RUN MODE:")
    print(f"   Scenario: {RUN_SCENARIO}")
    print(f"   Test mode: {TEST_MODE} ({TEST_MODE_BRANDS} brands)")
    
    print(f"\n🔄 PIPELINE STEPS:")
    print(f"   {'✅' if RUN_EDA else '❌'} Run EDA")
    print(f"   {'✅' if RUN_TRAINING else '❌'} Run Training")
    print(f"   {'✅' if RUN_SUBMISSION else '❌'} Generate Submissions")
    print(f"   {'✅' if RUN_VALIDATION else '❌'} Validate Submissions")
    
    print(f"\n🤖 MODELS ENABLED:")
    enabled_count = sum(1 for v in MODELS_ENABLED.values() if v)
    print(f"   {enabled_count}/{len(MODELS_ENABLED)} models enabled")
    for model, enabled in MODELS_ENABLED.items():
        print(f"   {'✅' if enabled else '❌'} {model}")
    
    print(f"\n📤 SUBMISSION MODEL: {SUBMISSION_MODEL}")
    print("=" * 70)


def run_eda():
    """Run EDA visualization notebook/script."""
    print("\n" + "=" * 70)
    print("📊 STEP 1: RUNNING EDA VISUALIZATION")
    print("=" * 70)
    
    eda_script = PROJECT_ROOT / "notebooks" / "01_eda_visualization.py"
    if eda_script.exists():
        result = subprocess.run(
            [sys.executable, str(eda_script)],
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            print("\n✅ EDA completed successfully!")
        else:
            print(f"\n⚠️ EDA finished with return code: {result.returncode}")
    else:
        print(f"\n⚠️ EDA script not found: {eda_script}")


def run_training():
    """Run model training."""
    print("\n" + "=" * 70)
    print("🤖 STEP 2: TRAINING MODELS")
    print("=" * 70)
    
    train_script = PROJECT_ROOT / "scripts" / "train_models.py"
    
    # Build command - the script will read config for scenario and test mode
    cmd = [sys.executable, str(train_script)]
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
    else:
        print(f"\n⚠️ Training finished with return code: {result.returncode}")


def run_submission():
    """Generate submission files."""
    print("\n" + "=" * 70)
    print("📤 STEP 3: GENERATING SUBMISSIONS")
    print("=" * 70)
    
    submission_script = PROJECT_ROOT / "scripts" / "generate_final_submissions.py"
    
    # Build command - the script will read config for model type
    cmd = [sys.executable, str(submission_script)]
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode == 0:
        print("\n✅ Submissions generated successfully!")
    else:
        print(f"\n⚠️ Submission generation finished with return code: {result.returncode}")


def run_validation():
    """Validate submission files."""
    print("\n" + "=" * 70)
    print("🔍 STEP 4: VALIDATING SUBMISSIONS")
    print("=" * 70)
    
    validate_script = PROJECT_ROOT / "scripts" / "validate_submissions.py"
    
    cmd = [sys.executable, str(validate_script)]
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode == 0:
        print("\n✅ Validation completed!")
    else:
        print(f"\n⚠️ Validation finished with return code: {result.returncode}")


def main():
    """Run the complete pipeline based on config toggles."""
    
    print("\n" + "🚀" * 35)
    print("   NOVARTIS DATATHON 2025 - GENERIC EROSION PIPELINE")
    print("🚀" * 35)
    
    # Print configuration
    print_config_summary()
    
    # Confirm before running
    print("\nPress Enter to start the pipeline (or Ctrl+C to cancel)...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline cancelled.")
        return
    
    # Track steps
    steps_run = []
    
    # Step 1: EDA
    if RUN_EDA:
        run_eda()
        steps_run.append("EDA")
    else:
        print("\n⏭️ Skipping EDA (disabled in config)")
    
    # Step 2: Training
    if RUN_TRAINING:
        run_training()
        steps_run.append("Training")
    else:
        print("\n⏭️ Skipping Training (disabled in config)")
    
    # Step 3: Submission
    if RUN_SUBMISSION:
        run_submission()
        steps_run.append("Submission")
    else:
        print("\n⏭️ Skipping Submission Generation (disabled in config)")
    
    # Step 4: Validation
    if RUN_VALIDATION:
        run_validation()
        steps_run.append("Validation")
    else:
        print("\n⏭️ Skipping Validation (disabled in config)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    
    if steps_run:
        print(f"\nSteps completed: {', '.join(steps_run)}")
    else:
        print("\n⚠️ No steps were run. Enable steps in config.py:")
        print("   RUN_EDA = True")
        print("   RUN_TRAINING = True")
        print("   RUN_SUBMISSION = True")
        print("   RUN_VALIDATION = True")
    
    print(f"\n📁 Output directories:")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Reports: {REPORTS_DIR}")
    print(f"   Submissions: {SUBMISSIONS_DIR}")
    
    print("\n" + "🏆" * 35)


if __name__ == "__main__":
    main()
