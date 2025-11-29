# =============================================================================
# File: scripts/run_pipeline.py
# Description: Master pipeline orchestrator - runs all steps based on config.py
#
# 🚀 USAGE:
#    python scripts/run_pipeline.py
#
# All settings are read from src/config.py:
#    - TEST_MODE: Fast testing with subset of brands
#    - TRAIN_MODE: "separate" or "unified" training mode
#    - MULTI_CONFIG_MODE: Run multiple configs for comparison
#    - RUN_EDA: Run EDA visualization
#    - RUN_TRAINING: Train models
#    - RUN_SUBMISSION: Generate unified submission file
#    - RUN_VALIDATION: Validate submission against template
#
# 📤 OUTPUT:
#    - Models: models/*.joblib
#    - Reports: reports/
#    - Submission: submissions/submission_baseline_final.csv (7,488 rows)
#
# =============================================================================

import sys
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *


class PipelineRunner:
    """Unified pipeline runner with timing and result tracking."""
    
    def __init__(self):
        self.start_time = None
        self.step_times = {}
        self.results = {}
        
    def print_header(self):
        """Print pipeline header."""
        print("\n" + "🚀" * 35)
        print("   NOVARTIS DATATHON 2025 - GENERIC EROSION PIPELINE")
        print("🚀" * 35)
        
    def print_config(self, config_id: str = None):
        """Print current configuration summary."""
        print("\n" + "=" * 70)
        print("📋 PIPELINE CONFIGURATION")
        print("=" * 70)
        
        if config_id:
            print(f"\n🔧 ACTIVE CONFIG: {config_id}")
            cfg = get_config_by_id(config_id)
            if cfg and 'description' in cfg:
                print(f"   {cfg['description']}")
        
        print(f"\n🎮 RUN MODE:")
        print(f"   Test mode: {TEST_MODE}" + (f" ({TEST_MODE_BRANDS} brands)" if TEST_MODE else " (full)"))
        print(f"   Train mode: {TRAIN_MODE}")
        print(f"   Multi-config: {MULTI_CONFIG_MODE}")
        if TRAIN_MODE == "separate":
            print(f"      → S1 and S2 trained separately")
        else:
            print(f"      → Single unified model for both scenarios")
        print(f"   Scenario(s): {RUN_SCENARIO}")
        
        print(f"\n🔄 PIPELINE STEPS:")
        print(f"   {'✅' if RUN_EDA else '❌'} [1] EDA Visualization")
        print(f"   {'✅' if RUN_TRAINING else '❌'} [2] Model Training")
        print(f"   {'✅' if RUN_SUBMISSION else '❌'} [3] Generate Submission")
        print(f"   {'✅' if RUN_VALIDATION else '❌'} [4] Validate Submission")
        
        if RUN_TRAINING:
            print(f"\n🤖 MODELS TO TRAIN:")
            enabled = [k for k, v in MODELS_ENABLED.items() if v]
            disabled = [k for k, v in MODELS_ENABLED.items() if not v]
            print(f"   Enabled: {', '.join(enabled)}")
            if disabled:
                print(f"   Disabled: {', '.join(disabled)}")
        
        if RUN_SUBMISSION:
            print(f"\n📤 SUBMISSION:")
            print(f"   Output: submission_baseline_final.csv")
            print(f"   Format: S1 (228×24) + S2 (112×18) = 7,488 rows")
        
        print("=" * 70)
    
    def run_eda(self) -> bool:
        """Run EDA visualization step."""
        print("\n" + "=" * 70)
        print("📊 STEP 1: EDA VISUALIZATION")
        print("=" * 70)
        
        try:
            from data_loader import load_all_data, merge_datasets
            from bucket_calculator import create_auxiliary_file
            from eda_analysis import run_full_eda
            
            print("\n📂 Loading data...")
            volume, generics, medicine = load_all_data(train=True)
            merged = merge_datasets(volume, generics, medicine)
            
            if TEST_MODE:
                brands = merged[['country', 'brand_name']].drop_duplicates().head(TEST_MODE_BRANDS)
                merged = merged.merge(brands, on=['country', 'brand_name'])
                print(f"   ⚠️ TEST MODE: Using {len(brands)} brands")
            
            aux_df = create_auxiliary_file(merged, save=True)
            
            print("\n🔍 Running EDA analysis...")
            eda_results = run_full_eda(merged, aux_df)
            
            print(f"\n✅ EDA completed!")
            self.results['eda'] = eda_results
            return True
            
        except Exception as e:
            print(f"\n❌ EDA failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_training(self) -> bool:
        """Run model training step."""
        print("\n" + "=" * 70)
        print("🤖 STEP 2: MODEL TRAINING")
        print("=" * 70)
        
        try:
            if TRAIN_MODE == "unified":
                from training.train_unified import train_unified
                print(f"\n🔀 Training mode: UNIFIED")
                results = train_unified(test_mode=TEST_MODE)
            else:
                from training.train_separate import train_separate
                print(f"\n🔀 Training mode: SEPARATE")
                results = train_separate(test_mode=TEST_MODE)
            
            print(f"\n✅ Training completed!")
            self.results['training'] = results
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_submission(self) -> bool:
        """Generate submission files for ALL enabled models."""
        print("\n" + "=" * 70)
        print("📤 STEP 3: GENERATE SUBMISSIONS")
        print("=" * 70)
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "generate_submission",
                Path(__file__).parent / "generate_submission.py"
            )
            gen_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen_module)
            
            # Generate submissions for ALL enabled models
            enabled_models = [m for m, enabled in SUBMISSIONS_ENABLED.items() if enabled]
            
            if not enabled_models:
                print("\n⚠️ No models enabled in SUBMISSIONS_ENABLED!")
                return False
            
            print(f"\n📝 Generating submissions for {len(enabled_models)} model(s):")
            for m in enabled_models:
                print(f"   • {m}")
            
            generated = []
            for model_type in enabled_models:
                print(f"\n{'─' * 50}")
                print(f"📝 Generating {model_type} submission...")
                try:
                    submission = gen_module.generate_submission(
                        model_type=model_type,
                        decay_rate=DEFAULT_DECAY_RATE,
                        save=True
                    )
                    generated.append(model_type)
                except Exception as e:
                    print(f"   ❌ Failed for {model_type}: {e}")
            
            print(f"\n✅ Generated {len(generated)}/{len(enabled_models)} submissions!")
            print(f"   Successful: {', '.join(generated)}")
            self.results['submissions_generated'] = generated
            return len(generated) > 0
            
        except Exception as e:
            print(f"\n❌ Submission generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_validation(self) -> bool:
        """Validate submission against template."""
        print("\n" + "=" * 70)
        print("🔍 STEP 4: VALIDATE SUBMISSION")
        print("=" * 70)
        
        try:
            import pandas as pd
            
            # Look for latest or timestamped submission files
            submission_files = list(SUBMISSIONS_DIR.glob("submission_*_latest.csv"))
            if not submission_files:
                # Fall back to old naming pattern
                submission_files = list(SUBMISSIONS_DIR.glob("submission_*_final.csv"))
            if not submission_files:
                # Try timestamped files
                submission_files = [f for f in SUBMISSIONS_DIR.glob("submission_*.csv")
                                   if 'template' not in f.name and 'example' not in f.name]
            
            if not submission_files:
                print("   ⚠️ No submission files found")
                return False
            
            latest = max(submission_files, key=lambda p: p.stat().st_mtime)
            print(f"\n📂 Validating: {latest.name}")
            
            template_path = SUBMISSIONS_DIR / "submission_template.csv"
            if not template_path.exists():
                print(f"   ⚠️ Template not found: {template_path}")
                return False
            
            template = pd.read_csv(template_path)
            submission = pd.read_csv(latest)
            
            errors = []
            
            if len(submission) != len(template):
                errors.append(f"Row count: got {len(submission)}, expected {len(template)}")
            
            expected_cols = ['country', 'brand_name', 'months_postgx', 'volume']
            if list(submission.columns) != expected_cols:
                errors.append(f"Columns mismatch: got {list(submission.columns)}")
            
            nan_count = submission['volume'].isna().sum()
            if nan_count > 0:
                errors.append(f"Found {nan_count} NaN values")
            
            template_keys = set(zip(template['country'], template['brand_name'], template['months_postgx']))
            submission_keys = set(zip(submission['country'], submission['brand_name'], submission['months_postgx']))
            
            missing = template_keys - submission_keys
            extra = submission_keys - template_keys
            
            if missing:
                errors.append(f"Missing {len(missing)} combinations")
            if extra:
                errors.append(f"Extra {len(extra)} combinations")
            
            if errors:
                print("\n❌ Validation FAILED:")
                for err in errors:
                    print(f"   {err}")
                return False
            
            print("\n✅ Validation PASSED!")
            print(f"\n📊 Submission Statistics:")
            print(f"   Total rows: {len(submission)}")
            print(f"   Unique brands: {submission[['country', 'brand_name']].drop_duplicates().shape[0]}")
            print(f"   Volume range: [{submission['volume'].min():.2f}, {submission['volume'].max():.2f}]")
            print(f"   Volume mean: {submission['volume'].mean():.2f}")
            
            brand_starts = submission.groupby(['country', 'brand_name'])['months_postgx'].min()
            s1_count = (brand_starts == 0).sum()
            s2_count = (brand_starts == 6).sum()
            print(f"   S1 brands: {s1_count}, S2 brands: {s2_count}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, confirm: bool = True, config_id: str = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            confirm: If True, wait for user confirmation before starting
            config_id: ID of the config being used (for display)
            
        Returns:
            Dictionary with results from each step
        """
        self.start_time = time.time()
        self.print_header()
        self.print_config(config_id=config_id)
        
        if confirm:
            print("\nPress Enter to start the pipeline (or Ctrl+C to cancel)...")
            try:
                input()
            except KeyboardInterrupt:
                print("\n\n❌ Pipeline cancelled.")
                return {}
        
        steps_run = []
        steps_failed = []
        
        # Step 1: EDA
        if RUN_EDA:
            step_start = time.time()
            if self.run_eda():
                steps_run.append("EDA")
            else:
                steps_failed.append("EDA")
            self.step_times['EDA'] = time.time() - step_start
        else:
            print("\n⏭️ Skipping EDA (disabled in config)")
        
        # Step 2: Training
        if RUN_TRAINING:
            step_start = time.time()
            if self.run_training():
                steps_run.append("Training")
            else:
                steps_failed.append("Training")
            self.step_times['Training'] = time.time() - step_start
        else:
            print("\n⏭️ Skipping Training (disabled in config)")
        
        # Step 3: Submission
        if RUN_SUBMISSION:
            step_start = time.time()
            if self.run_submission():
                steps_run.append("Submission")
            else:
                steps_failed.append("Submission")
            self.step_times['Submission'] = time.time() - step_start
        else:
            print("\n⏭️ Skipping Submission (disabled in config)")
        
        # Step 4: Validation
        if RUN_VALIDATION:
            step_start = time.time()
            if self.run_validation():
                steps_run.append("Validation")
            else:
                steps_failed.append("Validation")
            self.step_times['Validation'] = time.time() - step_start
        else:
            print("\n⏭️ Skipping Validation (disabled in config)")
        
        # Summary
        total_time = time.time() - self.start_time
        self._print_summary(steps_run, steps_failed, total_time)
        
        return self.results
    
    def _print_summary(self, steps_run: list, steps_failed: list, total_time: float):
        """Print final pipeline summary."""
        print("\n" + "=" * 70)
        if steps_failed:
            print("⚠️ PIPELINE COMPLETED WITH ERRORS")
        else:
            print("✅ PIPELINE COMPLETE!")
        print("=" * 70)
        
        if steps_run:
            print(f"\n✅ Steps completed: {', '.join(steps_run)}")
        if steps_failed:
            print(f"❌ Steps failed: {', '.join(steps_failed)}")
        
        if not steps_run and not steps_failed:
            print("\n⚠️ No steps were run. Enable steps in config.py:")
            print("   RUN_EDA = True")
            print("   RUN_TRAINING = True")
            print("   RUN_SUBMISSION = True")
            print("   RUN_VALIDATION = True")
        
        if self.step_times:
            print(f"\n⏱️ Execution times:")
            for step, duration in self.step_times.items():
                print(f"   {step}: {duration:.1f}s")
            print(f"   TOTAL: {total_time:.1f}s")
        
        print(f"\n📁 Output directories:")
        print(f"   Models: {MODELS_DIR}")
        print(f"   Reports: {REPORTS_DIR}")
        print(f"   Submissions: {SUBMISSIONS_DIR}")
        
        print("\n" + "🏆" * 35)


def run_single_config(config_id: str = None, confirm: bool = True) -> Dict[str, Any]:
    """Run pipeline with a single config."""
    if config_id:
        from config import apply_config, get_config_by_id
        cfg = get_config_by_id(config_id)
        if cfg:
            apply_config(cfg)
            print(f"\n🔧 Applied config: {config_id}")
    
    runner = PipelineRunner()
    return runner.run(confirm=confirm, config_id=config_id)


def run_multi_config(confirm: bool = True) -> Dict[str, Any]:
    """
    Run pipeline across all configs defined in MULTI_CONFIGS.
    Compares results and saves summary.
    """
    from config import MULTI_CONFIGS, apply_config, reset_to_defaults, REPORTS_DIR
    from datetime import datetime
    
    print("\n" + "🔄" * 35)
    print("   MULTI-CONFIG MODE")
    print("🔄" * 35)
    
    print(f"\n📋 Configs to run: {len(MULTI_CONFIGS)}")
    for i, cfg in enumerate(MULTI_CONFIGS):
        print(f"   [{i}] {cfg['id']}: {cfg.get('description', 'No description')}")
    
    if confirm:
        print("\nPress Enter to start multi-config run (or Ctrl+C to cancel)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n\n❌ Multi-config run cancelled.")
            return {}
    
    all_results = []
    total_start = time.time()
    
    for i, cfg in enumerate(MULTI_CONFIGS):
        config_id = cfg['id']
        print("\n" + "=" * 70)
        print(f"🔧 CONFIG {i+1}/{len(MULTI_CONFIGS)}: {config_id}")
        print("=" * 70)
        
        # Reset to defaults first, then apply this config
        reset_to_defaults()
        apply_config(cfg)
        
        # Run pipeline for this config
        runner = PipelineRunner()
        results = runner.run(confirm=False, config_id=config_id)
        
        # Store results
        config_result = {
            'config_id': config_id,
            'description': cfg.get('description', ''),
            'results': results,
            'step_times': runner.step_times.copy(),
        }
        
        # If training results exist, extract scores
        if 'training' in results and results['training']:
            training = results['training']
            if isinstance(training, dict) and 'best_score' in training:
                config_result['best_score'] = training['best_score']
                config_result['best_model'] = training.get('best_model', 'unknown')
        
        all_results.append(config_result)
    
    total_time = time.time() - total_start
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("📊 MULTI-CONFIG COMPARISON SUMMARY")
    print("=" * 70)
    
    # Create comparison table
    comparison_data = []
    for r in all_results:
        row = {
            'config_id': r['config_id'],
            'description': r['description'],
            'total_time': sum(r['step_times'].values()) if r['step_times'] else 0,
        }
        if 'best_score' in r:
            row['best_score'] = r['best_score']
            row['best_model'] = r['best_model']
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Find best config
    if 'best_score' in comparison_df.columns:
        best_idx = comparison_df['best_score'].idxmin()
        best_config = comparison_df.loc[best_idx, 'config_id']
        best_score = comparison_df.loc[best_idx, 'best_score']
        print(f"\n🏆 BEST CONFIG: {best_config} (Score: {best_score:.4f})")
    
    # Save comparison CSV with date+timestamp (no overwrites)
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    timestamp = f"{date_str}_{time_str}"
    
    comparison_path = REPORTS_DIR / f"multi_config_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n📁 Comparison saved: {comparison_path}")
    
    # Also save latest version for easy access
    latest_path = REPORTS_DIR / "multi_config_comparison_latest.csv"
    comparison_df.to_csv(latest_path, index=False)
    print(f"   Latest: {latest_path.name}")
    
    # Save detailed JSON summary with full config for each run
    json_summary = {
        "run_info": {
            "timestamp": timestamp,
            "date": date_str,
            "time": time_str,
            "saved_at": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "num_configs": len(MULTI_CONFIGS),
        },
        "best_result": {
            "config_id": best_config if 'best_score' in comparison_df.columns else None,
            "best_score": float(best_score) if 'best_score' in comparison_df.columns else None,
        },
        "all_configs": [
            {
                "config_id": r['config_id'],
                "description": r['description'],
                "best_score": r.get('best_score'),
                "best_model": r.get('best_model'),
                "total_time": sum(r['step_times'].values()) if r['step_times'] else 0,
            }
            for r in all_results
        ],
        "comparison_table": comparison_df.to_dict(orient='records'),
    }
    
    json_path = REPORTS_DIR / f"multi_config_summary_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    print(f"   Summary JSON: {json_path.name}")
    
    print(f"\n⏱️ Total multi-config time: {total_time:.1f}s")
    print("\n" + "🏆" * 35)
    
    return {
        'all_results': all_results,
        'comparison_df': comparison_df,
        'best_config': best_config if 'best_score' in comparison_df.columns else None,
    }


def main():
    """Main entry point."""
    if MULTI_CONFIG_MODE:
        run_multi_config(confirm=True)
    else:
        run_single_config(confirm=True)


if __name__ == "__main__":
    main()
