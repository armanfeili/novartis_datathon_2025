# =============================================================================
# File: scripts/generate_final_submissions.py
# Description: Generate final submission files for competition
# =============================================================================

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *
from data_loader import load_all_data, merge_datasets
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import GradientBoostingModel, BaselineModels, HybridPhysicsMLModel
from submission import generate_submission, save_submission


def get_full_config() -> dict:
    """Get all configuration settings as a dictionary."""
    return {
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_raw": str(DATA_RAW),
            "data_processed": str(DATA_PROCESSED),
            "models_dir": str(MODELS_DIR),
            "submissions_dir": str(SUBMISSIONS_DIR),
            "reports_dir": str(REPORTS_DIR)
        },
        "constants": {
            "pre_entry_months": PRE_ENTRY_MONTHS,
            "post_entry_months": POST_ENTRY_MONTHS,
            "bucket_1_threshold": BUCKET_1_THRESHOLD,
            "bucket_1_weight": BUCKET_1_WEIGHT,
            "bucket_2_weight": BUCKET_2_WEIGHT
        },
        "metric_weights_scenario1": {
            "monthly_weight": S1_MONTHLY_WEIGHT,
            "sum_0_5_weight": S1_SUM_0_5_WEIGHT,
            "sum_6_11_weight": S1_SUM_6_11_WEIGHT,
            "sum_12_23_weight": S1_SUM_12_23_WEIGHT
        },
        "metric_weights_scenario2": {
            "monthly_weight": S2_MONTHLY_WEIGHT,
            "sum_6_11_weight": S2_SUM_6_11_WEIGHT,
            "sum_12_23_weight": S2_SUM_12_23_WEIGHT
        },
        "model_params": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_splits_cv": N_SPLITS_CV,
            "lgbm_params": LGBM_PARAMS,
            "xgb_params": XGB_PARAMS
        }
    }


def save_submission_summary(scenario: int,
                            model_type: str,
                            submission_path: Path,
                            timestamp: str,
                            n_brands: int,
                            n_rows: int,
                            volume_stats: dict,
                            decay_rate: float = None) -> Path:
    """
    Save a JSON summary alongside the submission file.
    
    Args:
        scenario: Scenario number (1 or 2)
        model_type: Model used for predictions
        submission_path: Path to the submission CSV
        timestamp: Submission timestamp
        n_brands: Number of brands in submission
        n_rows: Total rows in submission
        volume_stats: Statistics about predicted volumes
        decay_rate: Decay rate used (for baseline/hybrid)
        
    Returns:
        Path to saved JSON file
    """
    summary = {
        "submission_info": {
            "scenario": scenario,
            "model_type": model_type,
            "timestamp": timestamp,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "submission_file": str(submission_path.name)
        },
        "data_stats": {
            "n_brands": n_brands,
            "n_rows": n_rows,
            "months_predicted": list(range(0, 24)) if scenario == 1 else list(range(6, 24)),
            "expected_rows": n_brands * (24 if scenario == 1 else 18)
        },
        "volume_predictions": volume_stats,
        "model_config": {
            "decay_rate": decay_rate,
            "model_type": model_type
        },
        "full_config": get_full_config()
    }
    
    # Save JSON with same name as CSV
    json_path = submission_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Summary saved: {json_path.name}")
    return json_path


def generate_final_submissions(model_type: str = 'lightgbm'):
    """
    Generate final submission files for both scenarios.
    
    Args:
        model_type: 'lightgbm', 'xgboost', or 'baseline'
    """
    print("=" * 70)
    print("📝 GENERATING FINAL SUBMISSIONS")
    print("=" * 70)
    
    # =========================================================================
    # Load training data (for avg_j and features)
    # =========================================================================
    print("\n📂 Loading training data...")
    
    volume_train, generics_train, medicine_train = load_all_data(train=True)
    merged_train = merge_datasets(volume_train, generics_train, medicine_train)
    
    # Create auxiliary file
    aux_df = create_auxiliary_file(merged_train, save=True)
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    
    # =========================================================================
    # Load test data
    # =========================================================================
    print("\n📂 Loading test data...")
    
    volume_test, generics_test, medicine_test = load_all_data(train=False)
    merged_test = merge_datasets(volume_test, generics_test, medicine_test)
    
    test_brands = merged_test[['country', 'brand_name']].drop_duplicates()
    print(f"   Test brands: {len(test_brands)}")
    
    # Compute avg_j FROM TEST DATA (test has pre-entry months)
    print("   Computing avg_vol from test pre-entry data...")
    test_avg_j = compute_avg_j(merged_test)
    
    # Check for missing avg_vol
    n_missing = test_avg_j['avg_vol'].isna().sum()
    if n_missing > 0:
        # Use training median as fallback
        median_avg_vol = avg_j['avg_vol'].median()
        test_avg_j['avg_vol'] = test_avg_j['avg_vol'].fillna(median_avg_vol)
        print(f"   Filled {n_missing} missing avg_vol with training median: {median_avg_vol:.2f}")
    
    # =========================================================================
    # Generate predictions
    # =========================================================================
    
    for scenario in [1, 2]:
        print(f"\n{'=' * 50}")
        print(f"📊 SCENARIO {scenario}")
        print("=" * 50)
        
        if scenario == 1:
            months_to_predict = list(range(0, 24))
        else:
            months_to_predict = list(range(6, 24))
        
        if model_type == 'baseline':
            # Use exponential decay
            print("   Using exponential decay baseline...")
            predictions = BaselineModels.exponential_decay(
                test_avg_j,
                months_to_predict,
                decay_rate=0.05
            )
        elif model_type == 'hybrid':
            # Use hybrid (Physics + ML) model
            print("   Using hybrid (Physics + ML) model...")
            model = HybridPhysicsMLModel(ml_model_type='lightgbm')
            
            try:
                model.load(f"scenario{scenario}_hybrid")
            except FileNotFoundError:
                print(f"   ⚠️ Hybrid model not found. Run train_models.py first.")
                print(f"   Falling back to baseline...")
                predictions = BaselineModels.exponential_decay(
                    test_avg_j, months_to_predict, decay_rate=0.05
                )
                submission = generate_submission(predictions, scenario)
                filepath = save_submission(submission, scenario, suffix=f"baseline_final", include_timestamp=False)
                print(f"   ✅ Saved: {filepath}")
                continue
            
            # Create features for test data
            print("   Creating features...")
            test_featured = create_all_features(merged_test, test_avg_j)
            test_pred_data = test_featured[test_featured['months_postgx'].isin(months_to_predict)].copy()
            
            # Merge avg_vol
            test_pred_data = test_pred_data.merge(test_avg_j, on=['country', 'brand_name'], how='left')
            
            # Get feature columns
            feature_cols = model.feature_names
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in test_pred_data.columns:
                    test_pred_data[col] = 0
            
            X_test = test_pred_data[feature_cols].fillna(0)
            avg_vol_test = test_pred_data['avg_vol'].values
            months_test = test_pred_data['months_postgx'].values
            
            print("   Generating hybrid predictions...")
            test_pred_data['volume_pred'] = model.predict(X_test, avg_vol_test, months_test)
            
            predictions = test_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
            predictions.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        else:
            # Load trained model
            print(f"   Loading {model_type} model...")
            model = GradientBoostingModel(model_type=model_type)
            
            try:
                model.load(f"scenario{scenario}_{model_type}")
            except FileNotFoundError:
                print(f"   ⚠️ Model not found, training new model...")
                # Need to train model first
                from pipeline import run_pipeline
                run_pipeline(scenario=scenario, model_type=model_type, 
                            generate_test_submission=False)
                model.load(f"scenario{scenario}_{model_type}")
            
            # Create features for test data
            print("   Creating features...")
            test_featured = create_all_features(merged_test, test_avg_j)
            test_pred_data = test_featured[test_featured['months_postgx'].isin(months_to_predict)].copy()
            
            # Get feature columns
            feature_cols = model.feature_names
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in test_pred_data.columns:
                    test_pred_data[col] = 0
            
            X_test = test_pred_data[feature_cols].fillna(0)
            
            print("   Generating predictions...")
            test_pred_data['volume_pred'] = model.predict(X_test)
            
            predictions = test_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
            predictions.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        # Generate and save submission
        print("   Validating submission...")
        submission = generate_submission(predictions, scenario)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save submission with timestamp
        filepath_timestamped = save_submission(
            submission, scenario, 
            suffix=f"{model_type}_{timestamp}", 
            include_timestamp=False
        )
        
        # Also save a "latest" version without timestamp for easy access
        filepath_latest = save_submission(
            submission, scenario, 
            suffix=f"{model_type}_final", 
            include_timestamp=False
        )
        
        print(f"   ✅ Saved: {filepath_timestamped.name}")
        print(f"   ✅ Saved: {filepath_latest.name} (latest)")
        
        # Calculate volume statistics
        volume_stats = {
            "min": float(submission['volume'].min()),
            "max": float(submission['volume'].max()),
            "mean": float(submission['volume'].mean()),
            "median": float(submission['volume'].median()),
            "std": float(submission['volume'].std())
        }
        
        # Determine decay rate
        decay_rate = 0.05 if model_type in ['baseline', 'hybrid'] else None
        
        # Save JSON summary for timestamped version
        save_submission_summary(
            scenario=scenario,
            model_type=model_type,
            submission_path=filepath_timestamped,
            timestamp=timestamp,
            n_brands=len(test_brands),
            n_rows=len(submission),
            volume_stats=volume_stats,
            decay_rate=decay_rate
        )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ FINAL SUBMISSIONS GENERATED!")
    print("=" * 70)
    print(f"\nFiles created in: {SUBMISSIONS_DIR}")
    print("\nSubmission files:")
    for f in sorted(SUBMISSIONS_DIR.glob("scenario*.*")):
        print(f"   - {f.name}")
    
    print("\n📤 Ready for upload to competition platform!")


def main():
    parser = argparse.ArgumentParser(description='Generate final submissions')
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost', 'baseline', 'hybrid'])
    
    args = parser.parse_args()
    
    generate_final_submissions(model_type=args.model)


if __name__ == "__main__":
    main()
