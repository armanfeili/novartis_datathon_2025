# =============================================================================
# File: scripts/generate_final_submissions.py
# Description: Generate final submission files for competition
# =============================================================================

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *
from data_loader import load_all_data, merge_datasets
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import GradientBoostingModel, BaselineModels
from submission import generate_submission, save_submission


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
        
        filepath = save_submission(submission, scenario, suffix=f"{model_type}_final", include_timestamp=False)
        print(f"   ✅ Saved: {filepath}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ FINAL SUBMISSIONS GENERATED!")
    print("=" * 70)
    print(f"\nFiles created in: {SUBMISSIONS_DIR}")
    print("\nSubmission files:")
    for f in SUBMISSIONS_DIR.glob("*.csv"):
        print(f"   - {f.name}")
    
    print("\n📤 Ready for upload to competition platform!")


def main():
    parser = argparse.ArgumentParser(description='Generate final submissions')
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost', 'baseline'])
    
    args = parser.parse_args()
    
    generate_final_submissions(model_type=args.model)


if __name__ == "__main__":
    main()
