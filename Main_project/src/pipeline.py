# =============================================================================
# File: src/pipeline.py
# Description: End-to-end pipeline for training and prediction
# =============================================================================

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from data_loader import load_all_data, merge_datasets, split_train_validation
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import GradientBoostingModel, BaselineModels, prepare_training_data, train_and_evaluate
from evaluation import evaluate_model, analyze_worst_predictions, compare_models
from submission import generate_submission, save_submission


def run_pipeline(scenario: int = 1,
                 model_type: str = 'lightgbm',
                 test_mode: bool = False,
                 save_model: bool = True,
                 generate_test_submission: bool = True) -> dict:
    """
    Run complete training and prediction pipeline.
    
    Args:
        scenario: 1 or 2
        model_type: 'lightgbm', 'xgboost', or 'baseline'
        test_mode: If True, only use subset of data for quick testing
        save_model: If True, save trained model
        generate_test_submission: If True, generate submission for test set
        
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print(f"🚀 RUNNING PIPELINE - Scenario {scenario} - Model: {model_type}")
    print("=" * 70)
    
    results = {
        'scenario': scenario,
        'model_type': model_type
    }
    
    # =========================================================================
    # STEP 1: Load and merge data
    # =========================================================================
    print("\n📂 STEP 1: Loading data...")
    
    volume_train, generics_train, medicine_train = load_all_data(train=True)
    merged_train = merge_datasets(volume_train, generics_train, medicine_train)
    
    if test_mode:
        # Use subset for quick testing
        brands = merged_train[['country', 'brand_name']].drop_duplicates().head(50)
        merged_train = merged_train.merge(brands, on=['country', 'brand_name'])
        print(f"⚠️ TEST MODE: Using {len(brands)} brands only")
    
    # =========================================================================
    # STEP 2: Create auxiliary file (avg_vol, buckets)
    # =========================================================================
    print("\n📊 STEP 2: Creating auxiliary file...")
    
    aux_df = create_auxiliary_file(merged_train, save=True)
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    
    # =========================================================================
    # STEP 3: Feature engineering
    # =========================================================================
    print("\n🔧 STEP 3: Feature engineering...")
    
    featured = create_all_features(merged_train, avg_j)
    
    # =========================================================================
    # STEP 4: Split train/validation
    # =========================================================================
    print("\n✂️ STEP 4: Splitting train/validation...")
    
    train_df, val_df = split_train_validation(featured)
    
    # =========================================================================
    # STEP 5: Prepare training data
    # =========================================================================
    print("\n📦 STEP 5: Preparing training data...")
    
    feature_cols = get_feature_columns(featured)
    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_val, y_val = prepare_training_data(val_df, feature_cols)
    
    # =========================================================================
    # STEP 6: Train model
    # =========================================================================
    print("\n🎯 STEP 6: Training model...")
    
    if model_type == 'baseline':
        # Use exponential decay baseline
        print("Using exponential decay baseline model")
        model = None
        
        # Tune decay rate
        train_brands = train_df[['country', 'brand_name']].drop_duplicates()
        train_avg_j = avg_j.merge(train_brands, on=['country', 'brand_name'])
        
        actual_df = train_df[train_df['months_postgx'].between(0, 23)][
            ['country', 'brand_name', 'months_postgx', 'volume']
        ]
        
        best_rate, _ = BaselineModels.tune_decay_rate(
            actual_df, train_avg_j, 'exponential'
        )
        results['best_decay_rate'] = best_rate
        
    else:
        model, metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val, model_type
        )
        results['train_metrics'] = metrics
        
        if save_model:
            model.save(f"scenario{scenario}_{model_type}")
    
    # =========================================================================
    # STEP 7: Generate validation predictions
    # =========================================================================
    print("\n📈 STEP 7: Generating validation predictions...")
    
    val_brands = val_df[['country', 'brand_name']].drop_duplicates()
    val_avg_j = avg_j.merge(val_brands, on=['country', 'brand_name'])
    
    if scenario == 1:
        months_to_predict = list(range(0, 24))
    else:
        months_to_predict = list(range(6, 24))
    
    if model_type == 'baseline':
        val_predictions = BaselineModels.exponential_decay(
            val_avg_j, months_to_predict, best_rate
        )
    else:
        # Predict using gradient boosting model
        val_featured = val_df[val_df['months_postgx'].isin(months_to_predict)].copy()
        X_val_pred = val_featured[feature_cols].fillna(0)
        val_featured['volume_pred'] = model.predict(X_val_pred)
        
        val_predictions = val_featured[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        val_predictions.columns = ['country', 'brand_name', 'months_postgx', 'volume']
    
    # =========================================================================
    # STEP 8: Evaluate on validation set
    # =========================================================================
    print("\n📊 STEP 8: Evaluating on validation set...")
    
    actual_df = val_df[val_df['months_postgx'].isin(months_to_predict)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ].copy()
    
    eval_results = evaluate_model(actual_df, val_predictions, aux_df, scenario)
    results['eval_results'] = eval_results
    
    # Analyze worst predictions
    worst = analyze_worst_predictions(eval_results['pe_details'])
    results['worst_predictions'] = worst
    
    # =========================================================================
    # STEP 9: Generate test submission (if requested)
    # =========================================================================
    if generate_test_submission:
        print("\n📝 STEP 9: Generating test submission...")
        
        volume_test, generics_test, medicine_test = load_all_data(train=False)
        merged_test = merge_datasets(volume_test, generics_test, medicine_test)
        
        test_brands = merged_test[['country', 'brand_name']].drop_duplicates()
        test_avg_j = avg_j.merge(test_brands, on=['country', 'brand_name'], how='left')
        
        # Fill missing avg_vol with median
        median_avg_vol = test_avg_j['avg_vol'].median()
        test_avg_j['avg_vol'] = test_avg_j['avg_vol'].fillna(median_avg_vol)
        
        if model_type == 'baseline':
            test_predictions = BaselineModels.exponential_decay(
                test_avg_j, months_to_predict, best_rate
            )
        else:
            # Create features for test data
            test_featured = create_all_features(merged_test, test_avg_j)
            test_pred_data = test_featured[test_featured['months_postgx'].isin(months_to_predict)].copy()
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in test_pred_data.columns:
                    test_pred_data[col] = 0
            
            X_test = test_pred_data[feature_cols].fillna(0)
            test_pred_data['volume_pred'] = model.predict(X_test)
            
            test_predictions = test_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
            test_predictions.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        # Generate and save submission
        submission = generate_submission(test_predictions, scenario)
        submission_path = save_submission(submission, scenario, suffix=model_type)
        results['submission_path'] = submission_path
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"   Scenario: {scenario}")
    print(f"   Model: {model_type}")
    print(f"   Final Score: {eval_results['final_score']:.4f}")
    print(f"   Bucket 1 PE: {eval_results['bucket1_avg_pe']:.4f} (n={eval_results['n_bucket1']})")
    print(f"   Bucket 2 PE: {eval_results['bucket2_avg_pe']:.4f} (n={eval_results['n_bucket2']})")
    
    if generate_test_submission:
        print(f"   Submission: {results.get('submission_path', 'N/A')}")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Run Novartis Datathon Pipeline')
    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2],
                        help='Scenario to run (1 or 2)')
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost', 'baseline'],
                        help='Model type')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (subset of data)')
    parser.add_argument('--no-submission', action='store_true',
                        help='Skip test submission generation')
    
    args = parser.parse_args()
    
    results = run_pipeline(
        scenario=args.scenario,
        model_type=args.model,
        test_mode=args.test,
        generate_test_submission=not args.no_submission
    )
    
    return results


if __name__ == "__main__":
    main()
