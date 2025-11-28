# =============================================================================
# File: scripts/train_models.py
# Description: Train and compare all models for a given scenario
# =============================================================================

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *
from data_loader import load_all_data, merge_datasets, split_train_validation
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import GradientBoostingModel, BaselineModels, prepare_training_data
from evaluation import evaluate_model, compare_models


def train_all_models(scenario: int = 1, test_mode: bool = False):
    """
    Train and compare all models for a given scenario.
    
    Args:
        scenario: 1 or 2
        test_mode: If True, use subset of data
    """
    print("=" * 70)
    print(f"🎯 TRAINING ALL MODELS - SCENARIO {scenario}")
    print("=" * 70)
    
    # =========================================================================
    # Load and prepare data
    # =========================================================================
    print("\n📂 Loading data...")
    
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    
    if test_mode:
        brands = merged[['country', 'brand_name']].drop_duplicates().head(50)
        merged = merged.merge(brands, on=['country', 'brand_name'])
        print(f"⚠️ TEST MODE: Using {len(brands)} brands")
    
    # Create auxiliary file
    aux_df = create_auxiliary_file(merged, save=True)
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    
    # Feature engineering
    print("\n🔧 Feature engineering...")
    featured = create_all_features(merged, avg_j)
    
    # Split
    train_df, val_df = split_train_validation(featured)
    
    # Prepare features
    feature_cols = get_feature_columns(featured)
    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_val, y_val = prepare_training_data(val_df, feature_cols)
    
    # Get validation actuals
    if scenario == 1:
        months_to_predict = list(range(0, 24))
    else:
        months_to_predict = list(range(6, 24))
    
    val_actual = val_df[val_df['months_postgx'].isin(months_to_predict)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ].copy()
    
    val_brands = val_df[['country', 'brand_name']].drop_duplicates()
    val_avg_j = avg_j.merge(val_brands, on=['country', 'brand_name'])
    
    # =========================================================================
    # Train models
    # =========================================================================
    all_results = []
    model_names = []
    
    # 1. Baseline: No erosion (persistence)
    print("\n" + "=" * 50)
    print("📊 Model 1: Baseline - No Erosion")
    print("=" * 50)
    
    pred_baseline = BaselineModels.naive_persistence(val_avg_j, months_to_predict)
    results_baseline = evaluate_model(val_actual, pred_baseline, aux_df, scenario)
    all_results.append(results_baseline)
    model_names.append("Baseline-NoErosion")
    
    # 2. Baseline: Exponential decay
    print("\n" + "=" * 50)
    print("📊 Model 2: Baseline - Exponential Decay")
    print("=" * 50)
    
    # Tune decay rate
    train_brands = train_df[['country', 'brand_name']].drop_duplicates()
    train_avg_j = avg_j.merge(train_brands, on=['country', 'brand_name'])
    train_actual = train_df[train_df['months_postgx'].isin(months_to_predict)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ]
    
    best_rate, _ = BaselineModels.tune_decay_rate(train_actual, train_avg_j, 'exponential')
    
    pred_exp_decay = BaselineModels.exponential_decay(val_avg_j, months_to_predict, best_rate)
    results_exp_decay = evaluate_model(val_actual, pred_exp_decay, aux_df, scenario)
    all_results.append(results_exp_decay)
    model_names.append(f"Baseline-ExpDecay({best_rate:.3f})")
    
    # 3. LightGBM
    print("\n" + "=" * 50)
    print("📊 Model 3: LightGBM")
    print("=" * 50)
    
    lgbm_model = GradientBoostingModel(model_type='lightgbm')
    lgbm_model.fit(X_train, y_train, X_val, y_val)
    
    # Generate predictions
    val_pred_data = val_df[val_df['months_postgx'].isin(months_to_predict)].copy()
    X_val_pred = val_pred_data[feature_cols].fillna(0)
    val_pred_data['volume_pred'] = lgbm_model.predict(X_val_pred)
    
    pred_lgbm = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
    pred_lgbm.columns = ['country', 'brand_name', 'months_postgx', 'volume']
    
    results_lgbm = evaluate_model(val_actual, pred_lgbm, aux_df, scenario)
    all_results.append(results_lgbm)
    model_names.append("LightGBM")
    
    # Save model
    lgbm_model.save(f"scenario{scenario}_lightgbm")
    
    # 4. XGBoost
    print("\n" + "=" * 50)
    print("📊 Model 4: XGBoost")
    print("=" * 50)
    
    xgb_model = GradientBoostingModel(model_type='xgboost')
    xgb_model.fit(X_train, y_train, X_val, y_val)
    
    val_pred_data['volume_pred'] = xgb_model.predict(X_val_pred)
    
    pred_xgb = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
    pred_xgb.columns = ['country', 'brand_name', 'months_postgx', 'volume']
    
    results_xgb = evaluate_model(val_actual, pred_xgb, aux_df, scenario)
    all_results.append(results_xgb)
    model_names.append("XGBoost")
    
    # Save model
    xgb_model.save(f"scenario{scenario}_xgboost")
    
    # =========================================================================
    # Compare models
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON")
    print("=" * 70)
    
    comparison_df = compare_models(all_results, model_names)
    
    # Save comparison
    comparison_df.to_csv(REPORTS_DIR / f"model_comparison_scenario{scenario}.csv", index=False)
    print(f"\n✅ Comparison saved to: {REPORTS_DIR / f'model_comparison_scenario{scenario}.csv'}")
    
    # Best model
    best_idx = comparison_df['final_score'].idxmin()
    best_model = comparison_df.loc[best_idx, 'model']
    best_score = comparison_df.loc[best_idx, 'final_score']
    
    print(f"\n🏆 BEST MODEL: {best_model} (Score: {best_score:.4f})")
    
    # Feature importance for best GB model
    print("\n📊 Top 10 Features (LightGBM):")
    print(lgbm_model.get_feature_importance(10))
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Train all models')
    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2])
    parser.add_argument('--test', action='store_true', help='Test mode')
    
    args = parser.parse_args()
    
    train_all_models(scenario=args.scenario, test_mode=args.test)


if __name__ == "__main__":
    main()
