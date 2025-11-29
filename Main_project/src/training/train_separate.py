# =============================================================================
# File: src/training/train_separate.py
# Description: Separate training pipeline for Scenario 1 and Scenario 2
#
# This mode trains two separate sets of models:
#   - Scenario 1: Uses only pre-LOE features, predicts months 0-23
#   - Scenario 2: Uses pre-LOE + early post-LOE features, predicts months 6-23
#
# =============================================================================

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='.*unsupported index.*')

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # Import module for live access (multi-config)
from config import *
from data_loader import load_all_data, merge_datasets, split_train_validation
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import GradientBoostingModel, BaselineModels, HybridPhysicsMLModel, ARIHOWModel, prepare_training_data
from evaluation import evaluate_model, compare_models
from scenarios.scenarios import get_months_for_scenario, get_scenario_definition


def train_scenario(
    scenario: int,
    merged: pd.DataFrame,
    avg_j: pd.DataFrame,
    aux_df: pd.DataFrame,
    test_mode: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train all enabled models for a single scenario.
    
    Args:
        scenario: 1 or 2
        merged: Merged training data
        avg_j: DataFrame with avg_vol per brand
        aux_df: Auxiliary file with avg_vol and bucket
        test_mode: Use subset of data
        
    Returns:
        Tuple of (comparison_df, run_info)
    """
    # Show which models are enabled
    enabled_models = [k for k, v in MODELS_ENABLED.items() if v]
    disabled_models = [k for k, v in MODELS_ENABLED.items() if not v]
    
    print("=" * 70)
    print(f"🎯 TRAINING SCENARIO {scenario}")
    print("=" * 70)
    print(f"\n🤖 Models to train ({len(enabled_models)}):")
    for m in enabled_models:
        print(f"   ✅ {m}")
    if disabled_models:
        print(f"\n⏭️ Skipped models ({len(disabled_models)}):")
        for m in disabled_models:
            print(f"   ❌ {m}")
    
    scenario_def = get_scenario_definition(scenario)
    months_to_predict = scenario_def['horizon']
    
    # Feature engineering
    print("\n🔧 Feature engineering...")
    featured = create_all_features(merged, avg_j)
    
    # Split
    train_df, val_df = split_train_validation(featured)
    
    # Prepare features
    feature_cols = get_feature_columns(featured)
    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_val, y_val = prepare_training_data(val_df, feature_cols)
    
    # Get validation actuals for evaluation
    val_actual = val_df[val_df['months_postgx'].isin(months_to_predict)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ].copy()
    
    val_brands = val_df[['country', 'brand_name']].drop_duplicates()
    val_avg_j = avg_j.merge(val_brands, on=['country', 'brand_name'])
    
    # Track results
    all_results = []
    model_names = []
    models_trained = {}
    best_rate = DEFAULT_DECAY_RATE
    
    # =========================================================================
    # 1. Baseline: No erosion
    # =========================================================================
    if MODELS_ENABLED.get('baseline_no_erosion', False):
        print("\n📊 Training: Baseline - No Erosion")
        pred = BaselineModels.naive_persistence(val_avg_j, months_to_predict)
        results = evaluate_model(val_actual, pred, aux_df, scenario)
        all_results.append(results)
        model_names.append("Baseline-NoErosion")
    
    # =========================================================================
    # 2. Baseline: Exponential decay
    # =========================================================================
    if MODELS_ENABLED.get('baseline_exp_decay', False):
        print("\n📊 Training: Baseline - Exponential Decay")
        
        # Tune decay rate
        train_brands = train_df[['country', 'brand_name']].drop_duplicates()
        train_avg_j = avg_j.merge(train_brands, on=['country', 'brand_name'])
        train_actual = train_df[train_df['months_postgx'].isin(months_to_predict)][
            ['country', 'brand_name', 'months_postgx', 'volume']
        ]
        
        best_rate, _ = BaselineModels.tune_decay_rate(train_actual, train_avg_j, 'exponential')
        
        pred = BaselineModels.exponential_decay(val_avg_j, months_to_predict, best_rate)
        results = evaluate_model(val_actual, pred, aux_df, scenario)
        all_results.append(results)
        model_names.append(f"Baseline-ExpDecay({best_rate:.3f})")
        models_trained['best_decay_rate'] = best_rate
    else:
        # Still tune for hybrid models
        train_brands = train_df[['country', 'brand_name']].drop_duplicates()
        train_avg_j = avg_j.merge(train_brands, on=['country', 'brand_name'])
        train_actual = train_df[train_df['months_postgx'].isin(months_to_predict)][
            ['country', 'brand_name', 'months_postgx', 'volume']
        ]
        best_rate, _ = BaselineModels.tune_decay_rate(train_actual, train_avg_j, 'exponential')
    
    # =========================================================================
    # 3. LightGBM
    # =========================================================================
    lgbm_model = None
    if MODELS_ENABLED.get('lightgbm', False):
        print("\n📊 Training: LightGBM")
        
        lgbm_model = GradientBoostingModel(model_type='lightgbm')
        lgbm_model.fit(X_train, y_train, X_val, y_val)
        
        # Generate predictions
        val_pred_data = val_df[val_df['months_postgx'].isin(months_to_predict)].copy()
        X_val_pred = val_pred_data[feature_cols].fillna(0)
        val_pred_data['volume_pred'] = lgbm_model.predict(X_val_pred)
        
        pred = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        results = evaluate_model(val_actual, pred, aux_df, scenario)
        all_results.append(results)
        model_names.append("LightGBM")
        
        # Save model (with config prefix in multi-config mode)
        lgbm_model.save(config.get_model_filename(f"scenario{scenario}_lightgbm"))
        models_trained['lightgbm'] = lgbm_model
    
    # =========================================================================
    # 4. XGBoost
    # =========================================================================
    if MODELS_ENABLED.get('xgboost', False):
        print("\n📊 Training: XGBoost")
        
        xgb_model = GradientBoostingModel(model_type='xgboost')
        xgb_model.fit(X_train, y_train, X_val, y_val)
        
        val_pred_data = val_df[val_df['months_postgx'].isin(months_to_predict)].copy()
        X_val_pred = val_pred_data[feature_cols].fillna(0)
        val_pred_data['volume_pred'] = xgb_model.predict(X_val_pred)
        
        pred = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        results = evaluate_model(val_actual, pred, aux_df, scenario)
        all_results.append(results)
        model_names.append("XGBoost")
        
        # Save model (with config prefix in multi-config mode)
        xgb_model.save(config.get_model_filename(f"scenario{scenario}_xgboost"))
        models_trained['xgboost'] = xgb_model
    
    # =========================================================================
    # 5 & 6. Hybrid Models
    # =========================================================================
    # Prepare data for hybrid models
    train_pred_data = train_df[train_df['months_postgx'].isin(months_to_predict)].copy()
    if 'avg_vol' not in train_pred_data.columns:
        train_pred_data = train_pred_data.merge(avg_j, on=['country', 'brand_name'], how='left')
    
    val_pred_data_hybrid = val_df[val_df['months_postgx'].isin(months_to_predict)].copy()
    if 'avg_vol' not in val_pred_data_hybrid.columns:
        val_pred_data_hybrid = val_pred_data_hybrid.merge(avg_j, on=['country', 'brand_name'], how='left')
    
    X_train_hybrid = train_pred_data[feature_cols].fillna(0)
    y_train_hybrid = train_pred_data['volume']
    avg_vol_train = train_pred_data['avg_vol'].fillna(train_pred_data['avg_vol'].median()).values
    months_train = train_pred_data['months_postgx'].values
    
    X_val_hybrid = val_pred_data_hybrid[feature_cols].fillna(0)
    y_val_hybrid = val_pred_data_hybrid['volume']
    avg_vol_val = val_pred_data_hybrid['avg_vol'].fillna(val_pred_data_hybrid['avg_vol'].median()).values
    months_val = val_pred_data_hybrid['months_postgx'].values
    
    # 5. Hybrid LightGBM
    if MODELS_ENABLED.get('hybrid_lightgbm', False):
        print("\n📊 Training: Hybrid (Physics + LightGBM)")
        
        hybrid_model = HybridPhysicsMLModel(ml_model_type='lightgbm', decay_rate=best_rate)
        hybrid_model.fit(
            X_train_hybrid, y_train_hybrid, avg_vol_train, months_train,
            X_val_hybrid, y_val_hybrid, avg_vol_val, months_val
        )
        
        hybrid_preds = hybrid_model.predict(X_val_hybrid, avg_vol_val, months_val)
        val_pred_data_hybrid['volume_pred'] = hybrid_preds
        
        pred = val_pred_data_hybrid[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        results = evaluate_model(val_actual, pred, aux_df, scenario)
        all_results.append(results)
        model_names.append("Hybrid-Physics+LightGBM")
        
        # Save model (with config prefix in multi-config mode)
        hybrid_model.save(config.get_model_filename(f"scenario{scenario}_hybrid"))
        models_trained['hybrid_lightgbm'] = hybrid_model
    
    # 6. Hybrid XGBoost
    if MODELS_ENABLED.get('hybrid_xgboost', False):
        print("\n📊 Training: Hybrid (Physics + XGBoost)")
        
        hybrid_xgb = HybridPhysicsMLModel(ml_model_type='xgboost', decay_rate=best_rate)
        hybrid_xgb.fit(
            X_train_hybrid, y_train_hybrid, avg_vol_train, months_train,
            X_val_hybrid, y_val_hybrid, avg_vol_val, months_val
        )
        
        hybrid_xgb_preds = hybrid_xgb.predict(X_val_hybrid, avg_vol_val, months_val)
        val_pred_data_hybrid['volume_pred'] = hybrid_xgb_preds
        
        # Save hybrid_xgboost model
        hybrid_xgb.save(config.get_model_filename(f"scenario{scenario}_hybrid_xgboost"))
        
        pred = val_pred_data_hybrid[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        results = evaluate_model(val_actual, pred, aux_df, scenario)
        all_results.append(results)
        model_names.append("Hybrid-Physics+XGBoost")
    
    # =========================================================================
    # 7. ARHOW Model
    # =========================================================================
    if MODELS_ENABLED.get('arihow', False):
        print("\n📊 Training: ARHOW (SARIMAX + HW)")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                arihow_model = ARIHOWModel(
                    arima_order=ARIHOW_PARAMS['arima_order'],
                    seasonal_order=ARIHOW_PARAMS['seasonal_order'],
                    hw_trend=ARIHOW_PARAMS['hw_trend'],
                    hw_seasonal=ARIHOW_PARAMS['hw_seasonal'],
                    hw_seasonal_periods=ARIHOW_PARAMS['hw_seasonal_periods'],
                    weight_window=ARIHOW_PARAMS['weight_window'],
                    suppress_warnings=ARIHOW_PARAMS.get('suppress_warnings', True)
                )
                
                fit_data = featured[featured['months_postgx'] < 0].copy()
                if len(fit_data) == 0:
                    fit_data = featured[featured['months_postgx'] <= 12].copy()
                
                arihow_model.fit(fit_data, target_col='volume', 
                                min_history_months=ARIHOW_PARAMS.get('min_history_months', 6))
                
                pred = arihow_model.predict_with_decay_fallback(
                    val_df, avg_j, months_to_predict=months_to_predict, decay_rate=best_rate
                )
            
            results = evaluate_model(val_actual, pred, aux_df, scenario)
            all_results.append(results)
            model_names.append("ARHOW-SARIMAX+HW")
            
            # Save model (with config prefix in multi-config mode)
            arihow_model.save(config.get_model_filename(f"scenario{scenario}_arihow"))
            models_trained['arihow'] = arihow_model
            
        except Exception as e:
            print(f"   ⚠️ ARIHOW failed: {e}")
    
    # =========================================================================
    # Compare models
    # =========================================================================
    if len(all_results) == 0:
        print("\n⚠️ No models were trained!")
        return None, {}
    
    comparison_df = compare_models(all_results, model_names)
    
    # Best model
    best_idx = comparison_df['final_score'].idxmin()
    best_model = comparison_df.loc[best_idx, 'model']
    best_score = comparison_df.loc[best_idx, 'final_score']
    
    print(f"\n🏆 BEST MODEL (S{scenario}): {best_model} (Score: {best_score:.4f})")
    
    run_info = {
        'scenario': scenario,
        'best_model': best_model,
        'best_score': best_score,
        'best_decay_rate': best_rate,
        'n_models_trained': len(all_results),
        'feature_cols': feature_cols,
        'models': models_trained
    }
    
    return comparison_df, run_info


def train_separate(test_mode: bool = False) -> Dict[str, Any]:
    """
    Main entry point for separate training mode.
    
    Trains S1 and S2 models separately (current behavior).
    
    Args:
        test_mode: Use subset of data for quick testing
        
    Returns:
        Dictionary with results for both scenarios
    """
    print("\n" + "=" * 70)
    print("🔀 TRAINING MODE: SEPARATE (S1 & S2 trained independently)")
    print("=" * 70)
    
    # Load data once
    print("\n📂 Loading data...")
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    
    if test_mode:
        brands = merged[['country', 'brand_name']].drop_duplicates().head(TEST_MODE_BRANDS)
        merged = merged.merge(brands, on=['country', 'brand_name'])
        print(f"⚠️ TEST MODE: Using {len(brands)} brands")
    
    # Create auxiliary file
    aux_df = create_auxiliary_file(merged, save=True)
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    
    results = {}
    
    # Determine which scenarios to run
    scenarios = RUN_SCENARIO if isinstance(RUN_SCENARIO, list) else [RUN_SCENARIO]
    
    for scenario in scenarios:
        comparison_df, run_info = train_scenario(
            scenario=scenario,
            merged=merged,
            avg_j=avg_j,
            aux_df=aux_df,
            test_mode=test_mode
        )
        
        if comparison_df is not None:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = REPORTS_DIR / f"model_comparison_scenario{scenario}_{timestamp}.csv"
            comparison_df.to_csv(csv_path, index=False)
            
            # Also save latest
            comparison_df.to_csv(REPORTS_DIR / f"model_comparison_scenario{scenario}.csv", index=False)
            
            results[f'scenario{scenario}'] = {
                'comparison_df': comparison_df,
                'run_info': run_info
            }
    
    return results


if __name__ == "__main__":
    # Direct execution
    from config import TEST_MODE
    train_separate(test_mode=TEST_MODE)
