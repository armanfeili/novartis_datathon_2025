# =============================================================================
# File: scripts/train_models.py
# Description: Train and compare all models for a given scenario
# =============================================================================

import warnings
# Suppress statsmodels warnings globally before importing models
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='.*unsupported index.*')
warnings.filterwarnings('ignore', message='.*No supported index.*')
warnings.filterwarnings('ignore', message='.*No frequency information.*')

import sys
from pathlib import Path
import argparse
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *
from data_loader import load_all_data, merge_datasets, split_train_validation
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import GradientBoostingModel, BaselineModels, HybridPhysicsMLModel, ARIHOWModel, prepare_training_data
from evaluation import evaluate_model, compare_models


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


def save_run_summary(comparison_df: pd.DataFrame, 
                     scenario: int, 
                     best_model: str,
                     best_score: float,
                     timestamp: str,
                     run_id: str,
                     feature_importance: pd.DataFrame = None,
                     data_info: dict = None) -> Path:
    """
    Save a JSON summary of the training run.
    
    Args:
        comparison_df: Model comparison DataFrame
        scenario: Scenario number
        best_model: Name of best performing model
        best_score: Best PE score
        timestamp: Run timestamp
        run_id: Unique run identifier
        feature_importance: Top feature importance DataFrame
        data_info: Information about the data used
        
    Returns:
        Path to saved JSON file
    """
    summary = {
        "run_info": {
            "run_id": run_id,
            "timestamp": timestamp,
            "scenario": scenario,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        },
        "best_model": {
            "name": best_model,
            "final_score": float(best_score)
        },
        "all_results": comparison_df.to_dict(orient='records'),
        "config": get_full_config(),
        "data_info": data_info or {},
        "feature_importance": feature_importance.to_dict(orient='records') if feature_importance is not None else []
    }
    
    # Save JSON
    json_path = REPORTS_DIR / f"run_summary_scenario{scenario}_{run_id}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Run summary saved to: {json_path}")
    return json_path


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
    
    # 5. Hybrid Model (Physics + ML)
    print("\n" + "=" * 50)
    print("📊 Model 5: Hybrid (Physics + LightGBM)")
    print("=" * 50)
    
    # Prepare auxiliary data for hybrid model
    # train_df already has avg_vol from feature engineering, but let's ensure it
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
    
    # Train hybrid model with best decay rate from baseline
    hybrid_model = HybridPhysicsMLModel(
        ml_model_type='lightgbm',
        decay_rate=best_rate
    )
    hybrid_model.fit(
        X_train_hybrid, y_train_hybrid,
        avg_vol_train, months_train,
        X_val_hybrid, y_val_hybrid,
        avg_vol_val, months_val
    )
    
    # Generate predictions
    hybrid_preds = hybrid_model.predict(X_val_hybrid, avg_vol_val, months_val)
    val_pred_data_hybrid['volume_pred'] = hybrid_preds
    
    pred_hybrid = val_pred_data_hybrid[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
    pred_hybrid.columns = ['country', 'brand_name', 'months_postgx', 'volume']
    
    results_hybrid = evaluate_model(val_actual, pred_hybrid, aux_df, scenario)
    all_results.append(results_hybrid)
    model_names.append(f"Hybrid-Physics+LightGBM")
    
    # Save hybrid model
    hybrid_model.save(f"scenario{scenario}_hybrid")
    
    # 6. Hybrid Model with XGBoost
    print("\n" + "=" * 50)
    print("📊 Model 6: Hybrid (Physics + XGBoost)")
    print("=" * 50)
    
    hybrid_xgb_model = HybridPhysicsMLModel(
        ml_model_type='xgboost',
        decay_rate=best_rate
    )
    hybrid_xgb_model.fit(
        X_train_hybrid, y_train_hybrid,
        avg_vol_train, months_train,
        X_val_hybrid, y_val_hybrid,
        avg_vol_val, months_val
    )
    
    # Generate predictions
    hybrid_xgb_preds = hybrid_xgb_model.predict(X_val_hybrid, avg_vol_val, months_val)
    val_pred_data_hybrid['volume_pred'] = hybrid_xgb_preds
    
    pred_hybrid_xgb = val_pred_data_hybrid[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
    pred_hybrid_xgb.columns = ['country', 'brand_name', 'months_postgx', 'volume']
    
    results_hybrid_xgb = evaluate_model(val_actual, pred_hybrid_xgb, aux_df, scenario)
    all_results.append(results_hybrid_xgb)
    model_names.append(f"Hybrid-Physics+XGBoost")
    
    # 7. ARHOW Model (ARIMA + Holt-Winters with Learned Weights)
    print("\n" + "=" * 50)
    print("📊 Model 7: ARHOW (SARIMAX + HW + Weights)")
    print("=" * 50)
    
    try:
        import warnings
        # Suppress verbose statsmodels warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # ARHOW is a per-brand time series model with learned weights
            # Uses: y_hat = beta0 * ARIMA + beta1 * HW
            # We need to fit on ALL brands' historical data
            arihow_model = ARIHOWModel(
                arima_order=(1, 1, 1),           # ARIMA(1,1,1)
                seasonal_order=(0, 0, 0, 0),     # No seasonality (data is monthly, no yearly pattern)
                hw_trend='add',                   # Additive trend
                hw_seasonal=None,                 # No seasonality in HW
                hw_seasonal_periods=12,           # Monthly data
                weight_window=12,                 # Use last 12 months to estimate weights
                suppress_warnings=True
            )
            
            # Fit on ALL brands' historical data (use featured which has all brands)
            # Only use pre-generic entry data for fitting (months_postgx < 0)
            fit_data = featured[featured['months_postgx'] < 0].copy()
            if len(fit_data) == 0:
                # If no pre-entry data, use early post-entry data
                fit_data = featured[featured['months_postgx'] <= 12].copy()
            
            arihow_model.fit(fit_data, target_col='volume', min_history_months=6)
            
            # Generate predictions with decay fallback for failed brands
            pred_arihow = arihow_model.predict_with_decay_fallback(
                val_df,
                avg_j,
                months_to_predict=months_to_predict,
                decay_rate=best_rate
            )
        
        results_arihow = evaluate_model(val_actual, pred_arihow, aux_df, scenario)
        all_results.append(results_arihow)
        model_names.append("ARHOW-SARIMAX+HW")
        
        # Save model
        arihow_model.save(f"scenario{scenario}_arihow")
        
    except ImportError as e:
        print(f"   ⚠️ ARHOW skipped: statsmodels not installed ({e})")
        print("   Install with: pip install statsmodels")
    except Exception as e:
        print(f"   ⚠️ ARIHOW failed: {e}")
    
    # =========================================================================
    # Compare models
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON")
    print("=" * 70)
    
    comparison_df = compare_models(all_results, model_names)
    
    # Generate timestamp and run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    
    # Save comparison CSV with timestamp
    csv_filename = f"model_comparison_scenario{scenario}_{run_id}.csv"
    comparison_df.to_csv(REPORTS_DIR / csv_filename, index=False)
    print(f"\n✅ Comparison saved to: {REPORTS_DIR / csv_filename}")
    
    # Also save a "latest" version without timestamp for easy access
    comparison_df.to_csv(REPORTS_DIR / f"model_comparison_scenario{scenario}.csv", index=False)
    
    # Best model
    best_idx = comparison_df['final_score'].idxmin()
    best_model = comparison_df.loc[best_idx, 'model']
    best_score = comparison_df.loc[best_idx, 'final_score']
    
    print(f"\n🏆 BEST MODEL: {best_model} (Score: {best_score:.4f})")
    
    # Feature importance for best GB model
    print("\n📊 Top 10 Features (LightGBM):")
    feature_imp = lgbm_model.get_feature_importance(10)
    print(feature_imp)
    
    # Data info
    data_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_brands": train_df[['country', 'brand_name']].drop_duplicates().shape[0],
        "val_brands": val_df[['country', 'brand_name']].drop_duplicates().shape[0],
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "months_predicted": months_to_predict,
        "test_mode": test_mode
    }
    
    # Save JSON summary
    save_run_summary(
        comparison_df=comparison_df,
        scenario=scenario,
        best_model=best_model,
        best_score=best_score,
        timestamp=timestamp,
        run_id=run_id,
        feature_importance=feature_imp,
        data_info=data_info
    )
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Train all models')
    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2])
    parser.add_argument('--test', action='store_true', help='Test mode')
    
    args = parser.parse_args()
    
    train_all_models(scenario=args.scenario, test_mode=args.test)


if __name__ == "__main__":
    main()
