# =============================================================================
# File: scripts/train_models.py
# Description: Train and compare all models - supports separate and unified modes
#
# TRAIN_MODE (from config.py):
#   "separate" - Train S1 and S2 pipelines separately (default)
#   "unified"  - Train a single global model for both scenarios
#
# =============================================================================

import warnings
# Suppress statsmodels warnings globally before importing models
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='.*unsupported index.*')
warnings.filterwarnings('ignore', message='.*No supported index.*')
warnings.filterwarnings('ignore', message='.*No frequency information.*')

import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config  # Import module for live access (multi-config)
from config import *
from data_loader import load_all_data, merge_datasets, split_train_validation
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import GradientBoostingModel, BaselineModels, HybridPhysicsMLModel, ARIHOWModel, prepare_training_data
from evaluation import evaluate_model, compare_models


def get_full_config() -> dict:
    """Get all configuration settings as a dictionary."""
    return {
        "run_mode": {
            "scenario": RUN_SCENARIO,
            "test_mode": TEST_MODE,
            "test_mode_brands": TEST_MODE_BRANDS,
            "submissions_enabled": SUBMISSIONS_ENABLED
        },
        "pipeline_steps": {
            "run_eda": RUN_EDA,
            "run_training": RUN_TRAINING,
            "run_submission": RUN_SUBMISSION,
            "run_validation": RUN_VALIDATION
        },
        "models_enabled": MODELS_ENABLED,
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
            "xgb_params": XGB_PARAMS,
            "arihow_params": ARIHOW_PARAMS,
            "hybrid_params": HYBRID_PARAMS
        },
        "baseline_params": {
            "decay_rate_range": DECAY_RATE_RANGE,
            "decay_rate_steps": DECAY_RATE_STEPS,
            "default_decay_rate": DEFAULT_DECAY_RATE
        },
        "feature_params": {
            "lag_windows": LAG_WINDOWS,
            "rolling_windows": ROLLING_WINDOWS,
            "pct_change_windows": PCT_CHANGE_WINDOWS,
            "pre_entry_features": PRE_ENTRY_FEATURES
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
            "time": datetime.now().strftime("%H:%M:%S"),
            "saved_at": datetime.now().isoformat(),
            "config_id": config.ACTIVE_CONFIG_ID,
        },
        "best_model": {
            "name": best_model,
            "final_score": float(best_score)
        },
        "all_results": comparison_df.to_dict(orient='records'),
        "config_snapshot": config.get_current_config_snapshot(),  # Full config for reproducibility
        "data_info": data_info or {},
        "feature_importance": feature_importance.to_dict(orient='records') if feature_importance is not None else []
    }
    
    # Save JSON with config prefix in multi-config mode
    config_prefix = f"{config.ACTIVE_CONFIG_ID}_" if config.MULTI_CONFIG_MODE and config.ACTIVE_CONFIG_ID != 'default' else ""
    json_path = REPORTS_DIR / f"{config_prefix}run_summary_scenario{scenario}_{run_id}.json"
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
    # Show which models are enabled
    enabled_models = [k for k, v in MODELS_ENABLED.items() if v]
    disabled_models = [k for k, v in MODELS_ENABLED.items() if not v]
    
    print("=" * 70)
    print(f"🎯 TRAINING MODELS - SCENARIO {scenario}")
    print("=" * 70)
    print(f"\n🤖 Models to train ({len(enabled_models)}):")
    for m in enabled_models:
        print(f"   ✅ {m}")
    if disabled_models:
        print(f"\n⏭️ Skipped models ({len(disabled_models)}):")
        for m in disabled_models:
            print(f"   ❌ {m}")
    
    # =========================================================================
    # Load and prepare data
    # =========================================================================
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
    # Train models (only if enabled in MODELS_ENABLED config)
    # =========================================================================
    all_results = []
    model_names = []
    
    # Variables needed for multiple models
    best_rate = DEFAULT_DECAY_RATE  # Default, will be tuned if exp_decay is enabled
    
    # 1. Baseline: No erosion (persistence)
    if MODELS_ENABLED.get('baseline_no_erosion', True):
        print("\n" + "=" * 50)
        print("📊 Model 1: Baseline - No Erosion")
        print("=" * 50)
        
        pred_baseline = BaselineModels.naive_persistence(val_avg_j, months_to_predict)
        results_baseline = evaluate_model(val_actual, pred_baseline, aux_df, scenario)
        all_results.append(results_baseline)
        model_names.append("Baseline-NoErosion")
    else:
        print("\n⏭️ Skipping Baseline-NoErosion (disabled in config)")
    
    # 2. Baseline: Exponential decay
    if MODELS_ENABLED.get('baseline_exp_decay', True):
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
    else:
        print("\n⏭️ Skipping Baseline-ExpDecay (disabled in config)")
        # Still tune decay rate for hybrid models
        train_brands = train_df[['country', 'brand_name']].drop_duplicates()
        train_avg_j = avg_j.merge(train_brands, on=['country', 'brand_name'])
        train_actual = train_df[train_df['months_postgx'].isin(months_to_predict)][
            ['country', 'brand_name', 'months_postgx', 'volume']
        ]
        best_rate, _ = BaselineModels.tune_decay_rate(train_actual, train_avg_j, 'exponential')
    
    # 3. LightGBM
    if MODELS_ENABLED.get('lightgbm', True):
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
        
        # Save model (with config prefix in multi-config mode)
        lgbm_model.save(config.get_model_filename(f"scenario{scenario}_lightgbm"))
    else:
        print("\n⏭️ Skipping LightGBM (disabled in config)")
    
    # 4. XGBoost
    if MODELS_ENABLED.get('xgboost', True):
        print("\n" + "=" * 50)
        print("📊 Model 4: XGBoost")
        print("=" * 50)
        
        xgb_model = GradientBoostingModel(model_type='xgboost')
        xgb_model.fit(X_train, y_train, X_val, y_val)
        
        val_pred_data = val_df[val_df['months_postgx'].isin(months_to_predict)].copy()
        X_val_pred = val_pred_data[feature_cols].fillna(0)
        val_pred_data['volume_pred'] = xgb_model.predict(X_val_pred)
    
        pred_xgb = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_xgb.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        results_xgb = evaluate_model(val_actual, pred_xgb, aux_df, scenario)
        all_results.append(results_xgb)
        model_names.append("XGBoost")
        
        # Save model (with config prefix in multi-config mode)
        xgb_model.save(config.get_model_filename(f"scenario{scenario}_xgboost"))
    else:
        print("\n⏭️ Skipping XGBoost (disabled in config)")
    
    # Prepare auxiliary data for hybrid models (needed by both hybrid variants)
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
    
    # 5. Hybrid Model (Physics + LightGBM)
    if MODELS_ENABLED.get('hybrid_lightgbm', True):
        print("\n" + "=" * 50)
        print("📊 Model 5: Hybrid (Physics + LightGBM)")
        print("=" * 50)
        
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
        
        # Save hybrid model (with config prefix in multi-config mode)
        hybrid_model.save(config.get_model_filename(f"scenario{scenario}_hybrid"))
    else:
        print("\n⏭️ Skipping Hybrid-LightGBM (disabled in config)")
    
    # 6. Hybrid Model with XGBoost
    if MODELS_ENABLED.get('hybrid_xgboost', True):
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
    else:
        print("\n⏭️ Skipping Hybrid-XGBoost (disabled in config)")
    
    # 7. ARHOW Model (ARIMA + Holt-Winters with Learned Weights)
    if MODELS_ENABLED.get('arihow', True):
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
                # Parameters loaded from config.py ARIHOW_PARAMS
                arihow_model = ARIHOWModel(
                    arima_order=ARIHOW_PARAMS['arima_order'],
                    seasonal_order=ARIHOW_PARAMS['seasonal_order'],
                    hw_trend=ARIHOW_PARAMS['hw_trend'],
                    hw_seasonal=ARIHOW_PARAMS['hw_seasonal'],
                    hw_seasonal_periods=ARIHOW_PARAMS['hw_seasonal_periods'],
                    weight_window=ARIHOW_PARAMS['weight_window'],
                    suppress_warnings=ARIHOW_PARAMS.get('suppress_warnings', True)
                )
                
                # Fit on ALL brands' historical data (use featured which has all brands)
                # Only use pre-generic entry data for fitting (months_postgx < 0)
                fit_data = featured[featured['months_postgx'] < 0].copy()
                if len(fit_data) == 0:
                    # If no pre-entry data, use early post-entry data
                    fit_data = featured[featured['months_postgx'] <= 12].copy()
                
                arihow_model.fit(fit_data, target_col='volume', 
                           min_history_months=ARIHOW_PARAMS.get('min_history_months', 6))
                
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
            
            # Save model (with config prefix in multi-config mode)
            arihow_model.save(config.get_model_filename(f"scenario{scenario}_arihow"))
            
        except ImportError as e:
            print(f"   ⚠️ ARHOW skipped: statsmodels not installed ({e})")
            print("   Install with: pip install statsmodels")
        except Exception as e:
            print(f"   ⚠️ ARIHOW failed: {e}")
    else:
        print("\n⏭️ Skipping ARHOW (disabled in config)")
    
    # =========================================================================
    # Compare models
    # =========================================================================
    if len(all_results) == 0:
        print("\n⚠️ No models were trained. Enable at least one model in config.py MODELS_ENABLED")
        return None
    
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON")
    print("=" * 70)
    
    comparison_df = compare_models(all_results, model_names)
    
    # Generate date and timestamp
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    timestamp = f"{date_str}_{time_str}"
    run_id = timestamp
    
    # Config prefix for multi-config mode
    config_prefix = f"{config.ACTIVE_CONFIG_ID}_" if config.MULTI_CONFIG_MODE and config.ACTIVE_CONFIG_ID != 'default' else ""
    
    # Save comparison CSV with timestamp (no overwrites)
    csv_filename = f"{config_prefix}model_comparison_scenario{scenario}_{run_id}.csv"
    comparison_df.to_csv(REPORTS_DIR / csv_filename, index=False)
    print(f"\n✅ Comparison saved to: {REPORTS_DIR / csv_filename}")
    
    # Also save a "latest" version for easy access
    latest_filename = f"{config_prefix}model_comparison_scenario{scenario}_latest.csv"
    comparison_df.to_csv(REPORTS_DIR / latest_filename, index=False)
    print(f"   Latest: {latest_filename}")
    
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
    """
    Main entry point. All settings come from config.py.
    
    Supports two training modes (controlled by TRAIN_MODE in config.py):
    - "separate": Train S1 and S2 pipelines separately (default)
    - "unified":  Train a single global model for both scenarios
    
    Key config settings:
    - TRAIN_MODE: "separate" or "unified"
    - TEST_MODE: True for quick testing with 50 brands
    - RUN_SCENARIO: 1, 2, or [1, 2] for both
    """
    # All settings from config.py - no CLI overrides
    train_mode = TRAIN_MODE
    test_mode = TEST_MODE
    scenario = RUN_SCENARIO
    
    print("\n" + "=" * 70)
    print(f"🚀 TRAINING PIPELINE")
    print(f"   Mode: {train_mode.upper()}")
    print(f"   Test mode: {test_mode} ({TEST_MODE_BRANDS} brands)" if test_mode else f"   Test mode: {test_mode} (full)")
    print(f"   Scenario(s): {scenario}")
    print("=" * 70)
    
    # Dispatch based on training mode
    if train_mode == "unified":
        # Unified training: single model for both scenarios
        from training.train_unified import train_unified
        results = train_unified(test_mode=test_mode)
    else:
        # Separate training: S1 and S2 trained independently (default/legacy)
        # Handle multiple scenarios from config
        scenarios = [scenario] if isinstance(scenario, int) else scenario
        
        for sc in scenarios:
            print(f"\n{'='*70}")
            print(f"🚀 Running Scenario {sc} (test_mode={test_mode})")
            print(f"{'='*70}")
            train_all_models(scenario=sc, test_mode=test_mode)


if __name__ == "__main__":
    main()
