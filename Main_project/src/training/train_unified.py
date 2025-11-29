# =============================================================================
# File: src/training/train_unified.py
# Description: Unified training pipeline - single model for both scenarios
#
# This mode trains ONE global model that handles both S1 and S2:
#   - Uses scenario_flag feature to differentiate S1-style and S2-style samples
#   - months_postgx is used as a feature
#   - S2 samples have additional early post-LOE features (months 0-5 summary)
#   - Model learns to handle both regimes with one set of weights
#
# Advantages:
#   - Cuts training time in half (7 models instead of 14)
#   - Model can learn shared patterns across scenarios
#   - Simpler pipeline
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
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # Import module for live access (multi-config)
from config import *
from data_loader import load_all_data, merge_datasets, split_train_validation
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import GradientBoostingModel, BaselineModels, HybridPhysicsMLModel, prepare_training_data
from evaluation import evaluate_model, compare_models
from scenarios.scenarios import get_months_for_scenario, get_scenario_definition
from metric_calculation import compute_metric1, compute_metric2


def add_unified_features(df: pd.DataFrame, avg_j: pd.DataFrame) -> pd.DataFrame:
    """
    Add unified training features including scenario_flag and early post-LOE summaries.
    
    Creates two types of samples from each brand:
    - S1-style: scenario_flag=0, no early post-LOE features
    - S2-style: scenario_flag=1, includes early post-LOE summary features
    
    Args:
        df: DataFrame with all data (must have months_postgx)
        avg_j: DataFrame with avg_vol per brand
        
    Returns:
        DataFrame with unified features added
    """
    df = df.copy()
    
    # Ensure avg_vol is present
    if 'avg_vol' not in df.columns:
        df = df.merge(avg_j[['country', 'brand_name', 'avg_vol']], on=['country', 'brand_name'], how='left')
    
    # Initialize unified features
    df['scenario_flag'] = 0  # Default to S1-style
    
    # Time window flags
    df['is_early_post'] = (df['months_postgx'] >= 0) & (df['months_postgx'] <= 5)
    df['is_mid_post'] = (df['months_postgx'] >= 6) & (df['months_postgx'] <= 11)
    df['is_late_post'] = (df['months_postgx'] >= 12) & (df['months_postgx'] <= 23)
    df['is_pre_entry'] = df['months_postgx'] < 0
    
    # Early post-LOE summary features (computed per brand)
    # These will be filled only for S2-style samples
    df['early_post_mean'] = np.nan
    df['early_post_std'] = np.nan
    df['early_post_min'] = np.nan
    df['early_post_max'] = np.nan
    df['early_post_slope'] = np.nan
    df['early_post_last'] = np.nan
    df['early_erosion_ratio'] = np.nan
    
    # Compute early post-LOE features for each brand
    for (country, brand), group in df.groupby(['country', 'brand_name']):
        early_data = group[group['months_postgx'].between(0, 5)]['volume']
        avg_vol = group['avg_vol'].iloc[0]
        
        if len(early_data) >= 3:  # Need some early data
            # Summary statistics
            df.loc[group.index, 'early_post_mean'] = early_data.mean()
            df.loc[group.index, 'early_post_std'] = early_data.std()
            df.loc[group.index, 'early_post_min'] = early_data.min()
            df.loc[group.index, 'early_post_max'] = early_data.max()
            df.loc[group.index, 'early_post_last'] = early_data.iloc[-1] if len(early_data) > 0 else np.nan
            
            # Slope (linear fit)
            if len(early_data) >= 2:
                x = np.arange(len(early_data))
                try:
                    slope = np.polyfit(x, early_data.values, 1)[0]
                    df.loc[group.index, 'early_post_slope'] = slope
                except:
                    pass
            
            # Erosion ratio (mean early volume / avg_vol)
            if avg_vol > 0:
                df.loc[group.index, 'early_erosion_ratio'] = early_data.mean() / avg_vol
    
    return df


def create_unified_training_data(
    df: pd.DataFrame,
    avg_j: pd.DataFrame,
    include_s2_style: bool = True
) -> pd.DataFrame:
    """
    Create unified training dataset with both S1-style and S2-style samples.
    
    For each brand:
    - S1-style samples: months 0-23, early post features = 0/NaN
    - S2-style samples: months 6-23, early post features filled from months 0-5
    
    Args:
        df: DataFrame with features
        avg_j: DataFrame with avg_vol
        include_s2_style: Whether to include S2-style samples (default True)
        
    Returns:
        Concatenated DataFrame with scenario_flag differentiating styles
    """
    df = add_unified_features(df, avg_j)
    
    # S1-style samples: all months 0-23, early post features zeroed
    s1_data = df[df['months_postgx'].between(0, 23)].copy()
    s1_data['scenario_flag'] = 0
    
    # Zero out early post-LOE features for S1-style
    early_post_cols = ['early_post_mean', 'early_post_std', 'early_post_min', 
                       'early_post_max', 'early_post_slope', 'early_post_last',
                       'early_erosion_ratio']
    for col in early_post_cols:
        if col in s1_data.columns:
            s1_data[col] = 0
    
    if not include_s2_style:
        return s1_data
    
    # S2-style samples: months 6-23, early post features populated
    s2_data = df[df['months_postgx'].between(6, 23)].copy()
    s2_data['scenario_flag'] = 1
    # Early post features already computed in add_unified_features
    
    # Concatenate
    unified_data = pd.concat([s1_data, s2_data], ignore_index=True)
    
    return unified_data


def train_unified_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: List[str],
    avg_j: pd.DataFrame,
    aux_df: pd.DataFrame,
    val_actual_full: pd.DataFrame,
    best_decay_rate: float,
    **kwargs
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a single unified model and evaluate on both S1 and S2 metrics.
    
    Args:
        model_type: 'lightgbm', 'xgboost', 'hybrid_lightgbm', or 'hybrid_xgboost'
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_cols: Feature column names
        avg_j: Average volume per brand
        aux_df: Auxiliary data with buckets
        val_actual_full: Full validation actuals (months 0-23)
        best_decay_rate: Tuned decay rate for hybrid models
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    if model_type == 'lightgbm':
        model = GradientBoostingModel(model_type='lightgbm')
        model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_val)
        
    elif model_type == 'xgboost':
        model = GradientBoostingModel(model_type='xgboost')
        model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_val)
        
    elif model_type == 'hybrid_lightgbm':
        avg_vol_train = kwargs.get('avg_vol_train')
        months_train = kwargs.get('months_train')
        avg_vol_val = kwargs.get('avg_vol_val')
        months_val = kwargs.get('months_val')
        
        model = HybridPhysicsMLModel(ml_model_type='lightgbm', decay_rate=best_decay_rate)
        model.fit(X_train, y_train, avg_vol_train, months_train,
                  X_val, y_val, avg_vol_val, months_val)
        predictions = model.predict(X_val, avg_vol_val, months_val)
        
    elif model_type == 'hybrid_xgboost':
        avg_vol_train = kwargs.get('avg_vol_train')
        months_train = kwargs.get('months_train')
        avg_vol_val = kwargs.get('avg_vol_val')
        months_val = kwargs.get('months_val')
        
        model = HybridPhysicsMLModel(ml_model_type='xgboost', decay_rate=best_decay_rate)
        model.fit(X_train, y_train, avg_vol_train, months_train,
                  X_val, y_val, avg_vol_val, months_val)
        predictions = model.predict(X_val, avg_vol_val, months_val)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, predictions


def train_unified(test_mode: bool = False) -> Dict[str, Any]:
    """
    Main entry point for unified training mode.
    
    Trains ONE global model that handles both S1 and S2 scenarios.
    Uses scenario_flag feature and early post-LOE summaries.
    
    Args:
        test_mode: Use subset of data for quick testing
        
    Returns:
        Dictionary with results
    """
    # Show which models are enabled
    enabled_models = [k for k, v in MODELS_ENABLED.items() if v]
    disabled_models = [k for k, v in MODELS_ENABLED.items() if not v]
    
    print("\n" + "=" * 70)
    print("🔀 TRAINING MODE: UNIFIED (Single model for S1 & S2)")
    print("=" * 70)
    print(f"\n🤖 Models to train ({len(enabled_models)}):")
    for m in enabled_models:
        print(f"   ✅ {m}")
    if disabled_models:
        print(f"\n⏭️ Skipped models ({len(disabled_models)}):")
        for m in disabled_models:
            print(f"   ❌ {m}")
    
    # Load data once
    print("\n📂 Loading data...")
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    
    if test_mode:
        brands = merged[['country', 'brand_name']].drop_duplicates().head(TEST_MODE_BRANDS)
        merged = merged.merge(brands, on=['country', 'brand_name'])
        print(f"⚠️ TEST MODE: Using {len(brands)} brands")
    
    # Create auxiliary file (needed for official metric calculation)
    aux_df = create_auxiliary_file(merged, save=True)
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    # aux_df has: country, brand_name, bucket, avg_vol - used directly by official metrics
    
    # Feature engineering
    print("\n🔧 Feature engineering...")
    featured = create_all_features(merged, avg_j)
    
    # Create unified training data
    print("\n🔀 Creating unified training data (S1 + S2 style samples)...")
    unified_data = create_unified_training_data(featured, avg_j, include_s2_style=True)
    
    print(f"   Total samples: {len(unified_data)}")
    print(f"   S1-style: {(unified_data['scenario_flag'] == 0).sum()}")
    print(f"   S2-style: {(unified_data['scenario_flag'] == 1).sum()}")
    
    # Get unified feature columns (include new features)
    base_feature_cols = get_feature_columns(featured)
    unified_feature_cols = base_feature_cols + [
        'scenario_flag',
        'is_early_post', 'is_mid_post', 'is_late_post', 'is_pre_entry',
        'early_post_mean', 'early_post_std', 'early_post_min', 'early_post_max',
        'early_post_slope', 'early_post_last', 'early_erosion_ratio'
    ]
    # Filter to existing columns
    unified_feature_cols = [c for c in unified_feature_cols if c in unified_data.columns]
    
    # Split by brand (GroupKFold-like)
    train_df, val_df = split_train_validation(unified_data)
    
    X_train, y_train = prepare_training_data(train_df, unified_feature_cols)
    X_val, y_val = prepare_training_data(val_df, unified_feature_cols)
    
    # Get validation actuals for both scenarios
    val_brands = val_df[['country', 'brand_name']].drop_duplicates()
    val_avg_j = avg_j.merge(val_brands, on=['country', 'brand_name'])
    
    # Full validation actuals (all months 0-23)
    val_actual_full = featured.merge(val_brands, on=['country', 'brand_name'])
    val_actual_full = val_actual_full[val_actual_full['months_postgx'].between(0, 23)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ].copy()
    
    # Track results
    all_results = []
    model_names = []
    models_trained = {}
    
    # =========================================================================
    # 1. Tune decay rate for baselines/hybrid
    # =========================================================================
    print("\n📊 Tuning decay rate...")
    train_brands = train_df[train_df['scenario_flag'] == 0][['country', 'brand_name']].drop_duplicates()
    train_avg_j_subset = avg_j.merge(train_brands, on=['country', 'brand_name'])
    train_actual = train_df[(train_df['scenario_flag'] == 0) & train_df['months_postgx'].between(0, 23)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ]
    
    best_rate, _ = BaselineModels.tune_decay_rate(train_actual, train_avg_j_subset, 'exponential')
    print(f"   Best decay rate: {best_rate:.4f}")
    
    # =========================================================================
    # 2. Baseline: Exponential Decay (evaluated on both S1 and S2)
    # =========================================================================
    if MODELS_ENABLED.get('baseline_exp_decay', True):
        print("\n📊 Evaluating: Baseline - Exponential Decay (Unified)")
        
        # Generate predictions for all months 0-23
        pred_baseline = BaselineModels.exponential_decay(val_avg_j, list(range(0, 24)), best_rate)
        pred_baseline.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        # Evaluate S1 metric (official)
        s1_score = compute_metric1(val_actual_full, pred_baseline, aux_df)
        
        # Evaluate S2 metric (months 6-23, official)
        val_actual_s2 = val_actual_full[val_actual_full['months_postgx'] >= 6]
        pred_baseline_s2 = pred_baseline[pred_baseline['months_postgx'] >= 6]
        s2_score = compute_metric2(val_actual_s2, pred_baseline_s2, aux_df)
        
        combined = 0.5 * s1_score + 0.5 * s2_score
        
        print(f"   S1 Score: {s1_score:.4f}")
        print(f"   S2 Score: {s2_score:.4f}")
        print(f"   Combined: {combined:.4f}")
        
        all_results.append({
            'model': f"Baseline-ExpDecay({best_rate:.3f})",
            'final_score': combined,
            's1_score': s1_score,
            's2_score': s2_score,
        })
        model_names.append(f"Baseline-ExpDecay({best_rate:.3f})")
    
    # =========================================================================
    # 3. LightGBM (Unified)
    # =========================================================================
    lgbm_model = None
    if MODELS_ENABLED.get('lightgbm', True):
        print("\n📊 Training: LightGBM (Unified)")
        
        lgbm_model = GradientBoostingModel(model_type='lightgbm')
        lgbm_model.fit(X_train, y_train, X_val, y_val)
        
        # Predict for full validation (all months 0-23, S1-style)
        val_pred_data = featured.merge(val_brands, on=['country', 'brand_name'])
        val_pred_data = val_pred_data[val_pred_data['months_postgx'].between(0, 23)].copy()
        val_pred_data = add_unified_features(val_pred_data, avg_j)
        val_pred_data['scenario_flag'] = 0  # S1-style for evaluation
        
        # Zero early post features for S1-style
        early_cols = ['early_post_mean', 'early_post_std', 'early_post_min', 
                     'early_post_max', 'early_post_slope', 'early_post_last', 'early_erosion_ratio']
        for col in early_cols:
            if col in val_pred_data.columns:
                val_pred_data[col] = 0
        
        X_val_pred = val_pred_data[unified_feature_cols].fillna(0)
        val_pred_data['volume_pred'] = lgbm_model.predict(X_val_pred)
        
        pred_lgbm = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_lgbm.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        # Evaluate S1 (official)
        s1_score = compute_metric1(val_actual_full, pred_lgbm, aux_df)
        
        # For S2, use S2-style features (scenario_flag=1 with early post features)
        val_pred_s2 = featured.merge(val_brands, on=['country', 'brand_name'])
        val_pred_s2 = val_pred_s2[val_pred_s2['months_postgx'].between(6, 23)].copy()
        val_pred_s2 = add_unified_features(val_pred_s2, avg_j)
        val_pred_s2['scenario_flag'] = 1  # S2-style
        
        X_val_s2 = val_pred_s2[unified_feature_cols].fillna(0)
        val_pred_s2['volume_pred'] = lgbm_model.predict(X_val_s2)
        
        pred_lgbm_s2 = val_pred_s2[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_lgbm_s2.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        val_actual_s2 = val_actual_full[val_actual_full['months_postgx'] >= 6]
        s2_score = compute_metric2(val_actual_s2, pred_lgbm_s2, aux_df)
        
        combined = 0.5 * s1_score + 0.5 * s2_score
        
        print(f"   S1 Score: {s1_score:.4f}")
        print(f"   S2 Score: {s2_score:.4f}")
        print(f"   Combined: {combined:.4f}")
        
        all_results.append({
            'model': "LightGBM-Unified",
            'final_score': combined,
            's1_score': s1_score,
            's2_score': s2_score,
        })
        model_names.append("LightGBM-Unified")
        
        # Save model (with config prefix in multi-config mode)
        lgbm_model.save(config.get_model_filename("unified_lightgbm"))
        models_trained['lightgbm'] = lgbm_model
    
    # =========================================================================
    # 4. XGBoost (Unified)
    # =========================================================================
    if MODELS_ENABLED.get('xgboost', True):
        print("\n📊 Training: XGBoost (Unified)")
        
        xgb_model = GradientBoostingModel(model_type='xgboost')
        xgb_model.fit(X_train, y_train, X_val, y_val)
        
        # Similar evaluation pattern as LightGBM
        val_pred_data = featured.merge(val_brands, on=['country', 'brand_name'])
        val_pred_data = val_pred_data[val_pred_data['months_postgx'].between(0, 23)].copy()
        val_pred_data = add_unified_features(val_pred_data, avg_j)
        val_pred_data['scenario_flag'] = 0
        
        for col in ['early_post_mean', 'early_post_std', 'early_post_min', 
                   'early_post_max', 'early_post_slope', 'early_post_last', 'early_erosion_ratio']:
            if col in val_pred_data.columns:
                val_pred_data[col] = 0
        
        X_val_pred = val_pred_data[unified_feature_cols].fillna(0)
        val_pred_data['volume_pred'] = xgb_model.predict(X_val_pred)
        
        pred_xgb = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_xgb.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        s1_score = compute_metric1(val_actual_full, pred_xgb, aux_df)
        
        val_pred_s2 = featured.merge(val_brands, on=['country', 'brand_name'])
        val_pred_s2 = val_pred_s2[val_pred_s2['months_postgx'].between(6, 23)].copy()
        val_pred_s2 = add_unified_features(val_pred_s2, avg_j)
        val_pred_s2['scenario_flag'] = 1
        
        X_val_s2 = val_pred_s2[unified_feature_cols].fillna(0)
        val_pred_s2['volume_pred'] = xgb_model.predict(X_val_s2)
        
        pred_xgb_s2 = val_pred_s2[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_xgb_s2.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        val_actual_s2 = val_actual_full[val_actual_full['months_postgx'] >= 6]
        s2_score = compute_metric2(val_actual_s2, pred_xgb_s2, aux_df)
        
        combined = 0.5 * s1_score + 0.5 * s2_score
        
        print(f"   S1 Score: {s1_score:.4f}")
        print(f"   S2 Score: {s2_score:.4f}")
        print(f"   Combined: {combined:.4f}")
        
        all_results.append({
            'model': "XGBoost-Unified",
            'final_score': combined,
            's1_score': s1_score,
            's2_score': s2_score,
        })
        model_names.append("XGBoost-Unified")
        
        # Save model (with config prefix in multi-config mode)
        xgb_model.save(config.get_model_filename("unified_xgboost"))
        models_trained['xgboost'] = xgb_model
    
    # =========================================================================
    # 5. Hybrid LightGBM (Unified)
    # =========================================================================
    if MODELS_ENABLED.get('hybrid_lightgbm', True):
        print("\n📊 Training: Hybrid LightGBM (Unified)")
        
        # Prepare data for hybrid model
        train_pred_data = train_df.copy()
        if 'avg_vol' not in train_pred_data.columns:
            train_pred_data = train_pred_data.merge(avg_j, on=['country', 'brand_name'], how='left')
        
        X_train_hyb = train_pred_data[unified_feature_cols].fillna(0)
        y_train_hyb = train_pred_data['volume']
        avg_vol_train = train_pred_data['avg_vol'].fillna(train_pred_data['avg_vol'].median()).values
        months_train = train_pred_data['months_postgx'].values
        
        val_pred_hyb = val_df.copy()
        if 'avg_vol' not in val_pred_hyb.columns:
            val_pred_hyb = val_pred_hyb.merge(avg_j, on=['country', 'brand_name'], how='left')
        
        X_val_hyb = val_pred_hyb[unified_feature_cols].fillna(0)
        y_val_hyb = val_pred_hyb['volume']
        avg_vol_val = val_pred_hyb['avg_vol'].fillna(val_pred_hyb['avg_vol'].median()).values
        months_val = val_pred_hyb['months_postgx'].values
        
        hybrid_model = HybridPhysicsMLModel(ml_model_type='lightgbm', decay_rate=best_rate)
        hybrid_model.fit(X_train_hyb, y_train_hyb, avg_vol_train, months_train,
                        X_val_hyb, y_val_hyb, avg_vol_val, months_val)
        
        # Evaluate on S1 (all months)
        val_pred_data = featured.merge(val_brands, on=['country', 'brand_name'])
        val_pred_data = val_pred_data[val_pred_data['months_postgx'].between(0, 23)].copy()
        val_pred_data = add_unified_features(val_pred_data, avg_j)
        val_pred_data['scenario_flag'] = 0
        for col in ['early_post_mean', 'early_post_std', 'early_post_min', 
                   'early_post_max', 'early_post_slope', 'early_post_last', 'early_erosion_ratio']:
            if col in val_pred_data.columns:
                val_pred_data[col] = 0
        
        if 'avg_vol' not in val_pred_data.columns:
            val_pred_data = val_pred_data.merge(avg_j, on=['country', 'brand_name'], how='left')
        
        X_pred = val_pred_data[unified_feature_cols].fillna(0)
        avg_vol_pred = val_pred_data['avg_vol'].fillna(val_pred_data['avg_vol'].median()).values
        months_pred = val_pred_data['months_postgx'].values
        
        val_pred_data['volume_pred'] = hybrid_model.predict(X_pred, avg_vol_pred, months_pred)
        pred_hyb = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_hyb.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        s1_score = compute_metric1(val_actual_full, pred_hyb, aux_df)
        
        # Evaluate on S2
        val_pred_s2 = featured.merge(val_brands, on=['country', 'brand_name'])
        val_pred_s2 = val_pred_s2[val_pred_s2['months_postgx'].between(6, 23)].copy()
        val_pred_s2 = add_unified_features(val_pred_s2, avg_j)
        val_pred_s2['scenario_flag'] = 1
        
        if 'avg_vol' not in val_pred_s2.columns:
            val_pred_s2 = val_pred_s2.merge(avg_j, on=['country', 'brand_name'], how='left')
        
        X_s2 = val_pred_s2[unified_feature_cols].fillna(0)
        avg_vol_s2 = val_pred_s2['avg_vol'].fillna(val_pred_s2['avg_vol'].median()).values
        months_s2 = val_pred_s2['months_postgx'].values
        
        val_pred_s2['volume_pred'] = hybrid_model.predict(X_s2, avg_vol_s2, months_s2)
        pred_hyb_s2 = val_pred_s2[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_hyb_s2.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        val_actual_s2 = val_actual_full[val_actual_full['months_postgx'] >= 6]
        s2_score = compute_metric2(val_actual_s2, pred_hyb_s2, aux_df)
        
        combined = 0.5 * s1_score + 0.5 * s2_score
        
        print(f"   S1 Score: {s1_score:.4f}")
        print(f"   S2 Score: {s2_score:.4f}")
        print(f"   Combined: {combined:.4f}")
        
        all_results.append({
            'model': "Hybrid-LightGBM-Unified",
            'final_score': combined,
            's1_score': s1_score,
            's2_score': s2_score,
        })
        model_names.append("Hybrid-LightGBM-Unified")
        
        # Save model (with config prefix in multi-config mode)
        hybrid_model.save(config.get_model_filename("unified_hybrid"))
        models_trained['hybrid_lightgbm'] = hybrid_model
    
    # =========================================================================
    # 5. Hybrid XGBoost Model (Unified)
    # =========================================================================
    if MODELS_ENABLED.get('hybrid_xgboost', True):
        print("\n📊 Training: Hybrid XGBoost (Unified)")
        
        # Prepare data for hybrid model
        train_pred_data = train_df.copy()
        if 'avg_vol' not in train_pred_data.columns:
            train_pred_data = train_pred_data.merge(avg_j, on=['country', 'brand_name'], how='left')
        
        X_train_hyb = train_pred_data[unified_feature_cols].fillna(0)
        y_train_hyb = train_pred_data['volume']
        avg_vol_train = train_pred_data['avg_vol'].fillna(train_pred_data['avg_vol'].median()).values
        months_train = train_pred_data['months_postgx'].values
        
        val_pred_hyb = val_df.copy()
        if 'avg_vol' not in val_pred_hyb.columns:
            val_pred_hyb = val_pred_hyb.merge(avg_j, on=['country', 'brand_name'], how='left')
        
        X_val_hyb = val_pred_hyb[unified_feature_cols].fillna(0)
        y_val_hyb = val_pred_hyb['volume']
        avg_vol_val = val_pred_hyb['avg_vol'].fillna(val_pred_hyb['avg_vol'].median()).values
        months_val = val_pred_hyb['months_postgx'].values
        
        hybrid_xgb_model = HybridPhysicsMLModel(ml_model_type='xgboost', decay_rate=best_rate)
        hybrid_xgb_model.fit(X_train_hyb, y_train_hyb, avg_vol_train, months_train,
                            X_val_hyb, y_val_hyb, avg_vol_val, months_val)
        
        # Evaluate on S1 (all months)
        val_pred_data = featured.merge(val_brands, on=['country', 'brand_name'])
        val_pred_data = val_pred_data[val_pred_data['months_postgx'].between(0, 23)].copy()
        val_pred_data = add_unified_features(val_pred_data, avg_j)
        val_pred_data['scenario_flag'] = 0
        for col in ['early_post_mean', 'early_post_std', 'early_post_min', 
                   'early_post_max', 'early_post_slope', 'early_post_last', 'early_erosion_ratio']:
            if col in val_pred_data.columns:
                val_pred_data[col] = 0
        
        if 'avg_vol' not in val_pred_data.columns:
            val_pred_data = val_pred_data.merge(avg_j, on=['country', 'brand_name'], how='left')
        
        X_pred = val_pred_data[unified_feature_cols].fillna(0)
        avg_vol_pred = val_pred_data['avg_vol'].fillna(val_pred_data['avg_vol'].median()).values
        months_pred = val_pred_data['months_postgx'].values
        
        val_pred_data['volume_pred'] = hybrid_xgb_model.predict(X_pred, avg_vol_pred, months_pred)
        pred_hyb = val_pred_data[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_hyb.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        s1_score = compute_metric1(val_actual_full, pred_hyb, aux_df)
        
        # Evaluate on S2
        val_pred_s2 = featured.merge(val_brands, on=['country', 'brand_name'])
        val_pred_s2 = val_pred_s2[val_pred_s2['months_postgx'].between(6, 23)].copy()
        val_pred_s2 = add_unified_features(val_pred_s2, avg_j)
        val_pred_s2['scenario_flag'] = 1
        
        if 'avg_vol' not in val_pred_s2.columns:
            val_pred_s2 = val_pred_s2.merge(avg_j, on=['country', 'brand_name'], how='left')
        
        X_s2 = val_pred_s2[unified_feature_cols].fillna(0)
        avg_vol_s2 = val_pred_s2['avg_vol'].fillna(val_pred_s2['avg_vol'].median()).values
        months_s2 = val_pred_s2['months_postgx'].values
        
        val_pred_s2['volume_pred'] = hybrid_xgb_model.predict(X_s2, avg_vol_s2, months_s2)
        pred_hyb_s2 = val_pred_s2[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
        pred_hyb_s2.columns = ['country', 'brand_name', 'months_postgx', 'volume']
        
        val_actual_s2 = val_actual_full[val_actual_full['months_postgx'] >= 6]
        s2_score = compute_metric2(val_actual_s2, pred_hyb_s2, aux_df)
        
        combined = 0.5 * s1_score + 0.5 * s2_score
        
        print(f"   S1 Score: {s1_score:.4f}")
        print(f"   S2 Score: {s2_score:.4f}")
        print(f"   Combined: {combined:.4f}")
        
        all_results.append({
            'model': "Hybrid-XGBoost-Unified",
            'final_score': combined,
            's1_score': s1_score,
            's2_score': s2_score,
        })
        model_names.append("Hybrid-XGBoost-Unified")
        
        # Save model (with config prefix in multi-config mode)
        hybrid_xgb_model.save(config.get_model_filename("unified_hybrid_xgboost"))
        models_trained['hybrid_xgboost'] = hybrid_xgb_model
    
    # =========================================================================
    # Create comparison DataFrame
    # =========================================================================
    if len(all_results) == 0:
        print("\n⚠️ No models were trained!")
        return {}
    
    comparison_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("📊 UNIFIED MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    
    # Best model
    best_idx = comparison_df['final_score'].idxmin()
    best_model = comparison_df.loc[best_idx, 'model']
    best_score = comparison_df.loc[best_idx, 'final_score']
    
    print(f"\n🏆 BEST UNIFIED MODEL: {best_model} (Combined Score: {best_score:.4f})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_df.to_csv(REPORTS_DIR / f"model_comparison_unified_{timestamp}.csv", index=False)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison_unified.csv", index=False)
    
    return {
        'comparison_df': comparison_df,
        'best_model': best_model,
        'best_score': best_score,
        'best_decay_rate': best_rate,
        'models': models_trained,
        'unified_feature_cols': unified_feature_cols
    }


if __name__ == "__main__":
    from config import TEST_MODE
    train_unified(test_mode=TEST_MODE)
