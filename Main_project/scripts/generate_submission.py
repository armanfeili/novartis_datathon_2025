# =============================================================================
# File: scripts/generate_submission.py
# Description: Generate submission files for competition
#
# The template expects ONE unified CSV with:
#   - 228 S1 brands × 24 months (months 0-23) = 5,472 rows
#   - 112 S2 brands × 18 months (months 6-23) = 2,016 rows
#   - Total: 7,488 rows
#
# Usage:
#   python scripts/generate_submission.py
#
# All settings come from config.py
# =============================================================================

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config  # Import module for live access (multi-config)
from config import *
from data_loader import load_all_data, merge_datasets
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns
from models import BaselineModels, GradientBoostingModel, HybridPhysicsMLModel


# =============================================================================
# TEMPLATE UTILITIES
# =============================================================================

def load_template() -> pd.DataFrame:
    """Load the official submission template."""
    template_path = SUBMISSIONS_DIR / "submission_template.csv"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_path}\n"
            "Please ensure submission_template.csv is in the submissions folder."
        )
    return pd.read_csv(template_path)


def determine_brand_scenarios(template: pd.DataFrame) -> dict:
    """
    Determine which scenario each brand belongs to based on template.
    
    S1 brands: months start at 0 (predict months 0-23)
    S2 brands: months start at 6 (predict months 6-23)
    
    Returns:
        dict mapping (country, brand_name) -> scenario (1 or 2)
    """
    brand_scenarios = {}
    for (country, brand), group in template.groupby(['country', 'brand_name']):
        min_month = group['months_postgx'].min()
        scenario = 1 if min_month == 0 else 2
        brand_scenarios[(country, brand)] = scenario
    return brand_scenarios


# =============================================================================
# PREDICTION GENERATORS
# =============================================================================

def generate_baseline_predictions(test_avg_j: pd.DataFrame, 
                                   brand_scenarios: dict,
                                   decay_rate: float = None) -> pd.DataFrame:
    """
    Generate exponential decay predictions for all test brands.
    
    Args:
        test_avg_j: DataFrame with avg_vol per brand
        brand_scenarios: dict mapping (country, brand) -> scenario
        decay_rate: Decay rate (uses config default if None)
        
    Returns:
        DataFrame with predictions matching template structure
    """
    if decay_rate is None:
        decay_rate = DEFAULT_DECAY_RATE
    
    predictions = []
    for _, row in test_avg_j.iterrows():
        country = row['country']
        brand = row['brand_name']
        avg_vol = row['avg_vol']
        
        key = (country, brand)
        scenario = brand_scenarios.get(key, 1)
        
        if scenario == 1:
            months = list(range(0, 24))
        else:
            months = list(range(6, 24))
        
        for month in months:
            volume = avg_vol * np.exp(-decay_rate * month)
            predictions.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'volume': max(volume, 0)
            })
    
    return pd.DataFrame(predictions)


def generate_ml_predictions(test_data: pd.DataFrame,
                            test_avg_j: pd.DataFrame,
                            brand_scenarios: dict,
                            model_type: str = 'lightgbm') -> pd.DataFrame:
    """
    Generate ML model predictions for all test brands.
    
    Supports both training modes from config.TRAIN_MODE:
    - "separate": Loads scenario1_{model} and scenario2_{model}
    - "unified": Loads unified_{model} with scenario_flag feature
    
    Args:
        test_data: Merged test data with features
        test_avg_j: DataFrame with avg_vol per brand
        brand_scenarios: dict mapping (country, brand) -> scenario
        model_type: 'lightgbm', 'xgboost', or 'hybrid'
        
    Returns:
        DataFrame with predictions matching template structure
    """
    featured = create_all_features(test_data, test_avg_j)
    feature_cols = get_feature_columns(featured)
    
    all_predictions = []
    
    s1_brands = [(c, b) for (c, b), s in brand_scenarios.items() if s == 1]
    s2_brands = [(c, b) for (c, b), s in brand_scenarios.items() if s == 2]
    
    # Check training mode
    if TRAIN_MODE == "unified":
        # Load single unified model
        model_name = f"unified_{model_type}"
        print(f"   Using UNIFIED model: {model_name}")
        
        try:
            if model_type == 'hybrid':
                model = HybridPhysicsMLModel(ml_model_type='lightgbm')
            else:
                model = GradientBoostingModel(model_type=model_type)
            model.load(model_name)
            
            # Add unified features (scenario_flag, early_post features)
            from training.train_unified import add_unified_features
            
            # Get unified feature columns
            unified_feature_cols = feature_cols + [
                'scenario_flag',
                'is_early_post', 'is_mid_post', 'is_late_post', 'is_pre_entry',
                'early_post_mean', 'early_post_std', 'early_post_min', 'early_post_max',
                'early_post_slope', 'early_post_last', 'early_erosion_ratio'
            ]
            unified_feature_cols = [c for c in unified_feature_cols if c in model.feature_names] if hasattr(model, 'feature_names') else unified_feature_cols
            
            for scenario, brands in [(1, s1_brands), (2, s2_brands)]:
                if not brands:
                    continue
                
                months_to_predict = list(range(0, 24)) if scenario == 1 else list(range(6, 24))
                
                brand_df = pd.DataFrame(brands, columns=['country', 'brand_name'])
                scenario_data = featured.merge(brand_df, on=['country', 'brand_name'])
                pred_data = scenario_data[scenario_data['months_postgx'].isin(months_to_predict)].copy()
                
                if len(pred_data) == 0:
                    print(f"   ⚠️ No data for S{scenario}, using baseline")
                    scenario_avg_j = test_avg_j.merge(brand_df, on=['country', 'brand_name'])
                    baseline_preds = BaselineModels.exponential_decay(
                        scenario_avg_j, months_to_predict, DEFAULT_DECAY_RATE
                    )
                    all_predictions.append(baseline_preds)
                    continue
                
                # Add unified features
                pred_data = add_unified_features(pred_data, test_avg_j)
                pred_data['scenario_flag'] = 0 if scenario == 1 else 1
                
                # For S1, zero out early_post features (no actuals available)
                if scenario == 1:
                    early_cols = ['early_post_mean', 'early_post_std', 'early_post_min', 
                                 'early_post_max', 'early_post_slope', 'early_post_last', 'early_erosion_ratio']
                    for col in early_cols:
                        if col in pred_data.columns:
                            pred_data[col] = 0
                
                # Filter to available features
                available_cols = [c for c in unified_feature_cols if c in pred_data.columns]
                X = pred_data[available_cols].fillna(0)
                
                if model_type == 'hybrid':
                    pred_data = pred_data.merge(test_avg_j, on=['country', 'brand_name'], how='left', suffixes=('', '_y'))
                    avg_vol = pred_data['avg_vol'].fillna(pred_data['avg_vol'].median()).values
                    months = pred_data['months_postgx'].values
                    pred_data['volume'] = model.predict(X, avg_vol, months)
                else:
                    pred_data['volume'] = model.predict(X)
                
                preds = pred_data[['country', 'brand_name', 'months_postgx', 'volume']].copy()
                all_predictions.append(preds)
                
        except FileNotFoundError:
            print(f"   ⚠️ Unified model {model_name} not found, falling back to baseline")
            for scenario, brands in [(1, s1_brands), (2, s2_brands)]:
                if not brands:
                    continue
                months_to_predict = list(range(0, 24)) if scenario == 1 else list(range(6, 24))
                brand_df = pd.DataFrame(brands, columns=['country', 'brand_name'])
                scenario_avg_j = test_avg_j.merge(brand_df, on=['country', 'brand_name'])
                baseline_preds = BaselineModels.exponential_decay(
                    scenario_avg_j, months_to_predict, DEFAULT_DECAY_RATE
                )
                all_predictions.append(baseline_preds)
    
    else:
        # SEPARATE mode: Load scenario-specific models
        print(f"   Using SEPARATE models: scenario1_{model_type}, scenario2_{model_type}")
        
        for scenario, brands in [(1, s1_brands), (2, s2_brands)]:
            if not brands:
                continue
                
            months_to_predict = list(range(0, 24)) if scenario == 1 else list(range(6, 24))
            
            brand_df = pd.DataFrame(brands, columns=['country', 'brand_name'])
            scenario_data = featured.merge(brand_df, on=['country', 'brand_name'])
            pred_data = scenario_data[scenario_data['months_postgx'].isin(months_to_predict)].copy()
            
            if len(pred_data) == 0:
                print(f"   ⚠️ No prediction data for Scenario {scenario}, using baseline")
                scenario_avg_j = test_avg_j.merge(brand_df, on=['country', 'brand_name'])
                baseline_preds = BaselineModels.exponential_decay(
                    scenario_avg_j, months_to_predict, DEFAULT_DECAY_RATE
                )
                all_predictions.append(baseline_preds)
                continue
            
            model_name = f"scenario{scenario}_{model_type}"
            if model_type == 'hybrid':
                model_name = f"scenario{scenario}_hybrid"
            
            try:
                if model_type == 'hybrid':
                    model = HybridPhysicsMLModel(ml_model_type='lightgbm')
                    model.load(model_name)
                    
                    pred_data = pred_data.merge(test_avg_j, on=['country', 'brand_name'], how='left')
                    X = pred_data[feature_cols].fillna(0)
                    avg_vol = pred_data['avg_vol'].fillna(pred_data['avg_vol'].median()).values
                    months = pred_data['months_postgx'].values
                    
                    pred_data['volume'] = model.predict(X, avg_vol, months)
                else:
                    model = GradientBoostingModel(model_type=model_type)
                    model.load(model_name)
                    
                    X = pred_data[feature_cols].fillna(0)
                    pred_data['volume'] = model.predict(X)
                
                preds = pred_data[['country', 'brand_name', 'months_postgx', 'volume']].copy()
                all_predictions.append(preds)
                
            except FileNotFoundError:
                print(f"   ⚠️ Model {model_name} not found, using baseline")
                scenario_avg_j = test_avg_j.merge(brand_df, on=['country', 'brand_name'])
                baseline_preds = BaselineModels.exponential_decay(
                    scenario_avg_j, months_to_predict, DEFAULT_DECAY_RATE
                )
                all_predictions.append(baseline_preds)
    
    return pd.concat(all_predictions, ignore_index=True)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_against_template(submission: pd.DataFrame, template: pd.DataFrame) -> bool:
    """Validate submission matches template exactly."""
    errors = []
    
    if len(submission) != len(template):
        errors.append(f"Row count: got {len(submission)}, expected {len(template)}")
    
    expected_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    if list(submission.columns) != expected_cols:
        errors.append(f"Columns: got {list(submission.columns)}, expected {expected_cols}")
    
    nan_count = submission['volume'].isna().sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values in volume")
    
    template_keys = set(zip(template['country'], template['brand_name'], template['months_postgx']))
    submission_keys = set(zip(submission['country'], submission['brand_name'], submission['months_postgx']))
    
    missing = template_keys - submission_keys
    extra = submission_keys - template_keys
    
    if missing:
        errors.append(f"Missing {len(missing)} (country, brand, month) combinations")
    if extra:
        errors.append(f"Extra {len(extra)} (country, brand, month) combinations")
    
    if errors:
        print("❌ Validation FAILED:")
        for err in errors:
            print(f"   {err}")
        return False
    
    print("✅ Validation PASSED!")
    return True


# =============================================================================
# SUMMARY / METADATA
# =============================================================================

def get_full_config() -> dict:
    """Get ALL configuration settings as a dictionary for reproducibility."""
    # Use the centralized config snapshot function
    return config.get_current_config_snapshot()


def save_submission_summary(submission: pd.DataFrame,
                            model_type: str,
                            filepath: Path,
                            decay_rate: float = None) -> Path:
    """Save a JSON summary alongside the submission file."""
    
    brand_starts = submission.groupby(['country', 'brand_name'])['months_postgx'].min()
    s1_count = (brand_starts == 0).sum()
    s2_count = (brand_starts == 6).sum()
    
    summary = {
        "submission_info": {
            "model_type": model_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": str(filepath.name),
        },
        "data_stats": {
            "total_rows": len(submission),
            "unique_brands": len(brand_starts),
            "s1_brands": int(s1_count),
            "s2_brands": int(s2_count),
        },
        "volume_stats": {
            "min": float(submission['volume'].min()),
            "max": float(submission['volume'].max()),
            "mean": float(submission['volume'].mean()),
            "median": float(submission['volume'].median()),
            "std": float(submission['volume'].std()),
        },
        "model_config": {
            "decay_rate": decay_rate,
        },
        "full_config": get_full_config(),
    }
    
    json_path = filepath.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return json_path


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_submission(model_type: str = 'baseline', 
                        decay_rate: float = None,
                        save: bool = True) -> pd.DataFrame:
    """
    Main function to generate unified submission.
    
    Supports both training modes from config.TRAIN_MODE:
    - "separate": Uses scenario1_{model} and scenario2_{model}
    - "unified": Uses unified_{model} with scenario_flag
    
    Args:
        model_type: 'baseline', 'lightgbm', 'xgboost', or 'hybrid'
        decay_rate: Decay rate for baseline (uses config default if None)
        save: Whether to save to file
        
    Returns:
        Submission DataFrame
    """
    print("=" * 70)
    print("📝 GENERATING SUBMISSION")
    print(f"   Model: {model_type}")
    print(f"   Train Mode: {TRAIN_MODE}")
    print("=" * 70)
    
    # Load template
    print("\n📂 Loading template...")
    template = load_template()
    print(f"   Template rows: {len(template)}")
    print(f"   Template brands: {template[['country', 'brand_name']].drop_duplicates().shape[0]}")
    
    # Determine brand scenarios from template
    brand_scenarios = determine_brand_scenarios(template)
    s1_count = sum(1 for s in brand_scenarios.values() if s == 1)
    s2_count = sum(1 for s in brand_scenarios.values() if s == 2)
    print(f"   S1 brands: {s1_count}, S2 brands: {s2_count}")
    
    # Load test data
    print("\n📂 Loading test data...")
    volume_test, generics_test, medicine_test = load_all_data(train=False)
    merged_test = merge_datasets(volume_test, generics_test, medicine_test)
    
    # Compute avg_vol from test pre-entry data
    print("   Computing avg_vol...")
    test_avg_j = compute_avg_j(merged_test)
    
    # Fill missing with median from training
    n_missing = test_avg_j['avg_vol'].isna().sum()
    if n_missing > 0:
        volume_train, _, _ = load_all_data(train=True)
        merged_train = merge_datasets(volume_train, generics_test, medicine_test)
        train_avg_j = compute_avg_j(merged_train)
        median_avg_vol = train_avg_j['avg_vol'].median()
        test_avg_j['avg_vol'] = test_avg_j['avg_vol'].fillna(median_avg_vol)
        print(f"   Filled {n_missing} missing avg_vol with median: {median_avg_vol:.2f}")
    
    # Generate predictions
    print(f"\n🔮 Generating {model_type} predictions...")
    
    if model_type == 'baseline':
        predictions = generate_baseline_predictions(test_avg_j, brand_scenarios, decay_rate)
    else:
        predictions = generate_ml_predictions(merged_test, test_avg_j, brand_scenarios, model_type)
    
    # Ensure correct column order and sort
    predictions = predictions[['country', 'brand_name', 'months_postgx', 'volume']]
    predictions = predictions.sort_values(['country', 'brand_name', 'months_postgx']).reset_index(drop=True)
    
    # Validate
    print("\n🔍 Validating submission...")
    validate_against_template(predictions, template)
    
    # Statistics
    print(f"\n📊 Submission Statistics:")
    print(f"   Total rows: {len(predictions)}")
    print(f"   Unique brands: {predictions[['country', 'brand_name']].drop_duplicates().shape[0]}")
    print(f"   Volume range: [{predictions['volume'].min():.2f}, {predictions['volume'].max():.2f}]")
    print(f"   Volume mean: {predictions['volume'].mean():.2f}")
    
    # Save
    if save:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")
        timestamp = f"{date_str}_{time_str}"
        
        # Save timestamped version (permanent record - no overwrites)
        ts_filename = config.get_submission_filename(model_type, timestamp)
        ts_filepath = SUBMISSIONS_DIR / ts_filename
        predictions.to_csv(ts_filepath, index=False)
        print(f"\n✅ Saved: {ts_filepath}")
        
        # Save JSON summary with full config (for timestamped version)
        json_path = save_submission_summary(predictions, model_type, ts_filepath, decay_rate)
        print(f"   Summary: {json_path.name}")
        print(f"   Config ID: {config.ACTIVE_CONFIG_ID}")
        
        # Also save/overwrite "latest" version for easy access
        latest_filename = config.get_submission_filename(model_type, "latest")
        latest_filepath = SUBMISSIONS_DIR / latest_filename
        predictions.to_csv(latest_filepath, index=False)
        print(f"   Latest: {latest_filepath.name}")
    
    return predictions


def generate_all_submissions():
    """Generate submissions for ALL enabled models in config.SUBMISSIONS_ENABLED."""
    
    enabled_models = [model for model, enabled in SUBMISSIONS_ENABLED.items() if enabled]
    
    if not enabled_models:
        print("⚠️ No models enabled in SUBMISSIONS_ENABLED!")
        print("   Edit src/config.py to enable at least one model.")
        return
    
    print("=" * 70)
    print(f"📝 GENERATING SUBMISSIONS FOR {len(enabled_models)} MODELS")
    print(f"   Models: {', '.join(enabled_models)}")
    print("=" * 70)
    
    for model_type in enabled_models:
        print(f"\n{'─' * 70}")
        try:
            generate_submission(model_type=model_type, save=True)
        except Exception as e:
            print(f"   ❌ Failed for {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"✅ ALL SUBMISSIONS COMPLETE")
    print("=" * 70)


def main():
    """Main entry point - uses config settings."""
    # Determine which model to use (first enabled in SUBMISSIONS_ENABLED)
    model_type = 'baseline'
    for model, enabled in SUBMISSIONS_ENABLED.items():
        if enabled:
            model_type = model
            break
    
    generate_submission(
        model_type=model_type,
        decay_rate=DEFAULT_DECAY_RATE,
        save=True
    )


if __name__ == "__main__":
    main()
