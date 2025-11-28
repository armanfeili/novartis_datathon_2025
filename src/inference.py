"""
Inference and submission generation for Novartis Datathon 2025.

Handles:
- Test scenario detection
- Prediction generation with inverse transform
- Edge case handling
- Submission file validation
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from .utils import load_config, setup_logging, timer, get_project_root
from .data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values, get_panel
from .features import make_features, get_feature_columns, get_features
from .train import get_feature_matrix_and_meta

logger = logging.getLogger(__name__)


def detect_test_scenarios(test_volume: pd.DataFrame) -> Dict[int, List[Tuple[str, str]]]:
    """
    Identify which test series belong to Scenario 1 vs 2.
    
    Detection rules (Heuristic):
        - Scenario 1: Series starting at months_postgx = 0 (no actuals before)
        - Scenario 2: Series starting at months_postgx = 6 (has months 0-5 actuals)
        - Must validate against expected counts (228 S1, 112 S2).
    
    Args:
        test_volume: Test volume DataFrame with columns [country, brand_name, months_postgx, ...]
        
    Returns:
        {1: list of (country, brand_name) tuples,
         2: list of (country, brand_name) tuples}
    """
    # Expected counts from competition documentation
    EXPECTED_S1_COUNT = 228
    EXPECTED_S2_COUNT = 112
    
    series_keys = ['country', 'brand_name']
    
    # Get min months_postgx per series
    series_min_month = test_volume.groupby(series_keys)['months_postgx'].min().reset_index()
    series_min_month.columns = series_keys + ['min_month']
    
    # Scenario 1: series that need predictions from month 0
    # Scenario 2: series that need predictions from month 6 (have months 0-5)
    
    # Get max months_postgx to understand the series range
    series_max_month = test_volume.groupby(series_keys)['months_postgx'].max().reset_index()
    series_max_month.columns = series_keys + ['max_month']
    
    series_info = series_min_month.merge(series_max_month, on=series_keys)
    
    # Heuristic: if min_month is 0 and we need to predict 0-23, it's Scenario 1
    # If min_month is 6, it's Scenario 2 (first 6 months given as features)
    
    # Actually, for submission we need to detect based on what predictions are needed
    # Let's check if there's volume data for months 0-5
    
    has_early_months = test_volume[test_volume['months_postgx'].between(0, 5)]
    series_with_early = has_early_months[series_keys].drop_duplicates()
    series_with_early['has_early'] = True
    
    series_info = series_info.merge(series_with_early, on=series_keys, how='left')
    # Convert NaN to False properly without FutureWarning
    series_info['has_early'] = series_info['has_early'].isna().apply(lambda x: not x)
    
    # Scenario 2: has months 0-5 data
    # Scenario 1: does not have months 0-5 data
    scenario2_series = series_info[series_info['has_early']][series_keys]
    scenario1_series = series_info[~series_info['has_early']][series_keys]
    
    result = {
        1: list(scenario1_series.itertuples(index=False, name=None)),
        2: list(scenario2_series.itertuples(index=False, name=None))
    }
    
    n_s1 = len(result[1])
    n_s2 = len(result[2])
    
    logger.info(f"Detected {n_s1} Scenario 1 series")
    logger.info(f"Detected {n_s2} Scenario 2 series")
    
    # Warn if counts differ from expected
    if n_s1 != EXPECTED_S1_COUNT:
        logger.warning(
            f"Scenario 1 count mismatch: detected {n_s1}, expected {EXPECTED_S1_COUNT}. "
            "This may indicate changes in test data or detection logic issues."
        )
    if n_s2 != EXPECTED_S2_COUNT:
        logger.warning(
            f"Scenario 2 count mismatch: detected {n_s2}, expected {EXPECTED_S2_COUNT}. "
            "This may indicate changes in test data or detection logic issues."
        )
    
    total = n_s1 + n_s2
    expected_total = EXPECTED_S1_COUNT + EXPECTED_S2_COUNT
    if total != expected_total:
        logger.warning(
            f"Total series count mismatch: detected {total}, expected {expected_total}"
        )
    
    return result


def generate_submission(
    model_scenario1: Any,
    model_scenario2: Any,
    test_panel: pd.DataFrame,
    submission_template: pd.DataFrame,
    feature_cols_s1: Optional[List[str]] = None,
    feature_cols_s2: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate final submission file.
    
    Args:
        model_scenario1: Trained model for Scenario 1
        model_scenario2: Trained model for Scenario 2
        test_panel: Pre-processed test panel (must have avg_vol_12m computed)
        submission_template: Template DataFrame with required rows
        feature_cols_s1: Feature columns for Scenario 1 (must match training)
        feature_cols_s2: Feature columns for Scenario 2 (must match training)
    
    CRITICAL: Models output normalized volume (y_norm).
    Must inverse transform: volume = y_norm * avg_vol_12m
    
    Returns:
        Submission DataFrame with columns [country, brand_name, months_postgx, volume]
    """
    with timer("Generate submission"):
        # Detect scenarios
        scenario_split = detect_test_scenarios(test_panel)
        
        predictions = []
        
        # Process Scenario 1 series
        if len(scenario_split[1]) > 0:
            s1_series = pd.DataFrame(scenario_split[1], columns=['country', 'brand_name'])
            s1_panel = test_panel.merge(s1_series, on=['country', 'brand_name'])
            
            # Build features for Scenario 1
            s1_features = make_features(s1_panel, scenario=1, mode='test')
            
            # Filter to prediction rows (months 0-23)
            s1_pred_rows = s1_features[
                (s1_features['months_postgx'] >= 0) & 
                (s1_features['months_postgx'] <= 23)
            ].copy()
            
            if len(s1_pred_rows) > 0:
                X_s1, meta_s1 = get_feature_matrix_and_meta(s1_pred_rows)
                
                # Filter to training feature columns if specified
                if feature_cols_s1 is not None:
                    X_s1 = X_s1[[c for c in feature_cols_s1 if c in X_s1.columns]]
                
                # Predict
                y_norm_pred = model_scenario1.predict(X_s1)
                
                # Inverse transform
                volume_pred = y_norm_pred * meta_s1['avg_vol_12m'].values
                
                # Build prediction DataFrame
                pred_df = meta_s1[['country', 'brand_name', 'months_postgx']].copy()
                pred_df['volume'] = volume_pred
                predictions.append(pred_df)
                
                logger.info(f"Scenario 1: {len(pred_df)} predictions generated")
        
        # Process Scenario 2 series
        if len(scenario_split[2]) > 0:
            s2_series = pd.DataFrame(scenario_split[2], columns=['country', 'brand_name'])
            s2_panel = test_panel.merge(s2_series, on=['country', 'brand_name'])
            
            # Build features for Scenario 2
            s2_features = make_features(s2_panel, scenario=2, mode='test')
            
            # Filter to prediction rows (months 6-23)
            s2_pred_rows = s2_features[
                (s2_features['months_postgx'] >= 6) & 
                (s2_features['months_postgx'] <= 23)
            ].copy()
            
            if len(s2_pred_rows) > 0:
                X_s2, meta_s2 = get_feature_matrix_and_meta(s2_pred_rows)
                
                # Filter to training feature columns if specified
                if feature_cols_s2 is not None:
                    X_s2 = X_s2[[c for c in feature_cols_s2 if c in X_s2.columns]]
                
                # Predict
                y_norm_pred = model_scenario2.predict(X_s2)
                
                # Inverse transform
                volume_pred = y_norm_pred * meta_s2['avg_vol_12m'].values
                
                # Build prediction DataFrame
                pred_df = meta_s2[['country', 'brand_name', 'months_postgx']].copy()
                pred_df['volume'] = volume_pred
                predictions.append(pred_df)
                
                logger.info(f"Scenario 2: {len(pred_df)} predictions generated")
        
        # Combine all predictions
        if len(predictions) == 0:
            raise ValueError("No predictions generated!")
        
        submission = pd.concat(predictions, ignore_index=True)
        
        # Post-processing: clip negative volumes to 0
        n_negative = (submission['volume'] < 0).sum()
        if n_negative > 0:
            logger.warning(f"Clipping {n_negative} negative predictions to 0")
            submission['volume'] = submission['volume'].clip(lower=0)
        
        # Merge with template to ensure correct order and completeness
        submission = submission_template[['country', 'brand_name', 'months_postgx']].merge(
            submission,
            on=['country', 'brand_name', 'months_postgx'],
            how='left'
        )
        
        # Check for missing predictions
        n_missing = submission['volume'].isna().sum()
        if n_missing > 0:
            logger.warning(f"{n_missing} predictions missing after merge!")
        
    return submission


def apply_edge_case_fallback(
    predictions: pd.DataFrame,
    panel_df: pd.DataFrame,
    global_erosion_curve: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    For problematic series, use conservative fallback.
    
    Edge case criteria:
    - Very low baseline (avg_vol_12m < P5)
    - Short pre-entry history (< 6 months)
    - High pre-entry volatility
    
    Fallback: global_erosion_curve * avg_vol_12m
    
    Args:
        predictions: Current predictions DataFrame
        panel_df: Panel data with series statistics
        global_erosion_curve: Optional pre-computed erosion curve by month
        
    Returns:
        Predictions with edge cases handled
    """
    series_keys = ['country', 'brand_name']
    
    # Identify edge case series
    series_stats = panel_df[series_keys + ['avg_vol_12m', 'pre_entry_volatility']].drop_duplicates()
    
    # Define thresholds
    avg_vol_p5 = series_stats['avg_vol_12m'].quantile(0.05)
    volatility_p95 = series_stats['pre_entry_volatility'].quantile(0.95)
    
    edge_cases = series_stats[
        (series_stats['avg_vol_12m'] < avg_vol_p5) |
        (series_stats['pre_entry_volatility'] > volatility_p95)
    ][series_keys]
    
    if len(edge_cases) == 0:
        logger.info("No edge cases identified")
        return predictions
    
    logger.info(f"Identified {len(edge_cases)} edge case series")
    
    # Compute global erosion curve if not provided
    if global_erosion_curve is None:
        # Use median erosion by month from non-edge-case series
        non_edge = panel_df[~panel_df[series_keys].isin(edge_cases)]
        if 'y_norm' in non_edge.columns:
            global_erosion_curve = non_edge.groupby('months_postgx')['y_norm'].median()
        else:
            # Default fallback curve
            global_erosion_curve = pd.Series({m: max(0.1, 1 - 0.03 * m) for m in range(24)})
    
    # Apply fallback for edge cases
    result = predictions.copy()
    edge_case_mask = result.merge(edge_cases, on=series_keys, how='inner').index
    
    for idx in edge_case_mask:
        row = result.loc[idx]
        month = row['months_postgx']
        series_avg = panel_df[
            (panel_df['country'] == row['country']) & 
            (panel_df['brand_name'] == row['brand_name'])
        ]['avg_vol_12m'].iloc[0]
        
        if month in global_erosion_curve.index:
            result.loc[idx, 'volume'] = global_erosion_curve[month] * series_avg
    
    return result


def validate_submission_format(
    submission_df: pd.DataFrame,
    template_df: pd.DataFrame
) -> bool:
    """
    Final sanity checks before submission.
    
    Checks:
    1. Row count matches template
    2. Correct columns: country, brand_name, months_postgx, volume
    3. No missing values in volume
    4. No negative volumes
    5. Keys match template exactly
    6. No duplicate keys
    
    Args:
        submission_df: Generated submission
        template_df: Official template
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    errors = []
    
    # Check 1: Row count
    if len(submission_df) != len(template_df):
        errors.append(f"Row count mismatch: {len(submission_df)} vs template {len(template_df)}")
    
    # Check 2: Columns
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    missing_cols = set(required_cols) - set(submission_df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check 3: No missing values
    n_missing = submission_df['volume'].isna().sum()
    if n_missing > 0:
        errors.append(f"{n_missing} missing volume values")
    
    # Check 4: No negative values
    n_negative = (submission_df['volume'] < 0).sum()
    if n_negative > 0:
        errors.append(f"{n_negative} negative volume values")
    
    # Check 5: Keys match template
    key_cols = ['country', 'brand_name', 'months_postgx']
    template_keys = set(template_df[key_cols].apply(tuple, axis=1))
    submission_keys = set(submission_df[key_cols].apply(tuple, axis=1))
    
    missing_keys = template_keys - submission_keys
    extra_keys = submission_keys - template_keys
    
    if missing_keys:
        errors.append(f"{len(missing_keys)} keys missing from submission")
    if extra_keys:
        errors.append(f"{len(extra_keys)} extra keys in submission")
    
    # Check 6: No duplicates
    n_duplicates = submission_df[key_cols].duplicated().sum()
    if n_duplicates > 0:
        errors.append(f"{n_duplicates} duplicate keys")
    
    if errors:
        for e in errors:
            logger.error(e)
        raise ValueError(f"Submission validation failed: {errors}")
    
    logger.info("Submission validation passed!")
    return True


def main():
    """CLI entry point for generating submissions."""
    parser = argparse.ArgumentParser(description="Generate submission for Novartis Datathon 2025")
    parser.add_argument('--model-s1', type=str, required=True,
                        help="Path to trained Scenario 1 model")
    parser.add_argument('--model-s2', type=str, required=True,
                        help="Path to trained Scenario 2 model")
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
                        help="Output submission file path")
    parser.add_argument('--data-config', type=str, default='configs/data.yaml',
                        help="Path to data config")
    parser.add_argument('--force-rebuild', action='store_true',
                        help="Force rebuild of cached test panel")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load config
    data_config = load_config(args.data_config)
    
    # Load test data using cached panel
    with timer("Load test data"):
        test_panel = get_panel(
            split='test',
            config=data_config,
            use_cache=True,
            force_rebuild=args.force_rebuild
        )
    
    # Load submission template
    template_path = get_project_root() / data_config['files']['submission_template']
    template = pd.read_csv(template_path)
    
    # Load models
    import joblib
    model_s1 = joblib.load(args.model_s1)
    model_s2 = joblib.load(args.model_s2)
    
    # Generate submission
    submission = generate_submission(
        model_scenario1=model_s1,
        model_scenario2=model_s2,
        test_panel=test_panel,
        submission_template=template
    )
    
    # Validate
    validate_submission_format(submission, template)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")


if __name__ == "__main__":
    main()
