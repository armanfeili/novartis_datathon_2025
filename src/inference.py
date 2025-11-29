"""
Inference and submission generation for Novartis Datathon 2025.

Handles:
- Test scenario detection
- Prediction generation with inverse transform
- Batch prediction for large datasets
- Confidence intervals for ensemble models
- Prediction clipping and validation
- Edge case handling
- Submission file validation
- Auxiliary file generation
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd

from .utils import load_config, setup_logging, timer, get_project_root
from .data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values, get_panel
from .features import make_features, get_feature_columns, get_features
from .train import get_feature_matrix_and_meta

logger = logging.getLogger(__name__)

# Submission format constants
SUBMISSION_COLUMNS = ['country', 'brand_name', 'months_postgx', 'volume']
AUXILIARY_COLUMNS = ['country', 'brand_name', 'avg_vol', 'bucket']
PREDICTION_BOUNDS = (0.0, 2.0)  # Reasonable bounds for y_norm predictions


# =============================================================================
# Batch Prediction Functions
# =============================================================================

def predict_batch(
    model: Any,
    X: pd.DataFrame,
    batch_size: int = 10000,
    verbose: bool = True
) -> np.ndarray:
    """
    Perform batch prediction for large datasets.
    
    This function splits large feature matrices into smaller batches
    to avoid memory issues when making predictions.
    
    Args:
        model: Trained model with predict() method
        X: Feature matrix DataFrame
        batch_size: Number of rows per batch (default 10000)
        verbose: Whether to log progress
        
    Returns:
        np.ndarray of predictions
    """
    n_samples = len(X)
    
    if n_samples <= batch_size:
        # Small dataset, predict directly
        return model.predict(X)
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    predictions = []
    
    if verbose:
        logger.info(f"Batch prediction: {n_samples} samples in {n_batches} batches")
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        X_batch = X.iloc[start_idx:end_idx]
        batch_preds = model.predict(X_batch)
        predictions.append(batch_preds)
        
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"  Batch {i + 1}/{n_batches} complete")
    
    return np.concatenate(predictions)


def predict_with_confidence(
    models: List[Any],
    X: pd.DataFrame,
    confidence_level: float = 0.90,
    batch_size: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions with confidence intervals using ensemble of models.
    
    Uses the spread of predictions across ensemble members to estimate
    confidence intervals. This is useful for uncertainty quantification.
    
    Args:
        models: List of trained models
        X: Feature matrix DataFrame
        confidence_level: Confidence level for intervals (default 0.90)
        batch_size: Batch size for predictions
        
    Returns:
        Tuple of (mean_predictions, lower_bound, upper_bound)
    """
    if len(models) == 0:
        raise ValueError("At least one model is required")
    
    if len(models) == 1:
        # Single model: no confidence intervals, just predictions
        preds = predict_batch(models[0], X, batch_size, verbose=False)
        return preds, preds.copy(), preds.copy()
    
    # Collect predictions from all ensemble members
    all_preds = []
    for i, model in enumerate(models):
        preds = predict_batch(model, X, batch_size, verbose=False)
        all_preds.append(preds)
    
    all_preds = np.array(all_preds)  # Shape: (n_models, n_samples)
    
    # Compute mean and percentiles
    mean_preds = np.mean(all_preds, axis=0)
    
    # Compute confidence intervals using percentiles
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower_bound = np.percentile(all_preds, lower_percentile, axis=0)
    upper_bound = np.percentile(all_preds, upper_percentile, axis=0)
    
    logger.info(
        f"Confidence intervals (level={confidence_level:.0%}): "
        f"mean width = {np.mean(upper_bound - lower_bound):.4f}"
    )
    
    return mean_preds, lower_bound, upper_bound


# =============================================================================
# Prediction Clipping and Validation
# =============================================================================

def clip_predictions(
    predictions: np.ndarray,
    min_bound: float = PREDICTION_BOUNDS[0],
    max_bound: float = PREDICTION_BOUNDS[1],
    is_normalized: bool = True
) -> np.ndarray:
    """
    Clip predictions to reasonable bounds.
    
    For normalized predictions (y_norm), reasonable bounds are [0, 2]:
    - y_norm = 0: complete erosion (no volume)
    - y_norm = 1: volume equals pre-entry average
    - y_norm = 2: volume is double pre-entry (very rare, possible in early LOE)
    
    Args:
        predictions: Array of predictions
        min_bound: Minimum allowed value (default 0.0)
        max_bound: Maximum allowed value (default 2.0)
        is_normalized: Whether predictions are normalized y_norm values
        
    Returns:
        Clipped predictions array
    """
    predictions = np.asarray(predictions)
    
    n_below = np.sum(predictions < min_bound)
    n_above = np.sum(predictions > max_bound)
    
    if n_below > 0 or n_above > 0:
        logger.warning(
            f"Clipping predictions: {n_below} below {min_bound}, "
            f"{n_above} above {max_bound}"
        )
    
    clipped = np.clip(predictions, min_bound, max_bound)
    
    return clipped


def verify_inverse_transform(
    y_norm: np.ndarray,
    avg_vol_12m: np.ndarray,
    volume: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """
    Verify that inverse transform from y_norm to volume is correct.
    
    The inverse transform is: volume = y_norm * avg_vol_12m
    
    This function checks that the relationship holds within tolerance.
    
    Args:
        y_norm: Normalized predictions
        avg_vol_12m: Series-specific average volumes
        volume: Computed volumes (should equal y_norm * avg_vol_12m)
        tolerance: Maximum allowed relative error
        
    Returns:
        True if verification passes
        
    Raises:
        ValueError if verification fails
    """
    y_norm = np.asarray(y_norm)
    avg_vol_12m = np.asarray(avg_vol_12m)
    volume = np.asarray(volume)
    
    expected = y_norm * avg_vol_12m
    
    # Compute relative error, handling zero values
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(volume - expected) / (np.abs(expected) + 1e-10)
    
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    if max_error > tolerance:
        logger.error(
            f"Inverse transform verification failed: "
            f"max_error={max_error:.2e}, mean_error={mean_error:.2e}"
        )
        raise ValueError(
            f"Inverse transform verification failed: max relative error "
            f"{max_error:.2e} exceeds tolerance {tolerance:.2e}"
        )
    
    logger.info(
        f"Inverse transform verified: max_error={max_error:.2e}, "
        f"mean_error={mean_error:.2e}"
    )
    
    return True


def _expand_panel_for_prediction(
    panel: pd.DataFrame,
    start_month: int,
    end_month: int
) -> pd.DataFrame:
    """
    Expand panel to include rows for future months that need predictions.
    
    For test data, we often only have historical (pre-entry) data but need to
    create features for future months (0-23 for S1, 6-23 for S2).
    
    This function creates placeholder rows for the future months by:
    1. Creating rows for each series × future month combination
    2. Forward-filling static columns (country, brand_name, ther_area, etc.)
    3. Setting time-varying columns (volume, n_gxs) to NaN or appropriate values
    
    Args:
        panel: Existing panel DataFrame
        start_month: First prediction month (0 for S1, 6 for S2)
        end_month: Last prediction month (23)
        
    Returns:
        Expanded panel with future month rows
    """
    series_keys = ['country', 'brand_name']
    
    # Get unique series with their static info (use observed=True to avoid category expansion)
    series_info = panel.groupby(series_keys, observed=True).first().reset_index()
    
    # Get columns that are static per series (carry forward)
    # Include 'month' (calendar month) but we'll compute it for future months
    time_varying = ['months_postgx', 'volume', 'n_gxs', 'y_norm']
    static_cols = [c for c in panel.columns if c not in time_varying]
    
    # Get the last known calendar month and months_postgx to extrapolate
    last_known = panel.groupby(series_keys, observed=True).agg({
        'months_postgx': 'max',
        'month': 'last'
    }).reset_index()
    last_known.columns = series_keys + ['last_months_postgx', 'last_month']
    
    # Convert month from category to string for mapping
    last_known['last_month'] = last_known['last_month'].astype(str)
    
    # Calendar month names for mapping
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_to_num = {m: i+1 for i, m in enumerate(month_names)}
    num_to_month = {i+1: m for i, m in enumerate(month_names)}
    
    # Create rows for future months
    future_months = list(range(start_month, end_month + 1))
    
    # Build expanded rows
    future_rows = []
    for month_postgx in future_months:
        month_df = series_info[static_cols].copy()
        month_df['months_postgx'] = month_postgx
        month_df['volume'] = np.nan  # Will be predicted
        month_df['n_gxs'] = np.nan  # Will be filled below
        
        # Compute calendar month for each series
        month_df = month_df.merge(last_known, on=series_keys, how='left')
        
        # Calculate how many months to advance from last known
        months_advance = month_postgx - month_df['last_months_postgx']
        
        # Convert last_month to number, add advance, wrap around
        if 'last_month' in month_df.columns:
            last_month_num = month_df['last_month'].map(month_to_num).fillna(1).astype(int)
            new_month_num = ((last_month_num - 1 + months_advance) % 12) + 1
            month_df['month'] = new_month_num.map(num_to_month)
        
        # Drop helper columns
        month_df = month_df.drop(columns=['last_months_postgx', 'last_month'], errors='ignore')
        
        future_rows.append(month_df)
    
    future_df = pd.concat(future_rows, ignore_index=True)
    
    # Combine with existing panel (keep all historical data for feature engineering)
    expanded = pd.concat([panel, future_df], ignore_index=True)
    
    # Sort by series and month
    expanded = expanded.sort_values(series_keys + ['months_postgx']).reset_index(drop=True)
    
    # Forward-fill n_gxs within each series (use last known value for future months)
    expanded['n_gxs'] = expanded.groupby(series_keys, observed=True)['n_gxs'].transform(
        lambda x: x.ffill().bfill()
    )
    
    logger.info(
        f"Expanded panel: {len(panel)} -> {len(expanded)} rows "
        f"(added months {start_month}-{end_month})"
    )
    
    return expanded


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
    
    # Get min months_postgx per series (use observed=True to avoid creating all category combinations)
    series_min_month = test_volume.groupby(series_keys, observed=True)['months_postgx'].min().reset_index()
    series_min_month.columns = series_keys + ['min_month']
    
    # Scenario 1: series that need predictions from month 0
    # Scenario 2: series that need predictions from month 6 (have months 0-5)
    
    # Get max months_postgx to understand the series range
    series_max_month = test_volume.groupby(series_keys, observed=True)['months_postgx'].max().reset_index()
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
        
        # Process Scenario 1 series (need predictions for months 0-23)
        if len(scenario_split[1]) > 0:
            s1_series = pd.DataFrame(scenario_split[1], columns=['country', 'brand_name'])
            s1_panel = test_panel.merge(s1_series, on=['country', 'brand_name'])
            
            # Expand panel to include future months (0-23) for prediction
            s1_panel_expanded = _expand_panel_for_prediction(s1_panel, start_month=0, end_month=23)
            
            # Build features for Scenario 1
            s1_features = make_features(s1_panel_expanded, scenario=1, mode='test')
            
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
        
        # Process Scenario 2 series (need predictions for months 6-23)
        if len(scenario_split[2]) > 0:
            s2_series = pd.DataFrame(scenario_split[2], columns=['country', 'brand_name'])
            s2_panel = test_panel.merge(s2_series, on=['country', 'brand_name'])
            
            # Expand panel to include future months (6-23) for prediction  
            s2_panel_expanded = _expand_panel_for_prediction(s2_panel, start_month=6, end_month=23)
            
            # Build features for Scenario 2
            s2_features = make_features(s2_panel_expanded, scenario=2, mode='test')
            
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


# =============================================================================
# Enhanced Validation Functions
# =============================================================================

def validate_submission_format(
    submission_df: pd.DataFrame,
    template_df: pd.DataFrame,
    strict: bool = True
) -> bool:
    """
    Final sanity checks before submission.
    
    Checks:
    1. Row count matches template
    2. Correct columns: country, brand_name, months_postgx, volume
    3. Column order matches template exactly
    4. No missing values in volume
    5. No negative volumes
    6. No NaN/Inf values
    7. Keys match template exactly
    8. No duplicate keys
    9. months_postgx values are valid integers
    
    Args:
        submission_df: Generated submission
        template_df: Official template
        strict: If True, raise ValueError on failure; otherwise return False
        
    Returns:
        True if valid, raises ValueError (if strict) or returns False otherwise
    """
    errors = []
    warnings_list = []
    
    # Check 1: Row count
    if len(submission_df) != len(template_df):
        errors.append(f"Row count mismatch: {len(submission_df)} vs template {len(template_df)}")
    
    # Check 2: Columns exist
    required_cols = SUBMISSION_COLUMNS  # ['country', 'brand_name', 'months_postgx', 'volume']
    missing_cols = set(required_cols) - set(submission_df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check 3: Column order matches template
    template_cols = list(template_df.columns)
    submission_cols = list(submission_df.columns)
    if submission_cols != template_cols:
        # This is a warning, not an error - we'll reorder on save
        warnings_list.append(
            f"Column order differs from template. "
            f"Template: {template_cols}, Submission: {submission_cols}"
        )
    
    # Only continue with value checks if required columns exist
    if missing_cols:
        if strict:
            raise ValueError(f"Submission validation failed: {errors}")
        return False
    
    # Check 4: No missing values in volume
    n_missing = submission_df['volume'].isna().sum()
    if n_missing > 0:
        errors.append(f"{n_missing} missing volume values")
    
    # Check 5: No negative values
    n_negative = (submission_df['volume'] < 0).sum()
    if n_negative > 0:
        errors.append(f"{n_negative} negative volume values")
    
    # Check 6: No NaN/Inf values in numeric columns
    for col in ['volume', 'months_postgx']:
        if col in submission_df.columns:
            n_inf = np.isinf(submission_df[col].values).sum()
            if n_inf > 0:
                errors.append(f"{n_inf} Inf values in {col}")
    
    # Check 7: Keys match template
    key_cols = ['country', 'brand_name', 'months_postgx']
    template_keys = set(template_df[key_cols].apply(tuple, axis=1))
    submission_keys = set(submission_df[key_cols].apply(tuple, axis=1))
    
    missing_keys = template_keys - submission_keys
    extra_keys = submission_keys - template_keys
    
    if missing_keys:
        errors.append(f"{len(missing_keys)} keys missing from submission")
    if extra_keys:
        errors.append(f"{len(extra_keys)} extra keys in submission")
    
    # Check 8: No duplicates
    n_duplicates = submission_df[key_cols].duplicated().sum()
    if n_duplicates > 0:
        errors.append(f"{n_duplicates} duplicate keys")
    
    # Check 9: months_postgx values are valid integers (0-23 or 6-23 depending on scenario)
    unique_months = submission_df['months_postgx'].unique()
    invalid_months = [m for m in unique_months if m < 0 or m > 23]
    if invalid_months:
        errors.append(f"Invalid months_postgx values: {invalid_months}")
    
    # Log warnings
    for w in warnings_list:
        logger.warning(w)
    
    if errors:
        for e in errors:
            logger.error(e)
        if strict:
            raise ValueError(f"Submission validation failed: {errors}")
        return False
    
    logger.info("Submission validation passed!")
    return True


def check_submission_statistics(
    submission_df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quick sanity check with summary statistics.
    
    Computes and logs:
    - Mean, std, min, max of volume
    - Distribution by months_postgx
    - Distribution by country
    - Potential anomalies
    
    Args:
        submission_df: Submission DataFrame
        verbose: Whether to log statistics
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Overall volume statistics
    volume = submission_df['volume']
    stats['volume_mean'] = float(volume.mean())
    stats['volume_std'] = float(volume.std())
    stats['volume_min'] = float(volume.min())
    stats['volume_max'] = float(volume.max())
    stats['volume_median'] = float(volume.median())
    
    # Count by scenario (inferred from months_postgx range)
    series_keys = ['country', 'brand_name']
    series_min_month = submission_df.groupby(series_keys)['months_postgx'].min()
    n_s1_series = (series_min_month == 0).sum()
    n_s2_series = (series_min_month > 0).sum()
    stats['n_scenario1_series'] = int(n_s1_series)
    stats['n_scenario2_series'] = int(n_s2_series)
    stats['total_series'] = int(n_s1_series + n_s2_series)
    stats['total_rows'] = len(submission_df)
    
    # Volume by month (mean)
    vol_by_month = submission_df.groupby('months_postgx')['volume'].mean()
    stats['volume_by_month'] = vol_by_month.to_dict()
    
    # Potential anomalies
    stats['n_zero_volume'] = int((volume == 0).sum())
    stats['n_high_volume'] = int((volume > volume.quantile(0.99)).sum())
    
    if verbose:
        logger.info("=" * 50)
        logger.info("Submission Statistics:")
        logger.info(f"  Total rows: {stats['total_rows']}")
        logger.info(f"  Total series: {stats['total_series']} (S1: {n_s1_series}, S2: {n_s2_series})")
        logger.info(f"  Volume - mean: {stats['volume_mean']:.2f}, std: {stats['volume_std']:.2f}")
        logger.info(f"  Volume - min: {stats['volume_min']:.2f}, max: {stats['volume_max']:.2f}")
        logger.info(f"  Zero volume predictions: {stats['n_zero_volume']}")
        logger.info("=" * 50)
    
    return stats


# =============================================================================
# Auxiliary File Generation
# =============================================================================

def generate_auxiliary_file(
    submission_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate auxiliary file for local metric computation.
    
    The auxiliary file contains:
    - country: Country identifier
    - brand_name: Brand identifier
    - avg_vol: Average volume (avg_vol_12m from panel)
    - bucket: Market size bucket (1 or 2)
    
    Note: This file is for LOCAL metric computation only, not for competition
    submission. The competition platform has its own ground truth.
    
    Args:
        submission_df: Generated submission DataFrame
        panel_df: Panel with series metadata (must have avg_vol_12m and bucket)
        output_path: Optional path to save the auxiliary file
        
    Returns:
        Auxiliary DataFrame with columns [country, brand_name, avg_vol, bucket]
    """
    series_keys = ['country', 'brand_name']
    
    # Get unique series from submission
    unique_series = submission_df[series_keys].drop_duplicates()
    
    # Get metadata from panel
    if 'avg_vol_12m' not in panel_df.columns:
        raise ValueError("Panel must contain 'avg_vol_12m' column")
    if 'bucket' not in panel_df.columns:
        raise ValueError("Panel must contain 'bucket' column")
    
    # Get series-level metadata (should be constant per series)
    series_meta = panel_df[series_keys + ['avg_vol_12m', 'bucket']].drop_duplicates(subset=series_keys)
    
    # Merge with unique series
    auxiliary = unique_series.merge(series_meta, on=series_keys, how='left')
    
    # Rename columns to match expected format
    auxiliary = auxiliary.rename(columns={'avg_vol_12m': 'avg_vol'})
    
    # Ensure correct column order
    auxiliary = auxiliary[AUXILIARY_COLUMNS]
    
    # Check for missing values
    n_missing = auxiliary.isna().any(axis=1).sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} series missing metadata in auxiliary file")
    
    logger.info(f"Generated auxiliary file with {len(auxiliary)} series")
    
    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        auxiliary.to_csv(output_path, index=False)
        logger.info(f"Auxiliary file saved to {output_path}")
    
    return auxiliary


# =============================================================================
# Submission Versioning and Logging
# =============================================================================

def generate_submission_version(
    model_info: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None
) -> str:
    """
    Generate unique submission version identifier.
    
    Format: {timestamp}_{run_name}_{hash}
    
    Args:
        model_info: Optional dictionary with model information
        run_name: Optional run name
        
    Returns:
        Version string (e.g., "20250113_123456_baseline_abc123")
    """
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run name (sanitized)
    if run_name:
        run_name = run_name.replace(" ", "_").replace("/", "_")[:30]
    else:
        run_name = "submission"
    
    # Short hash based on model info
    if model_info:
        info_str = json.dumps(model_info, sort_keys=True, default=str)
        short_hash = hashlib.sha256(info_str.encode()).hexdigest()[:6]
    else:
        short_hash = hashlib.sha256(timestamp.encode()).hexdigest()[:6]
    
    version = f"{timestamp}_{run_name}_{short_hash}"
    
    return version


def log_submission(
    submission_df: pd.DataFrame,
    version: str,
    model_info: Optional[Dict[str, Any]] = None,
    validation_score: Optional[float] = None,
    notes: Optional[str] = None,
    log_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create submission log entry.
    
    Args:
        submission_df: Submission DataFrame
        version: Submission version string
        model_info: Optional model information
        validation_score: Optional CV/validation score
        notes: Optional notes about the submission
        log_path: Optional path to append log entry
        
    Returns:
        Log entry dictionary
    """
    stats = check_submission_statistics(submission_df, verbose=False)
    
    log_entry = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'n_rows': len(submission_df),
        'n_series': stats['total_series'],
        'volume_mean': stats['volume_mean'],
        'volume_std': stats['volume_std'],
        'volume_min': stats['volume_min'],
        'volume_max': stats['volume_max'],
        'validation_score': validation_score,
        'model_info': model_info or {},
        'notes': notes or ''
    }
    
    logger.info(f"Submission log: {version}")
    
    # Append to log file if path provided
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing log or create new
        if log_path.exists():
            with open(log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
        
        logger.info(f"Log entry appended to {log_path}")
    
    return log_entry


# =============================================================================
# Edge Case Handling
# =============================================================================

def handle_missing_pre_entry_data(
    panel_df: pd.DataFrame,
    fallback_avg_vol: float = 1000.0
) -> pd.DataFrame:
    """
    Handle series with missing pre-entry data.
    
    For series where avg_vol_12m cannot be computed (insufficient pre-entry history),
    use a fallback value or global average.
    
    Args:
        panel_df: Panel DataFrame
        fallback_avg_vol: Default value if avg_vol_12m is missing
        
    Returns:
        Panel with missing avg_vol_12m filled
    """
    result = panel_df.copy()
    
    # Identify missing avg_vol_12m
    missing_mask = result['avg_vol_12m'].isna() | (result['avg_vol_12m'] <= 0)
    n_missing = missing_mask.sum()
    
    if n_missing > 0:
        # Compute global average from non-missing series
        valid_avg = result.loc[~missing_mask, 'avg_vol_12m']
        if len(valid_avg) > 0:
            global_avg = valid_avg.median()
        else:
            global_avg = fallback_avg_vol
        
        result.loc[missing_mask, 'avg_vol_12m'] = global_avg
        logger.warning(
            f"Filled {n_missing} rows with missing avg_vol_12m using fallback value {global_avg:.2f}"
        )
    
    return result


def handle_zero_volume_series(
    predictions_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    replacement_strategy: str = 'global_erosion'
) -> pd.DataFrame:
    """
    Handle series with all-zero historical volumes.
    
    These series are problematic because:
    1. avg_vol_12m = 0 means inverse transform gives 0
    2. May indicate data quality issues
    
    Strategies:
    - 'global_erosion': Use global average erosion curve
    - 'small_constant': Use small constant value
    - 'remove': Keep as-is (zeros)
    
    Args:
        predictions_df: Predictions DataFrame
        panel_df: Panel with series metadata
        replacement_strategy: Strategy for handling zero-volume series
        
    Returns:
        Predictions with zero-volume series handled
    """
    series_keys = ['country', 'brand_name']
    
    # Find series with zero avg_vol_12m
    series_avg = panel_df[series_keys + ['avg_vol_12m']].drop_duplicates(subset=series_keys)
    zero_vol_series = series_avg[series_avg['avg_vol_12m'] == 0][series_keys]
    
    if len(zero_vol_series) == 0:
        logger.info("No zero-volume series found")
        return predictions_df
    
    logger.warning(f"Found {len(zero_vol_series)} series with zero avg_vol_12m")
    
    result = predictions_df.copy()
    
    if replacement_strategy == 'global_erosion':
        # Compute global average volume and erosion
        non_zero = panel_df[panel_df['avg_vol_12m'] > 0]
        global_avg_vol = non_zero['avg_vol_12m'].median() if len(non_zero) > 0 else 1000.0
        
        # Apply small scaled values
        zero_mask = result.merge(zero_vol_series, on=series_keys, how='inner').index
        for idx in zero_mask:
            month = result.loc[idx, 'months_postgx']
            # Simple erosion: starts at 10% of global avg, decays
            erosion_factor = max(0.01, 0.1 * (1 - 0.02 * month))
            result.loc[idx, 'volume'] = global_avg_vol * erosion_factor
            
    elif replacement_strategy == 'small_constant':
        zero_mask = result.merge(zero_vol_series, on=series_keys, how='inner').index
        result.loc[zero_mask, 'volume'] = 1.0  # Minimal non-zero value
        
    # 'remove' or unknown strategy: keep as-is
    
    return result


def handle_extreme_predictions(
    predictions_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    max_ratio: float = 2.0,
    min_ratio: float = 0.0
) -> pd.DataFrame:
    """
    Handle extreme predictions that are unrealistic.
    
    Clips predictions to reasonable bounds relative to avg_vol_12m.
    
    Args:
        predictions_df: Predictions DataFrame (with volume column)
        panel_df: Panel with series metadata (avg_vol_12m)
        max_ratio: Maximum volume/avg_vol_12m ratio (default 2.0)
        min_ratio: Minimum volume/avg_vol_12m ratio (default 0.0)
        
    Returns:
        Predictions with extreme values clipped
    """
    series_keys = ['country', 'brand_name']
    
    # Get avg_vol_12m per series
    series_avg = panel_df[series_keys + ['avg_vol_12m']].drop_duplicates(subset=series_keys)
    
    # Merge with predictions
    result = predictions_df.merge(series_avg, on=series_keys, how='left')
    
    # Compute bounds
    result['max_volume'] = result['avg_vol_12m'] * max_ratio
    result['min_volume'] = result['avg_vol_12m'] * min_ratio
    
    # Clip
    n_clipped_high = (result['volume'] > result['max_volume']).sum()
    n_clipped_low = (result['volume'] < result['min_volume']).sum()
    
    result['volume'] = result['volume'].clip(
        lower=result['min_volume'],
        upper=result['max_volume']
    )
    
    if n_clipped_high > 0 or n_clipped_low > 0:
        logger.warning(
            f"Clipped {n_clipped_high} predictions above max_ratio={max_ratio}, "
            f"{n_clipped_low} below min_ratio={min_ratio}"
        )
    
    # Drop helper columns
    result = result.drop(columns=['avg_vol_12m', 'max_volume', 'min_volume'])
    
    return result


# =============================================================================
# Complete Submission Workflow
# =============================================================================

def save_submission_with_versioning(
    submission_df: pd.DataFrame,
    output_dir: Union[str, Path],
    run_name: Optional[str] = None,
    model_info: Optional[Dict[str, Any]] = None,
    validation_score: Optional[float] = None,
    save_auxiliary: bool = True,
    panel_df: Optional[pd.DataFrame] = None
) -> Dict[str, Path]:
    """
    Save submission with automatic versioning, logging, and auxiliary file.
    
    Creates:
    - submissions/{version}/submission.csv
    - submissions/{version}/auxiliary.csv (if save_auxiliary=True)
    - submissions/{version}/metadata.json
    - submissions/submission_log.json (appended)
    
    Args:
        submission_df: Submission DataFrame
        output_dir: Base output directory
        run_name: Optional run name
        model_info: Optional model information
        validation_score: Optional validation score
        save_auxiliary: Whether to save auxiliary file
        panel_df: Panel DataFrame (required if save_auxiliary=True)
        
    Returns:
        Dictionary with paths to saved files
    """
    output_dir = Path(output_dir)
    
    # Generate version
    version = generate_submission_version(model_info, run_name)
    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Ensure column order matches expected format
    submission_ordered = submission_df[SUBMISSION_COLUMNS].copy()
    
    # Save submission
    submission_path = version_dir / 'submission.csv'
    submission_ordered.to_csv(submission_path, index=False)
    paths['submission'] = submission_path
    logger.info(f"Submission saved to {submission_path}")
    
    # Save auxiliary file
    if save_auxiliary:
        if panel_df is None:
            logger.warning("Cannot save auxiliary file: panel_df not provided")
        else:
            auxiliary_path = version_dir / 'auxiliary.csv'
            generate_auxiliary_file(submission_df, panel_df, auxiliary_path)
            paths['auxiliary'] = auxiliary_path
    
    # Save metadata
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'model_info': model_info or {},
        'validation_score': validation_score,
        'statistics': check_submission_statistics(submission_df, verbose=False)
    }
    
    metadata_path = version_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    paths['metadata'] = metadata_path
    
    # Append to global log
    log_path = output_dir / 'submission_log.json'
    log_submission(
        submission_df=submission_df,
        version=version,
        model_info=model_info,
        validation_score=validation_score,
        log_path=log_path
    )
    paths['log'] = log_path
    
    # Print summary
    stats = check_submission_statistics(submission_df, verbose=True)
    
    return paths


def main():
    """CLI entry point for generating submissions."""
    parser = argparse.ArgumentParser(description="Generate submission for Novartis Datathon 2025")
    parser.add_argument('--model-s1', type=str, required=True,
                        help="Path to trained Scenario 1 model")
    parser.add_argument('--model-s2', type=str, required=True,
                        help="Path to trained Scenario 2 model")
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
                        help="Output submission file path")
    parser.add_argument('--output-dir', type=str, default='submissions',
                        help="Output directory for versioned submissions")
    parser.add_argument('--data-config', type=str, default='configs/data.yaml',
                        help="Path to data config")
    parser.add_argument('--force-rebuild', action='store_true',
                        help="Force rebuild of cached test panel")
    parser.add_argument('--run-name', type=str, default=None,
                        help="Name for this submission run")
    parser.add_argument('--validation-score', type=float, default=None,
                        help="Validation score to log with submission")
    parser.add_argument('--use-versioning', action='store_true',
                        help="Save with automatic versioning and auxiliary files")
    parser.add_argument('--save-auxiliary', action='store_true',
                        help="Generate auxiliary file for local metric computation")
    
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
    
    # Handle edge cases in test panel
    test_panel = handle_missing_pre_entry_data(test_panel)
    
    # Load submission template
    template_path = get_project_root() / data_config['files']['submission_template']
    template = pd.read_csv(template_path)
    
    # Load models - detect format and use appropriate loader
    def load_model(path: str):
        """Load model with appropriate loader based on file extension."""
        import joblib
        from pathlib import Path
        path_obj = Path(path)
        
        # Check file extension to determine loader
        if path_obj.suffix in ['.cbm', '.bin']:
            # CatBoost native format
            try:
                from catboost import CatBoostRegressor
                model = CatBoostRegressor()
                model.load_model(path)
                return model
            except Exception:
                pass
        
        # Try joblib for other formats (pkl, joblib, etc.)
        try:
            return joblib.load(path)
        except Exception:
            pass
        
        # Try CatBoost as fallback
        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(path)
            return model
        except Exception as e:
            raise ValueError(f"Could not load model from {path}: {e}")
    
    model_s1 = load_model(args.model_s1)
    model_s2 = load_model(args.model_s2)
    
    # Model info for logging
    model_info = {
        'model_s1_path': args.model_s1,
        'model_s2_path': args.model_s2,
        'model_s1_type': type(model_s1).__name__,
        'model_s2_type': type(model_s2).__name__
    }
    
    # Generate submission
    submission = generate_submission(
        model_scenario1=model_s1,
        model_scenario2=model_s2,
        test_panel=test_panel,
        submission_template=template
    )
    
    # Handle edge cases in predictions
    submission = handle_zero_volume_series(submission, test_panel)
    submission = handle_extreme_predictions(submission, test_panel)
    
    # Validate
    validate_submission_format(submission, template)
    
    # Check statistics
    check_submission_statistics(submission, verbose=True)
    
    # Save
    if args.use_versioning:
        paths = save_submission_with_versioning(
            submission_df=submission,
            output_dir=args.output_dir,
            run_name=args.run_name,
            model_info=model_info,
            validation_score=args.validation_score,
            save_auxiliary=args.save_auxiliary,
            panel_df=test_panel
        )
        logger.info(f"Versioned submission saved to {paths['submission']}")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission[SUBMISSION_COLUMNS].to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        
        # Optionally save auxiliary file
        if args.save_auxiliary:
            aux_path = output_path.parent / 'auxiliary.csv'
            generate_auxiliary_file(submission, test_panel, aux_path)


if __name__ == "__main__":
    main()
