"""
Evaluation module for Novartis Datathon 2025.

Wraps the official metric_calculation.py for local validation.
Provides compute_metric1, compute_metric2, and helper functions.

IMPORTANT: Both df_actual and df_pred must contain ACTUAL volume (not normalized)
in a 'volume' column. The official metric operates on raw volumes, not y_norm.

METRIC WEIGHTS (from official metric_calculation.py):
====================================================
These values are defined in configs/run_defaults.yaml under 'official_metric'
but the fallback implementations here use hardcoded values to EXACTLY match
the official metric_calculation.py script.

Metric 1 (Scenario 1 / Phase 1A):
  - Monthly error (months 0-23): 0.2 weight
  - Accumulated error (months 0-5): 0.5 weight [CRITICAL]
  - Accumulated error (months 6-11): 0.2 weight
  - Accumulated error (months 12-23): 0.1 weight

Metric 2 (Scenario 2 / Phase 1B):
  - Monthly error (months 6-23): 0.2 weight
  - Accumulated error (months 6-11): 0.5 weight [CRITICAL]
  - Accumulated error (months 12-23): 0.3 weight

Bucket weights:
  - Bucket 1 (high erosion): 2x weight (2/n1)
  - Bucket 2 (low erosion): 1x weight (1/n2)

Bucket threshold:
  - Bucket 1: mean_erosion <= 0.25
  - Bucket 2: mean_erosion > 0.25
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .utils import get_project_root

logger = logging.getLogger(__name__)

# Add docs/guide to path for importing official metric
_guide_path = get_project_root() / 'docs' / 'guide'
if str(_guide_path) not in sys.path:
    sys.path.insert(0, str(_guide_path))

# Import official metric functions
try:
    from metric_calculation import compute_metric1 as _official_metric1
    from metric_calculation import compute_metric2 as _official_metric2
    OFFICIAL_METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Official metric_calculation.py not found. Using fallback implementations.")
    OFFICIAL_METRICS_AVAILABLE = False


def compute_metric1(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """
    Wrapper around official compute_metric1 for Scenario 1.
    
    IMPORTANT: df_actual and df_pred must have columns:
        [country, brand_name, months_postgx, volume]
    where 'volume' is ACTUAL volume (not normalized y_norm).
    
    Phase 1A weights:
    - 20%: Monthly error (months 0-23)
    - 50%: Accumulated error (months 0-5) [CRITICAL]
    - 20%: Accumulated error (months 6-11)
    - 10%: Accumulated error (months 12-23)
    
    Bucket 1 weighted 2×, Bucket 2 weighted 1×.
    
    Args:
        df_actual: Actual volume data with columns [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data with same columns
        df_aux: Auxiliary data with columns [country, brand_name, avg_vol, bucket]
        
    Returns:
        Metric 1 score (lower is better)
    """
    # Validate inputs
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    for df, name in [(df_actual, 'df_actual'), (df_pred, 'df_pred')]:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    
    aux_required = ['country', 'brand_name', 'avg_vol', 'bucket']
    missing_aux = set(aux_required) - set(df_aux.columns)
    if missing_aux:
        raise ValueError(f"df_aux missing columns: {missing_aux}")
    
    if OFFICIAL_METRICS_AVAILABLE:
        return _official_metric1(df_actual, df_pred, df_aux)
    else:
        return _fallback_metric1(df_actual, df_pred, df_aux)


def compute_metric2(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """
    Wrapper around official compute_metric2 for Scenario 2.
    
    IMPORTANT: df_actual and df_pred must have columns:
        [country, brand_name, months_postgx, volume]
    where 'volume' is ACTUAL volume (not normalized y_norm).
    
    Phase 1B weights:
    - 20%: Monthly error (months 6-23)
    - 50%: Accumulated error (months 6-11) [CRITICAL]
    - 30%: Accumulated error (months 12-23)
    
    Bucket 1 weighted 2×, Bucket 2 weighted 1×.
    
    Args:
        df_actual: Actual volume data
        df_pred: Predicted volume data
        df_aux: Auxiliary data with bucket and avg_vol
        
    Returns:
        Metric 2 score (lower is better)
    """
    # Validate inputs
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    for df, name in [(df_actual, 'df_actual'), (df_pred, 'df_pred')]:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    
    if OFFICIAL_METRICS_AVAILABLE:
        return _official_metric2(df_actual, df_pred, df_aux)
    else:
        return _fallback_metric2(df_actual, df_pred, df_aux)


def compute_bucket_metrics(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame,
    scenario
) -> Dict[str, float]:
    """
    Compute metrics separately for Bucket 1 and Bucket 2.
    
    Args:
        df_actual: Actual volume data
        df_pred: Predicted volume data
        df_aux: Auxiliary data with bucket classification
        scenario: 1, 2, "scenario1", or "scenario2"
        
    Returns:
        {'overall': float, 'bucket1': float, 'bucket2': float}
    """
    # Normalize scenario to integer
    if isinstance(scenario, str):
        scenario = int(scenario[-1]) if scenario.startswith('scenario') else int(scenario)
    
    # Overall metric
    if scenario == 1:
        overall = compute_metric1(df_actual, df_pred, df_aux)
    else:
        overall = compute_metric2(df_actual, df_pred, df_aux)
    
    # Per-bucket metrics
    bucket_metrics = {'overall': overall}
    
    for bucket in [1, 2]:
        bucket_series = df_aux[df_aux['bucket'] == bucket][['country', 'brand_name']]
        
        if len(bucket_series) == 0:
            bucket_metrics[f'bucket{bucket}'] = np.nan
            continue
        
        # Filter to this bucket's series
        actual_bucket = df_actual.merge(bucket_series, on=['country', 'brand_name'])
        pred_bucket = df_pred.merge(bucket_series, on=['country', 'brand_name'])
        aux_bucket = df_aux[df_aux['bucket'] == bucket]
        
        if len(actual_bucket) == 0:
            bucket_metrics[f'bucket{bucket}'] = np.nan
            continue
        
        # Compute metric for this bucket
        try:
            if scenario == 1:
                bucket_metrics[f'bucket{bucket}'] = _compute_single_bucket_metric1(
                    actual_bucket, pred_bucket, aux_bucket
                )
            else:
                bucket_metrics[f'bucket{bucket}'] = _compute_single_bucket_metric2(
                    actual_bucket, pred_bucket, aux_bucket
                )
        except Exception as e:
            logger.warning(f"Could not compute bucket {bucket} metric: {e}")
            bucket_metrics[f'bucket{bucket}'] = np.nan
    
    return bucket_metrics


def create_aux_file(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create auxiliary file for metric computation.
    
    Args:
        panel_df: Training panel with avg_vol_12m and bucket columns
        
    Returns:
        DataFrame with columns: [country, brand_name, avg_vol, bucket]
    
    NOTE: Only create from training data. For test, organizers have their own.
    """
    required_cols = ['country', 'brand_name', 'avg_vol_12m', 'bucket']
    missing = set(required_cols) - set(panel_df.columns)
    if missing:
        raise ValueError(f"panel_df missing columns: {missing}")
    
    # Get unique series with their stats
    aux = panel_df[['country', 'brand_name', 'avg_vol_12m', 'bucket']].drop_duplicates()
    
    # Rename to match expected format
    aux = aux.rename(columns={'avg_vol_12m': 'avg_vol'})
    
    logger.info(f"Created aux file with {len(aux)} series")
    logger.info(f"Bucket distribution:\n{aux['bucket'].value_counts().to_string()}")
    
    return aux


def _fallback_metric1(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """Fallback implementation of Metric 1 if official script unavailable."""
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_predict")
    ).merge(df_aux, on=["country", "brand_name"], how="left")
    
    # Filter to series starting at month 0
    merged["start_month"] = merged.groupby(["country", "brand_name"])["months_postgx"].transform("min")
    merged = merged[merged["start_month"] == 0].copy()
    
    def compute_pe(group):
        avg_vol = group["avg_vol"].iloc[0]
        if avg_vol == 0 or np.isnan(avg_vol):
            return np.nan
        
        def sum_abs_diff(m_start, m_end):
            subset = group[(group["months_postgx"] >= m_start) & (group["months_postgx"] <= m_end)]
            return (subset["volume_actual"] - subset["volume_predict"]).abs().sum()
        
        def abs_sum_diff(m_start, m_end):
            subset = group[(group["months_postgx"] >= m_start) & (group["months_postgx"] <= m_end)]
            return abs(subset["volume_actual"].sum() - subset["volume_predict"].sum())
        
        term1 = 0.2 * sum_abs_diff(0, 23) / (24 * avg_vol)
        term2 = 0.5 * abs_sum_diff(0, 5) / (6 * avg_vol)
        term3 = 0.2 * abs_sum_diff(6, 11) / (6 * avg_vol)
        term4 = 0.1 * abs_sum_diff(12, 23) / (12 * avg_vol)
        
        return term1 + term2 + term3 + term4
    
    pe_results = merged.groupby(["country", "brand_name", "bucket"]).apply(compute_pe, include_groups=False).reset_index(name="PE")
    
    bucket1 = pe_results[pe_results["bucket"] == 1]
    bucket2 = pe_results[pe_results["bucket"] == 2]
    
    n1 = bucket1[["country", "brand_name"]].drop_duplicates().shape[0]
    n2 = bucket2[["country", "brand_name"]].drop_duplicates().shape[0]
    
    if n1 == 0 or n2 == 0:
        logger.warning("One bucket is empty, returning partial metric")
        if n1 > 0:
            return 2 * bucket1["PE"].mean()
        elif n2 > 0:
            return bucket2["PE"].mean()
        return np.nan
    
    return round((2/n1) * bucket1["PE"].sum() + (1/n2) * bucket2["PE"].sum(), 4)


def _fallback_metric2(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """Fallback implementation of Metric 2 if official script unavailable."""
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_predict")
    ).merge(df_aux, on=["country", "brand_name"], how="left")
    
    # Filter to series starting at month 6
    merged["start_month"] = merged.groupby(["country", "brand_name"])["months_postgx"].transform("min")
    merged = merged[merged["start_month"] == 6].copy()
    
    def compute_pe(group):
        avg_vol = group["avg_vol"].iloc[0]
        if avg_vol == 0 or np.isnan(avg_vol):
            return np.nan
        
        def sum_abs_diff(m_start, m_end):
            subset = group[(group["months_postgx"] >= m_start) & (group["months_postgx"] <= m_end)]
            return (subset["volume_actual"] - subset["volume_predict"]).abs().sum()
        
        def abs_sum_diff(m_start, m_end):
            subset = group[(group["months_postgx"] >= m_start) & (group["months_postgx"] <= m_end)]
            return abs(subset["volume_actual"].sum() - subset["volume_predict"].sum())
        
        term1 = 0.2 * sum_abs_diff(6, 23) / (18 * avg_vol)
        term2 = 0.5 * abs_sum_diff(6, 11) / (6 * avg_vol)
        term3 = 0.3 * abs_sum_diff(12, 23) / (12 * avg_vol)
        
        return term1 + term2 + term3
    
    pe_results = merged.groupby(["country", "brand_name", "bucket"]).apply(compute_pe, include_groups=False).reset_index(name="PE")
    
    bucket1 = pe_results[pe_results["bucket"] == 1]
    bucket2 = pe_results[pe_results["bucket"] == 2]
    
    n1 = bucket1[["country", "brand_name"]].drop_duplicates().shape[0]
    n2 = bucket2[["country", "brand_name"]].drop_duplicates().shape[0]
    
    if n1 == 0 or n2 == 0:
        logger.warning("One bucket is empty, returning partial metric")
        if n1 > 0:
            return 2 * bucket1["PE"].mean()
        elif n2 > 0:
            return bucket2["PE"].mean()
        return np.nan
    
    return round((2/n1) * bucket1["PE"].sum() + (1/n2) * bucket2["PE"].sum(), 4)


def _compute_single_bucket_metric1(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """Compute Metric 1 for a single bucket (for analysis)."""
    return _fallback_metric1(df_actual, df_pred, df_aux)


def _compute_single_bucket_metric2(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """Compute Metric 2 for a single bucket (for analysis)."""
    return _fallback_metric2(df_actual, df_pred, df_aux)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


# =============================================================================
# ERROR ANALYSIS FUNCTIONS (Section 6.5)
# =============================================================================

def compute_per_series_error(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame,
    scenario
) -> pd.DataFrame:
    """
    Compute error metrics for each series.
    
    Args:
        df_actual: Actual volume data with columns [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data with same columns
        df_aux: Auxiliary data with columns [country, brand_name, avg_vol, bucket]
        scenario: 1, 2, "scenario1", or "scenario2"
        
    Returns:
        DataFrame with columns:
            [country, brand_name, bucket, avg_vol, mae, rmse, mape, 
             total_actual, total_pred, total_error, n_months]
    """
    # Normalize scenario
    if isinstance(scenario, str):
        scenario = int(scenario[-1]) if scenario.startswith('scenario') else int(scenario)
    
    # Merge actual and predicted
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_pred")
    ).merge(df_aux, on=["country", "brand_name"], how="left")
    
    # Filter based on scenario
    if scenario == 1:
        merged = merged[merged['months_postgx'] >= 0]
    else:
        merged = merged[merged['months_postgx'] >= 6]
    
    # Compute per-series metrics
    def compute_series_metrics(group):
        actual = group['volume_actual'].values
        pred = group['volume_pred'].values
        avg_vol = group['avg_vol'].iloc[0]
        
        error = pred - actual
        abs_error = np.abs(error)
        
        return pd.Series({
            'bucket': group['bucket'].iloc[0],
            'avg_vol': avg_vol,
            'mae': abs_error.mean(),
            'rmse': np.sqrt((error ** 2).mean()),
            'mape': (abs_error / (actual + 1e-8)).mean() * 100,
            'total_actual': actual.sum(),
            'total_pred': pred.sum(),
            'total_error': abs(actual.sum() - pred.sum()),
            'normalized_total_error': abs(actual.sum() - pred.sum()) / (avg_vol + 1e-8),
            'n_months': len(group)
        })
    
    per_series = merged.groupby(['country', 'brand_name']).apply(
        compute_series_metrics, include_groups=False
    ).reset_index()
    
    return per_series


def identify_worst_series(
    per_series_errors: pd.DataFrame,
    metric: str = 'mae',
    top_k: int = 10,
    bucket: Optional[int] = None
) -> pd.DataFrame:
    """
    Identify worst-performing series based on specified metric.
    
    Args:
        per_series_errors: Output from compute_per_series_error()
        metric: Metric to rank by ('mae', 'rmse', 'mape', 'normalized_total_error')
        top_k: Number of worst series to return
        bucket: If specified, filter to only this bucket (1 or 2)
        
    Returns:
        DataFrame with top_k worst series sorted by metric descending
    """
    df = per_series_errors.copy()
    
    if bucket is not None:
        df = df[df['bucket'] == bucket]
    
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")
    
    return df.nlargest(top_k, metric)


def analyze_errors_by_bucket(
    per_series_errors: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate error statistics by bucket.
    
    Args:
        per_series_errors: Output from compute_per_series_error()
        
    Returns:
        {
            'bucket1': {'n_series': int, 'mean_mae': float, ...},
            'bucket2': {'n_series': int, 'mean_mae': float, ...},
            'overall': {'n_series': int, 'mean_mae': float, ...}
        }
    """
    result = {}
    
    for bucket_name, df_bucket in [('bucket1', per_series_errors[per_series_errors['bucket'] == 1]),
                                    ('bucket2', per_series_errors[per_series_errors['bucket'] == 2]),
                                    ('overall', per_series_errors)]:
        if len(df_bucket) == 0:
            result[bucket_name] = {'n_series': 0}
            continue
            
        result[bucket_name] = {
            'n_series': len(df_bucket),
            'mean_mae': df_bucket['mae'].mean(),
            'std_mae': df_bucket['mae'].std(),
            'mean_rmse': df_bucket['rmse'].mean(),
            'std_rmse': df_bucket['rmse'].std(),
            'mean_mape': df_bucket['mape'].mean(),
            'median_mae': df_bucket['mae'].median(),
            'max_mae': df_bucket['mae'].max(),
            'mean_normalized_error': df_bucket['normalized_total_error'].mean()
        }
    
    return result


def analyze_errors_by_time_window(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    scenario
) -> Dict[str, Dict[str, float]]:
    """
    Analyze errors by time window (early/mid/late).
    
    Args:
        df_actual: Actual volume data
        df_pred: Predicted volume data
        scenario: 1 or 2
        
    Returns:
        {
            'early': {'months': '0-5'/'6-11', 'mae': float, ...},
            'mid': {'months': '6-11'/'12-17', 'mae': float, ...},
            'late': {'months': '12-23'/'18-23', 'mae': float, ...}
        }
    """
    if isinstance(scenario, str):
        scenario = int(scenario[-1]) if scenario.startswith('scenario') else int(scenario)
    
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_pred")
    )
    
    # Define time windows based on scenario
    if scenario == 1:
        windows = {
            'early': (0, 5),
            'mid': (6, 11),
            'late': (12, 23)
        }
    else:
        windows = {
            'early': (6, 11),
            'mid': (12, 17),
            'late': (18, 23)
        }
    
    result = {}
    for window_name, (start, end) in windows.items():
        df_window = merged[(merged['months_postgx'] >= start) & 
                           (merged['months_postgx'] <= end)]
        
        if len(df_window) == 0:
            result[window_name] = {
                'months': f'{start}-{end}',
                'n_rows': 0
            }
            continue
        
        error = df_window['volume_pred'] - df_window['volume_actual']
        abs_error = np.abs(error)
        
        result[window_name] = {
            'months': f'{start}-{end}',
            'n_rows': len(df_window),
            'mae': abs_error.mean(),
            'rmse': np.sqrt((error ** 2).mean()),
            'mean_error': error.mean(),  # Positive = overprediction
            'std_error': error.std()
        }
    
    return result


def check_systematic_bias(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame
) -> Dict[str, float]:
    """
    Check for systematic over/under prediction.
    
    Args:
        df_actual: Actual volume data
        df_pred: Predicted volume data
        
    Returns:
        {
            'mean_error': float,  # Positive = overprediction
            'median_error': float,
            'pct_overprediction': float,  # % of predictions above actual
            'bias_magnitude': float  # |mean_error| / mean_actual
        }
    """
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_pred")
    )
    
    error = merged['volume_pred'] - merged['volume_actual']
    
    return {
        'mean_error': error.mean(),
        'median_error': error.median(),
        'pct_overprediction': (error > 0).mean() * 100,
        'bias_magnitude': abs(error.mean()) / (merged['volume_actual'].mean() + 1e-8)
    }


# Canonical metric name constants (Section 6.7.1)
METRIC_NAME_S1 = "metric1_official"
METRIC_NAME_S2 = "metric2_official"
METRIC_NAME_RMSE = "rmse_y_norm"
METRIC_NAME_MAE = "mae_y_norm"

# =============================================================================
# OFFICIAL METRIC CONSTANTS (from metric_calculation.py)
# =============================================================================
# These are the exact values from the official metric script.
# They are also defined in configs/run_defaults.yaml for reference.
# If the official script changes, update BOTH this file AND the config.

OFFICIAL_BUCKET_THRESHOLD = 0.25
OFFICIAL_BUCKET1_WEIGHT = 2.0
OFFICIAL_BUCKET2_WEIGHT = 1.0

OFFICIAL_METRIC1_WEIGHTS = {
    'monthly': 0.2,
    'accumulated_0_5': 0.5,
    'accumulated_6_11': 0.2,
    'accumulated_12_23': 0.1,
}

OFFICIAL_METRIC2_WEIGHTS = {
    'monthly': 0.2,
    'accumulated_6_11': 0.5,
    'accumulated_12_23': 0.3,
}


def validate_config_matches_official(run_config: dict) -> bool:
    """
    Validate that run_config metric values match official metric_calculation.py.
    
    Args:
        run_config: Loaded run_defaults.yaml config dict
        
    Returns:
        True if all values match, raises ValueError if mismatch detected
    """
    official_metric = run_config.get('official_metric', {})
    mismatches = []
    
    # Check bucket threshold
    config_threshold = official_metric.get('bucket_threshold')
    if config_threshold is not None and config_threshold != OFFICIAL_BUCKET_THRESHOLD:
        mismatches.append(f"bucket_threshold: config={config_threshold}, official={OFFICIAL_BUCKET_THRESHOLD}")
    
    # Check bucket weights
    bucket_weights = official_metric.get('bucket_weights', {})
    if bucket_weights.get('bucket1') is not None and bucket_weights.get('bucket1') != OFFICIAL_BUCKET1_WEIGHT:
        mismatches.append(f"bucket1_weight: config={bucket_weights.get('bucket1')}, official={OFFICIAL_BUCKET1_WEIGHT}")
    if bucket_weights.get('bucket2') is not None and bucket_weights.get('bucket2') != OFFICIAL_BUCKET2_WEIGHT:
        mismatches.append(f"bucket2_weight: config={bucket_weights.get('bucket2')}, official={OFFICIAL_BUCKET2_WEIGHT}")
    
    # Check metric1 weights
    metric1 = official_metric.get('metric1', {})
    for key, official_val in [
        ('monthly_weight', OFFICIAL_METRIC1_WEIGHTS['monthly']),
        ('accumulated_0_5_weight', OFFICIAL_METRIC1_WEIGHTS['accumulated_0_5']),
        ('accumulated_6_11_weight', OFFICIAL_METRIC1_WEIGHTS['accumulated_6_11']),
        ('accumulated_12_23_weight', OFFICIAL_METRIC1_WEIGHTS['accumulated_12_23']),
    ]:
        config_val = metric1.get(key)
        if config_val is not None and config_val != official_val:
            mismatches.append(f"metric1.{key}: config={config_val}, official={official_val}")
    
    # Check metric2 weights
    metric2 = official_metric.get('metric2', {})
    for key, official_val in [
        ('monthly_weight', OFFICIAL_METRIC2_WEIGHTS['monthly']),
        ('accumulated_6_11_weight', OFFICIAL_METRIC2_WEIGHTS['accumulated_6_11']),
        ('accumulated_12_23_weight', OFFICIAL_METRIC2_WEIGHTS['accumulated_12_23']),
    ]:
        config_val = metric2.get(key)
        if config_val is not None and config_val != official_val:
            mismatches.append(f"metric2.{key}: config={config_val}, official={official_val}")
    
    if mismatches:
        raise ValueError(
            f"Config values don't match official metric_calculation.py:\n  " +
            "\n  ".join(mismatches) +
            "\n\nUpdate configs/run_defaults.yaml to match the official script."
        )
    
    logger.info("Config metric values validated against official metric_calculation.py ✓")
    return True
