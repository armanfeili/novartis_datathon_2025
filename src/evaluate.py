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

MAIN API SURFACE:
=================
Primary metric functions:
  - compute_metric1(df_actual, df_pred, df_aux) - Scenario 1 official metric
  - compute_metric2(df_actual, df_pred, df_aux) - Scenario 2 official metric
  - compute_bucket_metrics(...) - Per-bucket breakdown

Error analysis:
  - compute_per_series_error(...) - Per-series error metrics
  - identify_worst_series(...) - Find worst performing series
  - analyze_errors_by_bucket(...) - Bucket-level aggregates
  - analyze_errors_by_time_window(...) - Early/mid/late analysis
  - check_systematic_bias(...) - Over/under prediction detection

Metric logging (Section 6.7):
  - make_metric_record(...) - Create standardized metric record
  - save_metric_records(...) - Save to CSV
  - load_metric_records(...) - Load from CSV
  - get_metrics_for_run(...) - Filter metrics for analysis

Resilience analysis (Ghannem et al.):
  - compute_resilience_metrics(...) - Supply chain resilience
  - identify_high_risk_series(...) - High stock-out risk series
  - detect_under_forecast_patterns(...) - Systematic under-forecasting

Validation:
  - validate_config_matches_official(...) - Config vs official script alignment
  - create_aux_file(...) - Build auxiliary file for metrics
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Union

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


# =============================================================================
# SECTION B: INPUT VALIDATION AND NORMALIZATION HELPERS
# =============================================================================

# Valid phases for metric records (Section F.2)
# Note: phases are extensible - includes standard phases plus special cases
VALID_PHASES = {
    "train", "val", "cv", "simulation", "test_offline", "test_online",
    "adversarial_val",  # For adversarial validation
    "hpo",  # Hyperparameter optimization
}
# Valid splits - includes fold-based splits for cross-validation
# Note: splits are validated loosely to allow for special cases
VALID_SPLITS = {"train", "val", "test", "train_vs_test", "all"}
# Allow fold_N pattern for CV splits
VALID_SPLIT_PREFIXES = {"fold_"}  # Accepts fold_0, fold_1, ..., fold_N


def _normalize_scenario(scenario: Union[int, str, None]) -> int:
    """
    Normalize scenario to integer (1 or 2).
    
    Section B.2: Unified scenario handling across all functions.
    
    Accepts:
        - int: 1, 2
        - str: "1", "2", "scenario1", "scenario2", "s1", "s2"
        - None: Defaults to 1 with warning
        
    Args:
        scenario: Scenario identifier in various formats
        
    Returns:
        Normalized scenario as integer (1 or 2)
        
    Raises:
        ValueError: If scenario is not valid (except None which defaults to 1)
    """
    if scenario is None:
        logger.warning("_normalize_scenario: scenario is None, defaulting to 1")
        return 1
    
    if isinstance(scenario, int):
        if scenario not in {1, 2}:
            raise ValueError(f"Invalid scenario: {scenario}. Must be 1 or 2.")
        return scenario
    
    if isinstance(scenario, str):
        scenario_lower = scenario.lower().strip()
        if scenario_lower in {"1", "scenario1", "s1"}:
            return 1
        elif scenario_lower in {"2", "scenario2", "s2"}:
            return 2
        else:
            raise ValueError(
                f"Invalid scenario string: '{scenario}'. "
                "Expected: 1, 2, 'scenario1', 'scenario2', 's1', or 's2'."
            )
    
    raise ValueError(f"Invalid scenario type: {type(scenario)}. Expected int or str.")


def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    df_name: str,
    func_name: str = "unknown"
) -> None:
    """
    Validate that DataFrame has required columns.
    
    Section B.1: Common validation helper to avoid repetition.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        df_name: Name of DataFrame for error messages
        func_name: Name of calling function for error context
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{func_name}] {df_name} missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def _validate_predictions_columns(
    df: pd.DataFrame,
    expected_suffix: str = "_pred",
    volume_col: str = "volume"
) -> None:
    """
    Validate prediction DataFrame after merge.
    
    Section B.3: Helper for post-merge column validation.
    
    Checks that volume columns exist with expected suffixes after merge.
    
    Args:
        df: Merged DataFrame
        expected_suffix: Expected suffix for prediction column
        volume_col: Base volume column name
        
    Raises:
        ValueError: If expected columns are missing
    """
    actual_col = f"{volume_col}_actual"
    pred_col = f"{volume_col}{expected_suffix}"
    
    missing = []
    if actual_col not in df.columns:
        missing.append(actual_col)
    if pred_col not in df.columns:
        missing.append(pred_col)
    
    if missing:
        raise ValueError(
            f"After merge, expected columns {missing} not found. "
            f"Available: {sorted(df.columns)}. "
            "Check that source DataFrames have 'volume' column."
        )


def _get_time_windows_for_scenario(scenario: int) -> Dict[str, Tuple[int, int]]:
    """
    Get time window definitions for a scenario.
    
    Section E.4: Factor out time window logic for consistency.
    
    Scenario 1: 0-5 (early), 6-11 (mid), 12-23 (late)
    Scenario 2: 6-11 (early), 12-17 (mid), 18-23 (late)
    
    Args:
        scenario: 1 or 2
        
    Returns:
        Dict mapping window name to (start_month, end_month)
    """
    scenario = _normalize_scenario(scenario)
    
    if scenario == 1:
        return {
            'early': (0, 5),
            'mid': (6, 11),
            'late': (12, 23)
        }
    else:
        return {
            'early': (6, 11),
            'mid': (12, 17),
            'late': (18, 23)
        }


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
    scenario: Union[int, str]
) -> Dict[str, float]:
    """
    Compute metrics separately for Bucket 1 and Bucket 2.
    
    Section D.1: Uses fallback metric functions for bucket-level analysis.
    The overall metric uses official functions when available.
    
    Args:
        df_actual: Actual volume data [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data [country, brand_name, months_postgx, volume]
        df_aux: Auxiliary data with bucket classification [country, brand_name, avg_vol, bucket]
        scenario: 1, 2, "scenario1", "scenario2", "s1", or "s2"
        
    Returns:
        {'overall': float, 'bucket1': float, 'bucket2': float}
        
    Note:
        Bucket-level metrics always use fallback implementations to avoid
        issues with bucket weighting in the official metric. The overall
        metric uses official implementation when available.
    """
    # Section B.2: Normalize scenario
    scenario = _normalize_scenario(scenario)
    
    # Section D.2: Validate bucket column
    if 'bucket' not in df_aux.columns:
        raise ValueError("df_aux must contain 'bucket' column for bucket metrics")
    
    # Ensure bucket is integer type
    df_aux = df_aux.copy()
    if df_aux['bucket'].dtype != 'int64':
        df_aux['bucket'] = df_aux['bucket'].astype(int)
    
    # Overall metric
    if scenario == 1:
        overall = compute_metric1(df_actual, df_pred, df_aux)
    else:
        overall = compute_metric2(df_actual, df_pred, df_aux)
    
    # Per-bucket metrics
    bucket_metrics = {'overall': overall}
    
    for bucket in [1, 2]:
        # Section D.2: Deduplicate bucket series
        bucket_series = df_aux[df_aux['bucket'] == bucket][['country', 'brand_name']].drop_duplicates()
        
        if len(bucket_series) == 0:
            logger.warning(f"Bucket {bucket} has no series - returning NaN for bucket{bucket} metric")
            bucket_metrics[f'bucket{bucket}'] = np.nan
            continue
        
        # Filter to this bucket's series
        actual_bucket = df_actual.merge(bucket_series, on=['country', 'brand_name'])
        pred_bucket = df_pred.merge(bucket_series, on=['country', 'brand_name'])
        aux_bucket = df_aux[df_aux['bucket'] == bucket]
        
        if len(actual_bucket) == 0:
            logger.warning(f"Bucket {bucket} has no rows in df_actual after merge")
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
        scenario: 1, 2, "scenario1", "scenario2", "s1", or "s2"
        
    Returns:
        DataFrame with columns:
            [country, brand_name, bucket, avg_vol, mae, rmse, mape, 
             total_actual, total_pred, total_error, n_months]
             
    Note:
        - Section E.1: avg_vol and bucket come from df_aux; warns if NaN
        - Section E.1: Groups sorted by months_postgx for deterministic behavior
        - Section E.1: MAPE epsilon is configurable via the internal function
    """
    # Section B.2: Normalize scenario
    scenario = _normalize_scenario(scenario)
    
    # Section B.1: Validate required columns
    _validate_required_columns(df_actual, ['country', 'brand_name', 'months_postgx', 'volume'], 
                               'df_actual', 'compute_per_series_error')
    _validate_required_columns(df_pred, ['country', 'brand_name', 'months_postgx', 'volume'], 
                               'df_pred', 'compute_per_series_error')
    _validate_required_columns(df_aux, ['country', 'brand_name', 'avg_vol', 'bucket'], 
                               'df_aux', 'compute_per_series_error')
    
    # Merge actual and predicted
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_pred")
    ).merge(df_aux, on=["country", "brand_name"], how="left")
    
    # Section E.1: Warn about missing avg_vol or bucket
    if merged['avg_vol'].isna().any():
        n_missing = merged['avg_vol'].isna().sum()
        logger.warning(f"compute_per_series_error: {n_missing} rows have NaN avg_vol")
    if merged['bucket'].isna().any():
        n_missing = merged['bucket'].isna().sum()
        logger.warning(f"compute_per_series_error: {n_missing} rows have NaN bucket")
    
    # Filter based on scenario
    if scenario == 1:
        merged = merged[merged['months_postgx'] >= 0]
    else:
        merged = merged[merged['months_postgx'] >= 6]
    
    logger.info(f"compute_per_series_error: Processing {len(merged)} rows for scenario {scenario}")
    
    # Compute per-series metrics
    def compute_series_metrics(group):
        # Section E.1: Sort by months_postgx for deterministic behavior
        group = group.sort_values('months_postgx')
        
        actual = group['volume_actual'].values
        pred = group['volume_pred'].values
        avg_vol = group['avg_vol'].iloc[0]
        bucket = group['bucket'].iloc[0]
        
        # Handle NaN avg_vol
        if pd.isna(avg_vol):
            avg_vol = actual.mean() if len(actual) > 0 else 1.0
        
        error = pred - actual
        abs_error = np.abs(error)
        
        # Section E.1: Handle zeros in actual for MAPE
        # Use configurable epsilon to avoid division by zero
        mape_epsilon = 1e-8
        mape = (abs_error / (np.abs(actual) + mape_epsilon)).mean() * 100
        
        return pd.Series({
            'bucket': bucket,
            'avg_vol': avg_vol,
            'mae': abs_error.mean(),
            'rmse': np.sqrt((error ** 2).mean()),
            'mape': mape,
            'total_actual': actual.sum(),
            'total_pred': pred.sum(),
            'total_error': abs(actual.sum() - pred.sum()),
            'normalized_total_error': abs(actual.sum() - pred.sum()) / (avg_vol + 1e-8),
            'n_months': len(group)
        })
    
    per_series = merged.groupby(['country', 'brand_name']).apply(
        compute_series_metrics, include_groups=False
    ).reset_index()
    
    logger.info(f"compute_per_series_error: Computed metrics for {len(per_series)} series")
    
    return per_series


def identify_worst_series(
    per_series_errors: pd.DataFrame,
    metric: str = 'mae',
    top_k: int = 10,
    bucket: Optional[int] = None
) -> pd.DataFrame:
    """
    Identify worst-performing series based on specified metric.
    
    Section E.2: Validates inputs and handles edge cases robustly.
    
    Args:
        per_series_errors: Output from compute_per_series_error()
        metric: Metric to rank by ('mae', 'rmse', 'mape', 'normalized_total_error')
        top_k: Number of worst series to return (must be positive)
        bucket: If specified, filter to only this bucket (1 or 2)
        
    Returns:
        DataFrame with top_k worst series sorted by metric descending
        
    Raises:
        ValueError: If top_k is not positive or metric not found
    """
    # Section E.2: Validate top_k
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    
    df = per_series_errors.copy()
    
    if bucket is not None:
        df = df[df['bucket'] == bucket]
        if len(df) == 0:
            logger.warning(f"No series found in bucket {bucket} - returning empty DataFrame")
            return df
    
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")
    
    # Section E.2: Validate top_k against available series
    actual_k = min(top_k, len(df))
    if actual_k < top_k:
        logger.info(f"Requested top_k={top_k} but only {len(df)} series available")
    
    return df.nlargest(actual_k, metric)


def analyze_errors_by_bucket(
    per_series_errors: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate error statistics by bucket.
    
    Section E.3: Includes median and handles NaN gracefully.
    
    Args:
        per_series_errors: Output from compute_per_series_error()
        
    Returns:
        {
            'bucket1': {'n_series': int, 'mean_mae': float, 'median_mae': float, ...},
            'bucket2': {'n_series': int, 'mean_mae': float, 'median_mae': float, ...},
            'overall': {'n_series': int, 'mean_mae': float, 'median_mae': float, ...}
        }
    """
    result = {}
    
    for bucket_name, df_bucket in [('bucket1', per_series_errors[per_series_errors['bucket'] == 1]),
                                    ('bucket2', per_series_errors[per_series_errors['bucket'] == 2]),
                                    ('overall', per_series_errors)]:
        if len(df_bucket) == 0:
            result[bucket_name] = {'n_series': 0}
            continue
        
        # Section E.3: Handle NaN gracefully using nanmean/nanstd
        result[bucket_name] = {
            'n_series': len(df_bucket),
            'mean_mae': np.nanmean(df_bucket['mae']) if 'mae' in df_bucket else np.nan,
            'std_mae': np.nanstd(df_bucket['mae']) if 'mae' in df_bucket else np.nan,
            'mean_rmse': np.nanmean(df_bucket['rmse']) if 'rmse' in df_bucket else np.nan,
            'std_rmse': np.nanstd(df_bucket['rmse']) if 'rmse' in df_bucket else np.nan,
            'mean_mape': np.nanmean(df_bucket['mape']) if 'mape' in df_bucket else np.nan,
            'median_mae': np.nanmedian(df_bucket['mae']) if 'mae' in df_bucket else np.nan,
            'median_rmse': np.nanmedian(df_bucket['rmse']) if 'rmse' in df_bucket else np.nan,
            'max_mae': df_bucket['mae'].max() if 'mae' in df_bucket else np.nan,
            'mean_normalized_error': np.nanmean(df_bucket['normalized_total_error']) if 'normalized_total_error' in df_bucket else np.nan
        }
    
    return result


def analyze_errors_by_time_window(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    scenario: Union[int, str]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze errors by time window (early/mid/late).
    
    Section E.4: Time windows match the official metric definitions.
    
    Scenario 1 windows (Phase 1A):
        - early: months 0-5 (50% weight in official metric - CRITICAL)
        - mid: months 6-11 (20% weight)
        - late: months 12-23 (10% weight)
        
    Scenario 2 windows (Phase 1B):
        - early: months 6-11 (50% weight in official metric - CRITICAL)
        - mid: months 12-17 (part of 30% weight)
        - late: months 18-23 (part of 30% weight)
    
    Args:
        df_actual: Actual volume data [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data [country, brand_name, months_postgx, volume]
        scenario: 1, 2, "scenario1", "scenario2", "s1", or "s2"
        
    Returns:
        {
            'early': {'months': '0-5'/'6-11', 'n_rows': int, 'mae': float, ...},
            'mid': {'months': '6-11'/'12-17', 'n_rows': int, 'mae': float, ...},
            'late': {'months': '12-23'/'18-23', 'n_rows': int, 'mae': float, ...}
        }
    """
    # Section B.2: Normalize scenario
    scenario = _normalize_scenario(scenario)
    
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_pred")
    )
    
    # Section E.4: Use helper for time window definitions
    windows = _get_time_windows_for_scenario(scenario)
    
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
    
    Section E.5: Uses error = pred - actual (positive = overprediction).
    
    Sign convention:
        - Positive error = overprediction (pred > actual)
        - Negative error = underprediction (pred < actual)
    
    This convention is opposite to compute_forecast_bias which uses
    error = actual - pred (positive = under-forecast / stock-out risk).
    
    Args:
        df_actual: Actual volume data [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data [country, brand_name, months_postgx, volume]
        
    Returns:
        {
            'mean_error': float,           # Positive = overprediction
            'median_error': float,         # Positive = overprediction
            'pct_overprediction': float,   # % of predictions above actual (0-100)
            'pct_underprediction': float,  # % of predictions below actual (0-100)
            'bias_magnitude': float,       # |mean_error| / mean_actual (0-1+)
            'n_rows': int                  # Number of prediction points
        }
        
    Example:
        >>> bias = check_systematic_bias(df_actual, df_pred)
        >>> if bias['mean_error'] > 0:
        ...     print(f"Model overpredicts by {bias['mean_error']:.2f} on average")
        >>> else:
        ...     print(f"Model underpredicts by {abs(bias['mean_error']):.2f} on average")
    """
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_pred")
    )
    
    # Section E.5: Error = pred - actual (positive = overprediction)
    error = merged['volume_pred'] - merged['volume_actual']
    
    return {
        'mean_error': float(error.mean()),
        'median_error': float(error.median()),
        'pct_overprediction': float((error > 0).mean() * 100),
        'pct_underprediction': float((error < 0).mean() * 100),
        'bias_magnitude': float(abs(error.mean()) / (merged['volume_actual'].mean() + 1e-8)),
        'n_rows': len(merged)
    }


# =============================================================================
# CANONICAL METRIC NAME CONSTANTS (Section F.1)
# =============================================================================
# Use these constants throughout the codebase for consistent metric naming.
# This ensures metric records can be filtered and aggregated correctly.

METRIC_NAME_S1 = "metric1_official"      # Official Scenario 1 metric
METRIC_NAME_S2 = "metric2_official"      # Official Scenario 2 metric
METRIC_NAME_RMSE = "rmse_y_norm"         # RMSE on normalized target
METRIC_NAME_MAE = "mae_y_norm"           # MAE on normalized target
METRIC_NAME_MAPE = "mape_y_norm"         # MAPE on normalized target


# =============================================================================
# METRIC BREAKDOWN FUNCTIONS (Section 6.3)
# =============================================================================

def compute_metric_by_ther_area(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame,
    panel_df: pd.DataFrame,
    scenario
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by therapeutic area.
    
    Args:
        df_actual: Actual volume data with columns [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data with same columns
        df_aux: Auxiliary data with columns [country, brand_name, avg_vol, bucket]
        panel_df: Panel DataFrame containing ther_area column
        scenario: 1 or 2
        
    Returns:
        Dict mapping ther_area to {metric, n_series, mean_error, std_error}
    """
    # Get ther_area for each series
    series_ther_area = panel_df[['country', 'brand_name', 'ther_area']].drop_duplicates()
    
    # Normalize scenario
    if isinstance(scenario, str):
        scenario = int(scenario[-1]) if scenario.startswith('scenario') else int(scenario)
    
    results = {}
    
    for ther_area in series_ther_area['ther_area'].unique():
        area_series = series_ther_area[series_ther_area['ther_area'] == ther_area][['country', 'brand_name']]
        
        if len(area_series) == 0:
            continue
        
        # Filter actual, pred, aux to this therapeutic area
        actual_area = df_actual.merge(area_series, on=['country', 'brand_name'])
        pred_area = df_pred.merge(area_series, on=['country', 'brand_name'])
        aux_area = df_aux.merge(area_series, on=['country', 'brand_name'])
        
        if len(actual_area) == 0 or len(aux_area) == 0:
            continue
        
        try:
            if scenario == 1:
                metric = compute_metric1(actual_area, pred_area, aux_area)
            else:
                metric = compute_metric2(actual_area, pred_area, aux_area)
            
            # Compute additional statistics
            merged = actual_area.merge(
                pred_area, on=['country', 'brand_name', 'months_postgx'],
                suffixes=('_actual', '_pred')
            )
            error = merged['volume_pred'] - merged['volume_actual']
            
            results[ther_area] = {
                'metric': metric,
                'n_series': len(area_series),
                'n_rows': len(merged),
                'mean_error': error.mean(),
                'std_error': error.std(),
                'mean_abs_error': error.abs().mean()
            }
        except Exception as e:
            logger.warning(f"Could not compute metric for ther_area={ther_area}: {e}")
            results[ther_area] = {
                'metric': np.nan,
                'n_series': len(area_series),
                'error': str(e)
            }
    
    return results


def compute_metric_by_country(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame,
    scenario
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by country.
    
    Args:
        df_actual: Actual volume data with columns [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data with same columns
        df_aux: Auxiliary data with columns [country, brand_name, avg_vol, bucket]
        scenario: 1 or 2
        
    Returns:
        Dict mapping country to {metric, n_series, mean_error, std_error}
    """
    # Normalize scenario
    if isinstance(scenario, str):
        scenario = int(scenario[-1]) if scenario.startswith('scenario') else int(scenario)
    
    results = {}
    
    for country in df_actual['country'].unique():
        # Filter to this country
        actual_country = df_actual[df_actual['country'] == country]
        pred_country = df_pred[df_pred['country'] == country]
        aux_country = df_aux[df_aux['country'] == country]
        
        if len(actual_country) == 0 or len(aux_country) == 0:
            continue
        
        n_series = actual_country[['country', 'brand_name']].drop_duplicates().shape[0]
        
        try:
            if scenario == 1:
                metric = compute_metric1(actual_country, pred_country, aux_country)
            else:
                metric = compute_metric2(actual_country, pred_country, aux_country)
            
            # Compute additional statistics
            merged = actual_country.merge(
                pred_country, on=['country', 'brand_name', 'months_postgx'],
                suffixes=('_actual', '_pred')
            )
            error = merged['volume_pred'] - merged['volume_actual']
            
            results[country] = {
                'metric': metric,
                'n_series': n_series,
                'n_rows': len(merged),
                'mean_error': error.mean(),
                'std_error': error.std(),
                'mean_abs_error': error.abs().mean()
            }
        except Exception as e:
            logger.warning(f"Could not compute metric for country={country}: {e}")
            results[country] = {
                'metric': np.nan,
                'n_series': n_series,
                'error': str(e)
            }
    
    return results


def create_evaluation_dataframe(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame,
    panel_df: Optional[pd.DataFrame] = None,
    scenario: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a comprehensive evaluation DataFrame for analysis.
    
    Args:
        df_actual: Actual volume data
        df_pred: Predicted volume data
        df_aux: Auxiliary data
        panel_df: Optional panel with additional metadata (ther_area, etc.)
        scenario: Optional scenario number for labeling
        
    Returns:
        DataFrame with columns:
            series_id, country, brand_name, months_postgx, scenario, bucket,
            y_true, y_pred, error, abs_error, pct_error, avg_vol
    """
    # Merge actual and predicted
    merged = df_actual.merge(
        df_pred,
        on=['country', 'brand_name', 'months_postgx'],
        how='inner',
        suffixes=('_actual', '_pred')
    )
    
    # Rename columns
    merged = merged.rename(columns={
        'volume_actual': 'y_true',
        'volume_pred': 'y_pred'
    })
    
    # Add aux info
    merged = merged.merge(df_aux, on=['country', 'brand_name'], how='left')
    
    # Compute error metrics
    merged['error'] = merged['y_pred'] - merged['y_true']
    merged['abs_error'] = merged['error'].abs()
    merged['pct_error'] = merged['error'] / (merged['y_true'] + 1e-8) * 100
    
    # Add scenario
    if scenario is not None:
        merged['scenario'] = scenario
    
    # Add series_id
    merged['series_id'] = merged['country'] + '_' + merged['brand_name']
    
    # Add metadata from panel if available
    if panel_df is not None:
        meta_cols = ['country', 'brand_name', 'ther_area', 'main_package', 'biological']
        available_meta = [c for c in meta_cols if c in panel_df.columns]
        if len(available_meta) > 2:
            meta = panel_df[available_meta].drop_duplicates()
            merged = merged.merge(meta, on=['country', 'brand_name'], how='left')
    
    return merged

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


def validate_config_matches_official(
    run_config: dict,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that run_config metric values match official metric_calculation.py.
    
    Section H.1: Enhanced with float tolerance for comparison.
    
    Args:
        run_config: Loaded run_defaults.yaml config dict
        tolerance: Floating point comparison tolerance (default 1e-6)
        
    Returns:
        True if all values match within tolerance
        
    Raises:
        ValueError: If any mismatch is detected beyond tolerance
    """
    def floats_match(a: float, b: float) -> bool:
        """Compare floats with tolerance."""
        if a is None or b is None:
            return a is None and b is None
        return abs(a - b) < tolerance
    
    official_metric = run_config.get('official_metric', {})
    mismatches = []
    
    # Check bucket threshold
    config_threshold = official_metric.get('bucket_threshold')
    if config_threshold is not None and not floats_match(config_threshold, OFFICIAL_BUCKET_THRESHOLD):
        mismatches.append(
            f"bucket_threshold: config={config_threshold}, official={OFFICIAL_BUCKET_THRESHOLD}"
        )
    
    # Check bucket weights
    bucket_weights = official_metric.get('bucket_weights', {})
    if bucket_weights.get('bucket1') is not None:
        if not floats_match(bucket_weights.get('bucket1'), OFFICIAL_BUCKET1_WEIGHT):
            mismatches.append(
                f"bucket1_weight: config={bucket_weights.get('bucket1')}, official={OFFICIAL_BUCKET1_WEIGHT}"
            )
    if bucket_weights.get('bucket2') is not None:
        if not floats_match(bucket_weights.get('bucket2'), OFFICIAL_BUCKET2_WEIGHT):
            mismatches.append(
                f"bucket2_weight: config={bucket_weights.get('bucket2')}, official={OFFICIAL_BUCKET2_WEIGHT}"
            )
    
    # Check metric1 weights
    metric1 = official_metric.get('metric1', {})
    for key, official_val in [
        ('monthly_weight', OFFICIAL_METRIC1_WEIGHTS['monthly']),
        ('accumulated_0_5_weight', OFFICIAL_METRIC1_WEIGHTS['accumulated_0_5']),
        ('accumulated_6_11_weight', OFFICIAL_METRIC1_WEIGHTS['accumulated_6_11']),
        ('accumulated_12_23_weight', OFFICIAL_METRIC1_WEIGHTS['accumulated_12_23']),
    ]:
        config_val = metric1.get(key)
        if config_val is not None and not floats_match(config_val, official_val):
            mismatches.append(f"metric1.{key}: config={config_val}, official={official_val}")
    
    # Check metric2 weights
    metric2 = official_metric.get('metric2', {})
    for key, official_val in [
        ('monthly_weight', OFFICIAL_METRIC2_WEIGHTS['monthly']),
        ('accumulated_6_11_weight', OFFICIAL_METRIC2_WEIGHTS['accumulated_6_11']),
        ('accumulated_12_23_weight', OFFICIAL_METRIC2_WEIGHTS['accumulated_12_23']),
    ]:
        config_val = metric2.get(key)
        if config_val is not None and not floats_match(config_val, official_val):
            mismatches.append(f"metric2.{key}: config={config_val}, official={official_val}")
    
    if mismatches:
        raise ValueError(
            f"Config values don't match official metric_calculation.py (tolerance={tolerance}):\n  " +
            "\n  ".join(mismatches) +
            "\n\nUpdate configs/run_defaults.yaml to match the official script."
        )
    
    logger.info("validate_config_matches_official: Config metric values validated ✓")
    return True


# =============================================================================
# UNIFIED METRICS LOGGING SYSTEM (Section 6.7)
# =============================================================================

def make_metric_record(
    phase: str,
    split: str,
    scenario: Union[int, str],
    model_name: str,
    metric_name: str,
    value: float,
    run_id: Optional[str] = None,
    step: Optional[str] = None,
    bucket: Optional[int] = None,
    series_id: Optional[str] = None,
    extra: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create a standardized metric record.
    
    Section F.1-F.2: All phases (training, validation, CV, inference) use this
    function to ensure consistent metric schema with proper validation.
    
    Use canonical metric name constants for consistency:
        - METRIC_NAME_S1: Official Scenario 1 metric
        - METRIC_NAME_S2: Official Scenario 2 metric
        - METRIC_NAME_RMSE: RMSE on normalized target
        - METRIC_NAME_MAE: MAE on normalized target
        - METRIC_NAME_MAPE: MAPE on normalized target
    
    Args:
        phase: One of "train", "val", "cv", "simulation", "test_offline", "test_online",
               "adversarial_val", "hpo"
        split: "train", "val", "test", or "fold_N" (e.g., "fold_0", "fold_1")
        scenario: 1, 2, "scenario1", "scenario2", "s1", or "s2"
        model_name: E.g., "catboost", "lgbm"
        metric_name: Canonical metric name (use METRIC_NAME_* constants)
        value: Metric value (float, can be NaN or inf - will be converted to NaN)
        run_id: Unique run identifier (auto-generated if None)
        step: Epoch index, fold index, or "final"
        bucket: Optional bucket number (1 or 2)
        series_id: Optional series identifier (for per-series metrics)
        extra: Optional dict with additional info (JSON-serializable)
        
    Returns:
        Dict with standardized fields
        
    Raises:
        ValueError: If phase or split are not valid
    """
    from datetime import datetime, timezone
    
    # Section F.2: Validate phase
    if phase not in VALID_PHASES:
        raise ValueError(
            f"Invalid phase: '{phase}'. "
            f"Valid phases: {sorted(VALID_PHASES)}"
        )
    
    # Section F.2: Validate split (allow fold_N pattern for CV)
    def _is_valid_split(s: str) -> bool:
        if s in VALID_SPLITS:
            return True
        # Check if it's a fold pattern like "fold_0", "fold_1", etc.
        for prefix in VALID_SPLIT_PREFIXES:
            if s.startswith(prefix):
                # Verify the suffix is a valid number
                suffix = s[len(prefix):]
                if suffix.isdigit():
                    return True
        return False
    
    if not _is_valid_split(split):
        raise ValueError(
            f"Invalid split: '{split}'. "
            f"Valid splits: {sorted(VALID_SPLITS)} or 'fold_N' (e.g., fold_0, fold_1)"
        )
    
    # Section B.2: Normalize scenario
    scenario = _normalize_scenario(scenario)
    
    # Section F.2: Handle value edge cases
    if value is None:
        final_value = np.nan
    elif np.isinf(value):
        logger.warning(f"make_metric_record: value is inf for metric={metric_name}, converting to NaN")
        final_value = np.nan
    elif np.isnan(value):
        final_value = np.nan
    else:
        final_value = float(value)
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    record = {
        'run_id': run_id,
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'phase': phase,
        'split': split,
        'scenario': scenario,
        'model': model_name,
        'metric': metric_name,
        'value': final_value,
        'step': step,
        'bucket': bucket,
        'series_id': series_id,
        'extra': extra
    }
    
    return record


def save_metric_records(
    records: List[Dict],
    path: Path,
    append: bool = True
) -> None:
    """
    Save metric records to a CSV file.
    
    Section F.3: Creates parent directories if missing. Appends to existing file
    if append=True, otherwise overwrites. Logs path and record count.
    
    Args:
        records: List of metric record dicts (from make_metric_record)
        path: Path to output CSV file
        append: Whether to append to existing file
    """
    import json
    
    if not records:
        logger.warning("save_metric_records: No records to save")
        return
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert records to DataFrame
    df = pd.DataFrame(records)
    
    # Serialize 'extra' column as JSON string
    if 'extra' in df.columns:
        df['extra'] = df['extra'].apply(
            lambda x: json.dumps(x) if x is not None else None
        )
    
    # Define column order for consistency
    column_order = [
        'run_id', 'timestamp', 'phase', 'split', 'scenario', 'model',
        'metric', 'value', 'step', 'bucket', 'series_id', 'extra'
    ]
    
    # Ensure all columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    
    df = df[column_order]
    
    # Write to file
    if append and path.exists():
        df.to_csv(path, mode='a', header=False, index=False)
        logger.debug(f"Appended {len(records)} records to {path}")
    else:
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(records)} records to {path}")


def load_metric_records(path: Path) -> pd.DataFrame:
    """
    Load metric records from a CSV file.
    
    Section F.3: Resilient loading with JSON parsing for 'extra' column.
    Returns empty DataFrame if file not found or empty.
    
    Args:
        path: Path to metrics CSV file
        
    Returns:
        DataFrame with metric records (empty DataFrame if file not found)
        
    Note:
        - Returns empty DataFrame with proper columns if file doesn't exist
        - Handles empty files gracefully
        - Logs warnings for malformed JSON in 'extra' column
    """
    import json
    
    path = Path(path)
    
    # Section F.3: Return empty DataFrame if file doesn't exist
    if not path.exists():
        logger.warning(f"load_metric_records: Metrics file not found: {path}, returning empty DataFrame")
        return pd.DataFrame(columns=[
            'run_id', 'timestamp', 'phase', 'split', 'scenario', 'model',
            'metric', 'value', 'step', 'bucket', 'series_id', 'extra'
        ])
    
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        logger.warning(f"load_metric_records: Empty file: {path}, returning empty DataFrame")
        return pd.DataFrame(columns=[
            'run_id', 'timestamp', 'phase', 'split', 'scenario', 'model',
            'metric', 'value', 'step', 'bucket', 'series_id', 'extra'
        ])
    
    # Section F.3: Deserialize 'extra' column from JSON with error handling
    if 'extra' in df.columns:
        def safe_json_loads(x):
            if pd.isna(x) or x == '':
                return None
            try:
                return json.loads(x)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"load_metric_records: Failed to parse JSON in 'extra': {x}")
                return None
        
        df['extra'] = df['extra'].apply(safe_json_loads)
    
    logger.debug(f"load_metric_records: Loaded {len(df)} records from {path}")
    return df


def get_metrics_for_run(
    metrics_dir: Path,
    run_id: str,
    phase: Optional[str] = None,
    metric: Optional[str] = None,
    scenario: Optional[Union[int, str]] = None,
    split: Optional[str] = None,
    bucket: Optional[int] = None,
    model: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and filter metrics for a specific run.
    
    Section F.4: Extended with scenario, split, bucket, model filters.
    
    Args:
        metrics_dir: Directory containing metrics files
        run_id: Run identifier
        phase: Optional phase filter ("train", "val", "cv", etc.)
        metric: Optional metric name filter
        scenario: Optional scenario filter (1, 2, "scenario1", "scenario2")
        split: Optional split filter ("train", "val", "test")
        bucket: Optional bucket filter (1 or 2)
        model: Optional model name filter
        
    Returns:
        Filtered DataFrame (empty if no matches or file not found)
    """
    metrics_path = Path(metrics_dir) / 'metrics.csv'
    
    df = load_metric_records(metrics_path)
    
    if df.empty:
        return df
    
    df = df[df['run_id'] == run_id]
    
    if phase is not None:
        df = df[df['phase'] == phase]
    
    if metric is not None:
        df = df[df['metric'] == metric]
    
    # Section F.4: Additional filters
    if scenario is not None:
        norm_scenario = _normalize_scenario(scenario)
        df = df[df['scenario'] == norm_scenario]
    
    if split is not None:
        df = df[df['split'] == split]
    
    if bucket is not None:
        df = df[df['bucket'] == bucket]
    
    if model is not None:
        df = df[df['model'] == model]
    
    logger.debug(f"get_metrics_for_run: Filtered to {len(df)} records for run_id={run_id}")
    return df


def create_per_series_metrics_df(
    per_series_errors: pd.DataFrame,
    scenario: Union[int, str],
    run_id: Optional[str] = None,
    model_name: str = 'unknown'
) -> pd.DataFrame:
    """
    Convert per-series error DataFrame to standardized metrics format.
    
    Section F.5: Fixed to handle missing columns gracefully and normalize scenario.
    
    Args:
        per_series_errors: Output from compute_per_series_error()
        scenario: 1, 2, "scenario1", "scenario2", etc.
        run_id: Run identifier (auto-generated if None)
        model_name: Model name
        
    Returns:
        DataFrame with standardized per-series metrics
        
    Raises:
        ValueError: If required columns (country, brand_name) are missing
    """
    from datetime import datetime, timezone
    
    # Section F.5: Normalize scenario
    scenario = _normalize_scenario(scenario)
    
    # Section F.5: Validate required columns
    _validate_required_columns(
        per_series_errors,
        ['country', 'brand_name'],
        'per_series_errors',
        'create_per_series_metrics_df'
    )
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    records = []
    
    # Metric columns to convert (only include if present)
    metric_columns = ['mae', 'rmse', 'mape', 'normalized_total_error']
    available_metrics = [m for m in metric_columns if m in per_series_errors.columns]
    
    if not available_metrics:
        logger.warning(
            f"create_per_series_metrics_df: No metric columns found in per_series_errors. "
            f"Expected: {metric_columns}"
        )
    
    for _, row in per_series_errors.iterrows():
        series_id = f"{row['country']}_{row['brand_name']}"
        bucket = row.get('bucket') if 'bucket' in per_series_errors.columns else None
        
        # Add each metric as a separate record
        for metric_name in available_metrics:
            value = row[metric_name]
            # Handle NaN and inf values
            if pd.isna(value) or np.isinf(value):
                final_value = np.nan
            else:
                final_value = float(value)
            
            records.append({
                'run_id': run_id,
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'scenario': scenario,
                'model': model_name,
                'series_id': series_id,
                'country': row['country'],
                'brand_name': row['brand_name'],
                'bucket': bucket,
                'metric': metric_name,
                'value': final_value
            })
    
    result = pd.DataFrame(records)
    logger.debug(f"create_per_series_metrics_df: Created {len(result)} metric records from {len(per_series_errors)} series")
    return result


def compute_diagnostic_metrics(
    predictions: pd.DataFrame,
    column: str = 'volume'
) -> Dict[str, float]:
    """
    Compute diagnostic metrics for predictions when ground truth is not available.
    
    Used for test_online phase where we can't compute error metrics.
    
    Args:
        predictions: DataFrame with predicted values
        column: Column name containing predictions
        
    Returns:
        Dict with pred_mean, pred_std, pred_min, pred_max, pred_median
    """
    values = predictions[column]
    
    return {
        'pred_mean': values.mean(),
        'pred_std': values.std(),
        'pred_min': values.min(),
        'pred_max': values.max(),
        'pred_median': values.median(),
        'pred_count': len(values),
        'pred_null_count': values.isna().sum()
    }


# =============================================================================
# RESILIENCE METRICS (Ghannem et al., 2023)
# =============================================================================

from dataclasses import dataclass


@dataclass
class ResilienceResult:
    """
    Result container for resilience metrics computation.
    
    Attributes:
        scenario: Scenario number (1 or 2)
        series_key: Tuple (country, brand_name)
        under_forecast_count: Number of months with under-forecasting
        under_forecast_pct: Percentage of months with under-forecasting
        max_under_forecast: Maximum under-forecast magnitude
        avg_under_forecast: Average under-forecast magnitude
        stock_out_risk_score: Estimated stock-out risk (0-1)
        recovery_time: Months to recover from max under-forecast
        resilience_score: Overall resilience score (0-1, higher is better)
    """
    scenario: int
    series_key: Tuple[str, str]
    under_forecast_count: int
    under_forecast_pct: float
    max_under_forecast: float
    avg_under_forecast: float
    stock_out_risk_score: float
    recovery_time: int
    resilience_score: float


def compute_resilience_metrics(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    scenario: Union[int, str] = 1,
    under_forecast_threshold: float = 0.0,
    critical_under_forecast_pct: float = 0.1,
) -> Dict[str, Any]:
    """
    Compute supply-chain resilience metrics for forecasts.
    
    Section G.1: Enhanced with scenario normalization and input validation.
    
    Based on Ghannem et al. (2023) - focuses on under-forecasting risk
    which leads to stock-outs and supply chain disruptions.
    
    Args:
        df_actual: Actual volume data [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data [country, brand_name, months_postgx, volume]
        scenario: 1, 2, "scenario1", "scenario2", etc.
        under_forecast_threshold: Error threshold to consider as under-forecast
                                 (default 0: any pred < actual is under-forecast)
        critical_under_forecast_pct: Threshold for critical under-forecast 
                                     (as fraction of actual volume)
        
    Returns:
        Dict containing:
        - overall: Aggregate resilience metrics
        - per_series: List of ResilienceResult per series
        - summary: Summary statistics (DataFrame)
        
    Raises:
        ValueError: If required columns are missing
    """
    # Section G.1: Normalize scenario
    scenario = _normalize_scenario(scenario)
    
    # Section G.1: Validate required columns
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    _validate_required_columns(df_actual, required_cols, 'df_actual', 'compute_resilience_metrics')
    _validate_required_columns(df_pred, required_cols, 'df_pred', 'compute_resilience_metrics')
    
    # Merge actual and predicted
    merged = df_actual.merge(
        df_pred,
        on=['country', 'brand_name', 'months_postgx'],
        suffixes=('_actual', '_pred'),
        how='inner'
    )
    
    if merged.empty:
        logger.warning("compute_resilience_metrics: No matching rows after merge, returning empty results")
        return {
            'overall': {
                'total_series': 0,
                'total_under_forecasts': 0,
                'avg_under_forecast_pct': np.nan,
                'avg_stock_out_risk': np.nan,
                'avg_resilience_score': np.nan,
                'min_resilience_score': np.nan,
                'pct_high_risk_series': np.nan,
            },
            'per_series': [],
            'summary': pd.DataFrame(),
        }
    
    # Compute forecast error (positive = under-forecast)
    merged['error'] = merged['volume_actual'] - merged['volume_pred']
    merged['error_pct'] = merged['error'] / (merged['volume_actual'] + 1e-6)
    
    # Identify under-forecasts
    merged['is_under_forecast'] = merged['error'] > under_forecast_threshold
    merged['is_critical_under'] = merged['error_pct'] > critical_under_forecast_pct
    
    # Per-series resilience
    series_keys = ['country', 'brand_name']
    per_series_results = []
    
    for (country, brand), group in merged.groupby(series_keys):
        n_months = len(group)
        under_count = group['is_under_forecast'].sum()
        
        # Under-forecast metrics
        under_forecast_pct = under_count / n_months if n_months > 0 else 0
        
        under_errors = group.loc[group['is_under_forecast'], 'error_pct']
        max_under = under_errors.max() if len(under_errors) > 0 else 0
        avg_under = under_errors.mean() if len(under_errors) > 0 else 0
        
        # Stock-out risk score (based on frequency and severity of under-forecasts)
        critical_count = group['is_critical_under'].sum()
        stock_out_risk = (0.6 * under_forecast_pct + 
                         0.4 * (critical_count / n_months if n_months > 0 else 0))
        
        # Recovery time: months after max under-forecast to reach normal error
        sorted_group = group.sort_values('months_postgx')
        max_under_idx = sorted_group['error_pct'].idxmax() if len(sorted_group) > 0 else None
        recovery_time = 0
        
        if max_under_idx is not None and max_under > critical_under_forecast_pct:
            after_max = sorted_group.loc[max_under_idx:, 'is_critical_under']
            # Count months until first non-critical month
            for i, is_critical in enumerate(after_max):
                if not is_critical:
                    recovery_time = i
                    break
            else:
                recovery_time = len(after_max)  # Never recovered
        
        # Resilience score (higher is better)
        resilience_score = 1.0 - (
            0.4 * under_forecast_pct +
            0.3 * min(max_under, 1.0) +
            0.2 * stock_out_risk +
            0.1 * min(recovery_time / 12, 1.0)
        )
        resilience_score = max(0, min(1, resilience_score))
        
        result = ResilienceResult(
            scenario=scenario,
            series_key=(country, brand),
            under_forecast_count=int(under_count),
            under_forecast_pct=float(under_forecast_pct),
            max_under_forecast=float(max_under),
            avg_under_forecast=float(avg_under),
            stock_out_risk_score=float(stock_out_risk),
            recovery_time=int(recovery_time),
            resilience_score=float(resilience_score)
        )
        per_series_results.append(result)
    
    # Overall metrics
    overall = {
        'total_series': len(per_series_results),
        'total_under_forecasts': sum(r.under_forecast_count for r in per_series_results),
        'avg_under_forecast_pct': np.mean([r.under_forecast_pct for r in per_series_results]),
        'avg_stock_out_risk': np.mean([r.stock_out_risk_score for r in per_series_results]),
        'avg_resilience_score': np.mean([r.resilience_score for r in per_series_results]),
        'min_resilience_score': min(r.resilience_score for r in per_series_results) if per_series_results else 0,
        'pct_high_risk_series': sum(1 for r in per_series_results if r.stock_out_risk_score > 0.3) / len(per_series_results) if per_series_results else 0,
    }
    
    # Summary DataFrame
    summary_df = pd.DataFrame([
        {
            'country': r.series_key[0],
            'brand_name': r.series_key[1],
            'under_forecast_count': r.under_forecast_count,
            'under_forecast_pct': r.under_forecast_pct,
            'max_under_forecast': r.max_under_forecast,
            'stock_out_risk_score': r.stock_out_risk_score,
            'resilience_score': r.resilience_score,
        }
        for r in per_series_results
    ])
    
    return {
        'overall': overall,
        'per_series': per_series_results,
        'summary': summary_df,
    }


def detect_under_forecast_patterns(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    threshold_pct: float = 0.1,
) -> pd.DataFrame:
    """
    Detect patterns in under-forecasting across series.
    
    Section G.2: Enhanced with input validation and empty data handling.
    
    Identifies which series/months are most prone to under-forecasting
    and potential systematic biases.
    
    Args:
        df_actual: Actual volume data [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data [country, brand_name, months_postgx, volume]
        threshold_pct: Threshold percentage for under-forecasting
        
    Returns:
        DataFrame with under-forecast analysis per month
        
    Raises:
        ValueError: If required columns are missing
    """
    # Section G.2: Validate required columns
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    _validate_required_columns(df_actual, required_cols, 'df_actual', 'detect_under_forecast_patterns')
    _validate_required_columns(df_pred, required_cols, 'df_pred', 'detect_under_forecast_patterns')
    
    merged = df_actual.merge(
        df_pred,
        on=['country', 'brand_name', 'months_postgx'],
        suffixes=('_actual', '_pred'),
        how='inner'
    )
    
    if merged.empty:
        logger.warning("detect_under_forecast_patterns: No matching rows after merge")
        return pd.DataFrame(columns=[
            'months_postgx', 'under_forecast_count', 'under_forecast_rate',
            'mean_error_pct', 'std_error_pct', 'max_error_pct', 'mean_error'
        ])
    
    merged['error'] = merged['volume_actual'] - merged['volume_pred']
    merged['error_pct'] = merged['error'] / (merged['volume_actual'] + 1e-6)
    merged['is_under_forecast'] = merged['error_pct'] > threshold_pct
    
    # Aggregate by month
    monthly = merged.groupby('months_postgx').agg({
        'is_under_forecast': ['sum', 'mean'],
        'error_pct': ['mean', 'std', 'max'],
        'error': 'mean'
    }).reset_index()
    
    monthly.columns = [
        'months_postgx',
        'under_forecast_count',
        'under_forecast_rate',
        'mean_error_pct',
        'std_error_pct',
        'max_error_pct',
        'mean_error'
    ]
    
    logger.debug(f"detect_under_forecast_patterns: Analyzed {len(monthly)} months")
    return monthly


def identify_high_risk_series(
    resilience_results: Dict[str, Any],
    risk_threshold: float = 0.3,
    top_n: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Identify series with high stock-out risk.
    
    Section G.3: Enhanced with top_n filtering and sorting by risk score.
    
    Args:
        resilience_results: Output from compute_resilience_metrics()
        risk_threshold: Stock-out risk threshold for high-risk classification
        top_n: Return only top N highest risk series (None for all above threshold)
        
    Returns:
        List of (country, brand_name) tuples for high-risk series,
        sorted by risk score (descending)
    """
    per_series = resilience_results.get('per_series', [])
    
    if not per_series:
        logger.warning("identify_high_risk_series: No per_series results provided")
        return []
    
    # Filter and sort by risk score
    high_risk = sorted(
        [r for r in per_series if r.stock_out_risk_score >= risk_threshold],
        key=lambda r: r.stock_out_risk_score,
        reverse=True
    )
    
    # Apply top_n limit if specified
    if top_n is not None:
        high_risk = high_risk[:top_n]
    
    result = [r.series_key for r in high_risk]
    
    logger.debug(f"identify_high_risk_series: Found {len(result)} high-risk series (threshold={risk_threshold})")
    return result


def compute_forecast_bias(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute forecast bias metrics to detect systematic under/over-forecasting.
    
    Section G.4: Enhanced with input validation.
    
    Args:
        df_actual: Actual volume data [country, brand_name, months_postgx, volume]
        df_pred: Predicted volume data [country, brand_name, months_postgx, volume]
        
    Returns:
        Dict with bias metrics:
        - mean_error: Average error (actual - predicted)
        - mean_bias_ratio: Error relative to mean actual volume
        - pct_under_forecast: Fraction of points with under-forecasting
        - pct_over_forecast: Fraction of points with over-forecasting
        - pct_exact: Fraction with zero error
        - bias_direction: "under", "over", or "balanced"
        
    Raises:
        ValueError: If required columns are missing
    """
    # Section G.4: Validate required columns
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    _validate_required_columns(df_actual, required_cols, 'df_actual', 'compute_forecast_bias')
    _validate_required_columns(df_pred, required_cols, 'df_pred', 'compute_forecast_bias')
    
    merged = df_actual.merge(
        df_pred,
        on=['country', 'brand_name', 'months_postgx'],
        suffixes=('_actual', '_pred'),
        how='inner'
    )
    
    if merged.empty:
        logger.warning("compute_forecast_bias: No matching rows after merge")
        return {
            'mean_error': np.nan,
            'mean_bias_ratio': np.nan,
            'pct_under_forecast': np.nan,
            'pct_over_forecast': np.nan,
            'pct_exact': np.nan,
            'bias_direction': 'unknown',
        }
    
    errors = merged['volume_actual'] - merged['volume_pred']
    
    mean_error = errors.mean()
    pct_under = (errors > 0).mean()  # Positive error = under-forecast
    pct_over = (errors < 0).mean()   # Negative error = over-forecast
    
    # Mean bias ratio (positive = under-forecasting, negative = over-forecasting)
    mean_bias_ratio = mean_error / (merged['volume_actual'].mean() + 1e-6)
    
    bias_direction = 'under' if mean_bias_ratio > 0.01 else ('over' if mean_bias_ratio < -0.01 else 'balanced')
    
    logger.debug(f"compute_forecast_bias: bias_direction={bias_direction}, mean_bias_ratio={mean_bias_ratio:.4f}")
    
    return {
        'mean_error': float(mean_error),
        'mean_bias_ratio': float(mean_bias_ratio),
        'pct_under_forecast': float(pct_under),
        'pct_over_forecast': float(pct_over),
        'pct_exact': float(1 - pct_under - pct_over),
        'bias_direction': bias_direction,
    }
