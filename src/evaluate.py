"""
Evaluation module for Novartis Datathon 2025.

Wraps the official metric_calculation.py for local validation.
Provides compute_metric1, compute_metric2, and helper functions.

IMPORTANT: Both df_actual and df_pred must contain ACTUAL volume (not normalized)
in a 'volume' column. The official metric operates on raw volumes, not y_norm.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

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
    scenario: str
) -> Dict[str, float]:
    """
    Compute metrics separately for Bucket 1 and Bucket 2.
    
    Args:
        df_actual: Actual volume data
        df_pred: Predicted volume data
        df_aux: Auxiliary data with bucket classification
        scenario: "scenario1" or "scenario2"
        
    Returns:
        {'overall': float, 'bucket1': float, 'bucket2': float}
    """
    # Overall metric
    if scenario == 'scenario1':
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
            if scenario == 'scenario1':
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
    
    pe_results = merged.groupby(["country", "brand_name", "bucket"]).apply(compute_pe).reset_index(name="PE")
    
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
    
    pe_results = merged.groupby(["country", "brand_name", "bucket"]).apply(compute_pe).reset_index(name="PE")
    
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
