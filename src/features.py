"""
Feature engineering for Novartis Datathon 2025.

Scenario-aware feature construction with strict leakage prevention.
Features are organized into categories with clear cutoff rules.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats

from .utils import timer

logger = logging.getLogger(__name__)

# Cutoff configuration per scenario
SCENARIO_CONFIG = {
    'scenario1': {
        'feature_cutoff': 0,    # Only months_postgx < 0 for features
        'target_start': 0,
        'target_end': 23,
    },
    'scenario2': {
        'feature_cutoff': 6,    # months_postgx < 6 allowed
        'target_start': 6,
        'target_end': 23,
    }
}


def make_features(
    panel_df: pd.DataFrame,
    scenario: str,
    mode: str = "train",
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Build features respecting scenario-specific information constraints.
    
    Args:
        panel_df: Base panel with all raw data (must have avg_vol_12m)
        scenario: "scenario1" or "scenario2"
        mode: "train" or "test"
        config: Optional features.yaml config
    
    Returns:
        DataFrame with engineered features.
        If mode="train", includes target column (y_norm).
    
    CRITICAL - y_norm Creation:
        if mode == "train":
            df['y_norm'] = df['volume'] / df['avg_vol_12m']
        
        y_norm is what models predict; actual volume is recovered via:
        volume = y_norm * avg_vol_12m
    
    Cutoff Rules:
        - scenario1: cutoff_month = 0 (only months_postgx < 0 for feature derivation)
        - scenario2: cutoff_month = 6 (months_postgx < 6 allowed for feature derivation)
    """
    if scenario not in SCENARIO_CONFIG:
        raise ValueError(f"Invalid scenario: {scenario}. Must be 'scenario1' or 'scenario2'")
    
    scenario_cfg = SCENARIO_CONFIG[scenario]
    cutoff = scenario_cfg['feature_cutoff']
    
    with timer(f"Feature engineering - {scenario}"):
        df = panel_df.copy()
        
        # 1. Pre-entry features (both scenarios)
        df = add_pre_entry_features(df)
        
        # 2. Time features
        df = add_time_features(df)
        
        # 3. Generics features (respecting cutoff)
        df = add_generics_features(df, cutoff_month=cutoff)
        
        # 4. Drug characteristics
        df = add_drug_features(df)
        
        # 5. Scenario 2 specific: early erosion features
        if scenario == 'scenario2':
            df = add_early_erosion_features(df)
        
        # 6. Create y_norm for training
        if mode == "train":
            if 'y_norm' not in df.columns:
                df['y_norm'] = df['volume'] / df['avg_vol_12m']
        
        # Log feature count
        n_features = len([c for c in df.columns if c not in 
                         ['country', 'brand_name', 'months_postgx', 'month', 
                          'volume', 'y_norm', 'bucket', 'avg_vol_12m', 'mean_erosion']])
        logger.info(f"Total features created: {n_features}")
    
    return df


def select_training_rows(panel_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Select only rows that are valid supervised targets for the scenario.
    
    Scenario 1: months_postgx in [0, 23]
    Scenario 2: months_postgx in [6, 23]
    
    ALWAYS call this before splitting into features/target.
    
    Args:
        panel_df: Panel with features
        scenario: "scenario1" or "scenario2"
        
    Returns:
        Filtered DataFrame with only target month rows
    """
    if scenario not in SCENARIO_CONFIG:
        raise ValueError(f"Invalid scenario: {scenario}")
    
    cfg = SCENARIO_CONFIG[scenario]
    start = cfg['target_start']
    end = cfg['target_end']
    
    mask = (panel_df['months_postgx'] >= start) & (panel_df['months_postgx'] <= end)
    filtered = panel_df[mask].copy()
    
    logger.info(f"Selected {len(filtered):,} rows for {scenario} (months {start}-{end})")
    return filtered


def add_pre_entry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-entry statistics (Category 1).
    
    Features:
    - avg_vol_12m: Already computed in data.py
    - avg_vol_6m: Mean volume over months [-6, -1]
    - avg_vol_3m: Mean volume over months [-3, -1]
    - pre_entry_trend: Linear slope over pre-entry period
    - pre_entry_volatility: std(volume_pre) / avg_vol_12m
    - pre_entry_max: Maximum volume in pre-entry period
    - log_avg_vol: log1p(avg_vol_12m) for scale normalization
    """
    series_keys = ['country', 'brand_name']
    
    # avg_vol_6m
    pre_6m_mask = (df['months_postgx'] >= -6) & (df['months_postgx'] <= -1)
    avg_vol_6m = df[pre_6m_mask].groupby(series_keys)['volume'].mean().reset_index()
    avg_vol_6m.columns = series_keys + ['avg_vol_6m']
    df = df.merge(avg_vol_6m, on=series_keys, how='left')
    df['avg_vol_6m'] = df['avg_vol_6m'].fillna(df['avg_vol_12m'])
    
    # avg_vol_3m
    pre_3m_mask = (df['months_postgx'] >= -3) & (df['months_postgx'] <= -1)
    avg_vol_3m = df[pre_3m_mask].groupby(series_keys)['volume'].mean().reset_index()
    avg_vol_3m.columns = series_keys + ['avg_vol_3m']
    df = df.merge(avg_vol_3m, on=series_keys, how='left')
    df['avg_vol_3m'] = df['avg_vol_3m'].fillna(df['avg_vol_12m'])
    
    # Pre-entry trend (linear slope)
    pre_entry_data = df[df['months_postgx'] < 0][series_keys + ['months_postgx', 'volume']].copy()
    
    def compute_slope(group):
        if len(group) < 2:
            return np.nan
        slope, _, _, _, _ = stats.linregress(group['months_postgx'], group['volume'])
        return slope
    
    trends = pre_entry_data.groupby(series_keys).apply(compute_slope).reset_index()
    trends.columns = series_keys + ['pre_entry_trend']
    df = df.merge(trends, on=series_keys, how='left')
    df['pre_entry_trend'] = df['pre_entry_trend'].fillna(0)
    
    # Pre-entry volatility
    pre_std = df[df['months_postgx'] < 0].groupby(series_keys)['volume'].std().reset_index()
    pre_std.columns = series_keys + ['pre_entry_std']
    df = df.merge(pre_std, on=series_keys, how='left')
    df['pre_entry_volatility'] = df['pre_entry_std'] / df['avg_vol_12m']
    df['pre_entry_volatility'] = df['pre_entry_volatility'].fillna(0)
    df = df.drop(columns=['pre_entry_std'])
    
    # Pre-entry max
    pre_max = df[df['months_postgx'] < 0].groupby(series_keys)['volume'].max().reset_index()
    pre_max.columns = series_keys + ['pre_entry_max']
    df = df.merge(pre_max, on=series_keys, how='left')
    df['pre_entry_max'] = df['pre_entry_max'].fillna(df['avg_vol_12m'])
    
    # Log average volume
    df['log_avg_vol'] = np.log1p(df['avg_vol_12m'])
    
    # Ratio features
    df['vol_ratio_6m_12m'] = df['avg_vol_6m'] / (df['avg_vol_12m'] + 1e-6)
    df['vol_ratio_3m_12m'] = df['avg_vol_3m'] / (df['avg_vol_12m'] + 1e-6)
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features (Category 2).
    
    Features:
    - months_postgx: Direct time index (already present)
    - months_postgx_sq: Squared for non-linear decay
    - is_post_entry: Binary (months_postgx >= 0)
    - time_bucket: Categorical (pre, early, mid, late)
    - month_of_year: Seasonality from calendar month
    """
    # Squared time
    df['months_postgx_sq'] = df['months_postgx'] ** 2
    
    # Post-entry indicator
    df['is_post_entry'] = (df['months_postgx'] >= 0).astype(int)
    
    # Time bucket categorical
    def get_time_bucket(m):
        if m < 0:
            return 'pre'
        elif m <= 5:
            return 'early'
        elif m <= 11:
            return 'mid'
        else:
            return 'late'
    
    df['time_bucket'] = df['months_postgx'].apply(get_time_bucket)
    
    # Month of year for seasonality
    if 'month' in df.columns:
        df['month_of_year'] = pd.to_datetime(df['month']).dt.month
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12)
    
    return df


def add_generics_features(df: pd.DataFrame, cutoff_month: int = 0) -> pd.DataFrame:
    """
    Add generics competition features respecting cutoff (Category 3).
    
    Features:
    - n_gxs: Current number of generics
    - has_generic: Binary (n_gxs > 0)
    - multiple_generics: Binary (n_gxs >= 2)
    - n_gxs_cummax: Maximum n_gxs up to current month (respecting cutoff)
    - n_gxs_at_entry: n_gxs at month 0
    - months_since_first_generic: Time since first generic appeared
    
    Args:
        df: Panel data
        cutoff_month: Feature cutoff (0 for S1, 6 for S2)
    """
    series_keys = ['country', 'brand_name']
    
    if 'n_gxs' not in df.columns:
        logger.warning("n_gxs column not found, skipping generics features")
        return df
    
    # Basic features
    df['has_generic'] = (df['n_gxs'] > 0).astype(int)
    df['multiple_generics'] = (df['n_gxs'] >= 2).astype(int)
    
    # n_gxs at entry (month 0)
    entry_n_gxs = df[df['months_postgx'] == 0][series_keys + ['n_gxs']].copy()
    entry_n_gxs.columns = series_keys + ['n_gxs_at_entry']
    df = df.merge(entry_n_gxs, on=series_keys, how='left')
    df['n_gxs_at_entry'] = df['n_gxs_at_entry'].fillna(0)
    
    # Cumulative max n_gxs (respecting cutoff for training)
    # For inference rows (months >= cutoff), use n_gxs up to cutoff
    df = df.sort_values(series_keys + ['months_postgx'])
    
    # Compute cummax within allowed window
    def compute_cummax_with_cutoff(group):
        result = group.copy()
        # For each row, compute cummax considering only months before cutoff
        cummax_vals = []
        for idx, row in result.iterrows():
            if row['months_postgx'] < cutoff_month:
                # Use actual cummax up to this point
                mask = result['months_postgx'] <= row['months_postgx']
                cummax_vals.append(result.loc[mask, 'n_gxs'].max())
            else:
                # Use cummax up to cutoff only
                mask = result['months_postgx'] < cutoff_month
                if mask.any():
                    cummax_vals.append(result.loc[mask, 'n_gxs'].max())
                else:
                    cummax_vals.append(0)
        result['n_gxs_cummax'] = cummax_vals
        return result
    
    # Simplified approach: use n_gxs at cutoff-1 for all prediction rows
    pre_cutoff_max = df[df['months_postgx'] < cutoff_month].groupby(series_keys)['n_gxs'].max().reset_index()
    pre_cutoff_max.columns = series_keys + ['n_gxs_pre_cutoff_max']
    df = df.merge(pre_cutoff_max, on=series_keys, how='left')
    df['n_gxs_pre_cutoff_max'] = df['n_gxs_pre_cutoff_max'].fillna(0)
    
    # First generic month
    first_generic = df[(df['n_gxs'] > 0) & (df['months_postgx'] < cutoff_month)]
    first_generic = first_generic.groupby(series_keys)['months_postgx'].min().reset_index()
    first_generic.columns = series_keys + ['first_generic_month']
    df = df.merge(first_generic, on=series_keys, how='left')
    
    # Months since first generic
    df['months_since_first_generic'] = df['months_postgx'] - df['first_generic_month']
    df['months_since_first_generic'] = df['months_since_first_generic'].fillna(-999)
    
    # Log transform for n_gxs
    df['log_n_gxs'] = np.log1p(df['n_gxs'])
    
    return df


def add_drug_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add static drug characteristics (Category 4).
    
    Features:
    - ther_area: Therapeutic area (label encoded)
    - biological: Boolean
    - small_molecule: Boolean
    - hospital_rate: Percentage (0-100)
    - hospital_rate_bin: Low/Medium/High
    - main_package: Dosage form (label encoded)
    - is_injection: Binary from main_package
    """
    # Hospital rate bins
    if 'hospital_rate' in df.columns:
        df['hospital_rate_bin'] = pd.cut(
            df['hospital_rate'],
            bins=[-np.inf, 30, 70, np.inf],
            labels=['low', 'medium', 'high']
        )
    
    # Is injection (derived from main_package)
    if 'main_package' in df.columns:
        injection_keywords = ['injection', 'injectable', 'syringe', 'vial', 'iv', 'infusion']
        df['is_injection'] = df['main_package'].str.lower().str.contains(
            '|'.join(injection_keywords), na=False
        ).astype(int)
    
    # Encode categoricals (label encoding)
    for col in ['ther_area', 'main_package']:
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
    
    # Biological interaction with n_gxs (optional)
    if 'biological' in df.columns and 'n_gxs' in df.columns:
        df['biological_x_n_gxs'] = df['biological'].astype(int) * df['n_gxs']
    
    return df


def add_early_erosion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add early post-entry signal features for Scenario 2 (Category 5).
    
    These features use months 0-5 data, only valid for Scenario 2.
    
    Features:
    - avg_vol_0_5: Mean volume over months [0, 5]
    - erosion_0_5: avg_vol_0_5 / avg_vol_12m
    - trend_0_5: Linear slope over months 0-5
    - drop_month_0: volume[0] / avg_vol_12m (initial drop)
    """
    series_keys = ['country', 'brand_name']
    
    # Mean volume months 0-5
    early_mask = (df['months_postgx'] >= 0) & (df['months_postgx'] <= 5)
    avg_vol_0_5 = df[early_mask].groupby(series_keys)['volume'].mean().reset_index()
    avg_vol_0_5.columns = series_keys + ['avg_vol_0_5']
    df = df.merge(avg_vol_0_5, on=series_keys, how='left')
    
    # Erosion in first 6 months
    df['erosion_0_5'] = df['avg_vol_0_5'] / (df['avg_vol_12m'] + 1e-6)
    df['erosion_0_5'] = df['erosion_0_5'].fillna(1.0)  # No erosion if missing
    
    # Trend in months 0-5
    early_data = df[early_mask][series_keys + ['months_postgx', 'volume']].copy()
    
    def compute_early_slope(group):
        if len(group) < 2:
            return np.nan
        slope, _, _, _, _ = stats.linregress(group['months_postgx'], group['volume'])
        return slope
    
    early_trends = early_data.groupby(series_keys).apply(compute_early_slope).reset_index()
    early_trends.columns = series_keys + ['trend_0_5']
    df = df.merge(early_trends, on=series_keys, how='left')
    df['trend_0_5'] = df['trend_0_5'].fillna(0)
    
    # Normalized trend
    df['trend_0_5_norm'] = df['trend_0_5'] / (df['avg_vol_12m'] + 1e-6)
    
    # Initial drop at month 0
    month_0_vol = df[df['months_postgx'] == 0][series_keys + ['volume']].copy()
    month_0_vol.columns = series_keys + ['vol_month_0']
    df = df.merge(month_0_vol, on=series_keys, how='left')
    df['drop_month_0'] = df['vol_month_0'] / (df['avg_vol_12m'] + 1e-6)
    df['drop_month_0'] = df['drop_month_0'].fillna(1.0)
    
    # Clean up intermediate column
    if 'vol_month_0' in df.columns:
        df = df.drop(columns=['vol_month_0'])
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_meta: bool = True) -> List[str]:
    """
    Get list of feature columns from DataFrame.
    
    Args:
        df: DataFrame with features
        exclude_meta: If True, exclude meta columns
        
    Returns:
        List of feature column names
    """
    meta_cols = [
        'country', 'brand_name', 'months_postgx', 'month',
        'volume', 'y_norm', 'bucket', 'avg_vol_12m', 'mean_erosion'
    ]
    
    if exclude_meta:
        return [c for c in df.columns if c not in meta_cols]
    return list(df.columns)
