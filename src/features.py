"""
Feature engineering for Novartis Datathon 2025.

Scenario-aware feature construction with strict leakage prevention.
Features are organized into categories with clear cutoff rules.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .utils import timer

logger = logging.getLogger(__name__)

# Cutoff configuration per scenario
# Keys are integers (1 or 2) to align with CLI --scenario flag
SCENARIO_CONFIG = {
    1: {
        'feature_cutoff': 0,    # Only months_postgx < 0 for features
        'target_start': 0,
        'target_end': 23,
    },
    2: {
        'feature_cutoff': 6,    # months_postgx < 6 allowed
        'target_start': 6,
        'target_end': 23,
    }
}

# String aliases for backward compatibility
SCENARIO_CONFIG['scenario1'] = SCENARIO_CONFIG[1]
SCENARIO_CONFIG['scenario2'] = SCENARIO_CONFIG[2]

# Forbidden feature columns (leakage prevention)
FORBIDDEN_FEATURES = frozenset([
    'bucket', 'y_norm', 'volume', 'mean_erosion',
    'country', 'brand_name'
])


def _normalize_scenario(scenario) -> int:
    """
    Normalize scenario parameter to integer (1 or 2).
    
    Accepts: 1, 2, '1', '2', 'scenario1', 'scenario2'
    Returns: 1 or 2
    """
    if isinstance(scenario, int) and scenario in (1, 2):
        return scenario
    if isinstance(scenario, str):
        if scenario in ('1', '2'):
            return int(scenario)
        if scenario in ('scenario1', 'scenario2'):
            return int(scenario[-1])
    raise ValueError(f"Invalid scenario: {scenario}. Must be 1, 2, 'scenario1', or 'scenario2'")


def _load_feature_config(config: Optional[dict]) -> dict:
    """
    Load feature configuration with defaults.
    
    Args:
        config: Optional features.yaml config dict
        
    Returns:
        Config dict with all required keys
    """
    defaults = {
        'pre_entry': {
            'windows': [3, 6, 12],
            'compute_trend': True,
            'compute_volatility': True,
            'compute_max': True,
            'compute_min': True,
            'log_transform': True,
        },
        'time': {
            'include_months_postgx': True,
            'include_squared': True,
            'include_is_post_entry': True,
            'include_time_bucket': True,
            'include_month_of_year': True,
            'include_quarters': True,
            'include_decay': True,
            'decay_alpha': 0.1,
        },
        'generics': {
            'include_n_gxs': True,
            'include_has_generic': True,
            'include_multiple_generics': True,
            'include_cummax': True,
            'include_first_month': True,
            'include_entry_speed': True,
        },
        'drug': {
            'categoricals': ['ther_area', 'main_package'],
            'numerical': ['hospital_rate'],
            'boolean': ['biological', 'small_molecule'],
            'hospital_rate_bins': [30, 70],
            'derive_is_injection': True,
            'encoding': 'label',
        },
        'scenario2_early': {
            'compute_avg_0_5': True,
            'compute_erosion_0_5': True,
            'compute_trend_0_5': True,
            'compute_drop_month_0': True,
            'compute_change_windows': True,
            'compute_recovery_signal': True,
            'compute_competition_response': True,
        },
        'interactions': {
            'enabled': False,
            'pairs': [],
        },
    }
    
    if config is None:
        return defaults
    
    # Merge config with defaults
    result = {}
    for key, default_val in defaults.items():
        if key in config:
            if isinstance(default_val, dict):
                result[key] = {**default_val, **config[key]}
            else:
                result[key] = config[key]
        else:
            result[key] = default_val
    
    return result


def make_features(
    panel_df: pd.DataFrame,
    scenario,
    mode: str = "train",
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Build features respecting scenario-specific information constraints.
    
    Args:
        panel_df: Base panel with all raw data (must have avg_vol_12m)
        scenario: 1, 2, "scenario1", or "scenario2"
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
        - scenario 1: cutoff_month = 0 (only months_postgx < 0 for feature derivation)
        - scenario 2: cutoff_month = 6 (months_postgx < 6 allowed for feature derivation)
    """
    scenario = _normalize_scenario(scenario)
    if scenario not in SCENARIO_CONFIG:
        raise ValueError(f"Invalid scenario: {scenario}. Must be 1 or 2")
    
    scenario_cfg = SCENARIO_CONFIG[scenario]
    cutoff = scenario_cfg['feature_cutoff']
    
    # Load feature config with defaults
    feat_config = _load_feature_config(config)
    
    with timer(f"Feature engineering - Scenario {scenario}"):
        df = panel_df.copy()
        
        # 1. Pre-entry features (both scenarios)
        df = add_pre_entry_features(df, feat_config.get('pre_entry', {}))
        
        # 2. Time features
        df = add_time_features(df, feat_config.get('time', {}))
        
        # 3. Generics features (respecting cutoff)
        df = add_generics_features(df, cutoff_month=cutoff, config=feat_config.get('generics', {}))
        
        # 4. Drug characteristics
        df = add_drug_features(df, feat_config.get('drug', {}))
        
        # 5. Scenario 2 specific: early erosion features
        if scenario == 2:
            df = add_early_erosion_features(df, feat_config.get('scenario2_early', {}))
        
        # 6. Interaction features (if enabled)
        if feat_config.get('interactions', {}).get('enabled', False):
            df = add_interaction_features(df, feat_config.get('interactions', {}))
        
        # 7. Create y_norm for training (ONLY if mode="train")
        if mode == "train":
            if 'y_norm' not in df.columns:
                df['y_norm'] = df['volume'] / df['avg_vol_12m']
        
        # Log feature count
        n_features = len([c for c in df.columns if c not in FORBIDDEN_FEATURES 
                         and c not in ['month', 'avg_vol_12m', 'avg_vol_0_5', 'vol_month_0']])
        logger.info(f"Total features created: {n_features}")
    
    return df


def select_training_rows(panel_df: pd.DataFrame, scenario) -> pd.DataFrame:
    """
    Select only rows that are valid supervised targets for the scenario.
    
    Scenario 1: months_postgx in [0, 23]
    Scenario 2: months_postgx in [6, 23]
    
    ALWAYS call this before splitting into features/target.
    
    Args:
        panel_df: Panel with features
        scenario: 1, 2, "scenario1", or "scenario2"
        
    Returns:
        Filtered DataFrame with only target month rows
    """
    scenario = _normalize_scenario(scenario)
    if scenario not in SCENARIO_CONFIG:
        raise ValueError(f"Invalid scenario: {scenario}")
    
    cfg = SCENARIO_CONFIG[scenario]
    start = cfg['target_start']
    end = cfg['target_end']
    
    mask = (panel_df['months_postgx'] >= start) & (panel_df['months_postgx'] <= end)
    filtered = panel_df[mask].copy()
    
    logger.info(f"Selected {len(filtered):,} rows for {scenario} (months {start}-{end})")
    return filtered


def add_pre_entry_features(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Add pre-entry statistics (Category 1).
    
    Features:
    - avg_vol_12m: Already computed in data.py
    - avg_vol_6m: Mean volume over months [-6, -1]
    - avg_vol_3m: Mean volume over months [-3, -1]
    - pre_entry_trend: Linear slope over pre-entry period
    - pre_entry_volatility: std(volume_pre) / avg_vol_12m
    - pre_entry_max: Maximum volume in pre-entry period
    - pre_entry_min: Minimum volume in pre-entry period
    - volume_growth_rate: (vol_3m - vol_12m) / vol_12m
    - log_avg_vol: log1p(avg_vol_12m) for scale normalization
    - seasonal_amplitude: Amplitude of seasonal pattern
    
    Args:
        df: Panel data with volume and months_postgx
        config: Pre-entry feature config
    """
    if config is None:
        config = {}
    
    series_keys = ['country', 'brand_name']
    
    # avg_vol_6m
    pre_6m_mask = (df['months_postgx'] >= -6) & (df['months_postgx'] <= -1)
    avg_vol_6m = df[pre_6m_mask].groupby(series_keys, observed=False)['volume'].mean().reset_index()
    avg_vol_6m.columns = series_keys + ['avg_vol_6m']
    df = df.merge(avg_vol_6m, on=series_keys, how='left')
    df['avg_vol_6m'] = df['avg_vol_6m'].fillna(df['avg_vol_12m'])
    
    # avg_vol_3m
    pre_3m_mask = (df['months_postgx'] >= -3) & (df['months_postgx'] <= -1)
    avg_vol_3m = df[pre_3m_mask].groupby(series_keys, observed=False)['volume'].mean().reset_index()
    avg_vol_3m.columns = series_keys + ['avg_vol_3m']
    df = df.merge(avg_vol_3m, on=series_keys, how='left')
    df['avg_vol_3m'] = df['avg_vol_3m'].fillna(df['avg_vol_12m'])
    
    # Pre-entry trend (linear slope)
    if config.get('compute_trend', True):
        pre_entry_data = df[df['months_postgx'] < 0][series_keys + ['months_postgx', 'volume']].copy()
        
        def compute_slope(group):
            if len(group) < 2:
                return np.nan
            slope, _, _, _, _ = stats.linregress(group['months_postgx'], group['volume'])
            return slope
        
        trends = pre_entry_data.groupby(series_keys, group_keys=False, observed=False).apply(
            compute_slope, include_groups=False
        ).reset_index()
        trends.columns = series_keys + ['pre_entry_trend']
        df = df.merge(trends, on=series_keys, how='left')
        df['pre_entry_trend'] = df['pre_entry_trend'].fillna(0)
        
        # Normalized trend
        df['pre_entry_trend_norm'] = df['pre_entry_trend'] / (df['avg_vol_12m'] + 1e-6)
    
    # Pre-entry volatility
    if config.get('compute_volatility', True):
        pre_std = df[df['months_postgx'] < 0].groupby(series_keys, observed=False)['volume'].std().reset_index()
        pre_std.columns = series_keys + ['pre_entry_std']
        df = df.merge(pre_std, on=series_keys, how='left')
        df['pre_entry_volatility'] = df['pre_entry_std'] / (df['avg_vol_12m'] + 1e-6)
        df['pre_entry_volatility'] = df['pre_entry_volatility'].fillna(0)
        df = df.drop(columns=['pre_entry_std'])
    
    # Pre-entry max
    if config.get('compute_max', True):
        pre_max = df[df['months_postgx'] < 0].groupby(series_keys, observed=False)['volume'].max().reset_index()
        pre_max.columns = series_keys + ['pre_entry_max']
        df = df.merge(pre_max, on=series_keys, how='left')
        df['pre_entry_max'] = df['pre_entry_max'].fillna(df['avg_vol_12m'])
        
        # Max ratio
        df['pre_entry_max_ratio'] = df['pre_entry_max'] / (df['avg_vol_12m'] + 1e-6)
    
    # Pre-entry min
    if config.get('compute_min', True):
        pre_min = df[df['months_postgx'] < 0].groupby(series_keys, observed=False)['volume'].min().reset_index()
        pre_min.columns = series_keys + ['pre_entry_min']
        df = df.merge(pre_min, on=series_keys, how='left')
        df['pre_entry_min'] = df['pre_entry_min'].fillna(df['avg_vol_12m'])
        
        # Min ratio
        df['pre_entry_min_ratio'] = df['pre_entry_min'] / (df['avg_vol_12m'] + 1e-6)
        
        # Range ratio (max - min) / avg
        df['pre_entry_range_ratio'] = (df['pre_entry_max'] - df['pre_entry_min']) / (df['avg_vol_12m'] + 1e-6)
    
    # Log average volume
    if config.get('log_transform', True):
        df['log_avg_vol'] = np.log1p(df['avg_vol_12m'])
        df['log_avg_vol_6m'] = np.log1p(df['avg_vol_6m'])
        df['log_avg_vol_3m'] = np.log1p(df['avg_vol_3m'])
    
    # Volume growth rate: (vol_3m - vol_12m) / vol_12m
    df['volume_growth_rate'] = (df['avg_vol_3m'] - df['avg_vol_12m']) / (df['avg_vol_12m'] + 1e-6)
    
    # Ratio features
    df['vol_ratio_6m_12m'] = df['avg_vol_6m'] / (df['avg_vol_12m'] + 1e-6)
    df['vol_ratio_3m_12m'] = df['avg_vol_3m'] / (df['avg_vol_12m'] + 1e-6)
    df['vol_ratio_3m_6m'] = df['avg_vol_3m'] / (df['avg_vol_6m'] + 1e-6)
    
    return df


def add_time_features(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Add time-based features (Category 2).
    
    Features:
    - months_postgx: Direct time index (already present)
    - months_postgx_sq: Squared for non-linear decay
    - is_post_entry: Binary (months_postgx >= 0)
    - time_bucket: Categorical (pre, early, mid, late)
    - month_of_year: Seasonality from calendar month
    - quarter: Q1-Q4 for seasonality
    - is_year_end: December flag
    - time_decay: exp(-alpha * months_postgx) for erosion modeling
    
    Args:
        df: Panel data with months_postgx and month
        config: Time feature config
    """
    if config is None:
        config = {}
    
    # Squared time
    if config.get('include_squared', True):
        df['months_postgx_sq'] = df['months_postgx'] ** 2
        # Cube for more flexible decay
        df['months_postgx_cube'] = df['months_postgx'] ** 3
    
    # Post-entry indicator
    if config.get('include_is_post_entry', True):
        df['is_post_entry'] = (df['months_postgx'] >= 0).astype(int)
    
    # Time bucket categorical
    if config.get('include_time_bucket', True):
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
        # One-hot encode time bucket
        df['is_early'] = (df['time_bucket'] == 'early').astype(int)
        df['is_mid'] = (df['time_bucket'] == 'mid').astype(int)
        df['is_late'] = (df['time_bucket'] == 'late').astype(int)
    
    # Month of year for seasonality
    if config.get('include_month_of_year', True) and 'month' in df.columns:
        # Month column contains abbreviated names like 'Jan', 'Feb', etc.
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        df['month_of_year'] = df['month'].map(month_map)
        # Handle any unmapped values by trying to parse as datetime
        if df['month_of_year'].isna().any():
            mask = df['month_of_year'].isna()
            try:
                df.loc[mask, 'month_of_year'] = pd.to_datetime(df.loc[mask, 'month'], format='%b').dt.month
            except Exception:
                # Fallback: just fill with 1 if parsing fails
                df['month_of_year'] = df['month_of_year'].fillna(1).astype(int)
        df['month_of_year'] = df['month_of_year'].astype(int)
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12)
    
    # Quarters (Q1-Q4)
    if config.get('include_quarters', True) and 'month_of_year' in df.columns:
        df['quarter'] = ((df['month_of_year'] - 1) // 3) + 1
        # One-hot encode quarters
        for q in [1, 2, 3, 4]:
            df[f'is_q{q}'] = (df['quarter'] == q).astype(int)
    
    # Year-end flag (December)
    if 'month_of_year' in df.columns:
        df['is_year_end'] = (df['month_of_year'] == 12).astype(int)
        df['is_year_start'] = (df['month_of_year'] == 1).astype(int)
    
    # Time decay features
    if config.get('include_decay', True):
        alpha = config.get('decay_alpha', 0.1)
        # Only for post-entry months
        post_entry_mask = df['months_postgx'] >= 0
        df['time_decay'] = 0.0
        df.loc[post_entry_mask, 'time_decay'] = np.exp(-alpha * df.loc[post_entry_mask, 'months_postgx'])
        
        # Faster decay
        df['time_decay_fast'] = 0.0
        df.loc[post_entry_mask, 'time_decay_fast'] = np.exp(-0.2 * df.loc[post_entry_mask, 'months_postgx'])
        
        # Square root time (slower decay)
        df['sqrt_months_postgx'] = np.sqrt(np.maximum(df['months_postgx'], 0))
    
    return df


def add_generics_features(df: pd.DataFrame, cutoff_month: int = 0, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Add generics competition features respecting cutoff (Category 3).
    
    Features:
    - n_gxs: Current number of generics
    - has_generic: Binary (n_gxs > 0)
    - multiple_generics: Binary (n_gxs >= 2)
    - n_gxs_cummax: Maximum n_gxs up to current month (respecting cutoff)
    - n_gxs_at_entry: n_gxs at month 0
    - first_generic_month: Month when first generic appeared
    - months_since_first_generic: Time since first generic appeared
    - generic_entry_speed: Rate of new generic entries
    
    Args:
        df: Panel data
        cutoff_month: Feature cutoff (0 for S1, 6 for S2)
        config: Generics feature config
    """
    if config is None:
        config = {}
    
    series_keys = ['country', 'brand_name']
    
    if 'n_gxs' not in df.columns:
        logger.warning("n_gxs column not found, skipping generics features")
        return df
    
    # Basic features
    if config.get('include_has_generic', True):
        df['has_generic'] = (df['n_gxs'] > 0).astype(int)
    
    if config.get('include_multiple_generics', True):
        df['multiple_generics'] = (df['n_gxs'] >= 2).astype(int)
        df['many_generics'] = (df['n_gxs'] >= 5).astype(int)
    
    # n_gxs at entry (month 0)
    entry_n_gxs = df[df['months_postgx'] == 0][series_keys + ['n_gxs']].copy()
    entry_n_gxs.columns = series_keys + ['n_gxs_at_entry']
    df = df.merge(entry_n_gxs, on=series_keys, how='left')
    df['n_gxs_at_entry'] = df['n_gxs_at_entry'].fillna(0)
    
    # Cumulative max n_gxs (respecting cutoff for training)
    # For inference rows (months >= cutoff), use n_gxs up to cutoff
    df = df.sort_values(series_keys + ['months_postgx'])
    
    # Simplified approach: use n_gxs at cutoff-1 for all prediction rows
    if config.get('include_cummax', True):
        pre_cutoff_max = df[df['months_postgx'] < cutoff_month].groupby(series_keys, observed=False)['n_gxs'].max().reset_index()
        pre_cutoff_max.columns = series_keys + ['n_gxs_pre_cutoff_max']
        df = df.merge(pre_cutoff_max, on=series_keys, how='left')
        df['n_gxs_pre_cutoff_max'] = df['n_gxs_pre_cutoff_max'].fillna(0)
    
    # First generic month
    if config.get('include_first_month', True):
        first_generic = df[(df['n_gxs'] > 0) & (df['months_postgx'] < cutoff_month)]
        first_generic = first_generic.groupby(series_keys, observed=False)['months_postgx'].min().reset_index()
        first_generic.columns = series_keys + ['first_generic_month']
        df = df.merge(first_generic, on=series_keys, how='left')
        
        # Months since first generic
        df['months_since_first_generic'] = df['months_postgx'] - df['first_generic_month']
        df['months_since_first_generic'] = df['months_since_first_generic'].fillna(-999)
        
        # Binary: has generic before entry
        df['had_generic_pre_entry'] = (df['first_generic_month'] < 0).astype(int)
        df['had_generic_pre_entry'] = df['had_generic_pre_entry'].fillna(0).astype(int)
    
    # Generic entry speed: change in n_gxs per month (up to cutoff)
    if config.get('include_entry_speed', True):
        # Compute n_gxs change within allowed window
        pre_cutoff_data = df[df['months_postgx'] < cutoff_month].copy()
        if len(pre_cutoff_data) > 0:
            # Get n_gxs at first and last available month before cutoff
            first_n_gxs = pre_cutoff_data.groupby(series_keys, observed=False).apply(
                lambda g: g.loc[g['months_postgx'].idxmin(), 'n_gxs'] if len(g) > 0 else 0,
                include_groups=False
            ).reset_index()
            first_n_gxs.columns = series_keys + ['first_n_gxs']
            
            last_n_gxs = pre_cutoff_data.groupby(series_keys, observed=False).apply(
                lambda g: g.loc[g['months_postgx'].idxmax(), 'n_gxs'] if len(g) > 0 else 0,
                include_groups=False
            ).reset_index()
            last_n_gxs.columns = series_keys + ['last_n_gxs']
            
            n_months = pre_cutoff_data.groupby(series_keys, observed=False)['months_postgx'].apply(
                lambda g: g.max() - g.min() + 1 if len(g) > 0 else 1
            ).reset_index()
            n_months.columns = series_keys + ['n_months_pre_cutoff']
            
            df = df.merge(first_n_gxs, on=series_keys, how='left')
            df = df.merge(last_n_gxs, on=series_keys, how='left')
            df = df.merge(n_months, on=series_keys, how='left')
            
            df['generic_entry_speed'] = (df['last_n_gxs'] - df['first_n_gxs']) / (df['n_months_pre_cutoff'] + 1e-6)
            df['generic_entry_speed'] = df['generic_entry_speed'].fillna(0)
            
            # Clean up intermediate columns
            df = df.drop(columns=['first_n_gxs', 'last_n_gxs', 'n_months_pre_cutoff'], errors='ignore')
    
    # Log transform for n_gxs
    df['log_n_gxs'] = np.log1p(df['n_gxs'])
    df['log_n_gxs_at_entry'] = np.log1p(df['n_gxs_at_entry'])
    
    # Binned n_gxs
    df['n_gxs_bin'] = pd.cut(
        df['n_gxs'],
        bins=[-np.inf, 0, 1, 3, 10, np.inf],
        labels=['none', 'one', 'few', 'several', 'many']
    )
    
    return df


def add_drug_features(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
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
    - country_encoded: Country label encoded
    
    Args:
        df: Panel data with drug characteristics
        config: Drug feature config
    """
    if config is None:
        config = {}
    
    # Hospital rate bins
    if 'hospital_rate' in df.columns:
        bins = config.get('hospital_rate_bins', [30, 70])
        df['hospital_rate_bin'] = pd.cut(
            df['hospital_rate'],
            bins=[-np.inf] + bins + [np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # Normalized hospital rate
        df['hospital_rate_norm'] = df['hospital_rate'] / 100.0
        
        # Is predominantly hospital
        df['is_hospital_drug'] = (df['hospital_rate'] > 70).astype(int)
        df['is_retail_drug'] = (df['hospital_rate'] < 30).astype(int)
    
    # Is injection (derived from main_package)
    if 'main_package' in df.columns and config.get('derive_is_injection', True):
        injection_keywords = ['injection', 'injectable', 'syringe', 'vial', 'iv', 'infusion']
        df['is_injection'] = df['main_package'].str.lower().str.contains(
            '|'.join(injection_keywords), na=False
        ).astype(int)
        
        # Is oral
        oral_keywords = ['tablet', 'capsule', 'oral', 'pill']
        df['is_oral'] = df['main_package'].str.lower().str.contains(
            '|'.join(oral_keywords), na=False
        ).astype(int)
    
    # Encode categoricals (label encoding)
    encoding_method = config.get('encoding', 'label')
    categoricals = config.get('categoricals', ['ther_area', 'main_package'])
    
    for col in categoricals:
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
    
    # Country encoding
    if 'country' in df.columns:
        df['country_encoded'] = df['country'].astype('category').cat.codes
    
    # Biological interaction with n_gxs (optional)
    if 'biological' in df.columns and 'n_gxs' in df.columns:
        df['biological_x_n_gxs'] = df['biological'].astype(int) * df['n_gxs']
    
    # Hospital rate x time interaction
    if 'hospital_rate' in df.columns and 'months_postgx' in df.columns:
        df['hospital_rate_x_time'] = df['hospital_rate_norm'] * df['months_postgx']
    
    # Small molecule indicator (if not already boolean)
    if 'small_molecule' in df.columns:
        if df['small_molecule'].dtype != bool:
            df['small_molecule'] = df['small_molecule'].astype(int)
    
    # Biological indicator (if not already boolean)
    if 'biological' in df.columns:
        if df['biological'].dtype != bool:
            df['biological'] = df['biological'].astype(int)
    
    return df


def add_early_erosion_features(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Add early post-entry signal features for Scenario 2 (Category 5).
    
    These features use months 0-5 data, only valid for Scenario 2.
    
    Features:
    - avg_vol_0_5: Mean volume over months [0, 5]
    - erosion_0_5: avg_vol_0_5 / avg_vol_12m
    - trend_0_5: Linear slope over months 0-5
    - drop_month_0: volume[0] / avg_vol_12m (initial drop)
    - month_0_to_3_change: Short-term erosion rate
    - month_3_to_5_change: Medium-term erosion rate
    - recovery_signal: If volume increases after initial drop
    - competition_response: n_gxs change in months 0-5
    
    Args:
        df: Panel data with volume and months_postgx
        config: Scenario 2 feature config
    """
    if config is None:
        config = {}
    
    series_keys = ['country', 'brand_name']
    
    # Mean volume months 0-5
    if config.get('compute_avg_0_5', True):
        early_mask = (df['months_postgx'] >= 0) & (df['months_postgx'] <= 5)
        avg_vol_0_5 = df[early_mask].groupby(series_keys, observed=False)['volume'].mean().reset_index()
        avg_vol_0_5.columns = series_keys + ['avg_vol_0_5']
        df = df.merge(avg_vol_0_5, on=series_keys, how='left')
        
        # Erosion in first 6 months
        df['erosion_0_5'] = df['avg_vol_0_5'] / (df['avg_vol_12m'] + 1e-6)
        df['erosion_0_5'] = df['erosion_0_5'].fillna(1.0)  # No erosion if missing
    
    # Trend in months 0-5
    if config.get('compute_trend_0_5', True):
        early_mask = (df['months_postgx'] >= 0) & (df['months_postgx'] <= 5)
        early_data = df[early_mask][series_keys + ['months_postgx', 'volume']].copy()
        
        def compute_early_slope(group):
            if len(group) < 2:
                return np.nan
            slope, _, _, _, _ = stats.linregress(group['months_postgx'], group['volume'])
            return slope
        
        early_trends = early_data.groupby(series_keys, observed=False).apply(
            compute_early_slope, include_groups=False
        ).reset_index()
        early_trends.columns = series_keys + ['trend_0_5']
        df = df.merge(early_trends, on=series_keys, how='left')
        df['trend_0_5'] = df['trend_0_5'].fillna(0)
        
        # Normalized trend
        df['trend_0_5_norm'] = df['trend_0_5'] / (df['avg_vol_12m'] + 1e-6)
    
    # Initial drop at month 0
    if config.get('compute_drop_month_0', True):
        month_0_vol = df[df['months_postgx'] == 0][series_keys + ['volume']].copy()
        month_0_vol.columns = series_keys + ['vol_month_0']
        df = df.merge(month_0_vol, on=series_keys, how='left')
        df['drop_month_0'] = df['vol_month_0'] / (df['avg_vol_12m'] + 1e-6)
        df['drop_month_0'] = df['drop_month_0'].fillna(1.0)
    
    # Short-term and medium-term erosion windows
    if config.get('compute_change_windows', True):
        # Avg volume months 0-2
        mask_0_2 = (df['months_postgx'] >= 0) & (df['months_postgx'] <= 2)
        avg_vol_0_2 = df[mask_0_2].groupby(series_keys, observed=False)['volume'].mean().reset_index()
        avg_vol_0_2.columns = series_keys + ['avg_vol_0_2']
        df = df.merge(avg_vol_0_2, on=series_keys, how='left')
        
        # Avg volume months 3-5
        mask_3_5 = (df['months_postgx'] >= 3) & (df['months_postgx'] <= 5)
        avg_vol_3_5 = df[mask_3_5].groupby(series_keys, observed=False)['volume'].mean().reset_index()
        avg_vol_3_5.columns = series_keys + ['avg_vol_3_5']
        df = df.merge(avg_vol_3_5, on=series_keys, how='left')
        
        # Month 0 to 3 change: (avg_vol_0_2 - vol_month_0) / vol_month_0
        df['month_0_to_3_change'] = (df['avg_vol_0_2'] - df.get('vol_month_0', df['avg_vol_0_2'])) / (df.get('vol_month_0', df['avg_vol_12m']) + 1e-6)
        df['month_0_to_3_change'] = df['month_0_to_3_change'].fillna(0)
        
        # Month 3 to 5 change: (avg_vol_3_5 - avg_vol_0_2) / avg_vol_0_2
        df['month_3_to_5_change'] = (df['avg_vol_3_5'] - df['avg_vol_0_2']) / (df['avg_vol_0_2'] + 1e-6)
        df['month_3_to_5_change'] = df['month_3_to_5_change'].fillna(0)
        
        # Erosion ratios
        df['erosion_0_2'] = df['avg_vol_0_2'] / (df['avg_vol_12m'] + 1e-6)
        df['erosion_3_5'] = df['avg_vol_3_5'] / (df['avg_vol_12m'] + 1e-6)
    
    # Recovery signal: volume in months 3-5 higher than months 0-2
    if config.get('compute_recovery_signal', True):
        if 'avg_vol_3_5' in df.columns and 'avg_vol_0_2' in df.columns:
            df['recovery_signal'] = (df['avg_vol_3_5'] > df['avg_vol_0_2']).astype(int)
            # Recovery magnitude
            df['recovery_magnitude'] = (df['avg_vol_3_5'] - df['avg_vol_0_2']) / (df['avg_vol_0_2'] + 1e-6)
            df['recovery_magnitude'] = df['recovery_magnitude'].clip(-1, 1)  # Clip extreme values
    
    # Competition response: n_gxs change in months 0-5
    if config.get('compute_competition_response', True) and 'n_gxs' in df.columns:
        # n_gxs at month 0
        n_gxs_0 = df[df['months_postgx'] == 0][series_keys + ['n_gxs']].copy()
        n_gxs_0.columns = series_keys + ['n_gxs_month_0']
        df = df.merge(n_gxs_0, on=series_keys, how='left')
        
        # n_gxs at month 5
        n_gxs_5 = df[df['months_postgx'] == 5][series_keys + ['n_gxs']].copy()
        n_gxs_5.columns = series_keys + ['n_gxs_month_5']
        df = df.merge(n_gxs_5, on=series_keys, how='left')
        
        # Competition response: change in n_gxs from month 0 to 5
        df['competition_response'] = df['n_gxs_month_5'] - df['n_gxs_month_0']
        df['competition_response'] = df['competition_response'].fillna(0)
        
        # Competition intensity: erosion per new generic
        df['erosion_per_generic'] = df['trend_0_5'] / (df['competition_response'] + 1)
        df['erosion_per_generic'] = df['erosion_per_generic'].fillna(0)
        
        # Clean up intermediate columns
        df = df.drop(columns=['n_gxs_month_0', 'n_gxs_month_5'], errors='ignore')
    
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
    # Full list of meta columns that should not be treated as features
    meta_cols = list(FORBIDDEN_FEATURES) + [
        'month', 'months_postgx', 'avg_vol_12m', 'avg_vol_0_5',
        'avg_vol_0_2', 'avg_vol_3_5', 'vol_month_0'
    ]
    
    if exclude_meta:
        return [c for c in df.columns if c not in meta_cols]
    return list(df.columns)


def add_interaction_features(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Add interaction features (Category 6).
    
    Features are created by multiplying pairs of existing features.
    Configured via features.yaml interactions section.
    
    Args:
        df: Panel data with existing features
        config: Interaction feature config
        
    Returns:
        DataFrame with added interaction features
    """
    if config is None:
        config = {}
    
    if not config.get('enabled', False):
        return df
    
    pairs = config.get('pairs', [])
    
    for pair in pairs:
        if len(pair) != 2:
            logger.warning(f"Invalid interaction pair: {pair}, skipping")
            continue
        
        col1, col2 = pair
        
        # Check if columns exist
        if col1 not in df.columns:
            logger.warning(f"Column {col1} not found for interaction, skipping")
            continue
        if col2 not in df.columns:
            logger.warning(f"Column {col2} not found for interaction, skipping")
            continue
        
        # Create interaction feature
        interaction_name = f'{col1}_x_{col2}'
        
        # Handle different data types
        try:
            # For numeric columns, just multiply
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                df[interaction_name] = df[col1] * df[col2]
            # For categorical x numeric, use encoded value
            elif col1.endswith('_encoded') or col2.endswith('_encoded'):
                df[interaction_name] = df[col1].astype(float) * df[col2].astype(float)
            else:
                # Try to convert to numeric
                df[interaction_name] = pd.to_numeric(df[col1], errors='coerce') * pd.to_numeric(df[col2], errors='coerce')
            
            df[interaction_name] = df[interaction_name].fillna(0)
            logger.debug(f"Created interaction feature: {interaction_name}")
        except Exception as e:
            logger.warning(f"Failed to create interaction {interaction_name}: {e}")
    
    return df


def validate_feature_leakage(
    df: pd.DataFrame,
    scenario: int,
    mode: str = "train"
) -> Tuple[bool, List[str]]:
    """
    Validate that feature DataFrame doesn't contain leakage.
    
    Checks:
    1. No forbidden columns (bucket, y_norm, volume, mean_erosion, etc.)
    2. For Scenario 1, no early-erosion features (avg_vol_0_5, etc.)
    3. For mode="test", no target column
    
    Args:
        df: Feature DataFrame
        scenario: 1 or 2
        mode: "train" or "test"
        
    Returns:
        Tuple of (is_valid, list of violations)
    """
    scenario = _normalize_scenario(scenario)
    violations = []
    
    # Check forbidden features
    for col in FORBIDDEN_FEATURES:
        if col in df.columns:
            violations.append(f"Forbidden column present: {col}")
    
    # Check Scenario 1 doesn't have early erosion features
    if scenario == 1:
        early_erosion_features = [
            'avg_vol_0_5', 'erosion_0_5', 'trend_0_5', 'drop_month_0',
            'avg_vol_0_2', 'avg_vol_3_5', 'month_0_to_3_change', 'month_3_to_5_change',
            'recovery_signal', 'recovery_magnitude', 'competition_response',
            'erosion_0_2', 'erosion_3_5', 'erosion_per_generic'
        ]
        for col in early_erosion_features:
            if col in df.columns:
                violations.append(f"Scenario 1 should not have early erosion feature: {col}")
    
    # Check test mode doesn't have target
    if mode == "test":
        if 'y_norm' in df.columns:
            violations.append("Test mode should not have y_norm column")
    
    is_valid = len(violations) == 0
    return is_valid, violations


def validate_feature_cutoffs(
    df: pd.DataFrame,
    scenario: int
) -> Tuple[bool, List[str]]:
    """
    Validate that features respect scenario cutoff rules.
    
    For Scenario 1: Features should only use months_postgx < 0
    For Scenario 2: Features should only use months_postgx < 6
    
    This is a soft validation - it logs warnings but doesn't prevent operation.
    
    Args:
        df: Feature DataFrame with months_postgx
        scenario: 1 or 2
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    scenario = _normalize_scenario(scenario)
    cfg = SCENARIO_CONFIG[scenario]
    cutoff = cfg['feature_cutoff']
    
    warnings = []
    
    # Check if any derived features might use future data
    # This is a heuristic check based on column names
    if scenario == 1:
        suspect_cols = [c for c in df.columns if any(x in c.lower() for x in ['_0_5', '_0_2', '_3_5', 'month_0', 'month_5'])]
        if suspect_cols:
            warnings.append(f"Scenario 1 has potentially problematic columns: {suspect_cols}")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def validate_feature_matrix(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    meta_df: pd.DataFrame,
    mode: str = "train",
    raise_on_error: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate feature matrix structure and ensure no META_COLS leakage.
    
    This function should be called after building each feature matrix.
    
    Args:
        X: Feature DataFrame
        y: Target Series (should be present for mode="train")
        meta_df: Metadata DataFrame
        mode: "train" or "test"
        raise_on_error: If True, raises ValueError on validation failure
        
    Returns:
        Tuple of (is_valid, list of issues)
        
    Raises:
        ValueError: If raise_on_error=True and validation fails
    """
    issues = []
    
    # Check 1: X should not contain any META_COLS or FORBIDDEN_FEATURES
    meta_cols_in_X = set(FORBIDDEN_FEATURES) & set(X.columns)
    if meta_cols_in_X:
        issues.append(f"Feature matrix contains forbidden columns: {meta_cols_in_X}")
    
    # Also check for common meta columns
    common_meta = {'country', 'brand_name', 'bucket', 'mean_erosion', 'volume', 'y_norm'}
    common_meta_in_X = common_meta & set(X.columns)
    if common_meta_in_X:
        issues.append(f"Feature matrix contains meta columns: {common_meta_in_X}")
    
    # Check 2: For mode="train", y must be present and match X rows
    if mode == "train":
        if y is None:
            issues.append("mode='train' requires y (target) to be present")
        elif len(X) != len(y):
            issues.append(f"X rows ({len(X)}) != y length ({len(y)})")
    
    # Check 3: For mode="test", y should be None
    if mode == "test" and y is not None:
        issues.append("mode='test' should have y=None")
    
    # Check 4: Feature matrix should have at least some numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        issues.append("Feature matrix has no numeric columns")
    
    # Check 5: No all-NaN columns
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        issues.append(f"Feature matrix has all-NaN columns: {all_nan_cols[:5]}...")
    
    # Check 6: Log shapes for visibility
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info(
            f"Feature matrix validation passed: "
            f"X={X.shape}, y={'None' if y is None else len(y)}, meta={meta_df.shape}"
        )
    else:
        for issue in issues:
            logger.error(f"Feature matrix validation failed: {issue}")
        if raise_on_error:
            raise ValueError(f"Feature matrix validation failed:\n" + "\n".join(issues))
    
    return is_valid, issues


def split_features_target_meta(
    df: pd.DataFrame,
    meta_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
    """
    Split DataFrame into features (X), target (y), and metadata.
    
    Args:
        df: Full feature DataFrame
        meta_cols: List of meta columns to exclude from features.
                   If None, uses default FORBIDDEN_FEATURES + common meta.
        
    Returns:
        Tuple of (X, y, meta_df)
        - X: Feature DataFrame (numeric features only)
        - y: Target Series (y_norm) if present, else None
        - meta_df: Metadata DataFrame with keys and meta columns
    """
    if meta_cols is None:
        meta_cols = list(FORBIDDEN_FEATURES) + ['month', 'avg_vol_12m', 'avg_vol_0_5', 
                                                 'avg_vol_0_2', 'avg_vol_3_5', 'vol_month_0']
    
    # Extract target
    y = df['y_norm'] if 'y_norm' in df.columns else None
    
    # Extract meta columns that exist
    existing_meta_cols = [c for c in meta_cols if c in df.columns]
    meta_df = df[existing_meta_cols].copy() if existing_meta_cols else pd.DataFrame(index=df.index)
    
    # Feature columns: everything except meta, target, and non-numeric
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Filter to numeric features only (exclude categorical string columns)
    numeric_feature_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_feature_cols.append(col)
        elif df[col].dtype.name == 'category':
            # Keep category dtype columns (they'll be encoded)
            numeric_feature_cols.append(col)
    
    X = df[numeric_feature_cols].copy()
    
    return X, y, meta_df


def get_categorical_feature_names(df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical feature columns.
    
    Returns columns that are categorical dtype or known categorical columns.
    
    Args:
        df: Feature DataFrame
        
    Returns:
        List of categorical column names
    """
    known_categorical = [
        'time_bucket', 'hospital_rate_bin', 'n_gxs_bin',
        'ther_area', 'main_package', 'country'
    ]
    
    categorical_cols = []
    for col in df.columns:
        if col in known_categorical:
            categorical_cols.append(col)
        elif df[col].dtype.name == 'category':
            categorical_cols.append(col)
    
    return categorical_cols


def get_numeric_feature_names(df: pd.DataFrame, exclude_categorical: bool = True) -> List[str]:
    """
    Get list of numeric feature columns.
    
    Args:
        df: Feature DataFrame
        exclude_categorical: If True, exclude categorical columns
        
    Returns:
        List of numeric column names
    """
    categorical = get_categorical_feature_names(df) if exclude_categorical else []
    meta_cols = list(FORBIDDEN_FEATURES) + ['month']
    
    numeric_cols = []
    for col in df.columns:
        if col in meta_cols or col in categorical:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    return numeric_cols


# =============================================================================
# Persisted Feature Matrix Functions
# =============================================================================

def _get_feature_cache_path(
    processed_dir: 'Path',
    split: str,
    scenario: int,
    mode: str,
    suffix: str = "features"
) -> 'Path':
    """
    Get the cache file path for features/target/meta.
    
    Args:
        processed_dir: Base directory for processed data
        split: "train" or "test"
        scenario: 1 or 2
        mode: "train" or "test" (for mode-specific caching)
        suffix: "features", "target", or "meta"
        
    Returns:
        Path to cache file
    """
    from pathlib import Path
    return processed_dir / f"{suffix}_{split}_s{scenario}_{mode}.parquet"


def get_features(
    split: str,
    scenario: int,
    mode: str,
    data_config: dict,
    features_config: Optional[dict] = None,
    use_cache: bool = True,
    force_rebuild: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
    """
    Get feature matrix with caching support.
    
    This function is the primary entry point for obtaining features.
    It handles:
    - Loading cached feature matrices from parquet files
    - Building features from panel data if needed
    - Saving built features for future use
    
    Args:
        split: "train" or "test"
        scenario: 1 or 2
        mode: "train" or "test" (determines if y_norm is created)
        data_config: Loaded data.yaml config dict
        features_config: Optional features.yaml config dict
        use_cache: If True, try to load from cache first
        force_rebuild: If True, rebuild even if cache exists
        
    Returns:
        Tuple of (X, y, meta_df)
        - X: Feature DataFrame
        - y: Target Series (y_norm) if mode="train", else None
        - meta_df: Metadata DataFrame
        
    Cache locations:
        {processed_dir}/features_{split}_s{scenario}_{mode}.parquet
        {processed_dir}/target_{split}_s{scenario}_{mode}.parquet
        {processed_dir}/meta_{split}_s{scenario}_{mode}.parquet
    
    Example:
        data_config = load_config('configs/data.yaml')
        features_config = load_config('configs/features.yaml')
        X_train, y_train, meta_train = get_features(
            'train', scenario=1, mode='train',
            data_config=data_config, features_config=features_config
        )
    """
    from pathlib import Path
    from .data import get_panel, audit_data_leakage
    from .utils import get_project_root, is_colab
    
    scenario = _normalize_scenario(scenario)
    
    # Determine processed directory
    if is_colab():
        processed_dir = Path(data_config.get('drive', {}).get('processed_dir', 'data/processed'))
    else:
        processed_dir = get_project_root() / data_config['paths']['processed_dir']
    
    # Cache paths
    features_path = _get_feature_cache_path(processed_dir, split, scenario, mode, "features")
    target_path = _get_feature_cache_path(processed_dir, split, scenario, mode, "target")
    meta_path = _get_feature_cache_path(processed_dir, split, scenario, mode, "meta")
    
    # Check cache
    if use_cache and not force_rebuild:
        if features_path.exists():
            with timer(f"Load cached features for {split} S{scenario} {mode}"):
                logger.info(f"Loading cached features from {features_path}")
                X = pd.read_parquet(features_path)
                
                # Load target if exists
                y = None
                if mode == "train" and target_path.exists():
                    y = pd.read_parquet(target_path)['y_norm']
                
                # Load meta if exists
                if meta_path.exists():
                    meta_df = pd.read_parquet(meta_path)
                else:
                    meta_df = pd.DataFrame(index=X.index)
                
                logger.info(f"Loaded X: {X.shape}, y: {len(y) if y is not None else 'None'}")
                return X, y, meta_df
    
    # Build features from scratch
    logger.info(f"Building features for {split} S{scenario} {mode}...")
    
    # Get panel
    panel = get_panel(split=split, config=data_config, use_cache=use_cache, force_rebuild=force_rebuild)
    
    # Build features
    features_df = make_features(panel, scenario=scenario, mode=mode, config=features_config)
    
    # Select training rows (only for mode="train")
    if mode == "train":
        features_df = select_training_rows(features_df, scenario=scenario)
    
    # Split into X, y, meta
    X, y, meta_df = split_features_target_meta(features_df)
    
    # For test mode, y should be None (no target available for predictions)
    if mode == "test":
        y = None
    
    # Run leakage audit (non-strict to log warnings but not fail)
    is_clean, violations = audit_data_leakage(X, scenario=scenario, mode=mode, strict=False)
    if not is_clean:
        logger.warning(f"Leakage audit warnings: {violations}")
    
    # Validate feature matrix structure
    validate_feature_matrix(X, y, meta_df, mode=mode, raise_on_error=False)
    
    # Validate shapes
    if mode == "train" and y is not None:
        assert X.shape[0] == len(y), f"X rows ({X.shape[0]}) != y length ({len(y)})"
    
    logger.info(f"Built features: X={X.shape}, y={len(y) if y is not None else 'None'}, meta={meta_df.shape}")
    
    # Save to cache
    if use_cache:
        processed_dir.mkdir(parents=True, exist_ok=True)
        with timer(f"Save features to cache"):
            X.to_parquet(features_path, index=False)
            logger.info(f"Saved features to {features_path}")
            
            if y is not None:
                pd.DataFrame({'y_norm': y}).to_parquet(target_path, index=False)
                logger.info(f"Saved target to {target_path}")
            
            if len(meta_df.columns) > 0:
                meta_df.to_parquet(meta_path, index=False)
                logger.info(f"Saved meta to {meta_path}")
    
    return X, y, meta_df


def clear_feature_cache(data_config: dict, split: Optional[str] = None, scenario: Optional[int] = None) -> None:
    """
    Clear cached feature files.
    
    Args:
        data_config: Loaded data.yaml config dict
        split: If specified, only clear that split. Otherwise clear all.
        scenario: If specified, only clear that scenario. Otherwise clear all.
    """
    from pathlib import Path
    from .utils import get_project_root, is_colab
    
    if is_colab():
        processed_dir = Path(data_config.get('drive', {}).get('processed_dir', 'data/processed'))
    else:
        processed_dir = get_project_root() / data_config['paths']['processed_dir']
    
    splits = [split] if split else ['train', 'test']
    scenarios = [scenario] if scenario else [1, 2]
    modes = ['train', 'test']
    suffixes = ['features', 'target', 'meta']
    
    for s in splits:
        for sc in scenarios:
            for m in modes:
                for suffix in suffixes:
                    cache_path = _get_feature_cache_path(processed_dir, s, sc, m, suffix)
                    if cache_path.exists():
                        cache_path.unlink()
                        logger.info(f"Cleared cache: {cache_path}")
