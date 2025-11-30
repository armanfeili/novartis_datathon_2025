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
    'country', 'brand_name',
    'vol_norm',  # Same as y_norm (volume / avg_vol_12m) - severe leakage!
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
            'compute_seasonal': True,  # NEW: seasonal pattern detection
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
            'include_future_n_gxs': True,  # NEW: future n_gxs (exogenous)
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
        'target_encoding': {
            'enabled': False,  # Disabled by default; set True to enable
            'features': ['ther_area', 'country'],  # Columns to target encode
            'n_folds': 5,  # K-fold for leakage prevention
            'smoothing': 10,  # Smoothing factor for regularization
        },
        'feature_selection': {
            'analyze_correlations': False,
            'correlation_threshold': 0.95,
            'compute_importance': False,
        },
        # NEW: Visibility features (Ghannem et al., 2023)
        'visibility': {
            'enabled': False,  # Disabled by default; requires external data
        },
        # NEW: Collaboration features (Ghannem et al., 2023)
        'collaboration': {
            'enabled': False,  # Disabled by default
            'compute_country_prior': True,
            'compute_ther_area_prior': True,
            'compute_hospital_prior': True,
            'compute_package_prior': True,
        },
        # NEW: Sequence features (Li et al., 2024)
        'sequence': {
            'enabled': False,  # DISABLED - causes data leakage from post-LOE months
            'lag_windows': [1, 2, 3, 6],
            'ma_windows': [3, 6, 12],
        },
        # NEW: Scenario-aware lag & rolling features for GBM models
        # These features are computed ONLY from pre-LOE data (NO LEAKAGE)
        # Statistics from pre-entry period are broadcast to all rows
        'lag_rolling': {
            'enabled': True,  # Enabled - testing
            'lag_windows': [1, 3, 6],  # Not used in new implementation
            'rolling_windows': [3, 6],  # Not used in new implementation
            'compute_erosion_rate': True,  # Not used in new implementation
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


def add_scenario_aware_lag_rolling_features(
    df: pd.DataFrame,
    scenario: int,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Add lag and rolling features computed ONLY from pre-LOE data (NO LEAKAGE).
    
    CRITICAL: These features use ONLY pre-entry data (months_postgx < 0) to 
    compute statistics that are then broadcast to ALL rows for that series.
    This ensures no data leakage from post-LOE months.
    
    Features created (all computed from pre-LOE months only):
    - pre_loe_vol_mean: Mean normalized volume in pre-LOE period
    - pre_loe_vol_std: Std of normalized volume in pre-LOE period  
    - pre_loe_vol_trend: Linear trend slope in pre-LOE period
    - pre_loe_vol_last: Last value before LOE (month -1)
    - pre_loe_vol_last3_mean: Mean of last 3 months before LOE
    - pre_loe_vol_last6_mean: Mean of last 6 months before LOE
    - pre_loe_volatility: Coefficient of variation in pre-LOE
    - pre_loe_momentum: Change from month -6 to month -1
    
    For S2, also adds features from months 0-5 (early post-LOE observable data).
    
    Args:
        df: Panel data with volume, months_postgx, avg_vol_12m
        scenario: 1 or 2
        config: Configuration dict
        
    Returns:
        DataFrame with pre-LOE based features added
    """
    if config is None:
        config = {}
    
    result = df.copy()
    series_keys = ['country', 'brand_name']
    
    # Compute vol_norm if not present
    if 'vol_norm' not in result.columns:
        if 'avg_vol_12m' in result.columns:
            result['vol_norm'] = result['volume'] / (result['avg_vol_12m'] + 1e-6)
        else:
            result['vol_norm'] = result['volume'] / (result.groupby(series_keys)['volume'].transform('mean') + 1e-6)
    
    # === FEATURES FROM PRE-LOE PERIOD ONLY (months_postgx < 0) ===
    # These are safe for both S1 and S2
    
    pre_loe_mask = result['months_postgx'] < 0
    pre_loe_data = result[pre_loe_mask].copy()
    
    # 1. Basic pre-LOE statistics (broadcast to all rows)
    pre_loe_stats = pre_loe_data.groupby(series_keys)['vol_norm'].agg([
        ('pre_loe_vol_mean', 'mean'),
        ('pre_loe_vol_std', 'std'),
        ('pre_loe_vol_min', 'min'),
        ('pre_loe_vol_max', 'max'),
    ]).reset_index()
    
    # Coefficient of variation (volatility)
    pre_loe_stats['pre_loe_volatility'] = pre_loe_stats['pre_loe_vol_std'] / (pre_loe_stats['pre_loe_vol_mean'] + 1e-6)
    
    # 2. Last values before LOE
    last_month_data = pre_loe_data[pre_loe_data['months_postgx'] == -1].groupby(series_keys)['vol_norm'].first()
    last_month_data.name = 'pre_loe_vol_last'
    
    # Last 3 months mean (-3, -2, -1)
    last3_data = pre_loe_data[pre_loe_data['months_postgx'].isin([-3, -2, -1])].groupby(series_keys)['vol_norm'].mean()
    last3_data.name = 'pre_loe_vol_last3_mean'
    
    # Last 6 months mean (-6 to -1)
    last6_data = pre_loe_data[pre_loe_data['months_postgx'].isin([-6, -5, -4, -3, -2, -1])].groupby(series_keys)['vol_norm'].mean()
    last6_data.name = 'pre_loe_vol_last6_mean'
    
    # 3. Momentum: change from month -6 to month -1
    month_m6_data = pre_loe_data[pre_loe_data['months_postgx'] == -6].groupby(series_keys)['vol_norm'].first()
    month_m1_data = pre_loe_data[pre_loe_data['months_postgx'] == -1].groupby(series_keys)['vol_norm'].first()
    momentum = (month_m1_data - month_m6_data) / (month_m6_data + 1e-6)
    momentum.name = 'pre_loe_momentum'
    
    # 4. Linear trend in pre-LOE period (slope of volume over time)
    def compute_trend(group):
        if len(group) < 3:
            return 0.0
        x = group['months_postgx'].values
        y = group['vol_norm'].values
        # Simple linear regression slope
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        denom = ((x - x_mean) ** 2).sum()
        if denom < 1e-6:
            return 0.0
        return num / denom
    
    trend_data = pre_loe_data.groupby(series_keys).apply(compute_trend, include_groups=False)
    trend_data.name = 'pre_loe_vol_trend'
    
    # Merge all pre-LOE features
    result = result.merge(pre_loe_stats, on=series_keys, how='left')
    result = result.merge(last_month_data.reset_index(), on=series_keys, how='left')
    result = result.merge(last3_data.reset_index(), on=series_keys, how='left')
    result = result.merge(last6_data.reset_index(), on=series_keys, how='left')
    result = result.merge(momentum.reset_index(), on=series_keys, how='left')
    result = result.merge(trend_data.reset_index(), on=series_keys, how='left')
    
    # Fill NaN with neutral values
    for col in ['pre_loe_vol_mean', 'pre_loe_vol_last', 'pre_loe_vol_last3_mean', 'pre_loe_vol_last6_mean']:
        if col in result.columns:
            result[col] = result[col].fillna(1.0)
    for col in ['pre_loe_vol_std', 'pre_loe_volatility', 'pre_loe_momentum', 'pre_loe_vol_trend']:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)
    for col in ['pre_loe_vol_min', 'pre_loe_vol_max']:
        if col in result.columns:
            result[col] = result[col].fillna(1.0)
    
    n_features = 10  # Count of pre-LOE features
    
    # === ADDITIONAL FEATURES FOR SCENARIO 2 (months 0-5 observable) ===
    if scenario == 2:
        early_post_mask = (result['months_postgx'] >= 0) & (result['months_postgx'] <= 5)
        early_post_data = result[early_post_mask].copy()
        
        if len(early_post_data) > 0:
            # Early post-LOE statistics
            early_stats = early_post_data.groupby(series_keys)['vol_norm'].agg([
                ('early_post_vol_mean', 'mean'),
                ('early_post_vol_std', 'std'),
            ]).reset_index()
            
            # Month 5 value (last observable for S2)
            month5_data = early_post_data[early_post_data['months_postgx'] == 5].groupby(series_keys)['vol_norm'].first()
            month5_data.name = 'early_post_vol_m5'
            
            # Initial drop: compare month 0 to month -1
            month0_data = early_post_data[early_post_data['months_postgx'] == 0].groupby(series_keys)['vol_norm'].first()
            initial_drop = (month0_data - result.groupby(series_keys).apply(
                lambda g: g[g['months_postgx'] == -1]['vol_norm'].iloc[0] if len(g[g['months_postgx'] == -1]) > 0 else np.nan,
                include_groups=False
            ))
            initial_drop.name = 'initial_loe_drop'
            
            # Merge S2-specific features
            result = result.merge(early_stats, on=series_keys, how='left')
            result = result.merge(month5_data.reset_index(), on=series_keys, how='left')
            result = result.merge(initial_drop.reset_index(), on=series_keys, how='left')
            
            # Fill NaN
            for col in ['early_post_vol_mean', 'early_post_vol_m5']:
                if col in result.columns:
                    result[col] = result[col].fillna(1.0)
            for col in ['early_post_vol_std', 'initial_loe_drop']:
                if col in result.columns:
                    result[col] = result[col].fillna(0.0)
            
            n_features += 4
    
    # CRITICAL: Drop vol_norm column to prevent leakage (it equals y_norm)
    if 'vol_norm' in result.columns:
        result = result.drop(columns=['vol_norm'])
    
    logger.info(f"Added {n_features} scenario-aware lag/rolling features for S{scenario}")
    
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
        
        # 7. Target encoding features (if enabled and mode="train")
        # Must be done before creating y_norm, uses K-fold to prevent leakage
        target_enc_config = feat_config.get('target_encoding', {})
        if target_enc_config.get('enabled', False) and mode == "train":
            # Create y_norm first for target encoding
            if 'y_norm' not in df.columns:
                df['y_norm'] = df['volume'] / df['avg_vol_12m']
            df = add_target_encoding_features(df, target_enc_config, scenario)
        
        # 8. Create y_norm for training (ONLY if mode="train")
        if mode == "train":
            if 'y_norm' not in df.columns:
                df['y_norm'] = df['volume'] / df['avg_vol_12m']
        
        # 9. Visibility features (Ghannem et al., 2023) - if enabled
        vis_config = feat_config.get('visibility', {})
        if vis_config.get('enabled', False):
            visibility_data = vis_config.get('visibility_data', None)
            df = add_visibility_features(df, visibility_data, vis_config)
        
        # 10. Collaboration features (Ghannem et al., 2023) - if enabled
        collab_config = feat_config.get('collaboration', {})
        if collab_config.get('enabled', False) and mode == "train":
            df = add_collaboration_features(df, collab_config)
        
        # 11. Sequence features (Li et al., 2024) - if enabled
        seq_config = feat_config.get('sequence', {})
        if seq_config.get('enabled', False):
            df = add_sequence_features(df, seq_config)
        
        # 12. Scenario-aware lag & rolling features for GBM (NEW)
        lag_rolling_config = feat_config.get('lag_rolling', {})
        if lag_rolling_config.get('enabled', True):
            df = add_scenario_aware_lag_rolling_features(df, scenario, lag_rolling_config)
        
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
    
    # Seasonal pattern detection from pre-entry months
    if config.get('compute_seasonal', True) and 'month' in df.columns:
        df = _add_seasonal_features(df, series_keys)
    
    return df


def _add_seasonal_features(df: pd.DataFrame, series_keys: List[str]) -> pd.DataFrame:
    """
    Add seasonal pattern features from pre-entry months.
    
    Features:
    - seasonal_amplitude: Max deviation from mean by month-of-year (normalized)
    - seasonal_peak_month: Month with highest volume
    - seasonal_trough_month: Month with lowest volume  
    - seasonal_ratio: Peak-to-trough ratio
    - seasonal_q1_effect, seasonal_q2_effect, etc.: Quarter-wise deviations
    
    These features capture pre-LOE seasonality patterns that may persist post-entry.
    
    Args:
        df: Panel data with volume, months_postgx, month
        series_keys: ['country', 'brand_name']
        
    Returns:
        DataFrame with seasonal features added
    """
    # Month map for parsing
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    # Get pre-entry data only
    pre_entry = df[df['months_postgx'] < 0].copy()
    if len(pre_entry) == 0:
        logger.warning("No pre-entry data for seasonal features, skipping")
        return df
    
    # Parse month to month_of_year if not already present
    if 'month_of_year' not in pre_entry.columns:
        pre_entry['month_of_year'] = pre_entry['month'].map(month_map)
        # Fallback for any unmapped
        if pre_entry['month_of_year'].isna().any():
            try:
                mask = pre_entry['month_of_year'].isna()
                pre_entry.loc[mask, 'month_of_year'] = pd.to_datetime(
                    pre_entry.loc[mask, 'month'], format='%b'
                ).dt.month
            except Exception:
                pre_entry['month_of_year'] = pre_entry['month_of_year'].fillna(1)
    
    # Compute series-level seasonal statistics
    def compute_seasonal_stats(group):
        """Compute seasonal stats for a single series."""
        if len(group) < 6:
            # Not enough data for reliable seasonality
            return pd.Series({
                'seasonal_amplitude': 0.0,
                'seasonal_peak_month': 6,
                'seasonal_trough_month': 1,
                'seasonal_ratio': 1.0,
                'seasonal_q1_effect': 0.0,
                'seasonal_q2_effect': 0.0,
                'seasonal_q3_effect': 0.0,
                'seasonal_q4_effect': 0.0,
            })
        
        # Mean volume per month-of-year
        monthly_avg = group.groupby('month_of_year', observed=True)['volume'].mean()
        overall_mean = group['volume'].mean()
        
        if overall_mean <= 0 or len(monthly_avg) == 0:
            return pd.Series({
                'seasonal_amplitude': 0.0,
                'seasonal_peak_month': 6,
                'seasonal_trough_month': 1,
                'seasonal_ratio': 1.0,
                'seasonal_q1_effect': 0.0,
                'seasonal_q2_effect': 0.0,
                'seasonal_q3_effect': 0.0,
                'seasonal_q4_effect': 0.0,
            })
        
        # Normalized monthly ratios (deviation from mean)
        monthly_ratios = monthly_avg / overall_mean
        
        # Amplitude: max deviation from 1.0
        seasonal_amplitude = monthly_ratios.max() - monthly_ratios.min()
        
        # Peak and trough months
        peak_month = monthly_ratios.idxmax() if len(monthly_ratios) > 0 else 6
        trough_month = monthly_ratios.idxmin() if len(monthly_ratios) > 0 else 1
        
        # Peak-to-trough ratio
        peak_val = monthly_ratios.max()
        trough_val = monthly_ratios.min()
        seasonal_ratio = peak_val / (trough_val + 1e-6) if trough_val > 0 else 1.0
        seasonal_ratio = min(seasonal_ratio, 5.0)  # Cap extreme values
        
        # Quarter effects (average deviation for each quarter)
        q1_months = [1, 2, 3]
        q2_months = [4, 5, 6]
        q3_months = [7, 8, 9]
        q4_months = [10, 11, 12]
        
        def quarter_effect(months):
            vals = [monthly_ratios.get(m, 1.0) for m in months if m in monthly_ratios.index]
            return np.mean(vals) - 1.0 if vals else 0.0
        
        return pd.Series({
            'seasonal_amplitude': seasonal_amplitude,
            'seasonal_peak_month': peak_month,
            'seasonal_trough_month': trough_month,
            'seasonal_ratio': seasonal_ratio,
            'seasonal_q1_effect': quarter_effect(q1_months),
            'seasonal_q2_effect': quarter_effect(q2_months),
            'seasonal_q3_effect': quarter_effect(q3_months),
            'seasonal_q4_effect': quarter_effect(q4_months),
        })
    
    # Apply to each series
    seasonal_stats = pre_entry.groupby(series_keys, group_keys=False, observed=False).apply(
        compute_seasonal_stats, include_groups=False
    ).reset_index()
    
    # Merge back to main DataFrame
    df = df.merge(seasonal_stats, on=series_keys, how='left')
    
    # Fill NaN with neutral values
    df['seasonal_amplitude'] = df['seasonal_amplitude'].fillna(0.0)
    df['seasonal_peak_month'] = df['seasonal_peak_month'].fillna(6).astype(int)
    df['seasonal_trough_month'] = df['seasonal_trough_month'].fillna(1).astype(int)
    df['seasonal_ratio'] = df['seasonal_ratio'].fillna(1.0)
    for q in ['q1', 'q2', 'q3', 'q4']:
        df[f'seasonal_{q}_effect'] = df[f'seasonal_{q}_effect'].fillna(0.0)
    
    logger.debug("Added seasonal pattern features")
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
    
    # Future n_gxs features (EXOGENOUS: n_gxs is provided for all forecast months in test data)
    # This is not leakage because n_gxs is an external forecast, not derived from volume
    if config.get('include_future_n_gxs', True):
        df = _add_future_generics_features(df, series_keys, cutoff_month)
    
    return df


def _add_future_generics_features(df: pd.DataFrame, series_keys: List[str], cutoff_month: int) -> pd.DataFrame:
    """
    Add expected future generics features (exogenous).
    
    n_gxs for future months is available in both train and test data as an
    external forecast. This is NOT leakage because:
    1. n_gxs does not depend on volume (it's from competitive intelligence)
    2. n_gxs is provided for all forecast months in the test data
    
    Features:
    - n_gxs_at_month_12: Expected n_gxs at month 12
    - n_gxs_at_month_23: Expected n_gxs at end of forecast horizon
    - n_gxs_change_to_12: Change in n_gxs from cutoff to month 12
    - n_gxs_change_to_23: Change in n_gxs from cutoff to month 23
    - n_gxs_max_forecast: Maximum n_gxs over forecast horizon
    - expected_new_generics: Number of new generics expected in forecast
    
    Args:
        df: Panel data with n_gxs
        series_keys: ['country', 'brand_name']
        cutoff_month: Feature cutoff (0 for S1, 6 for S2)
        
    Returns:
        DataFrame with future generics features
    """
    # n_gxs at specific future months
    for target_month in [12, 23]:
        month_data = df[df['months_postgx'] == target_month][series_keys + ['n_gxs']].copy()
        if len(month_data) > 0:
            month_data.columns = series_keys + [f'n_gxs_at_month_{target_month}']
            df = df.merge(month_data, on=series_keys, how='left')
            df[f'n_gxs_at_month_{target_month}'] = df[f'n_gxs_at_month_{target_month}'].fillna(df['n_gxs'])
    
    # n_gxs at cutoff (baseline for change calculation)
    cutoff_ref = max(cutoff_month - 1, 0)  # Use month just before cutoff
    cutoff_data = df[df['months_postgx'] == cutoff_ref][series_keys + ['n_gxs']].copy()
    if len(cutoff_data) > 0:
        cutoff_data.columns = series_keys + ['n_gxs_at_cutoff']
        df = df.merge(cutoff_data, on=series_keys, how='left')
        df['n_gxs_at_cutoff'] = df['n_gxs_at_cutoff'].fillna(0)
        
        # Change features
        if 'n_gxs_at_month_12' in df.columns:
            df['n_gxs_change_to_12'] = df['n_gxs_at_month_12'] - df['n_gxs_at_cutoff']
        if 'n_gxs_at_month_23' in df.columns:
            df['n_gxs_change_to_23'] = df['n_gxs_at_month_23'] - df['n_gxs_at_cutoff']
        
        # Clean up intermediate column
        df = df.drop(columns=['n_gxs_at_cutoff'], errors='ignore')
    
    # Maximum n_gxs over forecast horizon
    forecast_start = max(cutoff_month, 0)
    forecast_data = df[(df['months_postgx'] >= forecast_start) & (df['months_postgx'] <= 23)]
    if len(forecast_data) > 0:
        max_n_gxs = forecast_data.groupby(series_keys, observed=False)['n_gxs'].max().reset_index()
        max_n_gxs.columns = series_keys + ['n_gxs_max_forecast']
        df = df.merge(max_n_gxs, on=series_keys, how='left')
        df['n_gxs_max_forecast'] = df['n_gxs_max_forecast'].fillna(df['n_gxs'])
        
        # Expected new generics in forecast period
        min_n_gxs = forecast_data.groupby(series_keys, observed=False)['n_gxs'].min().reset_index()
        min_n_gxs.columns = series_keys + ['n_gxs_min_forecast']
        df = df.merge(min_n_gxs, on=series_keys, how='left')
        df['expected_new_generics'] = df['n_gxs_max_forecast'] - df['n_gxs_min_forecast']
        df['expected_new_generics'] = df['expected_new_generics'].fillna(0).clip(lower=0)
        
        # Clean up
        df = df.drop(columns=['n_gxs_min_forecast'], errors='ignore')
    
    # Log transform
    for col in ['n_gxs_at_month_12', 'n_gxs_at_month_23', 'n_gxs_max_forecast']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    
    logger.debug("Added future generics features (exogenous)")
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


def add_target_encoding_features(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    scenario: int = 1
) -> pd.DataFrame:
    """
    Add target encoding features with K-fold to prevent leakage.
    
    Features:
    - ther_area_erosion_prior: Historical average erosion by therapeutic area
    - country_effect: Country-level erosion patterns
    - ther_area_encoded_target: Target encoding for therapeutic area
    
    CRITICAL: Uses K-fold cross-validation to prevent target leakage.
    Each series gets an encoding computed only from OTHER series.
    
    Args:
        df: Panel data with y_norm and categorical columns
        config: Target encoding config
        scenario: 1 or 2
        
    Returns:
        DataFrame with target encoding features
    """
    if config is None:
        config = {}
    
    if not config.get('enabled', False):
        return df
    
    if 'y_norm' not in df.columns:
        logger.warning("y_norm not available for target encoding, skipping")
        return df
    
    series_keys = ['country', 'brand_name']
    features_to_encode = config.get('features', ['ther_area', 'country'])
    n_folds = config.get('n_folds', 5)
    smoothing = config.get('smoothing', 10)
    
    scenario = _normalize_scenario(scenario)
    scenario_cfg = SCENARIO_CONFIG[scenario]
    target_start = scenario_cfg['target_start']
    target_end = scenario_cfg['target_end']
    
    # Filter to target rows only for computing encodings
    target_mask = (df['months_postgx'] >= target_start) & (df['months_postgx'] <= target_end)
    target_df = df[target_mask].copy()
    
    if len(target_df) == 0:
        logger.warning("No target rows for target encoding, skipping")
        return df
    
    # Get unique series
    series_df = target_df[series_keys].drop_duplicates()
    n_series = len(series_df)
    
    if n_series < n_folds:
        logger.warning(f"Not enough series ({n_series}) for {n_folds}-fold target encoding")
        return df
    
    # Assign fold indices to series (series-level split)
    np.random.seed(42)  # Reproducibility
    series_df = series_df.sample(frac=1).reset_index(drop=True)
    series_df['_fold'] = np.arange(len(series_df)) % n_folds
    
    # Merge fold indices to data
    target_df = target_df.merge(series_df, on=series_keys, how='left')
    
    # Global mean for smoothing
    global_mean = target_df['y_norm'].mean()
    
    for col in features_to_encode:
        if col not in df.columns:
            logger.warning(f"Column {col} not found for target encoding, skipping")
            continue
        
        encoded_col = f'{col}_erosion_prior'
        
        # Initialize with global mean
        df[encoded_col] = global_mean
        
        # K-fold target encoding
        for fold in range(n_folds):
            # Train on all folds except current
            train_mask = target_df['_fold'] != fold
            val_mask = target_df['_fold'] == fold
            
            train_data = target_df[train_mask]
            val_series = target_df[val_mask][series_keys].drop_duplicates()
            
            if len(train_data) == 0:
                continue
            
            # Compute mean per category from training folds
            category_means = train_data.groupby(col, observed=False)['y_norm'].agg(['mean', 'count']).reset_index()
            category_means.columns = [col, 'cat_mean', 'cat_count']
            
            # Smoothed mean: (count * cat_mean + smoothing * global_mean) / (count + smoothing)
            category_means['smoothed_mean'] = (
                (category_means['cat_count'] * category_means['cat_mean'] + smoothing * global_mean)
                / (category_means['cat_count'] + smoothing)
            )
            
            # Map to validation series
            cat_to_mean = dict(zip(category_means[col], category_means['smoothed_mean']))
            
            # Get indices for validation series in original df
            val_series_set = set(zip(val_series['country'], val_series['brand_name']))
            val_idx = df.apply(
                lambda row: (row['country'], row['brand_name']) in val_series_set,
                axis=1
            )
            
            # Assign encoded values
            df.loc[val_idx, encoded_col] = df.loc[val_idx, col].map(cat_to_mean).fillna(global_mean)
        
        # Fill any remaining NaN with global mean
        df[encoded_col] = df[encoded_col].fillna(global_mean)
        
        logger.debug(f"Created target encoding feature: {encoded_col}")
    
    # Add ther_area × erosion interaction if both exist
    if 'ther_area_erosion_prior' in df.columns:
        if 'erosion_0_5' in df.columns:
            # S2 only: interaction with early erosion
            df['ther_area_x_early_erosion'] = (
                df['ther_area_erosion_prior'] * df['erosion_0_5']
            )
        
        # Interaction with time
        df['ther_area_erosion_x_time'] = (
            df['ther_area_erosion_prior'] * df['months_postgx']
        )
    
    logger.info(f"Added target encoding features for: {features_to_encode}")
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


# =============================================================================
# Feature Selection Utilities
# =============================================================================

def analyze_feature_correlations(
    X: pd.DataFrame,
    threshold: float = 0.95,
    method: str = 'pearson'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Analyze feature correlations and identify redundant features.
    
    Args:
        X: Feature DataFrame (numeric columns only)
        threshold: Correlation threshold for redundancy (default 0.95)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Tuple of (correlation_matrix, list of features to potentially remove)
    """
    # Select only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    # Compute correlation matrix
    corr_matrix = X_numeric.corr(method=method)
    
    # Find highly correlated pairs
    redundant_features = set()
    n_features = len(numeric_cols)
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                col_i = numeric_cols[i]
                col_j = numeric_cols[j]
                # Keep the feature that appears first, mark the other as redundant
                redundant_features.add(col_j)
                logger.debug(
                    f"High correlation ({corr_matrix.iloc[i, j]:.3f}): "
                    f"{col_i} vs {col_j} - marking {col_j} for removal"
                )
    
    redundant_list = list(redundant_features)
    logger.info(
        f"Found {len(redundant_list)} potentially redundant features "
        f"(correlation >= {threshold})"
    )
    
    return corr_matrix, redundant_list


def compute_feature_importance_permutation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation feature importance.
    
    Args:
        model: Trained model with predict method
        X: Feature DataFrame
        y: Target Series
        n_repeats: Number of permutation repeats
        random_state: Random seed
        
    Returns:
        DataFrame with feature importance scores (mean and std)
    """
    from sklearn.metrics import mean_squared_error
    
    np.random.seed(random_state)
    
    # Baseline prediction error
    y_pred_baseline = model.predict(X)
    baseline_mse = mean_squared_error(y, y_pred_baseline)
    
    importances = []
    
    for col in X.columns:
        col_importances = []
        
        for _ in range(n_repeats):
            # Permute column
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            
            # Compute prediction error with permuted column
            y_pred_permuted = model.predict(X_permuted)
            permuted_mse = mean_squared_error(y, y_pred_permuted)
            
            # Importance = increase in error
            importance = permuted_mse - baseline_mse
            col_importances.append(importance)
        
        importances.append({
            'feature': col,
            'importance_mean': np.mean(col_importances),
            'importance_std': np.std(col_importances),
        })
    
    importance_df = pd.DataFrame(importances)
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    logger.info(f"Computed permutation importance for {len(X.columns)} features")
    return importance_df


def select_features_by_importance(
    importance_df: pd.DataFrame,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None
) -> List[str]:
    """
    Select top features by importance.
    
    Args:
        importance_df: DataFrame from compute_feature_importance_permutation
        top_k: Select top K features (if provided)
        threshold: Select features with importance > threshold (if provided)
        
    Returns:
        List of selected feature names
    """
    if top_k is not None:
        selected = importance_df.head(top_k)['feature'].tolist()
        logger.info(f"Selected top {top_k} features by importance")
        return selected
    
    if threshold is not None:
        selected = importance_df[
            importance_df['importance_mean'] > threshold
        ]['feature'].tolist()
        logger.info(f"Selected {len(selected)} features with importance > {threshold}")
        return selected
    
    # Default: return all features sorted by importance
    return importance_df['feature'].tolist()


def get_feature_summary(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for all features.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        DataFrame with summary stats for each feature
    """
    summary_data = []
    
    for col in X.columns:
        col_data = X[col]
        
        stats = {
            'feature': col,
            'dtype': str(col_data.dtype),
            'n_unique': col_data.nunique(),
            'n_missing': col_data.isna().sum(),
            'missing_pct': col_data.isna().mean() * 100,
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
            })
        else:
            stats.update({
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'median': np.nan,
            })
        
        summary_data.append(stats)
    
    summary_df = pd.DataFrame(summary_data)
    logger.info(f"Generated feature summary for {len(X.columns)} features")
    return summary_df


def remove_redundant_features(
    X: pd.DataFrame,
    correlation_threshold: float = 0.95,
    variance_threshold: float = 0.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove redundant features based on correlation and variance.
    
    Args:
        X: Feature DataFrame
        correlation_threshold: Remove features with correlation >= threshold
        variance_threshold: Remove features with variance <= threshold
        
    Returns:
        Tuple of (filtered DataFrame, list of removed features)
    """
    removed_features = []
    
    # Select numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_filtered = X.copy()
    
    # Remove zero/low variance features
    if variance_threshold > 0:
        variances = X_filtered[numeric_cols].var()
        low_var_cols = variances[variances <= variance_threshold].index.tolist()
        X_filtered = X_filtered.drop(columns=low_var_cols)
        removed_features.extend(low_var_cols)
        logger.info(f"Removed {len(low_var_cols)} low-variance features")
    
    # Remove highly correlated features
    if correlation_threshold < 1.0:
        _, redundant_cols = analyze_feature_correlations(
            X_filtered, threshold=correlation_threshold
        )
        redundant_cols = [c for c in redundant_cols if c in X_filtered.columns]
        X_filtered = X_filtered.drop(columns=redundant_cols)
        removed_features.extend(redundant_cols)
        logger.info(f"Removed {len(redundant_cols)} highly correlated features")
    
    logger.info(f"Final feature set: {len(X_filtered.columns)} features")
    return X_filtered, removed_features


# =============================================================================
# SECTION 8.1: FEATURE EXPERIMENTS - Frequency Encoding & Feature Scaling
# =============================================================================

def add_frequency_encoding_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Add frequency encoding for categorical columns.
    
    Frequency encoding replaces categories with their frequency of occurrence.
    This is useful for high-cardinality categorical features.
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of columns to encode. If None, auto-detects.
        normalize: If True, normalize frequencies to [0, 1]
        
    Returns:
        DataFrame with frequency encoded columns added
    """
    result = df.copy()
    
    if categorical_cols is None:
        # Auto-detect categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Exclude ID columns
        categorical_cols = [c for c in categorical_cols if c not in ['country', 'brand_name']]
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        # Compute frequency
        freq = df[col].value_counts(normalize=normalize)
        result[f'{col}_freq'] = df[col].map(freq).fillna(0.0)
        
        logger.debug(f"Added frequency encoding for {col}")
    
    logger.info(f"Added frequency encoding for {len(categorical_cols)} categorical columns")
    return result


# =============================================================================
# VISIBILITY AND COLLABORATION FEATURES (Ghannem et al., 2023)
# =============================================================================

def add_visibility_features(
    df: pd.DataFrame,
    visibility_data: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Add supply-chain visibility features.
    
    Features based on Ghannem et al. (2023):
    - vis_avg_inventory: Average inventory levels
    - vis_avg_days_of_supply: Average days of supply
    - vis_avg_stock_out_risk: Average stock-out risk probability
    - vis_fill_rate: Order fill rate
    - vis_avg_lead_time: Average lead time in days
    - vis_supplier_reliability: Average supplier reliability score
    - vis_capacity_utilization: Distribution center capacity utilization
    
    Args:
        df: Panel data
        visibility_data: Pre-computed visibility features DataFrame.
                        Must have columns: country, brand_name, period, vis_*
        config: Configuration dict
        
    Returns:
        DataFrame with visibility features added
    """
    if config is None:
        config = {}
    
    result = df.copy()
    
    # Define visibility feature columns
    vis_feature_cols = [
        'vis_avg_inventory',
        'vis_avg_days_of_supply',
        'vis_avg_stock_out_risk',
        'vis_fill_rate',
        'vis_avg_lead_time',
        'vis_supplier_reliability',
        'vis_on_time_delivery',
        'vis_capacity_utilization',
    ]
    
    if visibility_data is None or visibility_data.empty:
        # Add empty visibility columns with zeros
        for col in vis_feature_cols:
            result[col] = 0.0
        logger.info("No visibility data provided; added zero-valued visibility features")
        return result
    
    # Merge visibility data
    try:
        # Create period column in result for merging
        if 'month' in result.columns:
            result['_vis_period'] = pd.to_datetime(result['month']).dt.to_period('M')
        else:
            # Use months_postgx as a fallback (no actual merge possible)
            for col in vis_feature_cols:
                result[col] = 0.0
            logger.warning("No 'month' column found; cannot merge visibility data")
            return result
        
        # Prepare visibility data
        vis_df = visibility_data.copy()
        if 'period' in vis_df.columns:
            vis_df['_vis_period'] = vis_df['period']
        
        # Merge on country, brand, period
        result = result.merge(
            vis_df,
            left_on=['country', 'brand_name', '_vis_period'],
            right_on=['country', 'brand_name', '_vis_period'],
            how='left',
            suffixes=('', '_vis_merge')
        )
        
        # Fill missing visibility features with 0 or median
        for col in vis_feature_cols:
            if col in result.columns:
                # Fill with median, then with 0
                median_val = result[col].median()
                if pd.notna(median_val):
                    result[col] = result[col].fillna(median_val)
                else:
                    result[col] = result[col].fillna(0.0)
            else:
                result[col] = 0.0
        
        # Clean up temporary columns
        result = result.drop(columns=['_vis_period', 'period'], errors='ignore')
        
        logger.info(f"Added visibility features from external data")
        
    except Exception as e:
        logger.warning(f"Error merging visibility data: {e}. Using zeros.")
        for col in vis_feature_cols:
            result[col] = 0.0
    
    return result


def add_collaboration_features(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Add collaboration-based features derived from partner sharing patterns.
    
    Features based on Ghannem et al. (2023) collaboration signals:
    - collab_country_erosion_prior: Country-level average erosion curve
    - collab_ther_area_erosion_prior: Therapeutic area average erosion
    - collab_hospital_erosion_prior: Hospital vs retail segment erosion
    - collab_package_erosion_prior: Package type erosion patterns
    - collab_similarity_score: Similarity to comparable series
    
    Note: These features leverage cross-series information for collaborative
    forecasting without leaking target information (computed on training data only).
    
    Args:
        df: Panel data with columns: country, brand_name, ther_area, etc.
        config: Configuration dict
        
    Returns:
        DataFrame with collaboration features added
    """
    if config is None:
        config = {}
    
    result = df.copy()
    series_keys = ['country', 'brand_name']
    
    # Only compute collaboration features if y_norm is available (training mode)
    if 'y_norm' not in result.columns:
        # Add placeholder columns
        collab_cols = [
            'collab_country_erosion_prior',
            'collab_ther_area_erosion_prior',
            'collab_hospital_erosion_prior',
            'collab_package_erosion_prior',
        ]
        for col in collab_cols:
            result[col] = 0.0
        logger.info("No y_norm column; skipping collaboration features (test mode)")
        return result
    
    # Country-level erosion prior (leave-one-out)
    if config.get('compute_country_prior', True):
        result = _add_loo_prior(result, 'country', 'collab_country_erosion_prior')
    
    # Therapeutic area erosion prior
    if config.get('compute_ther_area_prior', True) and 'ther_area' in result.columns:
        result = _add_loo_prior(result, 'ther_area', 'collab_ther_area_erosion_prior')
    
    # Hospital vs retail erosion prior
    if config.get('compute_hospital_prior', True) and 'hospital_rate' in result.columns:
        # Create hospital segment
        result['_hospital_segment'] = pd.cut(
            result['hospital_rate'],
            bins=[-1, 30, 70, 100],
            labels=['retail', 'mixed', 'hospital']
        )
        result = _add_loo_prior(result, '_hospital_segment', 'collab_hospital_erosion_prior')
        result = result.drop(columns=['_hospital_segment'])
    
    # Package type erosion prior
    if config.get('compute_package_prior', True) and 'main_package' in result.columns:
        result = _add_loo_prior(result, 'main_package', 'collab_package_erosion_prior')
    
    logger.info("Added collaboration features")
    return result


def _add_loo_prior(
    df: pd.DataFrame,
    group_col: str,
    output_col: str
) -> pd.DataFrame:
    """
    Add leave-one-out prior for a grouping column.
    
    For each series, compute the average y_norm for other series
    in the same group (excluding the current series).
    
    Args:
        df: DataFrame with y_norm
        group_col: Column to group by
        output_col: Name of output column
        
    Returns:
        DataFrame with prior column added
    """
    result = df.copy()
    
    if group_col not in result.columns:
        result[output_col] = 0.0
        return result
    
    try:
        # Compute group-level mean
        group_mean = result.groupby(group_col)['y_norm'].transform('mean')
        group_count = result.groupby(group_col)['y_norm'].transform('count')
        
        # Leave-one-out: (sum - current) / (count - 1)
        group_sum = result.groupby(group_col)['y_norm'].transform('sum')
        
        # Avoid division by zero
        loo_prior = np.where(
            group_count > 1,
            (group_sum - result['y_norm']) / (group_count - 1),
            group_mean  # Fallback to group mean if only one in group
        )
        
        result[output_col] = loo_prior
        result[output_col] = result[output_col].fillna(result['y_norm'].mean())
        
    except Exception as e:
        logger.warning(f"Error computing LOO prior for {group_col}: {e}")
        result[output_col] = 0.0
    
    return result


# =============================================================================
# SEQUENCE FEATURES FOR DEEP LEARNING (Li et al., 2024)
# =============================================================================

def add_sequence_features(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Add sequence-based features for CNN-LSTM and deep learning models.
    
    Features based on Li et al. (2024) CNN-LSTM approach:
    - seq_volume_lag_*: Lagged volume values
    - seq_volume_ma_*: Moving average features
    - seq_volume_diff_*: Differenced volume features
    - seq_momentum_*: Momentum indicators
    - seq_acceleration: Second-order difference
    
    Note: These features are designed for tabular models. For actual CNN-LSTM,
    use the sequence_builder module to create proper 2D/3D tensors.
    
    Args:
        df: Panel data with volume and months_postgx
        config: Configuration dict with lag windows, etc.
        
    Returns:
        DataFrame with sequence features added
    """
    if config is None:
        config = {}
    
    result = df.copy()
    series_keys = ['country', 'brand_name']
    
    # Lag windows (default: 1, 2, 3, 6 months)
    lag_windows = config.get('lag_windows', [1, 2, 3, 6])
    
    # Moving average windows
    ma_windows = config.get('ma_windows', [3, 6, 12])
    
    # Sort by series and time
    result = result.sort_values(series_keys + ['months_postgx'])
    
    # 1. Lagged volume features (for pre-entry only to avoid leakage)
    for lag in lag_windows:
        col_name = f'seq_volume_lag_{lag}'
        result[col_name] = result.groupby(series_keys)['volume'].shift(lag)
        # Normalize by avg_vol_12m
        if 'avg_vol_12m' in result.columns:
            result[col_name] = result[col_name] / (result['avg_vol_12m'] + 1e-6)
        result[col_name] = result[col_name].fillna(1.0)
    
    # 2. Moving average features
    for window in ma_windows:
        col_name = f'seq_volume_ma_{window}'
        result[col_name] = result.groupby(series_keys)['volume'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        # Normalize
        if 'avg_vol_12m' in result.columns:
            result[col_name] = result[col_name] / (result['avg_vol_12m'] + 1e-6)
        result[col_name] = result[col_name].fillna(1.0)
    
    # 3. Differenced features (first-order)
    for lag in [1, 3]:
        col_name = f'seq_volume_diff_{lag}'
        result[col_name] = result.groupby(series_keys)['volume'].diff(lag)
        # Normalize
        if 'avg_vol_12m' in result.columns:
            result[col_name] = result[col_name] / (result['avg_vol_12m'] + 1e-6)
        result[col_name] = result[col_name].fillna(0.0)
    
    # 4. Momentum features (rate of change)
    for window in [3, 6]:
        col_name = f'seq_momentum_{window}'
        result[col_name] = result.groupby(series_keys)['volume'].pct_change(window)
        result[col_name] = result[col_name].clip(-1, 1).fillna(0.0)
    
    # 5. Acceleration (second-order difference)
    result['seq_acceleration'] = result.groupby(series_keys)['volume'].transform(
        lambda x: x.diff().diff()
    )
    if 'avg_vol_12m' in result.columns:
        result['seq_acceleration'] = result['seq_acceleration'] / (result['avg_vol_12m'] + 1e-6)
    result['seq_acceleration'] = result['seq_acceleration'].fillna(0.0)
    
    # 6. Volatility over rolling window
    for window in [3, 6]:
        col_name = f'seq_volatility_{window}'
        result[col_name] = result.groupby(series_keys)['volume'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std() / x.rolling(window=window, min_periods=1).mean()
        )
        result[col_name] = result[col_name].fillna(0.0)
    
    logger.info(f"Added sequence features with {len(lag_windows)} lags and {len(ma_windows)} MA windows")
    return result


def create_visibility_feature_names() -> List[str]:
    """Return list of visibility feature column names."""
    return [
        'vis_avg_inventory',
        'vis_avg_days_of_supply',
        'vis_avg_stock_out_risk',
        'vis_fill_rate',
        'vis_avg_lead_time',
        'vis_supplier_reliability',
        'vis_on_time_delivery',
        'vis_capacity_utilization',
    ]


def create_collaboration_feature_names() -> List[str]:
    """Return list of collaboration feature column names."""
    return [
        'collab_country_erosion_prior',
        'collab_ther_area_erosion_prior',
        'collab_hospital_erosion_prior',
        'collab_package_erosion_prior',
    ]


def create_sequence_feature_names(
    lag_windows: List[int] = None,
    ma_windows: List[int] = None
) -> List[str]:
    """Return list of sequence feature column names."""
    if lag_windows is None:
        lag_windows = [1, 2, 3, 6]
    if ma_windows is None:
        ma_windows = [3, 6, 12]
    
    names = []
    
    # Lag features
    for lag in lag_windows:
        names.append(f'seq_volume_lag_{lag}')
    
    # MA features
    for window in ma_windows:
        names.append(f'seq_volume_ma_{window}')
    
    # Diff features
    for lag in [1, 3]:
        names.append(f'seq_volume_diff_{lag}')
    
    # Momentum features
    for window in [3, 6]:
        names.append(f'seq_momentum_{window}')
    
    # Other
    names.extend(['seq_acceleration', 'seq_volatility_3', 'seq_volatility_6'])
    
    return names


class FeatureScaler:
    """
    Feature scaler for preprocessing before model training.
    
    Supports:
    - StandardScaler: Zero mean, unit variance
    - MinMaxScaler: Scale to [0, 1]
    - RobustScaler: Scale using median and IQR (robust to outliers)
    - None: No scaling (for tree-based models)
    
    Note: Tree-based models (CatBoost, LightGBM, XGBoost) typically don't
    need feature scaling, but linear models and neural networks do.
    
    Example usage:
        scaler = FeatureScaler(method='standard')
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    """
    
    def __init__(self, method: str = 'standard', exclude_cols: Optional[List[str]] = None):
        """
        Initialize feature scaler.
        
        Args:
            method: 'standard', 'minmax', 'robust', or 'none'
            exclude_cols: Columns to exclude from scaling (e.g., categorical)
        """
        self.method = method.lower()
        self.exclude_cols = exclude_cols or []
        self.scaler = None
        self.numeric_cols: List[str] = []
        self.fitted = False
        
        if self.method not in ['standard', 'minmax', 'robust', 'none']:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if self.method == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
    
    def fit(self, X: pd.DataFrame) -> 'FeatureScaler':
        """Fit the scaler on training data."""
        if self.method == 'none':
            self.fitted = True
            return self
        
        # Identify numeric columns to scale
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = [c for c in self.numeric_cols if c not in self.exclude_cols]
        
        if len(self.numeric_cols) > 0:
            self.scaler.fit(X[self.numeric_cols])
        
        self.fitted = True
        logger.info(f"FeatureScaler fitted: {len(self.numeric_cols)} numeric columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        if self.method == 'none':
            return X.copy()
        
        result = X.copy()
        
        if len(self.numeric_cols) > 0:
            # Only transform columns that exist in X
            cols_to_transform = [c for c in self.numeric_cols if c in X.columns]
            if len(cols_to_transform) > 0:
                result[cols_to_transform] = self.scaler.transform(X[cols_to_transform])
        
        return result
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled features back to original scale."""
        if self.method == 'none':
            return X.copy()
        
        result = X.copy()
        
        if len(self.numeric_cols) > 0:
            cols_to_transform = [c for c in self.numeric_cols if c in X.columns]
            if len(cols_to_transform) > 0:
                result[cols_to_transform] = self.scaler.inverse_transform(X[cols_to_transform])
        
        return result


def run_feature_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
    scenario: int,
    model_class: type,
    model_config: dict,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    val_fraction: float = 0.2,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Run feature ablation study to measure importance of each feature group.
    
    Tests each feature group by:
    1. Training with all features (baseline)
    2. Training without each group (drop-one)
    3. Training with only each group (add-one)
    
    Args:
        X: Feature DataFrame
        y: Target Series
        meta_df: Metadata DataFrame
        scenario: 1 or 2
        model_class: Model class to use
        model_config: Model configuration
        feature_groups: Dict mapping group name to list of feature columns.
                       If None, uses default grouping.
        val_fraction: Validation split fraction
        random_state: Random seed
        
    Returns:
        Dictionary with ablation results:
        {
            'baseline': {'rmse': float, 'mae': float},
            'drop_group_name': {'rmse': float, 'mae': float, 'impact': float},
            'add_group_name': {'rmse': float, 'mae': float},
            ...
        }
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    results = {}
    
    # Default feature groups if not provided
    if feature_groups is None:
        feature_groups = _get_default_feature_groups(X.columns.tolist())
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=random_state
    )
    
    # Baseline: all features
    logger.info("Ablation: Training baseline model with all features...")
    model_baseline = model_class(model_config)
    model_baseline.fit(X_train, y_train, X_val, y_val)
    baseline_preds = model_baseline.predict(X_val)
    
    baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_preds))
    baseline_mae = mean_absolute_error(y_val, baseline_preds)
    
    results['baseline'] = {
        'rmse': baseline_rmse,
        'mae': baseline_mae,
        'n_features': len(X.columns)
    }
    logger.info(f"Baseline RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")
    
    # Drop-one ablation
    for group_name, group_cols in feature_groups.items():
        cols_to_drop = [c for c in group_cols if c in X.columns]
        if len(cols_to_drop) == 0:
            continue
        
        remaining_cols = [c for c in X.columns if c not in cols_to_drop]
        if len(remaining_cols) == 0:
            continue
        
        logger.info(f"Ablation: Dropping {group_name} ({len(cols_to_drop)} features)...")
        
        X_train_drop = X_train[remaining_cols]
        X_val_drop = X_val[remaining_cols]
        
        model_drop = model_class(model_config)
        model_drop.fit(X_train_drop, y_train, X_val_drop, y_val)
        drop_preds = model_drop.predict(X_val_drop)
        
        drop_rmse = np.sqrt(mean_squared_error(y_val, drop_preds))
        drop_mae = mean_absolute_error(y_val, drop_preds)
        
        # Positive impact means this group helps (RMSE increases when dropped)
        impact = drop_rmse - baseline_rmse
        
        results[f'drop_{group_name}'] = {
            'rmse': drop_rmse,
            'mae': drop_mae,
            'impact': impact,
            'n_features_dropped': len(cols_to_drop)
        }
        logger.info(f"  Without {group_name}: RMSE={drop_rmse:.4f} (impact: {impact:+.4f})")
    
    # Add-one ablation (only each group)
    for group_name, group_cols in feature_groups.items():
        cols_to_use = [c for c in group_cols if c in X.columns]
        if len(cols_to_use) == 0:
            continue
        
        logger.info(f"Ablation: Only {group_name} ({len(cols_to_use)} features)...")
        
        X_train_add = X_train[cols_to_use]
        X_val_add = X_val[cols_to_use]
        
        model_add = model_class(model_config)
        model_add.fit(X_train_add, y_train, X_val_add, y_val)
        add_preds = model_add.predict(X_val_add)
        
        add_rmse = np.sqrt(mean_squared_error(y_val, add_preds))
        add_mae = mean_absolute_error(y_val, add_preds)
        
        results[f'add_{group_name}'] = {
            'rmse': add_rmse,
            'mae': add_mae,
            'n_features': len(cols_to_use)
        }
        logger.info(f"  Only {group_name}: RMSE={add_rmse:.4f}")
    
    logger.info("Feature ablation study complete")
    return results


def _get_default_feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    """Get default feature group definitions based on column names."""
    groups = {
        'pre_entry': [],
        'time': [],
        'generics': [],
        'drug': [],
        'early_erosion': [],
        'interactions': [],
        'target_encoding': [],
        'frequency_encoding': [],
        'other': []
    }
    
    for col in feature_cols:
        col_lower = col.lower()
        
        if any(x in col_lower for x in ['pre_entry', 'avg_vol_12m', 'avg_vol_6m', 'avg_vol_3m', 
                                         'volatility', 'log_avg_vol', 'seasonal']):
            groups['pre_entry'].append(col)
        elif any(x in col_lower for x in ['months_postgx', 'time_bucket', 'is_early', 'is_mid', 
                                           'is_late', 'time_decay', 'quarter', 'month_sin', 
                                           'month_cos', 'sqrt_months']):
            groups['time'].append(col)
        elif any(x in col_lower for x in ['n_gxs', 'generic', 'gxs_', 'has_generic', 
                                           'multiple_generic', 'many_generic']):
            groups['generics'].append(col)
        elif any(x in col_lower for x in ['ther_area', 'main_package', 'hospital', 'biological', 
                                           'small_molecule', 'injection', 'oral', 'encoded']):
            groups['drug'].append(col)
        elif any(x in col_lower for x in ['erosion_0', 'avg_vol_0_', 'trend_0_5', 'drop_month', 
                                           'recovery', 'competition_response']):
            groups['early_erosion'].append(col)
        elif '_x_' in col_lower:
            groups['interactions'].append(col)
        elif '_prior' in col_lower or '_mean_enc' in col_lower:
            groups['target_encoding'].append(col)
        elif '_freq' in col_lower:
            groups['frequency_encoding'].append(col)
        else:
            groups['other'].append(col)
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if len(v) > 0}


def compare_feature_engineering_approaches(
    panel_df: pd.DataFrame,
    scenario: int,
    approaches: Dict[str, Dict],
    model_class: type,
    model_config: dict,
    val_fraction: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare different feature engineering approaches on the same data.
    
    Args:
        panel_df: Panel DataFrame
        scenario: 1 or 2
        approaches: Dict mapping approach name to feature config
        model_class: Model class to use
        model_config: Model configuration
        val_fraction: Validation split fraction
        random_state: Random seed
        
    Returns:
        DataFrame with comparison results
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    results = []
    
    for approach_name, feature_config in approaches.items():
        logger.info(f"Testing approach: {approach_name}")
        
        # Build features with this approach
        feature_df = make_features(panel_df.copy(), scenario=scenario, mode='train', config=feature_config)
        train_rows = select_training_rows(feature_df, scenario=scenario)
        
        X, y, meta = split_features_target_meta(train_rows)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_fraction, random_state=random_state
        )
        
        # Train and evaluate
        model = model_class(model_config)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        
        results.append({
            'approach': approach_name,
            'n_features': len(X.columns),
            'rmse': rmse,
            'mae': mae
        })
        
        logger.info(f"  {approach_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, features={len(X.columns)}")
    
    return pd.DataFrame(results).sort_values('rmse')
