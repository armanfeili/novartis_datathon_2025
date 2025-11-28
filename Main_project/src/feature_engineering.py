# =============================================================================
# File: src/feature_engineering.py
# Description: Functions to create all features for modeling
# 
# 🔧 ENHANCED VERSION - Implements EDA Report Recommendations:
#    - Time-based features (squared, sqrt, period indicators)
#    - Competition features (log transform, thresholds, intensity)
#    - Lag features (1, 3, 6, 12 month lags)
#    - Rolling statistics (mean, std, min, max)
#    - Interaction features (time × competition, etc.)
#    - Target encoding for categorical variables
#    - Hospital rate bucketing
#    - Therapeutic area erosion ranking
#    - Outlier handling (n_gxs capping)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).parent))
from config import *


# =============================================================================
# THERAPEUTIC AREA EROSION RANKINGS (from EDA)
# =============================================================================
# Ranked by mean erosion (lower = more erosion)
THER_AREA_EROSION_RANK = {
    'Anti-infectives': 1,                          # 0.515 - Highest erosion
    'Antineoplastic_and_immunology': 2,            # 0.551
    'Muscoskeletal_Rheumatology_and_Osteology': 3, # 0.557
    'Parasitology': 4,                             # 0.591
    'Cardiovascular_Metabolic': 5,                 # 0.592
    'Haematology': 6,                              # 0.597
    'Nervous_system': 7,                           # 0.607
    'Obstetrics_Gynaecology': 8,                   # 0.609
    'Systemic_Hormones': 9,                        # 0.628
    'Others': 10,                                  # 0.630
    'Dermatology': 11,                             # 0.644
    'Endocrinology_and_Metabolic_Disease': 12,     # 0.665
    'Respiratory_and_Immuno-inflammatory': 13,     # 0.698
    'Sensory_organs': 14,                          # 0.725 - Lowest erosion
}

# Mean erosion values by therapeutic area (from EDA)
THER_AREA_MEAN_EROSION = {
    'Anti-infectives': 0.515,
    'Antineoplastic_and_immunology': 0.551,
    'Muscoskeletal_Rheumatology_and_Osteology': 0.557,
    'Parasitology': 0.591,
    'Cardiovascular_Metabolic': 0.592,
    'Haematology': 0.597,
    'Nervous_system': 0.607,
    'Obstetrics_Gynaecology': 0.609,
    'Systemic_Hormones': 0.628,
    'Others': 0.630,
    'Dermatology': 0.644,
    'Endocrinology_and_Metabolic_Disease': 0.665,
    'Respiratory_and_Immuno-inflammatory': 0.698,
    'Sensory_organs': 0.725,
}

# High erosion therapeutic areas (from EDA: mean < 0.56)
HIGH_EROSION_AREAS = ['Anti-infectives', 'Antineoplastic_and_immunology', 
                      'Muscoskeletal_Rheumatology_and_Osteology']

# Competition thresholds (from EDA: diminishing returns after 6)
N_GXS_CAP = 15  # Cap outliers at 99th percentile
HIGH_COMPETITION_THRESHOLD = 5  # Volume drops rapidly with first 5-6 competitors


# =============================================================================
# SAMPLE WEIGHTS FOR CLASS IMBALANCE (EDA Section 13.1)
# =============================================================================

def compute_sample_weights(df: pd.DataFrame, 
                           bucket_col: str = 'bucket',
                           bucket1_weight: float = None,
                           bucket2_weight: float = None) -> np.ndarray:
    """
    Compute sample weights based on bucket membership.
    
    From EDA Report (Section 3 - Bucket Distribution):
    - Bucket 1 (High Erosion): Only 6.7% of brands, but 2× weight in scoring
    - Bucket 2 (Low Erosion): 93.3% of brands, normal weight
    - Imbalance Ratio: 14:1
    
    Applying higher weights to Bucket 1 samples helps the model focus on
    these critical high-erosion cases.
    
    Args:
        df: DataFrame with bucket column
        bucket_col: Name of bucket column
        bucket1_weight: Weight for Bucket 1 samples (default from config)
        bucket2_weight: Weight for Bucket 2 samples (default from config)
        
    Returns:
        numpy array of sample weights
    """
    from config import USE_SAMPLE_WEIGHTS, BUCKET_1_SAMPLE_WEIGHT, BUCKET_2_SAMPLE_WEIGHT
    
    if not USE_SAMPLE_WEIGHTS:
        return np.ones(len(df))
    
    bucket1_weight = bucket1_weight or BUCKET_1_SAMPLE_WEIGHT
    bucket2_weight = bucket2_weight or BUCKET_2_SAMPLE_WEIGHT
    
    if bucket_col not in df.columns:
        print(f"⚠️ Column '{bucket_col}' not found, returning uniform weights")
        return np.ones(len(df))
    
    weights = df[bucket_col].map({1: bucket1_weight, 2: bucket2_weight})
    weights = weights.fillna(1.0).values
    
    n_bucket1 = (df[bucket_col] == 1).sum()
    n_bucket2 = (df[bucket_col] == 2).sum()
    
    print(f"📊 Sample weights computed:")
    print(f"   Bucket 1: {n_bucket1} samples × {bucket1_weight}×")
    print(f"   Bucket 2: {n_bucket2} samples × {bucket2_weight}×")
    
    return weights


def compute_bucket_from_data(df: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """
    Compute bucket assignment for each brand based on mean normalized volume
    in months 18-23.
    
    From EDA Report (Section 3.2 - Bucket Definitions):
    - Bucket 1: Mean vol_norm ≤ 0.25 in months 18-23 (75%+ volume loss)
    - Bucket 2: Mean vol_norm > 0.25 in months 18-23
    
    Args:
        df: DataFrame with vol_norm and months_postgx columns
        threshold: Bucket 1 threshold (default 0.25 from competition rules)
        
    Returns:
        DataFrame with bucket assignment per country-brand
    """
    # Filter to equilibrium period (months 18-23)
    equilibrium_df = df[df['months_postgx'].between(18, 23)].copy()
    
    if len(equilibrium_df) == 0:
        print("⚠️ No data in months 18-23, cannot compute buckets")
        return pd.DataFrame()
    
    # Compute mean vol_norm per brand
    bucket_df = equilibrium_df.groupby(['country', 'brand_name'])['vol_norm'].mean().reset_index()
    bucket_df.columns = ['country', 'brand_name', 'mean_vol_norm_18_23']
    
    # Assign bucket
    bucket_df['bucket'] = np.where(bucket_df['mean_vol_norm_18_23'] <= threshold, 1, 2)
    
    n_bucket1 = (bucket_df['bucket'] == 1).sum()
    n_bucket2 = (bucket_df['bucket'] == 2).sum()
    
    print(f"📊 Bucket assignment computed:")
    print(f"   Bucket 1 (high erosion): {n_bucket1} brands ({100*n_bucket1/len(bucket_df):.1f}%)")
    print(f"   Bucket 2 (low erosion): {n_bucket2} brands ({100*n_bucket2/len(bucket_df):.1f}%)")
    
    return bucket_df[['country', 'brand_name', 'bucket']]


# =============================================================================
# LAG FEATURES (Critical for time-series - EDA Section 12.3)
# =============================================================================

def create_lag_features(df: pd.DataFrame, 
                        lags: list = None,
                        target_col: str = 'volume') -> pd.DataFrame:
    """
    Create lag features for volume.
    
    From EDA Report:
    - Lag features capture volume trajectory patterns
    - vol_norm_lag1, lag3, lag6 are most important
    - Also compute month-over-month changes and pct changes
    
    Args:
        df: Input DataFrame (must be sorted by brand and time)
        lags: List of lag periods (default from config: [1, 3, 6, 12])
        target_col: Column to create lags for
        
    Returns:
        DataFrame with added lag features
    """
    lags = lags or LAG_WINDOWS  # From config
    df = df.sort_values(['country', 'brand_name', 'months_postgx']).copy()
    
    for lag in lags:
        # Basic lag
        col_name = f'{target_col}_lag_{lag}'
        df[col_name] = df.groupby(['country', 'brand_name'])[target_col].shift(lag)
        
        # Difference from lag (month-over-month change)
        if lag <= 6:  # Only for shorter lags
            diff_col = f'{target_col}_diff_{lag}'
            df[diff_col] = df[target_col] - df[f'{target_col}_lag_{lag}']
    
    # Percentage change (avoiding division by zero)
    df[f'{target_col}_pct_change_1'] = df.groupby(['country', 'brand_name'])[target_col].pct_change()
    df[f'{target_col}_pct_change_1'] = df[f'{target_col}_pct_change_1'].replace([np.inf, -np.inf], np.nan)
    
    # 3-month percentage change
    if 3 in lags:
        df[f'{target_col}_pct_change_3'] = (df[target_col] - df[f'{target_col}_lag_3']) / df[f'{target_col}_lag_3'].replace(0, np.nan)
        df[f'{target_col}_pct_change_3'] = df[f'{target_col}_pct_change_3'].replace([np.inf, -np.inf], np.nan)
    
    print(f"✅ Created lag features for lags: {lags}")
    return df


# =============================================================================
# ROLLING FEATURES (EDA Section 12.4)
# =============================================================================

def create_rolling_features(df: pd.DataFrame,
                            windows: list = None,
                            target_col: str = 'volume') -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    From EDA Report:
    - Rolling mean captures local trend
    - Rolling std captures volatility
    - Rolling min helps identify rapid declines
    - 3-month and 6-month windows most important
    
    Args:
        df: Input DataFrame
        windows: List of window sizes (default from config: [3, 6, 12])
        target_col: Column to compute rolling stats for
        
    Returns:
        DataFrame with added rolling features
    """
    windows = windows or ROLLING_WINDOWS  # From config
    df = df.sort_values(['country', 'brand_name', 'months_postgx']).copy()
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = df.groupby(['country', 'brand_name'])[target_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Rolling std (volatility)
        df[f'{target_col}_rolling_std_{window}'] = df.groupby(['country', 'brand_name'])[target_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        # Rolling min (captures rapid declines)
        df[f'{target_col}_rolling_min_{window}'] = df.groupby(['country', 'brand_name'])[target_col].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        # Rolling max
        df[f'{target_col}_rolling_max_{window}'] = df.groupby(['country', 'brand_name'])[target_col].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
    
    # Erosion rate over last 3 months (from EDA recommendations)
    if 3 in windows:
        # Calculate as (current - rolling_min) / rolling_mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['erosion_rate_3m'] = (df[f'{target_col}_rolling_max_3'] - df[target_col]) / df[f'{target_col}_rolling_max_3'].replace(0, np.nan)
            df['erosion_rate_3m'] = df['erosion_rate_3m'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"✅ Created rolling features for windows: {windows}")
    return df


# =============================================================================
# PRE-ENTRY FEATURES (Critical for Scenario 1)
# =============================================================================

def create_pre_entry_features(df: pd.DataFrame, avg_j_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pre-entry aggregate features per country-brand.
    These are the ONLY features available at prediction time for Scenario 1.
    
    Features created:
    - avg_vol: Pre-entry average (from avg_j_df)
    - pre_entry_slope: Linear trend before entry
    - pre_entry_volatility: Std dev of pre-entry volumes
    - pre_entry_growth_rate: Growth rate in pre-entry period
    - pre_entry_min/max: Min and max volumes before entry
    
    Args:
        df: Full dataset
        avg_j_df: DataFrame with avg_vol per brand
        
    Returns:
        DataFrame with pre-entry features per brand
    """
    pre_entry = df[df['months_postgx'] < 0].copy()
    
    # 1. Pre-entry slope (linear trend)
    def calc_slope(group):
        if len(group) < 2:
            return 0
        x = group['months_postgx'].values
        y = group['volume'].values
        if np.std(x) == 0 or np.std(y) == 0:
            return 0
        try:
            return np.polyfit(x, y, 1)[0]
        except:
            return 0
    
    slopes = pre_entry.groupby(['country', 'brand_name'], as_index=False).apply(
        lambda g: pd.Series({'pre_entry_slope': calc_slope(g)}), 
        include_groups=False
    ).reset_index(drop=True)
    # Flatten and add back group columns
    slopes = pre_entry[['country', 'brand_name']].drop_duplicates().reset_index(drop=True)
    slopes['pre_entry_slope'] = pre_entry.groupby(['country', 'brand_name']).apply(calc_slope, include_groups=False).values
    
    # 2. Pre-entry volatility
    volatility = pre_entry.groupby(['country', 'brand_name'])['volume'].std().reset_index()
    volatility.columns = ['country', 'brand_name', 'pre_entry_volatility']
    
    # 3. Pre-entry min/max
    pre_min = pre_entry.groupby(['country', 'brand_name'])['volume'].min().reset_index()
    pre_min.columns = ['country', 'brand_name', 'pre_entry_min']
    
    pre_max = pre_entry.groupby(['country', 'brand_name'])['volume'].max().reset_index()
    pre_max.columns = ['country', 'brand_name', 'pre_entry_max']
    
    # 4. Pre-entry growth rate
    def calc_growth_rate(group):
        group = group.sort_values('months_postgx')
        if len(group) < 2:
            return 0
        first_vol = group['volume'].iloc[0]
        last_vol = group['volume'].iloc[-1]
        if first_vol == 0:
            return 0
        return (last_vol - first_vol) / first_vol
    
    # Fix FutureWarning - use include_groups=False
    growth = pre_entry[['country', 'brand_name']].drop_duplicates().reset_index(drop=True)
    growth['pre_entry_growth_rate'] = pre_entry.groupby(['country', 'brand_name']).apply(calc_growth_rate, include_groups=False).values
    
    # 5. Last pre-entry volume (month -1)
    last_pre = pre_entry[pre_entry['months_postgx'] == -1][['country', 'brand_name', 'volume']].copy()
    last_pre.columns = ['country', 'brand_name', 'pre_entry_last_volume']
    
    # Merge all features
    features = avg_j_df.copy()
    for feat_df in [slopes, volatility, pre_min, pre_max, growth, last_pre]:
        features = features.merge(feat_df, on=['country', 'brand_name'], how='left')
    
    print(f"✅ Created 7 pre-entry features for {len(features)} brands")
    return features


# =============================================================================
# COMPETITION FEATURES (EDA Section 12.2 & 5)
# =============================================================================

def create_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from generics competition data.
    
    From EDA Report (Section 5 - Competition Impact):
    - Volume drops rapidly with first 6 competitors, then plateaus
    - Log-transform n_gxs to capture diminishing marginal impact
    - Cap n_gxs at 15 (99th percentile) for outlier handling
    - Competition intensity = n_gxs / (months_postgx + 1)
    
    Features created:
    - n_gxs_capped: Capped at 15 (outlier handling)
    - n_gxs_log: Log transform (diminishing returns)
    - n_gxs_squared: Quadratic term
    - n_gxs_cummax: Cumulative max generics seen
    - n_gxs_change: Change in generics from previous month
    - has_generics: Binary indicator (n_gxs > 0)
    - high_competition: Binary (n_gxs >= 5)
    - months_with_generics: Cumulative months with competitors
    - competition_intensity: Rate of buildup (n_gxs / time)
    
    Args:
        df: DataFrame with n_gxs column
        
    Returns:
        DataFrame with added competition features
    """
    df = df.sort_values(['country', 'brand_name', 'months_postgx']).copy()
    
    # 1. OUTLIER HANDLING: Cap n_gxs at 15 (EDA Section 11.2)
    df['n_gxs_capped'] = df['n_gxs'].clip(upper=N_GXS_CAP)
    
    # 2. LOG TRANSFORM: Diminishing returns pattern (EDA Section 5.3)
    df['n_gxs_log'] = np.log1p(df['n_gxs_capped'])
    
    # 3. QUADRATIC TERM: Non-linear impact
    df['n_gxs_squared'] = df['n_gxs_capped'] ** 2
    
    # 4. Cumulative max generics
    df['n_gxs_cummax'] = df.groupby(['country', 'brand_name'])['n_gxs'].cummax()
    
    # 5. Change in number of generics
    df['n_gxs_change'] = df.groupby(['country', 'brand_name'])['n_gxs'].diff().fillna(0)
    
    # 6. Binary: has any generics
    df['has_generics'] = (df['n_gxs'] > 0).astype(int)
    
    # 7. Binary: high competition (EDA: plateau starts around 5-6 competitors)
    df['high_competition'] = (df['n_gxs'] >= HIGH_COMPETITION_THRESHOLD).astype(int)
    
    # 8. Cumulative months with generics
    df['months_with_generics'] = df.groupby(['country', 'brand_name'])['has_generics'].cumsum()
    
    # 9. Competition intensity: rate of buildup (EDA Section 12.2)
    # Avoid division by zero for month 0
    df['competition_intensity'] = df['n_gxs'] / (df['months_postgx'].clip(lower=0) + 1)
    
    # 10. Competition buildup rate (change in last 3 months)
    df['n_gxs_change_3m'] = df.groupby(['country', 'brand_name'])['n_gxs'].diff(3).fillna(0)
    
    print(f"✅ Created 10 competition features (with outlier handling)")
    return df


# =============================================================================
# TIME FEATURES (EDA Section 12.1)
# =============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    
    From EDA Report (Section 10.1 - Temporal Patterns):
    - Month 0-3: Rapid initial drop (most critical period)
    - Month 3-6: Continued steep decline  
    - Month 6-12: Erosion slows, pattern emerges
    - Month 12-18: Approaching equilibrium
    - Month 18-23: Stable equilibrium level
    
    CRITICAL INSIGHT: First 6 months determine bucket membership!
    
    Features created:
    - months_postgx_squared: Squared term for non-linear erosion
    - months_postgx_sqrt: Square root (early period emphasis)
    - months_postgx_log: Log transform
    - is_early_period: Months 0-5 (critical window)
    - is_mid_period: Months 6-11
    - is_late_period: Months 12-17
    - is_equilibrium: Months 18-23
    - time_bucket: Categorical (0-6, 6-12, 12-18, 18-24)
    
    Args:
        df: DataFrame with months_postgx column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Clip negative months for transforms (pre-entry period)
    months_clipped = df['months_postgx'].clip(lower=0)
    
    # Non-linear time features (EDA Section 12.1)
    df['months_postgx_squared'] = df['months_postgx'] ** 2
    df['months_postgx_sqrt'] = np.sqrt(months_clipped)  # Early period emphasis
    df['months_postgx_log'] = np.log1p(months_clipped)
    df['months_postgx_cubed'] = df['months_postgx'] ** 3  # Capture S-curve
    
    # Period indicators (aligned with EDA temporal patterns)
    df['is_early_period'] = (df['months_postgx'].between(0, 5)).astype(int)     # Critical first 6 months
    df['is_mid_period'] = (df['months_postgx'].between(6, 11)).astype(int)       # Pattern emergence
    df['is_late_period'] = (df['months_postgx'].between(12, 17)).astype(int)     # Approaching equilibrium
    df['is_equilibrium'] = (df['months_postgx'] >= 18).astype(int)               # Stable equilibrium
    
    # Aligned with metric weights (from competition rules)
    df['is_first_6_months'] = (df['months_postgx'].between(0, 5)).astype(int)    # sum_0-5 metric
    df['is_months_6_11'] = (df['months_postgx'].between(6, 11)).astype(int)      # sum_6-11 metric
    df['is_months_12_plus'] = (df['months_postgx'] >= 12).astype(int)            # sum_12-23 metric
    
    # Time bucket categorical (for tree-based models)
    df['time_bucket'] = pd.cut(
        df['months_postgx'], 
        bins=[-float('inf'), 0, 6, 12, 18, 24],
        labels=['pre_entry', 'early', 'mid', 'late', 'equilibrium']
    )
    df['time_bucket_encoded'] = df['time_bucket'].cat.codes
    
    # Quarter indicator (seasonality)
    df['quarter'] = ((df['months_postgx'] % 12) // 3) + 1
    
    # Decay phase indicator
    df['decay_phase'] = np.where(
        df['months_postgx'] < 0, 0,  # Pre-entry
        np.where(df['months_postgx'] <= 6, 1,  # Rapid decay
                np.where(df['months_postgx'] <= 12, 2,  # Slowing
                         3))  # Stable
    )
    
    print(f"✅ Created 15 time features")
    return df


# =============================================================================
# CATEGORICAL ENCODING
# =============================================================================

def encode_categorical_features(df: pd.DataFrame,
                                columns: list = None) -> tuple:
    """
    Label encode categorical columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to encode (default: from config)
        
    Returns:
        Tuple of (encoded DataFrame, dict of encoders)
    """
    df = df.copy()
    columns = columns or CATEGORICAL_COLS
    
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            # Handle missing values
            df[col] = df[col].fillna('UNKNOWN')
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    print(f"✅ Encoded {len(encoders)} categorical columns: {list(encoders.keys())}")
    return df, encoders


# =============================================================================
# HOSPITAL RATE FEATURES (EDA Section 8)
# =============================================================================

def create_hospital_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hospital rate bucket features.
    
    From EDA Report (Section 8 - Hospital Rate Impact):
    - 75-100% (Hospital): Fastest erosion (final: 0.446)
    - 0-25% (Retail): Moderate erosion (final: 0.509)
    - 25-50%: Slowest erosion (final: 0.556)
    - 50-75%: Moderate erosion (final: 0.485)
    
    Features:
    - hospital_rate_bucket: Categorical (4 buckets)
    - is_high_hospital_rate: Binary (>75%)
    - is_retail_focused: Binary (<25%)
    
    Args:
        df: DataFrame with hospital_rate column
        
    Returns:
        DataFrame with hospital rate features
    """
    df = df.copy()
    
    if 'hospital_rate' not in df.columns:
        print("⚠️ hospital_rate column not found, skipping hospital features")
        return df
    
    # Bucket hospital rate (from EDA Section 8.1)
    df['hospital_rate_bucket'] = pd.cut(
        df['hospital_rate'],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=['0-25%', '25-50%', '50-75%', '75-100%']
    )
    df['hospital_rate_bucket_encoded'] = df['hospital_rate_bucket'].cat.codes
    
    # Binary indicators
    df['is_high_hospital_rate'] = (df['hospital_rate'] >= 0.75).astype(int)
    df['is_retail_focused'] = (df['hospital_rate'] <= 0.25).astype(int)
    
    # Hospital rate squared (non-linear impact)
    df['hospital_rate_squared'] = df['hospital_rate'] ** 2
    
    print(f"✅ Created 4 hospital rate features")
    return df


# =============================================================================
# THERAPEUTIC AREA FEATURES (EDA Section 6)
# =============================================================================

def create_therapeutic_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create therapeutic area features based on EDA erosion analysis.
    
    From EDA Report (Section 6 - Therapeutic Area Analysis):
    - Anti-infectives: Highest erosion (0.515)
    - Sensory_organs: Lowest erosion (0.725)
    - 21 percentage points spread between areas
    
    Features:
    - ther_area_erosion_rank: 1-14 (1=highest erosion)
    - ther_area_mean_erosion: Mean vol_norm for area (target encoding)
    - is_high_erosion_area: Binary flag for top 3 erosion areas
    - is_low_erosion_area: Binary flag for bottom 3 erosion areas
    
    Args:
        df: DataFrame with ther_area column
        
    Returns:
        DataFrame with therapeutic area features
    """
    df = df.copy()
    
    if 'ther_area' not in df.columns:
        print("⚠️ ther_area column not found, skipping therapeutic features")
        return df
    
    # Map erosion rank (from EDA)
    df['ther_area_erosion_rank'] = df['ther_area'].map(THER_AREA_EROSION_RANK)
    df['ther_area_erosion_rank'] = df['ther_area_erosion_rank'].fillna(10)  # Default middle rank
    
    # Map mean erosion value (target encoding from EDA)
    df['ther_area_mean_erosion'] = df['ther_area'].map(THER_AREA_MEAN_EROSION)
    df['ther_area_mean_erosion'] = df['ther_area_mean_erosion'].fillna(0.615)  # Default median
    
    # Binary flags for high/low erosion areas
    df['is_high_erosion_area'] = df['ther_area'].isin(HIGH_EROSION_AREAS).astype(int)
    
    low_erosion_areas = ['Sensory_organs', 'Respiratory_and_Immuno-inflammatory', 
                         'Endocrinology_and_Metabolic_Disease']
    df['is_low_erosion_area'] = df['ther_area'].isin(low_erosion_areas).astype(int)
    
    print(f"✅ Created 4 therapeutic area features")
    return df


# =============================================================================
# INTERACTION FEATURES (EDA Section 12.5)
# =============================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features based on EDA insights.
    
    From EDA Report (Section 12.5 - Interaction Features):
    - time_x_competition: months_postgx * n_gxs
    - time_x_hospital: months_postgx * hospital_rate
    - competition_x_hospital: n_gxs * hospital_rate
    - early_high_competition: is_early_period * high_competition
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with added interaction features
    """
    df = df.copy()
    
    # Time × Competition (EDA: competition builds over time)
    if 'n_gxs' in df.columns:
        df['time_x_competition'] = df['months_postgx'] * df['n_gxs']
        df['time_x_n_gxs_log'] = df['months_postgx'] * df.get('n_gxs_log', np.log1p(df['n_gxs']))
    
    # Time × Hospital Rate (EDA: hospital erosion accelerates over time)
    if 'hospital_rate' in df.columns:
        df['time_x_hospital'] = df['months_postgx'] * df['hospital_rate']
    
    # Competition × Hospital Rate
    if 'n_gxs' in df.columns and 'hospital_rate' in df.columns:
        df['competition_x_hospital'] = df['n_gxs'] * df['hospital_rate']
    
    # Early period × High competition (critical combination)
    if 'is_early_period' in df.columns and 'high_competition' in df.columns:
        df['early_high_competition'] = df['is_early_period'] * df['high_competition']
    
    # Biological × time (biosimilars may erode slower)
    if 'biological' in df.columns:
        df['biological_x_months'] = df['biological'] * df['months_postgx']
    
    # Therapeutic area erosion × time
    if 'ther_area_mean_erosion' in df.columns:
        df['ther_erosion_x_time'] = df['ther_area_mean_erosion'] * df['months_postgx']
    
    # High erosion area × early period
    if 'is_high_erosion_area' in df.columns and 'is_early_period' in df.columns:
        df['high_erosion_early'] = df['is_high_erosion_area'] * df['is_early_period']
    
    print(f"✅ Created 8 interaction features")
    return df


# =============================================================================
# FULL FEATURE PIPELINE (Enhanced with EDA Recommendations)
# =============================================================================

def create_all_features(df: pd.DataFrame, 
                        avg_j_df: pd.DataFrame = None,
                        include_lags: bool = True,
                        include_rolling: bool = True) -> pd.DataFrame:
    """
    Run complete feature engineering pipeline with EDA recommendations.
    
    Pipeline Order (important for dependencies):
    1. Competition features (n_gxs processing, outlier handling)
    2. Time features (period indicators needed for interactions)
    3. Hospital rate features (bucket indicators)
    4. Therapeutic area features (erosion ranking)
    5. Lag features (volume trajectory)
    6. Rolling features (momentum indicators)
    7. Interaction features (cross-feature combinations)
    8. Categorical encoding
    9. Pre-entry features
    
    Args:
        df: Merged dataset
        avg_j_df: Pre-computed avg_vol per brand (optional)
        include_lags: Whether to include lag features
        include_rolling: Whether to include rolling features
        
    Returns:
        DataFrame with all features
    """
    print("\n" + "=" * 60)
    print("🔧 FEATURE ENGINEERING PIPELINE (EDA-Enhanced)")
    print("=" * 60)
    
    initial_cols = len(df.columns)
    
    # 1. Competition features (outlier handling, log transform, thresholds)
    df = create_competition_features(df)
    
    # 2. Time features (period indicators, decay phases)
    df = create_time_features(df)
    
    # 3. Hospital rate features (bucket indicators)
    df = create_hospital_rate_features(df)
    
    # 4. Therapeutic area features (erosion ranking, target encoding)
    df = create_therapeutic_area_features(df)
    
    # 5. Lag features (volume trajectory)
    if include_lags:
        df = create_lag_features(df)
    
    # 6. Rolling features (momentum, volatility)
    if include_rolling:
        df = create_rolling_features(df)
    
    # 7. Interaction features (cross-feature combinations)
    df = create_interaction_features(df)
    
    # 8. Categorical encoding (label encoding for remaining categoricals)
    df, encoders = encode_categorical_features(df)
    
    # 9. Add pre-entry features if avg_j provided
    if avg_j_df is not None:
        pre_entry_feats = create_pre_entry_features(df, avg_j_df)
        df = df.merge(pre_entry_feats, on=['country', 'brand_name'], how='left')
    
    # 10. Final cleanup - drop non-numeric categoricals from feature set
    # (Keep them in df for reference, but they shouldn't be used in modeling)
    
    # Fill NaN values created by lags/rolling with forward fill then 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    final_cols = len(df.columns)
    new_features = final_cols - initial_cols
    
    print(f"\n" + "=" * 60)
    print(f"✅ FEATURE ENGINEERING COMPLETE!")
    print(f"=" * 60)
    print(f"   Initial columns: {initial_cols}")
    print(f"   Final columns: {final_cols}")
    print(f"   New features created: {new_features}")
    print(f"   Final shape: {df.shape}")
    
    return df


def get_feature_columns(df: pd.DataFrame, 
                        exclude_cols: list = None) -> list:
    """
    Get list of feature columns for modeling.
    
    Automatically excludes:
    - Identifiers (country, brand_name)
    - Targets (volume, vol_norm)
    - Non-numeric categorical columns
    - Time bucket categorical (keep encoded version)
    
    Args:
        df: DataFrame with features
        exclude_cols: Additional columns to exclude
        
    Returns:
        List of feature column names
    """
    default_exclude = [
        # Identifiers
        'country', 'brand_name', 
        # Time columns
        'month',
        # Targets
        'volume', 'vol_norm', 'mean_erosion', 'bucket',
        # Categorical (use encoded versions instead)
        'ther_area', 'main_package', 'time_bucket', 'hospital_rate_bucket',
        # Pre-computed avg_vol (may cause leakage)
        'avg_vol',
    ]
    
    exclude = set(default_exclude)
    if exclude_cols:
        exclude.update(exclude_cols)
    
    # Get only numeric columns that aren't excluded
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude]
    
    return feature_cols


def get_feature_importance_groups() -> dict:
    """
    Return feature groupings for analysis.
    
    Based on EDA Report Section 10.2 (Expected Feature Importance):
    1. months_postgx - Time is primary driver
    2. n_gxs - Competition pressure
    3. ther_area - Category effects
    4. hospital_rate - Distribution channel
    5. avg_vol - Baseline volume
    6. biological - Limited impact
    
    Returns:
        Dictionary of feature groups
    """
    return {
        'time_features': [
            'months_postgx', 'months_postgx_squared', 'months_postgx_sqrt',
            'months_postgx_log', 'months_postgx_cubed',
            'is_early_period', 'is_mid_period', 'is_late_period', 'is_equilibrium',
            'is_first_6_months', 'is_months_6_11', 'is_months_12_plus',
            'time_bucket_encoded', 'quarter', 'decay_phase'
        ],
        'competition_features': [
            'n_gxs', 'n_gxs_capped', 'n_gxs_log', 'n_gxs_squared',
            'n_gxs_cummax', 'n_gxs_change', 'n_gxs_change_3m',
            'has_generics', 'high_competition',
            'months_with_generics', 'competition_intensity'
        ],
        'therapeutic_features': [
            'ther_area_encoded', 'ther_area_erosion_rank', 'ther_area_mean_erosion',
            'is_high_erosion_area', 'is_low_erosion_area'
        ],
        'hospital_features': [
            'hospital_rate', 'hospital_rate_bucket_encoded', 'hospital_rate_squared',
            'is_high_hospital_rate', 'is_retail_focused'
        ],
        'lag_features': [
            'volume_lag_1', 'volume_lag_3', 'volume_lag_6', 'volume_lag_12',
            'volume_diff_1', 'volume_diff_3', 'volume_diff_6',
            'volume_pct_change_1', 'volume_pct_change_3'
        ],
        'rolling_features': [
            'volume_rolling_mean_3', 'volume_rolling_mean_6', 'volume_rolling_mean_12',
            'volume_rolling_std_3', 'volume_rolling_std_6', 'volume_rolling_std_12',
            'volume_rolling_min_3', 'volume_rolling_min_6', 'volume_rolling_min_12',
            'volume_rolling_max_3', 'volume_rolling_max_6', 'volume_rolling_max_12',
            'erosion_rate_3m'
        ],
        'interaction_features': [
            'time_x_competition', 'time_x_n_gxs_log', 'time_x_hospital',
            'competition_x_hospital', 'early_high_competition',
            'biological_x_months', 'ther_erosion_x_time', 'high_erosion_early'
        ],
        'pre_entry_features': [
            'pre_entry_slope', 'pre_entry_volatility', 'pre_entry_growth_rate',
            'pre_entry_min', 'pre_entry_max', 'pre_entry_last_volume'
        ],
        'other_features': [
            'biological', 'small_molecule', 'country_encoded', 'main_package_encoded'
        ]
    }


if __name__ == "__main__":
    # Demo: Feature engineering with EDA enhancements
    print("=" * 60)
    print("🔧 FEATURE ENGINEERING DEMO (EDA-Enhanced)")
    print("=" * 60)
    
    from data_loader import load_all_data, merge_datasets
    from bucket_calculator import compute_avg_j
    
    # Load data
    print("\n📂 Loading data...")
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    print(f"   Merged shape: {merged.shape}")
    
    # Compute avg_j
    print("\n📊 Computing avg_j...")
    avg_j = compute_avg_j(merged)
    print(f"   Brands with avg_j: {len(avg_j)}")
    
    # Create all features
    print("\n🔧 Creating features...")
    featured = create_all_features(merged, avg_j)
    
    # Get feature columns
    feature_cols = get_feature_columns(featured)
    
    # Show feature groups
    print("\n📋 Feature Groups:")
    groups = get_feature_importance_groups()
    for group_name, features in groups.items():
        available = [f for f in features if f in featured.columns]
        print(f"   {group_name}: {len(available)} features")
    
    print(f"\n📊 Total feature columns: {len(feature_cols)}")
    print("\n✅ Feature engineering demo complete!")
