# =============================================================================
# File: src/feature_engineering.py
# Description: Functions to create all features for modeling
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import *


# =============================================================================
# LAG FEATURES
# =============================================================================

def create_lag_features(df: pd.DataFrame, 
                        lags: list = [1, 2, 3, 6, 12],
                        target_col: str = 'volume') -> pd.DataFrame:
    """
    Create lag features for volume.
    
    Args:
        df: Input DataFrame (must be sorted by brand and time)
        lags: List of lag periods
        target_col: Column to create lags for
        
    Returns:
        DataFrame with added lag features
    """
    df = df.sort_values(['country', 'brand_name', 'months_postgx']).copy()
    
    for lag in lags:
        col_name = f'{target_col}_lag_{lag}'
        df[col_name] = df.groupby(['country', 'brand_name'])[target_col].shift(lag)
    
    print(f"✅ Created {len(lags)} lag features: {[f'{target_col}_lag_{l}' for l in lags]}")
    return df


# =============================================================================
# ROLLING FEATURES
# =============================================================================

def create_rolling_features(df: pd.DataFrame,
                            windows: list = [3, 6, 12],
                            target_col: str = 'volume') -> pd.DataFrame:
    """
    Create rolling mean and std features.
    
    Args:
        df: Input DataFrame
        windows: List of window sizes
        target_col: Column to compute rolling stats for
        
    Returns:
        DataFrame with added rolling features
    """
    df = df.sort_values(['country', 'brand_name', 'months_postgx']).copy()
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = df.groupby(['country', 'brand_name'])[target_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Rolling std
        df[f'{target_col}_rolling_std_{window}'] = df.groupby(['country', 'brand_name'])[target_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    print(f"✅ Created {len(windows) * 2} rolling features for windows: {windows}")
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
# COMPETITION FEATURES (n_gxs)
# =============================================================================

def create_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from generics competition data.
    
    Features created:
    - n_gxs: Number of generics at each month (from data)
    - n_gxs_cummax: Cumulative max generics seen
    - n_gxs_change: Change in generics from previous month
    - has_generics: Binary indicator (n_gxs > 0)
    - months_with_generics: Cumulative months with competitors
    
    Args:
        df: DataFrame with n_gxs column
        
    Returns:
        DataFrame with added competition features
    """
    df = df.sort_values(['country', 'brand_name', 'months_postgx']).copy()
    
    # Cumulative max generics
    df['n_gxs_cummax'] = df.groupby(['country', 'brand_name'])['n_gxs'].cummax()
    
    # Change in number of generics
    df['n_gxs_change'] = df.groupby(['country', 'brand_name'])['n_gxs'].diff().fillna(0)
    
    # Binary: has any generics
    df['has_generics'] = (df['n_gxs'] > 0).astype(int)
    
    # Cumulative months with generics
    df['months_with_generics'] = df.groupby(['country', 'brand_name'])['has_generics'].cumsum()
    
    # Log transform of n_gxs (diminishing returns)
    df['n_gxs_log'] = np.log1p(df['n_gxs'])
    
    print(f"✅ Created 5 competition features")
    return df


# =============================================================================
# TIME FEATURES
# =============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    
    Features created:
    - months_postgx_squared: Squared term for non-linear erosion
    - months_postgx_log: Log transform
    - is_first_6_months: Indicator for months 0-5
    - is_months_6_11: Indicator for months 6-11
    - is_months_12_plus: Indicator for months 12+
    - quarter: Quarter within year
    
    Args:
        df: DataFrame with months_postgx column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Non-linear time features
    df['months_postgx_squared'] = df['months_postgx'] ** 2
    df['months_postgx_log'] = np.log1p(df['months_postgx'].clip(lower=0))
    df['months_postgx_sqrt'] = np.sqrt(df['months_postgx'].clip(lower=0))
    
    # Period indicators (aligned with metric weights)
    df['is_first_6_months'] = (df['months_postgx'].between(0, 5)).astype(int)
    df['is_months_6_11'] = (df['months_postgx'].between(6, 11)).astype(int)
    df['is_months_12_plus'] = (df['months_postgx'] >= 12).astype(int)
    
    # Quarter indicator
    df['quarter'] = ((df['months_postgx'] % 12) // 3) + 1
    
    print(f"✅ Created 7 time features")
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
# INTERACTION FEATURES
# =============================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features based on EDA insights.
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with added interaction features
    """
    df = df.copy()
    
    # Biological × time (biosimilars may erode slower)
    if 'biological' in df.columns:
        df['biological_x_months'] = df['biological'] * df['months_postgx']
    
    # Hospital rate × time (tender dynamics)
    if 'hospital_rate' in df.columns:
        df['hospital_rate_x_months'] = df['hospital_rate'] * df['months_postgx']
    
    # n_gxs × time
    if 'n_gxs' in df.columns:
        df['n_gxs_x_months'] = df['n_gxs'] * df['months_postgx']
    
    print(f"✅ Created 3 interaction features")
    return df


# =============================================================================
# FULL FEATURE PIPELINE
# =============================================================================

def create_all_features(df: pd.DataFrame, 
                        avg_j_df: pd.DataFrame = None,
                        include_lags: bool = True,
                        include_rolling: bool = True) -> pd.DataFrame:
    """
    Run complete feature engineering pipeline.
    
    Args:
        df: Merged dataset
        avg_j_df: Pre-computed avg_vol per brand (optional)
        include_lags: Whether to include lag features
        include_rolling: Whether to include rolling features
        
    Returns:
        DataFrame with all features
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # 1. Lag features
    if include_lags:
        df = create_lag_features(df)
    
    # 2. Rolling features
    if include_rolling:
        df = create_rolling_features(df)
    
    # 3. Competition features
    df = create_competition_features(df)
    
    # 4. Time features
    df = create_time_features(df)
    
    # 5. Interaction features
    df = create_interaction_features(df)
    
    # 6. Categorical encoding
    df, encoders = encode_categorical_features(df)
    
    # 7. Add pre-entry features if avg_j provided
    if avg_j_df is not None:
        pre_entry_feats = create_pre_entry_features(df, avg_j_df)
        df = df.merge(pre_entry_feats, on=['country', 'brand_name'], how='left')
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Final shape: {df.shape}")
    print(f"   Total features: {len(df.columns)}")
    
    return df


def get_feature_columns(df: pd.DataFrame, 
                        exclude_cols: list = None) -> list:
    """
    Get list of feature columns for modeling.
    
    Args:
        df: DataFrame with features
        exclude_cols: Columns to exclude
        
    Returns:
        List of feature column names
    """
    exclude = exclude_cols or [
        'country', 'brand_name', 'month', 'volume', 'vol_norm',
        'mean_erosion', 'bucket', 'avg_vol', 'ther_area', 'main_package'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols


if __name__ == "__main__":
    # Demo: Feature engineering
    print("=" * 60)
    print("FEATURE ENGINEERING DEMO")
    print("=" * 60)
    
    from data_loader import load_all_data, merge_datasets
    from bucket_calculator import compute_avg_j
    
    # Load data
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    
    # Compute avg_j
    avg_j = compute_avg_j(merged)
    
    # Create all features
    featured = create_all_features(merged, avg_j)
    
    # Get feature columns
    feature_cols = get_feature_columns(featured)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols[:10]:
        print(f"  - {col}")
    print(f"  ... and {len(feature_cols) - 10} more")
    
    print("\n✅ Feature engineering demo complete!")
