# =============================================================================
# File: src/data_loader.py
# Description: Functions to load, validate, clean, and merge all datasets
#
# 🔧 ENHANCED VERSION - Implements Data Pipeline Todo:
#    - Duplicate removal (Section 1.1)
#    - Sanity checks for months_postgx range
#    - Handle brands with short history
#    - Regression-based avg_vol imputation (Section 1.2)
#    - vol_norm_gt1 flag (Section 1.3)
#    - Comprehensive validation
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import *


def load_volume_data(train: bool = True) -> pd.DataFrame:
    """
    Load volume dataset (train or test).
    
    Args:
        train: If True, load training data; else load test data
        
    Returns:
        DataFrame with columns: country, brand_name, month, months_postgx, volume
    """
    path = VOLUME_TRAIN if train else VOLUME_TEST
    df = pd.read_csv(path)
    
    required_cols = ['country', 'brand_name', 'month', 'months_postgx', 'volume']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in volume data: {missing}")
    
    print(f"✅ Loaded volume {'train' if train else 'test'}: {df.shape}")
    return df


def load_generics_data(train: bool = True) -> pd.DataFrame:
    """
    Load generics dataset (train or test).
    
    Args:
        train: If True, load training data; else load test data
        
    Returns:
        DataFrame with columns: country, brand_name, months_postgx, n_gxs
    """
    path = GENERICS_TRAIN if train else GENERICS_TEST
    df = pd.read_csv(path)
    
    required_cols = ['country', 'brand_name', 'months_postgx', 'n_gxs']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in generics data: {missing}")
    
    print(f"✅ Loaded generics {'train' if train else 'test'}: {df.shape}")
    return df


def load_medicine_info(train: bool = True) -> pd.DataFrame:
    """
    Load medicine info dataset (train or test).
    
    Args:
        train: If True, load training data; else load test data
        
    Returns:
        DataFrame with columns: country, brand_name, ther_area, hospital_rate, 
                               main_package, biological, small_molecule
    """
    path = MEDICINE_INFO_TRAIN if train else MEDICINE_INFO_TEST
    df = pd.read_csv(path)
    
    print(f"✅ Loaded medicine_info {'train' if train else 'test'}: {df.shape}")
    return df


def load_all_data(train: bool = True) -> tuple:
    """
    Load all three datasets.
    
    Args:
        train: If True, load training data; else load test data
        
    Returns:
        Tuple of (volume_df, generics_df, medicine_df)
    """
    volume = load_volume_data(train)
    generics = load_generics_data(train)
    medicine = load_medicine_info(train)
    return volume, generics, medicine


def merge_datasets(volume_df: pd.DataFrame, 
                   generics_df: pd.DataFrame, 
                   medicine_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all datasets into unified modeling table.
    
    Merge strategy:
    1. volume + generics on (country, brand_name, months_postgx)
    2. result + medicine_info on (country, brand_name)
    
    Args:
        volume_df: Volume dataset
        generics_df: Generics dataset
        medicine_df: Medicine info dataset
        
    Returns:
        Merged DataFrame
    """
    # Merge volume + generics
    merged = volume_df.merge(
        generics_df,
        on=['country', 'brand_name', 'months_postgx'],
        how='left'
    )
    
    # Merge with medicine_info
    merged = merged.merge(
        medicine_df,
        on=['country', 'brand_name'],
        how='left'
    )
    
    print(f"✅ Merged dataset shape: {merged.shape}")
    return merged


# =============================================================================
# DATA CLEANING FUNCTIONS (Section 1.1 - Basic Sanity Checks)
# =============================================================================

def remove_duplicates(df: pd.DataFrame, 
                      subset: list = None,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Remove exact duplicate records.
    
    From Todo Section 1.1:
    - Remove duplicates based on (country, brand_name, months_postgx)
    - Verify no multiple rows exist for same months_postgx per brand
    
    Args:
        df: Input DataFrame
        subset: Columns to check for duplicates (default: country, brand_name, months_postgx)
        verbose: Print removal stats
        
    Returns:
        DataFrame with duplicates removed
    """
    if subset is None:
        subset = ['country', 'brand_name', 'months_postgx']
    
    n_before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep='first')
    n_after = len(df_clean)
    n_removed = n_before - n_after
    
    if verbose:
        if n_removed > 0:
            print(f"⚠️ Removed {n_removed} duplicate rows ({n_removed/n_before*100:.2f}%)")
        else:
            print(f"✅ No duplicates found")
    
    return df_clean


def verify_months_postgx_range(df: pd.DataFrame, 
                               min_month: int = -24, 
                               max_month: int = 23,
                               verbose: bool = True) -> dict:
    """
    Verify months_postgx spans reasonable range.
    
    From Todo Section 1.1:
    - months_postgx should span -24 to +23 typically
    - Flag brands with unusual ranges
    
    Args:
        df: DataFrame with months_postgx column
        min_month: Expected minimum months_postgx
        max_month: Expected maximum months_postgx
        verbose: Print verification results
        
    Returns:
        Dictionary with verification results
    """
    brand_ranges = df.groupby(['country', 'brand_name'])['months_postgx'].agg(['min', 'max', 'count'])
    brand_ranges.columns = ['min_month', 'max_month', 'n_months']
    
    # Brands with unexpected ranges
    unusual_min = brand_ranges[brand_ranges['min_month'] > min_month]
    unusual_max = brand_ranges[brand_ranges['max_month'] < max_month]
    short_history = brand_ranges[brand_ranges['n_months'] < 12]
    
    result = {
        'total_brands': len(brand_ranges),
        'brands_missing_early_months': len(unusual_min),
        'brands_missing_late_months': len(unusual_max),
        'brands_with_short_history': len(short_history),
        'short_history_brands': short_history.reset_index() if len(short_history) > 0 else None
    }
    
    if verbose:
        print(f"📊 months_postgx Range Verification:")
        print(f"   Total brands: {result['total_brands']}")
        print(f"   Brands missing early months (>{min_month}): {result['brands_missing_early_months']}")
        print(f"   Brands missing late months (<{max_month}): {result['brands_missing_late_months']}")
        print(f"   Brands with short history (<12 months): {result['brands_with_short_history']}")
    
    return result


def check_multiple_rows_per_month(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Check for brands with multiple rows for same months_postgx.
    
    From Todo Section 1.1:
    - For each brand, verify no multiple rows exist for same months_postgx
    
    Args:
        df: DataFrame
        verbose: Print results
        
    Returns:
        DataFrame of problematic brand-month combinations
    """
    counts = df.groupby(['country', 'brand_name', 'months_postgx']).size().reset_index(name='count')
    duplicates = counts[counts['count'] > 1]
    
    if verbose:
        if len(duplicates) > 0:
            print(f"⚠️ Found {len(duplicates)} brand-month combinations with multiple rows")
        else:
            print(f"✅ No multiple rows for same months_postgx per brand")
    
    return duplicates


# =============================================================================
# MISSING VALUE HANDLING (Section 1.2)
# =============================================================================

def create_time_to_50pct_features(df: pd.DataFrame, 
                                   vol_norm_col: str = 'vol_norm') -> pd.DataFrame:
    """
    Create time_to_50pct related features.
    
    From Todo Section 1.2:
    - Create reached_50pct = 1 if time_to_50pct is not null, else 0
    - Create time_to_50pct_imputed: Use real value when available, else 24
    
    Args:
        df: DataFrame with vol_norm column
        vol_norm_col: Name of normalized volume column
        
    Returns:
        DataFrame with brand-level time_to_50pct features
    """
    # Filter to post-LOE data
    post_loe = df[df['months_postgx'] >= 0].copy()
    
    # Find first month where vol_norm <= 0.5 for each brand
    def find_time_to_50pct(group):
        group = group.sort_values('months_postgx')
        below_50 = group[group[vol_norm_col] <= 0.5]
        if len(below_50) > 0:
            return below_50.iloc[0]['months_postgx']
        return None
    
    time_to_50 = post_loe.groupby(['country', 'brand_name']).apply(
        find_time_to_50pct, include_groups=False
    ).reset_index()
    time_to_50.columns = ['country', 'brand_name', 'time_to_50pct']
    
    # Create features
    time_to_50['reached_50pct'] = (time_to_50['time_to_50pct'].notna()).astype(int)
    time_to_50['time_to_50pct_imputed'] = time_to_50['time_to_50pct'].fillna(24)  # 24 = never reached
    
    n_reached = time_to_50['reached_50pct'].sum()
    n_total = len(time_to_50)
    
    print(f"✅ Created time_to_50pct features:")
    print(f"   Brands that reached 50%: {n_reached} ({100*n_reached/n_total:.1f}%)")
    print(f"   Brands that didn't reach 50%: {n_total - n_reached} ({100*(n_total-n_reached)/n_total:.1f}%)")
    
    return time_to_50


def impute_avg_vol_regression(df: pd.DataFrame, 
                               avg_j_df: pd.DataFrame,
                               features: list = None) -> pd.DataFrame:
    """
    Impute missing avg_vol using regression on brand characteristics.
    
    From Todo Section 1.2:
    - Fit regression of log(avg_vol) on country, ther_area, hospital_rate, etc.
    - Use for brands with less than 12 months of pre-LOE data
    
    Args:
        df: Full dataset with brand characteristics
        avg_j_df: DataFrame with avg_vol per brand
        features: Features to use for regression (default: country, ther_area, hospital_rate)
        
    Returns:
        avg_j_df with imputed values
    """
    if features is None:
        features = ['ther_area', 'hospital_rate']
    
    # Get brand characteristics
    brand_info = df[['country', 'brand_name', 'ther_area', 'hospital_rate']].drop_duplicates()
    avg_j_with_info = avg_j_df.merge(brand_info, on=['country', 'brand_name'], how='left')
    
    # Identify missing avg_vol
    has_avg_vol = avg_j_with_info['avg_vol'].notna()
    n_missing = (~has_avg_vol).sum()
    
    if n_missing == 0:
        print(f"✅ No missing avg_vol values to impute")
        return avg_j_df
    
    # Encode categorical features
    avg_j_encoded = avg_j_with_info.copy()
    le = LabelEncoder()
    
    if 'ther_area' in features and 'ther_area' in avg_j_encoded.columns:
        avg_j_encoded['ther_area_encoded'] = le.fit_transform(
            avg_j_encoded['ther_area'].fillna('UNKNOWN').astype(str)
        )
        features = ['ther_area_encoded' if f == 'ther_area' else f for f in features]
    
    # Prepare data for regression
    X_cols = [f for f in features if f in avg_j_encoded.columns]
    
    if len(X_cols) == 0:
        print(f"⚠️ No valid features for regression, using median imputation")
        median_val = avg_j_df['avg_vol'].median()
        avg_j_df['avg_vol'] = avg_j_df['avg_vol'].fillna(median_val)
        return avg_j_df
    
    # Fit on non-missing data
    train_mask = avg_j_encoded['avg_vol'].notna()
    X_train = avg_j_encoded.loc[train_mask, X_cols].fillna(0)
    y_train = np.log1p(avg_j_encoded.loc[train_mask, 'avg_vol'])
    
    # Fit regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # Predict for missing
    X_missing = avg_j_encoded.loc[~train_mask, X_cols].fillna(0)
    y_pred = np.expm1(reg.predict(X_missing))
    
    # Fill missing values
    avg_j_df = avg_j_df.copy()
    avg_j_df.loc[~has_avg_vol, 'avg_vol'] = y_pred
    
    print(f"✅ Imputed {n_missing} missing avg_vol values using regression")
    print(f"   Features used: {X_cols}")
    print(f"   R² on training data: {reg.score(X_train, y_train):.3f}")
    
    return avg_j_df


def create_vol_norm_gt1_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create flag for vol_norm > 1 cases.
    
    From Todo Section 1.3:
    - Create vol_norm_gt1 = 1 if vol_norm > 1 else 0
    - Do NOT remove these rows, just flag them
    
    Args:
        df: DataFrame with vol_norm column
        
    Returns:
        DataFrame with vol_norm_gt1 column added
    """
    df = df.copy()
    
    if 'vol_norm' not in df.columns:
        print("⚠️ vol_norm column not found, skipping vol_norm_gt1 flag")
        return df
    
    df['vol_norm_gt1'] = (df['vol_norm'] > 1).astype(int)
    n_gt1 = df['vol_norm_gt1'].sum()
    
    print(f"✅ Created vol_norm_gt1 flag: {n_gt1} rows ({100*n_gt1/len(df):.2f}%) have vol_norm > 1")
    
    return df


# =============================================================================
# COMPREHENSIVE DATA CLEANING PIPELINE
# =============================================================================

def clean_data(df: pd.DataFrame, 
               remove_dups: bool = True,
               verify_range: bool = True,
               add_vol_norm_flag: bool = True,
               verbose: bool = True) -> pd.DataFrame:
    """
    Run comprehensive data cleaning pipeline.
    
    Steps:
    1. Remove duplicates
    2. Verify months_postgx range
    3. Check for multiple rows per month
    4. Add vol_norm_gt1 flag
    
    Args:
        df: Raw merged DataFrame
        remove_dups: Remove duplicate rows
        verify_range: Verify months_postgx range
        add_vol_norm_flag: Add vol_norm > 1 flag
        verbose: Print progress
        
    Returns:
        Cleaned DataFrame
    """
    if verbose:
        print("\n" + "=" * 60)
        print("🧹 DATA CLEANING PIPELINE")
        print("=" * 60)
    
    df_clean = df.copy()
    
    # 1. Remove duplicates
    if remove_dups:
        df_clean = remove_duplicates(df_clean, verbose=verbose)
    
    # 2. Verify months_postgx range
    if verify_range:
        verify_months_postgx_range(df_clean, verbose=verbose)
    
    # 3. Check for multiple rows per month
    check_multiple_rows_per_month(df_clean, verbose=verbose)
    
    # 4. Add vol_norm_gt1 flag
    if add_vol_norm_flag and 'vol_norm' in df_clean.columns:
        df_clean = create_vol_norm_gt1_flag(df_clean)
    
    if verbose:
        print(f"\n✅ Data cleaning complete. Final shape: {df_clean.shape}")
    
    return df_clean


def validate_data(df: pd.DataFrame, name: str = "data") -> dict:
    """
    Validate dataset and return quality report.
    
    Args:
        df: DataFrame to validate
        name: Name for reporting
        
    Returns:
        Dictionary with validation results
    """
    report = {
        'name': name,
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict()
    }
    
    # Check for negative volumes
    if 'volume' in df.columns:
        report['negative_volumes'] = (df['volume'] < 0).sum()
    
    # Check hospital_rate range
    if 'hospital_rate' in df.columns:
        report['hospital_rate_out_of_range'] = (
            (df['hospital_rate'] < 0) | (df['hospital_rate'] > 100)
        ).sum()
    
    return report


def get_unique_brands(df: pd.DataFrame) -> pd.DataFrame:
    """Get unique country-brand combinations."""
    return df[['country', 'brand_name']].drop_duplicates()


def split_train_validation(df: pd.DataFrame, 
                           val_brands_ratio: float = 0.2,
                           random_state: int = RANDOM_STATE) -> tuple:
    """
    Split data into train and validation sets by brand (not by row).
    
    Args:
        df: Full dataset
        val_brands_ratio: Fraction of brands for validation
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df)
    """
    np.random.seed(random_state)
    
    # Get unique brands
    brands = df[['country', 'brand_name']].drop_duplicates()
    n_val = int(len(brands) * val_brands_ratio)
    
    # Random sample for validation
    val_brands = brands.sample(n=n_val, random_state=random_state)
    
    # Split
    val_mask = df.set_index(['country', 'brand_name']).index.isin(
        val_brands.set_index(['country', 'brand_name']).index
    )
    
    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()
    
    print(f"✅ Train: {len(train_df)} rows, {len(train_df[['country', 'brand_name']].drop_duplicates())} brands")
    print(f"✅ Validation: {len(val_df)} rows, {len(val_df[['country', 'brand_name']].drop_duplicates())} brands")
    
    return train_df, val_df


if __name__ == "__main__":
    # Demo: Load and validate data
    print("=" * 60)
    print("DATA LOADER DEMO")
    print("=" * 60)
    
    # Load training data
    volume, generics, medicine = load_all_data(train=True)
    
    # Merge
    merged = merge_datasets(volume, generics, medicine)
    
    # Validate
    report = validate_data(merged, "merged_train")
    print(f"\nValidation report:")
    print(f"  Shape: {report['shape']}")
    print(f"  Duplicates: {report['duplicates']}")
    
    # Split
    train_df, val_df = split_train_validation(merged)
    
    print("\n✅ Data loader demo complete!")
