# =============================================================================
# File: src/data_loader.py
# Description: Functions to load, validate, and merge all datasets
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys

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
