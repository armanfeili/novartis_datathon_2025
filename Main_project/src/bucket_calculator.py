# =============================================================================
# File: src/bucket_calculator.py
# Description: Functions to compute Avg_j, normalized volume, and erosion buckets
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def compute_avg_j(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-entry average volume (Avg_j) for each country-brand.
    
    Avg_j = mean(volume) over months_postgx in [-12, -1]
    This is the key normalization factor for the PE metric.
    
    Args:
        df: DataFrame with columns [country, brand_name, months_postgx, volume]
        
    Returns:
        DataFrame with columns [country, brand_name, avg_vol]
    """
    # Filter to pre-entry months (-12 to -1)
    pre_entry = df[
        (df['months_postgx'] >= -PRE_ENTRY_MONTHS) & 
        (df['months_postgx'] <= -1)
    ].copy()
    
    # Compute mean volume per brand
    avg_j = pre_entry.groupby(['country', 'brand_name'])['volume'].mean().reset_index()
    avg_j.columns = ['country', 'brand_name', 'avg_vol']
    
    # Handle brands with no pre-entry data
    all_brands = df[['country', 'brand_name']].drop_duplicates()
    avg_j = all_brands.merge(avg_j, on=['country', 'brand_name'], how='left')
    
    print(f"✅ Computed avg_vol for {len(avg_j)} brands")
    print(f"   Brands with valid avg_vol: {avg_j['avg_vol'].notna().sum()}")
    
    return avg_j


def compute_normalized_volume(df: pd.DataFrame, avg_j_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized volume: vol_norm = volume / Avg_j
    
    Args:
        df: DataFrame with volume data
        avg_j_df: DataFrame with avg_vol per brand
        
    Returns:
        DataFrame with added vol_norm column
    """
    merged = df.merge(avg_j_df, on=['country', 'brand_name'], how='left')
    
    # Compute normalized volume
    merged['vol_norm'] = merged['volume'] / merged['avg_vol']
    
    # Handle division by zero and inf
    merged['vol_norm'] = merged['vol_norm'].replace([np.inf, -np.inf], np.nan)
    
    return merged


def compute_mean_erosion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean generic erosion for each country-brand.
    
    Mean erosion = mean(vol_norm) over months_postgx in [0, 23]
    
    Args:
        df: DataFrame with vol_norm column
        
    Returns:
        DataFrame with columns [country, brand_name, mean_erosion]
    """
    # Filter to post-entry months (0-23)
    post_entry = df[
        (df['months_postgx'] >= 0) & 
        (df['months_postgx'] <= 23)
    ].copy()
    
    # Compute mean normalized volume (erosion)
    mean_erosion = post_entry.groupby(['country', 'brand_name'])['vol_norm'].mean().reset_index()
    mean_erosion.columns = ['country', 'brand_name', 'mean_erosion']
    
    return mean_erosion


def assign_buckets(mean_erosion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign erosion bucket based on mean erosion.
    
    Bucket 1: mean_erosion in [0, 0.25] → High erosion (weighted 2x in metric!)
    Bucket 2: mean_erosion > 0.25 → Lower erosion (weighted 1x)
    
    Args:
        mean_erosion_df: DataFrame with mean_erosion column
        
    Returns:
        DataFrame with added bucket column
    """
    df = mean_erosion_df.copy()
    
    df['bucket'] = np.where(
        (df['mean_erosion'] >= 0) & (df['mean_erosion'] <= BUCKET_1_THRESHOLD),
        1,  # Bucket 1: high erosion
        2   # Bucket 2: lower erosion
    )
    
    # Count distribution
    bucket_counts = df['bucket'].value_counts().sort_index()
    print(f"✅ Bucket distribution:")
    print(f"   Bucket 1 (high erosion): {bucket_counts.get(1, 0)} brands")
    print(f"   Bucket 2 (lower erosion): {bucket_counts.get(2, 0)} brands")
    
    return df


def create_auxiliary_file(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Create auxiliary file with avg_vol and bucket for metric calculation.
    
    This file is used during evaluation to:
    1. Normalize prediction errors by avg_vol
    2. Apply bucket weights (Bucket 1 = 2x)
    
    Args:
        df: Full merged dataset
        save: If True, save to data/processed/
        
    Returns:
        DataFrame with columns [country, brand_name, avg_vol, mean_erosion, bucket]
    """
    print("\n" + "=" * 60)
    print("CREATING AUXILIARY FILE")
    print("=" * 60)
    
    # Step 1: Compute avg_vol (pre-entry average)
    avg_j = compute_avg_j(df)
    
    # Step 2: Add normalized volume to data
    df_with_norm = compute_normalized_volume(df, avg_j)
    
    # Step 3: Compute mean erosion
    mean_erosion = compute_mean_erosion(df_with_norm)
    
    # Step 4: Assign buckets
    buckets = assign_buckets(mean_erosion)
    
    # Step 5: Merge avg_vol and bucket
    aux = avg_j.merge(
        buckets[['country', 'brand_name', 'mean_erosion', 'bucket']],
        on=['country', 'brand_name'],
        how='left'
    )
    
    if save:
        output_path = DATA_PROCESSED / "aux_bucket_avgvol.csv"
        aux.to_csv(output_path, index=False)
        print(f"\n✅ Saved auxiliary file to: {output_path}")
    
    return aux


def get_bucket_for_brand(aux_df: pd.DataFrame, country: str, brand_name: str) -> int:
    """Get bucket assignment for a specific brand."""
    mask = (aux_df['country'] == country) & (aux_df['brand_name'] == brand_name)
    if mask.sum() == 0:
        return None
    return aux_df.loc[mask, 'bucket'].iloc[0]


def get_avg_vol_for_brand(aux_df: pd.DataFrame, country: str, brand_name: str) -> float:
    """Get avg_vol for a specific brand."""
    mask = (aux_df['country'] == country) & (aux_df['brand_name'] == brand_name)
    if mask.sum() == 0:
        return None
    return aux_df.loc[mask, 'avg_vol'].iloc[0]


if __name__ == "__main__":
    # Demo: Create auxiliary file
    print("=" * 60)
    print("BUCKET CALCULATOR DEMO")
    print("=" * 60)
    
    from data_loader import load_all_data, merge_datasets
    
    # Load data
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    
    # Create auxiliary file
    aux = create_auxiliary_file(merged, save=True)
    
    print(f"\nAuxiliary file preview:")
    print(aux.head(10))
    
    print("\n✅ Bucket calculator demo complete!")
