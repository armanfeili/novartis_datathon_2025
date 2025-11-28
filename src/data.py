"""
Data loading and panel construction for Novartis Datathon 2025.

This module handles:
- Loading raw CSV files (volume, generics, medicine_info)
- Building unified panel dataset
- Computing pre-entry statistics (avg_vol_12m, bucket for training)
- Missing value handling
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import timer, get_project_root, is_colab

logger = logging.getLogger(__name__)


def load_raw_data(config: dict, split: str = "train") -> Dict[str, pd.DataFrame]:
    """
    Load all three datasets for train or test split using data.yaml config.
    
    Args:
        config: Loaded data.yaml config dict
        split: "train" or "test"
    
    Returns:
        Dictionary with keys: 'volume', 'generics', 'medicine_info'
    
    Example:
        config = load_config('configs/data.yaml')
        train_data = load_raw_data(config, split='train')
    """
    # Determine base path based on environment
    if is_colab():
        base_dir = Path(config.get('drive', {}).get('raw_dir', 'data/raw'))
    else:
        base_dir = get_project_root() / config['paths']['raw_dir']
    
    files = config['files'][split]
    
    data = {}
    with timer(f"Load {split} data"):
        for key in ['volume', 'generics', 'medicine_info']:
            file_path = base_dir / files[key]
            if file_path.exists():
                logger.info(f"Loading {key} from {file_path}")
                data[key] = pd.read_csv(file_path)
                logger.info(f"  -> {len(data[key]):,} rows loaded")
            else:
                raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return data


def prepare_base_panel(
    volume_df: pd.DataFrame,
    generics_df: pd.DataFrame,
    medicine_info_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create unified panel with all features joined.
    
    Joins:
    - volume ← generics on (country, brand_name, months_postgx)
    - result ← medicine_info on (country, brand_name)
    
    Args:
        volume_df: Volume data with columns [country, brand_name, month, months_postgx, volume]
        generics_df: Generics data with columns [country, brand_name, months_postgx, n_gxs]
        medicine_info_df: Static drug info with columns [country, brand_name, ther_area, ...]
    
    Returns:
        Panel DataFrame with all columns merged
    """
    with timer("Build base panel"):
        # Join keys
        time_keys = ['country', 'brand_name', 'months_postgx']
        static_keys = ['country', 'brand_name']
        
        # Merge volume with generics on time-varying keys
        panel = volume_df.merge(
            generics_df,
            on=time_keys,
            how='left'
        )
        logger.info(f"After volume-generics merge: {len(panel):,} rows")
        
        # Merge with static medicine info
        panel = panel.merge(
            medicine_info_df,
            on=static_keys,
            how='left'
        )
        logger.info(f"After medicine_info merge: {len(panel):,} rows")
        
        # Validate no duplicate keys
        duplicates = panel[time_keys].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate (country, brand_name, months_postgx) keys!")
        
        # Sort for consistent ordering
        panel = panel.sort_values(time_keys).reset_index(drop=True)
        
    return panel


def compute_pre_entry_stats(
    panel_df: pd.DataFrame,
    is_train: bool = True
) -> pd.DataFrame:
    """
    Compute pre-entry statistics for each series.
    
    Args:
        panel_df: Panel data (train or test)
        is_train: If True, computes target-dependent stats (bucket, mean_erosion)
    
    Always computes:
    - avg_vol_12m: Mean volume over months [-12, -1]
    
    If is_train=True, also computes:
    - mean_erosion: Mean of normalized post-entry volumes (y_norm for months 0-23)
    - bucket: 1 if mean_erosion <= 0.25 else 2
    
    CRITICAL: bucket is NEVER computed on test data and NEVER used as a feature.
    
    Returns:
        Panel with pre-entry stats added as columns (merged back to all rows)
    """
    with timer("Compute pre-entry stats"):
        df = panel_df.copy()
        series_keys = ['country', 'brand_name']
        
        # Compute avg_vol_12m: mean volume over months [-12, -1]
        pre_entry_mask = (df['months_postgx'] >= -12) & (df['months_postgx'] <= -1)
        pre_entry_data = df[pre_entry_mask].groupby(series_keys)['volume'].mean()
        pre_entry_data = pre_entry_data.reset_index()
        pre_entry_data.columns = series_keys + ['avg_vol_12m']
        
        # Handle series with no pre-entry data (use global fallback or mark)
        if pre_entry_data['avg_vol_12m'].isna().any():
            logger.warning("Some series have no pre-entry volume data (months -12 to -1)")
        
        # Merge back to panel
        df = df.merge(pre_entry_data, on=series_keys, how='left')
        
        # For series with missing avg_vol_12m, use available pre-entry data or global median
        if df['avg_vol_12m'].isna().any():
            # Try using any available pre-entry months
            all_pre_mask = df['months_postgx'] < 0
            fallback_avg = df[all_pre_mask].groupby(series_keys)['volume'].mean().reset_index()
            fallback_avg.columns = series_keys + ['avg_vol_fallback']
            
            df = df.merge(fallback_avg, on=series_keys, how='left')
            df['avg_vol_12m'] = df['avg_vol_12m'].fillna(df['avg_vol_fallback'])
            df = df.drop(columns=['avg_vol_fallback'])
            
            # Final fallback: global median (for test series with no pre-entry at all)
            global_median = df[df['avg_vol_12m'].notna()]['avg_vol_12m'].median()
            df['avg_vol_12m'] = df['avg_vol_12m'].fillna(global_median)
        
        logger.info(f"avg_vol_12m range: [{df['avg_vol_12m'].min():.2f}, {df['avg_vol_12m'].max():.2f}]")
        
        # Training-only: compute bucket classification
        if is_train:
            # First compute y_norm for post-entry months
            post_entry_mask = (df['months_postgx'] >= 0) & (df['months_postgx'] <= 23)
            
            # Compute mean erosion (mean of y_norm over months 0-23)
            erosion_df = df[post_entry_mask].copy()
            erosion_df['y_norm'] = erosion_df['volume'] / erosion_df['avg_vol_12m']
            
            mean_erosion = erosion_df.groupby(series_keys)['y_norm'].mean().reset_index()
            mean_erosion.columns = series_keys + ['mean_erosion']
            
            # Classify bucket
            mean_erosion['bucket'] = (mean_erosion['mean_erosion'] <= 0.25).astype(int)
            mean_erosion['bucket'] = mean_erosion['bucket'].map({1: 1, 0: 2})  # 1=high erosion, 2=low
            
            df = df.merge(mean_erosion, on=series_keys, how='left')
            
            # Log bucket distribution
            bucket_counts = df.drop_duplicates(series_keys)['bucket'].value_counts()
            logger.info(f"Bucket distribution:\n{bucket_counts.to_string()}")
        
        # Create y_norm column for training
        if is_train:
            df['y_norm'] = df['volume'] / df['avg_vol_12m']
        
    return df


def handle_missing_values(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply missing value strategy.
    
    | Column          | Strategy                              |
    |-----------------|---------------------------------------|
    | volume          | Keep as-is (rare NaN)                 |
    | n_gxs           | Forward-fill per series, then 0       |
    | hospital_rate   | Median by ther_area + flag            |
    | ther_area       | "Unknown" category                    |
    | main_package    | "Unknown" category                    |
    | biological      | False if missing + flag               |
    | small_molecule  | False if missing + flag               |
    
    Args:
        panel_df: Panel data with potential missing values
        
    Returns:
        Panel with missing values handled
    """
    with timer("Handle missing values"):
        df = panel_df.copy()
        series_keys = ['country', 'brand_name']
        
        # n_gxs: forward-fill per series, then fill with 0
        if 'n_gxs' in df.columns:
            original_na = df['n_gxs'].isna().sum()
            df['n_gxs'] = df.groupby(series_keys)['n_gxs'].ffill()
            df['n_gxs'] = df['n_gxs'].fillna(0)
            logger.info(f"n_gxs: filled {original_na} missing values")
        
        # hospital_rate: median by ther_area + missing flag
        if 'hospital_rate' in df.columns:
            df['hospital_rate_missing'] = df['hospital_rate'].isna().astype(int)
            
            # Compute median by ther_area
            if 'ther_area' in df.columns:
                ther_area_median = df.groupby('ther_area')['hospital_rate'].transform('median')
                df['hospital_rate'] = df['hospital_rate'].fillna(ther_area_median)
            
            # Global median fallback
            global_median = df['hospital_rate'].median()
            df['hospital_rate'] = df['hospital_rate'].fillna(global_median)
            logger.info(f"hospital_rate: filled with ther_area medians, flag added")
        
        # Categorical: fill with "Unknown"
        for col in ['ther_area', 'main_package']:
            if col in df.columns:
                original_na = df[col].isna().sum()
                df[col] = df[col].fillna("Unknown")
                if original_na > 0:
                    logger.info(f"{col}: filled {original_na} missing with 'Unknown'")
        
        # Boolean: fill with False + missing flag
        for col in ['biological', 'small_molecule']:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isna().astype(int)
                df[col] = df[col].fillna(False).astype(bool)
        
        # Log remaining NaN summary
        nan_summary = df.isna().sum()
        nan_cols = nan_summary[nan_summary > 0]
        if len(nan_cols) > 0:
            logger.warning(f"Remaining NaN counts:\n{nan_cols.to_string()}")
        else:
            logger.info("No missing values remaining")
    
    return df


def get_series_ids(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get unique series identifiers from panel.
    
    Args:
        panel_df: Panel data
        
    Returns:
        DataFrame with unique (country, brand_name) combinations
    """
    return panel_df[['country', 'brand_name']].drop_duplicates().reset_index(drop=True)


def get_series_count(panel_df: pd.DataFrame) -> int:
    """
    Count unique series in panel.
    
    Args:
        panel_df: Panel data
        
    Returns:
        Number of unique (country, brand_name) combinations
    """
    return panel_df[['country', 'brand_name']].drop_duplicates().shape[0]
