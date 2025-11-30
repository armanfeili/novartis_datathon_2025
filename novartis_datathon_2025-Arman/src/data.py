"""Data loading and panel construction for Novartis Datathon 2025.

This module handles:
- Loading raw CSV files (volume, generics, medicine_info)
- Building unified panel dataset
- Computing pre-entry statistics (avg_vol_12m, bucket for training)
- Missing value handling
- Panel caching for faster repeated loads
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import timer, get_project_root, is_colab, load_config

logger = logging.getLogger(__name__)

# =============================================================================
# Canonical Column Constants (aligned with configs/data.yaml)
# =============================================================================
ID_COLS: List[str] = ['country', 'brand_name']
TIME_COL: str = 'months_postgx'
CALENDAR_MONTH_COL: str = 'month'
RAW_TARGET_COL: str = 'volume'
MODEL_TARGET_COL: str = 'y_norm'

# Meta columns that must NEVER be used as model features
# Must stay in sync with META_COLS in train.py and columns.meta_cols in data.yaml
META_COLS: List[str] = [
    'country', 'brand_name', 'months_postgx', 'bucket', 
    'avg_vol_12m', 'y_norm', 'volume', 'mean_erosion', 'month',
    # Categorical identifiers - should not be features directly
    'ther_area', 'main_package', 'time_bucket',
]

# Expected column types for validation
EXPECTED_DTYPES = {
    'country': 'object',
    'brand_name': 'object',
    'months_postgx': 'int',
    'volume': 'float',
    'n_gxs': 'int',
    'ther_area': 'object',
    'main_package': 'object',
    'hospital_rate': 'float',
    'biological': 'bool',
    'small_molecule': 'bool',
}

# Expected value ranges for validation
VALUE_RANGES = {
    'months_postgx': (-24, 23),  # Pre-entry to 2 years post
    'volume': (0, None),  # Non-negative
    'n_gxs': (0, 50),  # Reasonable generic count
    'hospital_rate': (0, 100),  # Percentage
    'y_norm': (0, 5),  # Normalized volume typically 0-1.5, allow up to 5 for edge cases
}


# =============================================================================
# Validation Functions
# =============================================================================

def validate_dataframe_schema(df: pd.DataFrame, name: str, required_cols: List[str]) -> None:
    """
    Validate that a DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        name: Name for logging (e.g., 'volume', 'generics')
        required_cols: List of required column names
    
    Raises:
        ValueError: If required columns are missing
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} DataFrame missing required columns: {missing}")
    logger.debug(f"{name}: schema validated, {len(df.columns)} columns present")


def validate_value_ranges(df: pd.DataFrame, name: str) -> None:
    """
    Validate that numeric columns are within expected ranges.
    
    Args:
        df: DataFrame to validate
        name: Name for logging
    
    Logs warnings for out-of-range values but does not raise errors.
    """
    for col, (min_val, max_val) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        issues = []
        if min_val is not None and (col_data < min_val).any():
            below_count = (col_data < min_val).sum()
            issues.append(f"{below_count} values below {min_val}")
        if max_val is not None and (col_data > max_val).any():
            above_count = (col_data > max_val).sum()
            issues.append(f"{above_count} values above {max_val}")
        
        if issues:
            logger.warning(f"{name}.{col}: {', '.join(issues)}")


def validate_no_duplicates(df: pd.DataFrame, keys: List[str], name: str) -> None:
    """
    Validate that there are no duplicate keys in the DataFrame.
    
    Args:
        df: DataFrame to validate
        keys: Key columns that should be unique
        name: Name for logging
        
    Raises:
        ValueError: If duplicates are found
    """
    duplicates = df[keys].duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"{name}: Found {duplicates} duplicate {keys} combinations!")


def log_data_statistics(df: pd.DataFrame, name: str) -> None:
    """
    Log comprehensive statistics about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        name: Name for logging
    """
    logger.info(f"=== {name} Statistics ===")
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Missing values
    missing_pct = (df.isna().sum() / len(df) * 100)
    missing_cols = missing_pct[missing_pct > 0]
    if len(missing_cols) > 0:
        logger.info(f"  Missing values:")
        for col, pct in missing_cols.items():
            logger.info(f"    {col}: {pct:.1f}%")
    else:
        logger.info(f"  No missing values")
    
    # Unique series count (if ID columns present)
    if all(col in df.columns for col in ID_COLS):
        n_series = df[ID_COLS].drop_duplicates().shape[0]
        logger.info(f"  Unique series: {n_series:,}")
    
    # Time range (if time column present)
    if TIME_COL in df.columns:
        logger.info(f"  {TIME_COL} range: [{df[TIME_COL].min()}, {df[TIME_COL].max()}]")


# =============================================================================
# Panel Schema Validation
# =============================================================================

# Required columns that MUST be present after panel construction
PANEL_REQUIRED_COLUMNS = [
    'country', 'brand_name', 'months_postgx', 'volume', 'n_gxs',
    'ther_area', 'main_package', 'hospital_rate', 'biological', 'small_molecule'
]


def validate_panel_schema(
    panel_df: pd.DataFrame,
    split: str = "train",
    raise_on_error: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that a panel DataFrame has required columns and no duplicate keys.
    
    This function should be called after building each panel to ensure data integrity.
    
    Args:
        panel_df: Panel DataFrame to validate
        split: "train" or "test" (for logging)
        raise_on_error: If True, raises ValueError on validation failure
        
    Returns:
        Tuple of (is_valid, list of issues)
        
    Raises:
        ValueError: If raise_on_error=True and validation fails
    """
    issues = []
    
    # Check 1: Required columns present
    missing_cols = set(PANEL_REQUIRED_COLUMNS) - set(panel_df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check 2: No duplicate keys
    key_cols = ['country', 'brand_name', 'months_postgx']
    duplicates = panel_df.duplicated(subset=key_cols).sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate {key_cols} combinations")
    
    # Check 3: For train, bucket and mean_erosion should be present
    if split == "train":
        train_required = ['bucket', 'mean_erosion', 'avg_vol_12m', 'y_norm']
        missing_train = set(train_required) - set(panel_df.columns)
        if missing_train:
            issues.append(f"Training panel missing: {missing_train}")
    
    # Check 4: For test, should NOT have bucket or mean_erosion
    if split == "test":
        forbidden_test = {'bucket', 'mean_erosion', 'y_norm'}
        present_forbidden = forbidden_test & set(panel_df.columns)
        if present_forbidden:
            issues.append(f"Test panel should not have: {present_forbidden}")
    
    # Check 5: Data types are reasonable
    if 'months_postgx' in panel_df.columns:
        if not pd.api.types.is_integer_dtype(panel_df['months_postgx']):
            issues.append(f"months_postgx should be integer, got {panel_df['months_postgx'].dtype}")
    
    if 'volume' in panel_df.columns:
        if not pd.api.types.is_numeric_dtype(panel_df['volume']):
            issues.append(f"volume should be numeric, got {panel_df['volume'].dtype}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info(f"Panel schema validation passed for {split} ({len(panel_df):,} rows)")
    else:
        for issue in issues:
            logger.error(f"Panel schema validation failed: {issue}")
        if raise_on_error:
            raise ValueError(f"Panel schema validation failed:\n" + "\n".join(issues))
    
    return is_valid, issues


def verify_no_future_leakage(
    panel_df: pd.DataFrame,
    scenario: int,
    cutoff_column: str = 'months_postgx'
) -> Tuple[bool, List[str]]:
    """
    Verify that no features use data beyond the scenario cutoff.
    
    This is a critical leakage check that ensures:
    - Scenario 1: Only uses data from months_postgx < 0 (pre-entry only)
    - Scenario 2: Only uses data from months_postgx < 6 (pre-entry + first 6 months)
    
    Args:
        panel_df: Panel or feature DataFrame to check
        scenario: 1 or 2
        cutoff_column: Column to check for cutoff (default: months_postgx)
        
    Returns:
        Tuple of (is_clean, list of violations)
    """
    violations = []
    
    # Determine cutoff based on scenario
    if scenario == 1:
        cutoff = 0  # Only pre-entry data allowed
    elif scenario == 2:
        cutoff = 6  # Pre-entry + first 6 months allowed
    else:
        violations.append(f"Invalid scenario: {scenario}")
        return False, violations
    
    # Check if cutoff_column exists
    if cutoff_column not in panel_df.columns:
        logger.warning(f"Cutoff column '{cutoff_column}' not in DataFrame - cannot verify future leakage")
        return True, []
    
    # Get max months_postgx in the data
    max_month = panel_df[cutoff_column].max()
    
    if max_month >= cutoff:
        # This is expected for training rows, but features should be computed from < cutoff
        logger.debug(f"Data contains months_postgx up to {max_month} (cutoff={cutoff} for S{scenario})")
    
    # Check for feature columns that might contain future information
    suspicious_patterns = [
        f'_month_{cutoff}', f'_month_{cutoff+1}', f'_postgx_{cutoff}',
        '_future_', '_forecast_', '_pred_'
    ]
    
    for col in panel_df.columns:
        col_lower = col.lower()
        for pattern in suspicious_patterns:
            if pattern in col_lower:
                violations.append(f"Column '{col}' name suggests future data usage")
    
    # Log result
    is_clean = len(violations) == 0
    if is_clean:
        logger.info(f"No future leakage detected for Scenario {scenario} (cutoff={cutoff})")
    else:
        for v in violations:
            logger.warning(f"Potential future leakage: {v}")
    
    return is_clean, violations


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
    
    # Required columns for each file type
    required_cols = {
        'volume': ['country', 'brand_name', 'month', 'months_postgx', 'volume'],
        'generics': ['country', 'brand_name', 'months_postgx', 'n_gxs'],
        'medicine_info': ['country', 'brand_name', 'ther_area'],
    }
    
    data = {}
    with timer(f"Load {split} data"):
        for key in ['volume', 'generics', 'medicine_info']:
            file_path = base_dir / files[key]
            if file_path.exists():
                logger.info(f"Loading {key} from {file_path}")
                df = pd.read_csv(file_path)
                
                # Validate schema
                validate_dataframe_schema(df, f"{split}/{key}", required_cols[key])
                
                # Log statistics
                log_data_statistics(df, f"{split}/{key}")
                
                # Validate value ranges
                validate_value_ranges(df, f"{split}/{key}")
                
                data[key] = df
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
    - volume <- generics on (country, brand_name, months_postgx)
    - result <- medicine_info on (country, brand_name)
    
    Args:
        volume_df: Volume data with columns [country, brand_name, month, months_postgx, volume]
        generics_df: Generics data with columns [country, brand_name, months_postgx, n_gxs]
        medicine_info_df: Static drug info with columns [country, brand_name, ther_area, ...]
    
    Returns:
        Panel DataFrame with all columns merged
        
    Raises:
        ValueError: If duplicate keys are found after merge
    """
    with timer("Build base panel"):
        # Join keys
        time_keys = ['country', 'brand_name', 'months_postgx']
        static_keys = ['country', 'brand_name']
        
        # Log input shapes
        logger.info(f"Input shapes: volume={volume_df.shape}, generics={generics_df.shape}, medicine_info={medicine_info_df.shape}")
        
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
        
        # Validate no duplicate keys (CRITICAL check)
        validate_no_duplicates(panel, time_keys, "panel")
        
        # Sort for consistent ordering
        panel = panel.sort_values(time_keys).reset_index(drop=True)
        
        # Log panel statistics
        log_data_statistics(panel, "Base Panel")
        
        # Validate value ranges
        validate_value_ranges(panel, "panel")
        
    return panel


def compute_pre_entry_stats(
    panel_df: pd.DataFrame,
    is_train: bool = True,
    bucket_threshold: float = None,
    run_config: dict = None
) -> pd.DataFrame:
    """
    Compute pre-entry statistics for each series.
    
    Args:
        panel_df: Panel data (train or test)
        is_train: If True, computes target-dependent stats (bucket, mean_erosion)
        bucket_threshold: Threshold for bucket classification (mean_erosion <= threshold -> Bucket 1).
                         If None, loads from run_config or uses default 0.25.
        run_config: Run configuration dict containing official_metric.bucket_threshold.
                   If None and bucket_threshold is None, uses default 0.25.
    
    Always computes:
    - avg_vol_12m: Mean volume over months [-12, -1]
    - pre_entry_months_available: Count of pre-entry months for each series (for diagnostics)
    
    If is_train=True, also computes:
    - y_norm: volume / avg_vol_12m (normalized target)
    - mean_erosion: Mean of y_norm over months 0-23
    - bucket: 1 if mean_erosion <= bucket_threshold else 2
    
    CRITICAL: bucket is NEVER computed on test data and NEVER used as a feature.
    
    Edge case handling for series with < 12 pre-entry months:
    - Uses all available pre-entry months (months_postgx < 0)
    - If no pre-entry data at all, falls back to ther_area median, then global median
    
    Returns:
        Panel with pre-entry stats added as columns (merged back to all rows)
    """
    # Determine bucket threshold from config or default
    if bucket_threshold is None:
        if run_config is not None:
            bucket_threshold = run_config.get('official_metric', {}).get('bucket_threshold', 0.25)
        else:
            bucket_threshold = 0.25  # Default from official metric_calculation.py
    
    with timer("Compute pre-entry stats"):
        df = panel_df.copy()
        series_keys = ['country', 'brand_name']
        
        # =====================================================================
        # Step 1: Compute avg_vol_12m with robust fallbacks
        # =====================================================================
        
        # Primary: mean volume over months [-12, -1]
        pre_entry_mask = (df['months_postgx'] >= -12) & (df['months_postgx'] <= -1)
        pre_entry_data = df[pre_entry_mask].groupby(series_keys).agg(
            avg_vol_12m=('volume', 'mean'),
            pre_entry_months_available=('volume', 'count')
        ).reset_index()
        
        # Log series with less than 12 pre-entry months
        short_pre_entry = pre_entry_data[pre_entry_data['pre_entry_months_available'] < 12]
        if len(short_pre_entry) > 0:
            logger.warning(
                f"{len(short_pre_entry)} series have <12 pre-entry months "
                f"(min={short_pre_entry['pre_entry_months_available'].min()}, "
                f"median={short_pre_entry['pre_entry_months_available'].median():.0f})"
            )
        
        # Merge back to panel
        df = df.merge(pre_entry_data, on=series_keys, how='left')
        
        # =====================================================================
        # Step 2: Handle missing avg_vol_12m with fallback hierarchy
        # =====================================================================
        
        missing_avg_vol = df['avg_vol_12m'].isna()
        n_missing = df[missing_avg_vol][series_keys].drop_duplicates().shape[0]
        
        if n_missing > 0:
            logger.info(f"Handling {n_missing} series with no standard pre-entry data...")
            
            # Fallback 1: Use ANY pre-entry months (months_postgx < 0)
            all_pre_mask = df['months_postgx'] < 0
            fallback_avg = df[all_pre_mask].groupby(series_keys)['volume'].mean().reset_index()
            fallback_avg.columns = series_keys + ['avg_vol_fallback1']
            
            df = df.merge(fallback_avg, on=series_keys, how='left')
            filled_1 = df['avg_vol_12m'].isna() & df['avg_vol_fallback1'].notna()
            df.loc[filled_1, 'avg_vol_12m'] = df.loc[filled_1, 'avg_vol_fallback1']
            logger.info(f"  Fallback 1 (any pre-entry): filled {filled_1.sum()} rows")
            df = df.drop(columns=['avg_vol_fallback1'])
            
            # Fallback 2: ther_area median (if available)
            if 'ther_area' in df.columns:
                still_missing = df['avg_vol_12m'].isna()
                if still_missing.any():
                    ther_area_median = df[df['avg_vol_12m'].notna()].groupby('ther_area')['avg_vol_12m'].median()
                    df['_ther_area_fallback'] = df['ther_area'].map(ther_area_median)
                    filled_2 = still_missing & df['_ther_area_fallback'].notna()
                    df.loc[filled_2, 'avg_vol_12m'] = df.loc[filled_2, '_ther_area_fallback']
                    logger.info(f"  Fallback 2 (ther_area median): filled {filled_2.sum()} rows")
                    df = df.drop(columns=['_ther_area_fallback'])
            
            # Fallback 3: Global median (last resort)
            still_missing = df['avg_vol_12m'].isna()
            if still_missing.any():
                global_median = df[df['avg_vol_12m'].notna()]['avg_vol_12m'].median()
                df.loc[still_missing, 'avg_vol_12m'] = global_median
                logger.info(f"  Fallback 3 (global median={global_median:.2f}): filled {still_missing.sum()} rows")
        
        # Log avg_vol_12m distribution
        logger.info(f"avg_vol_12m distribution:")
        logger.info(f"  Range: [{df['avg_vol_12m'].min():.2f}, {df['avg_vol_12m'].max():.2f}]")
        logger.info(f"  Median: {df['avg_vol_12m'].median():.2f}")
        logger.info(f"  Mean: {df['avg_vol_12m'].mean():.2f}")
        
        # =====================================================================
        # Step 3: Training-only computations
        # =====================================================================
        
        if is_train:
            # Compute y_norm (normalized target)
            df['y_norm'] = df['volume'] / df['avg_vol_12m']
            
            # Handle edge cases in y_norm
            df['y_norm'] = df['y_norm'].replace([np.inf, -np.inf], np.nan)
            if df['y_norm'].isna().any():
                logger.warning(f"y_norm has {df['y_norm'].isna().sum()} NaN/inf values (will use 1.0)")
                df['y_norm'] = df['y_norm'].fillna(1.0)
            
            # Compute mean_erosion over post-entry months (0-23)
            post_entry_mask = (df['months_postgx'] >= 0) & (df['months_postgx'] <= 23)
            erosion_df = df[post_entry_mask].groupby(series_keys)['y_norm'].mean().reset_index()
            erosion_df.columns = series_keys + ['mean_erosion']
            
            # Classify bucket based on mean erosion using configurable threshold
            # Bucket 1: High erosion (mean_erosion <= bucket_threshold)
            # Bucket 2: Low erosion (mean_erosion > bucket_threshold)
            erosion_df['bucket'] = (erosion_df['mean_erosion'] <= bucket_threshold).astype(int)
            erosion_df['bucket'] = erosion_df['bucket'].map({1: 1, 0: 2})
            
            logger.info(f"Bucket classification using threshold={bucket_threshold}")
            
            df = df.merge(erosion_df, on=series_keys, how='left')
            
            # Log bucket distribution
            bucket_df = df[series_keys + ['bucket']].drop_duplicates()
            bucket_counts = bucket_df['bucket'].value_counts().sort_index()
            logger.info(f"Bucket distribution (train):")
            for bucket, count in bucket_counts.items():
                pct = count / len(bucket_df) * 100
                logger.info(f"  Bucket {bucket}: {count:,} series ({pct:.1f}%)")
            
            # Log y_norm distribution
            logger.info(f"y_norm distribution (train):")
            logger.info(f"  Range: [{df['y_norm'].min():.3f}, {df['y_norm'].max():.3f}]")
            logger.info(f"  Median: {df['y_norm'].median():.3f}")
            logger.info(f"  Mean: {df['y_norm'].mean():.3f}")
            
            # Validate y_norm range
            validate_value_ranges(df[['y_norm']], "y_norm")
        
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


# =============================================================================
# Panel Caching Functions
# =============================================================================

# Check for parquet support
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    try:
        import fastparquet
        PARQUET_AVAILABLE = True
    except ImportError:
        PARQUET_AVAILABLE = False
        logger.warning("Parquet support not available. Using pickle for caching.")


def _get_cache_path(interim_dir: Path, split: str) -> Path:
    """Get cache file path, using parquet if available, else pickle."""
    ext = ".parquet" if PARQUET_AVAILABLE else ".pkl"
    return interim_dir / f"panel_{split}{ext}"


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to cache file."""
    if PARQUET_AVAILABLE:
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)


def _load_cache(path: Path) -> pd.DataFrame:
    """Load DataFrame from cache file."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_pickle(path)


def get_panel(
    split: str,
    config: dict,
    use_cache: bool = True,
    force_rebuild: bool = False
) -> pd.DataFrame:
    """
    Get panel DataFrame with caching support.
    
    This function is the primary entry point for obtaining panels.
    It handles:
    - Loading cached panels from parquet files
    - Building panels from raw data if needed
    - Saving built panels for future use
    
    Args:
        split: "train" or "test"
        config: Loaded data.yaml config dict
        use_cache: If True, try to load from cache first
        force_rebuild: If True, rebuild even if cache exists
    
    Returns:
        Panel DataFrame with all preprocessing applied
    
    Cache location:
        {interim_dir}/panel_{split}.parquet
    
    Example:
        config = load_config('configs/data.yaml')
        train_panel = get_panel('train', config)
        test_panel = get_panel('test', config, force_rebuild=True)
    """
    # Determine cache path
    if is_colab():
        interim_dir = Path(config.get('drive', {}).get('interim_dir', 'data/interim'))
    else:
        interim_dir = get_project_root() / config['paths']['interim_dir']
    
    cache_path = _get_cache_path(interim_dir, split)
    
    # Check cache (also check for other format if one doesn't exist)
    alt_path = interim_dir / f"panel_{split}.pkl" if PARQUET_AVAILABLE else interim_dir / f"panel_{split}.parquet"
    
    if use_cache and not force_rebuild:
        if cache_path.exists():
            with timer(f"Load cached {split} panel"):
                logger.info(f"Loading cached panel from {cache_path}")
                panel = _load_cache(cache_path)
                logger.info(f"Loaded {len(panel):,} rows, {panel.shape[1]} columns")
                return panel
        elif alt_path.exists():
            with timer(f"Load cached {split} panel (alternate format)"):
                logger.info(f"Loading cached panel from {alt_path}")
                panel = _load_cache(alt_path)
                logger.info(f"Loaded {len(panel):,} rows, {panel.shape[1]} columns")
                return panel
    
    # Build panel from scratch
    logger.info(f"Building {split} panel from raw data...")
    
    # Load raw data
    raw_data = load_raw_data(config, split=split)
    
    # Build base panel
    panel = prepare_base_panel(
        raw_data['volume'],
        raw_data['generics'],
        raw_data['medicine_info']
    )
    
    # Handle missing values
    panel = handle_missing_values(panel)
    
    # Compute pre-entry stats
    is_train = (split == 'train')
    panel = compute_pre_entry_stats(panel, is_train=is_train)
    
    # Validate panel schema
    validate_panel_schema(panel, split=split, raise_on_error=False)
    
    # Save to cache
    if use_cache:
        interim_dir.mkdir(parents=True, exist_ok=True)
        with timer(f"Save {split} panel to cache"):
            # Optimize dtypes before saving
            panel = _optimize_dtypes(panel)
            _save_cache(panel, cache_path)
            logger.info(f"Saved to {cache_path}")
    
    return panel


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes for storage efficiency.
    
    - Converts known categorical columns (country, brand_name, ther_area, main_package) to category dtype
    - Converts low-cardinality object columns to category
    - Downcasts numeric types (int64 -> int32, float64 -> float32) where safe
    - Preserves data precision for critical numeric columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    df = df.copy()
    
    # Known high-cardinality categoricals that benefit from category dtype
    known_categoricals = ['country', 'brand_name', 'ther_area', 'main_package', 'month']
    
    # Convert known categoricals
    for col in known_categoricals:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
            logger.debug(f"Converted {col} to category (known categorical)")
    
    # Convert other low-cardinality object columns to category
    for col in df.select_dtypes(include=['object']).columns:
        n_unique = df[col].nunique()
        n_total = len(df)
        if n_unique / n_total < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
            logger.debug(f"Converted {col} to category ({n_unique} unique values)")
    
    # Columns to preserve precision (don't downcast)
    preserve_precision = {'y_norm', 'volume', 'avg_vol_12m', 'mean_erosion', 'hospital_rate'}
    
    # Downcast integers (except months_postgx which could be negative)
    for col in df.select_dtypes(include=['int64']).columns:
        if col not in preserve_precision:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Downcast floats, but preserve precision for critical columns
    for col in df.select_dtypes(include=['float64']).columns:
        if col not in preserve_precision:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Log memory reduction summary
    return df


def clear_panel_cache(config: dict, split: Optional[str] = None) -> None:
    """
    Clear cached panel files.
    
    Args:
        config: Loaded data.yaml config dict
        split: If specified, only clear that split. Otherwise clear all.
    """
    if is_colab():
        interim_dir = Path(config.get('drive', {}).get('interim_dir', 'data/interim'))
    else:
        interim_dir = get_project_root() / config['paths']['interim_dir']
    
    splits = [split] if split else ['train', 'test']
    
    for s in splits:
        for ext in ['.parquet', '.pkl']:
            cache_path = interim_dir / f"panel_{s}{ext}"
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache: {cache_path}")


def verify_meta_cols_consistency(config: dict) -> bool:
    """
    Verify that META_COLS matches columns.meta_cols in config.
    
    Args:
        config: Loaded data.yaml config dict
        
    Returns:
        True if consistent, raises ValueError otherwise
    """
    config_meta_cols = set(config.get('columns', {}).get('meta_cols', []))
    code_meta_cols = set(META_COLS)
    
    if config_meta_cols != code_meta_cols:
        missing_in_code = config_meta_cols - code_meta_cols
        missing_in_config = code_meta_cols - config_meta_cols
        
        msg = "META_COLS mismatch between code and config!\n"
        if missing_in_code:
            msg += f"  In config but not code: {missing_in_code}\n"
        if missing_in_config:
            msg += f"  In code but not config: {missing_in_config}\n"
        
        raise ValueError(msg)
    
    logger.info("META_COLS consistency check passed")
    return True


# =============================================================================
# Data Leakage Audit
# =============================================================================

# Forbidden columns that should NEVER be used as features (target leakage)
LEAKAGE_COLUMNS = frozenset([
    'bucket', 'y_norm', 'volume', 'mean_erosion'
])

# ID columns that should NEVER be used as features (identifiers, not features)
ID_COLUMNS = frozenset([
    'country', 'brand_name'
])


def audit_data_leakage(
    feature_df: pd.DataFrame,
    scenario: int = 1,
    mode: str = "train",
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Systematically audit a feature DataFrame for data leakage.
    
    This function confirms that no feature uses:
    1. Future volume values beyond the scenario cutoff
    2. bucket, mean_erosion, or any other target-derived statistic
    3. Test-set statistics (checked via naming conventions)
    
    Args:
        feature_df: DataFrame with features to audit
        scenario: 1 or 2 (determines cutoff rules)
        mode: "train" or "test"
        strict: If True, raises ValueError on leakage detection
        
    Returns:
        Tuple of (is_clean, list of violations)
        
    Raises:
        ValueError: If strict=True and leakage is detected
    """
    violations = []
    
    # Check 1: No forbidden columns (target leakage)
    present_leakage_cols = LEAKAGE_COLUMNS & set(feature_df.columns)
    if present_leakage_cols:
        violations.append(f"LEAKAGE: Target-derived columns present: {present_leakage_cols}")
    
    # Check 2: No ID columns as features (should be in meta)
    present_id_cols = ID_COLUMNS & set(feature_df.columns)
    if present_id_cols:
        violations.append(f"WARNING: ID columns present (should be in meta): {present_id_cols}")
    
    # Check 3: Scenario 1 should not have early erosion features
    if scenario == 1:
        early_erosion_patterns = [
            'avg_vol_0_', 'erosion_0_', 'trend_0_', 'drop_month_0',
            'month_0_to', 'month_3_to', 'recovery_', 'competition_response',
            'erosion_per_generic'
        ]
        for col in feature_df.columns:
            for pattern in early_erosion_patterns:
                if pattern in col.lower():
                    violations.append(f"LEAKAGE: Scenario 1 has early-erosion feature: {col}")
    
    # Check 4: Test mode should not have y_norm
    if mode == "test" and 'y_norm' in feature_df.columns:
        violations.append("LEAKAGE: Test mode should not have y_norm column")
    
    # Check 5: Look for suspicious column names
    suspicious_patterns = ['_test_', '_future_', '_target_', '_label_']
    for col in feature_df.columns:
        col_lower = col.lower()
        for pattern in suspicious_patterns:
            if pattern in col_lower:
                violations.append(f"SUSPICIOUS: Column name suggests potential leakage: {col}")
    
    # Check 6: Validate no META_COLS are used as features
    meta_cols_present = set(META_COLS) & set(feature_df.columns)
    # Filter out months_postgx as it's often needed for row identification
    meta_cols_present = meta_cols_present - {'months_postgx'}
    if meta_cols_present:
        violations.append(f"WARNING: META_COLS present in features: {meta_cols_present}")
    
    is_clean = len([v for v in violations if v.startswith('LEAKAGE')]) == 0
    
    # Log results
    if violations:
        for v in violations:
            if v.startswith('LEAKAGE'):
                logger.error(v)
            elif v.startswith('WARNING'):
                logger.warning(v)
            else:
                logger.info(v)
    else:
        logger.info(f"Data leakage audit passed for scenario {scenario}, mode {mode}")
    
    if strict and not is_clean:
        raise ValueError(f"Data leakage detected! Violations:\n" + "\n".join(violations))
    
    return is_clean, violations


def run_pre_training_leakage_check(
    X: pd.DataFrame,
    scenario: int,
    mode: str = "train"
) -> bool:
    """
    Run automated leakage check before training.
    
    This is a convenience wrapper for audit_data_leakage() that should be
    called at the start of any training or inference pipeline.
    
    Args:
        X: Feature DataFrame (without target or meta columns)
        scenario: 1 or 2
        mode: "train" or "test"
        
    Returns:
        True if check passes
        
    Raises:
        ValueError: If leakage is detected
    """
    with timer("Pre-training leakage check"):
        is_clean, violations = audit_data_leakage(X, scenario, mode, strict=True)
    return is_clean


# =============================================================================
# Date Continuity Validation
# =============================================================================

def validate_date_continuity(
    panel_df: pd.DataFrame,
    min_months: int = -12,
    max_months: int = 23
) -> Tuple[bool, pd.DataFrame]:
    """
    Validate that there are no gaps in months_postgx per series.
    
    Args:
        panel_df: Panel DataFrame with months_postgx column
        min_months: Minimum expected months_postgx (default -12)
        max_months: Maximum expected months_postgx (default 23)
        
    Returns:
        Tuple of (is_valid, issues_df)
        - is_valid: True if no gaps found
        - issues_df: DataFrame with series that have gaps, containing:
            - country, brand_name
            - expected_months: count of expected months
            - actual_months: count of actual months
            - missing_months: list of missing month values
            - gap_count: number of gaps
    """
    series_keys = ['country', 'brand_name']
    
    # Group by series and check for gaps
    issues_list = []
    
    for (country, brand), group in panel_df.groupby(series_keys):
        actual_months = set(group['months_postgx'].unique())
        min_actual = group['months_postgx'].min()
        max_actual = group['months_postgx'].max()
        
        # Expected months for this series (based on its actual range)
        expected_months = set(range(min_actual, max_actual + 1))
        missing = expected_months - actual_months
        
        if missing:
            issues_list.append({
                'country': country,
                'brand_name': brand,
                'min_month': min_actual,
                'max_month': max_actual,
                'expected_count': len(expected_months),
                'actual_count': len(actual_months),
                'missing_months': sorted(missing),
                'gap_count': len(missing)
            })
    
    issues_df = pd.DataFrame(issues_list) if issues_list else pd.DataFrame()
    is_valid = len(issues_df) == 0
    
    if is_valid:
        logger.info("Date continuity validation passed: no gaps in months_postgx")
    else:
        logger.warning(f"Date continuity issues found in {len(issues_df)} series:")
        total_gaps = issues_df['gap_count'].sum()
        logger.warning(f"  Total gaps: {total_gaps}")
        # Log worst offenders
        worst = issues_df.nlargest(5, 'gap_count')
        for _, row in worst.iterrows():
            logger.warning(
                f"  {row['country']}/{row['brand_name']}: "
                f"{row['gap_count']} gaps in months [{row['min_month']}, {row['max_month']}]"
            )
    
    return is_valid, issues_df


def get_series_month_coverage(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get coverage statistics for each series.
    
    Args:
        panel_df: Panel DataFrame
        
    Returns:
        DataFrame with columns:
        - country, brand_name
        - min_month, max_month (range of months_postgx)
        - pre_entry_months: count of months < 0
        - post_entry_months: count of months >= 0
        - total_months: total month count
        - has_full_pre_entry: True if has all months from -12 to -1
        - has_full_post_entry: True if has all months from 0 to 23
    """
    series_keys = ['country', 'brand_name']
    
    def compute_coverage(group):
        months = group['months_postgx'].unique()
        min_m, max_m = months.min(), months.max()
        
        pre_entry = set(range(-12, 0))
        post_entry = set(range(0, 24))
        actual_months = set(months)
        
        return pd.Series({
            'min_month': min_m,
            'max_month': max_m,
            'pre_entry_months': len(actual_months & pre_entry),
            'post_entry_months': len(actual_months & post_entry),
            'total_months': len(actual_months),
            'has_full_pre_entry': pre_entry.issubset(actual_months),
            'has_full_post_entry': post_entry.issubset(actual_months)
        })
    
    coverage_df = panel_df.groupby(series_keys).apply(
        compute_coverage, include_groups=False
    ).reset_index()
    
    logger.info(f"Series coverage stats:")
    logger.info(f"  Total series: {len(coverage_df)}")
    logger.info(f"  With full pre-entry (12 months): {coverage_df['has_full_pre_entry'].sum()}")
    logger.info(f"  With full post-entry (24 months): {coverage_df['has_full_post_entry'].sum()}")
    
    return coverage_df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    from .utils import setup_logging
    
    parser = argparse.ArgumentParser(
        description="Build and cache data panels and feature matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build train panel only
  python -m src.data --split train --data-config configs/data.yaml

  # Build test panel only  
  python -m src.data --split test --data-config configs/data.yaml

  # Build train Scenario 1 features
  python -m src.data --split train --scenario 1 --mode train --data-config configs/data.yaml --features-config configs/features.yaml

  # Build train Scenario 2 features
  python -m src.data --split train --scenario 2 --mode train --data-config configs/data.yaml --features-config configs/features.yaml

  # Build test features for both scenarios
  python -m src.data --split test --scenario 1 --mode test --data-config configs/data.yaml --features-config configs/features.yaml
  python -m src.data --split test --scenario 2 --mode test --data-config configs/data.yaml --features-config configs/features.yaml

  # Force rebuild (ignore cache)
  python -m src.data --split train --scenario 1 --mode train --force-rebuild --data-config configs/data.yaml --features-config configs/features.yaml

  # Clear all cache files
  python -m src.data --clear-cache --data-config configs/data.yaml

  # Validate date continuity
  python -m src.data --split train --validate-continuity --data-config configs/data.yaml
"""
    )
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Which split to build (default: both)")
    parser.add_argument("--scenario", type=int, choices=[1, 2],
                        help="Scenario for feature building (1 or 2). If not specified, only builds panel.")
    parser.add_argument("--mode", choices=["train", "test"],
                        help="Mode for feature building (train creates y_norm, test does not)")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild even if cache exists")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear existing cache files")
    parser.add_argument("--validate-continuity", action="store_true",
                        help="Validate date continuity in panel")
    parser.add_argument("--data-config", default="configs/data.yaml",
                        help="Path to data config YAML (default: configs/data.yaml)")
    parser.add_argument("--features-config", default=None,
                        help="Path to features config YAML (required if --scenario is specified)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Load config
    config = load_config(args.data_config)
    
    # Verify META_COLS consistency
    verify_meta_cols_consistency(config)
    
    # Clear cache if requested
    if args.clear_cache:
        clear_panel_cache(config)
        # Also clear feature cache
        if args.features_config:
            from .features import clear_feature_cache
            clear_feature_cache(config)
        logger.info("Cache cleared")
        exit(0)
    
    # Determine splits to process
    splits = ["train", "test"] if args.split == "both" else [args.split]
    
    # If scenario is specified, build features
    if args.scenario is not None:
        if args.mode is None:
            logger.error("--mode is required when --scenario is specified")
            exit(1)
        if args.features_config is None:
            logger.warning("--features-config not specified, using default feature configuration")
        
        # Load features config if provided
        features_config = load_config(args.features_config) if args.features_config else None
        
        # Import get_features here to avoid circular imports
        from .features import get_features, clear_feature_cache
        
        for split in splits:
            logger.info(f"\n{'='*60}")
            logger.info(f"Building {split} features for Scenario {args.scenario} (mode={args.mode})")
            logger.info('='*60)
            
            X, y, meta_df = get_features(
                split=split,
                scenario=args.scenario,
                mode=args.mode,
                data_config=config,
                features_config=features_config,
                use_cache=True,
                force_rebuild=args.force_rebuild
            )
            
            logger.info(f"\n{split} S{args.scenario} feature summary:")
            logger.info(f"  X shape: {X.shape}")
            logger.info(f"  y length: {len(y) if y is not None else 'None'}")
            logger.info(f"  Feature columns: {len(X.columns)}")
            
            # Run leakage audit
            is_clean, violations = audit_data_leakage(X, args.scenario, args.mode, strict=False)
            if not is_clean:
                logger.warning(f"Leakage audit found issues!")
    else:
        # Just build panels
        for split in splits:
            logger.info(f"\n{'='*60}")
            logger.info(f"Building {split} panel")
            logger.info('='*60)
            
            panel = get_panel(
                split=split,
                config=config,
                use_cache=True,
                force_rebuild=args.force_rebuild
            )
            
            logger.info(f"\n{split} panel summary:")
            logger.info(f"  Shape: {panel.shape}")
            logger.info(f"  Series: {get_series_count(panel):,}")
            logger.info(f"  Columns: {list(panel.columns)}")
            
            # Validate date continuity if requested
            if args.validate_continuity:
                logger.info(f"\nValidating date continuity for {split}...")
                is_valid, issues_df = validate_date_continuity(panel)
                if not is_valid:
                    logger.warning(f"Found {len(issues_df)} series with date gaps")
                    # Save issues to CSV
                    issues_path = Path(config['paths']['interim_dir']) / f"date_continuity_issues_{split}.csv"
                    issues_df.to_csv(issues_path, index=False)
                    logger.info(f"Saved issues to {issues_path}")
