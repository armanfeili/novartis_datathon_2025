"""
Data augmentation techniques for small-data optimization.

Implements training-only augmentation strategies from small_data_leaderboard_tricks_todo_copilot.md:
- 2.1: Noise-perturbed series for GBMs
- 2.2: Curve-shape augmentation via residual transfer
- 2.3: Bootstrapped residuals

CRITICAL: These augmentations are TRAINING-ONLY.
Never apply to validation OOF predictions or test inference.
"""

import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


# ==============================================================================
# 2.1 NOISE-PERTURBED SERIES FOR GBMs
# ==============================================================================

def augment_with_noise(
    df: pd.DataFrame,
    sigma: float = 0.05,
    target_col: str = 'y_norm',
    n_copies: int = 1,
    random_state: int = 42,
    mark_augmented: bool = True
) -> pd.DataFrame:
    """
    Create noise-perturbed copies of training data.
    
    For each row, creates synthetic copies with:
        y_norm_aug = y_norm * (1 + ε), where ε ~ N(0, σ²)
    
    Args:
        df: Training panel data with y_norm
        sigma: Standard deviation of noise (recommended: 0.03-0.07)
        target_col: Target column to perturb
        n_copies: Number of augmented copies per row
        random_state: For reproducibility
        mark_augmented: If True, add 'is_augmented' column
        
    Returns:
        DataFrame with original + augmented rows
        
    CRITICAL: Only apply to training folds, NEVER to validation or test.
    """
    np.random.seed(random_state)
    
    if target_col not in df.columns:
        logger.warning(f"{target_col} not found, returning original data")
        return df
    
    original_len = len(df)
    
    # Mark original data
    df = df.copy()
    if mark_augmented:
        df['is_augmented'] = 0
    
    augmented_dfs = [df]
    
    for copy_idx in range(n_copies):
        aug_df = df.copy()
        
        # Sample noise
        epsilon = np.random.normal(0, sigma, len(aug_df))
        
        # Perturb target
        aug_df[target_col] = aug_df[target_col] * (1 + epsilon)
        
        # Clip to valid range [0, 2] for normalized volume
        aug_df[target_col] = aug_df[target_col].clip(0, 2)
        
        if mark_augmented:
            aug_df['is_augmented'] = 1
        
        augmented_dfs.append(aug_df)
    
    result = pd.concat(augmented_dfs, ignore_index=True)
    
    logger.info(
        f"Noise augmentation: {original_len} -> {len(result)} rows "
        f"(sigma={sigma}, n_copies={n_copies})"
    )
    
    return result


def augment_bucket1_only(
    df: pd.DataFrame,
    sigma: float = 0.05,
    target_col: str = 'y_norm',
    bucket_col: str = 'bucket',
    n_copies: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Augment only Bucket 1 (high erosion) samples to address class imbalance.
    
    Bucket 1 brands are rare (~6.7%) but count 2× in scoring.
    This augmentation helps the model learn more about high-erosion patterns.
    
    Args:
        df: Training panel with bucket column
        sigma: Noise standard deviation
        target_col: Target column
        bucket_col: Bucket column
        n_copies: Number of augmented copies for Bucket 1
        random_state: For reproducibility
        
    Returns:
        DataFrame with original data + augmented Bucket 1 samples
    """
    np.random.seed(random_state)
    
    if bucket_col not in df.columns:
        logger.warning(f"{bucket_col} not found, returning original data")
        return df
    
    df = df.copy()
    df['is_augmented'] = 0
    
    # Separate buckets
    bucket1 = df[df[bucket_col] == 1].copy()
    bucket2 = df[df[bucket_col] == 2].copy()
    
    if len(bucket1) == 0:
        logger.warning("No Bucket 1 samples found")
        return df
    
    # Augment only Bucket 1
    augmented_b1 = []
    for copy_idx in range(n_copies):
        aug_b1 = bucket1.copy()
        epsilon = np.random.normal(0, sigma, len(aug_b1))
        aug_b1[target_col] = aug_b1[target_col] * (1 + epsilon)
        aug_b1[target_col] = aug_b1[target_col].clip(0, 2)
        aug_b1['is_augmented'] = 1
        augmented_b1.append(aug_b1)
    
    result = pd.concat([df] + augmented_b1, ignore_index=True)
    
    n_b1_orig = len(bucket1)
    n_b1_aug = len(bucket1) * n_copies
    logger.info(
        f"Bucket 1 augmentation: {n_b1_orig} -> {n_b1_orig + n_b1_aug} B1 samples "
        f"(total: {len(df)} -> {len(result)})"
    )
    
    return result


# ==============================================================================
# 2.2 CURVE-SHAPE AUGMENTATION VIA RESIDUAL TRANSFER
# ==============================================================================

def fit_simple_decay_curve(
    series_df: pd.DataFrame,
    target_col: str = 'y_norm',
    time_col: str = 'months_postgx'
) -> Tuple[float, float]:
    """
    Fit a simple exponential decay curve to a brand's erosion pattern.
    
    Model: y = a * exp(-b * t) where t = months_postgx
    In log space: log(y) = log(a) - b * t
    
    Args:
        series_df: Single brand's time series
        target_col: Target column
        time_col: Time column
        
    Returns:
        (a, b) decay parameters
    """
    df = series_df[series_df[time_col] >= 0].copy()
    
    if len(df) < 2:
        return 1.0, 0.0
    
    # Filter positive values for log transform
    df = df[df[target_col] > 0.01]
    
    if len(df) < 2:
        return 1.0, 0.0
    
    # Linear regression in log space
    X = df[time_col].values.reshape(-1, 1)
    y_log = np.log(df[target_col].values)
    
    try:
        model = LinearRegression()
        model.fit(X, y_log)
        a = np.exp(model.intercept_)
        b = -model.coef_[0]
        
        # Clamp to reasonable values
        a = np.clip(a, 0.5, 2.0)
        b = np.clip(b, 0.0, 0.5)
        
        return float(a), float(b)
    except Exception:
        return 1.0, 0.0


def compute_residuals(
    df: pd.DataFrame,
    target_col: str = 'y_norm',
    time_col: str = 'months_postgx',
    series_keys: List[str] = ['country', 'brand_name']
) -> pd.DataFrame:
    """
    Compute residuals from simple decay curves for each brand.
    
    residual = y_norm - simple_decay(t)
    
    Args:
        df: Panel data
        target_col: Target column
        time_col: Time column
        series_keys: Columns identifying unique series
        
    Returns:
        DataFrame with 'residual' and 'decay_pred' columns added
    """
    df = df.copy()
    df['decay_pred'] = np.nan
    df['residual'] = np.nan
    
    for keys, group in df.groupby(series_keys, observed=False):
        idx = group.index
        a, b = fit_simple_decay_curve(group, target_col, time_col)
        
        # Predict decay
        t = group[time_col].values
        decay_pred = a * np.exp(-b * np.maximum(t, 0))
        df.loc[idx, 'decay_pred'] = decay_pred
        df.loc[idx, 'residual'] = group[target_col].values - decay_pred
    
    return df


def find_analog_brands(
    target_brand: Dict[str, Any],
    all_brands: pd.DataFrame,
    n_analogs: int = 3,
    match_cols: List[str] = ['ther_area', 'bucket']
) -> pd.DataFrame:
    """
    Find analog brands similar to target brand for residual transfer.
    
    Similarity based on:
    - Same therapeutic area
    - Same bucket (erosion pattern)
    - Similar avg_vol (within 2x)
    
    Args:
        target_brand: Dict with brand metadata (country, brand_name, ther_area, etc.)
        all_brands: DataFrame with all brand metadata
        n_analogs: Number of analog brands to find
        match_cols: Columns that must match exactly
        
    Returns:
        DataFrame of analog brand identifiers
    """
    candidates = all_brands.copy()
    
    # Exclude target brand itself
    mask = ~(
        (candidates['country'] == target_brand.get('country')) &
        (candidates['brand_name'] == target_brand.get('brand_name'))
    )
    candidates = candidates[mask]
    
    # Match on specified columns
    for col in match_cols:
        if col in candidates.columns and col in target_brand:
            candidates = candidates[candidates[col] == target_brand[col]]
    
    # If avg_vol available, filter to similar size
    if 'avg_vol_12m' in candidates.columns and 'avg_vol_12m' in target_brand:
        target_vol = target_brand['avg_vol_12m']
        if target_vol > 0:
            candidates = candidates[
                (candidates['avg_vol_12m'] >= target_vol * 0.5) &
                (candidates['avg_vol_12m'] <= target_vol * 2.0)
            ]
    
    # Return up to n_analogs
    if len(candidates) > n_analogs:
        candidates = candidates.sample(n=n_analogs, random_state=42)
    
    return candidates[['country', 'brand_name']]


def augment_with_residual_transfer(
    df: pd.DataFrame,
    target_col: str = 'y_norm',
    time_col: str = 'months_postgx',
    series_keys: List[str] = ['country', 'brand_name'],
    lambda_weight: float = 0.5,
    sample_weight: float = 0.5,
    n_analogs: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create synthetic series by transferring residuals from analog brands.
    
    For target brand B with analog brand A:
        y_synth_B(t) = decay_B(t) + λ * residual_A(t)
    
    Args:
        df: Panel data with y_norm
        target_col: Target column
        time_col: Time column
        series_keys: Series identifiers
        lambda_weight: Weight for residual transfer (0.5-0.8 recommended)
        sample_weight: Sample weight for synthetic rows (< 1.0)
        n_analogs: Number of analog brands per target
        random_state: For reproducibility
        
    Returns:
        DataFrame with original + synthetic rows
    """
    np.random.seed(random_state)
    
    # Compute residuals for all brands
    df = compute_residuals(df, target_col, time_col, series_keys)
    
    # Get brand metadata
    meta_cols = series_keys + ['ther_area', 'bucket', 'avg_vol_12m']
    meta_cols = [c for c in meta_cols if c in df.columns]
    brand_meta = df.groupby(series_keys, observed=False).first()[
        [c for c in meta_cols if c not in series_keys]
    ].reset_index()
    
    synthetic_rows = []
    
    for _, target_row in brand_meta.iterrows():
        target_dict = target_row.to_dict()
        analogs = find_analog_brands(target_dict, brand_meta, n_analogs)
        
        if len(analogs) == 0:
            continue
        
        # Get target brand's decay curve
        target_mask = (
            (df['country'] == target_dict['country']) &
            (df['brand_name'] == target_dict['brand_name'])
        )
        target_series = df[target_mask].copy()
        
        for _, analog_row in analogs.iterrows():
            # Get analog's residuals
            analog_mask = (
                (df['country'] == analog_row['country']) &
                (df['brand_name'] == analog_row['brand_name'])
            )
            analog_series = df[analog_mask].copy()
            
            # Create synthetic series
            synth = target_series.copy()
            
            # Match residuals by months_postgx
            residual_map = dict(zip(
                analog_series[time_col],
                analog_series['residual']
            ))
            
            synth['transferred_residual'] = synth[time_col].map(residual_map).fillna(0)
            synth[target_col] = synth['decay_pred'] + lambda_weight * synth['transferred_residual']
            synth[target_col] = synth[target_col].clip(0, 2)
            synth['is_augmented'] = 1
            synth['sample_weight_aug'] = sample_weight
            
            synthetic_rows.append(synth)
    
    if synthetic_rows:
        synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
        
        # Clean up intermediate columns
        drop_cols = ['decay_pred', 'residual', 'transferred_residual']
        synthetic_df = synthetic_df.drop(columns=drop_cols, errors='ignore')
        
        # Mark original data
        df['is_augmented'] = 0
        df['sample_weight_aug'] = 1.0
        df = df.drop(columns=['decay_pred', 'residual'], errors='ignore')
        
        result = pd.concat([df, synthetic_df], ignore_index=True)
        
        logger.info(
            f"Residual transfer augmentation: {len(df)} -> {len(result)} rows "
            f"(λ={lambda_weight}, n_analogs={n_analogs})"
        )
        
        return result
    
    return df


# ==============================================================================
# 2.3 BOOTSTRAPPED RESIDUALS
# ==============================================================================

def bootstrap_residuals(
    df: pd.DataFrame,
    target_col: str = 'y_norm',
    time_col: str = 'months_postgx',
    series_keys: List[str] = ['country', 'brand_name'],
    group_by: List[str] = ['bucket', 'ther_area'],
    n_bootstrap: int = 1,
    sample_weight: float = 0.7,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create synthetic series using bootstrapped residuals from similar brands.
    
    For each brand j:
        1. Fit baseline decay curve
        2. Sample residuals from brands in same group (bucket, ther_area)
        3. Create: y_synth_j(t) = baseline_j(t) + residual_bootstrap(t)
    
    Args:
        df: Panel data
        target_col: Target column
        time_col: Time column
        series_keys: Series identifiers
        group_by: Columns for grouping residuals
        n_bootstrap: Number of bootstrap samples per brand
        sample_weight: Weight for synthetic rows
        random_state: For reproducibility
        
    Returns:
        DataFrame with original + bootstrapped synthetic rows
    """
    np.random.seed(random_state)
    
    # Compute residuals
    df = compute_residuals(df, target_col, time_col, series_keys)
    
    # Build residual pools by group
    group_cols = [c for c in group_by if c in df.columns]
    if not group_cols:
        group_cols = ['bucket'] if 'bucket' in df.columns else []
    
    if not group_cols:
        logger.warning("No grouping columns available for bootstrap")
        df = df.drop(columns=['decay_pred', 'residual'], errors='ignore')
        return df
    
    # Create residual pools indexed by (group_key, months_postgx)
    residual_pools = {}
    for group_key, group_df in df.groupby(group_cols, observed=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        
        for t in group_df[time_col].unique():
            pool_key = (group_key, t)
            residuals = group_df[group_df[time_col] == t]['residual'].dropna().values
            if len(residuals) > 0:
                residual_pools[pool_key] = residuals
    
    synthetic_rows = []
    
    for (keys), brand_df in df.groupby(series_keys, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        # Get brand's group
        brand_group = tuple(brand_df[group_cols].iloc[0].values)
        
        for bootstrap_idx in range(n_bootstrap):
            synth = brand_df.copy()
            
            # Sample residuals for each time point
            bootstrapped_residuals = []
            for t in synth[time_col]:
                pool_key = (brand_group, t)
                if pool_key in residual_pools:
                    pool = residual_pools[pool_key]
                    sampled = np.random.choice(pool)
                else:
                    sampled = 0.0
                bootstrapped_residuals.append(sampled)
            
            synth['bootstrapped_residual'] = bootstrapped_residuals
            synth[target_col] = synth['decay_pred'] + synth['bootstrapped_residual']
            synth[target_col] = synth[target_col].clip(0, 2)
            synth['is_augmented'] = 1
            synth['sample_weight_aug'] = sample_weight
            
            synthetic_rows.append(synth)
    
    if synthetic_rows:
        synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
        
        # Clean up
        drop_cols = ['decay_pred', 'residual', 'bootstrapped_residual']
        synthetic_df = synthetic_df.drop(columns=drop_cols, errors='ignore')
        
        df['is_augmented'] = 0
        df['sample_weight_aug'] = 1.0
        df = df.drop(columns=['decay_pred', 'residual'], errors='ignore')
        
        result = pd.concat([df, synthetic_df], ignore_index=True)
        
        logger.info(
            f"Bootstrap residuals augmentation: {len(df)} -> {len(result)} rows "
            f"(n_bootstrap={n_bootstrap})"
        )
        
        return result
    
    df = df.drop(columns=['decay_pred', 'residual'], errors='ignore')
    return df


# ==============================================================================
# AUGMENTATION PIPELINE
# ==============================================================================

def apply_training_augmentation(
    train_df: pd.DataFrame,
    config: Optional[Dict] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply configured augmentation strategies to training data.
    
    CRITICAL: Only call this on training folds.
    Never apply to validation or test data.
    
    Args:
        train_df: Training panel data
        config: Augmentation configuration dict with keys:
            - noise: {enabled, sigma, n_copies}
            - bucket1_noise: {enabled, sigma, n_copies}
            - residual_transfer: {enabled, lambda_weight, n_analogs}
            - bootstrap: {enabled, n_bootstrap, group_by}
        random_state: For reproducibility
        
    Returns:
        (augmented_df, sample_weights) where sample_weights includes
        augmentation weights (lower for synthetic rows)
    """
    if config is None:
        config = {}
    
    df = train_df.copy()
    
    # Track if any augmentation was applied
    augmentation_applied = False
    
    # 2.1a: General noise augmentation
    noise_cfg = config.get('noise', {})
    if noise_cfg.get('enabled', False):
        df = augment_with_noise(
            df,
            sigma=noise_cfg.get('sigma', 0.05),
            n_copies=noise_cfg.get('n_copies', 1),
            random_state=random_state
        )
        augmentation_applied = True
    
    # 2.1b: Bucket 1 focused noise augmentation
    b1_noise_cfg = config.get('bucket1_noise', {})
    if b1_noise_cfg.get('enabled', False):
        df = augment_bucket1_only(
            df,
            sigma=b1_noise_cfg.get('sigma', 0.05),
            n_copies=b1_noise_cfg.get('n_copies', 2),
            random_state=random_state
        )
        augmentation_applied = True
    
    # 2.2: Residual transfer augmentation
    residual_cfg = config.get('residual_transfer', {})
    if residual_cfg.get('enabled', False):
        df = augment_with_residual_transfer(
            df,
            lambda_weight=residual_cfg.get('lambda_weight', 0.5),
            sample_weight=residual_cfg.get('sample_weight', 0.5),
            n_analogs=residual_cfg.get('n_analogs', 2),
            random_state=random_state
        )
        augmentation_applied = True
    
    # 2.3: Bootstrapped residuals
    bootstrap_cfg = config.get('bootstrap', {})
    if bootstrap_cfg.get('enabled', False):
        df = bootstrap_residuals(
            df,
            n_bootstrap=bootstrap_cfg.get('n_bootstrap', 1),
            group_by=bootstrap_cfg.get('group_by', ['bucket', 'ther_area']),
            sample_weight=bootstrap_cfg.get('sample_weight', 0.7),
            random_state=random_state
        )
        augmentation_applied = True
    
    # Compute sample weights including augmentation weights
    if augmentation_applied and 'sample_weight_aug' in df.columns:
        sample_weights = df['sample_weight_aug'].values
        df = df.drop(columns=['sample_weight_aug'], errors='ignore')
    else:
        sample_weights = np.ones(len(df))
    
    return df, sample_weights
