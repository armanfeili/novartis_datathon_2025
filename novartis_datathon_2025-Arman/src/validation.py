"""
Validation and splitting module for Novartis Datathon 2025.

Implements series-level stratified validation mimicking true scenario constraints.
CRITICAL: Never mix months of the same series across train/val.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import timer
from .features import _normalize_scenario
from .evaluate import make_metric_record, save_metric_records

logger = logging.getLogger(__name__)


def create_validation_split(
    panel_df: pd.DataFrame,
    val_fraction: float = 0.2,
    stratify_by: Union[str, List[str]] = 'bucket',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split at SERIES level (not row level).
    
    CRITICAL: Never mix months of the same series across train/val.
    
    Args:
        panel_df: Full training panel
        val_fraction: Fraction of series for validation
        stratify_by: Column(s) to stratify by. Can be:
            - Single string: 'bucket' or 'ther_area'
            - List of strings: ['bucket', 'ther_area'] for multi-column stratification
        random_state: For reproducibility
    
    Returns:
        (train_df, val_df) - both contain full series
    """
    with timer("Create validation split"):
        series_keys = ['country', 'brand_name']
        
        # Handle stratify_by as string or list
        if isinstance(stratify_by, str):
            stratify_cols = [stratify_by]
        else:
            stratify_cols = list(stratify_by)
        
        # Get unique series with their stratification labels
        required_cols = series_keys + stratify_cols
        available_cols = [c for c in required_cols if c in panel_df.columns]
        series_info = panel_df[available_cols].drop_duplicates()
        
        # Check for series with multiple values for stratification columns (shouldn't happen)
        for col in stratify_cols:
            if col in series_info.columns:
                counts = series_info.groupby(series_keys, observed=False)[col].nunique()
                if (counts > 1).any():
                    logger.warning(f"Some series have multiple {col} values!")
        
        # Create combined stratification column for multi-column stratification
        stratify_available = [c for c in stratify_cols if c in series_info.columns]
        
        if len(stratify_available) > 0:
            if len(stratify_available) == 1:
                stratify_values = series_info[stratify_available[0]].astype(str)
            else:
                # Multi-column stratification: combine columns
                series_info['_stratify_combined'] = series_info[stratify_available].astype(str).agg('_'.join, axis=1)
                stratify_values = series_info['_stratify_combined']
                
                # Log combined stratification distribution
                combined_dist = stratify_values.value_counts()
                logger.info(f"Combined stratification groups ({len(combined_dist)} unique):")
                for group, count in combined_dist.head(10).items():
                    logger.info(f"  {group}: {count} series")
            
            # Handle rare classes by grouping them
            value_counts = stratify_values.value_counts()
            min_class_size = max(2, int(len(series_info) * val_fraction * 0.5))
            
            rare_classes = value_counts[value_counts < min_class_size].index
            if len(rare_classes) > 0:
                logger.warning(
                    f"Grouping {len(rare_classes)} rare stratification classes "
                    f"(< {min_class_size} series) into 'OTHER'"
                )
                stratify_values = stratify_values.copy()
                stratify_values[stratify_values.isin(rare_classes)] = 'OTHER'
            
            train_series, val_series = train_test_split(
                series_info,
                test_size=val_fraction,
                stratify=stratify_values,
                random_state=random_state
            )
        else:
            logger.warning(f"No stratification columns available, using random split")
            train_series, val_series = train_test_split(
                series_info,
                test_size=val_fraction,
                random_state=random_state
            )
        
        # Filter panel to train/val series
        train_df = panel_df.merge(
            train_series[series_keys],
            on=series_keys,
            how='inner'
        )
        val_df = panel_df.merge(
            val_series[series_keys],
            on=series_keys,
            how='inner'
        )
        
        # Log split statistics
        n_train_series = train_series.shape[0]
        n_val_series = val_series.shape[0]
        
        logger.info(f"Train series: {n_train_series}, Val series: {n_val_series}")
        logger.info(f"Train rows: {len(train_df):,}, Val rows: {len(val_df):,}")
        
        # Log stratification distribution for each column
        for col in stratify_available:
            train_dist = train_series[col].value_counts(normalize=True)
            val_dist = val_series[col].value_counts(normalize=True)
            logger.info(f"Train {col} distribution:\n{train_dist.to_string()}")
            logger.info(f"Val {col} distribution:\n{val_dist.to_string()}")
    
    return train_df, val_df


def create_temporal_cv_split(
    panel_df: pd.DataFrame,
    n_folds: int = 5,
    min_train_months: int = 12,
    gap_months: int = 0,
    stratify_by: Union[str, List[str], None] = 'bucket'
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time-based cross-validation splits.
    
    This implements a temporal CV where:
    - Each fold uses progressively more historical data for training
    - Validation is always on later time periods
    - Series are kept intact (no leakage across time for same series)
    
    Strategy: For each fold, we use a different temporal cutoff to simulate
    having data up to month T and validating on months > T.
    
    Args:
        panel_df: Full training panel with months_postgx column
        n_folds: Number of temporal folds
        min_train_months: Minimum months_postgx value to include in first training fold
        gap_months: Gap between train and val (purged CV)
        stratify_by: Column(s) to stratify series selection within temporal windows
        
    Returns:
        List of (train_df, val_df) tuples
    """
    series_keys = ['country', 'brand_name']
    
    # Get the range of months_postgx
    min_month = panel_df['months_postgx'].min()
    max_month = panel_df['months_postgx'].max()
    
    logger.info(f"Temporal CV: months_postgx range [{min_month}, {max_month}]")
    
    # Calculate fold boundaries (cutoff months)
    # We want validation windows that cover the target months (0-23 for S1, 6-23 for S2)
    # Each fold has a different cutoff
    post_entry_months = panel_df[panel_df['months_postgx'] >= 0]['months_postgx'].unique()
    post_entry_months = sorted(post_entry_months)
    
    if len(post_entry_months) < n_folds:
        logger.warning(f"Not enough post-entry months for {n_folds} folds, reducing to {len(post_entry_months)}")
        n_folds = max(1, len(post_entry_months))
    
    # Create cutoffs: evenly spaced through post-entry period
    cutoff_indices = np.linspace(0, len(post_entry_months) - 1, n_folds + 1)[1:-1].astype(int)
    cutoffs = [post_entry_months[i] for i in cutoff_indices]
    
    # Ensure we have reasonable cutoffs (not too early, not too late)
    # Minimum cutoff should allow some validation data
    min_cutoff = min_train_months if min_train_months >= 0 else 0
    cutoffs = [max(c, min_cutoff) for c in cutoffs]
    
    # Add final fold that uses all data
    if len(cutoffs) < n_folds - 1:
        cutoffs.append(max_month - 1)
    
    logger.info(f"Temporal CV cutoffs: {cutoffs}")
    
    folds = []
    for i, cutoff in enumerate(cutoffs):
        # Training: all months up to cutoff - gap
        train_cutoff = cutoff - gap_months
        train_mask = panel_df['months_postgx'] <= train_cutoff
        train_df = panel_df[train_mask].copy()
        
        # Validation: months after cutoff
        val_mask = panel_df['months_postgx'] > cutoff
        val_df = panel_df[val_mask].copy()
        
        # Ensure series in validation also have training data
        train_series = set(zip(train_df['country'], train_df['brand_name']))
        val_series = set(zip(val_df['country'], val_df['brand_name']))
        common_series = train_series & val_series
        
        if len(common_series) < len(val_series):
            logger.warning(
                f"Fold {i+1}: {len(val_series) - len(common_series)} val series "
                f"have no training data, excluding them"
            )
            val_df = val_df[
                val_df.apply(lambda r: (r['country'], r['brand_name']) in common_series, axis=1)
            ]
        
        logger.info(
            f"Fold {i+1}: train months <= {train_cutoff} ({len(train_df):,} rows), "
            f"val months > {cutoff} ({len(val_df):,} rows)"
        )
        
        folds.append((train_df, val_df))
    
    return folds


def create_holdout_set(
    panel_df: pd.DataFrame,
    holdout_fraction: float = 0.1,
    stratify_by: Union[str, List[str]] = 'bucket',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a holdout set for final validation before submission.
    
    This holdout should be set aside at the beginning and only used once
    for final model selection before submission.
    
    Args:
        panel_df: Full training panel
        holdout_fraction: Fraction of series for holdout (default 10%)
        stratify_by: Column(s) to stratify by
        random_state: For reproducibility
        
    Returns:
        (main_df, holdout_df) - both contain full series
    """
    # Use create_validation_split with holdout fraction
    main_df, holdout_df = create_validation_split(
        panel_df,
        val_fraction=holdout_fraction,
        stratify_by=stratify_by,
        random_state=random_state
    )
    
    logger.info(f"Holdout set created: {len(holdout_df[['country', 'brand_name']].drop_duplicates())} series")
    
    return main_df, holdout_df


def simulate_scenario(
    val_df: pd.DataFrame,
    scenario
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare validation data mimicking true scenario constraints.
    
    Note: In the main pipeline, scenario constraints are typically enforced 
    during feature engineering (make_features) and row selection (select_training_rows).
    This function is useful for explicit history/horizon splitting experiments.
    
    Scenario 1:
        - features_df: months_postgx < 0 only
        - targets_df: months_postgx in [0, 23]
    
    Scenario 2:
        - features_df: months_postgx < 6
        - targets_df: months_postgx in [6, 23]
    
    Args:
        val_df: Validation panel data
        scenario: 1, 2, "scenario1", or "scenario2"
        
    Returns:
        (features_df, targets_df)
    """
    scenario = _normalize_scenario(scenario)
    
    if scenario == 1:
        cutoff = 0
        target_start = 0
        target_end = 23
    elif scenario == 2:
        cutoff = 6
        target_start = 6
        target_end = 23
    else:
        raise ValueError(f"Invalid scenario: {scenario}")
    
    # History for feature extraction
    features_df = val_df[val_df['months_postgx'] < cutoff].copy()
    
    # Target months for prediction
    targets_df = val_df[
        (val_df['months_postgx'] >= target_start) & 
        (val_df['months_postgx'] <= target_end)
    ].copy()
    
    logger.info(f"Scenario {scenario}: {len(features_df):,} feature rows, {len(targets_df):,} target rows")
    
    return features_df, targets_df


def adversarial_validation(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_folds: int = 5,
    random_state: int = 42,
    run_id: Optional[str] = None,
    metrics_dir: Optional[Path] = None
) -> Dict:
    """
    Detect train/test distribution shift.
    
    Train classifier to distinguish train vs test rows.
    
    Args:
        train_features: Training feature matrix
        test_features: Test feature matrix
        feature_cols: Columns to use (if None, use all numeric)
        n_folds: Number of CV folds
        random_state: For reproducibility
        run_id: Optional run ID for metrics logging
        metrics_dir: Optional directory to save unified metric records
        
    Returns:
        {
            'mean_auc': float,  # ~0.5 = good, >0.7 = shift detected
            'auc_scores': list,
            'top_shift_features': DataFrame with feature importances
        }
    
    If AUC > 0.7:
        - Inspect top features driving shift
        - Consider dropping or simplifying those features
        - Increase regularization
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.warning("sklearn not available for adversarial validation")
        return {'mean_auc': np.nan, 'auc_scores': [], 'top_shift_features': pd.DataFrame()}
    
    with timer("Adversarial validation"):
        # Prepare features
        if feature_cols is None:
            # Use all numeric columns
            numeric_cols = train_features.select_dtypes(include=[np.number]).columns
            feature_cols = [c for c in numeric_cols if c not in 
                          ['country', 'brand_name', 'months_postgx', 'bucket', 'y_norm', 'volume']]
        
        # Filter to common columns
        common_cols = list(set(feature_cols) & set(train_features.columns) & set(test_features.columns))
        
        if len(common_cols) == 0:
            logger.warning("No common feature columns for adversarial validation")
            return {'mean_auc': np.nan, 'auc_scores': [], 'top_shift_features': pd.DataFrame()}
        
        X_train = train_features[common_cols].fillna(-999)
        X_test = test_features[common_cols].fillna(-999)
        
        # Create labels
        y_train = np.zeros(len(X_train))
        y_test = np.ones(len(X_test))
        
        # Combine
        X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y = np.concatenate([y_train, y_test])
        
        # Train classifier with CV
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        auc_scores = cross_val_score(clf, X, y, cv=n_folds, scoring='roc_auc')
        
        # Fit on full data for feature importances
        clf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': common_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mean_auc = auc_scores.mean()
        auc_std = auc_scores.std()
        
        logger.info(f"Adversarial validation AUC: {mean_auc:.4f} (std: {auc_std:.4f})")
        
        if mean_auc > 0.7:
            logger.warning(f"High AUC ({mean_auc:.3f}) suggests train/test distribution shift!")
            logger.warning(f"Top shift features:\n{feature_importance.head(10).to_string()}")
        elif mean_auc > 0.6:
            logger.info(f"Moderate AUC ({mean_auc:.3f}), some distribution difference detected")
        else:
            logger.info(f"Low AUC ({mean_auc:.3f}), train/test distributions appear similar")
        
        # Save unified metric records if metrics_dir is provided
        if metrics_dir is not None:
            metrics_dir = Path(metrics_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = metrics_dir / 'metrics.csv'
            
            records = [
                make_metric_record(
                    phase='adversarial_val', split='train_vs_test', scenario=None,
                    model_name='random_forest', metric_name='auc_mean',
                    value=mean_auc, run_id=run_id, step='final'
                ),
                make_metric_record(
                    phase='adversarial_val', split='train_vs_test', scenario=None,
                    model_name='random_forest', metric_name='auc_std',
                    value=auc_std, run_id=run_id, step='final'
                ),
            ]
            save_metric_records(records, metrics_path, append=True)
            logger.debug(f"Saved {len(records)} adversarial validation metric records to {metrics_path}")
        
        return {
            'mean_auc': mean_auc,
            'auc_scores': auc_scores.tolist(),
            'top_shift_features': feature_importance
        }


# =============================================================================
# STANDARDIZED CROSS-VALIDATION (Use this for all models!)
# =============================================================================

def create_stratified_group_kfold(
    panel_df: pd.DataFrame,
    n_folds: int = 5,
    stratify_by: str = 'bucket',
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    STANDARDIZED cross-validation for all models.
    
    This is the SINGLE SOURCE OF TRUTH for CV splits. All models (XGBoost,
    LightGBM, CatBoost, Neural Networks, etc.) should use this function.
    
    Key guarantees:
    1. All months of a given brand go to the SAME fold (no leakage)
    2. Buckets are reasonably balanced across folds (stratified)
    3. Reproducible with the same random_state
    
    Args:
        panel_df: Full panel data with columns:
            - 'country', 'brand_name': Series identifiers
            - 'bucket': Stratification column (1 or 2)
            - 'months_postgx': Time column
        n_folds: Number of CV folds (default: 5)
        stratify_by: Column to stratify by (default: 'bucket')
        random_state: For reproducibility (default: 42)
        
    Returns:
        List of tuples: (train_indices, val_indices, fold_info_df)
        - train_indices: numpy array of row indices for training
        - val_indices: numpy array of row indices for validation
        - fold_info_df: DataFrame with fold statistics
        
    Example:
        >>> folds = create_stratified_group_kfold(panel_df, n_folds=5)
        >>> for train_idx, val_idx, info in folds:
        ...     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        ...     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        ...     model.fit(X_train, y_train)
        ...     predictions = model.predict(X_val)
    """
    from sklearn.model_selection import StratifiedKFold
    
    series_keys = ['country', 'brand_name']
    
    # Get unique series with their bucket
    if stratify_by not in panel_df.columns:
        raise ValueError(f"Stratification column '{stratify_by}' not found in panel_df")
    
    series_info = panel_df[series_keys + [stratify_by]].drop_duplicates()
    
    # Validate: each series should have exactly one bucket value
    bucket_counts = series_info.groupby(series_keys, observed=False)[stratify_by].nunique()
    if (bucket_counts > 1).any():
        bad_series = bucket_counts[bucket_counts > 1].head()
        raise ValueError(f"Series have inconsistent {stratify_by} values: {bad_series.to_dict()}")
    
    # Create series-level index mapping
    series_info = series_info.reset_index(drop=True)
    series_to_idx = {
        (row['country'], row['brand_name']): idx 
        for idx, row in series_info.iterrows()
    }
    
    # Map each row in panel_df to its series index
    panel_df = panel_df.copy()
    panel_df['_series_idx'] = panel_df.apply(
        lambda row: series_to_idx.get((row['country'], row['brand_name']), -1), 
        axis=1
    )
    
    # Stratified K-Fold on series
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_series_idx, val_series_idx) in enumerate(
        skf.split(series_info, series_info[stratify_by])
    ):
        # Get row indices for this fold
        train_mask = panel_df['_series_idx'].isin(train_series_idx)
        val_mask = panel_df['_series_idx'].isin(val_series_idx)
        
        train_indices = panel_df.index[train_mask].to_numpy()
        val_indices = panel_df.index[val_mask].to_numpy()
        
        # Compute fold statistics
        train_series = series_info.iloc[train_series_idx]
        val_series = series_info.iloc[val_series_idx]
        
        fold_info = pd.DataFrame({
            'fold': [fold_idx],
            'n_train_series': [len(train_series)],
            'n_val_series': [len(val_series)],
            'n_train_rows': [len(train_indices)],
            'n_val_rows': [len(val_indices)],
            'train_bucket1_pct': [(train_series[stratify_by] == 1).mean()],
            'val_bucket1_pct': [(val_series[stratify_by] == 1).mean()],
            'train_bucket2_pct': [(train_series[stratify_by] == 2).mean()],
            'val_bucket2_pct': [(val_series[stratify_by] == 2).mean()],
        })
        
        folds.append((train_indices, val_indices, fold_info))
        
        logger.debug(
            f"Fold {fold_idx}: train={len(train_series)} series ({len(train_indices)} rows), "
            f"val={len(val_series)} series ({len(val_indices)} rows), "
            f"bucket1: train={fold_info['train_bucket1_pct'].iloc[0]:.1%}, "
            f"val={fold_info['val_bucket1_pct'].iloc[0]:.1%}"
        )
    
    # Clean up
    panel_df.drop('_series_idx', axis=1, inplace=True, errors='ignore')
    
    return folds


def validate_cv_splits(
    panel_df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    max_bucket_imbalance: float = 0.15
) -> Dict[str, any]:
    """
    Validate that CV splits meet requirements.
    
    Checks:
    1. No series overlap between train/val
    2. All months of each series in same split
    3. Bucket distribution is balanced (within threshold)
    
    Args:
        panel_df: Original panel data
        folds: Output from create_stratified_group_kfold
        max_bucket_imbalance: Max allowed bucket % difference between folds
        
    Returns:
        Dict with validation results
    """
    series_keys = ['country', 'brand_name']
    issues = []
    
    for fold_idx, (train_idx, val_idx, fold_info) in enumerate(folds):
        train_df = panel_df.iloc[train_idx]
        val_df = panel_df.iloc[val_idx]
        
        # Check 1: No series overlap
        train_series = set(train_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
        val_series = set(val_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
        
        overlap = train_series & val_series
        if overlap:
            issues.append(f"Fold {fold_idx}: {len(overlap)} series in both train and val!")
        
        # Check 2: All months of each series in same split
        for country, brand in train_series:
            months_in_train = train_df[
                (train_df['country'] == country) & (train_df['brand_name'] == brand)
            ]['months_postgx'].nunique()
            
            months_in_val = val_df[
                (val_df['country'] == country) & (val_df['brand_name'] == brand)
            ]['months_postgx'].nunique()
            
            if months_in_val > 0:
                issues.append(
                    f"Fold {fold_idx}: Series ({country}, {brand}) has months in both splits!"
                )
        
        # Check 3: Bucket balance
        bucket1_diff = abs(
            fold_info['train_bucket1_pct'].iloc[0] - fold_info['val_bucket1_pct'].iloc[0]
        )
        if bucket1_diff > max_bucket_imbalance:
            issues.append(
                f"Fold {fold_idx}: Bucket 1 imbalance={bucket1_diff:.1%} > {max_bucket_imbalance:.1%}"
            )
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'n_folds': len(folds)
    }


def get_fold_series(
    panel_df: pd.DataFrame,
    n_folds: int = 5,
    stratify_by: str = 'bucket',
    random_state: int = 42,
    save_indices: bool = False,
    output_path: Optional[str] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate K-fold splits at series level.
    
    Args:
        panel_df: Full panel data
        n_folds: Number of folds
        stratify_by: Column to stratify by
        random_state: For reproducibility
        save_indices: Whether to save fold indices for reproducibility
        output_path: Path to save fold indices (JSON file)
        
    Returns:
        List of (train_df, val_df) tuples
    """
    from sklearn.model_selection import StratifiedKFold
    import json
    
    series_keys = ['country', 'brand_name']
    series_info = panel_df[series_keys + [stratify_by]].drop_duplicates()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    fold_indices = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(series_info, series_info[stratify_by])):
        train_series = series_info.iloc[train_idx][series_keys]
        val_series = series_info.iloc[val_idx][series_keys]
        
        train_df = panel_df.merge(train_series, on=series_keys, how='inner')
        val_df = panel_df.merge(val_series, on=series_keys, how='inner')
        
        folds.append((train_df, val_df))
        
        # Store indices for reproducibility
        if save_indices:
            fold_indices.append({
                'fold': fold_idx,
                'train_series': [tuple(s) for s in train_series.values.tolist()],
                'val_series': [tuple(s) for s in val_series.values.tolist()]
            })
    
    # Save fold indices if requested
    if save_indices and output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'n_folds': n_folds,
                'stratify_by': stratify_by,
                'random_state': random_state,
                'folds': fold_indices
            }, f, indent=2)
        logger.info(f"Fold indices saved to {output_path}")
    
    return folds


def get_grouped_kfold_series(
    panel_df: pd.DataFrame,
    n_folds: int = 5,
    group_by: str = 'ther_area',
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate grouped K-fold splits where groups are kept together.
    
    This ensures all series from the same group (e.g., therapeutic area)
    are in the same fold. Useful when you want to test generalization
    across groups rather than within.
    
    Args:
        panel_df: Full panel data
        n_folds: Number of folds
        group_by: Column to group by (e.g., 'ther_area', 'country')
        random_state: For reproducibility
        
    Returns:
        List of (train_df, val_df) tuples
    """
    from sklearn.model_selection import GroupKFold
    
    series_keys = ['country', 'brand_name']
    series_info = panel_df[series_keys + [group_by]].drop_duplicates()
    
    # GroupKFold doesn't shuffle, so we need to shuffle groups first
    np.random.seed(random_state)
    unique_groups = series_info[group_by].unique()
    shuffled_groups = np.random.permutation(unique_groups)
    group_map = {g: i for i, g in enumerate(shuffled_groups)}
    series_info['_group_idx'] = series_info[group_by].map(group_map)
    
    gkf = GroupKFold(n_splits=min(n_folds, len(unique_groups)))
    
    folds = []
    for train_idx, val_idx in gkf.split(series_info, groups=series_info['_group_idx']):
        train_series = series_info.iloc[train_idx][series_keys]
        val_series = series_info.iloc[val_idx][series_keys]
        
        train_df = panel_df.merge(train_series, on=series_keys, how='inner')
        val_df = panel_df.merge(val_series, on=series_keys, how='inner')
        
        folds.append((train_df, val_df))
        
        # Log group distribution
        train_groups = series_info.iloc[train_idx][group_by].unique()
        val_groups = series_info.iloc[val_idx][group_by].unique()
        logger.debug(f"Fold: train groups={len(train_groups)}, val groups={len(val_groups)}")
    
    return folds


def create_purged_cv_split(
    panel_df: pd.DataFrame,
    n_folds: int = 5,
    gap_months: int = 3,
    min_train_months: int = -6,
    stratify_by: Optional[str] = 'bucket',
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create purged cross-validation splits with a time gap between train and val.
    
    This helps prevent data leakage from temporal proximity. The gap ensures
    that recent training data doesn't overlap with validation data.
    
    Args:
        panel_df: Full panel data
        n_folds: Number of folds
        gap_months: Number of months gap between train and val
        min_train_months: Minimum months_postgx to include in training
        stratify_by: Column to stratify by (if any)
        random_state: For reproducibility
        
    Returns:
        List of (train_df, val_df) tuples
    """
    series_keys = ['country', 'brand_name']
    
    # First, create series-level folds
    if stratify_by and stratify_by in panel_df.columns:
        series_info = panel_df[series_keys + [stratify_by]].drop_duplicates()
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = skf.split(series_info, series_info[stratify_by])
    else:
        series_info = panel_df[series_keys].drop_duplicates()
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = kf.split(series_info)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        train_series = series_info.iloc[train_idx][series_keys]
        val_series = series_info.iloc[val_idx][series_keys]
        
        train_panel = panel_df.merge(train_series, on=series_keys, how='inner')
        val_panel = panel_df.merge(val_series, on=series_keys, how='inner')
        
        # Apply purging: use only months up to (val_min_month - gap_months) for training
        val_min_month = val_panel['months_postgx'].min()
        train_cutoff = val_min_month - gap_months
        
        # Filter training data
        train_df = train_panel[
            (train_panel['months_postgx'] >= min_train_months) &
            (train_panel['months_postgx'] <= train_cutoff)
        ].copy()
        
        # Validation data stays as is
        val_df = val_panel.copy()
        
        if len(train_df) > 0 and len(val_df) > 0:
            folds.append((train_df, val_df))
            logger.info(
                f"Fold {fold_idx + 1}: train months [{min_train_months}, {train_cutoff}], "
                f"val starts at {val_min_month}, gap={gap_months}"
            )
        else:
            logger.warning(f"Fold {fold_idx + 1}: skipped (empty train or val)")
    
    return folds


def create_nested_cv(
    panel_df: pd.DataFrame,
    outer_folds: int = 5,
    inner_folds: int = 3,
    stratify_by: str = 'bucket',
    random_state: int = 42
) -> List[Dict]:
    """
    Create nested cross-validation for unbiased model selection.
    
    The outer loop is used for final model evaluation.
    The inner loop is used for hyperparameter tuning.
    
    Args:
        panel_df: Full panel data
        outer_folds: Number of outer CV folds
        inner_folds: Number of inner CV folds
        stratify_by: Column to stratify by
        random_state: For reproducibility
        
    Returns:
        List of dicts with keys:
            - 'outer_train': DataFrame for outer training (includes inner CV)
            - 'outer_val': DataFrame for outer validation (held out)
            - 'inner_folds': List of (inner_train_df, inner_val_df) tuples
    """
    series_keys = ['country', 'brand_name']
    series_info = panel_df[series_keys + [stratify_by]].drop_duplicates()
    
    from sklearn.model_selection import StratifiedKFold
    
    outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    
    nested_folds = []
    for outer_idx, (outer_train_idx, outer_val_idx) in enumerate(outer_skf.split(series_info, series_info[stratify_by])):
        outer_train_series = series_info.iloc[outer_train_idx][series_keys]
        outer_val_series = series_info.iloc[outer_val_idx][series_keys]
        
        outer_train_df = panel_df.merge(outer_train_series, on=series_keys, how='inner')
        outer_val_df = panel_df.merge(outer_val_series, on=series_keys, how='inner')
        
        # Create inner folds within outer training data
        inner_series_info = outer_train_df[series_keys + [stratify_by]].drop_duplicates()
        inner_skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state + outer_idx)
        
        inner_folds_list = []
        for inner_train_idx, inner_val_idx in inner_skf.split(inner_series_info, inner_series_info[stratify_by]):
            inner_train_series = inner_series_info.iloc[inner_train_idx][series_keys]
            inner_val_series = inner_series_info.iloc[inner_val_idx][series_keys]
            
            inner_train_df = outer_train_df.merge(inner_train_series, on=series_keys, how='inner')
            inner_val_df = outer_train_df.merge(inner_val_series, on=series_keys, how='inner')
            
            inner_folds_list.append((inner_train_df, inner_val_df))
        
        nested_folds.append({
            'outer_train': outer_train_df,
            'outer_val': outer_val_df,
            'inner_folds': inner_folds_list
        })
        
        logger.info(
            f"Outer fold {outer_idx + 1}: "
            f"train={len(outer_train_df[series_keys].drop_duplicates())} series, "
            f"val={len(outer_val_df[series_keys].drop_duplicates())} series, "
            f"inner folds={len(inner_folds_list)}"
        )
    
    return nested_folds


def validate_cv_respects_scenario_constraints(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scenario: int
) -> Tuple[bool, List[str]]:
    """
    Verify that CV split respects scenario constraints.
    
    Checks:
    1. No post-forecast-window information leaks into feature computation
    2. Respects the same history/horizon separation as competition scenarios
    3. Series integrity is maintained (no mixing months across splits)
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        scenario: 1 or 2
        
    Returns:
        (is_valid, list of violation messages)
    """
    from .features import _normalize_scenario
    scenario = _normalize_scenario(scenario)
    
    violations = []
    series_keys = ['country', 'brand_name']
    
    # Check 1: No series overlap
    train_series = set(train_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
    val_series = set(val_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
    overlap = train_series & val_series
    if overlap:
        violations.append(f"Series overlap between train and val: {len(overlap)} series")
    
    # Check 2: Scenario-specific cutoff validation
    if scenario == 1:
        # For S1, validation should only have months_postgx >= 0
        if 'months_postgx' in val_df.columns:
            val_min = val_df['months_postgx'].min()
            if val_min < 0:
                violations.append(f"S1 validation has pre-entry data (min month={val_min})")
    elif scenario == 2:
        # For S2, validation should only have months_postgx >= 6
        if 'months_postgx' in val_df.columns:
            val_min = val_df['months_postgx'].min()
            if val_min < 6:
                violations.append(f"S2 validation has months < 6 (min month={val_min})")
    
    # Check 3: Each series has all its months in one split
    if 'months_postgx' in train_df.columns:
        for country, brand in train_series:
            train_months = set(train_df[
                (train_df['country'] == country) & (train_df['brand_name'] == brand)
            ]['months_postgx'])
            val_months = set(val_df[
                (val_df['country'] == country) & (val_df['brand_name'] == brand)
            ]['months_postgx']) if (country, brand) in val_series else set()
            
            if train_months & val_months:
                violations.append(
                    f"Series ({country}, {brand}) has months in both splits: "
                    f"train={sorted(train_months)[:5]}..., val={sorted(val_months)[:5]}..."
                )
                break  # Just report first violation
    
    is_valid = len(violations) == 0
    if not is_valid:
        for v in violations:
            logger.warning(f"CV constraint violation: {v}")
    else:
        logger.info(f"CV split validated for scenario {scenario}")
    
    return is_valid, violations


def aggregate_cv_scores(
    cv_scores: List[Dict[str, float]],
    metric_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate CV scores with confidence intervals.
    
    Args:
        cv_scores: List of dicts, each containing metric scores for one fold
        metric_names: List of metric names to aggregate (if None, use all keys)
        
    Returns:
        Dict mapping metric names to {mean, std, ci_lower, ci_upper, min, max, n_folds}
    """
    from scipy import stats
    
    if not cv_scores:
        return {}
    
    if metric_names is None:
        metric_names = list(cv_scores[0].keys())
    
    results = {}
    for metric in metric_names:
        values = [score[metric] for score in cv_scores if metric in score and not np.isnan(score[metric])]
        
        if not values:
            results[metric] = {'mean': np.nan, 'std': np.nan, 'n_folds': 0}
            continue
        
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        n_folds = len(values)
        
        # 95% confidence interval
        if n_folds > 1:
            ci = stats.t.interval(0.95, df=n_folds-1, loc=mean, scale=std/np.sqrt(n_folds))
            ci_lower, ci_upper = ci
        else:
            ci_lower, ci_upper = mean, mean
        
        results[metric] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'min': min(values),
            'max': max(values),
            'n_folds': n_folds
        }
    
    return results


def create_cv_comparison_table(
    cv_results: Dict[str, List[Dict[str, float]]],
    primary_metric: str = 'metric1_official'
) -> pd.DataFrame:
    """
    Create a comparison table for different models across CV folds.
    
    Args:
        cv_results: Dict mapping model names to list of fold scores
        primary_metric: Primary metric for ranking
        
    Returns:
        DataFrame with model comparison statistics
    """
    rows = []
    for model_name, fold_scores in cv_results.items():
        agg = aggregate_cv_scores(fold_scores, [primary_metric])
        
        if primary_metric in agg:
            stats = agg[primary_metric]
            rows.append({
                'model': model_name,
                'mean': stats['mean'],
                'std': stats['std'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'min': stats['min'],
                'max': stats['max'],
                'n_folds': stats['n_folds']
            })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('mean', ascending=True)  # Lower is better for error metrics
        df['rank'] = range(1, len(df) + 1)
    
    return df


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict[str, float]:
    """
    Perform paired t-test to compare two models.
    
    Args:
        scores_a: CV scores for model A
        scores_b: CV scores for model B (same folds)
        
    Returns:
        Dict with t_statistic, p_value, mean_diff, ci_diff_lower, ci_diff_upper
    """
    from scipy import stats
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length (same number of folds)")
    
    if len(scores_a) < 2:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean_diff': np.nan,
            'is_significant': False
        }
    
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Mean difference
    diff = scores_a - scores_b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    
    # 95% CI for difference
    ci = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=std_diff/np.sqrt(n))
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'ci_diff_lower': ci[0],
        'ci_diff_upper': ci[1],
        'is_significant': p_value < 0.05
    }


# ==============================================================================
# MULTIPLE CV SCHEMES (Section 4.1)
# ==============================================================================

def create_multi_seed_cv(
    panel_df: pd.DataFrame,
    n_folds: int = 5,
    seeds: List[int] = [42, 123, 456],
    stratify_by: str = 'bucket'
) -> Dict[int, List[Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Create multiple CV schemes with different random seeds.
    
    This helps identify models that are robust across different fold assignments.
    Models that perform consistently well across multiple CV schemes are more
    trustworthy.
    
    Args:
        panel_df: Full panel data
        n_folds: Number of folds per CV scheme
        seeds: List of random seeds to use
        stratify_by: Column to stratify by
        
    Returns:
        Dict mapping seed -> list of (train_df, val_df) tuples
    """
    cv_schemes = {}
    
    for seed in seeds:
        folds = get_fold_series(
            panel_df,
            n_folds=n_folds,
            stratify_by=stratify_by,
            random_state=seed
        )
        cv_schemes[seed] = folds
        logger.info(f"Created CV scheme with seed={seed}, {len(folds)} folds")
    
    return cv_schemes


def validate_model_across_cv_schemes(
    cv_schemes: Dict[int, List[Tuple[pd.DataFrame, pd.DataFrame]]],
    model_train_fn,
    metric_fn,
    X_cols: List[str],
    y_col: str = 'y_norm'
) -> Dict[str, Any]:
    """
    Validate a model across multiple CV schemes to assess robustness.
    
    Args:
        cv_schemes: Dict from create_multi_seed_cv
        model_train_fn: Function(X_train, y_train, X_val, y_val) -> predictions
        metric_fn: Function(y_true, y_pred) -> float
        X_cols: Feature columns
        y_col: Target column
        
    Returns:
        Dict with:
            - per_scheme_scores: Dict[seed -> List[fold_scores]]
            - mean_per_scheme: Dict[seed -> mean_score]
            - overall_mean: float
            - overall_std: float
            - is_consistent: bool (std across schemes < threshold)
    """
    per_scheme_scores = {}
    scheme_means = []
    
    for seed, folds in cv_schemes.items():
        fold_scores = []
        
        for train_df, val_df in folds:
            X_train = train_df[X_cols]
            y_train = train_df[y_col]
            X_val = val_df[X_cols]
            y_val = val_df[y_col]
            
            # Train and predict
            preds = model_train_fn(X_train, y_train, X_val, y_val)
            
            # Compute metric
            score = metric_fn(y_val.values, preds)
            fold_scores.append(score)
        
        per_scheme_scores[seed] = fold_scores
        scheme_means.append(np.mean(fold_scores))
    
    overall_mean = np.mean(scheme_means)
    overall_std = np.std(scheme_means)
    
    # Consider consistent if std across schemes is < 5% of mean
    consistency_threshold = 0.05 * abs(overall_mean) if overall_mean != 0 else 0.01
    is_consistent = overall_std < consistency_threshold
    
    return {
        'per_scheme_scores': per_scheme_scores,
        'mean_per_scheme': {seed: np.mean(scores) for seed, scores in per_scheme_scores.items()},
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'is_consistent': is_consistent,
        'consistency_ratio': overall_std / abs(overall_mean) if overall_mean != 0 else np.inf
    }

