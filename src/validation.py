"""
Validation and splitting module for Novartis Datathon 2025.

Implements series-level stratified validation mimicking true scenario constraints.
CRITICAL: Never mix months of the same series across train/val.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import timer
from .features import _normalize_scenario
from .evaluate import make_metric_record, save_metric_records
from .data import ID_COLS, TIME_COL, META_COLS  # Import centralized constants

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
        series_keys = ID_COLS  # Use centralized constant
        
        # Handle stratify_by as string or list
        if isinstance(stratify_by, str):
            stratify_cols = [stratify_by]
        else:
            stratify_cols = list(stratify_by)
        
        # Get unique series with their stratification labels
        required_cols = series_keys + stratify_cols
        available_cols = [c for c in required_cols if c in panel_df.columns]
        series_info = panel_df[available_cols].drop_duplicates()
        
        # W1.1: Log warning if requested stratify columns are missing
        missing_stratify = [c for c in stratify_cols if c not in panel_df.columns]
        if missing_stratify:
            logger.warning(
                f"Requested stratify columns not found in data: {missing_stratify}. "
                f"Using available columns only: {[c for c in stratify_cols if c in panel_df.columns]}"
            )
        
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
            
            # W1.2: Handle rare classes robustly with minimum absolute threshold
            # Minimum class size must allow at least 2-3 samples for train and 1 for val
            # to enable stratified splitting
            value_counts = stratify_values.value_counts()
            # Absolute minimum: at least 3 series per class for proper stratification
            min_absolute = 3
            # Also consider relative minimum based on val_fraction
            min_relative = max(2, int(len(series_info) * val_fraction * 0.5))
            min_class_size = max(min_absolute, min_relative)
            
            rare_classes = value_counts[value_counts < min_class_size].index
            if len(rare_classes) > 0:
                n_rare_series = value_counts[value_counts < min_class_size].sum()
                logger.warning(
                    f"Grouping {len(rare_classes)} rare stratification classes "
                    f"(< {min_class_size} series each, {n_rare_series} series total) into 'OTHER'"
                )
                stratify_values = stratify_values.copy()
                stratify_values[stratify_values.isin(rare_classes)] = 'OTHER'
                
                # Check if 'OTHER' class itself is now very large or small
                other_count = (stratify_values == 'OTHER').sum()
                if other_count > 0:
                    logger.info(f"'OTHER' class now contains {other_count} series")
            
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
        
        # Explicit output validation (W1.3)
        _validate_split_output(panel_df, train_df, val_df, series_keys)
    
    return train_df, val_df


def _validate_split_output(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    series_keys: List[str]
) -> None:
    """
    Validate that train/val splits are correct.
    
    Checks:
    1. Train and val are disjoint at series level
    2. Together they cover all series in the original DataFrame
    
    Args:
        original_df: Original panel DataFrame
        train_df: Training split
        val_df: Validation split
        series_keys: Column names for series identification
        
    Raises:
        ValueError: If validation fails
    """
    # Get series sets
    original_series = set(original_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
    train_series = set(train_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
    val_series = set(val_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
    
    # Check disjoint
    overlap = train_series & val_series
    if overlap:
        raise ValueError(f"Train/val series overlap: {len(overlap)} series appear in both splits")
    
    # Check coverage
    combined = train_series | val_series
    missing = original_series - combined
    extra = combined - original_series
    
    if missing:
        logger.warning(f"Split missing {len(missing)} series from original data")
    if extra:
        logger.warning(f"Split has {len(extra)} extra series not in original data")
    
    logger.debug(f"Split validation passed: {len(train_series)} train, {len(val_series)} val series")


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
    series_keys = ID_COLS  # Use centralized constant
    
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
    metrics_dir: Optional[Path] = None,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_per_fold: int = 50
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
        n_estimators: Number of trees in RandomForest
        max_depth: Max tree depth
        min_samples_per_fold: Minimum samples per fold; skips CV if dataset too small
        
    Returns:
        {
            'mean_auc': float,  # ~0.5 = good, >0.7 = shift detected
            'auc_scores': list,
            'top_shift_features': DataFrame with feature importances,
            'skipped': bool  # True if skipped due to small dataset
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
        return {'mean_auc': np.nan, 'auc_scores': [], 'top_shift_features': pd.DataFrame(), 'skipped': True}
    
    with timer("Adversarial validation"):
        # Check for very small datasets (X.1)
        total_samples = len(train_features) + len(test_features)
        if total_samples < min_samples_per_fold * n_folds:
            logger.warning(
                f"Dataset too small for adversarial validation "
                f"({total_samples} samples < {min_samples_per_fold * n_folds} required for {n_folds} folds)"
            )
            return {'mean_auc': np.nan, 'auc_scores': [], 'top_shift_features': pd.DataFrame(), 'skipped': True}
        
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
        
        # Train classifier with CV using configurable hyperparameters
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
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
        stratify_by: Column to stratify by (gracefully falls back to KFold if missing)
        random_state: For reproducibility
        save_indices: Whether to save fold indices for reproducibility
        output_path: Path to save fold indices (JSON file)
        
    Returns:
        List of (train_df, val_df) tuples
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    import json
    
    series_keys = ID_COLS  # Use centralized constant
    
    # W3.1: Handle missing stratify_by column gracefully
    if stratify_by not in panel_df.columns:
        logger.warning(
            f"Stratify column '{stratify_by}' not found in data. "
            f"Falling back to simple KFold without stratification."
        )
        series_info = panel_df[series_keys].drop_duplicates()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iterator = kf.split(series_info)
        use_stratification = False
    else:
        series_info = panel_df[series_keys + [stratify_by]].drop_duplicates()
        
        # Log stratification class distribution
        class_dist = series_info[stratify_by].value_counts()
        logger.info(f"Stratification by '{stratify_by}' class distribution:")
        for cls, count in class_dist.items():
            logger.info(f"  {cls}: {count} series ({count/len(series_info)*100:.1f}%)")
        
        # Check for imbalanced classes
        min_count = class_dist.min()
        if min_count < n_folds:
            logger.warning(
                f"Smallest class has only {min_count} series, "
                f"which is less than n_folds={n_folds}. "
                f"Some folds may have empty validation for this class."
            )
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iterator = skf.split(series_info, series_info[stratify_by])
        use_stratification = True
    
    folds = []
    fold_indices = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
        train_series = series_info.iloc[train_idx][series_keys]
        val_series = series_info.iloc[val_idx][series_keys]
        
        train_df = panel_df.merge(train_series, on=series_keys, how='inner')
        val_df = panel_df.merge(val_series, on=series_keys, how='inner')
        
        # Log per-fold statistics
        logger.info(
            f"Fold {fold_idx + 1}/{n_folds}: "
            f"train {len(train_series)} series ({len(train_df):,} rows), "
            f"val {len(val_series)} series ({len(val_df):,} rows)"
        )
        
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
                'stratify_by': stratify_by if use_stratification else None,
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
    
    series_keys = ID_COLS  # Use centralized constant
    
    # W3.2: Handle missing group_by column
    if group_by not in panel_df.columns:
        logger.warning(
            f"Group column '{group_by}' not found in data. "
            f"Falling back to regular get_fold_series."
        )
        return get_fold_series(panel_df, n_folds=n_folds, random_state=random_state)
    
    series_info = panel_df[series_keys + [group_by]].drop_duplicates()
    
    # GroupKFold doesn't shuffle, so we need to shuffle groups first
    np.random.seed(random_state)
    unique_groups = series_info[group_by].unique()
    
    # W3.2: Check if n_folds exceeds number of unique groups
    if len(unique_groups) < n_folds:
        logger.warning(
            f"Only {len(unique_groups)} unique groups for '{group_by}', "
            f"reducing n_folds from {n_folds} to {len(unique_groups)}"
        )
    actual_n_folds = min(n_folds, len(unique_groups))
    
    shuffled_groups = np.random.permutation(unique_groups)
    group_map = {g: i for i, g in enumerate(shuffled_groups)}
    series_info['_group_idx'] = series_info[group_by].map(group_map)
    
    gkf = GroupKFold(n_splits=actual_n_folds)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(series_info, groups=series_info['_group_idx'])):
        train_series = series_info.iloc[train_idx][series_keys]
        val_series = series_info.iloc[val_idx][series_keys]
        
        train_df = panel_df.merge(train_series, on=series_keys, how='inner')
        val_df = panel_df.merge(val_series, on=series_keys, how='inner')
        
        folds.append((train_df, val_df))
        
        # W3.2: Log group distribution for each fold
        train_groups = series_info.iloc[train_idx][group_by].unique()
        val_groups = series_info.iloc[val_idx][group_by].unique()
        logger.info(
            f"Fold {fold_idx + 1}/{actual_n_folds}: "
            f"train {len(train_series)} series ({len(train_groups)} {group_by} values), "
            f"val {len(val_series)} series ({len(val_groups)} {group_by} values)"
        )
    
    return folds


def create_purged_cv_split(
    panel_df: pd.DataFrame,
    n_folds: int = 5,
    gap_months: int = 3,
    min_train_months: int = -6,
    stratify_by: Optional[str] = 'bucket',
    random_state: int = 42,
    min_train_rows: int = 100,
    min_val_rows: int = 50
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create purged cross-validation splits with a time gap between train and val.
    
    This helps prevent data leakage from temporal proximity. The gap ensures
    that recent training data doesn't overlap with validation data.
    
    W4.1: Enhanced with explicit checks for non-empty folds.
    
    Args:
        panel_df: Full panel data
        n_folds: Number of folds
        gap_months: Number of months gap between train and val (recommended: 0-3 for S1, 3-6 for S2)
        min_train_months: Minimum months_postgx to include in training (S1: -12, S2: -6 typical)
        stratify_by: Column to stratify by (if any)
        random_state: For reproducibility
        min_train_rows: Minimum rows required for training data per fold
        min_val_rows: Minimum rows required for validation data per fold
        
    Returns:
        List of (train_df, val_df) tuples
    """
    series_keys = ID_COLS  # Use centralized constant
    
    # Log configuration for clarity
    logger.info(
        f"Creating purged CV: {n_folds} folds, gap={gap_months} months, "
        f"min_train_months={min_train_months}, stratify_by={stratify_by}"
    )
    
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
    skipped_folds = []
    
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
        
        # W4.1: Explicit checks for non-empty folds
        if len(train_df) < min_train_rows:
            logger.warning(
                f"Fold {fold_idx + 1}: train has only {len(train_df)} rows "
                f"(< min_train_rows={min_train_rows}). Consider reducing gap_months or adjusting min_train_months."
            )
            skipped_folds.append((fold_idx + 1, 'train too small', len(train_df)))
            continue
            
        if len(val_df) < min_val_rows:
            logger.warning(
                f"Fold {fold_idx + 1}: val has only {len(val_df)} rows "
                f"(< min_val_rows={min_val_rows})"
            )
            skipped_folds.append((fold_idx + 1, 'val too small', len(val_df)))
            continue
        
        folds.append((train_df, val_df))
        logger.info(
            f"Fold {fold_idx + 1}: train months [{min_train_months}, {train_cutoff}] "
            f"({len(train_df):,} rows), val starts at {val_min_month} ({len(val_df):,} rows)"
        )
    
    # Log summary of skipped folds
    if skipped_folds:
        logger.warning(f"Skipped {len(skipped_folds)}/{n_folds} folds due to insufficient data:")
        for fold_num, reason, count in skipped_folds:
            logger.warning(f"  Fold {fold_num}: {reason} ({count} rows)")
    
    if len(folds) == 0:
        logger.error(
            "All folds were skipped! Check your gap_months and min_train_months settings. "
            f"Recommended: gap_months=0-3 for S1, min_train_months=-12; gap_months=3-6 for S2, min_train_months=-6"
        )
    
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
    series_keys = ID_COLS  # Use centralized constant
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
            'inner_folds': inner_folds_list,
            'outer_fold_idx': outer_idx  # W4.2: Add fold index for tracking
        })
        
        logger.info(
            f"Outer fold {outer_idx + 1}/{outer_folds}: "
            f"train={len(outer_train_df[series_keys].drop_duplicates())} series ({len(outer_train_df):,} rows), "
            f"val={len(outer_val_df[series_keys].drop_duplicates())} series ({len(outer_val_df):,} rows), "
            f"inner folds={len(inner_folds_list)}"
        )
    
    # W4.2: Log usage documentation
    logger.info(
        "Nested CV structure ready. Usage:\n"
        "  - Use inner_folds for hyperparameter search within each outer fold\n"
        "  - Train final model on outer_train with best hyperparameters\n"
        "  - Evaluate on outer_val for unbiased performance estimate\n"
        "  - Aggregate outer_val scores across all outer folds for final metric"
    )
    
    return nested_folds


def validate_cv_respects_scenario_constraints(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scenario: int,
    allow_series_overlap: bool = False
) -> Tuple[bool, List[str]]:
    """
    Verify that CV split respects scenario constraints.
    
    W5.1: Enhanced with configurable series overlap behavior.
    
    Checks:
    1. Series overlap (configurable - some CV strategies like temporal CV may allow it)
    2. Scenario-specific cutoff validation (months_postgx ranges)
    3. Series integrity - if not allowing overlap, months shouldn't be split
    4. Training has sufficient history (optional check for S2)
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        scenario: 1 or 2
        allow_series_overlap: If True, allows same series in train and val
            (useful for temporal CV strategies). Default False for series-level CV.
        
    Returns:
        (is_valid, list of violation messages)
    """
    from .features import _normalize_scenario
    scenario = _normalize_scenario(scenario)
    
    violations = []
    series_keys = ID_COLS  # Use centralized constant
    
    # Check 1: Series overlap (configurable)
    train_series = set(train_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
    val_series = set(val_df[series_keys].drop_duplicates().itertuples(index=False, name=None))
    overlap = train_series & val_series
    
    if overlap:
        if allow_series_overlap:
            logger.info(
                f"{len(overlap)} series appear in both train and val "
                f"(allowed for temporal CV strategy)"
            )
        else:
            violations.append(
                f"Series overlap between train and val: {len(overlap)} series. "
                f"Set allow_series_overlap=True for temporal CV strategies."
            )
    
    # Check 2: Scenario-specific cutoff validation
    if scenario == 1:
        # For S1, validation should only have months_postgx in [0, 23]
        if 'months_postgx' in val_df.columns:
            val_min = val_df['months_postgx'].min()
            val_max = val_df['months_postgx'].max()
            if val_min < 0:
                violations.append(f"S1 validation has pre-entry data (min month={val_min})")
            if val_max > 23:
                violations.append(f"S1 validation has months > 23 (max month={val_max})")
    elif scenario == 2:
        # For S2, validation should only have months_postgx in [6, 23]
        if 'months_postgx' in val_df.columns:
            val_min = val_df['months_postgx'].min()
            val_max = val_df['months_postgx'].max()
            if val_min < 6:
                violations.append(f"S2 validation has months < 6 (min month={val_min})")
            if val_max > 23:
                violations.append(f"S2 validation has months > 23 (max month={val_max})")
    
    # Check 3: If not allowing overlap, months shouldn't be split for same series
    if not allow_series_overlap and 'months_postgx' in train_df.columns:
        for country, brand in list(train_series)[:10]:  # Check first 10 for performance
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
    
    # Check 4 (optional): For S2, train should include months < 6 if we want early erosion features
    if scenario == 2 and 'months_postgx' in train_df.columns:
        train_min = train_df['months_postgx'].min()
        train_max = train_df['months_postgx'].max()
        if train_min >= 6:
            logger.warning(
                f"S2 training has no early erosion data (min month={train_min}). "
                f"Consider including months 0-5 for early_erosion features."
            )
    
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
        
    Note:
        X.2: Enhanced robustness - handles missing metrics and all-NaN gracefully.
    """
    from scipy import stats
    
    if not cv_scores:
        logger.warning("aggregate_cv_scores called with empty cv_scores list")
        return {}
    
    if metric_names is None:
        # Collect all unique metric names from all folds
        all_metrics = set()
        for score in cv_scores:
            all_metrics.update(score.keys())
        metric_names = list(all_metrics)
    
    results = {}
    for metric in metric_names:
        # X.2: Handle missing metrics for some folds
        values = []
        missing_count = 0
        nan_count = 0
        
        for fold_idx, score in enumerate(cv_scores):
            if metric not in score:
                missing_count += 1
                continue
            val = score[metric]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                nan_count += 1
                continue
            values.append(val)
        
        # Log warnings for missing/NaN values
        if missing_count > 0:
            logger.warning(
                f"Metric '{metric}': missing from {missing_count}/{len(cv_scores)} folds"
            )
        if nan_count > 0:
            logger.warning(
                f"Metric '{metric}': NaN in {nan_count}/{len(cv_scores)} folds"
            )
        
        # X.2: Handle all-NaN gracefully
        if not values:
            logger.warning(f"Metric '{metric}': all values are NaN or missing")
            results[metric] = {
                'mean': np.nan, 
                'std': np.nan, 
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'min': np.nan,
                'max': np.nan,
                'n_folds': 0
            }
            continue
        
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        n_folds = len(values)
        
        # 95% confidence interval
        if n_folds > 1:
            try:
                ci = stats.t.interval(0.95, df=n_folds-1, loc=mean, scale=std/np.sqrt(n_folds))
                ci_lower, ci_upper = ci
            except Exception as e:
                logger.warning(f"Could not compute CI for '{metric}': {e}")
                ci_lower, ci_upper = mean, mean
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
    primary_metric: str = 'metric1_official',
    additional_metrics: Optional[List[str]] = None,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Create a comparison table for different models across CV folds.
    
    X.3: Enhanced with configurable sort order and additional metrics.
    
    Args:
        cv_results: Dict mapping model names to list of fold scores
        primary_metric: Primary metric for ranking
        additional_metrics: Optional list of additional metrics to include in table
        ascending: If True, sort ascending (lower is better, e.g. for error metrics).
            If False, sort descending (higher is better, e.g. for accuracy).
        
    Returns:
        DataFrame with model comparison statistics
    """
    rows = []
    
    # Determine all metrics to include
    metrics_to_include = [primary_metric]
    if additional_metrics:
        metrics_to_include.extend(additional_metrics)
    
    for model_name, fold_scores in cv_results.items():
        agg = aggregate_cv_scores(fold_scores, metrics_to_include)
        
        row = {'model': model_name}
        
        # Add primary metric stats
        if primary_metric in agg:
            stats = agg[primary_metric]
            row.update({
                'mean': stats['mean'],
                'std': stats['std'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'min': stats['min'],
                'max': stats['max'],
                'n_folds': stats['n_folds']
            })
        else:
            row.update({
                'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan,
                'ci_upper': np.nan, 'min': np.nan, 'max': np.nan, 'n_folds': 0
            })
        
        # X.3: Add additional metrics if requested
        if additional_metrics:
            for metric in additional_metrics:
                if metric in agg:
                    row[f'{metric}_mean'] = agg[metric]['mean']
                    row[f'{metric}_std'] = agg[metric]['std']
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        # X.3: Configurable sort order
        df = df.sort_values('mean', ascending=ascending)
        df['rank'] = range(1, len(df) + 1)
        
        # Add visual indicator for best model
        if not df['mean'].isna().all():
            best_idx = df['mean'].idxmin() if ascending else df['mean'].idxmax()
            df['is_best'] = df.index == best_idx
    
    return df


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform paired t-test to compare two models.
    
    X.4: Enhanced with explicit safeguards for NaN and edge cases.
    
    Args:
        scores_a: CV scores for model A
        scores_b: CV scores for model B (same folds)
        alpha: Significance level (default 0.05)
        
    Returns:
        Dict with t_statistic, p_value, mean_diff, ci_diff_lower, ci_diff_upper, is_significant
        
    Raises:
        ValueError: If score lists have different lengths
    """
    from scipy import stats
    
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must have same length (same number of folds). "
            f"Got {len(scores_a)} and {len(scores_b)}"
        )
    
    # X.4: Check for minimum number of folds
    if len(scores_a) < 2:
        logger.warning(
            f"paired_t_test requires at least 2 folds, got {len(scores_a)}. "
            f"Returning NaN values."
        )
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean_diff': np.nan,
            'ci_diff_lower': np.nan,
            'ci_diff_upper': np.nan,
            'is_significant': False,
            'n_valid_pairs': 0
        }
    
    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)
    
    # X.4: Filter out NaN values (only use pairs where both are valid)
    valid_mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
    n_valid = valid_mask.sum()
    
    if n_valid < 2:
        logger.warning(
            f"paired_t_test: only {n_valid} valid (non-NaN) pairs. "
            f"Need at least 2 for t-test."
        )
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean_diff': np.nan,
            'ci_diff_lower': np.nan,
            'ci_diff_upper': np.nan,
            'is_significant': False,
            'n_valid_pairs': n_valid
        }
    
    if n_valid < len(scores_a):
        logger.warning(
            f"paired_t_test: {len(scores_a) - n_valid} pairs excluded due to NaN values"
        )
    
    scores_a_valid = scores_a[valid_mask]
    scores_b_valid = scores_b[valid_mask]
    
    # Paired t-test
    try:
        t_stat, p_value = stats.ttest_rel(scores_a_valid, scores_b_valid)
    except Exception as e:
        logger.error(f"paired_t_test failed: {e}")
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean_diff': np.nan,
            'ci_diff_lower': np.nan,
            'ci_diff_upper': np.nan,
            'is_significant': False,
            'n_valid_pairs': n_valid
        }
    
    # Mean difference
    diff = scores_a_valid - scores_b_valid
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    
    # Confidence interval for difference
    try:
        ci = stats.t.interval(1 - alpha, df=n-1, loc=mean_diff, scale=std_diff/np.sqrt(n))
        ci_lower, ci_upper = ci
    except Exception as e:
        logger.warning(f"Could not compute CI: {e}")
        ci_lower, ci_upper = np.nan, np.nan
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'ci_diff_lower': ci_lower,
        'ci_diff_upper': ci_upper,
        'is_significant': p_value < alpha,
        'n_valid_pairs': n_valid
    }
