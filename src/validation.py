"""
Validation and splitting module for Novartis Datathon 2025.

Implements series-level stratified validation mimicking true scenario constraints.
CRITICAL: Never mix months of the same series across train/val.
"""

import logging
from typing import Tuple, List, Optional, Dict, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import timer
from .features import _normalize_scenario

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
                counts = series_info.groupby(series_keys)[col].nunique()
                if (counts > 1).any():
                    logger.warning(f"Some series have multiple {col} values!")
        
        # Create combined stratification column for multi-column stratification
        stratify_available = [c for c in stratify_cols if c in series_info.columns]
        
        if len(stratify_available) > 0:
            if len(stratify_available) == 1:
                stratify_values = series_info[stratify_available[0]]
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
    random_state: int = 42
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
        
        logger.info(f"Adversarial validation AUC: {mean_auc:.4f} (std: {auc_scores.std():.4f})")
        
        if mean_auc > 0.7:
            logger.warning(f"High AUC ({mean_auc:.3f}) suggests train/test distribution shift!")
            logger.warning(f"Top shift features:\n{feature_importance.head(10).to_string()}")
        elif mean_auc > 0.6:
            logger.info(f"Moderate AUC ({mean_auc:.3f}), some distribution difference detected")
        else:
            logger.info(f"Low AUC ({mean_auc:.3f}), train/test distributions appear similar")
        
        return {
            'mean_auc': mean_auc,
            'auc_scores': auc_scores.tolist(),
            'top_shift_features': feature_importance
        }


def get_fold_series(
    panel_df: pd.DataFrame,
    n_folds: int = 5,
    stratify_by: str = 'bucket',
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate K-fold splits at series level.
    
    Args:
        panel_df: Full panel data
        n_folds: Number of folds
        stratify_by: Column to stratify by
        random_state: For reproducibility
        
    Returns:
        List of (train_df, val_df) tuples
    """
    from sklearn.model_selection import StratifiedKFold
    
    series_keys = ['country', 'brand_name']
    series_info = panel_df[series_keys + [stratify_by]].drop_duplicates()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in skf.split(series_info, series_info[stratify_by]):
        train_series = series_info.iloc[train_idx][series_keys]
        val_series = series_info.iloc[val_idx][series_keys]
        
        train_df = panel_df.merge(train_series, on=series_keys, how='inner')
        val_df = panel_df.merge(val_series, on=series_keys, how='inner')
        
        folds.append((train_df, val_df))
    
    return folds
