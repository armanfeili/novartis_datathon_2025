"""
Validation and splitting module for Novartis Datathon 2025.

Implements series-level stratified validation mimicking true scenario constraints.
CRITICAL: Never mix months of the same series across train/val.
"""

import logging
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import timer

logger = logging.getLogger(__name__)


def create_validation_split(
    panel_df: pd.DataFrame,
    val_fraction: float = 0.2,
    stratify_by: str = 'bucket',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split at SERIES level (not row level).
    
    CRITICAL: Never mix months of the same series across train/val.
    
    Args:
        panel_df: Full training panel
        val_fraction: Fraction of series for validation
        stratify_by: Column to stratify by (usually 'bucket')
        random_state: For reproducibility
    
    Returns:
        (train_df, val_df) - both contain full series
    """
    with timer("Create validation split"):
        series_keys = ['country', 'brand_name']
        
        # Get unique series with their stratification labels
        series_info = panel_df[series_keys + [stratify_by]].drop_duplicates()
        
        # Check for series with multiple bucket values (shouldn't happen)
        bucket_counts = series_info.groupby(series_keys)[stratify_by].nunique()
        if (bucket_counts > 1).any():
            logger.warning("Some series have multiple bucket values!")
        
        # Stratified split at series level
        if stratify_by in series_info.columns:
            train_series, val_series = train_test_split(
                series_info,
                test_size=val_fraction,
                stratify=series_info[stratify_by],
                random_state=random_state
            )
        else:
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
        
        if stratify_by in train_series.columns:
            train_bucket_dist = train_series[stratify_by].value_counts(normalize=True)
            val_bucket_dist = val_series[stratify_by].value_counts(normalize=True)
            logger.info(f"Train bucket distribution:\n{train_bucket_dist.to_string()}")
            logger.info(f"Val bucket distribution:\n{val_bucket_dist.to_string()}")
    
    return train_df, val_df


def simulate_scenario(
    val_df: pd.DataFrame,
    scenario: str
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
        scenario: "scenario1" or "scenario2"
        
    Returns:
        (features_df, targets_df)
    """
    if scenario == 'scenario1':
        cutoff = 0
        target_start = 0
        target_end = 23
    elif scenario == 'scenario2':
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
