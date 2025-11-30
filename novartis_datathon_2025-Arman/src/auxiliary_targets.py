"""
Multi-Task Learning / Auxiliary Targets Module - Section 3.2

Creates auxiliary prediction tasks that can help the main regression task:
1. Bucket classification (auxiliary classifier)
2. Cumulative erosion predictor
3. Trend direction predictor

These auxiliary tasks can be used:
- As additional features (predicted probabilities/values)
- As multi-output training objectives
- To pre-train models that are then fine-tuned on the main task

Usage:
    from src.auxiliary_targets import (
        create_bucket_classifier_target,
        create_cumulative_erosion_target,
        create_trend_direction_target,
        train_auxiliary_classifier,
        add_auxiliary_predictions_as_features
    )
    
    # Create auxiliary targets
    train_df = create_bucket_classifier_target(train_df)
    train_df = create_cumulative_erosion_target(train_df)
    
    # Train auxiliary models and add predictions as features
    train_df = add_auxiliary_predictions_as_features(train_df, feature_cols)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ==============================================================================
# AUXILIARY TARGET CREATION
# ==============================================================================

def create_bucket_classifier_target(
    df: pd.DataFrame,
    bucket_col: str = 'bucket',
    target_name: str = 'aux_bucket_target'
) -> pd.DataFrame:
    """
    Create bucket classification target (1 = fast decay, 2 = slow decay).
    
    This can be used to train a classifier that predicts decay speed,
    which can then inform the regression predictions.
    
    Args:
        df: DataFrame with bucket column
        bucket_col: Name of bucket column
        target_name: Name for the new target column
        
    Returns:
        DataFrame with new auxiliary target
    """
    df = df.copy()
    
    if bucket_col in df.columns:
        # Binary target: is_fast_decay (bucket == 1)
        df[target_name] = (df[bucket_col] == 1).astype(int)
        df[f'{target_name}_multiclass'] = df[bucket_col]
        
        logger.info(f"Created bucket classifier target: "
                   f"{df[target_name].sum()} fast decay / {len(df)} total")
    else:
        logger.warning(f"Column {bucket_col} not found, skipping bucket target")
    
    return df


def create_cumulative_erosion_target(
    df: pd.DataFrame,
    target_col: str = 'y_norm',
    time_col: str = 'months_postgx',
    series_keys: List[str] = ['country', 'brand_name'],
    target_name: str = 'aux_cumul_erosion'
) -> pd.DataFrame:
    """
    Create cumulative erosion target: 1 - mean(y_norm) up to current month.
    
    This represents the total erosion experienced so far, which is a smoother
    target than the instantaneous erosion.
    
    Args:
        df: DataFrame with target and time columns
        target_col: Target column (y_norm)
        time_col: Time index column
        series_keys: Columns identifying each series
        target_name: Name for the new target
        
    Returns:
        DataFrame with cumulative erosion target
    """
    df = df.copy()
    
    if target_col not in df.columns:
        logger.warning(f"Column {target_col} not found, skipping cumulative erosion target")
        return df
    
    # Sort by series and time
    df = df.sort_values(series_keys + [time_col])
    
    # Calculate expanding mean of target within each series
    df[target_name] = df.groupby(series_keys)[target_col].transform(
        lambda x: 1 - x.expanding().mean()
    )
    
    logger.info(f"Created cumulative erosion target: "
               f"mean={df[target_name].mean():.3f}")
    
    return df


def create_trend_direction_target(
    df: pd.DataFrame,
    target_col: str = 'y_norm',
    time_col: str = 'months_postgx',
    series_keys: List[str] = ['country', 'brand_name'],
    window: int = 3,
    target_name: str = 'aux_trend_direction'
) -> pd.DataFrame:
    """
    Create trend direction target: -1/0/1 for decreasing/flat/increasing.
    
    This classification target captures the local trend direction,
    which can help the model understand momentum.
    
    Args:
        df: DataFrame with target and time columns
        target_col: Target column
        time_col: Time index column
        series_keys: Columns identifying each series
        window: Window size for trend calculation
        target_name: Name for the new target
        
    Returns:
        DataFrame with trend direction target
    """
    df = df.copy()
    
    if target_col not in df.columns:
        logger.warning(f"Column {target_col} not found, skipping trend direction target")
        return df
    
    # Sort by series and time
    df = df.sort_values(series_keys + [time_col])
    
    # Calculate rolling change
    df['_temp_diff'] = df.groupby(series_keys)[target_col].transform(
        lambda x: x.diff(window)
    )
    
    # Classify direction
    threshold = 0.02  # 2% change threshold for "flat"
    df[target_name] = 0  # Flat
    df.loc[df['_temp_diff'] < -threshold, target_name] = -1  # Decreasing
    df.loc[df['_temp_diff'] > threshold, target_name] = 1   # Increasing
    
    # Clean up
    df = df.drop('_temp_diff', axis=1)
    
    logger.info(f"Created trend direction target: "
               f"down={sum(df[target_name]==-1)}, "
               f"flat={sum(df[target_name]==0)}, "
               f"up={sum(df[target_name]==1)}")
    
    return df


def create_next_month_target(
    df: pd.DataFrame,
    target_col: str = 'y_norm',
    time_col: str = 'months_postgx',
    series_keys: List[str] = ['country', 'brand_name'],
    horizon: int = 1,
    target_name: str = 'aux_next_month'
) -> pd.DataFrame:
    """
    Create next-month prediction target for auxiliary task.
    
    This can be used to train a one-step-ahead predictor as an auxiliary task.
    
    Args:
        df: DataFrame with target and time columns
        target_col: Target column
        time_col: Time index column
        series_keys: Columns identifying each series
        horizon: Number of months ahead to predict
        target_name: Name for the new target
        
    Returns:
        DataFrame with next-month target
    """
    df = df.copy()
    
    # Sort by series and time
    df = df.sort_values(series_keys + [time_col])
    
    # Shift target backwards (future value becomes current row's target)
    df[target_name] = df.groupby(series_keys)[target_col].shift(-horizon)
    
    n_valid = df[target_name].notna().sum()
    logger.info(f"Created next-month target (h={horizon}): {n_valid} valid samples")
    
    return df


# ==============================================================================
# AUXILIARY MODEL TRAINING
# ==============================================================================

def train_auxiliary_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_folds: int = 5,
    model_type: str = 'lgbm'
) -> Tuple[Any, np.ndarray]:
    """
    Train an auxiliary classifier using cross-validation.
    
    Returns out-of-fold predictions that can be used as features
    for the main model.
    
    Args:
        df: Training DataFrame
        feature_cols: Feature columns
        target_col: Auxiliary target column
        n_folds: Number of CV folds
        model_type: 'lgbm', 'xgb', or 'catboost'
        
    Returns:
        Tuple of (trained_model, oof_predictions)
    """
    X = df[feature_cols].values
    y = df[target_col].values
    
    oof_preds = np.zeros(len(df))
    
    # Determine if classification or regression
    n_unique = len(np.unique(y[~np.isnan(y)]))
    is_classification = n_unique <= 10
    
    if model_type == 'lgbm':
        import lightgbm as lgb
        
        if is_classification:
            params = {
                'objective': 'binary' if n_unique == 2 else 'multiclass',
                'num_class': n_unique if n_unique > 2 else 1,
                'metric': 'auc' if n_unique == 2 else 'multi_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'verbose': -1
            }
        else:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'verbose': -1
            }
    else:
        # Default to simple gradient boosting
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        if is_classification:
            base_model = GradientBoostingClassifier(n_estimators=100, max_depth=4)
        else:
            base_model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
    
    # Cross-validation
    if is_classification:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    models = []
    valid_mask = ~np.isnan(y)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X[valid_mask], y[valid_mask])):
        # Map back to original indices
        original_train_idx = np.where(valid_mask)[0][train_idx]
        original_val_idx = np.where(valid_mask)[0][val_idx]
        
        X_train, X_val = X[original_train_idx], X[original_val_idx]
        y_train, y_val = y[original_train_idx], y[original_val_idx]
        
        if model_type == 'lgbm':
            import lightgbm as lgb
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            if is_classification:
                if n_unique == 2:
                    oof_preds[original_val_idx] = model.predict(X_val)
                else:
                    # For multiclass, use class 1 probability
                    oof_preds[original_val_idx] = model.predict(X_val)[:, 1]
            else:
                oof_preds[original_val_idx] = model.predict(X_val)
        else:
            model = base_model.__class__(**base_model.get_params())
            model.fit(X_train, y_train)
            
            if is_classification:
                oof_preds[original_val_idx] = model.predict_proba(X_val)[:, 1]
            else:
                oof_preds[original_val_idx] = model.predict(X_val)
        
        models.append(model)
        logger.debug(f"Fold {fold_idx + 1}/{n_folds} complete")
    
    logger.info(f"Trained auxiliary {model_type} on {target_col}: "
               f"{'classification' if is_classification else 'regression'}")
    
    return models, oof_preds


# ==============================================================================
# FEATURE AUGMENTATION WITH AUXILIARY PREDICTIONS
# ==============================================================================

def add_auxiliary_predictions_as_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    auxiliary_targets: List[str] = ['aux_bucket_target', 'aux_cumul_erosion'],
    n_folds: int = 5,
    prefix: str = 'aux_pred_'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train auxiliary models and add their predictions as features.
    
    This implements a form of stacking where auxiliary predictions
    become input features for the main model.
    
    Args:
        train_df: Training DataFrame with auxiliary targets
        test_df: Test DataFrame
        feature_cols: Feature columns for auxiliary models
        auxiliary_targets: List of auxiliary target column names
        n_folds: CV folds for OOF prediction
        prefix: Prefix for new feature columns
        
    Returns:
        Tuple of (augmented_train_df, augmented_test_df)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for aux_target in auxiliary_targets:
        if aux_target not in train_df.columns:
            logger.warning(f"Auxiliary target {aux_target} not found, skipping")
            continue
        
        # Train auxiliary model and get OOF predictions
        models, oof_preds = train_auxiliary_classifier(
            train_df, feature_cols, aux_target, n_folds=n_folds
        )
        
        # Add OOF predictions to training data
        feature_name = f"{prefix}{aux_target}"
        train_df[feature_name] = oof_preds
        
        # Average predictions for test data
        X_test = test_df[feature_cols].values
        test_preds = np.zeros(len(test_df))
        
        for model in models:
            if hasattr(model, 'predict'):
                test_preds += model.predict(X_test) / len(models)
        
        test_df[feature_name] = test_preds
        
        logger.info(f"Added auxiliary feature: {feature_name}")
    
    return train_df, test_df


def create_multi_output_targets(
    df: pd.DataFrame,
    primary_target: str = 'y_norm',
    auxiliary_targets: List[str] = ['aux_bucket_target', 'aux_cumul_erosion']
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare DataFrame for multi-output training.
    
    Some models (e.g., neural networks) can be trained on multiple
    targets simultaneously, which can improve generalization.
    
    Args:
        df: DataFrame with all targets
        primary_target: Main regression target
        auxiliary_targets: Auxiliary targets to include
        
    Returns:
        Tuple of (prepared_df, list_of_all_targets)
    """
    df = df.copy()
    
    all_targets = [primary_target] + [t for t in auxiliary_targets if t in df.columns]
    
    # Drop rows where any target is missing
    for target in all_targets:
        if target in df.columns:
            df = df[df[target].notna()]
    
    logger.info(f"Multi-output setup: {len(all_targets)} targets, {len(df)} samples")
    
    return df, all_targets


# ==============================================================================
# BUCKET-WEIGHTED TRAINING HELPER
# ==============================================================================

def compute_bucket_balanced_weights(
    df: pd.DataFrame,
    bucket_col: str = 'bucket',
    base_weight_col: Optional[str] = 'sample_weight'
) -> np.ndarray:
    """
    Compute sample weights that balance bucket representation.
    
    Bucket 1 (fast decay) is often underrepresented. This function
    computes weights that give equal importance to each bucket.
    
    Args:
        df: DataFrame with bucket column
        bucket_col: Bucket column name
        base_weight_col: Existing weight column to multiply (optional)
        
    Returns:
        Array of sample weights
    """
    if bucket_col not in df.columns:
        logger.warning(f"Column {bucket_col} not found, returning uniform weights")
        return np.ones(len(df))
    
    # Compute inverse frequency weights
    bucket_counts = df[bucket_col].value_counts()
    total = len(df)
    n_buckets = len(bucket_counts)
    
    weights = np.ones(len(df))
    
    for bucket, count in bucket_counts.items():
        bucket_mask = df[bucket_col] == bucket
        # Weight = total / (n_buckets * count_in_bucket)
        weights[bucket_mask] = total / (n_buckets * count)
    
    # Apply base weights if provided
    if base_weight_col and base_weight_col in df.columns:
        weights *= df[base_weight_col].values
    
    # Normalize to mean=1
    weights = weights / weights.mean()
    
    logger.info(f"Bucket-balanced weights: "
               f"bucket1_mean={weights[df[bucket_col]==1].mean():.2f}, "
               f"bucket2_mean={weights[df[bucket_col]==2].mean():.2f}")
    
    return weights
