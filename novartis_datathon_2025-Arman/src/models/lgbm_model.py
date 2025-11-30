"""
LightGBM model implementation for Novartis Datathon 2025.

Fast training and good for hyperparameter search.
"""

from typing import Optional, List
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from .base import BaseModel

logger = logging.getLogger(__name__)


class LGBMModel(BaseModel):
    """LightGBM implementation optimized for speed."""
    
    DEFAULT_CONFIG = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'verbose': -1,
        'seed': 42,
    }
    
    # Features that should have monotonic constraints (Section 3.4)
    # -1 = decreasing (as feature increases, prediction decreases)
    # +1 = increasing
    #  0 = no constraint
    MONOTONIC_FEATURES = {
        'months_postgx': -1,  # More time → lower volume (erosion)
        'n_gxs': -1,          # More generics → lower volume
        'n_gxs_cummax': -1,   # Max generics seen → lower volume
    }
    
    def __init__(self, config: dict):
        """
        Initialize LightGBM model.
        
        Args:
            config: Configuration dict with 'params' key for LGB parameters
                - monotonic_constraints: dict or 'auto' to use MONOTONIC_FEATURES
        """
        super().__init__(config)
        
        # Merge default config with provided config
        self.params = {**self.DEFAULT_CONFIG}
        if 'params' in config:
            self.params.update(config['params'])
        
        # Training settings
        self.training_config = config.get('training', {})
        
        # Monotonic constraints setting
        self.use_monotonic = config.get('monotonic_constraints', None)
        
        self.model = None
    
    def _get_monotonic_constraints(self, feature_names: List[str]) -> List[int]:
        """
        Build monotonic constraints array for feature list.
        
        Args:
            feature_names: List of feature column names
            
        Returns:
            List of constraint values (-1, 0, +1) for each feature
        """
        if self.use_monotonic is None or self.use_monotonic == False:
            return None
        
        if isinstance(self.use_monotonic, dict):
            # Use provided dict
            constraint_dict = self.use_monotonic
        elif self.use_monotonic == 'auto' or self.use_monotonic == True:
            # Use default monotonic features
            constraint_dict = self.MONOTONIC_FEATURES
        else:
            return None
        
        constraints = []
        for feat in feature_names:
            constraints.append(constraint_dict.get(feat, 0))
        
        # Only return if at least one non-zero constraint
        if any(c != 0 for c in constraints):
            logger.info(f"Applying monotonic constraints to {sum(c != 0 for c in constraints)} features")
            return constraints
        return None
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'LGBMModel':
        """
        Train LightGBM model with optional validation and sample weights.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features for early stopping
            y_val: Optional validation target
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Convert categorical columns to numeric codes for LightGBM
        X_train_proc = X_train.copy()
        categorical_features = []
        for i, col in enumerate(X_train_proc.columns):
            if X_train_proc[col].dtype.name == 'category':
                X_train_proc[col] = X_train_proc[col].cat.codes.astype(int)
                categorical_features.append(col)
            elif X_train_proc[col].dtype == 'object':
                X_train_proc[col] = pd.Categorical(X_train_proc[col]).codes.astype(int)
                categorical_features.append(col)
        
        # Convert to numpy to avoid potential pandas issues
        X_train_np = np.nan_to_num(X_train_proc.values.astype(np.float64), nan=0.0)
        y_train_np = y_train.values.astype(np.float64) if hasattr(y_train, 'values') else np.array(y_train, dtype=np.float64)
        weight_np = sample_weight.values.astype(np.float64) if sample_weight is not None else None
        
        # Create datasets with numpy arrays
        train_set = lgb.Dataset(
            X_train_np, 
            y_train_np,
            weight=weight_np,
            feature_name=list(X_train_proc.columns)
        )
        
        # Add monotonic constraints if enabled
        monotonic_constraints = self._get_monotonic_constraints(list(X_train_proc.columns))
        if monotonic_constraints is not None:
            self.params['monotone_constraints'] = monotonic_constraints
        
        valid_sets = [train_set]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            # Convert categorical columns in validation set
            X_val_proc = X_val.copy()
            for col in X_val_proc.columns:
                if X_val_proc[col].dtype.name == 'category':
                    X_val_proc[col] = X_val_proc[col].cat.codes.astype(int)
                elif X_val_proc[col].dtype == 'object':
                    X_val_proc[col] = pd.Categorical(X_val_proc[col]).codes.astype(int)
            X_val_np = np.nan_to_num(X_val_proc.values.astype(np.float64), nan=0.0)
            y_val_np = y_val.values.astype(np.float64) if hasattr(y_val, 'values') else np.array(y_val, dtype=np.float64)
            val_set = lgb.Dataset(X_val_np, y_val_np, feature_name=list(X_val_proc.columns))
            valid_sets.append(val_set)
            valid_names.append('valid')
        
        # Extract n_estimators from params
        n_estimators = self.params.pop('n_estimators', 1000)
        early_stopping_rounds = self.params.pop('early_stopping_rounds', 50)
        
        # Callbacks - use simpler form
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=self.training_config.get('verbose_eval', 100))
        ]
        
        # Train
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Restore params for potential reuse
        self.params['n_estimators'] = n_estimators
        self.params['early_stopping_rounds'] = early_stopping_rounds
        
        logger.info(f"LightGBM trained for {self.model.best_iteration} iterations")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        # Convert categorical columns to numeric codes
        X_proc = X.copy()
        for col in X_proc.columns:
            if X_proc[col].dtype.name == 'category':
                X_proc[col] = X_proc[col].cat.codes.astype(int)
            elif X_proc[col].dtype == 'object':
                X_proc[col] = pd.Categorical(X_proc[col]).codes.astype(int)
        X_np = np.nan_to_num(X_proc.values.astype(np.float64), nan=0.0)
        return self.model.predict(X_np)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LGBMModel':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({})
        instance.model = data['model']
        instance.feature_names = data.get('feature_names', [])
        instance.params = data.get('params', {})
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances.
        
        Returns:
            DataFrame with columns ['feature', 'importance']
        """
        if self.model is None or len(self.feature_names) == 0:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        importance = self.model.feature_importance(importance_type='gain')
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
