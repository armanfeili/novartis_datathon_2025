"""
CatBoost model implementation for Novartis Datathon 2025.

Primary hero model with native categorical support and robustness to overfitting.
"""

from typing import Optional, List, Dict
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .base import BaseModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseModel):
    """CatBoost implementation with native categorical support."""
    
    DEFAULT_CONFIG = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'loss_function': 'RMSE',
        'early_stopping_rounds': 50,
        'verbose': 100,
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
        Initialize CatBoost model.
        
        Args:
            config: Configuration dict with 'params' key for CatBoost parameters
                - monotonic_constraints: dict or 'auto' to use MONOTONIC_FEATURES
        """
        super().__init__(config)
        
        # Merge default config with provided config
        self.params = {**self.DEFAULT_CONFIG}
        if 'params' in config:
            self.params.update(config['params'])
        
        # Get categorical feature names if specified
        self.cat_features: List[str] = config.get('categorical_features', [])
        
        # Monotonic constraints setting
        self.use_monotonic = config.get('monotonic_constraints', None)
        
        # Initialize model
        self.model = CatBoostRegressor(**self.params)
    
    def _get_monotonic_constraints(self, feature_names: List[str]) -> Optional[str]:
        """
        Build monotonic constraints string for CatBoost.
        
        CatBoost accepts constraints as a string like "0,-1,0,1,0,..." or
        as a dict mapping feature index/name to constraint value.
        
        Args:
            feature_names: List of feature column names
            
        Returns:
            String of constraint values or None
        """
        if self.use_monotonic is None or self.use_monotonic == False:
            return None
        
        if isinstance(self.use_monotonic, dict):
            constraint_dict = self.use_monotonic
        elif self.use_monotonic == 'auto' or self.use_monotonic == True:
            constraint_dict = self.MONOTONIC_FEATURES
        else:
            return None
        
        constraints = []
        for feat in feature_names:
            constraints.append(str(constraint_dict.get(feat, 0)))
        
        # Only return if at least one non-zero constraint
        if any(c != '0' for c in constraints):
            n_constrained = sum(1 for c in constraints if c != '0')
            logger.info(f"Applying monotonic constraints to {n_constrained} features")
            return ','.join(constraints)
        return None
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'CatBoostModel':
        """
        Train CatBoost model with optional validation and sample weights.
        
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
        
        # Auto-detect categorical columns (pandas category dtype) and merge with config
        cat_cols_from_dtype = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
        all_cat_features = list(set(self.cat_features) | set(cat_cols_from_dtype))
        
        # Get indices of categorical features present in data
        cat_features_idx = []
        for cat_col in all_cat_features:
            if cat_col in X_train.columns:
                cat_features_idx.append(list(X_train.columns).index(cat_col))
        
        # Get monotonic constraints if enabled
        monotonic_constraints = self._get_monotonic_constraints(self.feature_names)
        
        # Apply monotonic constraints to model params
        if monotonic_constraints is not None:
            self.model.set_params(monotone_constraints=monotonic_constraints)
        
        # Prepare training pool
        train_pool = Pool(
            X_train,
            y_train,
            cat_features=cat_features_idx if cat_features_idx else None,
            weight=sample_weight.values if sample_weight is not None else None
        )
        
        # Prepare validation pool if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(
                X_val,
                y_val,
                cat_features=cat_features_idx if cat_features_idx else None
            )
        
        # Train
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=True if eval_set else False,
            verbose=self.params.get('verbose', 100)
        )
        
        logger.info(f"CatBoost trained for {self.model.get_best_iteration()} iterations")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CatBoostModel':
        """Load model from disk."""
        instance = cls({})
        instance.model = CatBoostRegressor()
        instance.model.load_model(path)
        instance.feature_names = instance.model.feature_names_ or []
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances.
        
        Returns:
            DataFrame with columns ['feature', 'importance']
        """
        if self.model is None or len(self.feature_names) == 0:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        importance = self.model.get_feature_importance()
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
