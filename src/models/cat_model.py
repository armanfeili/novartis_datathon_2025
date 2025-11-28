"""
CatBoost model implementation for Novartis Datathon 2025.

Primary hero model with native categorical support and robustness to overfitting.
"""

from typing import Optional, List
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
    
    def __init__(self, config: dict):
        """
        Initialize CatBoost model.
        
        Args:
            config: Configuration dict with 'params' key for CatBoost parameters
        """
        super().__init__(config)
        
        # Merge default config with provided config
        self.params = {**self.DEFAULT_CONFIG}
        if 'params' in config:
            self.params.update(config['params'])
        
        # Get categorical feature names if specified
        self.cat_features: List[str] = config.get('categorical_features', [])
        
        # Initialize model
        self.model = CatBoostRegressor(**self.params)
    
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
        
        # Identify categorical features present in data
        cat_features_idx = []
        for cat_col in self.cat_features:
            if cat_col in X_train.columns:
                cat_features_idx.append(list(X_train.columns).index(cat_col))
        
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
