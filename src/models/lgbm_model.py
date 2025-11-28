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
    
    def __init__(self, config: dict):
        """
        Initialize LightGBM model.
        
        Args:
            config: Configuration dict with 'params' key for LGB parameters
        """
        super().__init__(config)
        
        # Merge default config with provided config
        self.params = {**self.DEFAULT_CONFIG}
        if 'params' in config:
            self.params.update(config['params'])
        
        # Training settings
        self.training_config = config.get('training', {})
        
        self.model = None
    
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
        
        # Create datasets
        train_set = lgb.Dataset(
            X_train, 
            y_train,
            weight=sample_weight.values if sample_weight is not None else None
        )
        
        valid_sets = [train_set]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append('valid')
        
        # Extract n_estimators from params
        n_estimators = self.params.pop('n_estimators', 1000)
        early_stopping_rounds = self.params.pop('early_stopping_rounds', 50)
        
        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
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
        return self.model.predict(X)
    
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
