"""
XGBoost model wrapper for Novartis Datathon 2025.
"""

from typing import Optional

import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

from .base import BaseModel


# Default XGBoost configuration for generic erosion forecasting
DEFAULT_CONFIG = {
    'params': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'seed': 42
    },
    'training': {
        'num_boost_round': 2000,
        'early_stopping_rounds': 100,
        'verbose_eval': 100,
        'sample_weights': True
    }
}


class XGBModel(BaseModel):
    """XGBoost model with native sample weight support via DMatrix."""
    
    def __init__(self, config: dict):
        """
        Initialize XGBoost model.
        
        Args:
            config: Configuration dict with 'params' and 'training' sections
        """
        super().__init__(config)
        
        # Merge with defaults
        self.params = {**DEFAULT_CONFIG['params'], **config.get('params', {})}
        self.training_params = {**DEFAULT_CONFIG['training'], **config.get('training', {})}
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'XGBModel':
        """
        Train XGBoost model with optional sample weights.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional)
            sample_weight: Sample weights for training (optional)
        
        Returns:
            Self (for chaining)
        """
        self.feature_names = list(X_train.columns)
        
        # Convert categorical columns to numeric codes for XGBoost
        X_train_proc = X_train.copy()
        X_val_proc = X_val.copy() if X_val is not None else None
        
        for col in X_train_proc.columns:
            if X_train_proc[col].dtype.name == 'category':
                X_train_proc[col] = X_train_proc[col].cat.codes
                if X_val_proc is not None:
                    X_val_proc[col] = X_val_proc[col].cat.codes
        
        # Create DMatrix with sample weights
        weight = sample_weight.values if sample_weight is not None else None
        dtrain = xgb.DMatrix(X_train_proc, label=y_train, weight=weight)
        
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val_proc, label=y_val)
            evals.append((dval, 'eval'))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.training_params.get('num_boost_round', 2000),
            evals=evals,
            early_stopping_rounds=self.training_params.get('early_stopping_rounds', 100),
            verbose_eval=self.training_params.get('verbose_eval', 100)
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using trained model."""
        # Convert categorical columns to numeric codes
        X_proc = X.copy()
        for col in X_proc.columns:
            if X_proc[col].dtype.name == 'category':
                X_proc[col] = X_proc[col].cat.codes
        dtest = xgb.DMatrix(X_proc)
        return self.model.predict(dtest)
    
    def save(self, path: str) -> None:
        """Save model and metadata to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'XGBModel':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({'params': data.get('params', {})})
        instance.model = data['model']
        instance.feature_names = data.get('feature_names', [])
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with columns ['feature', 'importance']
        """
        if self.model is None or len(self.feature_names) == 0:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Get gain-based importance
        importance_dict = self.model.get_score(importance_type='gain')
        
        # Map feature names (XGBoost uses f0, f1, etc. internally)
        importance_data = []
        for i, name in enumerate(self.feature_names):
            key = f'f{i}'
            importance_data.append({
                'feature': name,
                'importance': importance_dict.get(key, 0)
            })
        
        return pd.DataFrame(importance_data).sort_values('importance', ascending=False)
