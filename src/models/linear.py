"""
Linear and baseline models for Novartis Datathon 2025.

Includes:
- LinearModel: Ridge/Lasso/ElasticNet with preprocessing
- GlobalMeanBaseline: Predict global average erosion curve
- FlatBaseline: Predict no erosion (y_norm = 1.0)
"""

from typing import Optional, Dict
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .base import BaseModel

logger = logging.getLogger(__name__)


class LinearModel(BaseModel):
    """Linear regression with preprocessing pipeline."""
    
    def __init__(self, config: dict):
        """
        Initialize linear model.
        
        Args:
            config: Configuration with model type and preprocessing settings
        """
        super().__init__(config)
        
        self.model_type = config.get('model', {}).get('type', 'ridge')
        self.params = config.get(self.model_type, {})
        self.preprocessing = config.get('preprocessing', {})
        
        # Define regressor
        if self.model_type == 'ridge':
            regressor = Ridge(**self.params)
        elif self.model_type == 'lasso':
            regressor = Lasso(**self.params)
        elif self.model_type == 'elasticnet':
            regressor = ElasticNet(**self.params)
        elif self.model_type == 'huber':
            regressor = HuberRegressor(**self.params)
        else:
            raise ValueError(f"Unknown linear model type: {self.model_type}")
        
        # Build pipeline
        steps = []
        if self.preprocessing.get('handle_missing'):
            steps.append(('imputer', SimpleImputer(strategy=self.preprocessing['handle_missing'])))
        
        if self.preprocessing.get('scale_features', True):
            steps.append(('scaler', StandardScaler()))
        
        steps.append(('model', regressor))
        self.model = Pipeline(steps)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'LinearModel':
        """
        Train linear model.
        
        Note: Linear models in sklearn don't use validation for early stopping.
        sample_weight is passed to the regressor if supported.
        """
        self.feature_names = list(X_train.columns)
        
        # Fit with sample weights if supported and provided
        if sample_weight is not None:
            # Pipeline fit_params need to be prefixed with step name
            self.model.fit(X_train, y_train, model__sample_weight=sample_weight.values)
        else:
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'LinearModel':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({'model': {'type': data.get('model_type', 'ridge')}})
        instance.model = data['model']
        instance.feature_names = data.get('feature_names', [])
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients as importance."""
        if len(self.feature_names) == 0:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Get coefficients from the model step in pipeline
        coefs = self.model.named_steps['model'].coef_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(coefs)
        }).sort_values('importance', ascending=False)


class GlobalMeanBaseline(BaseModel):
    """
    Predict using global average erosion curve from training data.
    
    This baseline learns the average y_norm (erosion) for each months_postgx
    from training data and predicts this average for all series.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.erosion_curve: Dict[int, float] = {}
        self.global_mean: float = 0.5
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'GlobalMeanBaseline':
        """
        Compute mean y_norm per months_postgx.
        
        Note: X_train must contain 'months_postgx' column.
        """
        if 'months_postgx' not in X_train.columns:
            raise ValueError("X_train must contain 'months_postgx' column for GlobalMeanBaseline")
        
        # Compute mean erosion by month
        df = pd.DataFrame({
            'months_postgx': X_train['months_postgx'],
            'y_norm': y_train
        })
        
        self.erosion_curve = df.groupby('months_postgx')['y_norm'].mean().to_dict()
        self.global_mean = y_train.mean()
        
        logger.info(f"GlobalMeanBaseline: learned erosion curve for {len(self.erosion_curve)} months")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply learned erosion curve.
        
        X must have 'months_postgx' column.
        """
        if 'months_postgx' not in X.columns:
            raise ValueError("X must contain 'months_postgx' column for prediction")
        
        return X['months_postgx'].map(self.erosion_curve).fillna(self.global_mean).values
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'erosion_curve': self.erosion_curve,
            'global_mean': self.global_mean
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'GlobalMeanBaseline':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({})
        instance.erosion_curve = data['erosion_curve']
        instance.global_mean = data['global_mean']
        return instance


class FlatBaseline(BaseModel):
    """
    Predict 1.0 (no erosion) as normalized volume.
    
    This is the simplest baseline - predicts that volume remains at pre-entry level.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.prediction_value: float = 1.0
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'FlatBaseline':
        """No fitting needed - always predicts 1.0."""
        logger.info("FlatBaseline: no fitting needed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1.0 for all predictions."""
        return np.ones(len(X)) * self.prediction_value
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({'prediction_value': self.prediction_value}, path)
    
    @classmethod
    def load(cls, path: str) -> 'FlatBaseline':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({})
        instance.prediction_value = data.get('prediction_value', 1.0)
        return instance


class TrendBaseline(BaseModel):
    """
    Extrapolate pre-entry trend into post-entry period.
    
    This baseline uses the pre-entry volume trend (slope) to predict
    future volumes, applying a decay factor.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.decay_factor: float = config.get('decay_factor', 0.95)
        self.global_trend: float = 0.0
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'TrendBaseline':
        """Learn global decay trend from training data."""
        # Compute average y_norm change per month
        if 'months_postgx' in X_train.columns:
            df = pd.DataFrame({
                'months_postgx': X_train['months_postgx'],
                'y_norm': y_train
            })
            # Average slope
            monthly_avg = df.groupby('months_postgx')['y_norm'].mean()
            if len(monthly_avg) > 1:
                self.global_trend = monthly_avg.diff().mean()
        
        logger.info(f"TrendBaseline: learned global trend = {self.global_trend:.4f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using trend extrapolation.
        
        prediction = 1 + months_postgx * global_trend * decay_factor
        """
        if 'months_postgx' not in X.columns:
            return np.ones(len(X))
        
        predictions = 1.0 + X['months_postgx'] * self.global_trend * self.decay_factor
        return np.clip(predictions, 0, 2).values  # Clip to reasonable range
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'decay_factor': self.decay_factor,
            'global_trend': self.global_trend
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'TrendBaseline':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({})
        instance.decay_factor = data.get('decay_factor', 0.95)
        instance.global_trend = data.get('global_trend', 0.0)
        return instance
