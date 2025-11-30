"""
Base model interface for Novartis Datathon 2025.

All models must implement this interface for consistent usage.
"""

from abc import ABC, abstractmethod
from typing import Optional, List

import pandas as pd
import numpy as np


class BaseModel(ABC):
    """Abstract base for all models ensuring consistent interface."""
    
    def __init__(self, config: dict):
        """
        Initialize model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
    
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'BaseModel':
        """
        Train model with optional validation and sample weights.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features for early stopping
            y_val: Optional validation target
            sample_weight: Optional sample weights for weighted training
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for input features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
            
        Returns:
            Loaded model instance
        """
        pass
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance if available.
        
        Returns:
            DataFrame with columns ['feature', 'importance']
        """
        return pd.DataFrame(columns=['feature', 'importance'])
