from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass
