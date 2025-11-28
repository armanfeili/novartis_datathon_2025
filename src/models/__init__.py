"""
Model implementations for Novartis Datathon 2025.

All models follow the BaseModel interface:
- fit(X_train, y_train, X_val, y_val, sample_weight)
- predict(X)
- save(path)
- load(path)
- get_feature_importance()
"""

from .base import BaseModel
from .cat_model import CatBoostModel
from .lgbm_model import LGBMModel
from .xgb_model import XGBModel
from .linear import LinearModel, GlobalMeanBaseline, FlatBaseline, TrendBaseline
from .nn import NNModel

__all__ = [
    'BaseModel',
    'CatBoostModel',
    'LGBMModel',
    'XGBModel',
    'LinearModel',
    'GlobalMeanBaseline',
    'FlatBaseline',
    'TrendBaseline',
    'NNModel',
]


def get_model_class(name: str):
    """
    Get model class by name.
    
    Args:
        name: Model name ('catboost', 'lightgbm', 'xgboost', 'linear', 'nn', 
              'global_mean', 'flat')
    
    Returns:
        Model class
    """
    mapping = {
        'catboost': CatBoostModel,
        'lightgbm': LGBMModel,
        'lgbm': LGBMModel,
        'xgboost': XGBModel,
        'xgb': XGBModel,
        'linear': LinearModel,
        'nn': NNModel,
        'neural': NNModel,
        'global_mean': GlobalMeanBaseline,
        'flat': FlatBaseline,
        'trend': TrendBaseline,
    }
    
    name_lower = name.lower()
    if name_lower not in mapping:
        raise ValueError(f"Unknown model: {name}. Available: {list(mapping.keys())}")
    
    return mapping[name_lower]
