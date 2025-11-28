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
from .linear import LinearModel, GlobalMeanBaseline, FlatBaseline, TrendBaseline

# Lazy imports for models that require native libraries
# These will only fail when actually used, not on package import
_CatBoostModel = None
_LGBMModel = None
_XGBModel = None
_NNModel = None


def _import_catboost():
    global _CatBoostModel
    if _CatBoostModel is None:
        from .cat_model import CatBoostModel
        _CatBoostModel = CatBoostModel
    return _CatBoostModel


def _import_lgbm():
    global _LGBMModel
    if _LGBMModel is None:
        from .lgbm_model import LGBMModel
        _LGBMModel = LGBMModel
    return _LGBMModel


def _import_xgb():
    global _XGBModel
    if _XGBModel is None:
        from .xgb_model import XGBModel
        _XGBModel = XGBModel
    return _XGBModel


def _import_nn():
    global _NNModel
    if _NNModel is None:
        from .nn import NNModel
        _NNModel = NNModel
    return _NNModel


__all__ = [
    'BaseModel',
    'LinearModel',
    'GlobalMeanBaseline',
    'FlatBaseline',
    'TrendBaseline',
    'get_model_class',
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
        'catboost': _import_catboost,
        'cat': _import_catboost,
        'lightgbm': _import_lgbm,
        'lgbm': _import_lgbm,
        'xgboost': _import_xgb,
        'xgb': _import_xgb,
        'linear': lambda: LinearModel,
        'nn': _import_nn,
        'neural': _import_nn,
        'global_mean': lambda: GlobalMeanBaseline,
        'flat': lambda: FlatBaseline,
        'trend': lambda: TrendBaseline,
    }
    
    name_lower = name.lower()
    if name_lower not in mapping:
        raise ValueError(f"Unknown model: {name}. Available: {list(mapping.keys())}")
    
    return mapping[name_lower]()
