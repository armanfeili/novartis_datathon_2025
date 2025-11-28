"""
Model implementations for Novartis Datathon 2025.

All models follow the BaseModel interface:
- fit(X_train, y_train, X_val, y_val, sample_weight)
- predict(X)
- save(path)
- load(path)
- get_feature_importance()

Available models:
- BaseModel: Abstract base class
- LinearModel: Ridge/Lasso/ElasticNet/Huber with polynomial features support
- GlobalMeanBaseline: Predict global average erosion curve
- FlatBaseline: Predict no erosion (y_norm = 1.0)
- TrendBaseline: Extrapolate pre-entry trend
- HistoricalCurveBaseline: KNN-based matching to similar historical series
- CatBoostModel: CatBoost with native categorical support
- LGBMModel: LightGBM fast gradient boosting
- XGBModel: XGBoost gradient boosting
- NNModel: Simple MLP neural network

Ensemble models:
- AveragingEnsemble: Simple averaging of predictions
- WeightedAveragingEnsemble: Weighted average with optimizable weights
- StackingEnsemble: Two-level stacking with meta-learner
- BlendingEnsemble: Blending with holdout predictions
"""

from .base import BaseModel
from .linear import (
    LinearModel,
    GlobalMeanBaseline,
    FlatBaseline,
    TrendBaseline,
    HistoricalCurveBaseline
)

# Ensemble models (pure Python, no native dependencies)
from .ensemble import (
    AveragingEnsemble,
    WeightedAveragingEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    create_ensemble
)

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
    # Base class
    'BaseModel',
    # Linear and baseline models
    'LinearModel',
    'GlobalMeanBaseline',
    'FlatBaseline',
    'TrendBaseline',
    'HistoricalCurveBaseline',
    # Ensemble models
    'AveragingEnsemble',
    'WeightedAveragingEnsemble',
    'StackingEnsemble',
    'BlendingEnsemble',
    'create_ensemble',
    # Factory function
    'get_model_class',
]


def get_model_class(name: str):
    """
    Get model class by name.

    Args:
        name: Model name (case-insensitive). Options:
            - Tree boosters: 'catboost', 'cat', 'lightgbm', 'lgbm', 'xgboost', 'xgb'
            - Linear: 'linear', 'ridge', 'lasso', 'elasticnet', 'huber'
            - Neural: 'nn', 'neural', 'mlp'
            - Baselines: 'global_mean', 'flat', 'trend', 'historical_curve', 'knn_curve'
            - Ensembles: 'averaging', 'weighted', 'stacking', 'blending'

    Returns:
        Model class

    Raises:
        ValueError: If model name is not recognized
    """
    mapping = {
        # Tree boosters (lazy import)
        'catboost': _import_catboost,
        'cat': _import_catboost,
        'lightgbm': _import_lgbm,
        'lgbm': _import_lgbm,
        'xgboost': _import_xgb,
        'xgb': _import_xgb,
        # Neural network (lazy import)
        'nn': _import_nn,
        'neural': _import_nn,
        'mlp': _import_nn,
        # Linear models (eager import - no native deps)
        'linear': lambda: LinearModel,
        'ridge': lambda: LinearModel,
        'lasso': lambda: LinearModel,
        'elasticnet': lambda: LinearModel,
        'huber': lambda: LinearModel,
        # Baselines (eager import)
        'global_mean': lambda: GlobalMeanBaseline,
        'flat': lambda: FlatBaseline,
        'trend': lambda: TrendBaseline,
        'historical_curve': lambda: HistoricalCurveBaseline,
        'knn_curve': lambda: HistoricalCurveBaseline,
        # Ensembles (eager import)
        'averaging': lambda: AveragingEnsemble,
        'averaging_ensemble': lambda: AveragingEnsemble,
        'weighted': lambda: WeightedAveragingEnsemble,
        'weighted_averaging': lambda: WeightedAveragingEnsemble,
        'weighted_ensemble': lambda: WeightedAveragingEnsemble,
        'stacking': lambda: StackingEnsemble,
        'stacking_ensemble': lambda: StackingEnsemble,
        'blending': lambda: BlendingEnsemble,
        'blending_ensemble': lambda: BlendingEnsemble,
    }

    name_lower = name.lower()
    if name_lower not in mapping:
        available = sorted(set(mapping.keys()))
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return mapping[name_lower]()
