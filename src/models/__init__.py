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

Specialized models:
- BaselineModels: Pure deterministic baselines (naive, linear/exp decay)
- HybridPhysicsMLModel: Physics-based decay + ML residual learning
- ARIHOWModel: ARIMA + Holt-Winters hybrid for time series

Deep Learning models (require PyTorch):
- KGGCNLSTMModel: Knowledge Graph GCN + LSTM (KG-GCN-LSTM paper)
- CNNLSTMModel: CNN-LSTM for drug sales prediction (Li et al. 2024)
- LSTMModel: Pure LSTM for ablation comparison

Ensemble models:
- AveragingEnsemble: Simple averaging of predictions
- WeightedAveragingEnsemble: Weighted average with optimizable weights
- StackingEnsemble: Two-level stacking with meta-learner
- BlendingEnsemble: Blending with holdout predictions
- EnsembleBlender: Lightweight prediction combiner for numpy arrays
"""

from .base import BaseModel
from .linear import (
    LinearModel,
    GlobalMeanBaseline,
    FlatBaseline,
    TrendBaseline,
    HistoricalCurveBaseline
)

# Specialized models (no native deps)
from .baselines import BaselineModels
from .hybrid_physics_ml import HybridPhysicsMLModel

# ARIHOW model (requires statsmodels, imported lazily)
_ARIHOWModel = None

def _import_arihow():
    global _ARIHOWModel
    if _ARIHOWModel is None:
        from .arihow import ARIHOWModel
        _ARIHOWModel = ARIHOWModel
    return _ARIHOWModel

# Ensemble models (pure Python, no native dependencies)
from .ensemble import (
    AveragingEnsemble,
    WeightedAveragingEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    EnsembleBlender,
    create_ensemble,
    optimize_ensemble_weights
)

# Lazy imports for models that require native libraries
# These will only fail when actually used, not on package import
_CatBoostModel = None
_LGBMModel = None
_XGBModel = None
_NNModel = None
_KGGCNLSTMModel = None
_CNNLSTMModel = None
_LSTMModel = None


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


def _import_kg_gcn_lstm():
    global _KGGCNLSTMModel
    if _KGGCNLSTMModel is None:
        from .kg_gcn_lstm import KGGCNLSTMModel
        _KGGCNLSTMModel = KGGCNLSTMModel
    return _KGGCNLSTMModel


def _import_cnn_lstm():
    global _CNNLSTMModel
    if _CNNLSTMModel is None:
        from .cnn_lstm import CNNLSTMModel
        _CNNLSTMModel = CNNLSTMModel
    return _CNNLSTMModel


def _import_lstm():
    global _LSTMModel
    if _LSTMModel is None:
        from .cnn_lstm import LSTMModel
        _LSTMModel = LSTMModel
    return _LSTMModel


__all__ = [
    # Base class
    'BaseModel',
    # Linear and baseline models
    'LinearModel',
    'GlobalMeanBaseline',
    'FlatBaseline',
    'TrendBaseline',
    'HistoricalCurveBaseline',
    # Specialized models
    'BaselineModels',
    'HybridPhysicsMLModel',
    'ARIHOWModel',  # Lazy import - requires statsmodels
    # Deep learning models (lazy import - require PyTorch)
    'KGGCNLSTMModel',  # KG-GCN-LSTM paper
    'CNNLSTMModel',  # Li et al. 2024
    'LSTMModel',  # Ablation model
    # Ensemble models
    'AveragingEnsemble',
    'WeightedAveragingEnsemble',
    'StackingEnsemble',
    'BlendingEnsemble',
    'EnsembleBlender',
    'create_ensemble',
    'optimize_ensemble_weights',
    # Factory function
    'get_model_class',
]


def __getattr__(name: str):
    """Lazy import for models with heavy dependencies."""
    if name == 'ARIHOWModel':
        return _import_arihow()
    elif name == 'CatBoostModel':
        return _import_catboost()
    elif name == 'LGBMModel':
        return _import_lgbm()
    elif name == 'XGBModel':
        return _import_xgb()
    elif name == 'NNModel':
        return _import_nn()
    elif name == 'KGGCNLSTMModel':
        return _import_kg_gcn_lstm()
    elif name == 'CNNLSTMModel':
        return _import_cnn_lstm()
    elif name == 'LSTMModel':
        return _import_lstm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_model_class(name: str):
    """
    Get model class by name.

    Args:
        name: Model name (case-insensitive). Options:
            - Tree boosters: 'catboost', 'cat', 'lightgbm', 'lgbm', 'xgboost', 'xgb'
            - Linear: 'linear', 'ridge', 'lasso', 'elasticnet', 'huber'
            - Neural: 'nn', 'neural', 'mlp'
            - Deep learning: 'kg_gcn_lstm', 'cnn_lstm', 'lstm'
            - Baselines: 'global_mean', 'flat', 'trend', 'historical_curve', 'knn_curve'
            - Specialized: 'hybrid_lgbm', 'hybrid_xgb', 'arihow'
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
        # Deep learning models (lazy import - require PyTorch)
        'kg_gcn_lstm': _import_kg_gcn_lstm,
        'kggcnlstm': _import_kg_gcn_lstm,
        'gcn_lstm': _import_kg_gcn_lstm,
        'cnn_lstm': _import_cnn_lstm,
        'cnnlstm': _import_cnn_lstm,
        'lstm': _import_lstm,
        'lstm_only': _import_lstm,
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
        # Deterministic baselines
        'baseline_naive': lambda: BaselineModels,
        'baseline_linear': lambda: BaselineModels,
        'baseline_exp': lambda: BaselineModels,
        'baseline': lambda: BaselineModels,
        # Specialized models
        'hybrid_lgbm': lambda: HybridPhysicsMLModel,
        'hybrid_xgb': lambda: HybridPhysicsMLModel,
        'hybrid': lambda: HybridPhysicsMLModel,
        'arihow': _import_arihow,
        'ts_hybrid': _import_arihow,
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
        'blender': lambda: EnsembleBlender,
    }

    name_lower = name.lower()
    if name_lower not in mapping:
        available = sorted(set(mapping.keys()))
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return mapping[name_lower]()
