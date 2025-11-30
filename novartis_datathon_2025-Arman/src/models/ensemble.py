"""
Ensemble model implementations for Novartis Datathon 2025.

Implements:
- AveragingEnsemble: Simple averaging of predictions
- WeightedAveragingEnsemble: Weighted average with tunable weights
- StackingEnsemble: Two-level stacking with meta-learner
- BlendingEnsemble: Blending with holdout predictions

All ensembles follow the BaseModel interface.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

from .base import BaseModel

logger = logging.getLogger(__name__)


class AveragingEnsemble(BaseModel):
    """
    Simple averaging ensemble that averages predictions from multiple models.
    
    This is the simplest ensemble method - just takes the mean of all model predictions.
    
    Example usage:
        models = [catboost_model, lgbm_model, xgb_model]
        ensemble = AveragingEnsemble({'models': models})
        ensemble.fit(X_train, y_train, X_val, y_val)  # Already fitted models
        predictions = ensemble.predict(X_test)
    """
    
    def __init__(self, config: dict):
        """
        Initialize averaging ensemble.
        
        Args:
            config: Configuration dict with:
                - models: List of pre-trained model instances (optional, can be added later)
                - clip_predictions: Whether to clip predictions to [0, 2] (default: True)
        """
        super().__init__(config)
        
        self.models: List[BaseModel] = config.get('models', [])
        self.clip_predictions = config.get('clip_predictions', True)
        self.clip_min = config.get('clip_min', 0.0)
        self.clip_max = config.get('clip_max', 2.0)
    
    def add_model(self, model: BaseModel) -> 'AveragingEnsemble':
        """Add a model to the ensemble."""
        self.models.append(model)
        return self
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'AveragingEnsemble':
        """
        Fit the ensemble.
        
        If models are already fitted (which is the typical case for ensembles),
        this just stores feature names. Otherwise, trains each model.
        """
        self.feature_names = list(X_train.columns)
        
        # If models have no trained state, fit them
        for i, model in enumerate(self.models):
            if model.model is None:
                logger.info(f"Training model {i+1}/{len(self.models)}")
                model.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        logger.info(f"AveragingEnsemble: {len(self.models)} models ready")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions by averaging all model predictions."""
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models first.")
        
        predictions = []
        for model in self.models:
            preds = model.predict(X)
            predictions.append(preds)
        
        # Average predictions
        avg_preds = np.mean(predictions, axis=0)
        
        # Clip if requested
        if self.clip_predictions:
            avg_preds = np.clip(avg_preds, self.clip_min, self.clip_max)
        
        return avg_preds
    
    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        # Save each model to a subdirectory structure
        joblib.dump({
            'n_models': len(self.models),
            'clip_predictions': self.clip_predictions,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'feature_names': self.feature_names,
            'models': self.models,  # joblib handles nested objects
        }, path)
        logger.info(f"AveragingEnsemble saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AveragingEnsemble':
        """Load ensemble from disk."""
        data = joblib.load(path)
        
        config = {
            'clip_predictions': data.get('clip_predictions', True),
            'clip_min': data.get('clip_min', 0.0),
            'clip_max': data.get('clip_max', 2.0),
        }
        instance = cls(config)
        instance.feature_names = data.get('feature_names', [])
        instance.models = data.get('models', [])
        
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get average feature importance across all models."""
        if len(self.models) == 0:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        all_importances = []
        for model in self.models:
            imp = model.get_feature_importance()
            if len(imp) > 0:
                all_importances.append(imp.set_index('feature')['importance'])
        
        if not all_importances:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Average importances
        combined = pd.concat(all_importances, axis=1)
        avg_importance = combined.mean(axis=1)
        
        return pd.DataFrame({
            'feature': avg_importance.index,
            'importance': avg_importance.values
        }).sort_values('importance', ascending=False).reset_index(drop=True)


class WeightedAveragingEnsemble(BaseModel):
    """
    Weighted averaging ensemble with optimizable weights.
    
    Weights can be:
    1. Specified manually
    2. Optimized on validation data to minimize MSE
    3. Optimized to minimize official metric (if compute_metric_fn provided)
    
    Example usage:
        models = [catboost_model, lgbm_model, xgb_model]
        ensemble = WeightedAveragingEnsemble({
            'models': models,
            'optimize_weights': True
        })
        ensemble.fit(X_train, y_train, X_val, y_val)
        predictions = ensemble.predict(X_test)
    """
    
    def __init__(self, config: dict):
        """
        Initialize weighted averaging ensemble.
        
        Args:
            config: Configuration dict with:
                - models: List of pre-trained model instances
                - weights: Optional list of weights (must sum to 1)
                - optimize_weights: If True, optimize weights on validation data
                - optimization_metric: 'mse' or 'mae' (default: 'mse')
                - clip_predictions: Whether to clip predictions (default: True)
        """
        super().__init__(config)
        
        self.models: List[BaseModel] = config.get('models', [])
        self.weights: Optional[np.ndarray] = None
        
        if 'weights' in config:
            self.weights = np.array(config['weights'])
            # Normalize weights to sum to 1
            self.weights = self.weights / self.weights.sum()
        
        self.optimize_weights = config.get('optimize_weights', False)
        self.optimization_metric = config.get('optimization_metric', 'mse')
        self.clip_predictions = config.get('clip_predictions', True)
        self.clip_min = config.get('clip_min', 0.0)
        self.clip_max = config.get('clip_max', 2.0)
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> 'WeightedAveragingEnsemble':
        """Add a model with an initial weight."""
        self.models.append(model)
        
        if self.weights is None:
            self.weights = np.array([weight])
        else:
            self.weights = np.append(self.weights, weight)
            # Re-normalize
            self.weights = self.weights / self.weights.sum()
        
        return self
    
    def _optimize_weights_mse(
        self, 
        predictions: List[np.ndarray], 
        y_true: np.ndarray
    ) -> np.ndarray:
        """
        Optimize weights to minimize MSE using constrained optimization.
        
        Weights are constrained to be non-negative and sum to 1.
        """
        n_models = len(predictions)
        
        # Stack predictions
        pred_matrix = np.column_stack(predictions)
        
        def objective(weights):
            weighted_pred = pred_matrix @ weights
            return np.mean((weighted_pred - y_true) ** 2)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights >= 0
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial weights (uniform)
        w0 = np.ones(n_models) / n_models
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            logger.warning(f"Weight optimization failed: {result.message}. Using uniform weights.")
            return w0
    
    def _optimize_weights_mae(
        self, 
        predictions: List[np.ndarray], 
        y_true: np.ndarray
    ) -> np.ndarray:
        """Optimize weights to minimize MAE."""
        n_models = len(predictions)
        pred_matrix = np.column_stack(predictions)
        
        def objective(weights):
            weighted_pred = pred_matrix @ weights
            return np.mean(np.abs(weighted_pred - y_true))
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        w0 = np.ones(n_models) / n_models
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else w0
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'WeightedAveragingEnsemble':
        """
        Fit the ensemble, optionally optimizing weights on validation data.
        """
        self.feature_names = list(X_train.columns)
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models first.")
        
        # Train models if not already fitted
        for i, model in enumerate(self.models):
            if model.model is None:
                logger.info(f"Training model {i+1}/{len(self.models)}")
                model.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        # Optimize weights if requested and validation data is available
        if self.optimize_weights and X_val is not None and y_val is not None:
            logger.info("Optimizing ensemble weights on validation data...")
            
            # Get predictions from all models on validation set
            val_predictions = [model.predict(X_val) for model in self.models]
            
            if self.optimization_metric == 'mae':
                self.weights = self._optimize_weights_mae(val_predictions, y_val.values)
            else:
                self.weights = self._optimize_weights_mse(val_predictions, y_val.values)
            
            logger.info(f"Optimized weights: {self.weights}")
        
        # If weights still not set, use uniform
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        logger.info(f"WeightedAveragingEnsemble: {len(self.models)} models, "
                   f"weights={self.weights.round(3)}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted average predictions."""
        if len(self.models) == 0:
            raise ValueError("No models in ensemble.")
        
        predictions = []
        for model in self.models:
            preds = model.predict(X)
            predictions.append(preds)
        
        # Weighted average
        pred_matrix = np.column_stack(predictions)
        weighted_preds = pred_matrix @ self.weights
        
        if self.clip_predictions:
            weighted_preds = np.clip(weighted_preds, self.clip_min, self.clip_max)
        
        return weighted_preds
    
    def get_weights(self) -> Dict[int, float]:
        """Get current model weights."""
        return {i: w for i, w in enumerate(self.weights)} if self.weights is not None else {}
    
    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        joblib.dump({
            'models': self.models,
            'weights': self.weights,
            'optimize_weights': self.optimize_weights,
            'optimization_metric': self.optimization_metric,
            'clip_predictions': self.clip_predictions,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'feature_names': self.feature_names,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'WeightedAveragingEnsemble':
        """Load ensemble from disk."""
        data = joblib.load(path)
        
        config = {
            'optimize_weights': data.get('optimize_weights', False),
            'optimization_metric': data.get('optimization_metric', 'mse'),
            'clip_predictions': data.get('clip_predictions', True),
            'clip_min': data.get('clip_min', 0.0),
            'clip_max': data.get('clip_max', 2.0),
        }
        instance = cls(config)
        instance.models = data.get('models', [])
        instance.weights = data.get('weights')
        instance.feature_names = data.get('feature_names', [])
        
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get weighted average feature importance."""
        if len(self.models) == 0 or self.weights is None:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        all_importances = []
        for model, weight in zip(self.models, self.weights):
            imp = model.get_feature_importance()
            if len(imp) > 0:
                imp_series = imp.set_index('feature')['importance'] * weight
                all_importances.append(imp_series)
        
        if not all_importances:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        combined = pd.concat(all_importances, axis=1)
        weighted_importance = combined.sum(axis=1)
        
        return pd.DataFrame({
            'feature': weighted_importance.index,
            'importance': weighted_importance.values
        }).sort_values('importance', ascending=False).reset_index(drop=True)


class StackingEnsemble(BaseModel):
    """
    Two-level stacking ensemble with a meta-learner.
    
    Level 1: Base models make predictions (out-of-fold for training)
    Level 2: Meta-learner combines base model predictions
    
    This implementation uses cross-validation to generate OOF predictions
    for training the meta-learner, avoiding overfitting.
    
    Example usage:
        base_models = [
            ('catboost', CatBoostModel(config1)),
            ('lgbm', LGBMModel(config2)),
            ('xgb', XGBModel(config3)),
        ]
        ensemble = StackingEnsemble({
            'base_models': base_models,
            'meta_learner': Ridge(alpha=1.0),
            'n_folds': 5
        })
        ensemble.fit(X_train, y_train, X_val, y_val)
    """
    
    def __init__(self, config: dict):
        """
        Initialize stacking ensemble.
        
        Args:
            config: Configuration dict with:
                - base_models: List of (name, model) tuples
                - meta_learner: Sklearn regressor for level 2 (default: Ridge)
                - n_folds: Number of CV folds for OOF predictions (default: 5)
                - use_original_features: Include original features in meta-learner (default: False)
                - clip_predictions: Whether to clip final predictions (default: True)
        """
        super().__init__(config)
        
        self.base_models: List[Tuple[str, BaseModel]] = config.get('base_models', [])
        self.meta_learner = config.get('meta_learner', Ridge(alpha=1.0))
        self.n_folds = config.get('n_folds', 5)
        self.use_original_features = config.get('use_original_features', False)
        self.clip_predictions = config.get('clip_predictions', True)
        self.clip_min = config.get('clip_min', 0.0)
        self.clip_max = config.get('clip_max', 2.0)
        
        # Fitted models for each fold (for OOF training)
        self._fitted_base_models: List[List[BaseModel]] = []
        # Single fitted model per base for test prediction
        self._final_base_models: List[BaseModel] = []
    
    def add_base_model(self, name: str, model: BaseModel) -> 'StackingEnsemble':
        """Add a base model to the ensemble."""
        self.base_models.append((name, model))
        return self
    
    def _get_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Generate out-of-fold predictions for training meta-learner.
        
        Uses K-fold CV to ensure no information leakage.
        """
        from sklearn.model_selection import KFold
        
        n_samples = len(X)
        n_models = len(self.base_models)
        
        oof_predictions = np.zeros((n_samples, n_models))
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        self._fitted_base_models = [[] for _ in range(n_models)]
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            
            sw_fold = None
            if sample_weight is not None:
                sw_fold = sample_weight.iloc[train_idx]
            
            for model_idx, (name, base_model) in enumerate(self.base_models):
                # Clone and train model
                model_copy = base_model.__class__(base_model.config)
                model_copy.fit(X_fold_train, y_fold_train, sample_weight=sw_fold)
                
                # Store for later if needed
                self._fitted_base_models[model_idx].append(model_copy)
                
                # Get OOF predictions
                oof_predictions[val_idx, model_idx] = model_copy.predict(X_fold_val)
            
            logger.info(f"  Fold {fold_idx + 1}/{self.n_folds} complete")
        
        return oof_predictions
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        1. Generate OOF predictions from base models using CV
        2. Train meta-learner on OOF predictions
        3. Train final base models on full training data (for test prediction)
        """
        self.feature_names = list(X_train.columns)
        
        if len(self.base_models) == 0:
            raise ValueError("No base models. Add models with add_base_model().")
        
        logger.info(f"Fitting StackingEnsemble with {len(self.base_models)} base models")
        
        # Step 1: Generate OOF predictions
        logger.info("Generating out-of-fold predictions...")
        oof_predictions = self._get_oof_predictions(X_train, y_train, sample_weight)
        
        # Step 2: Prepare meta-features
        if self.use_original_features:
            meta_features = np.hstack([oof_predictions, X_train.values])
        else:
            meta_features = oof_predictions
        
        # Step 3: Train meta-learner
        logger.info("Training meta-learner...")
        self.meta_learner.fit(meta_features, y_train)
        
        # Step 4: Train final base models on full data
        logger.info("Training final base models on full data...")
        self._final_base_models = []
        for name, base_model in self.base_models:
            model_copy = base_model.__class__(base_model.config)
            model_copy.fit(X_train, y_train, X_val, y_val, sample_weight)
            self._final_base_models.append(model_copy)
        
        logger.info("StackingEnsemble fitting complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the stacking ensemble."""
        if len(self._final_base_models) == 0:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from all base models
        base_predictions = []
        for model in self._final_base_models:
            base_predictions.append(model.predict(X))
        
        # Stack predictions
        meta_features = np.column_stack(base_predictions)
        
        # Add original features if configured
        if self.use_original_features:
            meta_features = np.hstack([meta_features, X.values])
        
        # Get meta-learner predictions
        predictions = self.meta_learner.predict(meta_features)
        
        if self.clip_predictions:
            predictions = np.clip(predictions, self.clip_min, self.clip_max)
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        # Serialize base model configs for reconstruction
        base_model_configs = [
            (name, model.__class__.__name__, model.config)
            for name, model in self.base_models
        ]
        
        joblib.dump({
            'base_model_configs': base_model_configs,
            '_final_base_models': self._final_base_models,
            'meta_learner': self.meta_learner,
            'n_folds': self.n_folds,
            'use_original_features': self.use_original_features,
            'clip_predictions': self.clip_predictions,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'feature_names': self.feature_names,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'StackingEnsemble':
        """Load ensemble from disk."""
        data = joblib.load(path)
        
        config = {
            'n_folds': data.get('n_folds', 5),
            'use_original_features': data.get('use_original_features', False),
            'clip_predictions': data.get('clip_predictions', True),
            'clip_min': data.get('clip_min', 0.0),
            'clip_max': data.get('clip_max', 2.0),
            'meta_learner': data.get('meta_learner', Ridge(alpha=1.0)),
        }
        instance = cls(config)
        instance._final_base_models = data.get('_final_base_models', [])
        instance.feature_names = data.get('feature_names', [])
        
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (meta-learner coefficients if available)."""
        if hasattr(self.meta_learner, 'coef_'):
            n_base = len(self.base_models)
            coefs = self.meta_learner.coef_
            
            # Base model importance
            importance_data = []
            for i, (name, _) in enumerate(self.base_models):
                if i < len(coefs):
                    importance_data.append({
                        'feature': f'{name}_prediction',
                        'importance': abs(coefs[i])
                    })
            
            return pd.DataFrame(importance_data).sort_values(
                'importance', ascending=False
            ).reset_index(drop=True)
        
        return pd.DataFrame(columns=['feature', 'importance'])


class BlendingEnsemble(BaseModel):
    """
    Blending ensemble using holdout data for meta-learner training.
    
    Unlike stacking, blending uses a fixed holdout set rather than CV
    for generating meta-learner training data. This is simpler but uses
    less data for training base models.
    
    Example usage:
        models = [catboost_model, lgbm_model, xgb_model]
        ensemble = BlendingEnsemble({
            'models': models,
            'holdout_fraction': 0.2
        })
        ensemble.fit(X_train, y_train)
    """
    
    def __init__(self, config: dict):
        """
        Initialize blending ensemble.
        
        Args:
            config: Configuration dict with:
                - models: List of base model instances
                - meta_learner: Sklearn regressor (default: Ridge)
                - holdout_fraction: Fraction of data for blending (default: 0.2)
                - clip_predictions: Whether to clip predictions (default: True)
        """
        super().__init__(config)
        
        self.models: List[BaseModel] = config.get('models', [])
        self.meta_learner = config.get('meta_learner', Ridge(alpha=1.0))
        self.holdout_fraction = config.get('holdout_fraction', 0.2)
        self.clip_predictions = config.get('clip_predictions', True)
        self.clip_min = config.get('clip_min', 0.0)
        self.clip_max = config.get('clip_max', 2.0)
        
        self._fitted_models: List[BaseModel] = []
    
    def add_model(self, model: BaseModel) -> 'BlendingEnsemble':
        """Add a model to the ensemble."""
        self.models.append(model)
        return self
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'BlendingEnsemble':
        """
        Fit the blending ensemble.
        
        1. Split training data into base train and holdout
        2. Train base models on base train
        3. Get predictions on holdout
        4. Train meta-learner on holdout predictions
        """
        self.feature_names = list(X_train.columns)
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models first.")
        
        logger.info(f"Fitting BlendingEnsemble with {len(self.models)} base models")
        
        # Split data
        n_holdout = int(len(X_train) * self.holdout_fraction)
        indices = np.random.permutation(len(X_train))
        
        holdout_idx = indices[:n_holdout]
        base_idx = indices[n_holdout:]
        
        X_base = X_train.iloc[base_idx]
        y_base = y_train.iloc[base_idx]
        X_holdout = X_train.iloc[holdout_idx]
        y_holdout = y_train.iloc[holdout_idx]
        
        sw_base = None
        if sample_weight is not None:
            sw_base = sample_weight.iloc[base_idx]
        
        # Train base models
        self._fitted_models = []
        holdout_predictions = []
        
        for i, model in enumerate(self.models):
            logger.info(f"Training base model {i+1}/{len(self.models)}")
            
            # Clone and train
            model_copy = model.__class__(model.config)
            model_copy.fit(X_base, y_base, sample_weight=sw_base)
            self._fitted_models.append(model_copy)
            
            # Get holdout predictions
            holdout_predictions.append(model_copy.predict(X_holdout))
        
        # Stack holdout predictions
        meta_features = np.column_stack(holdout_predictions)
        
        # Train meta-learner
        logger.info("Training meta-learner on holdout predictions...")
        self.meta_learner.fit(meta_features, y_holdout)
        
        # Re-train base models on full training data for final predictions
        logger.info("Re-training base models on full data...")
        self._fitted_models = []
        for i, model in enumerate(self.models):
            model_copy = model.__class__(model.config)
            model_copy.fit(X_train, y_train, X_val, y_val, sample_weight)
            self._fitted_models.append(model_copy)
        
        logger.info("BlendingEnsemble fitting complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate blended predictions."""
        if len(self._fitted_models) == 0:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from all models
        base_predictions = []
        for model in self._fitted_models:
            base_predictions.append(model.predict(X))
        
        # Stack and get meta-learner predictions
        meta_features = np.column_stack(base_predictions)
        predictions = self.meta_learner.predict(meta_features)
        
        if self.clip_predictions:
            predictions = np.clip(predictions, self.clip_min, self.clip_max)
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        joblib.dump({
            '_fitted_models': self._fitted_models,
            'meta_learner': self.meta_learner,
            'holdout_fraction': self.holdout_fraction,
            'clip_predictions': self.clip_predictions,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'feature_names': self.feature_names,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'BlendingEnsemble':
        """Load ensemble from disk."""
        data = joblib.load(path)
        
        config = {
            'holdout_fraction': data.get('holdout_fraction', 0.2),
            'clip_predictions': data.get('clip_predictions', True),
            'clip_min': data.get('clip_min', 0.0),
            'clip_max': data.get('clip_max', 2.0),
            'meta_learner': data.get('meta_learner', Ridge(alpha=1.0)),
        }
        instance = cls(config)
        instance._fitted_models = data.get('_fitted_models', [])
        instance.feature_names = data.get('feature_names', [])
        
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get meta-learner coefficients as importance."""
        if hasattr(self.meta_learner, 'coef_'):
            coefs = self.meta_learner.coef_
            
            importance_data = []
            for i, coef in enumerate(coefs):
                importance_data.append({
                    'feature': f'model_{i}_prediction',
                    'importance': abs(coef)
                })
            
            return pd.DataFrame(importance_data).sort_values(
                'importance', ascending=False
            ).reset_index(drop=True)
        
        return pd.DataFrame(columns=['feature', 'importance'])


def create_ensemble(
    models: List[BaseModel],
    method: str = 'averaging',
    **kwargs
) -> BaseModel:
    """
    Factory function to create an ensemble of the specified type.
    
    Args:
        models: List of fitted or unfitted model instances
        method: Ensemble method - 'averaging', 'weighted', 'stacking', 'blending'
        **kwargs: Additional configuration for the ensemble
        
    Returns:
        Ensemble model instance
    """
    config = {'models': models, **kwargs}
    
    if method == 'averaging':
        return AveragingEnsemble(config)
    elif method == 'weighted':
        return WeightedAveragingEnsemble(config)
    elif method == 'stacking':
        # Convert models to (name, model) tuples for stacking
        base_models = [(f'model_{i}', m) for i, m in enumerate(models)]
        config['base_models'] = base_models
        del config['models']
        return StackingEnsemble(config)
    elif method == 'blending':
        return BlendingEnsemble(config)
    else:
        raise ValueError(f"Unknown ensemble method: {method}. "
                        f"Available: averaging, weighted, stacking, blending")


# ==============================================================================
# ENSEMBLE BLENDER - Lightweight prediction combiner
# ==============================================================================

class EnsembleBlender:
    """
    Lightweight ensemble blender for combining prediction arrays.
    
    Unlike the BaseModel-based ensembles above, this class works directly
    with numpy arrays of predictions, making it suitable for:
    - Combining predictions from heterogeneous model types
    - Post-hoc blending of already-generated predictions
    - Quick experimentation with ensemble weights
    
    Example usage:
        blender = EnsembleBlender(constrain_weights=True)
        blender.fit(
            predictions={'catboost': preds_cat, 'lgbm': preds_lgbm, 'xgb': preds_xgb},
            y_true=y_val
        )
        blended = blender.predict({'catboost': preds_cat_test, ...})
        print(blender.get_weights())
    """
    
    def __init__(self, constrain_weights: bool = True):
        """
        Initialize ensemble blender.
        
        Args:
            constrain_weights: If True, weights are constrained to sum to 1
        """
        self.constrain_weights = constrain_weights
        self.model_names: List[str] = []
        self.weights: Optional[np.ndarray] = None
        self.is_fitted: bool = False
    
    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> 'EnsembleBlender':
        """
        Learn optimal ensemble weights via constrained optimization.
        
        Minimizes weighted MSE subject to:
        - weights >= 0
        - sum(weights) == 1 (if constrain_weights=True)
        
        Args:
            predictions: Dict mapping model names to prediction arrays
            y_true: True target values
            sample_weights: Optional per-sample weights
            
        Returns:
            self (fitted blender)
        """
        self.model_names = list(predictions.keys())
        n_models = len(self.model_names)
        
        if n_models == 0:
            raise ValueError("No predictions provided")
        
        # Stack predictions into matrix
        pred_matrix = np.column_stack([predictions[name] for name in self.model_names])
        
        # Define objective function
        def objective(weights):
            weighted_pred = pred_matrix @ weights
            if sample_weights is not None:
                return np.average((weighted_pred - y_true) ** 2, weights=sample_weights)
            return np.mean((weighted_pred - y_true) ** 2)
        
        # Bounds: weights >= 0
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Constraints
        constraints = []
        if self.constrain_weights:
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Initial weights (uniform)
        w0 = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints if constraints else None
        )
        
        if result.success:
            self.weights = result.x
        else:
            logger.warning(f"Weight optimization failed: {result.message}. Using uniform weights.")
            self.weights = w0
        
        self.is_fitted = True
        
        # Log metrics
        blended_pred = pred_matrix @ self.weights
        if sample_weights is not None:
            rmse = np.sqrt(np.average((blended_pred - y_true) ** 2, weights=sample_weights))
            mae = np.average(np.abs(blended_pred - y_true), weights=sample_weights)
        else:
            rmse = np.sqrt(np.mean((blended_pred - y_true) ** 2))
            mae = np.mean(np.abs(blended_pred - y_true))
        
        logger.info(f"Ensemble blender fitted: RMSE={rmse:.4f}, MAE={mae:.4f}")
        logger.info(f"Weights: {dict(zip(self.model_names, self.weights.round(4)))}")
        
        # Also compute per-model RMSE
        for i, name in enumerate(self.model_names):
            solo_rmse = np.sqrt(np.mean((pred_matrix[:, i] - y_true) ** 2))
            logger.info(f"  {name} solo RMSE: {solo_rmse:.4f}")
        
        return self
    
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate blended predictions using learned weights.
        
        Args:
            predictions: Dict mapping model names to prediction arrays
            
        Returns:
            Blended prediction array
        """
        if not self.is_fitted:
            raise ValueError("Blender not fitted. Call fit() first.")
        
        # Ensure same models in same order
        missing = set(self.model_names) - set(predictions.keys())
        if missing:
            raise ValueError(f"Missing predictions for models: {missing}")
        
        pred_matrix = np.column_stack([predictions[name] for name in self.model_names])
        return pred_matrix @ self.weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get learned weights as a dictionary."""
        if not self.is_fitted:
            return {}
        return dict(zip(self.model_names, self.weights.tolist()))


def optimize_ensemble_weights(
    val_df: pd.DataFrame,
    model_predictions: Dict[str, np.ndarray],
    target_col: str = 'volume',
    sample_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Convenience function to optimize ensemble weights on validation data.
    
    Args:
        val_df: Validation DataFrame containing target column
        model_predictions: Dict mapping model names to prediction arrays
        target_col: Name of target column in val_df
        sample_weights: Optional per-sample weights
        
    Returns:
        Dict with:
        - 'weights': Dict[str, float] of model weights
        - 'blended_predictions': np.ndarray of blended predictions
        - 'blender': Fitted EnsembleBlender instance
    """
    y_true = val_df[target_col].values
    
    blender = EnsembleBlender(constrain_weights=True)
    blender.fit(model_predictions, y_true, sample_weights)
    
    blended_preds = blender.predict(model_predictions)
    
    return {
        'weights': blender.get_weights(),
        'blended_predictions': blended_preds,
        'blender': blender
    }
