"""
Hybrid Physics + ML model for Novartis Datathon 2025.

Combines a physics-based exponential decay baseline with ML residual learning.
The physics component captures the expected erosion pattern, while the ML
component learns deviations from this baseline.

This approach leverages domain knowledge about pharmaceutical erosion
while allowing the model to learn complex patterns in the residuals.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging

import numpy as np
import pandas as pd

from .base import BaseModel

logger = logging.getLogger(__name__)


class HybridPhysicsMLModel:
    """
    Hybrid model combining physics-based decay with ML residual learning.
    
    Architecture:
    1. Physics baseline: volume_physics = avg_vol * exp(-decay_rate * months_postgx)
    2. ML residual: residual = actual - physics_baseline
    3. Final prediction: volume = physics_baseline + ML_pred(residual)
    
    Benefits:
    - Physics baseline provides reasonable predictions even with limited data
    - ML component focuses on learning systematic deviations
    - More interpretable than pure ML approach
    - Often better extrapolation to unseen scenarios
    
    Example usage:
        model = HybridPhysicsMLModel(ml_model_type='lightgbm', decay_rate=0.05)
        model.fit(
            X_train, y_train,
            avg_vol_train=avg_vol_train,
            months_train=months_train,
            X_val=X_val, y_val=y_val,
            avg_vol_val=avg_vol_val,
            months_val=months_val
        )
        predictions = model.predict(X_test, avg_vol_test, months_test)
    """
    
    def __init__(
        self,
        ml_model_type: str = 'lightgbm',
        decay_rate: float = 0.05,
        params: Optional[Dict[str, Any]] = None,
        clip_predictions: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 2.0
    ):
        """
        Initialize hybrid model.
        
        Args:
            ml_model_type: 'lightgbm' or 'xgboost' for residual learning
            decay_rate: Exponential decay rate for physics baseline
            params: Override parameters for ML model (optional)
            clip_predictions: Whether to clip final predictions
            clip_min: Minimum prediction value
            clip_max: Maximum prediction value
        """
        self.ml_model_type = ml_model_type.lower()
        self.decay_rate = decay_rate
        self.params = params or self._default_params()
        self.clip_predictions = clip_predictions
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        self.ml_model = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        
        # Training statistics for debugging
        self._train_stats: Dict[str, float] = {}
    
    def _default_params(self) -> Dict[str, Any]:
        """Get conservative default parameters for the ML model."""
        if self.ml_model_type == 'lightgbm':
            return {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': -1,
                'random_state': 42
            }
        elif self.ml_model_type == 'xgboost':
            return {
                'booster': 'gbtree',
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 1,
                'verbosity': 0,
                'n_jobs': -1,
                'random_state': 42
            }
        elif self.ml_model_type == 'catboost':
            return {
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6,
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 50
            }
        else:
            raise ValueError(f"Unknown ml_model_type: {self.ml_model_type}")
    
    def _compute_physics_baseline(
        self,
        avg_vol: np.ndarray,
        months_postgx: np.ndarray
    ) -> np.ndarray:
        """
        Compute physics-based exponential decay baseline.
        
        Args:
            avg_vol: Pre-LOE average volume for each sample
            months_postgx: Months since generic entry for each sample
            
        Returns:
            Physics baseline predictions (normalized by avg_vol if y is normalized)
        """
        # Exponential decay: y_norm = exp(-decay_rate * months)
        # Note: We predict y_norm (normalized volume), not absolute volume
        return np.exp(-self.decay_rate * months_postgx)
    
    def _create_ml_model(self):
        """Create the ML model instance."""
        if self.ml_model_type == 'lightgbm':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self.params)
        elif self.ml_model_type == 'xgboost':
            from xgboost import XGBRegressor
            return XGBRegressor(**self.params)
        elif self.ml_model_type == 'catboost':
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**self.params)
        else:
            raise ValueError(f"Unknown ml_model_type: {self.ml_model_type}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        avg_vol_train: np.ndarray,
        months_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        avg_vol_val: Optional[np.ndarray] = None,
        months_val: Optional[np.ndarray] = None,
        sample_weight_train: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50
    ) -> 'HybridPhysicsMLModel':
        """
        Fit the hybrid model.
        
        Three-step process:
        1. Compute physics baseline for training data
        2. Calculate residuals (actual - physics)
        3. Train ML model on residuals
        
        Args:
            X_train: Feature matrix for training
            y_train: Target values (normalized volume, y_norm)
            avg_vol_train: Pre-LOE average volume for each training sample
            months_train: Months post-GX for each training sample
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets
            avg_vol_val: Validation avg volumes
            months_val: Validation months
            sample_weight_train: Sample weights for training
            sample_weight_val: Sample weights for validation
            early_stopping_rounds: Early stopping patience
            
        Returns:
            self (fitted model)
        """
        self.feature_names = list(X_train.columns)
        
        # Convert to numpy if needed
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(months_train, pd.Series):
            months_train = months_train.values
        if isinstance(avg_vol_train, pd.Series):
            avg_vol_train = avg_vol_train.values
        
        # Step 1: Compute physics baseline
        physics_pred_train = self._compute_physics_baseline(avg_vol_train, months_train)
        
        # Step 2: Calculate residuals
        residuals_train = y_train - physics_pred_train
        
        # Store training statistics for debugging
        self._train_stats = {
            'physics_rmse': np.sqrt(np.mean((physics_pred_train - y_train) ** 2)),
            'physics_mae': np.mean(np.abs(physics_pred_train - y_train)),
            'physics_mean': np.mean(physics_pred_train),
            'residual_mean': np.mean(residuals_train),
            'residual_std': np.std(residuals_train),
            'n_train': len(y_train)
        }
        
        logger.info(f"Physics baseline RMSE: {self._train_stats['physics_rmse']:.4f}")
        logger.info(f"Residual mean: {self._train_stats['residual_mean']:.4f}, "
                   f"std: {self._train_stats['residual_std']:.4f}")
        
        # Step 3: Train ML model on residuals
        self.ml_model = self._create_ml_model()
        
        fit_params = {}
        
        if sample_weight_train is not None:
            fit_params['sample_weight'] = sample_weight_train
        
        # Prepare validation data for early stopping
        if X_val is not None and y_val is not None and months_val is not None:
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            if isinstance(months_val, pd.Series):
                months_val = months_val.values
            if isinstance(avg_vol_val, pd.Series):
                avg_vol_val = avg_vol_val.values
            
            physics_pred_val = self._compute_physics_baseline(avg_vol_val, months_val)
            residuals_val = y_val - physics_pred_val
            
            if self.ml_model_type == 'lightgbm':
                fit_params['eval_set'] = [(X_val, residuals_val)]
                if hasattr(self.ml_model, 'set_params'):
                    self.ml_model.set_params(early_stopping_rounds=early_stopping_rounds)
                # LightGBM uses callbacks for early stopping in newer versions
                try:
                    from lightgbm import early_stopping, log_evaluation
                    fit_params['callbacks'] = [
                        early_stopping(early_stopping_rounds, verbose=False),
                        log_evaluation(period=0)
                    ]
                except ImportError:
                    # Older LightGBM version
                    fit_params['early_stopping_rounds'] = early_stopping_rounds
                    fit_params['verbose'] = False
                    
            elif self.ml_model_type == 'xgboost':
                fit_params['eval_set'] = [(X_val, residuals_val)]
                fit_params['early_stopping_rounds'] = early_stopping_rounds
                fit_params['verbose'] = False
                if sample_weight_val is not None:
                    fit_params['sample_weight_eval_set'] = [sample_weight_val]
            
            elif self.ml_model_type == 'catboost':
                from catboost import Pool
                # Identify categorical features
                cat_features = [col for col in X_val.columns if X_val[col].dtype.name == 'category' or X_val[col].dtype == object]
                # CatBoost uses Pool for eval_set
                eval_pool = Pool(X_val, residuals_val, cat_features=cat_features if cat_features else None)
                fit_params['eval_set'] = eval_pool
                fit_params['early_stopping_rounds'] = early_stopping_rounds
                fit_params['verbose'] = False
        
        # For CatBoost, identify categorical features in training data
        if self.ml_model_type == 'catboost':
            cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category' or X_train[col].dtype == object]
            if cat_features:
                fit_params['cat_features'] = cat_features
        
        # Fit the model
        self.ml_model.fit(X_train, residuals_train, **fit_params)
        self.is_fitted = True
        
        # Compute validation metrics if available
        if X_val is not None and y_val is not None and months_val is not None:
            val_preds = self.predict(X_val, avg_vol_val, months_val)
            val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
            logger.info(f"Hybrid model validation RMSE: {val_rmse:.4f}")
            self._train_stats['val_rmse'] = val_rmse
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        avg_vol: np.ndarray,
        months_postgx: np.ndarray
    ) -> np.ndarray:
        """
        Generate predictions using physics + ML residual.
        
        Args:
            X: Feature matrix
            avg_vol: Pre-LOE average volume for each sample
            months_postgx: Months since generic entry
            
        Returns:
            Final predictions (physics baseline + residual predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(avg_vol, pd.Series):
            avg_vol = avg_vol.values
        if isinstance(months_postgx, pd.Series):
            months_postgx = months_postgx.values
        
        # Physics baseline
        physics_pred = self._compute_physics_baseline(avg_vol, months_postgx)
        
        # ML residual prediction
        residual_pred = self.ml_model.predict(X)
        
        # Combine
        final_pred = physics_pred + residual_pred
        
        # Clip predictions
        if self.clip_predictions:
            final_pred = np.clip(final_pred, self.clip_min, self.clip_max)
        
        return final_pred
    
    def predict_components(
        self,
        X: pd.DataFrame,
        avg_vol: np.ndarray,
        months_postgx: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions broken down by component.
        
        Useful for understanding model behavior and debugging.
        
        Args:
            X: Feature matrix
            avg_vol: Pre-LOE average volume
            months_postgx: Months since generic entry
            
        Returns:
            Dict with 'physics', 'residual', and 'final' prediction arrays
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(avg_vol, pd.Series):
            avg_vol = avg_vol.values
        if isinstance(months_postgx, pd.Series):
            months_postgx = months_postgx.values
        
        physics_pred = self._compute_physics_baseline(avg_vol, months_postgx)
        residual_pred = self.ml_model.predict(X)
        final_pred = physics_pred + residual_pred
        
        if self.clip_predictions:
            final_pred = np.clip(final_pred, self.clip_min, self.clip_max)
        
        return {
            'physics': physics_pred,
            'residual': residual_pred,
            'final': final_pred
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the ML component.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with 'feature' and 'importance' columns
        """
        if not self.is_fitted:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        importances = self.ml_model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n).reset_index(drop=True)
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics for debugging."""
        return self._train_stats.copy()
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'ml_model_type': self.ml_model_type,
            'decay_rate': self.decay_rate,
            'params': self.params,
            'clip_predictions': self.clip_predictions,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max
        }


class HybridPhysicsMLWrapper(BaseModel):
    """
    BaseModel-compliant wrapper for HybridPhysicsMLModel.
    
    This wrapper adapts the HybridPhysicsMLModel interface to match the 
    standard BaseModel interface used by the training pipeline:
    
        fit(X_train, y_train, X_val, y_val, sample_weight)
        predict(X)
    
    It extracts months_postgx and avg_vol from X features or from metadata
    columns that may be present in the DataFrame.
    
    Config options:
        decay_rate: float, exponential decay rate (default: 0.05)
        ml_model_type: str, 'lightgbm' or 'xgboost' (default: 'lightgbm')
        clip_min: float, minimum prediction (default: 0.0)
        clip_max: float, maximum prediction (default: 2.0)
        ml_params: dict, parameters for the underlying ML model
        
    Example:
        config = {
            'decay_rate': 0.05,
            'ml_model_type': 'lightgbm',
            'ml_params': {'num_leaves': 31, 'learning_rate': 0.05}
        }
        model = HybridPhysicsMLWrapper(config)
        model.fit(X_train, y_train, X_val, y_val, sample_weight)
        predictions = model.predict(X_test)
    """
    
    # Columns that should be available for physics computation
    REQUIRED_COLS = ['months_postgx', 'avg_vol_12m']
    MONTHS_COL = 'months_postgx'
    AVG_VOL_COL = 'avg_vol_12m'
    
    def __init__(self, config: dict):
        """
        Initialize the wrapper.
        
        Args:
            config: Configuration dict with model parameters
        """
        super().__init__(config)
        
        # Extract parameters
        decay_rate = config.get('decay_rate', 0.05)
        ml_model_type = config.get('ml_model_type', 'lightgbm')
        clip_min = config.get('clip_min', 0.0)
        clip_max = config.get('clip_max', 2.0)
        ml_params = config.get('ml_params', None)
        
        # Create underlying model
        self._model = HybridPhysicsMLModel(
            ml_model_type=ml_model_type,
            decay_rate=decay_rate,
            params=ml_params,
            clip_predictions=True,
            clip_min=clip_min,
            clip_max=clip_max
        )
        
        self.feature_names: List[str] = []
        self._trained_with_meta: bool = False
    
    def _extract_meta(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Extract feature matrix and meta columns (months, avg_vol) from X.
        
        Returns:
            Tuple of (X_features, months, avg_vol)
        """
        months = None
        avg_vol = None
        
        # Try to get months_postgx
        if self.MONTHS_COL in X.columns:
            months = X[self.MONTHS_COL].values
        else:
            # Cannot proceed without months
            raise ValueError(
                f"HybridPhysicsMLWrapper requires '{self.MONTHS_COL}' column in X. "
                "Include this column in features or use train_hybrid_model directly."
            )
        
        # Try to get avg_vol_12m
        if self.AVG_VOL_COL in X.columns:
            avg_vol = X[self.AVG_VOL_COL].values
        else:
            # Use default (1.0 means no normalization adjustment)
            logger.warning(
                f"HybridPhysicsMLWrapper: '{self.AVG_VOL_COL}' not in X, using 1.0 for all samples"
            )
            avg_vol = np.ones(len(X))
        
        # Get feature columns (exclude meta)
        meta_cols = [self.MONTHS_COL, self.AVG_VOL_COL]
        feature_cols = [c for c in X.columns if c not in meta_cols]
        X_features = X[feature_cols]
        
        return X_features, months, avg_vol
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'HybridPhysicsMLWrapper':
        """
        Fit the hybrid model.
        
        Args:
            X_train: Training features (must include months_postgx, optionally avg_vol_12m)
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights (optional)
            
        Returns:
            self (fitted model)
        """
        # Extract meta from training data
        X_train_features, months_train, avg_vol_train = self._extract_meta(X_train)
        self.feature_names = list(X_train_features.columns)
        
        # Extract meta from validation data if provided
        if X_val is not None and y_val is not None:
            X_val_features, months_val, avg_vol_val = self._extract_meta(X_val)
        else:
            X_val_features, months_val, avg_vol_val = None, None, None
        
        # Convert sample_weight to numpy if needed
        sample_weight_np = None
        if sample_weight is not None:
            sample_weight_np = sample_weight.values if isinstance(sample_weight, pd.Series) else sample_weight
        
        # Fit the underlying model
        self._model.fit(
            X_train=X_train_features,
            y_train=y_train.values if isinstance(y_train, pd.Series) else y_train,
            avg_vol_train=avg_vol_train,
            months_train=months_train,
            X_val=X_val_features,
            y_val=y_val.values if isinstance(y_val, pd.Series) else y_val if y_val is not None else None,
            avg_vol_val=avg_vol_val,
            months_val=months_val,
            sample_weight_train=sample_weight_np,
            early_stopping_rounds=self.config.get('early_stopping_rounds', 50)
        )
        
        self._trained_with_meta = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame (must include months_postgx, optionally avg_vol_12m)
            
        Returns:
            Array of predictions
        """
        if not self._model.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract meta from X
        X_features, months, avg_vol = self._extract_meta(X)
        
        return self._model.predict(X_features, avg_vol, months)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump({
            'model': self._model,
            'feature_names': self.feature_names,
            'config': self.config,
            '_trained_with_meta': self._trained_with_meta
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'HybridPhysicsMLWrapper':
        """Load model from disk."""
        import joblib
        data = joblib.load(path)
        instance = cls(data.get('config', {}))
        instance._model = data['model']
        instance.feature_names = data.get('feature_names', [])
        instance._trained_with_meta = data.get('_trained_with_meta', True)
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the underlying ML model."""
        return self._model.get_feature_importance()
