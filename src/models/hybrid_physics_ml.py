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
