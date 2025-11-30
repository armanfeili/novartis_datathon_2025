"""
CatBoost model implementation for Novartis Datathon 2025.

HERO MODEL - Primary model for competition submission with:
- Native categorical support (no encoding needed)
- Robust regularization preventing overfitting
- GPU support for faster training
- Sample weight support aligned with official metric
"""

from typing import Optional, List, Union
import logging
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .base import BaseModel

logger = logging.getLogger(__name__)


class CatBoostModel(BaseModel):
    """
    CatBoost implementation with native categorical support.
    
    Hero model for the competition with key features:
    - Handles categoricals natively without encoding
    - Supports sample weights for metric-aligned training
    - GPU acceleration when available
    - Robust early stopping
    """
    
    # Default configuration aligned with model_cat.yaml
    DEFAULT_CONFIG = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'min_data_in_leaf': 20,
        'random_strength': 1.0,
        'bagging_temperature': 1.0,
        'early_stopping_rounds': 100,
        'random_seed': 42,
        'verbose': 100,
        'thread_count': -1,
    }
    
    def __init__(self, config: dict):
        """
        Initialize CatBoost model.
        
        Args:
            config: Configuration dict with:
                - 'params': CatBoost hyperparameters
                - 'categorical_features': List of categorical column names
                - 'gpu': Dict with 'enabled' and 'device_id' for GPU settings
        """
        super().__init__(config)
        
        # Merge default config with provided config
        self.params = {**self.DEFAULT_CONFIG}
        if 'params' in config:
            self.params.update(config['params'])
        
        # Get categorical feature names if specified
        self.cat_features: List[str] = config.get('categorical_features', [])
        
        # Handle GPU configuration
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False):
            try:
                # Check if GPU is available
                self.params['task_type'] = 'GPU'
                self.params['devices'] = str(gpu_config.get('device_id', 0))
                logger.info(f"CatBoost GPU mode enabled on device {self.params['devices']}")
            except Exception as e:
                logger.warning(f"GPU not available, falling back to CPU: {e}")
                self.params['task_type'] = 'CPU'
        else:
            self.params['task_type'] = 'CPU'
        
        # Handle small_data_mode (BONUS: G4)
        small_data_config = config.get('small_data_mode', {})
        if small_data_config.get('enabled', False):
            overrides = small_data_config.get('overrides', {})
            self.params.update(overrides)
            logger.info(f"Small data mode enabled: {overrides}")
        
        # Store monotonicity config for later (needs feature names)
        self.monotonicity_config = config.get('monotonicity', {})
        
        # Initialize model
        self.model = CatBoostRegressor(**self.params)
        self._config = config  # Store full config for save/load
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'CatBoostModel':
        """
        Train CatBoost model with optional validation and sample weights.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features for early stopping
            y_val: Optional validation target
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Auto-detect categorical features if not explicitly provided
        if not self.cat_features:
            from ..features import get_categorical_feature_names
            self.cat_features = get_categorical_feature_names(X_train)
            logger.info(f"Auto-detected {len(self.cat_features)} categorical features")
        
        # Identify categorical features present in data
        cat_features_idx = []
        for cat_col in self.cat_features:
            if cat_col in X_train.columns:
                cat_features_idx.append(list(X_train.columns).index(cat_col))
        
        # Also check for category dtype columns that weren't explicitly listed
        for i, col in enumerate(X_train.columns):
            if X_train[col].dtype.name == 'category' and i not in cat_features_idx:
                cat_features_idx.append(i)
                logger.debug(f"Added category dtype column '{col}' to categorical features")
        
        # Handle monotonicity constraints (BONUS: B9)
        monotone_constraints = None
        if self.monotonicity_config.get('enabled', False):
            constraints_dict = self.monotonicity_config.get('constraints', {})
            if constraints_dict:
                # Convert feature name constraints to index-based constraints
                monotone_constraints = []
                for feat_name in self.feature_names:
                    if feat_name in constraints_dict:
                        constraint_val = constraints_dict[feat_name]
                        monotone_constraints.append(constraint_val)
                    else:
                        monotone_constraints.append(0)  # No constraint
                
                logger.info(f"Monotonicity constraints applied: {sum(c != 0 for c in monotone_constraints)} features")
                # Update params for this training
                train_params = self.params.copy()
                train_params['monotone_constraints'] = monotone_constraints
        
        # Prepare training pool
        train_pool = Pool(
            X_train,
            y_train,
            cat_features=cat_features_idx if cat_features_idx else None,
            weight=sample_weight.values if sample_weight is not None else None
        )
        
        # Prepare validation pool if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(
                X_val,
                y_val,
                cat_features=cat_features_idx if cat_features_idx else None
            )
        
        # Train (with monotonicity if enabled)
        fit_params = {
            'use_best_model': True if eval_set else False,
            'verbose': self.params.get('verbose', 100)
        }
        
        if monotone_constraints is not None:
            # Need to recreate model with monotonicity constraints
            train_params = self.params.copy()
            train_params['monotone_constraints'] = monotone_constraints
            self.model = CatBoostRegressor(**train_params)
        
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            **fit_params
        )
        
        logger.info(f"CatBoost trained for {self.model.get_best_iteration()} iterations")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        # Ensure feature order matches model EXACTLY
        X_proc = X.copy()
        
        # CRITICAL: Reorder columns to match model's feature order exactly
        if hasattr(self.model, 'feature_names_') and self.model.feature_names_:
            # Get the exact feature order from the model
            model_feature_names = self.model.feature_names_
            # Create DataFrame with features in exact model order
            X_ordered = pd.DataFrame(index=X_proc.index)
            for feat_name in model_feature_names:
                if feat_name in X_proc.columns:
                    X_ordered[feat_name] = X_proc[feat_name]
                else:
                    # Feature missing - fill with 0 (or appropriate default)
                    logger.warning(f"Feature '{feat_name}' missing in prediction data, filling with 0")
                    X_ordered[feat_name] = 0
            X_proc = X_ordered
        elif hasattr(self, 'feature_names') and self.feature_names:
            # Use stored feature names if model doesn't have feature_names_
            model_feature_names = self.feature_names
            X_ordered = pd.DataFrame(index=X_proc.index)
            for feat_name in model_feature_names:
                if feat_name in X_proc.columns:
                    X_ordered[feat_name] = X_proc[feat_name]
                else:
                    logger.warning(f"Feature '{feat_name}' missing in prediction data, filling with 0")
                    X_ordered[feat_name] = 0
            X_proc = X_ordered
        
        # AGGRESSIVE: Convert ALL columns that could be categorical to integers
        # CatBoost requires categorical features to be integers, not floats
        # Convert any column matching categorical patterns OR in cat_features list
        for col in X_proc.columns:
            is_categorical = (
                col in self.cat_features or
                col.endswith('_bin') or
                col.endswith('_encoded') or
                X_proc[col].dtype.name == 'category'
            )
            
            if is_categorical:
                try:
                    if X_proc[col].dtype.name == 'category':
                        X_proc[col] = X_proc[col].cat.codes.astype(int)
                        X_proc[col] = X_proc[col].replace(-1, 0).astype(int)
                    elif X_proc[col].dtype in ['float64', 'float32']:
                        # Convert float to int (e.g., 0.0 -> 0)
                        # Check if values are whole numbers (likely categorical)
                        if X_proc[col].notna().any():
                            sample_values = X_proc[col].dropna().head(100)
                            if len(sample_values) > 0 and (sample_values == sample_values.astype(int)).all():
                                X_proc[col] = X_proc[col].fillna(0).astype(int)
                    elif pd.api.types.is_integer_dtype(X_proc[col]):
                        X_proc[col] = X_proc[col].fillna(0).astype(int)
                except Exception as e:
                    logger.warning(f"Could not convert categorical feature {col}: {e}")
        
        # AGGRESSIVE: Convert ALL float columns that are whole numbers to integers
        # CatBoost may have stored some features as categorical that we don't know about
        # This ensures any float that should be int is converted
        for col in X_proc.columns:
            if X_proc[col].dtype in ['float64', 'float32']:
                # Check if values are whole numbers (likely should be integers)
                non_null = X_proc[col].dropna()
                if len(non_null) > 0:
                    # Sample check: if first 1000 values are whole numbers, convert all
                    sample = non_null.head(1000)
                    if len(sample) > 0 and (sample == sample.astype(int)).all():
                        # This column has whole number floats - convert to int
                        X_proc[col] = X_proc[col].fillna(0).astype(int)
                    # Also convert if it's in categorical features list or matches patterns
                    elif col in self.cat_features or col.endswith('_bin') or col.endswith('_encoded'):
                        X_proc[col] = X_proc[col].fillna(0).astype(int)
        
        # Use CatBoost Pool to handle categorical features properly
        # This ensures CatBoost handles categorical conversion internally
        try:
            from catboost import Pool
            # Get categorical feature indices
            cat_feature_indices = []
            for i, col in enumerate(X_proc.columns):
                if col in self.cat_features or col.endswith('_bin') or col.endswith('_encoded'):
                    cat_feature_indices.append(i)
            
            # Create Pool with categorical features specified
            pool = Pool(X_proc, cat_features=cat_feature_indices if cat_feature_indices else None)
            return self.model.predict(pool)
        except Exception as e:
            # Fallback to direct prediction if Pool fails
            logger.warning(f"Pool prediction failed, using direct prediction: {e}")
            return self.model.predict(X_proc)
    
    def save(self, path: str) -> None:
        """
        Save model to disk with metadata.
        
        Saves:
        - Model file (.cbm format)
        - Metadata file (.json) with config, feature names, categorical features
        
        Args:
            path: Path to save model (without extension)
        """
        import json
        
        # Ensure path has proper extension
        model_path = path if path.endswith('.cbm') else f"{path}.cbm"
        
        # Save CatBoost model
        self.model.save_model(model_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'categorical_features': self.cat_features,
            'params': self.params,
            'model_path': model_path,
        }
        
        meta_path = model_path.replace('.cbm', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_path} with metadata")
    
    @classmethod
    def load(cls, path: str, config: Optional[dict] = None) -> 'CatBoostModel':
        """
        Load model from disk with metadata.
        
        Args:
            path: Path to model file (with or without .cbm extension)
            config: Optional config to override loaded settings
            
        Returns:
            Loaded CatBoostModel instance
        """
        import json
        
        # Handle path with or without extension
        model_path = path if path.endswith('.cbm') else f"{path}.cbm"
        
        # Try to load metadata
        meta_path = model_path.replace('.cbm', '_meta.json')
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        # Create instance with loaded or provided config
        load_config = config or {}
        if 'params' not in load_config and 'params' in metadata:
            load_config['params'] = metadata['params']
        if 'categorical_features' not in load_config and 'categorical_features' in metadata:
            load_config['categorical_features'] = metadata['categorical_features']
        
        instance = cls(load_config)
        instance.model = CatBoostRegressor()
        instance.model.load_model(model_path)
        instance.feature_names = metadata.get('feature_names', instance.model.feature_names_ or [])
        instance.cat_features = metadata.get('categorical_features', [])
        
        logger.info(f"Model loaded from {model_path}")
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances.
        
        Returns:
            DataFrame with columns ['feature', 'importance']
        """
        if self.model is None or len(self.feature_names) == 0:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        importance = self.model.get_feature_importance()
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
