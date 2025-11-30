"""
Teacher-Student Distillation for model ensembles.

Implements Section 3.3 from small_data_leaderboard_tricks_todo_copilot.md:
- Train a simpler student model using teacher ensemble predictions
- Loss = α * MSE(y_true, y_pred) + (1-α) * MSE(y_teacher, y_pred)
- Student uses stronger regularization for better generalization
"""

import logging
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

from .models.base import BaseModel

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """
    Train student models using knowledge distillation from teacher ensemble.
    
    The student learns from both:
    1. True labels (hard targets)
    2. Teacher predictions (soft targets)
    
    This helps create a simpler, more generalizable model.
    """
    
    def __init__(
        self,
        teacher_models: List[BaseModel],
        teacher_weights: Optional[List[float]] = None,
        alpha: float = 0.5,
        student_type: str = 'lgbm'
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher_models: List of trained teacher models (ensemble)
            teacher_weights: Optional weights for teacher averaging
            alpha: Balance between true labels and teacher predictions
                   α=1.0: Only true labels (no distillation)
                   α=0.0: Only teacher predictions
                   Recommended: 0.3-0.7
            student_type: Type of student model ('lgbm', 'ridge')
        """
        self.teacher_models = teacher_models
        self.teacher_weights = teacher_weights or [1.0 / len(teacher_models)] * len(teacher_models)
        self.alpha = alpha
        self.student_type = student_type
        self.student_model = None
        self.feature_names = None
    
    def get_teacher_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate weighted ensemble predictions from teacher models.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Averaged teacher predictions
        """
        predictions = []
        for model, weight in zip(self.teacher_models, self.teacher_weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def create_distillation_target(
        self,
        y_true: np.ndarray,
        y_teacher: np.ndarray
    ) -> np.ndarray:
        """
        Create blended target for student training.
        
        y_blend = α * y_true + (1-α) * y_teacher
        
        Args:
            y_true: Ground truth labels
            y_teacher: Teacher predictions
            
        Returns:
            Blended target
        """
        return self.alpha * y_true + (1 - self.alpha) * y_teacher
    
    def _create_student_model(self, config: Optional[Dict] = None):
        """Create student model with stronger regularization."""
        if config is None:
            config = {}
        
        if self.student_type == 'lgbm':
            # Student uses fewer trees, higher regularization
            student_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 15,  # Smaller than teacher
                'learning_rate': 0.03,
                'n_estimators': 500,  # Fewer trees
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,  # L2 regularization
                'min_data_in_leaf': 30,  # Larger minimum
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42,
            }
            student_params.update(config)
            return lgb.LGBMRegressor(**student_params)
        
        elif self.student_type == 'ridge':
            # Simple linear model as student
            return Ridge(alpha=config.get('alpha', 1.0))
        
        else:
            raise ValueError(f"Unknown student type: {self.student_type}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        student_config: Optional[Dict] = None
    ) -> 'DistillationTrainer':
        """
        Train student model using distillation.
        
        Args:
            X_train: Training features
            y_train: True training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            sample_weight: Optional sample weights
            student_config: Optional config overrides for student
            
        Returns:
            self
        """
        self.feature_names = list(X_train.columns)
        
        # Get teacher predictions on training data
        logger.info("Generating teacher predictions for distillation...")
        y_teacher_train = self.get_teacher_predictions(X_train)
        
        # Create blended target
        y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        y_distill = self.create_distillation_target(y_train_arr, y_teacher_train)
        
        logger.info(
            f"Distillation target created (α={self.alpha}): "
            f"mean={y_distill.mean():.4f}, std={y_distill.std():.4f}"
        )
        
        # Create and train student
        self.student_model = self._create_student_model(student_config)
        
        if self.student_type == 'lgbm':
            # LightGBM with early stopping
            callbacks = [lgb.early_stopping(50, verbose=False)]
            
            eval_set = None
            if X_val is not None and y_val is not None:
                y_val_arr = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
                y_teacher_val = self.get_teacher_predictions(X_val)
                y_val_distill = self.create_distillation_target(y_val_arr, y_teacher_val)
                eval_set = [(X_val, y_val_distill)]
            
            self.student_model.fit(
                X_train, y_distill,
                sample_weight=sample_weight,
                eval_set=eval_set,
                callbacks=callbacks
            )
            
            logger.info(f"Student trained for {self.student_model.best_iteration_} iterations")
        
        else:
            # Ridge or other sklearn models
            self.student_model.fit(X_train, y_distill, sample_weight=sample_weight)
            logger.info("Student (Ridge) trained")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using student model."""
        if self.student_model is None:
            raise ValueError("Student model not trained. Call fit() first.")
        
        return self.student_model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from student model."""
        if self.student_model is None or self.feature_names is None:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        if self.student_type == 'lgbm':
            importance = self.student_model.feature_importances_
        elif self.student_type == 'ridge':
            importance = np.abs(self.student_model.coef_)
        else:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


def train_distilled_model(
    teacher_models: List[BaseModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    alpha: float = 0.5,
    teacher_weights: Optional[List[float]] = None,
    sample_weight: Optional[np.ndarray] = None,
    student_type: str = 'lgbm'
) -> DistillationTrainer:
    """
    Convenience function to train a distilled student model.
    
    Args:
        teacher_models: Pre-trained teacher models
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        alpha: Distillation alpha (0.3-0.7 recommended)
        teacher_weights: Optional weights for teacher ensemble
        sample_weight: Optional sample weights
        student_type: 'lgbm' or 'ridge'
        
    Returns:
        Trained DistillationTrainer
    """
    trainer = DistillationTrainer(
        teacher_models=teacher_models,
        teacher_weights=teacher_weights,
        alpha=alpha,
        student_type=student_type
    )
    
    trainer.fit(
        X_train, y_train,
        X_val, y_val,
        sample_weight=sample_weight
    )
    
    return trainer
