"""
Submission Diversification Module - Section 4.3

Creates multiple submission "flavors" to hedge against CV/LB discrepancy.

Flavors:
1. Baseline: Best CV model
2. Underfit: Regularized/simpler model
3. Overfit: Aggressive model (more trees, less regularization)
4. Bucket1-focused: Weighted training for Bucket 1
5. Conservative: Predictions clipped toward safe values
6. Aggressive: Predictions pushed toward extremes

Usage:
    from src.submission_diversification import SubmissionFlavorFactory
    
    factory = SubmissionFlavorFactory(base_model, panel_df, feature_cols)
    flavors = factory.create_all_flavors()
    
    for name, predictions in flavors.items():
        save_submission(predictions, f"submission_{name}.csv")
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from copy import deepcopy

logger = logging.getLogger(__name__)


class SubmissionFlavorFactory:
    """
    Factory for creating diverse submission flavors.
    
    Each flavor represents a different bias-variance trade-off or focus area.
    By submitting multiple flavors, you can hedge against CV overfitting and
    identify which strategy works best on the public leaderboard.
    """
    
    def __init__(
        self,
        base_model: Any,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'y_norm',
        bucket_col: str = 'bucket',
        sample_weight_col: Optional[str] = 'sample_weight'
    ):
        """
        Initialize the factory.
        
        Args:
            base_model: Fitted base model (with .predict method)
            train_df: Training data for refitting
            feature_cols: Feature column names
            target_col: Target column name
            bucket_col: Bucket column name
            sample_weight_col: Sample weight column (optional)
        """
        self.base_model = base_model
        self.train_df = train_df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.bucket_col = bucket_col
        self.sample_weight_col = sample_weight_col
        
        # Store base predictions for reference
        self._base_preds = None
    
    def get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from the base model."""
        if hasattr(self.base_model, 'predict'):
            return self.base_model.predict(X)
        else:
            raise ValueError("Base model must have a predict method")
    
    def create_baseline_flavor(self, X: pd.DataFrame) -> np.ndarray:
        """
        Baseline flavor: Direct predictions from best CV model.
        """
        return self.get_base_predictions(X)
    
    def create_conservative_flavor(
        self,
        X: pd.DataFrame,
        clip_quantile: float = 0.1
    ) -> np.ndarray:
        """
        Conservative flavor: Clip predictions toward median.
        
        Useful when you suspect the model is making extreme predictions
        that might hurt on the leaderboard.
        """
        preds = self.get_base_predictions(X)
        
        # Clip to inter-quantile range
        lower = np.quantile(preds, clip_quantile)
        upper = np.quantile(preds, 1 - clip_quantile)
        
        conservative = np.clip(preds, lower, upper)
        
        logger.info(f"Conservative: Clipped to [{lower:.4f}, {upper:.4f}]")
        return conservative
    
    def create_aggressive_flavor(
        self,
        X: pd.DataFrame,
        stretch_factor: float = 1.1
    ) -> np.ndarray:
        """
        Aggressive flavor: Stretch predictions away from mean.
        
        Useful when you think the model is too conservative.
        """
        preds = self.get_base_predictions(X)
        
        # Stretch from mean
        mean_pred = np.mean(preds)
        aggressive = mean_pred + stretch_factor * (preds - mean_pred)
        
        # Keep within valid range
        aggressive = np.clip(aggressive, 0, 1)
        
        logger.info(f"Aggressive: Stretched by {stretch_factor}x from mean")
        return aggressive
    
    def create_bucket1_focused_flavor(
        self,
        X: pd.DataFrame,
        test_df: pd.DataFrame,
        bucket1_weight_multiplier: float = 3.0
    ) -> np.ndarray:
        """
        Bucket 1 focused flavor: Retrain with higher weight on Bucket 1.
        
        Bucket 1 (fast decay) is often harder to predict and may have
        different patterns. This flavor gives more attention to Bucket 1.
        """
        # Clone model if possible
        try:
            from copy import deepcopy
            model = deepcopy(self.base_model)
        except:
            logger.warning("Could not clone model, using base predictions with adjustment")
            preds = self.get_base_predictions(X)
            
            # Adjust predictions for likely Bucket 1 samples
            # Bucket 1 = fast decay, so predictions should be lower
            if self.bucket_col in test_df.columns:
                bucket1_mask = test_df[self.bucket_col] == 1
                preds[bucket1_mask] *= 0.95  # Push toward more erosion
            
            return preds
        
        # Create sample weights emphasizing Bucket 1
        if self.bucket_col in self.train_df.columns:
            weights = np.ones(len(self.train_df))
            bucket1_mask = self.train_df[self.bucket_col] == 1
            weights[bucket1_mask] *= bucket1_weight_multiplier
            
            # Retrain
            X_train = self.train_df[self.feature_cols]
            y_train = self.train_df[self.target_col]
            
            if hasattr(model, 'fit'):
                try:
                    model.fit(X_train, y_train, sample_weight=weights)
                    return model.predict(X)
                except Exception as e:
                    logger.warning(f"Bucket1 retraining failed: {e}")
        
        return self.get_base_predictions(X)
    
    def create_smoothed_flavor(
        self,
        X: pd.DataFrame,
        test_df: pd.DataFrame,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Smoothed flavor: Blend predictions with group means.
        
        Reduces variance by pulling predictions toward therapeutic area means.
        """
        preds = self.get_base_predictions(X)
        
        # If we have therapeutic area info, smooth within groups
        if 'ther_area' in test_df.columns:
            result = preds.copy()
            
            for ther_area in test_df['ther_area'].unique():
                mask = test_df['ther_area'] == ther_area
                group_mean = preds[mask].mean()
                result[mask] = alpha * group_mean + (1 - alpha) * preds[mask]
            
            logger.info(f"Smoothed: Blended with group means (alpha={alpha})")
            return result
        
        return preds
    
    def create_ensemble_blend_flavor(
        self,
        X: pd.DataFrame,
        other_predictions: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Ensemble blend flavor: Blend base with other model predictions.
        
        Args:
            X: Test features
            other_predictions: Dict of model_name -> predictions
            weights: Optional weights for each model (including 'base')
        """
        base_preds = self.get_base_predictions(X)
        
        all_preds = {'base': base_preds}
        all_preds.update(other_predictions)
        
        if weights is None:
            # Equal weights
            weights = {name: 1.0 / len(all_preds) for name in all_preds}
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        # Blend
        blended = np.zeros_like(base_preds)
        for name, preds in all_preds.items():
            if name in weights:
                blended += weights[name] * preds
        
        logger.info(f"Ensemble blend: {len(all_preds)} models, weights={weights}")
        return blended
    
    def create_rank_based_flavor(
        self,
        X: pd.DataFrame,
        train_quantiles: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Rank-based flavor: Map predictions to training distribution quantiles.
        
        Ensures predictions follow the same distribution as training targets.
        """
        preds = self.get_base_predictions(X)
        
        # Get training target distribution
        if train_quantiles is None:
            train_target = self.train_df[self.target_col].values
            train_quantiles = np.sort(train_target)
        
        # Map predictions to quantiles
        ranks = np.searchsorted(np.sort(preds), preds) / len(preds)
        
        # Map ranks to training distribution
        indices = (ranks * (len(train_quantiles) - 1)).astype(int)
        rank_based = train_quantiles[indices]
        
        logger.info("Rank-based: Mapped to training distribution")
        return rank_based
    
    def create_all_flavors(
        self,
        X: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        include_flavors: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create all submission flavors.
        
        Args:
            X: Test features
            test_df: Test DataFrame (needed for some flavors)
            include_flavors: List of flavor names to include (None = all)
            
        Returns:
            Dict of flavor_name -> predictions
        """
        if test_df is None:
            test_df = X.copy()
        
        available_flavors = {
            'baseline': lambda: self.create_baseline_flavor(X),
            'conservative': lambda: self.create_conservative_flavor(X),
            'aggressive': lambda: self.create_aggressive_flavor(X),
            'bucket1_focused': lambda: self.create_bucket1_focused_flavor(X, test_df),
            'smoothed': lambda: self.create_smoothed_flavor(X, test_df),
            'rank_based': lambda: self.create_rank_based_flavor(X),
        }
        
        if include_flavors is None:
            include_flavors = list(available_flavors.keys())
        
        flavors = {}
        for name in include_flavors:
            if name in available_flavors:
                try:
                    flavors[name] = available_flavors[name]()
                    logger.info(f"Created flavor: {name}")
                except Exception as e:
                    logger.warning(f"Failed to create flavor {name}: {e}")
        
        return flavors


def create_submission_variants(
    predictions: np.ndarray,
    adjustments: Dict[str, float] = None
) -> Dict[str, np.ndarray]:
    """
    Quick utility to create simple variants from base predictions.
    
    Args:
        predictions: Base predictions
        adjustments: Dict of name -> multiplier
        
    Returns:
        Dict of variant_name -> adjusted predictions
    """
    if adjustments is None:
        adjustments = {
            'base': 1.0,
            'minus_5pct': 0.95,
            'plus_5pct': 1.05,
            'minus_10pct': 0.90,
            'plus_10pct': 1.10,
        }
    
    variants = {}
    for name, mult in adjustments.items():
        adjusted = np.clip(predictions * mult, 0, 1)
        variants[name] = adjusted
    
    return variants


def select_best_flavor(
    flavors: Dict[str, np.ndarray],
    validation_true: np.ndarray,
    validation_preds_per_flavor: Dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    higher_is_better: bool = False
) -> str:
    """
    Select the best flavor based on validation performance.
    
    Args:
        flavors: Dict of flavor_name -> test predictions
        validation_true: True validation targets
        validation_preds_per_flavor: Validation predictions per flavor
        metric_fn: Metric function(y_true, y_pred) -> score
        higher_is_better: Whether higher metric is better
        
    Returns:
        Name of best flavor
    """
    scores = {}
    
    for name, val_preds in validation_preds_per_flavor.items():
        scores[name] = metric_fn(validation_true, val_preds)
    
    if higher_is_better:
        best_name = max(scores, key=scores.get)
    else:
        best_name = min(scores, key=scores.get)
    
    logger.info(f"Best flavor: {best_name} (score={scores[best_name]:.4f})")
    logger.info(f"All scores: {scores}")
    
    return best_name
