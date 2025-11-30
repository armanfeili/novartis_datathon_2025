"""
Bayesian Stacking Implementation for Novartis Datathon 2025.

Implements Bayesian model averaging with Dirichlet priors, optimized
for the official competition metric with time-window and bucket weights.

Usage:
    from src.stacking import BayesianStacker, train_stacker_for_scenario
    
    # Train stacker
    stacker = train_stacker_for_scenario(scenario=1, config=config)
    
    # Apply to test
    ensemble_preds = apply_ensemble_to_test(scenario=1, stacker=stacker)
"""

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


# ==============================================================================
# SAMPLE WEIGHT COMPUTATION (Section 3.2)
# ==============================================================================

def compute_sample_weight(
    month: int,
    bucket: int,
    scenario: int,
    time_weights: Optional[Dict[str, float]] = None,
    bucket_weights: Optional[Dict[int, float]] = None
) -> float:
    """
    Compute sample weight based on official metric structure.
    
    The official metric gives more importance to:
    - Early months (0-5 in S1, 6-11 in both)
    - Bucket 1 (weighted 2x vs Bucket 2)
    
    Args:
        month: months_postgx value
        bucket: 1 or 2
        scenario: 1 or 2
        time_weights: Optional custom time weights
        bucket_weights: Optional custom bucket weights
        
    Returns:
        Combined sample weight
    """
    # Default time weights based on official metric
    if time_weights is None:
        if scenario == 1:
            if 0 <= month <= 5:
                time_w = 3.0  # Early months (S1 only)
            elif 6 <= month <= 11:
                time_w = 4.0  # Mid months (both scenarios, critical)
            else:
                time_w = 2.0  # Late months
        else:  # scenario == 2
            if 6 <= month <= 11:
                time_w = 4.0  # Mid months (critical for S2)
            else:
                time_w = 2.0  # Late months
    else:
        # Use custom time weights
        if scenario == 1:
            if 0 <= month <= 5:
                time_w = time_weights.get('early', 3.0)
            elif 6 <= month <= 11:
                time_w = time_weights.get('mid', 4.0)
            else:
                time_w = time_weights.get('late', 2.0)
        else:
            if 6 <= month <= 11:
                time_w = time_weights.get('mid', 4.0)
            else:
                time_w = time_weights.get('late', 2.0)
    
    # Default bucket weights
    if bucket_weights is None:
        bucket_w = 2.0 if bucket == 1 else 1.0
    else:
        bucket_w = bucket_weights.get(bucket, 1.0)
    
    return time_w * bucket_w


def compute_sample_weights_vectorized(
    months: np.ndarray,
    buckets: np.ndarray,
    scenario: int
) -> np.ndarray:
    """
    Vectorized version of compute_sample_weight for efficiency.
    
    Args:
        months: Array of months_postgx values
        buckets: Array of bucket values
        scenario: 1 or 2
        
    Returns:
        Array of sample weights
    """
    weights = np.ones(len(months))
    
    # Time weights
    if scenario == 1:
        weights[months <= 5] = 3.0
        weights[(months >= 6) & (months <= 11)] = 4.0
        weights[months >= 12] = 2.0
    else:  # scenario == 2
        weights[(months >= 6) & (months <= 11)] = 4.0
        weights[months >= 12] = 2.0
    
    # Bucket weights
    bucket_multiplier = np.where(buckets == 1, 2.0, 1.0)
    
    return weights * bucket_multiplier


# ==============================================================================
# BAYESIAN STACKER (Section 4)
# ==============================================================================

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(z - np.max(z))
    return e / e.sum()


def fit_dirichlet_weighted_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    regularization_strength: float = 1.0
) -> np.ndarray:
    """
    Fit ensemble weights using MAP optimization with Dirichlet prior.
    
    The optimization minimizes:
        L(w) = sum_n(s_n * (y_n - X_n @ w)^2) - log p(w)
    
    where p(w) ~ Dirichlet(alpha).
    
    Args:
        X: (N, M) matrix of base model predictions
        y: (N,) true targets
        sample_weight: (N,) sample weights (time * bucket)
        alpha: (M,) Dirichlet prior params (default: all ones)
        regularization_strength: Strength of Dirichlet prior
        
    Returns:
        Optimal weights w of shape (M,)
    """
    N, M = X.shape
    
    if alpha is None:
        alpha = np.ones(M)
    
    # Scale alpha by regularization strength
    alpha_scaled = alpha * regularization_strength
    
    def loss_fn(z):
        """Loss function on unconstrained params z."""
        w = softmax(z)  # shape (M,)
        y_pred = X @ w  # shape (N,)
        resid = y - y_pred
        
        # Weighted squared error
        mse_part = np.sum(sample_weight * resid ** 2)
        
        # Negative log Dirichlet prior (up to constant)
        # p(w) ∝ prod_m w_m^(alpha_m - 1)
        # -log p(w) = -sum_m (alpha_m - 1) * log(w_m)
        prior_part = -np.sum((alpha_scaled - 1.0) * np.log(w + 1e-12))
        
        return mse_part + prior_part
    
    def grad_fn(z):
        """Gradient of loss function."""
        w = softmax(z)
        y_pred = X @ w
        resid = y - y_pred
        
        # Gradient of MSE w.r.t. w
        grad_mse_w = -2 * X.T @ (sample_weight * resid)
        
        # Gradient of prior w.r.t. w
        grad_prior_w = -(alpha_scaled - 1.0) / (w + 1e-12)
        
        grad_w = grad_mse_w + grad_prior_w
        
        # Chain rule: dL/dz = dL/dw * dw/dz
        # dw/dz (Jacobian of softmax) = diag(w) - w @ w.T
        jacobian = np.diag(w) - np.outer(w, w)
        grad_z = jacobian @ grad_w
        
        return grad_z
    
    # Initialize with uniform weights (z = 0)
    z0 = np.zeros(M)
    
    # Try multiple initializations
    best_loss = np.inf
    best_z = z0
    
    for init_seed in range(5):
        if init_seed == 0:
            z_init = z0
        else:
            np.random.seed(init_seed * 42)
            z_init = np.random.randn(M) * 0.1
        
        try:
            result = minimize(
                loss_fn,
                z_init,
                method='L-BFGS-B',
                jac=grad_fn,
                options={'maxiter': 1000, 'disp': False}
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_z = result.x
        except Exception as e:
            logger.warning(f"Optimization failed with seed {init_seed}: {e}")
            continue
    
    # Convert to weights
    w_opt = softmax(best_z)
    
    logger.info(f"Bayesian stacking fitted with loss={best_loss:.4f}")
    logger.info(f"Learned weights: {w_opt}")
    
    return w_opt


class BayesianStacker:
    """
    Bayesian model stacker with Dirichlet prior on weights.
    
    Uses MAP optimization to find optimal convex combination of
    base model predictions, with sample weights matching the
    official competition metric.
    
    Example:
        stacker = BayesianStacker(alpha=np.ones(5))
        stacker.fit(X_oof, y_true, sample_weight)
        y_ensemble = stacker.predict(X_test)
    """
    
    def __init__(
        self,
        alpha: Optional[np.ndarray] = None,
        regularization_strength: float = 1.0,
        clip_predictions: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 2.0
    ):
        """
        Initialize Bayesian stacker.
        
        Args:
            alpha: Dirichlet prior parameters (default: uniform)
            regularization_strength: Strength of Dirichlet prior
            clip_predictions: Whether to clip predictions
            clip_min: Minimum prediction value
            clip_max: Maximum prediction value
        """
        self.alpha = alpha
        self.regularization_strength = regularization_strength
        self.clip_predictions = clip_predictions
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        self.weights_: Optional[np.ndarray] = None
        self.model_names_: List[str] = []
        self.n_models_: int = 0
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        model_names: Optional[List[str]] = None
    ) -> 'BayesianStacker':
        """
        Fit the stacker using MAP optimization.
        
        Args:
            X: (N, M) matrix of base model predictions
            y: (N,) true targets
            sample_weight: (N,) optional sample weights
            model_names: Optional list of model names for logging
            
        Returns:
            self
        """
        N, M = X.shape
        self.n_models_ = M
        
        if model_names is not None:
            self.model_names_ = model_names
        else:
            self.model_names_ = [f"model_{i}" for i in range(M)]
        
        # Default to uniform weights if not provided
        if sample_weight is None:
            sample_weight = np.ones(N)
        
        # Default to uniform prior
        if self.alpha is None:
            alpha = np.ones(M)
        else:
            alpha = self.alpha
        
        # Fit using MAP optimization
        self.weights_ = fit_dirichlet_weighted_ensemble(
            X, y, sample_weight, alpha, self.regularization_strength
        )
        
        # Log weights with model names
        for name, w in zip(self.model_names_, self.weights_):
            logger.info(f"  {name}: {w:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: (N, M) matrix of base model predictions
            
        Returns:
            (N,) ensemble predictions
        """
        if self.weights_ is None:
            raise ValueError("Stacker not fitted. Call fit() first.")
        
        y_pred = X @ self.weights_
        
        if self.clip_predictions:
            y_pred = np.clip(y_pred, self.clip_min, self.clip_max)
        
        return y_pred
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Return weights as a dictionary."""
        return dict(zip(self.model_names_, self.weights_))
    
    def save(self, path: str) -> None:
        """Save stacker to disk."""
        joblib.dump({
            'weights': self.weights_,
            'model_names': self.model_names_,
            'alpha': self.alpha,
            'regularization_strength': self.regularization_strength,
            'clip_predictions': self.clip_predictions,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
        }, path)
        logger.info(f"BayesianStacker saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BayesianStacker':
        """Load stacker from disk."""
        data = joblib.load(path)
        
        instance = cls(
            alpha=data.get('alpha'),
            regularization_strength=data.get('regularization_strength', 1.0),
            clip_predictions=data.get('clip_predictions', True),
            clip_min=data.get('clip_min', 0.0),
            clip_max=data.get('clip_max', 2.0),
        )
        instance.weights_ = data['weights']
        instance.model_names_ = data.get('model_names', [])
        instance.n_models_ = len(instance.weights_)
        
        return instance


# ==============================================================================
# OOF PREDICTION GENERATION (Section 2)
# ==============================================================================

def generate_oof_predictions(
    model_class,
    model_config: dict,
    df_train: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'y_norm',
    scenario: int = 1,
    n_folds: int = 5,
    group_col: str = 'brand_name',
    stratify_col: str = 'bucket',
    random_state: int = 42,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate out-of-fold predictions for a single model using GroupKFold.
    
    Args:
        model_class: Model class to instantiate
        model_config: Configuration dict for model
        df_train: Training DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        scenario: 1 or 2
        n_folds: Number of CV folds
        group_col: Column for grouping (brand_name)
        stratify_col: Column for stratification (bucket)
        random_state: Random seed
        save_path: Optional path to save OOF predictions
        
    Returns:
        DataFrame with columns [country, brand_name, months_postgx, y_true, y_pred, bucket]
    """
    # Filter by scenario
    if scenario == 1:
        df = df_train[df_train['months_postgx'].between(0, 23)].copy()
    else:
        df = df_train[df_train['months_postgx'].between(6, 23)].copy()
    
    # Create unique group identifiers
    groups = df.groupby(['country', group_col]).ngroup()
    
    # Initialize OOF predictions
    oof_preds = np.full(len(df), np.nan)
    
    # GroupKFold CV
    gkf = GroupKFold(n_splits=n_folds)
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        logger.info(f"Fold {fold_idx + 1}/{n_folds}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        # Compute sample weights for training
        train_weights = compute_sample_weights_vectorized(
            train_df['months_postgx'].values,
            train_df['bucket'].values,
            scenario
        )
        
        # Train model
        model = model_class(model_config)
        model.fit(X_train, y_train, X_val, y_val, 
                  sample_weight=pd.Series(train_weights, index=train_df.index))
        
        # Predict on validation fold
        oof_preds[val_idx] = model.predict(X_val)
    
    # Build result DataFrame
    result = df[['country', 'brand_name', 'months_postgx', 'bucket']].copy()
    result['y_true'] = df[target_col].values
    result['y_pred'] = oof_preds
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(save_path, index=False)
        logger.info(f"OOF predictions saved to {save_path}")
    
    return result


# ==============================================================================
# META-DATASET BUILDING (Section 3)
# ==============================================================================

def build_meta_dataset_for_scenario(
    oof_files: Dict[str, str],
    scenario: int,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Build meta-dataset by combining OOF predictions from multiple models.
    
    Args:
        oof_files: Dict mapping model name -> OOF parquet file path
        scenario: 1 or 2
        save_path: Optional path to save meta-dataset
        
    Returns:
        DataFrame with columns [country, brand_name, months_postgx, y_true, bucket,
                               sample_weight, pred_model1, pred_model2, ...]
    """
    model_names = list(oof_files.keys())
    
    # Load first model as base
    first_model = model_names[0]
    df_meta = pd.read_parquet(oof_files[first_model])
    df_meta = df_meta.rename(columns={'y_pred': f'pred_{first_model}'})
    
    # Merge other models
    for model_name in model_names[1:]:
        df_model = pd.read_parquet(oof_files[model_name])
        
        # Keep only prediction column for merge
        df_model = df_model[['country', 'brand_name', 'months_postgx', 'y_pred']]
        df_model = df_model.rename(columns={'y_pred': f'pred_{model_name}'})
        
        # Merge on keys
        df_meta = df_meta.merge(
            df_model,
            on=['country', 'brand_name', 'months_postgx'],
            how='inner'
        )
    
    # Add sample weights
    df_meta['sample_weight'] = compute_sample_weights_vectorized(
        df_meta['months_postgx'].values,
        df_meta['bucket'].values,
        scenario
    )
    
    logger.info(f"Meta-dataset built: {len(df_meta)} rows, "
               f"{len(model_names)} models")
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_meta.to_parquet(save_path, index=False)
        logger.info(f"Meta-dataset saved to {save_path}")
    
    return df_meta


def train_stacker_for_scenario(
    df_meta: pd.DataFrame,
    scenario: int,
    alpha: Optional[np.ndarray] = None,
    regularization_strength: float = 1.0,
    save_dir: Optional[str] = None
) -> BayesianStacker:
    """
    Train Bayesian stacker for a scenario using the meta-dataset.
    
    Args:
        df_meta: Meta-dataset from build_meta_dataset_for_scenario
        scenario: 1 or 2
        alpha: Dirichlet prior parameters
        regularization_strength: Prior strength
        save_dir: Optional directory to save stacker
        
    Returns:
        Fitted BayesianStacker
    """
    # Get prediction columns
    pred_cols = [c for c in df_meta.columns if c.startswith('pred_')]
    model_names = [c.replace('pred_', '') for c in pred_cols]
    
    # Extract arrays
    X = df_meta[pred_cols].values
    y = df_meta['y_true'].values
    sample_weight = df_meta['sample_weight'].values
    
    # Create and fit stacker
    stacker = BayesianStacker(
        alpha=alpha,
        regularization_strength=regularization_strength
    )
    stacker.fit(X, y, sample_weight, model_names=model_names)
    
    # Save if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save stacker
        stacker.save(save_dir / f'stacker_scenario{scenario}.joblib')
        
        # Save weights as numpy
        np.save(save_dir / f'weights_scenario{scenario}.npy', stacker.weights_)
        
        # Save weights as text
        with open(save_dir / f'weights_scenario{scenario}.txt', 'w') as f:
            f.write(f"Scenario {scenario} Bayesian Stacking Weights\n")
            f.write("=" * 50 + "\n")
            for name, w in zip(stacker.model_names_, stacker.weights_):
                f.write(f"{name}: {w:.6f}\n")
        
        logger.info(f"Stacker saved to {save_dir}")
    
    return stacker


# ==============================================================================
# TEST PREDICTION ENSEMBLE (Section 5)
# ==============================================================================

def build_test_meta_dataset(
    test_pred_files: Dict[str, str],
    scenario: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build test meta-dataset from individual model test predictions.
    
    Args:
        test_pred_files: Dict mapping model name -> test prediction file path
        scenario: 1 or 2
        
    Returns:
        Tuple of (DataFrame with keys + predictions, X matrix)
    """
    model_names = list(test_pred_files.keys())
    
    # Load first model as base
    first_model = model_names[0]
    df_test = pd.read_parquet(test_pred_files[first_model])
    
    # Rename prediction column
    pred_col = 'volume' if 'volume' in df_test.columns else 'y_pred'
    df_test = df_test.rename(columns={pred_col: f'pred_{first_model}'})
    
    # Merge other models
    for model_name in model_names[1:]:
        df_model = pd.read_parquet(test_pred_files[model_name])
        
        pred_col = 'volume' if 'volume' in df_model.columns else 'y_pred'
        df_model = df_model[['country', 'brand_name', 'months_postgx', pred_col]]
        df_model = df_model.rename(columns={pred_col: f'pred_{model_name}'})
        
        df_test = df_test.merge(
            df_model,
            on=['country', 'brand_name', 'months_postgx'],
            how='inner'
        )
    
    # Extract prediction matrix (same order as training)
    pred_cols = [f'pred_{name}' for name in model_names]
    X = df_test[pred_cols].values
    
    return df_test, X


def apply_ensemble_to_test(
    stacker: BayesianStacker,
    test_pred_files: Dict[str, str],
    scenario: int,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply trained Bayesian stacker to test predictions.
    
    Args:
        stacker: Fitted BayesianStacker
        test_pred_files: Dict mapping model name -> test prediction file path
        scenario: 1 or 2
        save_path: Optional path to save ensemble predictions
        
    Returns:
        DataFrame with columns [country, brand_name, months_postgx, volume]
    """
    # Build test meta-dataset
    df_test, X = build_test_meta_dataset(test_pred_files, scenario)
    
    # Apply ensemble weights
    y_ensemble = stacker.predict(X)
    
    # Build result DataFrame
    result = df_test[['country', 'brand_name', 'months_postgx']].copy()
    result['volume'] = y_ensemble
    
    # Sanitize: clip negatives to 0
    result['volume'] = result['volume'].clip(lower=0)
    
    # Verify scenario months
    if scenario == 1:
        expected_months = set(range(0, 24))
    else:
        expected_months = set(range(6, 24))
    
    actual_months = set(result['months_postgx'].unique())
    if not expected_months.issubset(actual_months):
        logger.warning(f"Missing months in test predictions: "
                      f"{expected_months - actual_months}")
    
    logger.info(f"Ensemble predictions generated: {len(result)} rows")
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
        logger.info(f"Ensemble submission saved to {save_path}")
    
    return result


# ==============================================================================
# HIERARCHICAL BAYESIAN DECAY MODEL (Section 7 - Optional)
# ==============================================================================

class HierarchicalBayesianDecay:
    """
    Hierarchical Bayesian decay model as extra base model.
    
    For each brand j:
        vol_norm_j(t) = a_j * exp(-b_j * t) + c_j
    
    With hierarchical priors on (a, b, c) by therapeutic area/bucket.
    Priors are learned from the population of brands.
    
    This is a simplified implementation using Empirical Bayes (MAP estimation).
    For full Bayesian inference, use PyMC or NumPyro.
    """
    
    def __init__(
        self,
        prior_a_mean: float = 1.0,
        prior_a_std: float = 0.5,
        prior_b_mean: float = 0.05,
        prior_b_std: float = 0.02,
        prior_c_mean: float = 0.3,
        prior_c_std: float = 0.2,
        use_hierarchical_priors: bool = True,
        shrinkage_strength: float = 0.3
    ):
        """
        Initialize with prior parameters.
        
        Args:
            prior_a_mean: Prior mean for amplitude parameter a
            prior_a_std: Prior std for a
            prior_b_mean: Prior mean for decay rate b
            prior_b_std: Prior std for b
            prior_c_mean: Prior mean for asymptote c
            prior_c_std: Prior std for c
            use_hierarchical_priors: Whether to learn hierarchical priors by bucket
            shrinkage_strength: How much to shrink individual estimates toward group mean
        """
        self.prior_a = (prior_a_mean, prior_a_std)
        self.prior_b = (prior_b_mean, prior_b_std)
        self.prior_c = (prior_c_mean, prior_c_std)
        self.use_hierarchical_priors = use_hierarchical_priors
        self.shrinkage_strength = shrinkage_strength
        
        self.brand_params_: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        self.bucket_priors_: Dict[int, Dict[str, Tuple[float, float]]] = {}
        self.global_priors_: Dict[str, Tuple[float, float]] = {}
    
    def _fit_brand(
        self,
        t: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        priors: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[float, float, float]:
        """Fit decay parameters for a single brand."""
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        
        if priors is None:
            priors = {
                'a': self.prior_a,
                'b': self.prior_b,
                'c': self.prior_c
            }
        
        def decay_model(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        def loss_fn(params):
            a, b, c = params
            y_pred = decay_model(t, a, b, c)
            
            # Weighted MSE
            mse = np.sum(sample_weight * (y - y_pred) ** 2)
            
            # Gaussian priors (negative log)
            prior_a = ((a - priors['a'][0]) / priors['a'][1]) ** 2
            prior_b = ((b - priors['b'][0]) / priors['b'][1]) ** 2
            prior_c = ((c - priors['c'][0]) / priors['c'][1]) ** 2
            
            return mse + 0.5 * (prior_a + prior_b + prior_c)
        
        # Initialize with prior means
        x0 = [priors['a'][0], priors['b'][0], priors['c'][0]]
        
        # Bounds: a > 0, b > 0, c >= 0
        bounds = [(0.01, 2.0), (0.001, 0.5), (0.0, 1.0)]
        
        result = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds)
        
        return tuple(result.x)
    
    def _estimate_hierarchical_priors(
        self,
        df: pd.DataFrame,
        target_col: str = 'y_norm',
        time_col: str = 'months_postgx'
    ) -> None:
        """
        Estimate hierarchical priors from data using Empirical Bayes.
        
        First fits all brands with global priors, then estimates
        bucket-specific priors from the fitted parameters.
        """
        logger.info("Estimating hierarchical priors (Empirical Bayes)...")
        
        # First pass: fit with global priors
        initial_params = {}
        for (country, brand), group in df.groupby(['country', 'brand_name']):
            t = group[time_col].values
            y = group[target_col].values
            bucket = group['bucket'].iloc[0] if 'bucket' in group.columns else 1
            
            try:
                params = self._fit_brand(t, y)
                initial_params[(country, brand)] = (params, bucket)
            except Exception:
                continue
        
        # Compute global priors from all fitted parameters
        all_params = np.array([p[0] for p in initial_params.values()])
        if len(all_params) > 0:
            self.global_priors_ = {
                'a': (np.mean(all_params[:, 0]), np.std(all_params[:, 0]) + 0.01),
                'b': (np.mean(all_params[:, 1]), np.std(all_params[:, 1]) + 0.001),
                'c': (np.mean(all_params[:, 2]), np.std(all_params[:, 2]) + 0.01)
            }
            logger.info(f"Global priors: a={self.global_priors_['a']}, "
                       f"b={self.global_priors_['b']}, c={self.global_priors_['c']}")
        
        # Compute bucket-specific priors
        for bucket in [1, 2]:
            bucket_params = np.array([
                p[0] for p in initial_params.values() if p[1] == bucket
            ])
            
            if len(bucket_params) >= 3:
                self.bucket_priors_[bucket] = {
                    'a': (np.mean(bucket_params[:, 0]), np.std(bucket_params[:, 0]) + 0.01),
                    'b': (np.mean(bucket_params[:, 1]), np.std(bucket_params[:, 1]) + 0.001),
                    'c': (np.mean(bucket_params[:, 2]), np.std(bucket_params[:, 2]) + 0.01)
                }
                logger.info(f"Bucket {bucket} priors: a={self.bucket_priors_[bucket]['a']}, "
                           f"b={self.bucket_priors_[bucket]['b']}")
            else:
                self.bucket_priors_[bucket] = self.global_priors_.copy()
    
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'y_norm',
        time_col: str = 'months_postgx'
    ) -> 'HierarchicalBayesianDecay':
        """
        Fit decay model for each brand with hierarchical priors.
        
        Two-pass approach:
        1. Estimate population/bucket-level priors (Empirical Bayes)
        2. Refit each brand with hierarchical shrinkage
        
        Args:
            df: Training DataFrame
            target_col: Target column
            time_col: Time column
            
        Returns:
            self
        """
        # Estimate hierarchical priors
        if self.use_hierarchical_priors:
            self._estimate_hierarchical_priors(df, target_col, time_col)
        
        # Second pass: fit with hierarchical priors
        for (country, brand), group in df.groupby(['country', 'brand_name']):
            t = group[time_col].values
            y = group[target_col].values
            bucket = group['bucket'].iloc[0] if 'bucket' in group.columns else 1
            
            # Get appropriate priors
            if self.use_hierarchical_priors and bucket in self.bucket_priors_:
                priors = self.bucket_priors_[bucket]
            elif self.use_hierarchical_priors and self.global_priors_:
                priors = self.global_priors_
            else:
                priors = {'a': self.prior_a, 'b': self.prior_b, 'c': self.prior_c}
            
            try:
                params = self._fit_brand(t, y, priors=priors)
                
                # Apply shrinkage toward group mean
                if self.use_hierarchical_priors and self.shrinkage_strength > 0:
                    prior_means = np.array([priors['a'][0], priors['b'][0], priors['c'][0]])
                    params = tuple(
                        (1 - self.shrinkage_strength) * np.array(params) +
                        self.shrinkage_strength * prior_means
                    )
                
                self.brand_params_[(country, brand)] = params
            except Exception as e:
                # Fallback to prior means
                logger.warning(f"Failed to fit {country}/{brand}: {e}")
                if priors:
                    self.brand_params_[(country, brand)] = (
                        priors['a'][0], priors['b'][0], priors['c'][0]
                    )
                else:
                    self.brand_params_[(country, brand)] = (
                        self.prior_a[0], self.prior_b[0], self.prior_c[0]
                    )
        
        logger.info(f"Fitted hierarchical decay model for {len(self.brand_params_)} brands")
        return self
    
    def predict(
        self,
        df: pd.DataFrame,
        time_col: str = 'months_postgx'
    ) -> np.ndarray:
        """
        Generate predictions using fitted decay models.
        
        Args:
            df: DataFrame with country, brand_name, and time column
            time_col: Time column
            
        Returns:
            Array of predictions
        """
        predictions = np.zeros(len(df))
        
        for idx, row in df.iterrows():
            key = (row['country'], row['brand_name'])
            t = row[time_col]
            
            if key in self.brand_params_:
                a, b, c = self.brand_params_[key]
            else:
                # Use global prior means for unseen brands
                if self.global_priors_:
                    a = self.global_priors_['a'][0]
                    b = self.global_priors_['b'][0]
                    c = self.global_priors_['c'][0]
                else:
                    a, b, c = self.prior_a[0], self.prior_b[0], self.prior_c[0]
            
            y_pred = a * np.exp(-b * t) + c
            predictions[idx if isinstance(idx, int) else df.index.get_loc(idx)] = y_pred
        
        return predictions
    
    def predict_fast(
        self,
        df: pd.DataFrame,
        time_col: str = 'months_postgx'
    ) -> np.ndarray:
        """
        Faster vectorized prediction.
        
        Args:
            df: DataFrame with country, brand_name, and time column
            time_col: Time column
            
        Returns:
            Array of predictions
        """
        predictions = np.zeros(len(df))
        
        # Default params for unseen brands
        if self.global_priors_:
            default_params = (
                self.global_priors_['a'][0],
                self.global_priors_['b'][0],
                self.global_priors_['c'][0]
            )
        else:
            default_params = (self.prior_a[0], self.prior_b[0], self.prior_c[0])
        
        # Create lookup
        df_copy = df.copy()
        df_copy['_idx'] = range(len(df_copy))
        
        for (country, brand), group in df_copy.groupby(['country', 'brand_name']):
            key = (country, brand)
            t = group[time_col].values
            idx = group['_idx'].values
            
            if key in self.brand_params_:
                a, b, c = self.brand_params_[key]
            else:
                a, b, c = default_params
            
            predictions[idx] = a * np.exp(-b * t) + c
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'brand_params': self.brand_params_,
            'bucket_priors': self.bucket_priors_,
            'global_priors': self.global_priors_,
            'prior_a': self.prior_a,
            'prior_b': self.prior_b,
            'prior_c': self.prior_c,
            'use_hierarchical_priors': self.use_hierarchical_priors,
            'shrinkage_strength': self.shrinkage_strength,
        }, path)
        logger.info(f"HierarchicalBayesianDecay saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'HierarchicalBayesianDecay':
        """Load model from disk."""
        data = joblib.load(path)
        
        instance = cls(
            prior_a_mean=data['prior_a'][0],
            prior_a_std=data['prior_a'][1],
            prior_b_mean=data['prior_b'][0],
            prior_b_std=data['prior_b'][1],
            prior_c_mean=data['prior_c'][0],
            prior_c_std=data['prior_c'][1],
            use_hierarchical_priors=data.get('use_hierarchical_priors', True),
            shrinkage_strength=data.get('shrinkage_strength', 0.3),
        )
        instance.brand_params_ = data['brand_params']
        instance.bucket_priors_ = data.get('bucket_priors', {})
        instance.global_priors_ = data.get('global_priors', {})
        
        return instance


def generate_oof_for_hierarchical_decay(
    df_train: pd.DataFrame,
    scenario: int,
    target_col: str = 'y_norm',
    time_col: str = 'months_postgx',
    n_folds: int = 5,
    save_path: Optional[str] = None,
    **decay_kwargs
) -> pd.DataFrame:
    """
    Generate OOF predictions for the Hierarchical Bayesian Decay model.
    
    Args:
        df_train: Training DataFrame
        scenario: 1 or 2
        target_col: Target column
        time_col: Time column
        n_folds: Number of CV folds
        save_path: Optional path to save OOF predictions
        **decay_kwargs: Additional arguments for HierarchicalBayesianDecay
        
    Returns:
        DataFrame with OOF predictions
    """
    # Filter by scenario
    if scenario == 1:
        df = df_train[df_train['months_postgx'].between(0, 23)].copy()
    else:
        df = df_train[df_train['months_postgx'].between(6, 23)].copy()
    
    # Create groups for CV
    groups = df.groupby(['country', 'brand_name']).ngroup()
    
    # Initialize OOF predictions
    oof_preds = np.full(len(df), np.nan)
    
    # GroupKFold CV
    gkf = GroupKFold(n_splits=n_folds)
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        logger.info(f"Hierarchical Decay - Fold {fold_idx + 1}/{n_folds}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Fit model on training fold
        model = HierarchicalBayesianDecay(**decay_kwargs)
        model.fit(train_df, target_col=target_col, time_col=time_col)
        
        # Predict on validation fold
        oof_preds[val_idx] = model.predict_fast(val_df, time_col=time_col)
    
    # Build result DataFrame
    result = df[['country', 'brand_name', 'months_postgx', 'bucket']].copy()
    result['y_true'] = df[target_col].values
    result['y_pred'] = oof_preds
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(save_path, index=False)
        logger.info(f"Hierarchical Decay OOF saved to {save_path}")
    
    return result


def add_hierarchical_decay_to_meta_dataset(
    df_meta: pd.DataFrame,
    df_train: pd.DataFrame,
    scenario: int,
    target_col: str = 'y_norm',
    **decay_kwargs
) -> pd.DataFrame:
    """
    Add hierarchical decay predictions as an extra column to meta-dataset.
    
    Args:
        df_meta: Existing meta-dataset
        df_train: Training DataFrame for fitting decay model
        scenario: 1 or 2
        target_col: Target column
        **decay_kwargs: Arguments for HierarchicalBayesianDecay
        
    Returns:
        Meta-dataset with added pred_bayes_decay column
    """
    # Generate OOF for decay model
    df_decay_oof = generate_oof_for_hierarchical_decay(
        df_train, scenario, target_col=target_col, **decay_kwargs
    )
    
    # Merge with meta-dataset
    df_decay_oof = df_decay_oof[['country', 'brand_name', 'months_postgx', 'y_pred']]
    df_decay_oof = df_decay_oof.rename(columns={'y_pred': 'pred_bayes_decay'})
    
    df_meta_enhanced = df_meta.merge(
        df_decay_oof,
        on=['country', 'brand_name', 'months_postgx'],
        how='left'
    )
    
    # Fill any missing with global prediction
    if df_meta_enhanced['pred_bayes_decay'].isna().any():
        # Fit global model
        model = HierarchicalBayesianDecay(**decay_kwargs)
        model.fit(df_train, target_col=target_col)
        
        # Fill missing
        missing_mask = df_meta_enhanced['pred_bayes_decay'].isna()
        missing_df = df_meta_enhanced[missing_mask][['country', 'brand_name', 'months_postgx']]
        df_meta_enhanced.loc[missing_mask, 'pred_bayes_decay'] = model.predict_fast(missing_df)
    
    logger.info(f"Added pred_bayes_decay to meta-dataset")
    return df_meta_enhanced


# ==============================================================================
# SUBMISSION DIVERSIFICATION (Section 6)
# ==============================================================================

def generate_diversified_submissions(
    stacker: BayesianStacker,
    test_pred_files: Dict[str, str],
    scenario: int,
    n_variants: int = 5,
    noise_std: float = 0.1,
    save_dir: Optional[str] = None
) -> List[pd.DataFrame]:
    """
    Generate diversified ensemble submissions by perturbing weights.
    
    Args:
        stacker: Fitted BayesianStacker
        test_pred_files: Dict mapping model name -> test prediction file path
        scenario: 1 or 2
        n_variants: Number of variants to generate
        noise_std: Standard deviation of noise to add to weights (in log space)
        save_dir: Optional directory to save submissions
        
    Returns:
        List of submission DataFrames
    """
    df_test, X = build_test_meta_dataset(test_pred_files, scenario)
    base_weights = stacker.weights_.copy()
    
    submissions = []
    
    for i in range(n_variants):
        if i == 0:
            # First variant is the original
            weights = base_weights
            variant_name = 'base'
        else:
            # Add noise in log space and re-normalize via softmax
            log_weights = np.log(base_weights + 1e-12)
            np.random.seed(i * 42)
            log_weights_noisy = log_weights + np.random.randn(len(log_weights)) * noise_std
            weights = softmax(log_weights_noisy)
            variant_name = f'variant_{i}'
        
        # Generate predictions
        y_ensemble = X @ weights
        y_ensemble = np.clip(y_ensemble, 0, 2)
        
        # Build result DataFrame
        result = df_test[['country', 'brand_name', 'months_postgx']].copy()
        result['volume'] = y_ensemble
        
        submissions.append(result)
        
        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            result.to_csv(
                save_dir / f'ensemble_scenario{scenario}_{variant_name}.csv',
                index=False
            )
            
            logger.info(f"Saved variant {variant_name} with weights: {weights}")
    
    return submissions


def fit_stacker_multi_init(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    n_inits: int = 10,
    regularization_strength: float = 1.0
) -> List[np.ndarray]:
    """
    Fit stacker from multiple random initializations.
    
    Returns multiple weight vectors that are all local optima,
    enabling diversified submissions.
    
    Args:
        X: (N, M) matrix of base model predictions
        y: (N,) true targets
        sample_weight: (N,) sample weights
        alpha: (M,) Dirichlet prior params
        n_inits: Number of random initializations
        regularization_strength: Prior strength
        
    Returns:
        List of weight arrays from different local optima
    """
    N, M = X.shape
    
    if alpha is None:
        alpha = np.ones(M)
    
    alpha_scaled = alpha * regularization_strength
    
    def loss_fn(z):
        w = softmax(z)
        y_pred = X @ w
        resid = y - y_pred
        mse_part = np.sum(sample_weight * resid ** 2)
        prior_part = -np.sum((alpha_scaled - 1.0) * np.log(w + 1e-12))
        return mse_part + prior_part
    
    all_weights = []
    all_losses = []
    
    for init_seed in range(n_inits):
        np.random.seed(init_seed * 42)
        
        if init_seed == 0:
            z_init = np.zeros(M)  # Uniform start
        elif init_seed <= M:
            # Start biased toward each model in turn
            z_init = np.zeros(M)
            z_init[init_seed - 1] = 2.0
        else:
            z_init = np.random.randn(M) * 0.5
        
        try:
            result = minimize(
                loss_fn,
                z_init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'disp': False}
            )
            
            weights = softmax(result.x)
            all_weights.append(weights)
            all_losses.append(result.fun)
            
        except Exception as e:
            logger.warning(f"Optimization failed with init {init_seed}: {e}")
            continue
    
    # Sort by loss and return unique solutions
    sorted_pairs = sorted(zip(all_losses, all_weights), key=lambda x: x[0])
    
    # Filter to keep only diverse solutions
    unique_weights = []
    for loss, weights in sorted_pairs:
        is_unique = True
        for existing in unique_weights:
            if np.allclose(weights, existing, atol=0.05):
                is_unique = False
                break
        if is_unique:
            unique_weights.append(weights)
    
    logger.info(f"Found {len(unique_weights)} unique weight configurations")
    return unique_weights


def mcmc_sample_weights(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    n_samples: int = 100,
    step_size: float = 0.05,
    burn_in: int = 50,
    thin: int = 5
) -> List[np.ndarray]:
    """
    Light-weight MCMC sampling of ensemble weights.
    
    Uses Metropolis-Hastings with random walk proposals in log-weight space.
    
    Args:
        X: (N, M) matrix of base model predictions
        y: (N,) true targets
        sample_weight: (N,) sample weights
        alpha: (M,) Dirichlet prior params
        n_samples: Number of samples to collect after burn-in
        step_size: Standard deviation of random walk proposal
        burn_in: Number of burn-in steps
        thin: Thinning interval
        
    Returns:
        List of sampled weight arrays
    """
    N, M = X.shape
    
    if alpha is None:
        alpha = np.ones(M)
    
    def log_posterior(z):
        """Negative log posterior (we want to minimize)."""
        w = softmax(z)
        y_pred = X @ w
        resid = y - y_pred
        
        # Likelihood (Gaussian)
        log_lik = -0.5 * np.sum(sample_weight * resid ** 2)
        
        # Prior (Dirichlet)
        log_prior = np.sum((alpha - 1.0) * np.log(w + 1e-12))
        
        return log_lik + log_prior
    
    # Start from MAP solution
    map_weights = fit_dirichlet_weighted_ensemble(X, y, sample_weight, alpha)
    z_current = np.log(map_weights + 1e-12)
    log_p_current = log_posterior(z_current)
    
    samples = []
    n_accepted = 0
    total_steps = burn_in + n_samples * thin
    
    for step in range(total_steps):
        # Propose new z
        z_proposed = z_current + np.random.randn(M) * step_size
        log_p_proposed = log_posterior(z_proposed)
        
        # Accept/reject
        log_accept_ratio = log_p_proposed - log_p_current
        
        if np.log(np.random.rand()) < log_accept_ratio:
            z_current = z_proposed
            log_p_current = log_p_proposed
            n_accepted += 1
        
        # Collect sample after burn-in with thinning
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(softmax(z_current))
    
    accept_rate = n_accepted / total_steps
    logger.info(f"MCMC acceptance rate: {accept_rate:.2%}")
    logger.info(f"Collected {len(samples)} weight samples")
    
    return samples


def generate_bayesian_submission_variants(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray,
    test_pred_files: Dict[str, str],
    scenario: int,
    method: str = 'multi_init',
    n_variants: int = 5,
    alpha: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> List[pd.DataFrame]:
    """
    Generate multiple submission variants using Bayesian methods.
    
    Methods:
    - 'multi_init': Multiple random initializations of MAP optimization
    - 'mcmc': MCMC sampling of weight posterior
    - 'noise': Add noise to MAP weights
    
    Args:
        X_train: (N, M) matrix of OOF predictions
        y_train: (N,) true targets
        sample_weight_train: (N,) sample weights
        test_pred_files: Dict mapping model name -> test prediction file
        scenario: 1 or 2
        method: 'multi_init', 'mcmc', or 'noise'
        n_variants: Number of variants to generate
        alpha: Dirichlet prior parameters
        save_dir: Optional directory to save submissions
        **kwargs: Additional method-specific arguments
        
    Returns:
        List of submission DataFrames
    """
    df_test, X_test = build_test_meta_dataset(test_pred_files, scenario)
    
    # Get weight samples
    if method == 'multi_init':
        weight_samples = fit_stacker_multi_init(
            X_train, y_train, sample_weight_train, alpha,
            n_inits=max(n_variants * 2, 10)
        )[:n_variants]
    
    elif method == 'mcmc':
        weight_samples = mcmc_sample_weights(
            X_train, y_train, sample_weight_train, alpha,
            n_samples=n_variants,
            step_size=kwargs.get('step_size', 0.05),
            burn_in=kwargs.get('burn_in', 50),
            thin=kwargs.get('thin', 5)
        )
    
    elif method == 'noise':
        # Get MAP weights first
        map_weights = fit_dirichlet_weighted_ensemble(
            X_train, y_train, sample_weight_train, alpha
        )
        
        weight_samples = [map_weights]  # First is MAP
        noise_std = kwargs.get('noise_std', 0.1)
        
        for i in range(1, n_variants):
            np.random.seed(i * 42)
            log_w = np.log(map_weights + 1e-12)
            log_w_noisy = log_w + np.random.randn(len(log_w)) * noise_std
            weight_samples.append(softmax(log_w_noisy))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Generate submissions
    submissions = []
    
    for i, weights in enumerate(weight_samples):
        y_ensemble = X_test @ weights
        y_ensemble = np.clip(y_ensemble, 0, 2)
        
        result = df_test[['country', 'brand_name', 'months_postgx']].copy()
        result['volume'] = y_ensemble
        
        submissions.append(result)
        
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            
            variant_name = 'map' if i == 0 else f'{method}_{i}'
            result.to_csv(
                save_dir_path / f'ensemble_scenario{scenario}_{variant_name}.csv',
                index=False
            )
            
            logger.info(f"Saved {variant_name}: weights={weights.round(4)}")
    
    return submissions


def create_blend_of_blends(
    submissions: List[pd.DataFrame],
    weights: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Create a final submission by blending multiple submission variants.
    
    Args:
        submissions: List of submission DataFrames
        weights: Optional weights for each submission (default: uniform)
        
    Returns:
        Blended submission DataFrame
    """
    if weights is None:
        weights = np.ones(len(submissions)) / len(submissions)
    else:
        weights = np.array(weights) / np.sum(weights)
    
    # Start with first submission structure
    result = submissions[0][['country', 'brand_name', 'months_postgx']].copy()
    
    # Weighted average of volumes
    volume_blend = np.zeros(len(result))
    for sub, w in zip(submissions, weights):
        volume_blend += w * sub['volume'].values
    
    result['volume'] = volume_blend
    
    logger.info(f"Created blend of {len(submissions)} submissions")
    return result


# ==============================================================================
# OFFICIAL METRIC EVALUATION (Section 8)
# ==============================================================================

def evaluate_with_official_metric(
    y_pred_norm: np.ndarray,
    df_info: pd.DataFrame,
    df_aux: pd.DataFrame,
    scenario: int
) -> Dict[str, float]:
    """
    Evaluate predictions using the official competition metric.
    
    This function converts normalized predictions back to actual volume
    and computes the official metric1/metric2 scores.
    
    Args:
        y_pred_norm: Normalized predictions (y_norm scale)
        df_info: DataFrame with [country, brand_name, months_postgx, avg_vol_12m]
        df_aux: Auxiliary DataFrame with [country, brand_name, avg_vol, bucket]
        scenario: 1 or 2
        
    Returns:
        Dict with metric scores: {'overall': float, 'bucket1': float, 'bucket2': float}
    """
    try:
        from ..evaluate import compute_metric1, compute_metric2, compute_bucket_metrics
    except ImportError:
        logger.warning("Could not import evaluate module. Using MSE fallback.")
        return {'overall': np.mean((y_pred_norm - df_info['y_norm'].values) ** 2)}
    
    # Convert normalized predictions to actual volume
    # volume = y_norm * avg_vol_12m
    df_pred = df_info[['country', 'brand_name', 'months_postgx']].copy()
    
    if 'avg_vol_12m' in df_info.columns:
        df_pred['volume'] = y_pred_norm * df_info['avg_vol_12m'].values
    elif 'avg_vol' in df_info.columns:
        df_pred['volume'] = y_pred_norm * df_info['avg_vol'].values
    else:
        # Try to get avg_vol from aux
        df_pred = df_pred.merge(df_aux[['country', 'brand_name', 'avg_vol']], 
                                 on=['country', 'brand_name'], how='left')
        df_pred['volume'] = y_pred_norm * df_pred['avg_vol'].values
        df_pred = df_pred.drop(columns=['avg_vol'])
    
    # Get actual volume
    df_actual = df_info[['country', 'brand_name', 'months_postgx']].copy()
    if 'volume' in df_info.columns:
        df_actual['volume'] = df_info['volume'].values
    elif 'y_norm' in df_info.columns and 'avg_vol_12m' in df_info.columns:
        df_actual['volume'] = df_info['y_norm'].values * df_info['avg_vol_12m'].values
    else:
        logger.warning("Cannot compute actual volume. Missing volume or y_norm + avg_vol_12m")
        return {'overall': np.nan}
    
    # Clip negative predictions
    df_pred['volume'] = df_pred['volume'].clip(lower=0)
    
    try:
        metrics = compute_bucket_metrics(df_actual, df_pred, df_aux, scenario)
        return metrics
    except Exception as e:
        logger.error(f"Error computing official metric: {e}")
        return {'overall': np.nan}


def evaluate_oof_predictions(
    df_oof: pd.DataFrame,
    df_aux: pd.DataFrame,
    scenario: int,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Evaluate OOF predictions using official metric.
    
    Args:
        df_oof: OOF DataFrame with [country, brand_name, months_postgx, y_true, y_pred, bucket]
        df_aux: Auxiliary DataFrame
        scenario: 1 or 2
        model_name: Name of model for logging
        
    Returns:
        Dict with evaluation results
    """
    try:
        from ..evaluate import compute_metric1, compute_metric2
    except ImportError:
        # Fallback to simple metrics
        mse = np.mean((df_oof['y_pred'] - df_oof['y_true']) ** 2)
        mae = np.mean(np.abs(df_oof['y_pred'] - df_oof['y_true']))
        return {
            'model': model_name,
            'scenario': scenario,
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
    
    # Need avg_vol to convert to actual volume
    df_eval = df_oof.merge(df_aux[['country', 'brand_name', 'avg_vol']], 
                           on=['country', 'brand_name'], how='left')
    
    # Convert to actual volume
    df_actual = df_eval[['country', 'brand_name', 'months_postgx']].copy()
    df_actual['volume'] = df_eval['y_true'] * df_eval['avg_vol']
    
    df_pred = df_eval[['country', 'brand_name', 'months_postgx']].copy()
    df_pred['volume'] = df_eval['y_pred'] * df_eval['avg_vol']
    df_pred['volume'] = df_pred['volume'].clip(lower=0)
    
    try:
        if scenario == 1:
            official_metric = compute_metric1(df_actual, df_pred, df_aux)
        else:
            official_metric = compute_metric2(df_actual, df_pred, df_aux)
        
        # Also compute per-bucket
        results = {
            'model': model_name,
            'scenario': scenario,
            f'metric{scenario}_official': official_metric,
        }
        
        # Per-bucket metrics
        for bucket in [1, 2]:
            bucket_mask = df_eval['bucket'] == bucket
            if bucket_mask.sum() > 0:
                bucket_mse = np.mean((df_eval.loc[bucket_mask, 'y_pred'] - 
                                      df_eval.loc[bucket_mask, 'y_true']) ** 2)
                results[f'bucket{bucket}_mse'] = bucket_mse
        
        # Additional metrics
        results['mse'] = np.mean((df_oof['y_pred'] - df_oof['y_true']) ** 2)
        results['mae'] = np.mean(np.abs(df_oof['y_pred'] - df_oof['y_true']))
        results['rmse'] = np.sqrt(results['mse'])
        
        logger.info(f"{model_name} Scenario {scenario}: "
                   f"Official Metric = {official_metric:.6f}, "
                   f"RMSE = {results['rmse']:.6f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return {
            'model': model_name,
            'scenario': scenario,
            'error': str(e)
        }


def compare_models_on_oof(
    oof_files: Dict[str, str],
    df_aux: pd.DataFrame,
    scenario: int
) -> pd.DataFrame:
    """
    Compare multiple models using their OOF predictions and official metric.
    
    Args:
        oof_files: Dict mapping model name -> OOF parquet file path
        df_aux: Auxiliary DataFrame
        scenario: 1 or 2
        
    Returns:
        DataFrame with model comparison
    """
    results = []
    
    for model_name, oof_path in oof_files.items():
        try:
            df_oof = pd.read_parquet(oof_path)
            eval_result = evaluate_oof_predictions(df_oof, df_aux, scenario, model_name)
            results.append(eval_result)
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results.append({'model': model_name, 'error': str(e)})
    
    df_comparison = pd.DataFrame(results)
    
    # Sort by official metric (lower is better)
    metric_col = f'metric{scenario}_official'
    if metric_col in df_comparison.columns:
        df_comparison = df_comparison.sort_values(metric_col, ascending=True)
        df_comparison['rank'] = range(1, len(df_comparison) + 1)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Model Comparison - Scenario {scenario}")
    logger.info(f"{'='*60}")
    logger.info(f"\n{df_comparison.to_string()}")
    
    return df_comparison


def evaluate_ensemble_vs_single_models(
    stacker: 'BayesianStacker',
    df_meta: pd.DataFrame,
    df_aux: pd.DataFrame,
    scenario: int
) -> Dict[str, Any]:
    """
    Compare ensemble performance against individual base models.
    
    Args:
        stacker: Trained BayesianStacker
        df_meta: Meta-dataset with OOF predictions from all base models
        df_aux: Auxiliary DataFrame
        scenario: 1 or 2
        
    Returns:
        Dict with comparison results
    """
    pred_cols = [c for c in df_meta.columns if c.startswith('pred_')]
    
    results = []
    
    # Evaluate each base model
    for pred_col in pred_cols:
        model_name = pred_col.replace('pred_', '')
        
        df_oof = df_meta[['country', 'brand_name', 'months_postgx', 'bucket']].copy()
        df_oof['y_true'] = df_meta['y_true']
        df_oof['y_pred'] = df_meta[pred_col]
        
        eval_result = evaluate_oof_predictions(df_oof, df_aux, scenario, model_name)
        results.append(eval_result)
    
    # Evaluate ensemble
    X = df_meta[pred_cols].values
    y_ensemble = stacker.predict(X)
    
    df_oof_ensemble = df_meta[['country', 'brand_name', 'months_postgx', 'bucket']].copy()
    df_oof_ensemble['y_true'] = df_meta['y_true']
    df_oof_ensemble['y_pred'] = y_ensemble
    
    ensemble_result = evaluate_oof_predictions(df_oof_ensemble, df_aux, scenario, 'ENSEMBLE')
    results.append(ensemble_result)
    
    df_comparison = pd.DataFrame(results)
    
    # Sort by official metric
    metric_col = f'metric{scenario}_official'
    if metric_col in df_comparison.columns:
        df_comparison = df_comparison.sort_values(metric_col, ascending=True)
        df_comparison['rank'] = range(1, len(df_comparison) + 1)
    
    # Highlight if ensemble is best
    if metric_col in df_comparison.columns:
        ensemble_metric = df_comparison[df_comparison['model'] == 'ENSEMBLE'][metric_col].values[0]
        best_single = df_comparison[df_comparison['model'] != 'ENSEMBLE'][metric_col].min()
        
        improvement = (best_single - ensemble_metric) / best_single * 100
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Ensemble vs Single Model Comparison - Scenario {scenario}")
        logger.info(f"{'='*60}")
        logger.info(f"Ensemble: {ensemble_metric:.6f}")
        logger.info(f"Best Single: {best_single:.6f}")
        logger.info(f"Improvement: {improvement:.2f}%")
        logger.info(f"\nFull comparison:\n{df_comparison.to_string()}")
    
    return {
        'comparison': df_comparison,
        'ensemble_metric': ensemble_metric if metric_col in df_comparison.columns else None,
        'best_single_metric': best_single if metric_col in df_comparison.columns else None,
        'improvement_pct': improvement if metric_col in df_comparison.columns else None
    }
