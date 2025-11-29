"""
Linear and baseline models for Novartis Datathon 2025.

Includes:
- LinearModel: Ridge/Lasso/ElasticNet with preprocessing
- GlobalMeanBaseline: Predict global average erosion curve
- FlatBaseline: Predict no erosion (y_norm = 1.0)
- TrendBaseline: Extrapolate pre-entry trend
- HistoricalCurveBaseline: Match to similar historical series
"""

from typing import Optional, Dict, List, Tuple
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

from .base import BaseModel

logger = logging.getLogger(__name__)


class LinearModel(BaseModel):
    """Linear regression with preprocessing pipeline.
    
    Supports polynomial features for capturing non-linear relationships.
    Automatically handles categorical columns by either dropping them or
    one-hot encoding them (depending on config).
    
    Config options:
        model.type: 'ridge', 'lasso', 'elasticnet', 'huber'
        preprocessing.handle_missing: 'mean', 'median', etc.
        preprocessing.scale_features: True/False
        preprocessing.polynomial_degree: int (2 for quadratic features)
        preprocessing.handle_categoricals: 'drop', 'onehot', 'label' (default: 'drop')
        
    Example config:
        config = {
            'model': {'type': 'ridge'},
            'ridge': {'alpha': 1.0},
            'preprocessing': {
                'scale_features': True,
                'polynomial_degree': 2,  # Add quadratic features
                'handle_categoricals': 'drop'  # Drop categorical columns
            }
        }
    """
    
    # Columns that are known to be categorical
    CATEGORICAL_COLS = ['ther_area', 'main_package', 'time_bucket', 'country']
    
    def __init__(self, config: dict):
        """
        Initialize linear model.
        
        Args:
            config: Configuration with model type and preprocessing settings
        """
        super().__init__(config)
        
        self.model_type = config.get('model', {}).get('type', 'ridge')
        self.params = config.get(self.model_type, {})
        self.preprocessing = config.get('preprocessing', {})
        self.polynomial_degree = self.preprocessing.get('polynomial_degree', None)
        self.handle_categoricals = self.preprocessing.get('handle_categoricals', 'drop')
        
        # Track columns to drop (set during fit)
        self._categorical_cols_to_drop: List[str] = []
        self._onehot_encoder = None
        self._label_encoders: Dict[str, Any] = {}
        
        # Define regressor
        if self.model_type == 'ridge':
            regressor = Ridge(**self.params)
        elif self.model_type == 'lasso':
            regressor = Lasso(**self.params)
        elif self.model_type == 'elasticnet':
            regressor = ElasticNet(**self.params)
        elif self.model_type == 'huber':
            regressor = HuberRegressor(**self.params)
        else:
            raise ValueError(f"Unknown linear model type: {self.model_type}")
        
        # Build pipeline
        steps = []
        if self.preprocessing.get('handle_missing'):
            steps.append(('imputer', SimpleImputer(strategy=self.preprocessing['handle_missing'])))
        
        if self.preprocessing.get('scale_features', True):
            steps.append(('scaler', StandardScaler()))
        
        # Add polynomial features if specified
        if self.polynomial_degree is not None and self.polynomial_degree > 1:
            steps.append(('poly', PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False,
                interaction_only=self.preprocessing.get('interaction_only', False)
            )))
            # Re-scale after polynomial transformation
            steps.append(('post_poly_scaler', StandardScaler()))
        
        steps.append(('model', regressor))
        self.model = Pipeline(steps)
    
    def _preprocess_categoricals(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Handle categorical columns in the feature DataFrame.
        
        Args:
            X: Feature DataFrame
            is_training: Whether this is training (to fit encoders)
            
        Returns:
            DataFrame with categorical columns handled
        """
        X = X.copy()
        
        # Detect categorical columns
        categorical_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_cols.append(col)
            elif col in self.CATEGORICAL_COLS and col in X.columns:
                categorical_cols.append(col)
        
        # Remove duplicates
        categorical_cols = list(set(categorical_cols))
        
        if not categorical_cols:
            return X
        
        if self.handle_categoricals == 'drop':
            # Simply drop categorical columns
            if is_training:
                self._categorical_cols_to_drop = categorical_cols
                logger.info(f"LinearModel: Dropping categorical columns: {categorical_cols}")
            X = X.drop(columns=[c for c in self._categorical_cols_to_drop if c in X.columns])
            
        elif self.handle_categoricals == 'label':
            # Label encode categorical columns
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                if col not in X.columns:
                    continue
                if is_training:
                    le = LabelEncoder()
                    # Handle unknown categories by treating them as a special category
                    X[col] = X[col].fillna('__MISSING__').astype(str)
                    le.fit(X[col])
                    self._label_encoders[col] = le
                else:
                    X[col] = X[col].fillna('__MISSING__').astype(str)
                
                if col in self._label_encoders:
                    le = self._label_encoders[col]
                    # Handle unknown categories
                    known_classes = set(le.classes_)
                    X[col] = X[col].apply(lambda x: x if x in known_classes else '__UNKNOWN__')
                    if '__UNKNOWN__' not in known_classes:
                        # Add unknown to encoder
                        le.classes_ = np.append(le.classes_, '__UNKNOWN__')
                    X[col] = le.transform(X[col])
            
        elif self.handle_categoricals == 'onehot':
            # One-hot encode categorical columns
            from sklearn.preprocessing import OneHotEncoder
            if is_training:
                self._onehot_encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore'
                )
                cat_data = X[categorical_cols].fillna('__MISSING__').astype(str)
                encoded = self._onehot_encoder.fit_transform(cat_data)
                feature_names = self._onehot_encoder.get_feature_names_out(categorical_cols)
            else:
                if self._onehot_encoder is not None:
                    cat_data = X[categorical_cols].fillna('__MISSING__').astype(str)
                    encoded = self._onehot_encoder.transform(cat_data)
                    feature_names = self._onehot_encoder.get_feature_names_out(categorical_cols)
                else:
                    return X
            
            # Drop original categorical columns and add encoded ones
            X = X.drop(columns=categorical_cols)
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
        
        return X
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'LinearModel':
        """
        Train linear model.
        
        Note: Linear models in sklearn don't use validation for early stopping.
        sample_weight is passed to the regressor if supported.
        Categorical columns are handled according to handle_categoricals config.
        """
        # Handle categorical columns
        X_train = self._preprocess_categoricals(X_train, is_training=True)
        
        self.feature_names = list(X_train.columns)
        
        # Fit with sample weights if supported and provided
        if sample_weight is not None:
            # Pipeline fit_params need to be prefixed with step name
            self.model.fit(X_train, y_train, model__sample_weight=sample_weight.values)
        else:
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        # Handle categorical columns the same way as during training
        X = self._preprocess_categoricals(X, is_training=False)
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'handle_categoricals': self.handle_categoricals,
            '_categorical_cols_to_drop': self._categorical_cols_to_drop,
            '_onehot_encoder': self._onehot_encoder,
            '_label_encoders': self._label_encoders,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'LinearModel':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({'model': {'type': data.get('model_type', 'ridge')}})
        instance.model = data['model']
        instance.feature_names = data.get('feature_names', [])
        instance.handle_categoricals = data.get('handle_categoricals', 'drop')
        instance._categorical_cols_to_drop = data.get('_categorical_cols_to_drop', [])
        instance._onehot_encoder = data.get('_onehot_encoder')
        instance._label_encoders = data.get('_label_encoders', {})
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients as importance."""
        if len(self.feature_names) == 0:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Get coefficients from the model step in pipeline
        coefs = self.model.named_steps['model'].coef_
        
        # Handle polynomial features - coefs may be longer than feature_names
        # In this case, we report that we have polynomial features and return
        # the original feature names with their aggregated importance
        if len(coefs) != len(self.feature_names):
            # If polynomial features are used, we can't easily map back
            # Return generic feature names based on coef length
            if len(coefs) > len(self.feature_names):
                # Polynomial features expanded the feature space
                feature_names = [f'feature_{i}' for i in range(len(coefs))]
                logger.info(f"Polynomial features expanded {len(self.feature_names)} -> {len(coefs)} features")
            else:
                # Features were reduced (shouldn't happen, but handle it)
                feature_names = self.feature_names[:len(coefs)]
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coefs)
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(coefs)
        }).sort_values('importance', ascending=False)


class GlobalMeanBaseline(BaseModel):
    """
    Predict using global average erosion curve from training data.
    
    This baseline learns the average y_norm (erosion) for each months_postgx
    from training data and predicts this average for all series.
    
    Note: This model requires 'months_postgx' column in X_train. If using
    the standard training pipeline, ensure months_postgx is included in 
    the feature DataFrame or passed via the train function's special handling.
    
    When 'months_postgx' is not available in X, the model will:
    1. Try to get it from config (if provided as 'months_postgx_values')
    2. Fall back to using the global mean for all predictions
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.erosion_curve: Dict[int, float] = {}
        self.global_mean: float = 0.5
        self._use_global_fallback: bool = False
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
        months_postgx: Optional[np.ndarray] = None  # Accept months separately
    ) -> 'GlobalMeanBaseline':
        """
        Compute mean y_norm per months_postgx.
        
        Args:
            X_train: Feature DataFrame, optionally containing 'months_postgx'
            y_train: Target values
            X_val: Validation features (unused)
            y_val: Validation targets (unused)
            sample_weight: Sample weights (unused for this baseline)
            months_postgx: Optional array of months_postgx values if not in X_train
        
        Returns:
            self (fitted model)
        """
        # Get months_postgx from X_train or from explicit parameter
        if 'months_postgx' in X_train.columns:
            months = X_train['months_postgx'].values
        elif months_postgx is not None:
            months = months_postgx
        else:
            # Fall back to global mean only
            logger.warning(
                "GlobalMeanBaseline: 'months_postgx' not in X_train and not provided separately. "
                "Will predict global mean for all samples. To fix: include 'months_postgx' in "
                "features or pass months_postgx array to fit()."
            )
            self.global_mean = float(y_train.mean())
            self._use_global_fallback = True
            self.erosion_curve = {}
            return self
        
        # Compute mean erosion by month
        df = pd.DataFrame({
            'months_postgx': months,
            'y_norm': y_train.values if isinstance(y_train, pd.Series) else y_train
        })
        
        self.erosion_curve = df.groupby('months_postgx')['y_norm'].mean().to_dict()
        self.global_mean = float(y_train.mean())
        self._use_global_fallback = False
        
        logger.info(f"GlobalMeanBaseline: learned erosion curve for {len(self.erosion_curve)} months")
        
        return self
    
    def predict(
        self, 
        X: pd.DataFrame,
        months_postgx: Optional[np.ndarray] = None  # Accept months separately
    ) -> np.ndarray:
        """
        Apply learned erosion curve.
        
        Args:
            X: Feature DataFrame, optionally containing 'months_postgx'
            months_postgx: Optional array of months_postgx values if not in X
        
        Returns:
            Array of predictions
        """
        # If using global fallback, return global mean for all
        if self._use_global_fallback:
            return np.full(len(X), self.global_mean)
        
        # Get months_postgx from X or from explicit parameter
        if 'months_postgx' in X.columns:
            months = X['months_postgx']
        elif months_postgx is not None:
            months = pd.Series(months_postgx, index=X.index)
        else:
            logger.warning(
                "GlobalMeanBaseline: 'months_postgx' not in X and not provided. "
                "Returning global mean for all predictions."
            )
            return np.full(len(X), self.global_mean)
        
        return months.map(self.erosion_curve).fillna(self.global_mean).values
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'erosion_curve': self.erosion_curve,
            'global_mean': self.global_mean,
            '_use_global_fallback': self._use_global_fallback
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'GlobalMeanBaseline':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({})
        instance.erosion_curve = data['erosion_curve']
        instance.global_mean = data['global_mean']
        instance._use_global_fallback = data.get('_use_global_fallback', False)
        return instance


class FlatBaseline(BaseModel):
    """
    Predict 1.0 (no erosion) as normalized volume.
    
    This is the simplest baseline - predicts that volume remains at pre-entry level.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.prediction_value: float = 1.0
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'FlatBaseline':
        """No fitting needed - always predicts 1.0."""
        logger.info("FlatBaseline: no fitting needed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1.0 for all predictions."""
        return np.ones(len(X)) * self.prediction_value
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({'prediction_value': self.prediction_value}, path)
    
    @classmethod
    def load(cls, path: str) -> 'FlatBaseline':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({})
        instance.prediction_value = data.get('prediction_value', 1.0)
        return instance


class TrendBaseline(BaseModel):
    """
    Extrapolate pre-entry trend into post-entry period.
    
    This baseline uses the pre-entry volume trend (slope) to predict
    future volumes, applying a decay factor.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.decay_factor: float = config.get('decay_factor', 0.95)
        self.global_trend: float = 0.0
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'TrendBaseline':
        """Learn global decay trend from training data."""
        # Compute average y_norm change per month
        if 'months_postgx' in X_train.columns:
            df = pd.DataFrame({
                'months_postgx': X_train['months_postgx'],
                'y_norm': y_train
            })
            # Average slope
            monthly_avg = df.groupby('months_postgx')['y_norm'].mean()
            if len(monthly_avg) > 1:
                self.global_trend = monthly_avg.diff().mean()
        
        logger.info(f"TrendBaseline: learned global trend = {self.global_trend:.4f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using trend extrapolation.
        
        prediction = 1 + months_postgx * global_trend * decay_factor
        """
        if 'months_postgx' not in X.columns:
            return np.ones(len(X))
        
        predictions = 1.0 + X['months_postgx'] * self.global_trend * self.decay_factor
        return np.clip(predictions, 0, 2).values  # Clip to reasonable range
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'decay_factor': self.decay_factor,
            'global_trend': self.global_trend
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'TrendBaseline':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls({})
        instance.decay_factor = data.get('decay_factor', 0.95)
        instance.global_trend = data.get('global_trend', 0.0)
        return instance


class HistoricalCurveBaseline(BaseModel):
    """
    Match test series to similar historical series and use their erosion curves.
    
    This baseline uses K-nearest neighbors to find similar series based on
    pre-entry features and drug characteristics, then uses their averaged
    erosion curves as predictions.
    
    Features used for matching:
    - Drug characteristics: ther_area, hospital_rate, biological, small_molecule
    - Pre-entry statistics: log_avg_vol_12m, pre_entry_trend, pre_entry_volatility
    - Competition: n_gxs at entry
    
    Config options:
        n_neighbors: Number of similar series to average (default: 5)
        metric: Distance metric for KNN (default: 'cosine')
        weights: KNN weights ('uniform' or 'distance', default: 'distance')
        
    Example config:
        config = {
            'n_neighbors': 5,
            'metric': 'cosine',
            'weights': 'distance'
        }
    """
    
    # Features used for matching similar series
    MATCHING_FEATURES = [
        'log_avg_vol_12m', 'pre_entry_trend', 'pre_entry_volatility',
        'hospital_rate', 'n_gxs_at_entry', 'n_gxs'
    ]
    
    # Categorical features that need encoding
    CATEGORICAL_FEATURES = ['ther_area', 'main_package']
    
    def __init__(self, config: dict):
        """
        Initialize historical curve baseline.
        
        Args:
            config: Configuration dict with n_neighbors, metric, weights
        """
        super().__init__(config)
        
        self.n_neighbors = config.get('n_neighbors', 5)
        self.metric = config.get('metric', 'cosine')
        self.weights = config.get('weights', 'distance')
        
        self.knn_model: Optional[NearestNeighbors] = None
        self.scaler: Optional[StandardScaler] = None
        self.train_erosion_curves: Dict[Tuple[str, str], Dict[int, float]] = {}
        self.train_feature_matrix: Optional[np.ndarray] = None
        self.train_series_keys: List[Tuple[str, str]] = []
        self.global_mean: float = 0.5
        self.global_curve: Dict[int, float] = {}
        self._categorical_encodings: Dict[str, Dict[str, int]] = {}
        self._available_features: List[str] = []
    
    def _encode_categorical(
        self, 
        df: pd.DataFrame, 
        col: str, 
        is_training: bool = False
    ) -> np.ndarray:
        """Encode categorical column to integers."""
        if is_training:
            unique_vals = df[col].unique()
            self._categorical_encodings[col] = {
                val: idx for idx, val in enumerate(unique_vals)
            }
        
        encoding = self._categorical_encodings.get(col, {})
        default_val = len(encoding)  # Unknown category gets new index
        
        return np.array([encoding.get(v, default_val) for v in df[col]])
    
    def _extract_series_features(
        self, 
        X: pd.DataFrame, 
        is_training: bool = False
    ) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """
        Extract per-series features for KNN matching.
        
        Args:
            X: Feature DataFrame with multiple rows per series
            is_training: Whether this is training (to learn encodings)
            
        Returns:
            Feature matrix (n_series x n_features), list of (country, brand_name) keys
        """
        # Identify which features are available
        available = []
        for feat in self.MATCHING_FEATURES:
            if feat in X.columns:
                available.append(feat)
        
        if is_training:
            self._available_features = available
        
        # Check for required ID columns
        has_ids = 'country' in X.columns and 'brand_name' in X.columns
        
        if not has_ids:
            # Fallback: use all rows as individual "series"
            logger.warning("No country/brand_name columns - treating each row as a series")
            series_keys = [(str(i), '') for i in range(len(X))]
            
            feature_data = []
            for feat in self._available_features:
                feature_data.append(X[feat].values)
            
            # Add categorical features
            for cat_col in self.CATEGORICAL_FEATURES:
                if cat_col in X.columns:
                    encoded = self._encode_categorical(X, cat_col, is_training)
                    feature_data.append(encoded)
            
            if not feature_data:
                return np.zeros((len(X), 1)), series_keys
            
            return np.column_stack(feature_data), series_keys
        
        # Group by series and extract features
        series_data = []
        series_keys = []
        
        for (country, brand), group in X.groupby(['country', 'brand_name']):
            series_keys.append((country, brand))
            
            row_features = []
            for feat in self._available_features:
                if feat in group.columns:
                    # Use first available value (they should be same for static features)
                    val = group[feat].iloc[0] if len(group) > 0 else 0.0
                    row_features.append(val if not pd.isna(val) else 0.0)
            
            # Add categorical features
            for cat_col in self.CATEGORICAL_FEATURES:
                if cat_col in group.columns:
                    encoded = self._encode_categorical(
                        group.head(1), cat_col, is_training
                    )[0]
                    row_features.append(encoded)
            
            series_data.append(row_features)
        
        if not series_data:
            return np.zeros((1, 1)), series_keys
        
        return np.array(series_data), series_keys
    
    def _extract_erosion_curves(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[Tuple[str, str], Dict[int, float]]:
        """
        Extract erosion curves per series.
        
        Returns:
            Dict mapping (country, brand_name) to {months_postgx: y_norm}
        """
        if 'months_postgx' not in X.columns:
            return {}
        
        has_ids = 'country' in X.columns and 'brand_name' in X.columns
        
        if not has_ids:
            logger.warning("No country/brand_name columns - cannot extract series curves")
            return {}
        
        curves = {}
        
        df = X[['country', 'brand_name', 'months_postgx']].copy()
        df['y_norm'] = y.values
        
        for (country, brand), group in df.groupby(['country', 'brand_name']):
            curve = group.set_index('months_postgx')['y_norm'].to_dict()
            curves[(country, brand)] = curve
        
        return curves
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'HistoricalCurveBaseline':
        """
        Fit the historical curve baseline.
        
        Learns:
        1. Feature matrix for KNN matching
        2. Erosion curves for each training series
        3. Global average curve as fallback
        """
        # Extract series-level features
        self.train_feature_matrix, self.train_series_keys = self._extract_series_features(
            X_train, is_training=True
        )
        
        # Handle missing/nan values
        self.train_feature_matrix = np.nan_to_num(self.train_feature_matrix, nan=0.0)
        
        # Scale features
        self.scaler = StandardScaler()
        self.train_feature_matrix = self.scaler.fit_transform(self.train_feature_matrix)
        
        # Fit KNN model
        n_neighbors = min(self.n_neighbors, len(self.train_series_keys))
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=self.metric,
            algorithm='auto'
        )
        self.knn_model.fit(self.train_feature_matrix)
        
        # Extract erosion curves
        self.train_erosion_curves = self._extract_erosion_curves(X_train, y_train)
        
        # Compute global statistics
        self.global_mean = y_train.mean()
        
        if 'months_postgx' in X_train.columns:
            df = pd.DataFrame({
                'months_postgx': X_train['months_postgx'],
                'y_norm': y_train
            })
            self.global_curve = df.groupby('months_postgx')['y_norm'].mean().to_dict()
        
        n_curves = len(self.train_erosion_curves)
        logger.info(f"HistoricalCurveBaseline: fitted with {n_curves} historical curves, "
                   f"n_neighbors={n_neighbors}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict by matching to similar historical series.
        
        For each test series:
        1. Find k nearest neighbors in training data
        2. Average their erosion curves (weighted by distance)
        3. Return predictions for each row based on months_postgx
        """
        if self.knn_model is None:
            return np.ones(len(X)) * self.global_mean
        
        # Extract features for test series
        test_features, test_series_keys = self._extract_series_features(
            X, is_training=False
        )
        test_features = np.nan_to_num(test_features, nan=0.0)
        test_features = self.scaler.transform(test_features)
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(test_features)
        
        # Build prediction curves for each test series
        series_pred_curves: Dict[Tuple[str, str], Dict[int, float]] = {}
        
        for i, (test_country, test_brand) in enumerate(test_series_keys):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            # Compute weights (inverse distance)
            if self.weights == 'distance':
                # Add small epsilon to avoid division by zero
                weights = 1.0 / (neighbor_distances + 1e-6)
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(neighbor_indices)) / len(neighbor_indices)
            
            # Aggregate erosion curves from neighbors
            aggregated_curve: Dict[int, float] = {}
            weight_sums: Dict[int, float] = {}
            
            for j, idx in enumerate(neighbor_indices):
                if idx < len(self.train_series_keys):
                    train_key = self.train_series_keys[idx]
                    curve = self.train_erosion_curves.get(train_key, {})
                    
                    for month, y_val in curve.items():
                        if month not in aggregated_curve:
                            aggregated_curve[month] = 0.0
                            weight_sums[month] = 0.0
                        aggregated_curve[month] += weights[j] * y_val
                        weight_sums[month] += weights[j]
            
            # Normalize by weights
            for month in aggregated_curve:
                if weight_sums[month] > 0:
                    aggregated_curve[month] /= weight_sums[month]
            
            series_pred_curves[(test_country, test_brand)] = aggregated_curve
        
        # Generate predictions for each row
        predictions = np.ones(len(X)) * self.global_mean
        
        has_ids = 'country' in X.columns and 'brand_name' in X.columns
        has_months = 'months_postgx' in X.columns
        
        if has_ids and has_months:
            for i, row in X.iterrows():
                country = row['country']
                brand = row['brand_name']
                month = row['months_postgx']
                
                curve = series_pred_curves.get((country, brand), {})
                
                if month in curve:
                    predictions[X.index.get_loc(i)] = curve[month]
                elif month in self.global_curve:
                    predictions[X.index.get_loc(i)] = self.global_curve[month]
        elif has_months:
            # Use global curve when no series info
            for i, row in X.iterrows():
                month = row['months_postgx']
                if month in self.global_curve:
                    predictions[X.index.get_loc(i)] = self.global_curve[month]
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'weights': self.weights,
            'knn_model': self.knn_model,
            'scaler': self.scaler,
            'train_erosion_curves': self.train_erosion_curves,
            'train_feature_matrix': self.train_feature_matrix,
            'train_series_keys': self.train_series_keys,
            'global_mean': self.global_mean,
            'global_curve': self.global_curve,
            '_categorical_encodings': self._categorical_encodings,
            '_available_features': self._available_features,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'HistoricalCurveBaseline':
        """Load model from disk."""
        data = joblib.load(path)
        
        config = {
            'n_neighbors': data.get('n_neighbors', 5),
            'metric': data.get('metric', 'cosine'),
            'weights': data.get('weights', 'distance'),
        }
        instance = cls(config)
        
        instance.knn_model = data.get('knn_model')
        instance.scaler = data.get('scaler')
        instance.train_erosion_curves = data.get('train_erosion_curves', {})
        instance.train_feature_matrix = data.get('train_feature_matrix')
        instance.train_series_keys = data.get('train_series_keys', [])
        instance.global_mean = data.get('global_mean', 0.5)
        instance.global_curve = data.get('global_curve', {})
        instance._categorical_encodings = data.get('_categorical_encodings', {})
        instance._available_features = data.get('_available_features', [])
        
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance for matching.
        
        Since this is KNN-based, returns the features used for matching
        with uniform importance.
        """
        features = self._available_features + [
            f"{col}_encoded" for col in self.CATEGORICAL_FEATURES
        ]
        
        if not features:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        return pd.DataFrame({
            'feature': features,
            'importance': [1.0 / len(features)] * len(features)
        })
