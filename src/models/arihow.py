"""
ARIMA + Holt-Winters hybrid model (ARIHOW) for Novartis Datathon 2025.

Combines ARIMA and Holt-Winters forecasts using learned weights.
Falls back to exponential decay baseline when time series is too short
or models fail to converge.

This is a time-series approach that treats each brand independently,
complementing the cross-sectional ML models.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging
import warnings

import numpy as np
import pandas as pd

from .base import BaseModel

logger = logging.getLogger(__name__)


class ARIHOWModel:
    """
    ARIMA + Holt-Winters hybrid model for pharmaceutical erosion forecasting.
    
    For each (country, brand_name) series:
    1. Fit SARIMAX model to capture trends and autocorrelation
    2. Fit Holt-Winters model for exponential smoothing
    3. Learn optimal combination weights via linear regression
    
    Falls back to exponential decay when:
    - History is too short (< min_history_months)
    - ARIMA/HW fails to converge
    - Numerical issues occur
    
    Example usage:
        model = ARIHOWModel(arima_order=(1,1,1), hw_trend='add')
        model.fit(history_df, target_col='volume', min_history_months=6)
        predictions = model.predict_with_decay_fallback(
            brands_df, avg_j_df, months_to_predict=[0,1,...,23]
        )
    """
    
    def __init__(
        self,
        arima_order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        hw_trend: Optional[str] = 'add',
        hw_seasonal: Optional[str] = None,
        hw_seasonal_periods: int = 12,
        weight_window: int = 12,
        suppress_warnings: bool = True
    ):
        """
        Initialize ARIHOW model.
        
        Args:
            arima_order: (p, d, q) for SARIMAX
            seasonal_order: (P, D, Q, m) for seasonal ARIMA
            hw_trend: 'add', 'mul', or None for Holt-Winters trend
            hw_seasonal: 'add', 'mul', or None for HW seasonality
            hw_seasonal_periods: Seasonal period for HW (default 12 months)
            weight_window: Number of recent observations to use for weight fitting
            suppress_warnings: Whether to suppress statsmodels warnings
        """
        self.arima_order = arima_order
        self.seasonal_order = seasonal_order
        self.hw_trend = hw_trend
        self.hw_seasonal = hw_seasonal
        self.hw_seasonal_periods = hw_seasonal_periods
        self.weight_window = weight_window
        self.suppress_warnings = suppress_warnings
        
        # Store fitted models per brand
        self.brand_models: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.is_fitted: bool = False
    
    def _fit_brand_arihow(
        self,
        series: pd.Series,
        brand_key: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Fit ARIMA + Holt-Winters for a single brand and learn combination weights.
        
        Args:
            series: Time series indexed by months_postgx (or similar)
            brand_key: (country, brand_name) tuple for logging
            
        Returns:
            Dict with fitted models, weights, and metadata
        """
        result = {
            'arima_res': None,
            'hw_res': None,
            'beta': np.array([0.5, 0.5]),  # Default equal weights
            'success': False,
            'fallback_value': series.iloc[-1] if len(series) > 0 else 1.0,
            'last_value': series.iloc[-1] if len(series) > 0 else 1.0,
            'series_length': len(series)
        }
        
        if len(series) < 3:
            logger.debug(f"Brand {brand_key}: Too short ({len(series)} points), using fallback")
            return result
        
        try:
            # Suppress statsmodels warnings if requested
            with warnings.catch_warnings():
                if self.suppress_warnings:
                    warnings.filterwarnings('ignore')
                
                # Import here to avoid import errors if statsmodels not installed
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                
                # Fit ARIMA
                try:
                    arima_model = SARIMAX(
                        series,
                        order=self.arima_order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    arima_res = arima_model.fit(disp=False, maxiter=100)
                    result['arima_res'] = arima_res
                except Exception as e:
                    logger.debug(f"Brand {brand_key}: ARIMA fit failed: {e}")
                    arima_res = None
                
                # Fit Holt-Winters
                try:
                    # Need at least 2*seasonal_periods for seasonal HW
                    if self.hw_seasonal and len(series) < 2 * self.hw_seasonal_periods:
                        hw_seasonal = None
                    else:
                        hw_seasonal = self.hw_seasonal
                    
                    hw_model = ExponentialSmoothing(
                        series,
                        trend=self.hw_trend,
                        seasonal=hw_seasonal,
                        seasonal_periods=self.hw_seasonal_periods if hw_seasonal else None
                    )
                    hw_res = hw_model.fit(optimized=True)
                    result['hw_res'] = hw_res
                except Exception as e:
                    logger.debug(f"Brand {brand_key}: HW fit failed: {e}")
                    hw_res = None
                
                # Learn combination weights using recent data
                if arima_res is not None and hw_res is not None:
                    try:
                        # Get in-sample predictions
                        arima_fitted = arima_res.fittedvalues
                        hw_fitted = hw_res.fittedvalues
                        
                        # Use last weight_window observations
                        n_obs = min(len(series), self.weight_window)
                        y_recent = series.iloc[-n_obs:].values
                        arima_recent = arima_fitted.iloc[-n_obs:].values
                        hw_recent = hw_fitted.iloc[-n_obs:].values
                        
                        # Stack predictors
                        X = np.column_stack([arima_recent, hw_recent])
                        
                        # Solve for weights using least squares
                        # Constrain weights to be non-negative
                        from sklearn.linear_model import Ridge
                        lr = Ridge(alpha=0.1, fit_intercept=False, positive=True)
                        lr.fit(X, y_recent)
                        beta = lr.coef_
                        
                        # Normalize weights to sum to 1
                        if beta.sum() > 0:
                            beta = beta / beta.sum()
                        else:
                            beta = np.array([0.5, 0.5])
                        
                        result['beta'] = beta
                        result['success'] = True
                        
                    except Exception as e:
                        logger.debug(f"Brand {brand_key}: Weight learning failed: {e}")
                        # Use equal weights as fallback
                        result['beta'] = np.array([0.5, 0.5])
                        result['success'] = arima_res is not None or hw_res is not None
                
                elif arima_res is not None:
                    result['beta'] = np.array([1.0, 0.0])
                    result['success'] = True
                elif hw_res is not None:
                    result['beta'] = np.array([0.0, 1.0])
                    result['success'] = True
        
        except Exception as e:
            logger.warning(f"Brand {brand_key}: ARIHOW fit failed: {e}")
        
        return result
    
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'volume',
        min_history_months: int = 6
    ) -> 'ARIHOWModel':
        """
        Fit ARIHOW models for all brands in the dataframe.
        
        Args:
            df: DataFrame with columns ['country', 'brand_name', 'months_postgx', target_col]
            target_col: Column containing the target variable
            min_history_months: Minimum history required for ARIHOW (else fallback)
            
        Returns:
            self (fitted model)
        """
        required_cols = ['country', 'brand_name', 'months_postgx', target_col]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Group by brand
        grouped = df.groupby(['country', 'brand_name'], observed=True)
        n_brands = len(grouped)
        
        logger.info(f"Fitting ARIHOW models for {n_brands} brands...")
        
        success_count = 0
        for (country, brand_name), group in grouped:
            brand_key = (country, brand_name)
            
            # Build time series
            series = group.sort_values('months_postgx').set_index('months_postgx')[target_col]
            
            # Check minimum history
            if len(series) < min_history_months:
                self.brand_models[brand_key] = {
                    'success': False,
                    'fallback_value': series.iloc[-1] if len(series) > 0 else 1.0,
                    'last_value': series.iloc[-1] if len(series) > 0 else 1.0,
                    'series_length': len(series),
                    'arima_res': None,
                    'hw_res': None,
                    'beta': np.array([0.5, 0.5])
                }
                continue
            
            # Fit ARIHOW
            model_info = self._fit_brand_arihow(series, brand_key)
            self.brand_models[brand_key] = model_info
            
            if model_info['success']:
                success_count += 1
        
        self.is_fitted = True
        logger.info(f"ARIHOW fitting complete: {success_count}/{n_brands} brands successful")
        
        return self
    
    def _forecast_brand(
        self,
        model_info: Dict[str, Any],
        steps: int
    ) -> np.ndarray:
        """
        Generate forecasts for a single brand using ARIHOW combination.
        
        Args:
            model_info: Dict containing fitted models and weights
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        if not model_info.get('success', False):
            # Return constant (last value) as fallback
            return np.full(steps, model_info.get('fallback_value', 1.0))
        
        arima_res = model_info.get('arima_res')
        hw_res = model_info.get('hw_res')
        beta = model_info.get('beta', np.array([0.5, 0.5]))
        
        forecasts = np.zeros(steps)
        
        try:
            # ARIMA forecast
            if arima_res is not None:
                arima_fc = arima_res.get_forecast(steps=steps).predicted_mean.values
            else:
                arima_fc = np.full(steps, model_info.get('fallback_value', 1.0))
            
            # HW forecast
            if hw_res is not None:
                hw_fc = hw_res.forecast(steps)
                if isinstance(hw_fc, pd.Series):
                    hw_fc = hw_fc.values
            else:
                hw_fc = np.full(steps, model_info.get('fallback_value', 1.0))
            
            # Combine with learned weights
            forecasts = beta[0] * arima_fc + beta[1] * hw_fc
            
            # Clip to non-negative
            forecasts = np.maximum(forecasts, 0)
            
        except Exception as e:
            logger.debug(f"Forecast failed: {e}, using fallback")
            forecasts = np.full(steps, model_info.get('fallback_value', 1.0))
        
        return forecasts
    
    def predict(
        self,
        df: pd.DataFrame,
        months_to_predict: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate predictions for all brands in the dataframe.
        
        Args:
            df: DataFrame with at least ['country', 'brand_name'] columns
            months_to_predict: List of months to forecast (default 0-23)
            
        Returns:
            DataFrame with columns ['country', 'brand_name', 'months_postgx', 'volume']
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if months_to_predict is None:
            months_to_predict = list(range(24))
        
        # Get unique brands
        brands = df[['country', 'brand_name']].drop_duplicates()
        
        results = []
        for _, row in brands.iterrows():
            brand_key = (row['country'], row['brand_name'])
            
            if brand_key not in self.brand_models:
                logger.warning(f"Brand {brand_key} not in fitted models, using fallback")
                for month in months_to_predict:
                    results.append({
                        'country': row['country'],
                        'brand_name': row['brand_name'],
                        'months_postgx': month,
                        'volume': 1.0  # Default fallback
                    })
                continue
            
            model_info = self.brand_models[brand_key]
            forecasts = self._forecast_brand(model_info, len(months_to_predict))
            
            for i, month in enumerate(months_to_predict):
                results.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month,
                    'volume': forecasts[i]
                })
        
        return pd.DataFrame(results)
    
    def predict_with_decay_fallback(
        self,
        df: pd.DataFrame,
        avg_j: pd.DataFrame,
        months_to_predict: Optional[List[int]] = None,
        decay_rate: float = 0.05,
        volume_col: str = 'avg_vol'
    ) -> pd.DataFrame:
        """
        Generate predictions using ARIHOW when successful, decay baseline otherwise.
        
        Args:
            df: DataFrame with ['country', 'brand_name'] for brands to predict
            avg_j: DataFrame with ['country', 'brand_name', volume_col] for fallback
            months_to_predict: Months to forecast (default 0-23)
            decay_rate: Exponential decay rate for fallback
            volume_col: Column name for average volume in avg_j
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if months_to_predict is None:
            months_to_predict = list(range(24))
        
        # Get unique brands
        brands = df[['country', 'brand_name']].drop_duplicates()
        
        # Build avg_j lookup
        avg_j_dict = {}
        for _, row in avg_j.iterrows():
            key = (row['country'], row['brand_name'])
            avg_j_dict[key] = row[volume_col]
        
        results = []
        for _, row in brands.iterrows():
            brand_key = (row['country'], row['brand_name'])
            
            # Check if ARIHOW was successful for this brand
            model_info = self.brand_models.get(brand_key, {})
            use_arihow = model_info.get('success', False)
            
            if use_arihow:
                # Use ARIHOW forecast
                forecasts = self._forecast_brand(model_info, len(months_to_predict))
                for i, month in enumerate(months_to_predict):
                    results.append({
                        'country': row['country'],
                        'brand_name': row['brand_name'],
                        'months_postgx': month,
                        'volume': forecasts[i]
                    })
            else:
                # Use exponential decay fallback
                avg_vol = avg_j_dict.get(brand_key, 1.0)
                for month in months_to_predict:
                    volume = avg_vol * np.exp(-decay_rate * month)
                    results.append({
                        'country': row['country'],
                        'brand_name': row['brand_name'],
                        'months_postgx': month,
                        'volume': volume
                    })
        
        return pd.DataFrame(results)
    
    def get_brand_weights(self) -> pd.DataFrame:
        """
        Get learned combination weights for all brands.
        
        Returns:
            DataFrame with columns:
            ['country', 'brand_name', 'beta_arima', 'beta_hw', 'success', 'series_length']
        """
        results = []
        for (country, brand_name), info in self.brand_models.items():
            beta = info.get('beta', np.array([0.5, 0.5]))
            results.append({
                'country': country,
                'brand_name': brand_name,
                'beta_arima': beta[0],
                'beta_hw': beta[1],
                'success': info.get('success', False),
                'series_length': info.get('series_length', 0)
            })
        
        return pd.DataFrame(results)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get summary statistics about fitted models."""
        if not self.is_fitted:
            return {}
        
        n_total = len(self.brand_models)
        n_success = sum(1 for info in self.brand_models.values() if info.get('success', False))
        
        weights_df = self.get_brand_weights()
        
        return {
            'n_brands': n_total,
            'n_success': n_success,
            'success_rate': n_success / n_total if n_total > 0 else 0,
            'mean_beta_arima': weights_df['beta_arima'].mean(),
            'mean_beta_hw': weights_df['beta_hw'].mean(),
            'mean_series_length': weights_df['series_length'].mean()
        }


class ARIHOWWrapper(BaseModel):
    """
    BaseModel-compliant wrapper for ARIHOWModel.
    
    This wrapper adapts the ARIHOWModel interface to match the standard
    BaseModel interface used by the training pipeline:
    
        fit(X_train, y_train, X_val, y_val, sample_weight)
        predict(X)
    
    The ARIHOWModel is a time-series model that fits separate ARIMA + 
    Holt-Winters models per brand/country series. This wrapper handles
    the data transformation needed to work with the standard interface.
    
    Config options:
        arima_order: Tuple[int, int, int], ARIMA (p, d, q) order (default: (1, 1, 1))
        seasonal_order: Tuple[int, int, int, int], seasonal ARIMA order 
        hw_trend: str, 'add', 'mul', or None (default: 'add')
        hw_seasonal: str, seasonal component type (default: None)
        min_history_months: int, minimum series length (default: 6)
        decay_rate: float, fallback decay rate (default: 0.05)
        
    Example:
        config = {
            'arima_order': [1, 1, 1],
            'hw_trend': 'add',
            'min_history_months': 6
        }
        model = ARIHOWWrapper(config)
        model.fit(X_train, y_train, X_val, y_val, sample_weight)
        predictions = model.predict(X_test)
    """
    
    # Required columns for ARIHOW
    REQUIRED_COLS = ['country', 'brand_name', 'months_postgx']
    
    def __init__(self, config: dict):
        """
        Initialize the wrapper.
        
        Args:
            config: Configuration dict with ARIHOW parameters
        """
        super().__init__(config)
        
        # Extract parameters
        arima_order = tuple(config.get('arima_order', [1, 1, 1]))
        seasonal_order = tuple(config.get('seasonal_order', [0, 0, 0, 0]))
        hw_trend = config.get('hw_trend', 'add')
        hw_seasonal = config.get('hw_seasonal', None)
        hw_seasonal_periods = config.get('hw_seasonal_periods', 12)
        
        self.min_history_months = config.get('min_history_months', 6)
        self.decay_rate = config.get('decay_rate', 0.05)
        
        # Create underlying model
        self._model = ARIHOWModel(
            arima_order=arima_order,
            seasonal_order=seasonal_order,
            hw_trend=hw_trend,
            hw_seasonal=hw_seasonal,
            hw_seasonal_periods=hw_seasonal_periods
        )
        
        self.feature_names: List[str] = []
        self._avg_vol_col = 'avg_vol_12m'
        self._target_col = 'y_norm'
    
    def _validate_data(self, X: pd.DataFrame, context: str = 'data') -> None:
        """Validate that required columns are present."""
        missing = [c for c in self.REQUIRED_COLS if c not in X.columns]
        if missing:
            raise ValueError(
                f"ARIHOWWrapper requires columns {self.REQUIRED_COLS} in {context}. "
                f"Missing: {missing}. Include these in the feature DataFrame."
            )
    
    def _build_history_df(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Build a history DataFrame suitable for ARIHOW fitting.
        
        Returns:
            DataFrame with columns: country, brand_name, months_postgx, volume
        """
        self._validate_data(X, 'X')
        
        history = X[['country', 'brand_name', 'months_postgx']].copy()
        history['volume'] = y.values if isinstance(y, pd.Series) else y
        
        # If we have avg_vol_12m, denormalize the volume
        if self._avg_vol_col in X.columns:
            history['volume'] = history['volume'] * X[self._avg_vol_col].values
        
        return history
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'ARIHOWWrapper':
        """
        Fit the ARIHOW model.
        
        Args:
            X_train: Training features (must include country, brand_name, months_postgx)
            y_train: Training targets (y_norm values)
            X_val: Validation features (unused for ARIHOW)
            y_val: Validation targets (unused for ARIHOW)
            sample_weight: Sample weights (unused for ARIHOW)
            
        Returns:
            self (fitted model)
        """
        # Build history DataFrame
        history = self._build_history_df(X_train, y_train)
        
        # Store feature names (excluding meta columns)
        meta_cols = ['country', 'brand_name', 'months_postgx', self._avg_vol_col]
        self.feature_names = [c for c in X_train.columns if c not in meta_cols]
        
        # Fit the model
        self._model.fit(
            df=history,
            target_col='volume',
            min_history_months=self.min_history_months
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame (must include country, brand_name, months_postgx)
            
        Returns:
            Array of y_norm predictions
        """
        if not self._model.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self._validate_data(X, 'X')
        
        # Get unique (country, brand) pairs and their months to predict
        unique_series = X.groupby(['country', 'brand_name'], observed=True)['months_postgx'].apply(list).reset_index()
        
        # Generate predictions
        results = []
        for _, row in unique_series.iterrows():
            country = row['country']
            brand_name = row['brand_name']
            months = row['months_postgx']
            
            # Ensure months is a list
            if not isinstance(months, list):
                months = [months]
            
            # Get prediction for this series
            series_df = pd.DataFrame({
                'country': [country] * len(months),
                'brand_name': [brand_name] * len(months),
                'months_postgx': months
            })
            
            pred_df = self._model.predict(series_df, months_to_predict=sorted(set(months)))
            
            # Map predictions back to original months order
            pred_dict = {}
            for _, pred_row in pred_df.iterrows():
                pred_dict[pred_row['months_postgx']] = pred_row['volume']
            
            for month in months:
                results.append({
                    'country': country,
                    'brand_name': brand_name,
                    'months_postgx': month,
                    'volume': pred_dict.get(month, 1.0)
                })
        
        # Build result DataFrame with same order as input X
        result_df = pd.DataFrame(results)
        
        # Merge back to get predictions in same order as X
        merged = X[['country', 'brand_name', 'months_postgx']].merge(
            result_df,
            on=['country', 'brand_name', 'months_postgx'],
            how='left'
        )
        
        predictions = merged['volume'].values
        
        # Normalize predictions if we have avg_vol_12m
        if self._avg_vol_col in X.columns:
            avg_vol = X[self._avg_vol_col].values
            # Convert volume to y_norm
            predictions = predictions / np.where(avg_vol > 0, avg_vol, 1.0)
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump({
            'model': self._model,
            'feature_names': self.feature_names,
            'config': self.config,
            '_avg_vol_col': self._avg_vol_col,
            '_target_col': self._target_col
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'ARIHOWWrapper':
        """Load model from disk."""
        import joblib
        data = joblib.load(path)
        instance = cls(data.get('config', {}))
        instance._model = data['model']
        instance.feature_names = data.get('feature_names', [])
        instance._avg_vol_col = data.get('_avg_vol_col', 'avg_vol_12m')
        instance._target_col = data.get('_target_col', 'y_norm')
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (not applicable for ARIHOW)."""
        # ARIHOW doesn't have feature importance in the traditional sense
        return pd.DataFrame({
            'feature': ['arima_component', 'hw_component'],
            'importance': [0.5, 0.5]
        })