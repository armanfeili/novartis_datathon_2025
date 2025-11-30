# =============================================================================
# File: src/models.py
# Description: Model classes for baseline and gradient boosting models
# =============================================================================

import warnings
# Suppress statsmodels warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', message='.*unsupported index.*')
warnings.filterwarnings('ignore', message='.*No supported index.*')
warnings.filterwarnings('ignore', message='.*No frequency information.*')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
import joblib
from pathlib import Path
import sys
from datetime import datetime
import shutil

sys.path.insert(0, str(Path(__file__).parent))
import config  # Import module to get live values during multi-config
from config import *  # Also import for convenience


# =============================================================================
# BASELINE MODELS
# =============================================================================

class BaselineModels:
    """Simple baseline models for comparison."""
    
    @staticmethod
    def naive_persistence(avg_j_df: pd.DataFrame, 
                          months_to_predict: list) -> pd.DataFrame:
        """
        Predict avg_j (pre-entry average) for all future months.
        This is the "no erosion" baseline.
        
        Args:
            avg_j_df: DataFrame with avg_vol per brand
            months_to_predict: List of months to predict
            
        Returns:
            DataFrame with predictions
        """
        predictions = []
        for _, row in avg_j_df.iterrows():
            for month in months_to_predict:
                predictions.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month,
                    'volume': row['avg_vol']
                })
        return pd.DataFrame(predictions)
    
    @staticmethod
    def linear_decay(avg_j_df: pd.DataFrame, 
                     months_to_predict: list,
                     decay_rate: float = 0.03) -> pd.DataFrame:
        """
        Predict linear decay from avg_j.
        volume = avg_vol * (1 - decay_rate * months_postgx)
        
        Args:
            avg_j_df: DataFrame with avg_vol per brand
            months_to_predict: List of months to predict
            decay_rate: Monthly decay rate
            
        Returns:
            DataFrame with predictions
        """
        predictions = []
        for _, row in avg_j_df.iterrows():
            for month in months_to_predict:
                decayed_volume = row['avg_vol'] * (1 - decay_rate * month)
                decayed_volume = max(0, decayed_volume)  # No negative volumes
                predictions.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month,
                    'volume': decayed_volume
                })
        return pd.DataFrame(predictions)
    
    @staticmethod
    def exponential_decay(avg_j_df: pd.DataFrame,
                          months_to_predict: list,
                          decay_rate: float = 0.05) -> pd.DataFrame:
        """
        Predict exponential decay from avg_j.
        volume = avg_vol * exp(-decay_rate * months_postgx)
        
        Args:
            avg_j_df: DataFrame with avg_vol per brand
            months_to_predict: List of months to predict
            decay_rate: Decay rate parameter
            
        Returns:
            DataFrame with predictions
        """
        predictions = []
        for _, row in avg_j_df.iterrows():
            for month in months_to_predict:
                decayed_volume = row['avg_vol'] * np.exp(-decay_rate * month)
                predictions.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month,
                    'volume': decayed_volume
                })
        return pd.DataFrame(predictions)
    
    @staticmethod
    def tune_decay_rate(actual_df: pd.DataFrame,
                        avg_j_df: pd.DataFrame,
                        decay_type: str = 'exponential',
                        decay_rates: list = None) -> tuple:
        """
        Tune decay rate on actual data.
        
        Args:
            actual_df: DataFrame with actual volumes
            avg_j_df: DataFrame with avg_vol per brand
            decay_type: 'linear' or 'exponential'
            decay_rates: List of rates to try
            
        Returns:
            Tuple of (best_rate, results_df)
        """
        decay_rates = decay_rates or np.arange(0.01, 0.15, 0.01)
        months = sorted(actual_df['months_postgx'].unique())
        
        results = []
        for rate in decay_rates:
            if decay_type == 'linear':
                preds = BaselineModels.linear_decay(avg_j_df, months, rate)
            else:
                preds = BaselineModels.exponential_decay(avg_j_df, months, rate)
            
            merged = actual_df.merge(
                preds, on=['country', 'brand_name', 'months_postgx'],
                suffixes=('_actual', '_pred')
            )
            mse = mean_squared_error(merged['volume_actual'], merged['volume_pred'])
            mae = mean_absolute_error(merged['volume_actual'], merged['volume_pred'])
            
            results.append({'decay_rate': rate, 'mse': mse, 'mae': mae})
        
        results_df = pd.DataFrame(results)
        best_rate = results_df.loc[results_df['mae'].idxmin(), 'decay_rate']
        
        print(f"✅ Best {decay_type} decay rate: {best_rate:.3f}")
        return best_rate, results_df


# =============================================================================
# GRADIENT BOOSTING MODELS
# =============================================================================

class GradientBoostingModel:
    """LightGBM/XGBoost model wrapper for volume prediction."""
    
    def __init__(self, model_type: str = 'lightgbm', params: dict = None):
        """
        Initialize model.
        
        Args:
            model_type: 'lightgbm' or 'xgboost'
            params: Model hyperparameters (uses defaults if None)
        """
        self.model_type = model_type
        self.model = None
        self.params = params or self._default_params()
        self.feature_names = None
        
    def _default_params(self) -> dict:
        """Get default parameters for model type (reads live from config module)."""
        if self.model_type == 'lightgbm':
            return config.LGBM_PARAMS.copy()  # Read from module for multi-config
        else:
            return config.XGB_PARAMS.copy()  # Read from module for multi-config


# =============================================================================
# HYBRID MODEL (PHYSICS + ML)
# =============================================================================

class HybridPhysicsMLModel:
    """
    Hybrid model combining physics-based exponential decay with ML residual learning.
    
    Formula:
        base_prediction = avg_vol * exp(-decay_rate * months_postgx)  # Physics
        residual = ML_model.predict(features)                          # Learn residuals
        final = base_prediction + residual                             # Combine
    """
    
    def __init__(self, 
                 ml_model_type: str = 'lightgbm',
                 decay_rate: float = 0.05,
                 params: dict = None):
        """
        Initialize hybrid model.
        
        Args:
            ml_model_type: 'lightgbm' or 'xgboost' for residual model
            decay_rate: Exponential decay rate (lambda)
            params: ML model hyperparameters
        """
        self.ml_model_type = ml_model_type
        self.decay_rate = decay_rate
        self.ml_model = None
        self.params = params or self._default_params()
        self.feature_names = None
        self.is_fitted = False
        
    def _default_params(self) -> dict:
        """Get conservative default parameters for residual learning."""
        if self.ml_model_type == 'lightgbm':
            return {
                'n_estimators': 100,
                'max_depth': 4,           # Shallow trees for residuals
                'learning_rate': 0.05,
                'min_child_samples': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbosity': -1
            }
        else:
            return {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.05,
                'min_child_weight': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbosity': 0
            }
    
    def _compute_physics_baseline(self, 
                                   avg_vol: np.ndarray, 
                                   months_postgx: np.ndarray) -> np.ndarray:
        """
        Compute physics-based exponential decay prediction.
        
        Args:
            avg_vol: Pre-entry average volume for each sample
            months_postgx: Months since generic entry
            
        Returns:
            Physics baseline predictions
        """
        return avg_vol * np.exp(-self.decay_rate * months_postgx)
    
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series,
            avg_vol_train: np.ndarray,
            months_train: np.ndarray,
            X_val: pd.DataFrame = None, 
            y_val: pd.Series = None,
            avg_vol_val: np.ndarray = None,
            months_val: np.ndarray = None,
            early_stopping_rounds: int = 50) -> 'HybridPhysicsMLModel':
        """
        Train the hybrid model.
        
        Step 1: Compute physics baseline
        Step 2: Compute residuals (actual - physics)
        Step 3: Train ML model on residuals
        
        Args:
            X_train: Training features
            y_train: Training target (actual volumes)
            avg_vol_train: Pre-entry average volume for training samples
            months_train: Months post generic entry for training samples
            X_val, y_val: Validation data (optional)
            avg_vol_val, months_val: Validation auxiliary data (optional)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Self (fitted model)
        """
        self.feature_names = list(X_train.columns)
        
        # Step 1: Physics baseline
        physics_pred_train = self._compute_physics_baseline(avg_vol_train, months_train)
        
        # Step 2: Residuals
        residuals_train = y_train.values - physics_pred_train
        
        print(f"📊 Physics baseline RMSE: {np.sqrt(np.mean((y_train.values - physics_pred_train)**2)):.2f}")
        print(f"📊 Residual stats: mean={residuals_train.mean():.2f}, std={residuals_train.std():.2f}")
        
        # Step 3: Train ML model on residuals
        if self.ml_model_type == 'lightgbm':
            self.ml_model = lgb.LGBMRegressor(**self.params)
            
            if X_val is not None and y_val is not None and avg_vol_val is not None:
                physics_pred_val = self._compute_physics_baseline(avg_vol_val, months_val)
                residuals_val = y_val.values - physics_pred_val
                
                self.ml_model.fit(
                    X_train, residuals_train,
                    eval_set=[(X_val, residuals_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                )
            else:
                self.ml_model.fit(X_train, residuals_train)
                
        else:  # xgboost
            self.ml_model = xgb.XGBRegressor(**self.params)
            
            if X_val is not None and y_val is not None and avg_vol_val is not None:
                physics_pred_val = self._compute_physics_baseline(avg_vol_val, months_val)
                residuals_val = y_val.values - physics_pred_val
                
                self.ml_model.fit(
                    X_train, residuals_train,
                    eval_set=[(X_val, residuals_val)],
                    verbose=False
                )
            else:
                self.ml_model.fit(X_train, residuals_train)
        
        self.is_fitted = True
        print(f"✅ Hybrid {self.ml_model_type} model trained (decay_rate={self.decay_rate})")
        return self
    
    def predict(self, 
                X: pd.DataFrame, 
                avg_vol: np.ndarray, 
                months_postgx: np.ndarray) -> np.ndarray:
        """
        Generate predictions using hybrid approach.
        
        Args:
            X: Features
            avg_vol: Pre-entry average volume for each sample
            months_postgx: Months since generic entry
            
        Returns:
            Array of predictions (physics + ML residual)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Physics baseline
        physics_pred = self._compute_physics_baseline(avg_vol, months_postgx)
        
        # ML residual prediction
        residual_pred = self.ml_model.predict(X)
        
        # Combine
        final_pred = physics_pred + residual_pred
        
        # Ensure no negative predictions
        final_pred = np.clip(final_pred, 0, None)
        
        return final_pred
    
    def predict_components(self, 
                           X: pd.DataFrame, 
                           avg_vol: np.ndarray, 
                           months_postgx: np.ndarray) -> dict:
        """
        Get prediction components (for analysis).
        
        Returns:
            Dictionary with 'physics', 'residual', 'final' predictions
        """
        physics_pred = self._compute_physics_baseline(avg_vol, months_postgx)
        residual_pred = self.ml_model.predict(X)
        final_pred = np.clip(physics_pred + residual_pred, 0, None)
        
        return {
            'physics': physics_pred,
            'residual': residual_pred,
            'final': final_pred
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from the ML residual model."""
        if not self.is_fitted:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, name: str) -> Path:
        """Save hybrid model to disk with date+timestamp (no overwrites)."""
        date_str = datetime.now().strftime('%Y%m%d')
        time_str = datetime.now().strftime('%H%M%S')
        timestamp = f"{date_str}_{time_str}"
        
        # Always timestamped path (no overwrites)
        timestamped_path = MODELS_DIR / f"{name}_hybrid_{timestamp}.joblib"
        # Latest path (for easy loading)
        latest_path = MODELS_DIR / f"{name}_hybrid.joblib"
        
        save_data = {
            'ml_model': self.ml_model,
            'ml_model_type': self.ml_model_type,
            'decay_rate': self.decay_rate,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'timestamp': timestamp,
            'date': date_str,
            'time': time_str,
            'saved_at': datetime.now().isoformat(),
            'config_snapshot': config.get_current_config_snapshot(),  # Full config for reproducibility
        }
        
        # Save timestamped version (permanent record)
        joblib.dump(save_data, timestamped_path)
        # Save/overwrite latest version (for easy access)
        joblib.dump(save_data, latest_path)
        
        print(f"✅ Hybrid model saved to {timestamped_path}")
        print(f"   (latest copy: {latest_path})")
        print(f"   Config ID: {config.ACTIVE_CONFIG_ID}")
        return timestamped_path
    
    def load(self, name: str) -> 'HybridPhysicsMLModel':
        """Load hybrid model from disk."""
        path = MODELS_DIR / f"{name}_hybrid.joblib"
        data = joblib.load(path)
        self.ml_model = data['ml_model']
        self.ml_model_type = data['ml_model_type']
        self.decay_rate = data['decay_rate']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        print(f"✅ Hybrid model loaded from {path}")
        return self


# =============================================================================
# ARIHOW MODEL (ARIMA + Holt-Winters Hybrid with Learned Weights)
# =============================================================================

import warnings

class ARIHOWModel:
    """
    ARHOW hybrid model with learned weights:
        y_hat = beta0 * y_hat_ARIMA + beta1 * y_hat_HW
    
    This model combines:
    1. SARIMAX for capturing trend, autocorrelation, and seasonality
    2. Holt-Winters (Exponential Smoothing) for level/trend/seasonality patterns
    3. Linear regression to optimally weight the two forecasts
    
    The weights (beta0, beta1) are learned from the last `weight_window` observations
    to minimize prediction error on in-sample data.
    
    Note: This model works per-brand, fitting a time series model to each brand's
    historical data and forecasting future months.
    """
    
    def __init__(self,
                 arima_order: tuple = (1, 1, 1),
                 seasonal_order: tuple = None,
                 hw_trend: str = 'add',
                 hw_seasonal: str = None,
                 hw_seasonal_periods: int = 12,
                 weight_window: int = 12,
                 suppress_warnings: bool = True):
        """
        Initialize ARHOW model.
        
        Args:
            arima_order: (p, d, q) order for ARIMA
                        p = AR order, d = differencing, q = MA order
            seasonal_order: (P, D, Q, s) seasonal order for SARIMAX (optional)
            hw_trend: Holt-Winters trend type ('add', 'mul', None)
            hw_seasonal: Holt-Winters seasonal type ('add', 'mul', None)
            hw_seasonal_periods: Seasonal periods for Holt-Winters
            weight_window: Number of last observations to estimate ARHOW weights
            suppress_warnings: If True, suppress statsmodels warnings
        """
        self.arima_order = arima_order
        self.seasonal_order = seasonal_order or (0, 0, 0, 0)
        self.hw_trend = hw_trend
        self.hw_seasonal = hw_seasonal
        self.hw_seasonal_periods = hw_seasonal_periods
        self.weight_window = weight_window
        self.suppress_warnings = suppress_warnings
        
        self.brand_models = {}  # Store fitted models per brand
        self.is_fitted = False
        
    def _fit_brand_arhow(self, series: pd.Series, brand_key: str) -> dict:
        """
        Fit ARHOW model for a single brand using SARIMAX + Holt-Winters + regression weights.
        
        Args:
            series: Time series data for the brand (indexed by months_postgx)
            brand_key: Identifier for the brand
            
        Returns:
            Dictionary with fitted model info
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from sklearn.linear_model import LinearRegression
        except ImportError:
            raise ImportError("statsmodels and scikit-learn required.")
        
        result = {
            'arima_res': None,
            'hw_res': None,
            'beta': None,  # (beta0, beta1) weights
            'success': False,
            'fallback_value': float(series.mean()) if len(series) > 0 else 0,
            'last_value': float(series.iloc[-1]) if len(series) > 0 else 0
        }
        
        # Need enough data points
        min_points = max(self.weight_window + 2, 8)
        if len(series) < min_points:
            return result
        
        # Use context manager to suppress all statsmodels warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            try:
                # Sort and clean series
                series = series.sort_index().dropna().astype(float)
                
                # Reset to RangeIndex for statsmodels compatibility
                y = pd.Series(series.values, index=pd.RangeIndex(len(series)))
                
                # --- 1. Fit SARIMAX model ---
                arima_model = SARIMAX(
                    y,
                    order=self.arima_order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                arima_res = arima_model.fit(disp=False)
                result['arima_res'] = arima_res
                
                # --- 2. Fit Holt-Winters model ---
                # Determine if we have enough data for seasonality
                use_seasonal = (self.hw_seasonal is not None and 
                               self.hw_seasonal_periods is not None and
                               len(y) >= 2 * self.hw_seasonal_periods)
                
                hw_model = ExponentialSmoothing(
                    y,
                    trend=self.hw_trend,
                    seasonal=self.hw_seasonal if use_seasonal else None,
                    seasonal_periods=self.hw_seasonal_periods if use_seasonal else None,
                    initialization_method='estimated'
                )
                hw_res = hw_model.fit(optimized=True)
                result['hw_res'] = hw_res
                
                # --- 3. Build regression window to estimate weights ---
                effective_window = min(self.weight_window, len(y) - 2)
                if effective_window < 3:
                    # Not enough data for weight estimation, use equal weights
                    result['beta'] = np.array([0.5, 0.5])
                else:
                    start_idx = len(y) - effective_window
                    
                    # In-sample predictions from each model
                    arima_pred = arima_res.fittedvalues.iloc[start_idx:]
                    hw_pred = hw_res.fittedvalues.iloc[start_idx:]
                    y_true = y.iloc[start_idx:]
                    
                    # Align and create regression data
                    df_reg = pd.DataFrame({
                        'y': y_true.values,
                        'arima': arima_pred.values,
                        'hw': hw_pred.values
                    }).dropna()
                    
                    if len(df_reg) >= 3:
                        X = df_reg[['arima', 'hw']].values
                        y_reg = df_reg['y'].values
                        
                        # Estimate weights: y = beta0*arima + beta1*hw (no intercept)
                        lr = LinearRegression(fit_intercept=False)
                        lr.fit(X, y_reg)
                        result['beta'] = lr.coef_  # array([beta0, beta1])
                    else:
                        result['beta'] = np.array([0.5, 0.5])
                
                # Store original index for reference
                result['original_index'] = series.index.tolist()
                result['series_length'] = len(y)
                result['success'] = True
                
            except Exception as e:
                result['error'] = str(e)
            
        return result
    
    def fit(self, 
            df: pd.DataFrame,
            target_col: str = 'volume',
            min_history_months: int = 6) -> 'ARIHOWModel':
        """
        Fit ARHOW models for all brands in the dataset.
        
        Args:
            df: DataFrame with columns [country, brand_name, months_postgx, volume]
            target_col: Target column name
            min_history_months: Minimum months of history required to fit
            
        Returns:
            Self (fitted model)
        """
        print(f"🔧 Fitting ARHOW model (SARIMAX + HW + weights) for each brand...")
        
        # Suppress statsmodels warnings if requested
        if self.suppress_warnings:
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*unsupported index.*')
            warnings.filterwarnings('ignore', message='.*Non-invertible.*')
            warnings.filterwarnings('ignore', message='.*non-stationary.*')
        
        brands = df[['country', 'brand_name']].drop_duplicates()
        n_brands = len(brands)
        success_count = 0
        
        for idx, (_, brand_row) in enumerate(brands.iterrows()):
            brand_key = (brand_row['country'], brand_row['brand_name'])
            
            # Get brand's time series
            brand_data = df[
                (df['country'] == brand_row['country']) & 
                (df['brand_name'] == brand_row['brand_name'])
            ].sort_values('months_postgx')
            
            # Create series indexed by months_postgx
            series = brand_data.set_index('months_postgx')[target_col]
            
            if len(series) >= min_history_months:
                model_result = self._fit_brand_arhow(series, str(brand_key))
                self.brand_models[brand_key] = model_result
                if model_result['success']:
                    success_count += 1
            else:
                # Not enough data - store fallback
                self.brand_models[brand_key] = {
                    'success': False,
                    'fallback_value': float(series.mean()) if len(series) > 0 else 0,
                    'last_value': float(series.iloc[-1]) if len(series) > 0 else 0
                }
            
            if (idx + 1) % 200 == 0:
                print(f"   Processed {idx + 1}/{n_brands} brands...")
        
        self.is_fitted = True
        print(f"✅ ARHOW fitted: {success_count}/{n_brands} brands successfully")
        return self
    
    def _forecast_brand(self, model_info: dict, steps: int) -> np.ndarray:
        """
        Generate multi-step forecast for a single brand using ARHOW hybrid.
        
        Args:
            model_info: Dictionary with fitted model components
            steps: Number of periods to forecast
            
        Returns:
            Array of forecasts
        """
        if not model_info.get('success', False):
            return None
        
        arima_res = model_info['arima_res']
        hw_res = model_info['hw_res']
        beta = model_info['beta']
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            # ARIMA/SARIMAX forecast
            arima_fc = arima_res.forecast(steps=steps)
            
            # Holt-Winters forecast
            hw_fc = hw_res.forecast(steps=steps)
            
            # Ensure same length
            if len(hw_fc) != len(arima_fc):
                hw_fc = hw_fc[:len(arima_fc)]
            
            # Combine with learned weights: y_hat = beta0 * ARIMA + beta1 * HW
            beta0, beta1 = beta
            arhow_fc = beta0 * arima_fc.values + beta1 * hw_fc.values
            
            # Ensure non-negative
            arhow_fc = np.maximum(arhow_fc, 0)
            
            return arhow_fc
    
    def predict(self,
                df: pd.DataFrame,
                months_to_predict: list = None) -> pd.DataFrame:
        """
        Generate predictions for brands.
        
        Args:
            df: DataFrame with brands to predict for
            months_to_predict: List of months to predict (defaults to 0-23)
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        months_to_predict = months_to_predict or list(range(0, 24))
        max_steps = max(months_to_predict) + 1
        predictions = []
        
        brands = df[['country', 'brand_name']].drop_duplicates()
        
        for _, brand_row in brands.iterrows():
            brand_key = (brand_row['country'], brand_row['brand_name'])
            model_info = self.brand_models.get(brand_key, None)
            
            if model_info and model_info.get('success', False):
                # Generate full forecast
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        forecasts = self._forecast_brand(model_info, max_steps)
                        for month in months_to_predict:
                            pred_value = float(forecasts[month]) if forecasts is not None else model_info.get('fallback_value', 0)
                            predictions.append({
                                'country': brand_row['country'],
                                'brand_name': brand_row['brand_name'],
                                'months_postgx': month,
                                'volume': max(0, pred_value)
                            })
                    except:
                        # Fallback
                        for month in months_to_predict:
                            predictions.append({
                                'country': brand_row['country'],
                                'brand_name': brand_row['brand_name'],
                                'months_postgx': month,
                                'volume': model_info.get('fallback_value', 0)
                            })
            else:
                # Use fallback
                fallback = model_info.get('fallback_value', 0) if model_info else 0
                for month in months_to_predict:
                    predictions.append({
                        'country': brand_row['country'],
                        'brand_name': brand_row['brand_name'],
                        'months_postgx': month,
                        'volume': fallback
                    })
        
        return pd.DataFrame(predictions)
    
    def predict_with_decay_fallback(self,
                                     df: pd.DataFrame,
                                     avg_j: pd.DataFrame,
                                     months_to_predict: list = None,
                                     decay_rate: float = 0.05) -> pd.DataFrame:
        """
        Generate predictions with exponential decay fallback for failed brands.
        
        Uses ARHOW where it works, exponential decay otherwise.
        
        Args:
            df: DataFrame with brands to predict
            avg_j: DataFrame with avg_vol per brand
            months_to_predict: List of months
            decay_rate: Fallback decay rate
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        months_to_predict = months_to_predict or list(range(0, 24))
        max_steps = max(months_to_predict) + 1
        predictions = []
        
        brands = df[['country', 'brand_name']].drop_duplicates()
        brands = brands.merge(avg_j[['country', 'brand_name', 'avg_vol']], 
                             on=['country', 'brand_name'], how='left')
        
        arhow_used = 0
        decay_used = 0
        
        for _, brand_row in brands.iterrows():
            brand_key = (brand_row['country'], brand_row['brand_name'])
            avg_vol = brand_row.get('avg_vol', 0)
            if pd.isna(avg_vol):
                avg_vol = 0
            
            model_info = self.brand_models.get(brand_key, None)
            use_arhow = model_info and model_info.get('success', False)
            
            if use_arhow:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        forecasts = self._forecast_brand(model_info, max_steps)
                        if forecasts is not None:
                            for month in months_to_predict:
                                pred_value = float(forecasts[month])
                                predictions.append({
                                    'country': brand_row['country'],
                                    'brand_name': brand_row['brand_name'],
                                    'months_postgx': month,
                                    'volume': max(0, pred_value)
                                })
                                arhow_used += 1
                        else:
                            raise ValueError("Forecast failed")
                    except:
                        # Fallback to decay
                        for month in months_to_predict:
                            pred_value = avg_vol * np.exp(-decay_rate * month)
                            predictions.append({
                                'country': brand_row['country'],
                                'brand_name': brand_row['brand_name'],
                                'months_postgx': month,
                                'volume': max(0, pred_value)
                            })
                            decay_used += 1
            else:
                # Use exponential decay fallback
                for month in months_to_predict:
                    pred_value = avg_vol * np.exp(-decay_rate * month) if avg_vol > 0 else 0
                    predictions.append({
                        'country': brand_row['country'],
                        'brand_name': brand_row['brand_name'],
                        'months_postgx': month,
                        'volume': max(0, pred_value)
                    })
                    decay_used += 1
        
        print(f"   📊 ARHOW predictions: {arhow_used}, Decay fallback: {decay_used}")
        return pd.DataFrame(predictions)
    
    def get_brand_weights(self) -> pd.DataFrame:
        """Get the learned ARHOW weights for each brand."""
        weights = []
        for brand_key, model_info in self.brand_models.items():
            if model_info.get('success', False) and model_info.get('beta') is not None:
                weights.append({
                    'country': brand_key[0],
                    'brand_name': brand_key[1],
                    'beta_arima': model_info['beta'][0],
                    'beta_hw': model_info['beta'][1],
                    'success': True
                })
            else:
                weights.append({
                    'country': brand_key[0],
                    'brand_name': brand_key[1],
                    'beta_arima': None,
                    'beta_hw': None,
                    'success': False
                })
        return pd.DataFrame(weights)
    
    def save(self, name: str) -> Path:
        """Save ARHOW model to disk with date+timestamp (no overwrites)."""
        date_str = datetime.now().strftime('%Y%m%d')
        time_str = datetime.now().strftime('%H%M%S')
        timestamp = f"{date_str}_{time_str}"
        
        # Always timestamped path (no overwrites)
        timestamped_path = MODELS_DIR / f"{name}_arihow_{timestamp}.joblib"
        # Latest path (for easy loading)
        latest_path = MODELS_DIR / f"{name}_arihow.joblib"
        
        # Save configuration and brand weights (statsmodels objects don't serialize well)
        save_data = {
            'arima_order': self.arima_order,
            'seasonal_order': self.seasonal_order,
            'hw_trend': self.hw_trend,
            'hw_seasonal': self.hw_seasonal,
            'hw_seasonal_periods': self.hw_seasonal_periods,
            'weight_window': self.weight_window,
            'is_fitted': self.is_fitted,
            'timestamp': timestamp,
            'date': date_str,
            'time': time_str,
            'saved_at': datetime.now().isoformat(),
            'config_snapshot': config.get_current_config_snapshot(),  # Full config for reproducibility
            # Save weights and fallback values for each brand
            'brand_data': {
                str(k): {
                    'success': v.get('success', False),
                    'fallback_value': v.get('fallback_value', 0),
                    'last_value': v.get('last_value', 0),
                    'beta': v.get('beta', [0.5, 0.5]) if v.get('beta') is not None else [0.5, 0.5]
                }
                for k, v in self.brand_models.items()
            }
        }
        
        # Save timestamped version (permanent record)
        joblib.dump(save_data, timestamped_path)
        # Save/overwrite latest version (for easy access)
        joblib.dump(save_data, latest_path)
        
        print(f"✅ ARHOW model config saved to {timestamped_path}")
        print(f"   (latest copy: {latest_path})")
        print(f"   Config ID: {config.ACTIVE_CONFIG_ID}")
        return timestamped_path
    
    @classmethod
    def load(cls, name: str) -> 'ARIHOWModel':
        """Load ARHOW model from disk."""
        path = MODELS_DIR / f"{name}_arihow.joblib"
        data = joblib.load(path)
        
        model = cls(
            arima_order=data['arima_order'],
            seasonal_order=data['seasonal_order'],
            hw_trend=data['hw_trend'],
            hw_seasonal=data['hw_seasonal'],
            hw_seasonal_periods=data['hw_seasonal_periods'],
            weight_window=data.get('weight_window', 12)
        )
        model.is_fitted = data['is_fitted']
        
        print(f"✅ ARHOW model config loaded from {path}")
        return model


class GradientBoostingModel:
    """LightGBM/XGBoost model wrapper for volume prediction."""
    
    def __init__(self, model_type: str = 'lightgbm', params: dict = None):
        """
        Initialize model.
        
        Args:
            model_type: 'lightgbm' or 'xgboost'
            params: Model hyperparameters (uses defaults if None)
        """
        self.model_type = model_type
        self.model = None
        self.params = params or self._default_params()
        self.feature_names = None
        
    def _default_params(self) -> dict:
        """Get default parameters for model type (reads live from config module)."""
        if self.model_type == 'lightgbm':
            return config.LGBM_PARAMS.copy()  # Read from module for multi-config
        else:
            return config.XGB_PARAMS.copy()  # Read from module for multi-config
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            early_stopping_rounds: int = 50) -> 'GradientBoostingModel':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Self (fitted model)
        """
        self.feature_names = list(X_train.columns)
        
        if self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**self.params)
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                )
            else:
                self.model.fit(X_train, y_train)
                
        else:  # xgboost
            self.model = xgb.XGBRegressor(**self.params)
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        
        print(f"✅ {self.model_type} model trained successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model.predict(X)
        # Ensure no negative predictions
        predictions = np.clip(predictions, 0, None)
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 5) -> dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        if self.model_type == 'lightgbm':
            model = lgb.LGBMRegressor(**self.params)
        else:
            model = xgb.XGBRegressor(**self.params)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores_neg_mse = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        scores_neg_mae = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        results = {
            'rmse_mean': np.sqrt(-scores_neg_mse.mean()),
            'rmse_std': np.sqrt(-scores_neg_mse).std(),
            'mae_mean': -scores_neg_mae.mean(),
            'mae_std': (-scores_neg_mae).std()
        }
        
        print(f"✅ Cross-validation results:")
        print(f"   RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        print(f"   MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        
        return results
    
    def cross_validate_grouped(self, X: pd.DataFrame, y: pd.Series,
                                groups: pd.Series,
                                n_splits: int = 5,
                                stratify_col: pd.Series = None) -> dict:
        """
        Perform cross-validation with GroupKFold to ensure brands stay together.
        
        From Todo Section 4 (Cross-Validation Design):
        - Use GroupKFold or StratifiedGroupKFold for CV
        - Group by brand_name
        - All months for each brand appear in only one fold
        
        Args:
            X: Features DataFrame
            y: Target Series
            groups: Group labels (brand_name) for each row
            n_splits: Number of CV folds
            stratify_col: Optional column for stratification (e.g., bucket)
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
        
        if self.model_type == 'lightgbm':
            base_model = lgb.LGBMRegressor(**self.params)
        else:
            base_model = xgb.XGBRegressor(**self.params)
        
        # Choose CV strategy
        if stratify_col is not None:
            try:
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_splits = cv.split(X, stratify_col, groups)
                cv_name = "StratifiedGroupKFold"
            except:
                # Fall back to GroupKFold if stratification fails
                cv = GroupKFold(n_splits=n_splits)
                cv_splits = cv.split(X, y, groups)
                cv_name = "GroupKFold (stratification failed)"
        else:
            cv = GroupKFold(n_splits=n_splits)
            cv_splits = cv.split(X, y, groups)
            cv_name = "GroupKFold"
        
        # Manual cross-validation to get detailed metrics
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Clone and fit model
            if self.model_type == 'lightgbm':
                model = lgb.LGBMRegressor(**self.params)
            else:
                model = xgb.XGBRegressor(**self.params)
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            # Compute metrics
            rmse = np.sqrt(np.mean((y_val_fold - y_pred) ** 2))
            mae = np.mean(np.abs(y_val_fold - y_pred))
            
            # Count unique brands in fold
            n_train_brands = groups.iloc[train_idx].nunique()
            n_val_brands = groups.iloc[val_idx].nunique()
            
            fold_results.append({
                'fold': fold_idx + 1,
                'rmse': rmse,
                'mae': mae,
                'n_train_brands': n_train_brands,
                'n_val_brands': n_val_brands,
                'n_train_samples': len(train_idx),
                'n_val_samples': len(val_idx)
            })
        
        # Aggregate results
        rmse_values = [r['rmse'] for r in fold_results]
        mae_values = [r['mae'] for r in fold_results]
        
        results = {
            'cv_method': cv_name,
            'n_splits': n_splits,
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
            'mae_mean': np.mean(mae_values),
            'mae_std': np.std(mae_values),
            'fold_details': fold_results
        }
        
        print(f"✅ {cv_name} Cross-validation results ({n_splits} folds):")
        print(f"   RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        print(f"   MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"   Fold details:")
        for r in fold_results:
            print(f"     Fold {r['fold']}: RMSE={r['rmse']:.4f}, "
                  f"Train={r['n_train_brands']} brands, Val={r['n_val_brands']} brands")
        
        return results


# =============================================================================
# CATBOOST MODEL
# =============================================================================

class CatBoostModel:
    """CatBoost wrapper with simple defaults."""

    def __init__(self, params: dict = None):
        self.params = params or {
            'depth': 6,
            'learning_rate': 0.03,
            'iterations': 300,
            'loss_function': 'RMSE',
            'verbose': False,
            'random_seed': RANDOM_STATE,
        }
        self.model = None
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'CatBoostModel':
        self.model = CatBoostRegressor(**self.params)
        self.feature_names = list(X_train.columns)
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], use_best_model=True, verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
        print("✅ CatBoost model trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted.")
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def save(self, name: str) -> Path:
        latest_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({'model': self.model, 'params': self.params, 'feature_names': self.feature_names}, latest_path)
        print(f"✅ CatBoost model saved to {latest_path}")
        return latest_path


# =============================================================================
# LINEAR MODELS
# =============================================================================

class LinearModel:
    """Simple linear model selector."""

    def __init__(self, model_type: str = 'ridge', params: dict = None):
        self.model_type = model_type
        defaults = {
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 0.001, 'max_iter': 5000},
            'elasticnet': {'alpha': 0.001, 'l1_ratio': 0.5, 'max_iter': 5000},
            'huber': {'epsilon': 1.35, 'alpha': 0.0001},
        }
        self.params = params or defaults.get(model_type, defaults['ridge'])
        self.model = None
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LinearModel':
        if self.model_type == 'lasso':
            self.model = Lasso(**self.params)
        elif self.model_type == 'elasticnet':
            self.model = ElasticNet(**self.params)
        elif self.model_type == 'huber':
            self.model = HuberRegressor(**self.params)
        else:
            self.model = Ridge(**self.params)
        self.model.fit(X_train, y_train)
        self.feature_names = list(X_train.columns)
        print(f"✅ Linear model trained ({self.model_type})")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted.")
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def save(self, name: str) -> Path:
        latest_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({'model': self.model, 'params': self.params, 'feature_names': self.feature_names}, latest_path)
        print(f"✅ Linear model saved to {latest_path}")
        return latest_path


# =============================================================================
# SIMPLE NEURAL NETWORK (MLP)
# =============================================================================

class SimpleNNModel:
    """Sklearn MLPRegressor wrapper for quick NN baseline."""

    def __init__(self, params: dict = None):
        self.params = params or {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'random_state': RANDOM_STATE,
        }
        self.model = None
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'SimpleNNModel':
        self.model = MLPRegressor(**self.params)
        self.model.fit(X_train, y_train)
        self.feature_names = list(X_train.columns)
        print("✅ Simple NN (MLP) trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted.")
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def save(self, name: str) -> Path:
        latest_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({'model': self.model, 'params': self.params, 'feature_names': self.feature_names}, latest_path)
        print(f"✅ Simple NN model saved to {latest_path}")
        return latest_path


# =============================================================================
# ADDITIONAL TIME-SERIES MODELS (from other project reference)
# =============================================================================

class SeasonalNaiveModel:
    """Seasonal naive forecaster using last seasonal period (default 12 months)."""

    def __init__(self, season_length: int = 12):
        self.season_length = season_length
        self.history = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = 'volume') -> 'SeasonalNaiveModel':
        series = df.sort_values(['country', 'brand_name', 'months_postgx'])
        self.history = {
            (c, b): grp[target_col].values for (c, b), grp in series.groupby(['country', 'brand_name'])
        }
        self.is_fitted = True
        print(f"✅ Seasonal naive fitted on {len(self.history)} series")
        return self

    def predict(self, df: pd.DataFrame, target_col: str = 'volume') -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        preds = []
        for _, row in df.iterrows():
            key = (row['country'], row['brand_name'])
            hist = self.history.get(key, [])
            if len(hist) == 0:
                pred = 0
            else:
                idx = row['months_postgx'] % max(1, self.season_length)
                pred = hist[-self.season_length:][idx % len(hist[-self.season_length:])]
            preds.append(pred)
        out = df[['country', 'brand_name', 'months_postgx']].copy()
        out[target_col] = np.maximum(preds, 0)
        return out


class SimpleARIMAModel:
    """Per-brand ARIMA wrapper (non-seasonal) using statsmodels."""

    def __init__(self, order: tuple = (1, 1, 1)):
        self.order = order
        self.models = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = 'volume') -> 'SimpleARIMAModel':
        grouped = df.sort_values(['country', 'brand_name', 'months_postgx']).groupby(['country', 'brand_name'])
        for key, grp in grouped:
            try:
                series = grp[target_col].astype(float).values
                if len(series) < sum(self.order):
                    continue
                model = ARIMA(series, order=self.order)
                self.models[key] = model.fit()
            except Exception:
                continue
        self.is_fitted = True
        print(f"✅ ARIMA fitted for {len(self.models)} series")
        return self

    def predict(self, df: pd.DataFrame, target_col: str = 'volume') -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        preds = []
        for _, row in df.iterrows():
            key = (row['country'], row['brand_name'])
            model = self.models.get(key)
            horizon = int(row['months_postgx']) + 1
            if model is None:
                preds.append(0)
                continue
            try:
                fc = model.forecast(steps=horizon)
                preds.append(fc[-1])
            except Exception:
                preds.append(0)
        out = df[['country', 'brand_name', 'months_postgx']].copy()
        out[target_col] = np.maximum(preds, 0)
        return out


class AutoARIMAModel:
    """Auto-ARIMA wrapper using pmdarima (optional dependency)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = 'volume') -> 'AutoARIMAModel':
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError("pmdarima is required for AutoARIMAModel.")

        grouped = df.sort_values(['country', 'brand_name', 'months_postgx']).groupby(['country', 'brand_name'])
        for key, grp in grouped:
            series = grp[target_col].astype(float).values
            if len(series) < 6:
                continue
            try:
                self.models[key] = auto_arima(series, error_action='ignore', suppress_warnings=True, **self.kwargs)
            except Exception:
                continue
        self.is_fitted = True
        print(f"✅ Auto-ARIMA fitted for {len(self.models)} series")
        return self

    def predict(self, df: pd.DataFrame, target_col: str = 'volume') -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        preds = []
        for _, row in df.iterrows():
            key = (row['country'], row['brand_name'])
            model = self.models.get(key)
            horizon = int(row['months_postgx']) + 1
            if model is None:
                preds.append(0)
                continue
            try:
                fc = model.predict(n_periods=horizon)
                preds.append(fc[-1])
            except Exception:
                preds.append(0)
        out = df[['country', 'brand_name', 'months_postgx']].copy()
        out[target_col] = np.maximum(preds, 0)
        return out


class ProphetModel:
    """Per-brand Prophet wrapper (optional dependency)."""

    def __init__(self, params: dict = None):
        self.params = params or {}
        self.models = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = 'volume') -> 'ProphetModel':
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet is required for ProphetModel.")

        grouped = df.sort_values(['country', 'brand_name', 'months_postgx']).groupby(['country', 'brand_name'])
        for key, grp in grouped:
            if len(grp) < 3:
                continue
            tmp = grp[['months_postgx', target_col]].copy()
            tmp.rename(columns={'months_postgx': 'ds', target_col: 'y'}, inplace=True)
            # Convert months_postgx to a pseudo-date to keep Prophet happy
            tmp['ds'] = pd.to_datetime(tmp['ds'], unit='M', origin='unix')
            try:
                m = Prophet(**self.params)
                m.fit(tmp)
                self.models[key] = m
            except Exception:
                continue
        self.is_fitted = True
        print(f"✅ Prophet fitted for {len(self.models)} series")
        return self

    def predict(self, df: pd.DataFrame, target_col: str = 'volume') -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        preds = []
        for _, row in df.iterrows():
            key = (row['country'], row['brand_name'])
            model = self.models.get(key)
            if model is None:
                preds.append(0)
                continue
            horizon = int(row['months_postgx'])
            future = pd.DataFrame({'ds': pd.to_datetime([horizon], unit='M', origin='unix')})
            try:
                fc = model.predict(future)
                preds.append(float(fc['yhat'].iloc[0]))
            except Exception:
                preds.append(0)
        out = df[['country', 'brand_name', 'months_postgx']].copy()
        out[target_col] = np.maximum(preds, 0)
        return out


class LSTMModelPlaceholder:
    """
    Placeholder for LSTM model from the reference project.
    Dependency-heavy (PyTorch/TF); intentionally minimal to avoid breaking installs.
    """

    def __init__(self):
        raise NotImplementedError("LSTM model requires a deep learning stack; integrate separately if needed.")
    
    def save(self, name: str) -> Path:
        """Save model to disk with date+timestamp (no overwrites)."""
        date_str = datetime.now().strftime('%Y%m%d')
        time_str = datetime.now().strftime('%H%M%S')
        timestamp = f"{date_str}_{time_str}"
        
        # Always timestamped path (no overwrites)
        timestamped_path = MODELS_DIR / f"{name}_{timestamp}.joblib"
        # Latest path (for easy loading)
        latest_path = MODELS_DIR / f"{name}.joblib"
        
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'timestamp': timestamp,
            'date': date_str,
            'time': time_str,
            'saved_at': datetime.now().isoformat(),
            'config_snapshot': config.get_current_config_snapshot(),  # Full config for reproducibility
        }
        
        # Save timestamped version (permanent record)
        joblib.dump(save_data, timestamped_path)
        # Save/overwrite latest version (for easy access)
        joblib.dump(save_data, latest_path)
        
        print(f"✅ Model saved to {timestamped_path}")
        print(f"   (latest copy: {latest_path})")
        print(f"   Config ID: {config.ACTIVE_CONFIG_ID}")
        return timestamped_path
    
    def load(self, name: str) -> 'GradientBoostingModel':
        """Load model from disk."""
        path = MODELS_DIR / f"{name}.joblib"
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.params = data['params']
        self.feature_names = data['feature_names']
        print(f"✅ Model loaded from {path}")
        return self


# =============================================================================
# MODEL TRAINING UTILITIES
# =============================================================================

def prepare_training_data(df: pd.DataFrame,
                          feature_cols: list,
                          target_col: str = 'volume',
                          filter_post_entry: bool = True) -> tuple:
    """
    Prepare data for model training.
    
    Args:
        df: Featured dataset
        feature_cols: List of feature columns
        target_col: Target column name
        filter_post_entry: If True, only use post-entry months
        
    Returns:
        Tuple of (X, y)
    """
    data = df.copy()
    
    if filter_post_entry:
        data = data[data['months_postgx'] >= 0]
    
    X = data[feature_cols].fillna(0)
    y = data[target_col]
    
    print(f"✅ Prepared training data: X={X.shape}, y={y.shape}")
    return X, y


# =============================================================================
# ENSEMBLE BLENDING WEIGHTS (Section 3.5 - Hybrid/Ensemble Strategy)
# =============================================================================

class EnsembleBlender:
    """
    Ensemble blender that learns optimal weights for combining model predictions.
    
    From Todo Section 3.5 (Ensemble Strategy):
    - Combine: y_pred = w_phys * y_phys + w_ml * y_ml + w_ts * y_ts
    - Fit blending weights on validation data
    - Constrain weights to sum to 1
    """
    
    def __init__(self, constrain_weights: bool = True):
        """
        Initialize blender.
        
        Args:
            constrain_weights: If True, weights must sum to 1
        """
        self.constrain_weights = constrain_weights
        self.weights = None
        self.model_names = None
        
    def fit(self, predictions: dict, y_true: np.ndarray,
            sample_weights: np.ndarray = None) -> 'EnsembleBlender':
        """
        Learn optimal blending weights from validation data.
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            y_true: True target values
            sample_weights: Optional sample weights
            
        Returns:
            self
        """
        from scipy.optimize import minimize
        
        self.model_names = list(predictions.keys())
        pred_matrix = np.column_stack([predictions[name] for name in self.model_names])
        
        def loss_fn(weights):
            """Weighted MSE loss."""
            blended = pred_matrix @ weights
            errors = (y_true - blended) ** 2
            if sample_weights is not None:
                errors = errors * sample_weights
            return np.mean(errors)
        
        # Initial weights: equal
        n_models = len(self.model_names)
        x0 = np.ones(n_models) / n_models
        
        # Constraints
        constraints = []
        if self.constrain_weights:
            # Weights sum to 1
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(loss_fn, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        self.weights = result.x
        
        # Compute metrics
        blended_pred = pred_matrix @ self.weights
        rmse = np.sqrt(np.mean((y_true - blended_pred) ** 2))
        mae = np.mean(np.abs(y_true - blended_pred))
        
        # Individual model RMSEs
        individual_rmse = {}
        for i, name in enumerate(self.model_names):
            individual_rmse[name] = np.sqrt(np.mean((y_true - pred_matrix[:, i]) ** 2))
        
        print(f"✅ Ensemble blending weights learned:")
        for i, name in enumerate(self.model_names):
            print(f"   {name}: {self.weights[i]:.3f} (solo RMSE: {individual_rmse[name]:.4f})")
        print(f"   Blended RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        
        # Check if blending improves over best single model
        best_single_rmse = min(individual_rmse.values())
        improvement = (best_single_rmse - rmse) / best_single_rmse * 100
        if improvement > 0:
            print(f"   ✅ Blending improves over best single model by {improvement:.1f}%")
        else:
            print(f"   ⚠️ Blending worse than best single model by {-improvement:.1f}%")
        
        return self
    
    def predict(self, predictions: dict) -> np.ndarray:
        """
        Generate blended predictions.
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            
        Returns:
            Blended predictions
        """
        if self.weights is None:
            raise ValueError("Must call fit() before predict()")
        
        pred_matrix = np.column_stack([predictions[name] for name in self.model_names])
        return pred_matrix @ self.weights
    
    def get_weights(self) -> dict:
        """Get learned weights as dictionary."""
        if self.weights is None:
            return {}
        return {name: weight for name, weight in zip(self.model_names, self.weights)}


def optimize_ensemble_weights(val_df: pd.DataFrame,
                               model_predictions: dict,
                               target_col: str = 'volume',
                               sample_weights: np.ndarray = None) -> dict:
    """
    Optimize ensemble weights using validation data.
    
    Convenience function that wraps EnsembleBlender.
    
    Args:
        val_df: Validation DataFrame
        model_predictions: Dict of {model_name: predictions_array}
        target_col: Target column name
        sample_weights: Optional sample weights
        
    Returns:
        Dictionary with optimized weights and metrics
    """
    y_true = val_df[target_col].values
    
    blender = EnsembleBlender(constrain_weights=True)
    blender.fit(model_predictions, y_true, sample_weights)
    
    blended_pred = blender.predict(model_predictions)
    
    return {
        'weights': blender.get_weights(),
        'blended_predictions': blended_pred,
        'blender': blender
    }


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       model_type: str = 'lightgbm') -> tuple:
    """
    Train model and evaluate on validation set.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: 'lightgbm' or 'xgboost'
        
    Returns:
        Tuple of (model, metrics_dict)
    """
    model = GradientBoostingModel(model_type=model_type)
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
        'train_mae': mean_absolute_error(y_train, train_preds),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_preds)),
        'val_mae': mean_absolute_error(y_val, val_preds)
    }
    
    print(f"\n📊 {model_type.upper()} Evaluation:")
    print(f"   Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"   Val RMSE:   {metrics['val_rmse']:.4f}")
    
    return model, metrics


if __name__ == "__main__":
    # Demo: Train a simple model
    print("=" * 60)
    print("MODELS DEMO")
    print("=" * 60)
    
    from data_loader import load_all_data, merge_datasets, split_train_validation
    from bucket_calculator import compute_avg_j
    from feature_engineering import create_all_features, get_feature_columns
    
    # Load and prepare data
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    avg_j = compute_avg_j(merged)
    
    # Create features
    featured = create_all_features(merged, avg_j)
    
    # Split
    train_df, val_df = split_train_validation(featured)
    
    # Prepare training data
    feature_cols = get_feature_columns(featured)
    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_val, y_val = prepare_training_data(val_df, feature_cols)
    
    # Train model
    model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val, 'lightgbm')
    
    # Feature importance
    print("\n📊 Top 10 Features:")
    print(model.get_feature_importance(10))
    
    # Save model
    model.save("demo_lightgbm")
    
    print("\n✅ Models demo complete!")
