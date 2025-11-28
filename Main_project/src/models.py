# =============================================================================
# File: src/models.py
# Description: Model classes for baseline and gradient boosting models
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import *


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
        """Get default parameters for model type."""
        if self.model_type == 'lightgbm':
            return LGBM_PARAMS.copy()
        else:
            return XGB_PARAMS.copy()
    
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
    
    def save(self, name: str) -> Path:
        """Save model to disk."""
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names
        }, path)
        print(f"✅ Model saved to {path}")
        return path
    
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
