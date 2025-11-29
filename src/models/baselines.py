"""
Baseline models for Novartis Datathon 2025.

Implements simple deterministic forecasting baselines:
- Naive persistence (flat forecast)
- Linear decay
- Exponential decay
- Decay rate tuning

These serve as sanity checks and lower bounds for model performance.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class BaselineModels:
    """
    Collection of baseline forecasting methods for pharmaceutical erosion.
    
    All methods are static and pure functions - no model state to save/load.
    Input/output format matches the competition structure.
    
    Example usage:
        # Prepare avg_j dataframe with pre-LOE average volumes
        avg_j_df = train_panel.groupby(['country', 'brand_name']).agg(
            avg_vol=('volume', 'mean')  # or pre-computed avg_vol_12m
        ).reset_index()
        
        # Generate predictions
        months = list(range(0, 24))  # Scenario 1
        preds_naive = BaselineModels.naive_persistence(avg_j_df, months)
        preds_exp = BaselineModels.exponential_decay(avg_j_df, months, decay_rate=0.05)
    """
    
    @staticmethod
    def naive_persistence(
        avg_j_df: pd.DataFrame,
        months_to_predict: List[int],
        volume_col: str = 'avg_vol'
    ) -> pd.DataFrame:
        """
        Naive persistence baseline: predict constant volume for all future months.
        
        Assumes no erosion - the simplest possible baseline.
        
        Args:
            avg_j_df: DataFrame with columns ['country', 'brand_name', volume_col]
                      where volume_col is the pre-LOE average volume
            months_to_predict: List of months to forecast (e.g., [0, 1, ..., 23])
            volume_col: Name of the volume column in avg_j_df
            
        Returns:
            DataFrame with columns ['country', 'brand_name', 'months_postgx', 'volume']
        """
        # Validate input
        required_cols = ['country', 'brand_name', volume_col]
        missing = set(required_cols) - set(avg_j_df.columns)
        if missing:
            raise ValueError(f"Missing columns in avg_j_df: {missing}")
        
        # Create all (brand, month) combinations
        results = []
        for _, row in avg_j_df.iterrows():
            for month in months_to_predict:
                results.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month,
                    'volume': row[volume_col]  # Constant prediction
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def linear_decay(
        avg_j_df: pd.DataFrame,
        months_to_predict: List[int],
        decay_rate: float = 0.03,
        volume_col: str = 'avg_vol'
    ) -> pd.DataFrame:
        """
        Linear decay baseline: volume decreases linearly with time.
        
        Formula: volume = avg_vol * (1 - decay_rate * months_postgx)
        Clipped at 0 to avoid negative volumes.
        
        Args:
            avg_j_df: DataFrame with columns ['country', 'brand_name', volume_col]
            months_to_predict: List of months to forecast
            decay_rate: Monthly decay rate (default 0.03 = 3% per month)
            volume_col: Name of the volume column
            
        Returns:
            DataFrame with columns ['country', 'brand_name', 'months_postgx', 'volume']
        """
        required_cols = ['country', 'brand_name', volume_col]
        missing = set(required_cols) - set(avg_j_df.columns)
        if missing:
            raise ValueError(f"Missing columns in avg_j_df: {missing}")
        
        results = []
        for _, row in avg_j_df.iterrows():
            avg_vol = row[volume_col]
            for month in months_to_predict:
                # Linear decay formula, clipped at 0
                volume = avg_vol * max(0, 1 - decay_rate * month)
                results.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month,
                    'volume': volume
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def exponential_decay(
        avg_j_df: pd.DataFrame,
        months_to_predict: List[int],
        decay_rate: float = 0.05,
        volume_col: str = 'avg_vol'
    ) -> pd.DataFrame:
        """
        Exponential decay baseline: volume decreases exponentially.
        
        Formula: volume = avg_vol * exp(-decay_rate * months_postgx)
        
        This is often more realistic than linear decay as it:
        - Never goes negative
        - Shows faster initial erosion, slower later
        
        Args:
            avg_j_df: DataFrame with columns ['country', 'brand_name', volume_col]
            months_to_predict: List of months to forecast
            decay_rate: Monthly decay rate (default 0.05)
            volume_col: Name of the volume column
            
        Returns:
            DataFrame with columns ['country', 'brand_name', 'months_postgx', 'volume']
        """
        required_cols = ['country', 'brand_name', volume_col]
        missing = set(required_cols) - set(avg_j_df.columns)
        if missing:
            raise ValueError(f"Missing columns in avg_j_df: {missing}")
        
        results = []
        for _, row in avg_j_df.iterrows():
            avg_vol = row[volume_col]
            for month in months_to_predict:
                # Exponential decay formula
                volume = avg_vol * np.exp(-decay_rate * month)
                results.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month,
                    'volume': volume
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def tune_decay_rate(
        actual_df: pd.DataFrame,
        avg_j_df: pd.DataFrame,
        decay_type: str = 'exponential',
        decay_rates: Optional[List[float]] = None,
        volume_col: str = 'avg_vol',
        target_col: str = 'volume'
    ) -> Tuple[float, pd.DataFrame]:
        """
        Find optimal decay rate by grid search on actual data.
        
        Args:
            actual_df: DataFrame with true volumes
                       Columns: ['country', 'brand_name', 'months_postgx', target_col]
            avg_j_df: DataFrame with pre-LOE averages
                      Columns: ['country', 'brand_name', volume_col]
            decay_type: 'linear' or 'exponential'
            decay_rates: List of decay rates to try (default: 0.01 to 0.15)
            volume_col: Column name for avg volume in avg_j_df
            target_col: Column name for actual volume in actual_df
            
        Returns:
            (best_decay_rate, results_df) where results_df has columns:
            ['decay_rate', 'mse', 'rmse', 'mae', 'mape']
        """
        if decay_rates is None:
            decay_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
        
        # Get unique months from actual data
        months_to_predict = sorted(actual_df['months_postgx'].unique().tolist())
        
        # Select decay function
        if decay_type == 'linear':
            decay_fn = BaselineModels.linear_decay
        elif decay_type == 'exponential':
            decay_fn = BaselineModels.exponential_decay
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}. Use 'linear' or 'exponential'.")
        
        results = []
        for rate in decay_rates:
            # Generate predictions
            preds_df = decay_fn(avg_j_df, months_to_predict, decay_rate=rate, volume_col=volume_col)
            
            # Merge with actual values
            merged = actual_df.merge(
                preds_df,
                on=['country', 'brand_name', 'months_postgx'],
                how='inner',
                suffixes=('_actual', '_pred')
            )
            
            if len(merged) == 0:
                continue
            
            y_true = merged[target_col].values if target_col in merged.columns else merged[f'{target_col}_actual'].values
            y_pred = merged['volume'].values if 'volume' in merged.columns else merged['volume_pred'].values
            
            # Compute metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            
            # MAPE (avoid division by zero)
            mask = y_true > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan
            
            results.append({
                'decay_rate': rate,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            })
        
        results_df = pd.DataFrame(results)
        
        # Find best rate (minimum RMSE)
        if len(results_df) > 0:
            best_idx = results_df['rmse'].idxmin()
            best_rate = results_df.loc[best_idx, 'decay_rate']
        else:
            best_rate = 0.05  # Default fallback
        
        return best_rate, results_df
    
    @staticmethod
    def bucket_specific_decay(
        avg_j_df: pd.DataFrame,
        months_to_predict: List[int],
        bucket_decay_rates: dict,
        volume_col: str = 'avg_vol',
        bucket_col: str = 'bucket'
    ) -> pd.DataFrame:
        """
        Apply different exponential decay rates per bucket.
        
        Bucket 1 (fast erosion) typically has higher decay rates than Bucket 2.
        
        Args:
            avg_j_df: DataFrame with columns ['country', 'brand_name', volume_col, bucket_col]
            months_to_predict: List of months to forecast
            bucket_decay_rates: Dict mapping bucket -> decay_rate
                               e.g., {1: 0.08, 2: 0.03}
            volume_col: Name of the volume column
            bucket_col: Name of the bucket column
            
        Returns:
            DataFrame with predictions
        """
        required_cols = ['country', 'brand_name', volume_col, bucket_col]
        missing = set(required_cols) - set(avg_j_df.columns)
        if missing:
            raise ValueError(f"Missing columns in avg_j_df: {missing}")
        
        results = []
        for _, row in avg_j_df.iterrows():
            avg_vol = row[volume_col]
            bucket = row[bucket_col]
            decay_rate = bucket_decay_rates.get(bucket, 0.05)  # Default decay
            
            for month in months_to_predict:
                volume = avg_vol * np.exp(-decay_rate * month)
                results.append({
                    'country': row['country'],
                    'brand_name': row['brand_name'],
                    'months_postgx': month,
                    'volume': volume
                })
        
        return pd.DataFrame(results)
