import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.metrics = config.get('metrics', {})

    def calculate_metrics(self, y_true, y_pred) -> dict:
        """Calculate metrics defined in config."""
        results = {}
        
        # Always calculate primary metric
        primary = self.metrics.get('primary', 'rmse')
        results[primary] = self._get_metric_func(primary)(y_true, y_pred)
        
        # Calculate secondary metrics
        for metric in self.metrics.get('secondary', []):
            results[metric] = self._get_metric_func(metric)(y_true, y_pred)
            
        return results

    def _get_metric_func(self, name: str):
        if name == 'rmse':
            return lambda y, p: np.sqrt(mean_squared_error(y, p))
        if name == 'mae':
            return mean_absolute_error
        if name == 'r2':
            return r2_score
        if name == 'mape':
            return lambda y, p: np.mean(np.abs((y - p) / y)) * 100
        raise ValueError(f"Unknown metric: {name}")

    def plot_residuals(self, y_true, y_pred, save_path: str = None):
        """Plot residuals."""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
