import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

class Validator:
    def __init__(self, config: dict):
        self.config = config
        self.cv_config = config.get('cv', {})
        self.strategy = self.cv_config.get('strategy', 'kfold')
        self.n_splits = self.cv_config.get('n_splits', 5)
        self.seed = config.get('reproducibility', {}).get('seed', 42)

    def get_splits(self, df: pd.DataFrame, target_col: str = None):
        """Generate train/val indices based on strategy."""
        if self.strategy == 'time_series':
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            return list(tscv.split(df))
            
        elif self.strategy == 'stratified_kfold':
            if target_col is None:
                raise ValueError("Target column required for StratifiedKFold")
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return list(skf.split(df, df[target_col]))
            
        else: # kfold
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return list(kf.split(df))

    def adversarial_validation(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Check if train and test distributions are similar."""
        # Implementation for adversarial validation
        pass
