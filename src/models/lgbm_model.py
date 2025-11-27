import lightgbm as lgb
import numpy as np
import joblib
from .base import BaseModel

class LGBMModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.params = config.get('params', {})
        self.training_params = config.get('training', {})

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        train_set = lgb.Dataset(X_train, y_train)
        valid_sets = [train_set]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, y_val, reference=train_set)
            valid_sets.append(val_set)

        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.params.get('early_stopping_rounds', 50)),
                lgb.log_evaluation(period=self.training_params.get('verbose_eval', 100))
            ]
        )

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
