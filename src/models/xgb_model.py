import xgboost as xgb
import numpy as np
import joblib
from .base import BaseModel

class XGBModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.params = config.get('params', {})
        self.training_params = config.get('training', {})

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'eval'))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 1000),
            evals=evals,
            early_stopping_rounds=self.params.get('early_stopping_rounds', 50),
            verbose_eval=self.training_params.get('verbose_eval', 100)
        )

    def predict(self, X) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
