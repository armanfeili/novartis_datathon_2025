from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import numpy as np
from .base import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.params = config.get('params', {})
        self.task = config.get('model', {}).get('task', 'regression')
        
        if self.task == 'classification':
            self.model = CatBoostClassifier(**self.params)
        else:
            self.model = CatBoostRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            use_best_model=True,
            verbose=self.params.get('verbose', 100)
        )

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model.load_model(path)
