import numpy as np
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .base import BaseModel

class LinearModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_type = config.get('model', {}).get('type', 'ridge')
        self.params = config.get(self.model_type, {})
        self.preprocessing = config.get('preprocessing', {})
        
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
        
        if self.preprocessing.get('scale_features'):
            steps.append(('scaler', StandardScaler()))
            
        steps.append(('model', regressor))
        self.model = Pipeline(steps)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Linear models in sklearn don't typically use validation set for early stopping
        self.model.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
