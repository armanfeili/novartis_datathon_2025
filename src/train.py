import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from .utils import setup_logging, load_config, set_seed, timer
from .data import DataManager
from .features import FeatureEngineer
from .validation import Validator
from .models.lgbm_model import LGBMModel
from .models.xgb_model import XGBModel
from .models.cat_model import CatBoostModel
from .models.linear import LinearModel
from .models.nn import NNModel
from .evaluate import Evaluator

def get_model_class(model_name):
    if model_name == 'lightgbm': return LGBMModel
    if model_name == 'xgboost': return XGBModel
    if model_name == 'catboost': return CatBoostModel
    if model_name == 'linear': return LinearModel
    if model_name == 'neural_network': return NNModel
    raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/run_defaults.yaml')
    parser.add_argument('--model-config', type=str, required=True)
    args = parser.parse_args()

    # Load configs
    run_config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # Setup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = f"{timestamp}_{model_config['model']['name']}"
    
    # Setup paths
    artifacts_dir = Path(run_config['paths']['artifacts_dir']) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_path=artifacts_dir / "logs.txt")
    set_seed(run_config['reproducibility']['seed'])
    
    logging.info(f"Starting run: {run_id}")
    
    # Data Pipeline
    data_mgr = DataManager(load_config('configs/data.yaml'))
    raw_data = data_mgr.load_raw_data()
    interim_df = data_mgr.make_interim(raw_data)
    
    # Feature Engineering
    fe = FeatureEngineer(load_config('configs/features.yaml'))
    processed_df = fe.build_features(interim_df)
    
    # Validation Split
    validator = Validator(run_config)
    target_col = load_config('configs/data.yaml')['columns']['target']
    splits = validator.get_splits(processed_df, target_col=target_col)
    
    # Training Loop
    oof_preds = np.zeros(len(processed_df))
    scores = []
    
    ModelClass = get_model_class(model_config['model']['name'])
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        logging.info(f"Training Fold {fold+1}/{len(splits)}")
        
        X_train = processed_df.iloc[train_idx].drop(columns=[target_col])
        y_train = processed_df.iloc[train_idx][target_col]
        X_val = processed_df.iloc[val_idx].drop(columns=[target_col])
        y_val = processed_df.iloc[val_idx][target_col]
        
        model = ModelClass(model_config)
        model.fit(X_train, y_train, X_val, y_val)
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        # Save model
        model.save(str(artifacts_dir / f"model_fold_{fold}.bin"))
        
    # Evaluation
    evaluator = Evaluator(run_config)
    metrics = evaluator.calculate_metrics(processed_df[target_col], oof_preds)
    logging.info(f"CV Metrics: {metrics}")
    
    # Save results
    pd.DataFrame({'actual': processed_df[target_col], 'pred': oof_preds}).to_csv(artifacts_dir / "oof_preds.csv", index=False)

if __name__ == "__main__":
    main()
