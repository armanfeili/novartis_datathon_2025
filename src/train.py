import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml

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

def run_experiment(model_name: str, model_config_path: str, run_name: str = None, config_path: str = 'configs/run_defaults.yaml'):
    """
    Run a full experiment: load data, train model, evaluate, and save artifacts.
    """
    # Load configs
    run_config = load_config(config_path)
    model_config = load_config(model_config_path)
    
    # Setup Run ID
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if run_name:
        run_id = f"{timestamp}_{run_name}"
    else:
        run_id = f"{timestamp}_{model_config['model']['name']}"
    
    # Setup paths
    artifacts_dir = Path(run_config['paths']['artifacts_dir']) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_path=artifacts_dir / "logs.txt")
    set_seed(run_config['reproducibility']['seed'])
    
    logging.info(f"Starting run: {run_id}")
    
    # Save config snapshot
    with open(artifacts_dir / "config_used.yaml", "w") as f:
        yaml.dump({"run_config": run_config, "model_config": model_config}, f)

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
    
    # Save metrics
    with open(artifacts_dir / "metrics.json", "w") as f:
        import json
        json.dump(metrics, f, indent=4)

    # Save results
    pd.DataFrame({'actual': processed_df[target_col], 'pred': oof_preds}).to_csv(artifacts_dir / "oof_preds.csv", index=False)
    
    return run_id, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model name (optional override)")
    parser.add_argument('--model-config', type=str, required=True, help="Path to model config yaml")
    parser.add_argument('--config', type=str, default='configs/run_defaults.yaml', help="Path to run defaults yaml")
    parser.add_argument('--run-name', type=str, help="Custom run name")
    
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        model_config_path=args.model_config,
        run_name=args.run_name,
        config_path=args.config
    )

if __name__ == "__main__":
    main()
