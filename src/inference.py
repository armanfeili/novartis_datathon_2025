import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from .utils import load_config, setup_logging
from .data import DataManager
from .features import FeatureEngineer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, required=True, help="Run ID to load model from")
    parser.add_argument('--input-file', type=str, default='test.csv')
    args = parser.parse_args()
    
    # Load run config
    # Assuming config is saved in artifacts
    artifacts_dir = Path(f"artifacts/runs/{args.run_id}")
    # In a real scenario, we'd load the config used for that run
    
    setup_logging()
    
    # Load Data
    # This part needs to be adapted to load test data specifically
    # For now, placeholder
    df_test = pd.read_csv(args.input_file)
    
    # Feature Engineering (must match training)
    # fe = FeatureEngineer(...)
    # df_test = fe.build_features(df_test)
    
    # Load Model(s) and Predict
    # If we have multiple folds, we average predictions
    models = list(artifacts_dir.glob("model_fold_*.bin"))
    preds = []
    
    for model_path in models:
        model = joblib.load(model_path)
        # Check if model object or raw model
        if hasattr(model, 'predict'):
            p = model.predict(df_test)
        else:
            # Handle raw booster if necessary
            p = model.predict(df_test)
        preds.append(p)
        
    avg_preds = np.mean(preds, axis=0)
    
    # Save submission
    sub = pd.DataFrame({'id': df_test.index, 'prediction': avg_preds})
    sub.to_csv(f"submissions/sub_{args.run_id}.csv", index=False)

if __name__ == "__main__":
    main()
