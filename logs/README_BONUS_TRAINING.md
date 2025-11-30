# Bonus Features Training Results

This directory contains logs and results from training CatBoost with all bonus features enabled.

## Configuration Used

**Config File:** `configs/run_bonus_all.yaml`

**Enabled Features:**
- ✅ B2: Bucket Specialization
- ✅ B3: Post-hoc Calibration  
- ✅ B4: Temporal Smoothing
- ✅ B6: Bias Correction
- ✅ B8: Multi-Seed Training (best seed selection)
- ✅ B10: Target Transform (log1p)
- ✅ G6: Data Augmentation

**Disabled Features:**
- ⚠️ B5: Residual Model (requires feature reconstruction)
- ⚠️ B7: Feature Pruning (evaluated separately)
- ⚠️ B9: Monotonicity Constraints (experimental)

## Training Process

1. **Scenario 1 Training**: Trains model for months 0-23 (no post-entry actuals)
2. **Scenario 2 Training**: Trains model for months 6-23 (first 6 months available)
3. **Submission Generation**: Combines predictions from both scenarios

## Log Files

- `training_s1_*.log`: Scenario 1 training logs
- `training_s2_*.log`: Scenario 2 training logs  
- `inference_*.log`: Submission generation logs
- `results_*.json`: Complete results summary with metrics

## Results Structure

Each `results_*.json` file contains:
- Run name and timestamp
- Enabled features list
- Scenario 1 metrics and model path
- Scenario 2 metrics and model path
- Submission file path
- Artifacts directories

## Artifacts

Models and artifacts are stored in:
- `artifacts/{run_name}_s1/`: Scenario 1 artifacts
- `artifacts/{run_name}_s2/`: Scenario 2 artifacts

Each artifacts directory contains:
- `model_*.bin`: Trained model(s)
- `calibration_params.json`: Calibration parameters (if enabled)
- `bias_corrections.json`: Bias corrections (if enabled)
- `target_transform_params_*.json`: Transform parameters (if enabled)
- `metrics.json`: Validation metrics
- `metadata.json`: Training metadata

## Submissions

Final submissions are saved to:
- `submissions/{run_name}_submission.csv`

## Usage

To run training with all bonus features:

```bash
python train_and_submit_complete.py
```

Or train scenarios individually:

```bash
# Scenario 1
python -m src.train --scenario 1 --model catboost --run-config configs/run_bonus_all.yaml --run-name "my_run_s1"

# Scenario 2  
python -m src.train --scenario 2 --model catboost --run-config configs/run_bonus_all.yaml --run-name "my_run_s2"

# Generate submission
python -m src.inference --model-s1 artifacts/my_run_s1/model.bin --model-s2 artifacts/my_run_s2/model.bin
```

