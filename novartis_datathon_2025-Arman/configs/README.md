# Configuration Reference

This directory contains YAML configuration files that control all aspects of the training pipeline.

---

## Overview

| File | Purpose |
|------|---------|
| `data.yaml` | Data paths, column definitions, schema |
| `features.yaml` | Feature engineering settings |
| `run_defaults.yaml` | Training defaults, seeds, validation, metrics |
| `model_cat.yaml` | CatBoost hyperparameters |
| `model_lgbm.yaml` | LightGBM hyperparameters |
| `model_xgb.yaml` | XGBoost hyperparameters |
| `model_linear.yaml` | Linear model settings |
| `model_nn.yaml` | Neural network configuration |

---

## data.yaml

Controls data loading, column names, and missing value handling.

### Paths

```yaml
paths:
  raw_dir: "data/raw"           # Raw data directory
  interim_dir: "data/interim"   # Intermediate files
  processed_dir: "data/processed"
  artifacts_dir: "artifacts"
```

### File Locations

```yaml
files:
  train:
    volume: "TRAIN/df_volume_train.csv"
    generics: "TRAIN/df_generics_train.csv"
    medicine_info: "TRAIN/df_medicine_info_train.csv"
  test:
    volume: "TEST/df_volume_test.csv"
    generics: "TEST/df_generics_test.csv"
    medicine_info: "TEST/df_medicine_info_test.csv"
```

### Column Definitions

```yaml
columns:
  id_keys:                      # Primary identifiers
    - "country"
    - "brand_name"
  time_key: "months_postgx"     # Time column
  raw_target: "volume"          # Raw target column
  model_target: "y_norm"        # Normalized target for training
  
  # Meta columns - NEVER used as features
  meta_cols:
    - "country"
    - "brand_name"
    - "months_postgx"
    - "bucket"
    - "avg_vol_12m"
    - "y_norm"
    - "volume"
    - "mean_erosion"
    - "month"
```

### Missing Value Handling

```yaml
missing_values:
  n_gxs:
    strategy: "ffill_then_zero"     # Forward-fill per series, then 0
  hospital_rate:
    strategy: "median_by_group"
    group_by: "ther_area"
    add_missing_flag: true
  ther_area: "Unknown"
  main_package: "Unknown"
  biological:
    default: false
    add_missing_flag: true
```

---

## features.yaml

Controls feature engineering with strict leakage prevention.

### Pre-Entry Statistics

```yaml
pre_entry:
  windows: [3, 6, 12]           # Rolling windows for avg_vol
  compute_trend: true           # Linear trend from pre-entry
  compute_volatility: true      # Volatility measure
  compute_seasonal: true        # Seasonal pattern detection
  log_transform: true           # log1p(avg_vol_12m)
```

### Time Features

```yaml
time:
  include_months_postgx: true   # Direct time index
  include_squared: true         # months_postgx²
  include_time_bucket: true     # Categorical: pre, 0-5, 6-11, 12-23
  include_month_of_year: true   # Seasonality
  include_decay: true           # Exponential decay
  decay_alpha: 0.1              # Decay rate
```

### Generics Features

```yaml
generics:
  include_n_gxs: true           # Current generic count
  include_has_generic: true     # Binary: any generics?
  include_cummax: true          # Maximum generics so far
  include_first_month: true     # When did generics enter?
  include_future_n_gxs: true    # Exogenous future info (allowed)
```

### Scenario 2 Features

Only computed when `scenario=2`:

```yaml
scenario2_early:
  compute_avg_0_5: true         # Mean volume months 0-5
  compute_erosion_0_5: true     # avg_0_5 / avg_vol_12m
  compute_trend_0_5: true       # Early trend
  compute_drop_month_0: true    # Initial drop magnitude
```

### Leakage Prevention

```yaml
leakage_prevention:
  forbidden_features:           # Never included in features
    - "bucket"
    - "y_norm"
    - "volume"
    - "mean_erosion"
    - "country"
    - "brand_name"
  
  scenario1:
    feature_cutoff: 0           # Only months < 0 allowed
    target_range: [0, 23]
  
  scenario2:
    feature_cutoff: 6           # Only months < 6 allowed
    target_range: [6, 23]
```

---

## run_defaults.yaml

Training defaults, reproducibility, validation, and metrics.

### Reproducibility

```yaml
reproducibility:
  seed: 42                      # Global random seed
  deterministic: true           # Enable deterministic mode
```

### Validation Settings

```yaml
validation:
  val_fraction: 0.2             # 20% for validation
  stratify_by: "bucket"         # Stratify by bucket
  split_level: "series"         # CRITICAL: series-level split
```

### Scenario Definitions

```yaml
scenarios:
  scenario1:
    name: "Scenario 1 - No Post-Entry Actuals"
    forecast_start: 0
    forecast_end: 23
    feature_cutoff: 0
    
  scenario2:
    name: "Scenario 2 - First 6 Months Available"
    forecast_start: 6
    forecast_end: 23
    feature_cutoff: 6
```

### Official Metric Configuration

```yaml
official_metric:
  bucket_threshold: 0.25        # Bucket 1 if mean_erosion ≤ 0.25
  
  bucket_weights:
    bucket1: 2.0                # High erosion - 2× weight
    bucket2: 1.0                # Low erosion - 1× weight
  
  metric1:                      # Scenario 1
    monthly_weight: 0.2
    accumulated_0_5_weight: 0.5
    accumulated_6_11_weight: 0.2
    accumulated_12_23_weight: 0.1
  
  metric2:                      # Scenario 2
    monthly_weight: 0.2
    accumulated_6_11_weight: 0.5
    accumulated_12_23_weight: 0.3
```

### Sample Weights

```yaml
sample_weights:
  scenario1:
    months_0_5: 3.0             # Highest priority
    months_6_11: 1.5
    months_12_23: 1.0
    
  scenario2:
    months_6_11: 2.5            # Highest priority
    months_12_23: 1.0
    
  bucket_weights:
    bucket1: 2.0                # Match metric weighting
    bucket2: 1.0
```

### Experiment Tracking

```yaml
experiment_tracking:
  enabled: false                # Set to true to enable
  backend: "mlflow"             # mlflow or wandb
  experiment_name: "novartis-datathon-2025"
```

---

## model_cat.yaml (CatBoost)

```yaml
model:
  name: "catboost"
  task: "regression"

params:
  loss_function: "RMSE"
  eval_metric: "RMSE"
  depth: 6                      # Tree depth
  min_data_in_leaf: 20
  learning_rate: 0.03
  iterations: 2000
  l2_leaf_reg: 3.0
  early_stopping_rounds: 100
  random_seed: 42
  verbose: 100

# Native categorical handling
categorical_features:
  - "ther_area"
  - "main_package"
  - "time_bucket"
  - "hospital_rate_bin"

# Optuna search space
tuning:
  enabled: false
  n_trials: 100
  search_space:
    depth: [4, 8]
    learning_rate: [0.01, 0.1]
    l2_leaf_reg: [1.0, 10.0]
```

---

## model_lgbm.yaml (LightGBM)

```yaml
model:
  name: "lightgbm"
  task: "regression"

params:
  boosting_type: "gbdt"
  objective: "regression"
  metric: "rmse"
  num_leaves: 31
  max_depth: -1                 # No limit
  min_data_in_leaf: 20
  learning_rate: 0.05
  n_estimators: 1000
  feature_fraction: 0.8
  bagging_fraction: 0.8
  early_stopping_rounds: 50
  verbose: -1
  seed: 42

tuning:
  search_space:
    num_leaves: [15, 255]
    learning_rate: [0.01, 0.3]
    feature_fraction: [0.5, 1.0]
```

---

## model_xgb.yaml (XGBoost)

```yaml
model:
  name: "xgboost"
  task: "regression"

params:
  booster: "gbtree"
  objective: "reg:squarederror"
  eval_metric: "rmse"
  max_depth: 6
  min_child_weight: 1
  learning_rate: 0.05
  n_estimators: 1000
  subsample: 0.8
  colsample_bytree: 0.8
  early_stopping_rounds: 50
  verbosity: 0
  seed: 42
  # GPU: tree_method: "gpu_hist"

tuning:
  search_space:
    max_depth: [3, 12]
    learning_rate: [0.01, 0.3]
    subsample: [0.5, 1.0]
```

---

## model_linear.yaml

```yaml
model:
  name: "linear"
  type: "ridge"                 # ridge, lasso, elasticnet, huber

ridge:
  alpha: 1.0
  fit_intercept: true

lasso:
  alpha: 1.0
  max_iter: 1000

elasticnet:
  alpha: 1.0
  l1_ratio: 0.5

preprocessing:
  scale_features: true
  scaler: "standard"            # standard, minmax, robust

tuning:
  search_space:
    ridge:
      alpha: [0.001, 1000.0]
    elasticnet:
      alpha: [0.001, 100.0]
      l1_ratio: [0.0, 1.0]
```

---

## model_nn.yaml (Neural Network)

```yaml
model:
  name: "neural_network"
  framework: "pytorch"

architecture:
  type: "mlp"
  mlp:
    hidden_layers: [256, 128, 64]
    activation: "relu"
    dropout: 0.2
    batch_norm: true

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 1e-5
  optimizer: "adam"
  
  scheduler:
    type: "reduce_on_plateau"
    patience: 10
    factor: 0.5
  
  early_stopping:
    enabled: true
    patience: 20
    restore_best_weights: true

amp:
  enabled: true                 # Mixed precision training

seed: 42
```

---

## Usage Examples

### Loading Configs

```python
from src.utils import load_config

# Load individual config
data_config = load_config('configs/data.yaml')
features_config = load_config('configs/features.yaml')
run_config = load_config('configs/run_defaults.yaml')
model_config = load_config('configs/model_cat.yaml')

# Access values
seed = run_config['reproducibility']['seed']
val_fraction = run_config['validation']['val_fraction']
```

### Overriding via CLI

```bash
# Configs are merged: CLI overrides file settings
python -m src.train \
    --scenario 1 \
    --model catboost \
    --seed 123 \                    # Overrides reproducibility.seed
    --val-fraction 0.15             # Overrides validation.val_fraction
```

### Creating Custom Config

```yaml
# configs/model_custom.yaml
model:
  name: "catboost"
  task: "regression"

params:
  depth: 8                          # Deeper trees
  learning_rate: 0.01               # Slower learning
  iterations: 5000
  # ... other params
```

---

## Validation

Use the test suite to validate configurations:

```bash
pytest tests/test_smoke.py -k "config" -v
```
