# 🤖 Models Documentation - Novartis Datathon 2025

This document provides comprehensive documentation for all machine learning models and their configuration files used in the Novartis Datathon 2025 pharmaceutical erosion forecasting competition.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Available Models](#available-models)
   - [XGBoost (Primary)](#1-xgboost-model-primary)
   - [LightGBM (Secondary)](#2-lightgbm-model-secondary)
   - [CatBoost (Tertiary)](#3-catboost-model-tertiary)
   - [Linear Models](#4-linear-models)
   - [Neural Network](#5-neural-network-model)
   - [Hybrid Physics+ML](#6-hybrid-physics--ml-model)
   - [ARIHOW (Time Series)](#7-arihow-model-arima--holt-winters)
4. [Baseline Models](#baseline-models)
5. [Ensemble Models](#ensemble-models)
6. [Configuration Files](#configuration-files)
7. [Model Interface (BaseModel)](#model-interface-basemodel)
8. [GPU Support](#gpu-support)
9. [Hyperparameter Sweeps](#hyperparameter-sweeps)
10. [Usage Examples](#usage-examples)
11. [Model Factory Function](#model-factory-function)

---

## Overview

The Novartis Datathon 2025 models are designed to forecast pharmaceutical volume erosion following generic entry (Loss of Exclusivity - LOE). The models predict normalized volume (`y_norm`) for 24 months post-generic entry.

### Competition Scenarios

| Scenario | Forecast Period | Available Data | Description |
|----------|-----------------|----------------|-------------|
| **Scenario 1** | Months 0-23 | Pre-entry only | Predict entire erosion curve before generic entry |
| **Scenario 2** | Months 6-23 | Pre-entry + Months 0-5 | Predict remaining curve with early erosion observed |

### Key Metrics

- **Official Metric**: Prediction Error (PE) - Used for model selection and leaderboard ranking
- **RMSE**: Used for early stopping during training (faster, more stable)
- **MAE**: Alternative metric for interpretability

---

## Model Architecture

All models implement the `BaseModel` abstract interface, ensuring consistent usage across the pipeline:

```
BaseModel (Abstract)
├── fit(X_train, y_train, X_val, y_val, sample_weight)
├── predict(X)
├── save(path)
├── load(path)  [classmethod]
└── get_feature_importance()
```

### Model Hierarchy

```
BaseModel (base.py)
├── Tree-Based Models
│   ├── XGBModel (xgb_model.py)
│   ├── LGBMModel (lgbm_model.py)
│   └── CatBoostModel (cat_model.py)
├── Linear Models (linear.py)
│   ├── LinearModel (Ridge/Lasso/ElasticNet/Huber)
│   ├── GlobalMeanBaseline
│   ├── FlatBaseline
│   ├── TrendBaseline
│   └── HistoricalCurveBaseline
├── Neural Network (nn.py)
│   └── NNModel (PyTorch MLP)
├── Specialized Models
│   ├── HybridPhysicsMLModel (hybrid_physics_ml.py)
│   ├── ARIHOWModel (arihow.py)
│   └── BaselineModels (baselines.py)
└── Ensemble Models (ensemble.py)
    ├── AveragingEnsemble
    ├── WeightedAveragingEnsemble
    ├── StackingEnsemble
    ├── BlendingEnsemble
    └── EnsembleBlender
```

---

## Available Models

### 1. XGBoost Model (Primary)

**File**: `xgb_model.py`  
**Config**: `configs/model_xgb.yaml`  
**Priority**: 1 (Highest)  
**GPU Support**: ✅ Yes

XGBoost is the primary model, delivering the best performance on the official metric.

#### Features

- **Native DMatrix Support**: Efficient handling of sample weights
- **GPU Acceleration**: `tree_method='gpu_hist'` for fast training on Colab
- **Early Stopping**: Automatic stopping based on RMSE to prevent overfitting
- **Feature Importance**: Gain-based importance scores
- **Categorical Handling**: Automatic encoding of categorical columns to numeric codes

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` (eta) | 0.03 | Learning rate |
| `n_estimators` | 3000 | Maximum number of trees |
| `subsample` | 0.8 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Column subsampling ratio |
| `reg_lambda` | 1.0 | L2 regularization |
| `reg_alpha` | 0.0 | L1 regularization |
| `min_child_weight` | 1 | Minimum child weight |
| `early_stopping_rounds` | 50 | Early stopping patience |

#### Best Configurations (from sweeps)

| Scenario | max_depth | learning_rate | reg_lambda | Official Metric |
|----------|-----------|---------------|------------|-----------------|
| Scenario 1 | 6 | 0.03 | 1 | 0.7499 |
| Scenario 2 | 4 | 0.05 | 1 | 0.2659 |

#### Named Configurations

| ID | Description | Key Settings |
|----|-------------|--------------|
| `default` | Balanced configuration | depth=6, lr=0.03 |
| `low_lr` | More trees, better generalization | depth=6, lr=0.02, n_est=5000 |
| `shallow` | Reduce overfitting | depth=4, lr=0.05, lambda=3 |
| `deep` | Complex patterns | depth=8, lr=0.02, lambda=5 |
| `regularized` | Strong regularization | depth=5, subsample=0.7 |
| `s1_best` | Best for Scenario 1 | depth=6, lr=0.03 |
| `s2_best` | Best for Scenario 2 | depth=4, lr=0.05 |

#### GPU Configuration

```yaml
gpu:
  enabled: true
  device_id: 0
# Applied params when GPU enabled:
#   tree_method: 'gpu_hist'
#   predictor: 'gpu_predictor'
```

---

### 2. LightGBM Model (Secondary)

**File**: `lgbm_model.py`  
**Config**: `configs/model_lgbm.yaml`  
**Priority**: 2  
**GPU Support**: ✅ Yes

LightGBM is the secondary model, offering fast training and excellent ensemble partner for XGBoost.

#### Features

- **Histogram-Based Learning**: Extremely fast training
- **Leaf-Wise Growth**: Different tree structure than XGBoost
- **Native Categorical Support**: No manual encoding needed
- **GPU Acceleration**: OpenCL-based GPU support
- **Early Stopping**: Via callbacks API

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_leaves` | 63 | Maximum number of leaves |
| `max_depth` | -1 | No depth limit (-1) |
| `learning_rate` | 0.05 | Learning rate |
| `n_estimators` | 3000 | Maximum boosting iterations |
| `min_data_in_leaf` | 20 | Minimum samples per leaf |
| `feature_fraction` | 0.8 | Feature subsampling ratio |
| `bagging_fraction` | 0.8 | Row subsampling ratio |
| `lambda_l1` | 0.0 | L1 regularization |
| `lambda_l2` | 0.0 | L2 regularization |
| `early_stopping_rounds` | 50 | Early stopping patience |

#### Best Configurations

| Scenario | num_leaves | learning_rate | min_data_in_leaf |
|----------|------------|---------------|------------------|
| Scenario 1 | 63 | 0.05 | 20 |
| Scenario 2 | 31 | 0.05 | 20 |

#### Named Configurations

| ID | Description | Key Settings |
|----|-------------|--------------|
| `default` | Balanced | leaves=63, lr=0.05 |
| `low_lr` | More iterations | leaves=63, lr=0.02, n_est=5000 |
| `high_leaves` | Complex patterns | leaves=127, lr=0.03 |
| `conservative` | Prevent overfitting | leaves=31, min_data=40 |
| `regularized` | L1/L2 regularization | lambda_l1=0.1, lambda_l2=0.1 |

#### GPU Configuration

```yaml
gpu:
  enabled: true
  platform_id: 0
  device_id: 0
# Applied params when GPU enabled:
#   device: 'gpu'
```

---

### 3. CatBoost Model (Tertiary)

**File**: `cat_model.py`  
**Config**: `configs/model_cat.yaml`  
**Priority**: 3  
**GPU Support**: ✅ Yes

CatBoost provides native categorical handling and ensemble diversity.

#### Features

- **Native Categorical Support**: No need for encoding
- **Ordered Boosting**: Reduces overfitting
- **GPU Acceleration**: Efficient CUDA implementation
- **Robust to Overfitting**: Built-in regularization
- **Pool API**: Efficient data handling

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | 6 | Tree depth |
| `learning_rate` | 0.03 | Learning rate |
| `iterations` | 3000 | Maximum iterations |
| `l2_leaf_reg` | 3.0 | L2 regularization |
| `random_strength` | 1.0 | Randomization strength |
| `bagging_temperature` | 1.0 | Bayesian bootstrap temperature |
| `min_data_in_leaf` | 20 | Minimum samples per leaf |
| `early_stopping_rounds` | 100 | Early stopping patience |

#### Categorical Features (Native Support)

```yaml
categorical_features:
  - "ther_area"
  - "main_package"
  - "time_bucket"
  - "hospital_rate_bin"
  - "n_gxs_bin"
```

#### Named Configurations

| ID | Description | Key Settings |
|----|-------------|--------------|
| `default` | Balanced | depth=6, lr=0.03, l2=3.0 |
| `shallow` | Simpler patterns | depth=4, lr=0.05, l2=1.0 |
| `conservative` | Strong regularization | depth=5, lr=0.02, l2=10.0 |

#### GPU Configuration

```yaml
gpu:
  enabled: true
  device_id: 0
# Applied params when GPU enabled:
#   task_type: 'GPU'
#   devices: '0'
```

> **Note**: CatBoost consistently underperforms XGBoost on the official metric. Use primarily for ensemble diversity.

---

### 4. Linear Models

**File**: `linear.py`  
**Config**: `configs/model_linear.yaml`  
**Priority**: 4  
**GPU Support**: ❌ No (CPU only)

Linear models serve as interpretable baselines with built-in regularization.

#### Available Types

| Model | Description | Key Parameter |
|-------|-------------|---------------|
| **Ridge** | L2 regularization | `alpha` |
| **Lasso** | L1 regularization (sparse) | `alpha` |
| **ElasticNet** | L1 + L2 combined | `alpha`, `l1_ratio` |
| **Huber** | Robust to outliers | `epsilon`, `alpha` |

#### Features

- **Pipeline Integration**: Automatic scaling and preprocessing
- **Sample Weight Support**: Via sklearn's `sample_weight` parameter
- **Polynomial Features**: Optional quadratic feature expansion
- **Feature Importance**: Coefficient magnitudes

#### Key Parameters

```yaml
# Ridge
ridge:
  alpha: 1.0
  fit_intercept: true
  solver: "auto"

# Lasso
lasso:
  alpha: 1.0
  max_iter: 1000
  tol: 1e-4

# ElasticNet
elasticnet:
  alpha: 1.0
  l1_ratio: 0.5  # 0=L2, 1=L1

# Huber
huber:
  epsilon: 1.35
  alpha: 0.0001
```

#### Preprocessing Options

```yaml
preprocessing:
  scale_features: true
  scaler: "standard"        # standard, minmax, robust
  handle_missing: "mean"    # mean, median, zero
  polynomial_degree: null   # Set to 2 for quadratic features
```

#### Named Configurations

| ID | Description | Settings |
|----|-------------|----------|
| `default` | Standard Ridge | type=ridge, alpha=1.0 |
| `lasso` | Feature selection | type=lasso, alpha=1.0 |
| `lasso_strong` | Sparse solution | type=lasso, alpha=10.0 |
| `elasticnet` | Balanced L1/L2 | type=elasticnet, l1_ratio=0.5 |
| `huber` | Robust regression | type=huber, epsilon=1.35 |

---

### 5. Neural Network Model

**File**: `nn.py`  
**Config**: `configs/model_nn.yaml`  
**Priority**: 5 (Experimental)  
**GPU Support**: ✅ Yes (PyTorch)

Simple MLP neural network for tabular regression.

#### Architecture

```
Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear → Output
```

#### Features

- **PyTorch Backend**: Automatic GPU detection (CUDA/MPS)
- **Batch Normalization**: Stabilizes training
- **Dropout**: Regularization
- **Sample Weights**: Via WeightedRandomSampler
- **Early Stopping**: Patience-based with best model restoration
- **Learning Rate Scheduling**: ReduceLROnPlateau

#### Key Parameters

```yaml
architecture:
  mlp:
    hidden_layers: [256, 128, 64]
    activation: "relu"      # relu, leaky_relu, gelu, silu
    dropout: 0.2
    batch_norm: true

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.00001
  optimizer: "adam"
  
  scheduler:
    type: "reduce_on_plateau"
    patience: 10
    factor: 0.5
    
  early_stopping:
    enabled: true
    patience: 20
```

#### Named Configurations

| ID | Description | Architecture |
|----|-------------|--------------|
| `default` | Standard MLP | [256, 128, 64], dropout=0.2 |
| `small` | Faster training | [128, 64], dropout=0.1 |
| `large` | Complex patterns | [512, 256, 128, 64], dropout=0.3 |
| `deep` | More layers | [256, 256, 128, 128, 64], dropout=0.2 |

#### Device Selection

```yaml
device: "auto"  # auto, cuda, mps, cpu

# Automatic detection order:
# 1. CUDA (NVIDIA GPU)
# 2. MPS (Apple Silicon)
# 3. CPU (fallback)
```

#### Mixed Precision Training

```yaml
amp:
  enabled: true  # Use automatic mixed precision (faster)
```

---

### 6. Hybrid Physics + ML Model

**File**: `hybrid_physics_ml.py`  
**Config**: `configs/model_hybrid.yaml`  
**Priority**: 2  
**GPU Support**: ✅ Yes (via underlying ML model)

Combines a physics-based exponential decay baseline with ML residual learning.

#### Architecture

```
final_prediction = physics_baseline + ML_residual

Where:
  physics_baseline = exp(-decay_rate × months_postgx)
  ML_residual = LightGBM/XGBoost(features)
```

#### Benefits

- **Domain Knowledge Integration**: Leverages expected erosion patterns
- **Better Generalization**: Physics provides reasonable baseline even with limited data
- **Interpretability**: Can decompose predictions into components
- **Extrapolation Safety**: Physics baseline prevents unreasonable predictions

#### Key Parameters

```yaml
physics:
  decay_rate: 0.05  # Monthly decay rate (0.03-0.10 typical)
  scenario_decay_rates:
    scenario1: 0.05
    scenario2: 0.05

ml_model:
  type: "lightgbm"  # or "xgboost"
  
  lightgbm:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 500
    
prediction:
  clip_predictions: true
  clip_min: 0.0
  clip_max: 2.0
```

#### Prediction Components

```python
# Get breakdown of predictions
components = model.predict_components(X_test, avg_vol_test, months_test)
# Returns:
#   'physics': Physics baseline predictions
#   'residual': ML residual predictions
#   'final': Combined final predictions
```

#### Named Configurations

| ID | Description | Settings |
|----|-------------|----------|
| `default` | Balanced | decay=0.05, lr=0.05, leaves=31 |
| `slow_decay` | More ML learning | decay=0.03, leaves=63 |
| `fast_decay` | Less ML correction | decay=0.10, lr=0.03 |
| `ml_heavy` | More ML capacity | decay=0.05, leaves=63, n_est=1000 |

---

### 7. ARIHOW Model (ARIMA + Holt-Winters)

**File**: `arihow.py`  
**Config**: `configs/model_arihow.yaml`  
**Priority**: 4  
**GPU Support**: ❌ No (statsmodels-based)  
**Dependencies**: statsmodels, scipy

ARIMA + Holt-Winters hybrid with learned combination weights.

#### Architecture

```
prediction = β × ARIMA_forecast + (1-β) × HoltWinters_forecast

Where β is learned via ridge regression on recent observations
```

#### Features

- **Per-Brand Fitting**: Separate ARIMA/HW models for each (country, brand_name)
- **Learned Weights**: Optimal blend via constrained optimization
- **Fallback Strategy**: Exponential decay for series with insufficient history
- **Parallel Fitting**: Optional multiprocessing for faster training

#### Key Parameters

```yaml
arima:
  order: [1, 1, 1]              # (p, d, q)
  seasonal_order: [1, 0, 1, 12] # (P, D, Q, m)
  trend: 'c'                     # constant
  enforce_stationarity: false
  enforce_invertibility: false

holt_winters:
  trend: 'add'                  # 'add', 'mul', or null
  seasonal: null                # 'add', 'mul', or null
  seasonal_periods: 12
  damped_trend: true

weights:
  method: 'grid'                # grid, brent, fixed
  grid_values: [0.0, 0.1, ..., 1.0]
  
fallback:
  min_history_months: 12        # Use fallback if less history
  decay_rate: 0.02              # Exponential decay rate
```

#### Named Configurations

| ID | Description | Settings |
|----|-------------|----------|
| `default` | Balanced ARIMA+HW | order=[1,1,1], hw_trend='add' |
| `arima_heavy` | Favor ARIMA | order=[2,1,1], beta=0.7 |
| `hw_heavy` | Favor Holt-Winters | order=[1,0,0], beta=0.3 |
| `simple` | Stable series | order=[1,1,0], no seasonal |

#### Model Statistics

```python
# Get learned weights for all brands
weights_df = model.get_brand_weights()
# Returns: country, brand_name, beta_arima, beta_hw, success, series_length

# Get summary statistics
stats = model.get_model_stats()
# Returns: n_brands, n_success, success_rate, mean_beta_arima, mean_beta_hw
```

---

## Baseline Models

**File**: `baselines.py`

Simple deterministic forecasting baselines for sanity checks.

### Available Baselines

| Baseline | Formula | Description |
|----------|---------|-------------|
| **Naive Persistence** | `volume = avg_vol` | Constant (no erosion) |
| **Linear Decay** | `volume = avg_vol × (1 - rate × t)` | Linear decrease |
| **Exponential Decay** | `volume = avg_vol × exp(-rate × t)` | Exponential decrease |
| **Bucket-Specific** | Different rates per bucket | Tailored decay |

### Usage

```python
from src.models import BaselineModels

# Simple predictions
preds = BaselineModels.naive_persistence(avg_j_df, months=[0,1,...,23])
preds = BaselineModels.exponential_decay(avg_j_df, months, decay_rate=0.05)

# Tune decay rate on training data
best_rate, results = BaselineModels.tune_decay_rate(
    actual_df, avg_j_df, 
    decay_type='exponential',
    decay_rates=[0.01, 0.02, ..., 0.15]
)
```

---

## Ensemble Models

**File**: `ensemble.py`

Multiple ensemble strategies for combining model predictions.

### 1. AveragingEnsemble

Simple arithmetic mean of predictions.

```python
ensemble = AveragingEnsemble({'models': [model1, model2, model3]})
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

### 2. WeightedAveragingEnsemble

Weighted combination with optimizable weights.

```python
ensemble = WeightedAveragingEnsemble({
    'models': [model1, model2, model3],
    'optimize_weights': True,      # Learn weights on validation
    'optimization_metric': 'mse'   # or 'mae'
})
ensemble.fit(X_train, y_train, X_val, y_val)
weights = ensemble.get_weights()  # {0: 0.4, 1: 0.35, 2: 0.25}
```

### 3. StackingEnsemble

Two-level stacking with meta-learner.

```python
ensemble = StackingEnsemble({
    'base_models': [
        ('xgb', XGBModel(config)),
        ('lgbm', LGBMModel(config)),
    ],
    'meta_learner': Ridge(alpha=1.0),
    'n_folds': 5,                      # OOF CV folds
    'use_original_features': False     # Only use base predictions
})
```

### 4. BlendingEnsemble

Uses holdout data for meta-learner training.

```python
ensemble = BlendingEnsemble({
    'models': [model1, model2],
    'meta_learner': Ridge(alpha=1.0),
    'holdout_fraction': 0.2
})
```

### 5. EnsembleBlender

Lightweight prediction combiner for numpy arrays.

```python
blender = EnsembleBlender(constrain_weights=True)
blender.fit(
    predictions={'xgb': preds_xgb, 'lgbm': preds_lgbm},
    y_true=y_val
)
blended = blender.predict({'xgb': preds_xgb_test, 'lgbm': preds_lgbm_test})
print(blender.get_weights())  # {'xgb': 0.55, 'lgbm': 0.45}
```

---

## Configuration Files

All model configurations are stored in `configs/` with a consistent structure:

```
configs/
├── model_xgb.yaml      # XGBoost configuration
├── model_lgbm.yaml     # LightGBM configuration
├── model_cat.yaml      # CatBoost configuration
├── model_linear.yaml   # Linear models configuration
├── model_nn.yaml       # Neural network configuration
├── model_hybrid.yaml   # Hybrid Physics+ML configuration
└── model_arihow.yaml   # ARIHOW configuration
```

### Common Configuration Structure

```yaml
# =============================================================================
# Model Identification
# =============================================================================
model:
  name: "model_name"
  task: "regression"
  priority: 1  # Lower = higher priority

# =============================================================================
# Active Configuration
# =============================================================================
active_config_id: null  # or "config_id" from sweep_configs

# =============================================================================
# GPU Configuration
# =============================================================================
gpu:
  enabled: false
  device_id: 0

# =============================================================================
# Sweep Configuration
# =============================================================================
sweep:
  enabled: false
  mode: "configs"  # "configs" or "grid"
  selection_metric: "official_metric"
  n_folds: 3
  
  grid:
    param1: [val1, val2, val3]
    param2: [val1, val2]

# =============================================================================
# Named Configurations
# =============================================================================
sweep_configs:
  - id: "default"
    description: "Default configuration"
    params:
      param1: value1

  - id: "custom"
    description: "Custom configuration"
    params:
      param1: value2

# =============================================================================
# Base Parameters
# =============================================================================
params:
  param1: default_value
  param2: default_value

# =============================================================================
# Training Settings
# =============================================================================
training:
  use_early_stopping: true
  eval_metric: "rmse"
  use_sample_weights: true

# =============================================================================
# Best Parameters (from previous experiments)
# =============================================================================
best_params:
  scenario1:
    param1: best_value
  scenario2:
    param1: best_value

# =============================================================================
# Ensemble Settings
# =============================================================================
ensemble:
  include: true
  weight_optimization: true
  default_weight: 0.5
```

---

## Model Interface (BaseModel)

All models implement the `BaseModel` abstract base class:

```python
from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Abstract base for all models ensuring consistent interface."""
    
    def __init__(self, config: dict):
        """Initialize model with configuration."""
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
    
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'BaseModel':
        """Train model with optional validation and sample weights."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input features."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        pass
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance if available."""
        return pd.DataFrame(columns=['feature', 'importance'])
```

### Serialization

| Model | Save Format | Load Method |
|-------|-------------|-------------|
| XGBoost | joblib (model + metadata) | `XGBModel.load(path)` |
| LightGBM | joblib (model + metadata) | `LGBMModel.load(path)` |
| CatBoost | Native CatBoost format | `CatBoostModel.load(path)` |
| Linear | joblib (pipeline) | `LinearModel.load(path)` |
| Neural Net | PyTorch checkpoint | `NNModel.load(path)` |

---

## GPU Support

### Enabling GPU

```python
# Method 1: Via config file
gpu:
  enabled: true
  device_id: 0

# Method 2: Programmatically
from src.utils import get_gpu_info

gpu_info = get_gpu_info()
if gpu_info['gpu_available']:
    config['params']['tree_method'] = 'gpu_hist'  # XGBoost
    config['params']['device'] = 'gpu'            # LightGBM
    config['params']['task_type'] = 'GPU'         # CatBoost
```

### Model-Specific GPU Parameters

| Model | GPU Parameter | Value |
|-------|---------------|-------|
| XGBoost | `tree_method` | `'gpu_hist'` |
| XGBoost | `predictor` | `'gpu_predictor'` |
| LightGBM | `device` | `'gpu'` |
| CatBoost | `task_type` | `'GPU'` |
| Neural Net | (automatic) | Via PyTorch |

### GPU Detection

```python
import torch

# Check CUDA
if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Check MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("MPS available")
```

---

## Hyperparameter Sweeps

### Sweep Modes

1. **configs**: Iterate over named configurations from `sweep_configs`
2. **grid**: Cartesian product of all parameter values in `sweep.grid`

### Configuration

```yaml
sweep:
  enabled: true
  mode: "configs"  # or "grid"
  selection_metric: "official_metric"  # PRIMARY criterion
  n_folds: 3  # K-fold CV for robustness
```

### Running Sweeps

```python
from src.config_sweep import generate_sweep_runs, get_config_by_id

# Get all configurations to sweep
sweep_runs = generate_sweep_runs(model_config, mode='configs')

# Or get a specific configuration
specific_config = get_config_by_id(model_config, 'low_lr')
```

### Sweep Presets

```yaml
sweep_presets:
  fast:      # Quick exploration
    param1: [val1, val2]
  full:      # Comprehensive search
    param1: [val1, val2, val3, val4]
  focused:   # Around known good values
    param1: [best-1, best, best+1]
```

---

## Usage Examples

### Basic Model Training

```python
from src.models import get_model_class
from src.utils import load_config

# Load configuration
config = load_config('configs/model_xgb.yaml')

# Get model class and instantiate
ModelClass = get_model_class('xgboost')
model = ModelClass(config)

# Train
model.fit(X_train, y_train, X_val, y_val, sample_weight=weights)

# Predict
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance()

# Save/Load
model.save('model.bin')
loaded_model = ModelClass.load('model.bin')
```

### Training with Cross-Validation

```python
from src.train import run_cross_validation

models, cv_results, oof_predictions = run_cross_validation(
    panel_features=panel_features,
    scenario=1,
    model_type='xgboost',
    model_config=config,
    run_config=run_config,
    n_folds=5,
    save_oof=True,
    artifacts_dir=Path('artifacts/run_001'),
)

print(f"CV Official Metric: {cv_results['cv_official_mean']:.4f}")
```

### Creating Ensembles

```python
from src.models import (
    get_model_class, 
    WeightedAveragingEnsemble,
    EnsembleBlender
)

# Train individual models
xgb = get_model_class('xgboost')(xgb_config)
lgbm = get_model_class('lightgbm')(lgbm_config)

xgb.fit(X_train, y_train, X_val, y_val)
lgbm.fit(X_train, y_train, X_val, y_val)

# Method 1: WeightedAveragingEnsemble
ensemble = WeightedAveragingEnsemble({
    'models': [xgb, lgbm],
    'optimize_weights': True
})
ensemble.fit(X_train, y_train, X_val, y_val)
preds = ensemble.predict(X_test)

# Method 2: EnsembleBlender (post-hoc)
preds_xgb = xgb.predict(X_val)
preds_lgbm = lgbm.predict(X_val)

blender = EnsembleBlender()
blender.fit(
    predictions={'xgboost': preds_xgb, 'lightgbm': preds_lgbm},
    y_true=y_val
)
final_preds = blender.predict({
    'xgboost': xgb.predict(X_test),
    'lightgbm': lgbm.predict(X_test)
})
```

### Using Hybrid Model

```python
from src.models import HybridPhysicsMLModel

model = HybridPhysicsMLModel(
    ml_model_type='lightgbm',
    decay_rate=0.05,
    clip_predictions=True
)

# Fit requires additional inputs
model.fit(
    X_train, y_train,
    avg_vol_train=avg_vol_train,
    months_train=months_train,
    X_val=X_val, y_val=y_val,
    avg_vol_val=avg_vol_val,
    months_val=months_val
)

# Predict also requires additional inputs
predictions = model.predict(X_test, avg_vol_test, months_test)

# Get prediction breakdown
components = model.predict_components(X_test, avg_vol_test, months_test)
print(f"Physics contribution: {components['physics'].mean():.3f}")
print(f"ML residual: {components['residual'].mean():.3f}")
```

---

## Model Factory Function

Use `get_model_class()` to retrieve model classes by name:

```python
from src.models import get_model_class

# Tree boosters
XGBModel = get_model_class('xgboost')     # or 'xgb'
LGBMModel = get_model_class('lightgbm')   # or 'lgbm'
CatBoostModel = get_model_class('catboost') # or 'cat'

# Linear models
LinearModel = get_model_class('linear')   # or 'ridge', 'lasso', 'elasticnet'

# Neural network
NNModel = get_model_class('nn')           # or 'neural', 'mlp'

# Specialized
HybridModel = get_model_class('hybrid')   # or 'hybrid_lgbm', 'hybrid_xgb'
ARIHOWModel = get_model_class('arihow')   # or 'ts_hybrid'

# Baselines
GlobalMean = get_model_class('global_mean')
FlatBaseline = get_model_class('flat')
TrendBaseline = get_model_class('trend')
KNNBaseline = get_model_class('historical_curve')

# Ensembles
AvgEnsemble = get_model_class('averaging')
WeightedEnsemble = get_model_class('weighted')
StackingEnsemble = get_model_class('stacking')
BlendingEnsemble = get_model_class('blending')
```

### Available Model Names

```python
# Full list of recognized model names
available_models = [
    # Tree boosters
    'catboost', 'cat',
    'lightgbm', 'lgbm',
    'xgboost', 'xgb',
    
    # Neural network
    'nn', 'neural', 'mlp',
    
    # Linear
    'linear', 'ridge', 'lasso', 'elasticnet', 'huber',
    
    # Baselines
    'global_mean', 'flat', 'trend', 'historical_curve', 'knn_curve',
    'baseline_naive', 'baseline_linear', 'baseline_exp', 'baseline',
    
    # Specialized
    'hybrid_lgbm', 'hybrid_xgb', 'hybrid',
    'arihow', 'ts_hybrid',
    
    # Ensembles
    'averaging', 'averaging_ensemble',
    'weighted', 'weighted_averaging', 'weighted_ensemble',
    'stacking', 'stacking_ensemble',
    'blending', 'blending_ensemble',
    'blender',
]
```

---

## Summary

| Model | Priority | GPU | Best Use Case |
|-------|----------|-----|---------------|
| **XGBoost** | 1 | ✅ | Primary model, best official_metric |
| **LightGBM** | 2 | ✅ | Fast training, ensemble partner |
| **Hybrid** | 2 | ✅ | Interpretable, domain knowledge |
| **CatBoost** | 3 | ✅ | Categorical features, diversity |
| **Linear** | 4 | ❌ | Baseline, interpretability |
| **ARIHOW** | 4 | ❌ | Time series patterns |
| **Neural Net** | 5 | ✅ | Experimental, complex patterns |

### Recommended Workflow

1. **Start with XGBoost** - Best standalone performance
2. **Add LightGBM** - Create XGB+LGBM ensemble
3. **Optimize ensemble weights** - Use EnsembleBlender on validation
4. **Optional: Add Hybrid** - If domain knowledge improves predictions
5. **Optional: Add CatBoost** - Only if improves ensemble

---

## License

This project is part of the Novartis Datathon 2025 competition. See LICENSE file for details.

---

**Last Updated**: November 2025
