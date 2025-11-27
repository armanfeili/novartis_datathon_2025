# Novartis Datathon 2025 - Project Functionality Documentation

> **Last Updated:** November 2025  
> **Author:** Arman Feili  
> **Repository:** [novartis_datathon_2025](https://github.com/armanfeili/novartis_datathon_2025)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Core Modules](#core-modules)
   - [Utilities](#1-utilities-srcutilspy)
   - [Data Management](#2-data-management-srcdatapy)
   - [Feature Engineering](#3-feature-engineering-srcfeaturespy)
   - [Validation](#4-validation-srcvalidationpy)
   - [Evaluation](#5-evaluation-srcevaluatepy)
   - [Training](#6-training-srctrainpy)
   - [Inference](#7-inference-srcinferencepy)
5. [Model Implementations](#model-implementations)
6. [Configuration System](#configuration-system)
7. [Main Notebook](#main-notebook-workflow)
8. [Complete Workflow](#complete-workflow)
9. [Usage Guide](#usage-guide)

---

## Overview

This project is a comprehensive machine learning framework designed for the **Novartis Datathon 2025** competition. It implements a hybrid development workflow that combines:

- **Local Development:** VS Code with GitHub Copilot for code editing and version control
- **Cloud Training:** Google Colab with GPU acceleration (T4/A100) for model training
- **Persistent Storage:** Google Drive for datasets, artifacts, and submissions

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | LightGBM, XGBoost, CatBoost, Linear Models, Neural Networks |
| **Configuration-Driven** | All settings externalized to YAML files |
| **Reproducibility** | Deterministic seeds across all libraries |
| **Experiment Tracking** | Automatic artifact management with config snapshots |
| **Cross-Validation** | KFold, Stratified, and Time Series CV strategies |
| **Environment Agnostic** | Auto-detection of Colab vs local development |

---

## Architecture

The project follows a three-tier architecture separating code, compute, and storage:

```
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL DEVELOPMENT (VS Code + Copilot)                          │
│  ├─ Edit src/, configs/, notebooks/                             │
│  ├─ Commit & push to GitHub                                     │
│  └─ No data/runs stored locally                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  COLAB COMPUTE (GPU Runtime)                                    │
│  ├─ Clone repo from GitHub (code only)                          │
│  ├─ Mount Google Drive at /content/drive                        │
│  ├─ Install dependencies from requirements.txt                  │
│  └─ Execute training → outputs to Drive                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  GOOGLE DRIVE STORAGE                                           │
│  novartis-datathon-2025/                                        │
│  ├─ data/raw/              ← Original datasets                  │
│  ├─ data/interim/          ← Cleaned/merged tables              │
│  ├─ data/processed/        ← Feature-engineered data            │
│  ├─ artifacts/runs/<id>/   ← Training outputs per run           │
│  │   ├─ config_used.yaml   ← Frozen config snapshot             │
│  │   ├─ metrics.json       ← Evaluation metrics                 │
│  │   ├─ oof_preds.csv      ← Out-of-fold predictions            │
│  │   ├─ model_fold_*.bin   ← Trained model weights              │
│  │   └─ logs.txt           ← Training logs                      │
│  └─ submissions/           ← Final submission files             │
└─────────────────────────────────────────────────────────────────┘
```

**Design Principle:** Code travels through Git. Data stays in Drive.

---

## Project Structure

```
novartis_datathon_2025/
│
├── configs/                    # Configuration files
│   ├── data.yaml              # Data paths and column definitions
│   ├── features.yaml          # Feature engineering settings
│   ├── run_defaults.yaml      # Experiment-level settings
│   ├── model_lgbm.yaml        # LightGBM hyperparameters
│   ├── model_xgb.yaml         # XGBoost hyperparameters
│   ├── model_cat.yaml         # CatBoost hyperparameters
│   ├── model_linear.yaml      # Linear model settings
│   └── model_nn.yaml          # Neural network architecture
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── utils.py               # Utility functions
│   ├── data.py                # Data loading and processing
│   ├── features.py            # Feature engineering
│   ├── validation.py          # Cross-validation strategies
│   ├── evaluate.py            # Metrics and visualization
│   ├── train.py               # Training pipeline
│   ├── inference.py           # Prediction generation
│   └── models/                # Model implementations
│       ├── base.py            # Abstract base class
│       ├── lgbm_model.py      # LightGBM wrapper
│       ├── xgb_model.py       # XGBoost wrapper
│       ├── cat_model.py       # CatBoost wrapper
│       ├── linear.py          # Sklearn linear models
│       └── nn.py              # PyTorch neural network
│
├── notebooks/                  # Jupyter notebooks
│   ├── 00_eda.ipynb           # Exploratory Data Analysis
│   ├── 01_feature_prototype.ipynb
│   ├── 01_train.ipynb
│   ├── 02_model_sanity.ipynb
│   └── colab/
│       └── main.ipynb         # Main Colab training notebook
│
├── data/                       # Data directories (local)
│   ├── raw/                   # Original data files
│   ├── interim/               # Intermediate processed data
│   └── processed/             # Final feature tables
│
├── docs/                       # Documentation
├── tests/                      # Unit tests
├── env/                        # Environment files
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

---

## Core Modules

### 1. Utilities (`src/utils.py`)

The utilities module provides foundational functions used throughout the project.

#### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `set_seed` | `(seed: int = 42)` | Sets random seed across Python, NumPy, and PyTorch for reproducibility |
| `get_device` | `() -> torch.device` | Returns CUDA device if available, otherwise CPU |
| `setup_logging` | `(log_path: str = None, level=INFO)` | Configures logging to console and optional file |
| `timer` | `(name: str)` | Context manager to time and log code block execution |
| `load_config` | `(config_path: str) -> dict` | Loads and parses YAML configuration files |
| `get_project_root` | `() -> Path` | Returns the project root directory |
| `resolve_path` | `(path_str: str, drive_base: str = None) -> Path` | Resolves paths with Drive variable substitution |

#### Example Usage

```python
from src.utils import set_seed, load_config, timer

# Set reproducibility
set_seed(42)

# Load configuration
config = load_config('configs/model_lgbm.yaml')

# Time a code block
with timer("Data Loading"):
    data = load_data()
# Output: [Data Loading] done in 2.345 s
```

---

### 2. Data Management (`src/data.py`)

The `DataManager` class handles all data loading, processing, and path resolution with automatic environment detection.

#### Class: `DataManager`

```python
class DataManager:
    def __init__(self, config: dict)
    def load_raw_data(self) -> dict[str, pd.DataFrame]
    def make_interim(self, data: dict) -> pd.DataFrame
    def make_processed(self, df: pd.DataFrame) -> pd.DataFrame
    def save_processed(self, df: pd.DataFrame, filename: str)
```

#### Environment Detection

The class automatically detects the runtime environment:

| Environment | Detection | Path Resolution |
|-------------|-----------|-----------------|
| **Colab** | `/content/drive` exists | Uses `config['drive']` paths |
| **Local** | Default | Uses `config['local']` paths |

#### Methods

| Method | Description |
|--------|-------------|
| `__init__` | Initializes paths based on environment, creates directories if needed |
| `_resolve_path` | Handles `${drive.base_path}` variable substitution |
| `load_raw_data` | Loads CSV files defined in config, returns dict of DataFrames |
| `make_interim` | Transforms raw data to interim format (cleaning, merging) |
| `make_processed` | Applies final processing to create feature tables |
| `save_processed` | Persists processed data to the processed directory |

#### Example Usage

```python
from src.data import DataManager
from src.utils import load_config

data_config = load_config('configs/data.yaml')
data_mgr = DataManager(data_config)

# Load all raw datasets
raw_data = data_mgr.load_raw_data()
# Returns: {'train': DataFrame, 'test': DataFrame, ...}

# Process to interim format
interim_df = data_mgr.make_interim(raw_data)
```

---

### 3. Feature Engineering (`src/features.py`)

The `FeatureEngineer` class creates features based on configuration settings.

#### Class: `FeatureEngineer`

```python
class FeatureEngineer:
    def __init__(self, config: dict)
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame
```

#### Feature Groups

All feature groups are configurable via `configs/features.yaml`:

| Group | Description | Config Key |
|-------|-------------|------------|
| **Basic** | Basic transformations | `feature_groups.basic` |
| **Time-Based** | Date/time component extraction | `feature_groups.time_based` |
| **Lags** | Lag features | `feature_groups.lags` |
| **Rolling** | Rolling window statistics | `feature_groups.rolling` |
| **Diff** | Difference features | `feature_groups.diff` |
| **Interactions** | Feature interactions | `feature_groups.interactions` |

#### Configuration Options

```yaml
# configs/features.yaml
feature_groups:
  basic: true
  lags: true
  rolling: true
  time_based: true

lags:
  windows: [1, 7, 14, 30]    # Lag periods
  columns: []                 # Columns to apply lags

rolling:
  windows: [7, 14, 30]       # Rolling window sizes
  functions:
    - mean
    - std
    - min
    - max
    - median

time_features:
  extract:
    - day_of_week
    - month
    - year
    - is_weekend
  cyclical_encoding: true    # sin/cos encoding
```

#### Example Usage

```python
from src.features import FeatureEngineer
from src.utils import load_config

features_config = load_config('configs/features.yaml')
fe = FeatureEngineer(features_config)

# Build all features
processed_df = fe.build_features(interim_df)
```

---

### 4. Validation (`src/validation.py`)

The `Validator` class implements multiple cross-validation strategies.

#### Class: `Validator`

```python
class Validator:
    def __init__(self, config: dict)
    def get_splits(self, df: pd.DataFrame, target_col: str = None) -> list
    def adversarial_validation(self, train_df, test_df)
```

#### Supported Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `kfold` | Standard K-Fold CV | General tabular data |
| `stratified_kfold` | Stratified K-Fold | Classification tasks |
| `time_series` | TimeSeriesSplit | Temporal/time-series data |

#### Configuration

```yaml
# configs/run_defaults.yaml
cv:
  strategy: "time_series"  # kfold, stratified_kfold, time_series
  n_splits: 5
  shuffle: false           # Usually false for time series
  
  time_series:
    gap: 0                 # Gap between train and validation
    max_train_size: null   # Maximum training set size
```

#### Example Usage

```python
from src.validation import Validator
from src.utils import load_config

run_config = load_config('configs/run_defaults.yaml')
validator = Validator(run_config)

# Get CV splits
splits = validator.get_splits(df, target_col='target')
# Returns: [(train_idx, val_idx), ...]

for fold, (train_idx, val_idx) in enumerate(splits):
    X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
```

---

### 5. Evaluation (`src/evaluate.py`)

The `Evaluator` class calculates metrics and creates visualizations.

#### Class: `Evaluator`

```python
class Evaluator:
    def __init__(self, config: dict)
    def calculate_metrics(self, y_true, y_pred) -> dict
    def plot_residuals(self, y_true, y_pred, save_path: str = None)
```

#### Supported Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `rmse` | Root Mean Squared Error | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ |
| `mae` | Mean Absolute Error | $\frac{1}{n}\sum|y - \hat{y}|$ |
| `r2` | R-squared | $1 - \frac{SS_{res}}{SS_{tot}}$ |
| `mape` | Mean Absolute Percentage Error | $\frac{100}{n}\sum|\frac{y - \hat{y}}{y}|$ |

#### Configuration

```yaml
# configs/run_defaults.yaml
metrics:
  primary: "rmse"          # Primary metric for model selection
  secondary:
    - mae
    - mape
    - r2
```

#### Example Usage

```python
from src.evaluate import Evaluator
from src.utils import load_config

run_config = load_config('configs/run_defaults.yaml')
evaluator = Evaluator(run_config)

# Calculate all metrics
metrics = evaluator.calculate_metrics(y_true, y_pred)
# Returns: {'rmse': 0.123, 'mae': 0.098, 'r2': 0.95, 'mape': 5.2}

# Create residual plot
evaluator.plot_residuals(y_true, y_pred, save_path='residuals.png')
```

---

### 6. Training (`src/train.py`)

The training module orchestrates the complete ML pipeline.

#### Main Function

```python
def run_experiment(
    model_name: str,
    model_config_path: str,
    run_name: str = None,
    config_path: str = 'configs/run_defaults.yaml'
) -> tuple[str, dict]
```

#### Training Pipeline Flow

```
┌────────────────────────────────────────────────────────────┐
│                    run_experiment()                         │
├────────────────────────────────────────────────────────────┤
│  1. Load Configurations                                     │
│     └─ run_config + model_config                           │
│                                                            │
│  2. Setup Run                                              │
│     ├─ Generate run_id (timestamp_model_name)              │
│     ├─ Create artifacts directory                          │
│     ├─ Setup logging                                       │
│     ├─ Set random seed                                     │
│     └─ Save config snapshot                                │
│                                                            │
│  3. Data Pipeline                                          │
│     ├─ DataManager.load_raw_data()                         │
│     └─ DataManager.make_interim()                          │
│                                                            │
│  4. Feature Engineering                                    │
│     └─ FeatureEngineer.build_features()                    │
│                                                            │
│  5. Validation Setup                                       │
│     └─ Validator.get_splits()                              │
│                                                            │
│  6. Training Loop (per fold)                               │
│     ├─ Split data into train/val                           │
│     ├─ Initialize model                                    │
│     ├─ model.fit(X_train, y_train, X_val, y_val)          │
│     ├─ Generate validation predictions                     │
│     ├─ Store OOF predictions                               │
│     └─ Save model checkpoint                               │
│                                                            │
│  7. Evaluation                                             │
│     ├─ Calculate CV metrics                                │
│     ├─ Save metrics.json                                   │
│     └─ Save oof_preds.csv                                  │
│                                                            │
│  8. Return (run_id, metrics)                               │
└────────────────────────────────────────────────────────────┘
```

#### Run Artifacts

Each training run creates an isolated directory:

```
artifacts/runs/2025-11-27_14-30_colab_run_01/
├── config_used.yaml     # Frozen configuration snapshot
├── metrics.json         # Evaluation metrics
├── oof_preds.csv        # Out-of-fold predictions
├── model_fold_0.bin     # Fold 0 model weights
├── model_fold_1.bin     # Fold 1 model weights
├── ...
└── logs.txt             # Training logs
```

#### Example Usage

```python
from src.train import run_experiment

# Run a LightGBM experiment
run_id, metrics = run_experiment(
    model_name='lightgbm',
    model_config_path='configs/model_lgbm.yaml',
    run_name='experiment_v1'
)

print(f"Run ID: {run_id}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

#### CLI Usage

```bash
python -m src.train \
    --model lightgbm \
    --model-config configs/model_lgbm.yaml \
    --config configs/run_defaults.yaml \
    --run-name my_experiment
```

---

### 7. Inference (`src/inference.py`)

Generates predictions using trained models.

#### Inference Pipeline

```
1. Load test data
2. Apply same feature engineering as training
3. Load all fold models from artifacts directory
4. Generate predictions from each model
5. Average predictions across folds (ensemble)
6. Save submission CSV
```

#### Example Usage

```python
from src.inference import generate_submission

# Generate submission from a trained run
submission = generate_submission(
    run_id='2025-11-27_14-30_experiment_v1',
    test_df=test_processed,
    id_col='id'
)
# Saves to: submissions/submission_<run_id>.csv
```

---

## Model Implementations

All models inherit from `BaseModel` abstract class, ensuring a unified interface.

### Base Model Interface

```python
# src/models/base.py
class BaseModel(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass
```

### Available Models

#### 1. LightGBM (`src/models/lgbm_model.py`)

```yaml
# configs/model_lgbm.yaml
model:
  name: "lightgbm"
  task: "regression"

params:
  boosting_type: "gbdt"
  objective: "regression"
  metric: "rmse"
  num_leaves: 31
  max_depth: -1
  learning_rate: 0.05
  n_estimators: 1000
  feature_fraction: 0.8
  bagging_fraction: 0.8
  early_stopping_rounds: 50
```

#### 2. XGBoost (`src/models/xgb_model.py`)

```yaml
# configs/model_xgb.yaml
model:
  name: "xgboost"
  task: "regression"

params:
  booster: "gbtree"
  objective: "reg:squarederror"
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 1000
  subsample: 0.8
  colsample_bytree: 0.8
  # GPU support:
  # tree_method: "gpu_hist"
```

#### 3. CatBoost (`src/models/cat_model.py`)

```yaml
# configs/model_cat.yaml
model:
  name: "catboost"
  task: "regression"

params:
  loss_function: "RMSE"
  depth: 6
  learning_rate: 0.05
  iterations: 1000
  l2_leaf_reg: 3.0
  # GPU support:
  # task_type: "GPU"
```

#### 4. Linear Models (`src/models/linear.py`)

```yaml
# configs/model_linear.yaml
model:
  name: "linear"
  type: "ridge"  # ridge, lasso, elasticnet, huber

ridge:
  alpha: 1.0

preprocessing:
  scale_features: true
  handle_missing: "mean"
```

Includes preprocessing pipeline with:
- Missing value imputation
- Feature scaling (StandardScaler)
- Choice of regressor (Ridge, Lasso, ElasticNet, Huber)

#### 5. Neural Network (`src/models/nn.py`)

```yaml
# configs/model_nn.yaml
model:
  name: "neural_network"
  task: "regression"

architecture:
  mlp:
    hidden_layers: [256, 128, 64]
    dropout: 0.2

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
```

Architecture:
```
Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear → Output
```

### Model Selection

```python
from src.train import get_model_class

# Get model class by name
ModelClass = get_model_class('lightgbm')  # Returns LGBMModel
ModelClass = get_model_class('xgboost')   # Returns XGBModel
ModelClass = get_model_class('catboost')  # Returns CatBoostModel
ModelClass = get_model_class('linear')    # Returns LinearModel
ModelClass = get_model_class('neural_network')  # Returns NNModel
```

---

## Configuration System

### Configuration Files Overview

| File | Purpose |
|------|---------|
| `data.yaml` | Data paths, file names, column definitions |
| `features.yaml` | Feature engineering settings |
| `run_defaults.yaml` | Experiment settings, CV, metrics, paths |
| `model_*.yaml` | Model-specific hyperparameters |

### `configs/data.yaml`

```yaml
# Google Drive paths (Colab)
drive:
  base_path: "/content/drive/MyDrive/novartis-datathon-2025"
  raw_dir: "/content/drive/MyDrive/novartis-datathon-2025/data/raw"
  interim_dir: "/content/drive/MyDrive/novartis-datathon-2025/data/interim"
  processed_dir: "/content/drive/MyDrive/novartis-datathon-2025/data/processed"

# Local development paths
local:
  base_path: "."
  raw_dir: "./data/raw"
  interim_dir: "./data/interim"
  processed_dir: "./data/processed"

# Data files (update with your filenames)
files:
  train: "train.csv"
  test: "test.csv"

# Column definitions
columns:
  target: null          # Set your target column
  categorical: []
  numerical: []

keys:
  primary_key: "id"
  time_key: "date"
```

### `configs/run_defaults.yaml`

```yaml
reproducibility:
  seed: 42
  deterministic: true

cv:
  strategy: "time_series"
  n_splits: 5
  shuffle: false

paths:
  artifacts_dir: "artifacts/runs"
  submissions_dir: "submissions"

metrics:
  primary: "rmse"
  secondary: [mae, mape, r2]

hardware:
  use_gpu: true
  mixed_precision: true
  num_workers: 4
```

---

## Main Notebook Workflow

The main notebook (`notebooks/colab/main.ipynb`) provides a comprehensive 10-section workflow:

### Sections Overview

| # | Section | Description |
|---|---------|-------------|
| 1 | **Environment Setup** | Mount Drive, clone repo, detect environment |
| 2 | **Import Modules** | Import all project modules, check GPU |
| 3 | **Load Configurations** | Load YAML configs, set random seed |
| 4 | **Data Loading & EDA** | Initialize DataManager, explore data |
| 5 | **Feature Engineering** | Build features with FeatureEngineer |
| 6 | **Model Training** | Configure and run experiment |
| 7 | **Evaluation** | Analyze metrics, visualize predictions |
| 8 | **Inference** | Generate predictions and submissions |
| 9 | **Experiment Tracking** | View run history, compare metrics |
| 10 | **Utilities** | Helper functions for common operations |

### Quick Start in Colab

```python
# Section 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Section 6: Run Training
run_id, metrics = run_experiment(
    model_name='lightgbm',
    model_config_path='configs/model_lgbm.yaml',
    run_name='my_experiment'
)
```

### Utility Functions

```python
# Available utilities
sync_to_drive()           # Sync changes to Google Drive
download_submission(id)   # Download submission CSV
upload_data()             # Upload data files
show_gpu_info()           # Display GPU information
clear_gpu_cache()         # Clear GPU memory
```

---

## Complete Workflow

### Step-by-Step Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                      PREPARATION PHASE                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Upload raw data to Google Drive                              │
│     └─ MyDrive/novartis-datathon-2025/data/raw/                 │
│                                                                  │
│  2. Update configs/data.yaml                                     │
│     ├─ Set file names in `files` section                        │
│     ├─ Set target column                                        │
│     └─ Define categorical/numerical columns                     │
│                                                                  │
│  3. Configure features (configs/features.yaml)                   │
│     └─ Enable/disable feature groups                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                              │
├─────────────────────────────────────────────────────────────────┤
│  4. Open notebook in Colab                                       │
│     └─ Enable GPU runtime (T4 or A100)                          │
│                                                                  │
│  5. Run all setup cells (1-5)                                    │
│                                                                  │
│  6. Configure experiment                                         │
│     ├─ MODEL_NAME = "lightgbm"                                  │
│     └─ RUN_NAME = "experiment_v1"                               │
│                                                                  │
│  7. Execute training                                             │
│     └─ run_experiment(model_name, config_path, run_name)        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION PHASE                             │
├─────────────────────────────────────────────────────────────────┤
│  8. Review training outputs                                      │
│     ├─ Check metrics.json                                       │
│     ├─ Analyze OOF predictions                                  │
│     └─ View actual vs predicted plots                           │
│                                                                  │
│  9. Compare experiments                                          │
│     └─ Use experiment tracking section                          │
│                                                                  │
│  10. Iterate on hyperparameters                                  │
│      └─ Modify model config and re-train                        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SUBMISSION PHASE                             │
├─────────────────────────────────────────────────────────────────┤
│  11. Generate predictions                                        │
│      └─ generate_submission(run_id, test_df)                    │
│                                                                  │
│  12. Download submission file                                    │
│      └─ download_submission(run_id)                             │
│                                                                  │
│  13. Submit to competition                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Guide

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/armanfeili/novartis_datathon_2025.git
cd novartis_datathon_2025

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training locally (if you have data)
python -m src.train \
    --model lightgbm \
    --model-config configs/model_lgbm.yaml \
    --run-name local_test
```

### Colab Training

1. Click the Colab badge in README or open `notebooks/colab/main.ipynb`
2. Enable GPU: **Runtime** → **Change runtime type** → **GPU**
3. Run all cells in order

### Adding a New Model

1. Create `src/models/my_model.py` inheriting from `BaseModel`
2. Implement `fit`, `predict`, `save`, `load` methods
3. Create `configs/model_mymodel.yaml` with hyperparameters
4. Register in `src/train.py` `get_model_class()` function

### Adding New Features

1. Add feature generation method to `FeatureEngineer` class
2. Update `configs/features.yaml` with new feature group
3. Enable in `feature_groups` section

---

## Dependencies

Core dependencies listed in `requirements.txt`:

```
# ML/DL Frameworks
torch
torchvision
lightgbm
xgboost
catboost
scikit-learn

# Data Processing
numpy
pandas

# Visualization
matplotlib
seaborn
plotly

# Utilities
pyyaml
joblib
tqdm
ipykernel
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors in Colab | Run dependency installation cell first |
| Data not found | Check `configs/data.yaml` paths and file names |
| GPU not detected | Ensure GPU runtime is enabled in Colab |
| Out of memory | Reduce batch size or use gradient accumulation |
| Training fails | Check logs.txt in artifacts directory |

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Open Pull Request

---

## License

MIT License - see [LICENSE](../LICENSE) for details.

---

## Contact

**Author:** Arman Feili  
**GitHub:** [@armanfeili](https://github.com/armanfeili)  
**Repository:** [novartis_datathon_2025](https://github.com/armanfeili/novartis_datathon_2025)
