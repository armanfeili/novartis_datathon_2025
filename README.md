# Novartis Datathon 2025 – Generic Erosion Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-372%20passed-green.svg)](#testing)

> **🚀 Quick Access:**
> - **Google Colab (Recommended):** [Open in Colab](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb)
> - **Local Jupyter:** `jupyter notebook notebooks/colab/main.ipynb` → [http://localhost:8888](http://localhost:8888)

Predict pharmaceutical brand volume erosion following Loss of Exclusivity (LOE) events when generic competitors enter the market.

---

## Table of Contents

1. [Competition Overview](#competition-overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Overview](#data-overview)
6. [Configuration](#configuration)
7. [CLI Reference](#cli-reference)
8. [Feature Engineering](#feature-engineering)
9. [Models](#models)
10. [Validation Strategy](#validation-strategy)
11. [Metric Calculation](#metric-calculation)
12. [Hyperparameter Optimization](#hyperparameter-optimization)
13. [Testing](#testing)
14. [Notebooks](#notebooks)
15. [Reproducibility](#reproducibility)
16. [Troubleshooting](#troubleshooting)

---

## Competition Overview

### Problem Statement

When a pharmaceutical drug loses patent protection (Loss of Exclusivity / LOE), generic manufacturers can enter the market, causing volume erosion for the original branded product. This competition challenges participants to **forecast 24-month post-LOE volume trajectories** for pharmaceutical brands across multiple countries.

### Two Scenarios

| Scenario | Available Data | Forecast Horizon | Metric |
|----------|---------------|------------------|--------|
| **Scenario 1** | Pre-LOE history only (months < 0) | Months 0–23 | Metric 1 |
| **Scenario 2** | Pre-LOE + first 6 months (months 0–5) | Months 6–23 | Metric 2 |

### Target Variable

- **Normalized volume** (`y_norm`): `volume[t] / avg_vol_12m`
- `avg_vol_12m`: 12-month average pre-LOE volume (baseline)
- Final submission requires **absolute volume** (inverse transform: `volume = y_norm × avg_vol_12m`)

### Bucket Classification

Series are classified into buckets based on mean erosion:
- **Bucket 1** (high erosion): `mean_erosion ≤ 0.25` → Weight 2× in metric
- **Bucket 2** (low erosion): `mean_erosion > 0.25` → Weight 1× in metric

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd novartis_datathon_2025

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests to verify installation
pytest tests/ -v

# 5. Train a model (Scenario 1, CatBoost)
python -m src.train --scenario 1 --model catboost --output artifacts/catboost_s1

# 6. Generate predictions
python -m src.inference --model-s1 artifacts/catboost_s1/model.bin \
                         --model-s2 artifacts/catboost_s2/model.bin \
                         --output submissions/submission.csv
```

---

## Project Structure

```
novartis_datathon_2025/
├── configs/                    # YAML configuration files
│   ├── data.yaml               # Data paths, columns, schema
│   ├── features.yaml           # Feature engineering settings
│   ├── run_defaults.yaml       # Training defaults, seeds, metrics
│   ├── model_cat.yaml          # CatBoost hyperparameters
│   ├── model_lgbm.yaml         # LightGBM hyperparameters
│   ├── model_xgb.yaml          # XGBoost hyperparameters
│   ├── model_linear.yaml       # Linear model settings
│   └── model_nn.yaml           # Neural network settings
│
├── data/                       # Data directory (gitignored raw data)
│   ├── raw/                    # Original competition data
│   │   ├── TRAIN/              # Training data files
│   │   └── TEST/               # Test data files
│   ├── interim/                # Intermediate processed data
│   ├── processed/              # Final processed datasets
│   └── external/               # External data sources
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data.py                 # Data loading, panel creation
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # Training pipeline with CLI
│   ├── inference.py            # Prediction and submission generation
│   ├── evaluate.py             # Metric computation
│   ├── validation.py           # CV splits, series-level validation
│   ├── utils.py                # Logging, config loading, utilities
│   └── models/                 # Model implementations
│       ├── base.py             # BaseModel abstract class
│       ├── cat_model.py        # CatBoost wrapper
│       ├── lgbm_model.py       # LightGBM wrapper
│       ├── xgb_model.py        # XGBoost wrapper
│       ├── linear.py           # Linear models (Ridge, Lasso)
│       ├── nn.py               # Neural network (PyTorch)
│       └── ensemble.py         # Ensemble methods
│
├── scripts/                    # Utility scripts
│   ├── reproduce.sh            # Full reproduction script
│   ├── train_and_submit_bonus.sh   # Training with bonus features
│   ├── train_and_submit_complete.py
│   ├── train_catboost_bonus_complete.py
│   ├── run_full_training_bonus.py
│   └── generate_submission_from_trained.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── 00_eda.ipynb            # Exploratory data analysis
│   ├── 01_feature_prototype.ipynb  # Feature prototyping
│   ├── 01_train.ipynb          # Interactive training
│   ├── 02_model_sanity.ipynb   # Model sanity checks
│   └── colab/                  # Google Colab notebook
│       ├── main.ipynb
│       └── colab_requirements.txt
│
├── docs/                       # Documentation
│   ├── CODEBASE_DOCUMENTATION.md
│   ├── MODEL_COMPARISON.md
│   ├── guide/                  # Competition guide files
│   │   ├── metric_calculation.py   # Official metric script
│   │   ├── submission_template.csv
│   │   └── auxiliar_metric_computation_example.csv
│   ├── planning/               # Planning & TODO documents
│   │   ├── TODO.md
│   │   ├── TODO_2.md
│   │   ├── TODO_BONUS.md
│   │   ├── approach.md
│   │   └── functionality.md
│   └── results/                # Experiment results
│
├── tests/                      # Test suite
│   ├── conftest.py             # Pytest fixtures
│   └── test_smoke.py           # Smoke tests
│
├── submissions/                # Generated submissions
├── artifacts/                  # Trained models, logs, metrics
├── logs/                       # Training logs
│
├── requirements.txt            # Pip requirements
├── environment.yml             # Conda environment
├── pytest.ini                  # Pytest configuration
├── .gitignore                  # Git ignore rules
├── CONTRIBUTING.md             # Contribution guidelines
└── LICENSE                     # MIT License
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda
- ~8GB RAM recommended
- GPU optional (supports CUDA for neural networks)

### Option 1: pip (Recommended)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import catboost; print('OK')"
```

### Option 2: Conda

```bash
conda env create -f env/environment.yml
conda activate novartis
```

### Option 3: Google Colab

Upload `notebooks/colab/main.ipynb` to Colab and follow in-notebook instructions.

---

## Data Overview

### Source Files

Located in `data/raw/TRAIN/` and `data/raw/TEST/`:

| File | Description | Key Columns |
|------|-------------|-------------|
| `df_volume_*.csv` | Monthly volume data | country, brand_name, month, months_postgx, volume |
| `df_generics_*.csv` | Generic competitor count | country, brand_name, months_postgx, n_gxs |
| `df_medicine_info_*.csv` | Drug characteristics | ther_area, hospital_rate, main_package, biological, small_molecule |

### Key Concepts

- **Series**: Unique (country, brand_name) combination
- **months_postgx**: Months relative to LOE (0 = LOE month, negative = pre-LOE)
- **n_gxs**: Number of generic competitors at time t

### Data Statistics

| Metric | Train | Test S1 | Test S2 |
|--------|-------|---------|---------|
| Series | 1,953 | 228 | 112 |
| Total Rows | ~47,000 | ~5,500 | ~2,000 |

---

## Configuration

All settings are in `configs/`. See [`configs/README.md`](configs/README.md) for detailed documentation.

### Key Configuration Files

| File | Purpose |
|------|---------|
| `data.yaml` | Data paths, column names, missing value handling |
| `features.yaml` | Feature engineering options, leakage prevention |
| `run_defaults.yaml` | Seeds, validation, sample weights, metric definitions |
| `model_*.yaml` | Model-specific hyperparameters |

### Example: Changing Random Seed

```yaml
# configs/run_defaults.yaml
reproducibility:
  seed: 42  # Change this value
  deterministic: true
```

---

## CLI Reference

### Training

```bash
# Basic training
python -m src.train --scenario 1 --model catboost

# Full options
python -m src.train \
    --scenario 1 \                      # 1 or 2
    --model catboost \                  # catboost, lightgbm, xgboost, linear, nn
    --model-config configs/model_cat.yaml \
    --run-config configs/run_defaults.yaml \
    --data-config configs/data.yaml \
    --features-config configs/features.yaml \
    --output artifacts/my_run \
    --run-name "experiment_v1" \
    --val-fraction 0.2 \
    --seed 42 \
    --use-metric-weights \              # Align training weights with metric
    --enable-checkpoints \              # Save checkpoints
    --enable-profiling                  # Memory profiling
```

### Cross-Validation

```bash
python -m src.train \
    --scenario 1 \
    --model catboost \
    --cv 5 \                            # 5-fold CV
    --stratify bucket                   # Stratify by bucket
```

### Hyperparameter Optimization

```bash
python -m src.train \
    --scenario 1 \
    --model catboost \
    --hpo \                             # Enable HPO
    --hpo-trials 100 \                  # Number of Optuna trials
    --hpo-timeout 3600                  # Timeout in seconds
```

### Inference

```bash
python -m src.inference \
    --model-s1 artifacts/catboost_s1/model.bin \
    --model-s2 artifacts/catboost_s2/model.bin \
    --output submissions/submission.csv \
    --use-versioning \                  # Automatic versioning
    --save-auxiliary                    # Generate aux file for local metrics
```

### Get Help

```bash
python -m src.train --help
python -m src.inference --help
```

---

## Feature Engineering

### Feature Categories

| Category | Features | Scenario |
|----------|----------|----------|
| **Pre-entry stats** | avg_vol_3m, avg_vol_6m, avg_vol_12m, volatility, trend | Both |
| **Time features** | months_postgx, months_postgx², time_bucket, quarter | Both |
| **Generics** | n_gxs, has_generic, cummax_n_gxs, first_generic_month | Both |
| **Drug characteristics** | ther_area, hospital_rate, biological, small_molecule | Both |
| **Early signal** | avg_vol_0_5, erosion_0_5, trend_0_5 | S2 only |

### Leakage Prevention

**Critical**: The following columns are **NEVER** used as features:
- `bucket` – derived from target
- `y_norm`, `volume` – target variables
- `mean_erosion` – derived from target
- `country`, `brand_name` – meta columns (used for grouping only)

These are defined in `META_COLS` in `src/data.py`.

---

## Models

### Supported Models

| Model | CLI Name | Config File | Notes |
|-------|----------|-------------|-------|
| CatBoost | `catboost` | `model_cat.yaml` | **Primary model**, handles categoricals natively |
| LightGBM | `lightgbm` | `model_lgbm.yaml` | Fast training, good for HPO |
| XGBoost | `xgboost` | `model_xgb.yaml` | GPU support |
| Linear | `linear` | `model_linear.yaml` | Ridge, Lasso, ElasticNet |
| Neural Network | `nn` | `model_nn.yaml` | PyTorch MLP |

### Model Interface

All models inherit from `BaseModel` and implement:
```python
class BaseModel:
    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None): ...
    def predict(self, X): ...
    def save(self, path): ...
    def load(self, path): ...
    def get_feature_importance(self): ...
```

### Ensemble

```python
from src.models.ensemble import WeightedEnsemble

ensemble = WeightedEnsemble(
    models=[model_cat, model_lgbm],
    weights=[0.6, 0.4]
)
predictions = ensemble.predict(X_test)
```

---

## Validation Strategy

### Series-Level Split

**Critical**: Split at the **series level**, not row level, to prevent data leakage.

```python
from src.validation import create_validation_split

train_df, val_df = create_validation_split(
    df,
    val_fraction=0.2,
    stratify_by='bucket',  # Ensures balanced bucket distribution
    random_state=42
)
```

### Cross-Validation

```python
from src.validation import get_fold_series

for fold_idx, (train_series, val_series) in enumerate(
    get_fold_series(df, n_folds=5, stratify_by='bucket')
):
    train_df = df[df[['country', 'brand_name']].apply(tuple, axis=1).isin(train_series)]
    val_df = df[df[['country', 'brand_name']].apply(tuple, axis=1).isin(val_series)]
    # ... train and evaluate
```

---

## Metric Calculation

### Using the Official Metric Script

The official metric script is at `docs/guide/metric_calculation.py`. Here's how to use it:

```python
import pandas as pd
from docs.guide.metric_calculation import compute_metric1, compute_metric2

# Prepare DataFrames
df_actual = pd.DataFrame({
    'country': [...],
    'brand_name': [...],
    'months_postgx': [...],
    'volume': [...]  # Actual volumes
})

df_pred = pd.DataFrame({
    'country': [...],
    'brand_name': [...],
    'months_postgx': [...],
    'volume': [...]  # Predicted volumes
})

# Auxiliary file with bucket and avg_vol
df_aux = pd.DataFrame({
    'country': [...],
    'brand_name': [...],
    'avg_vol': [...],     # avg_vol_12m
    'bucket': [...]       # 1 or 2
})

# Compute metrics
metric1_score = compute_metric1(df_actual, df_pred, df_aux)  # Scenario 1
metric2_score = compute_metric2(df_actual, df_pred, df_aux)  # Scenario 2

print(f"Metric 1: {metric1_score:.4f}")
print(f"Metric 2: {metric2_score:.4f}")
```

### Our Wrapper Functions

We also provide wrapper functions in `src/evaluate.py`:

```python
from src.evaluate import compute_metric1, compute_metric2, create_aux_file

# Create aux file from panel data
df_aux = create_aux_file(panel_df)

# Compute metrics
m1 = compute_metric1(df_actual, df_pred, df_aux)
m2 = compute_metric2(df_actual, df_pred, df_aux)
```

### Metric Formulas

**Metric 1 (Scenario 1)**:
```
PE = 0.2 × Σ|actual-pred|/(24×avg_vol)      [months 0-23, monthly]
   + 0.5 × |Σ(actual)-Σ(pred)|/(6×avg_vol)  [months 0-5, accumulated]
   + 0.2 × |Σ(actual)-Σ(pred)|/(6×avg_vol)  [months 6-11, accumulated]
   + 0.1 × |Σ(actual)-Σ(pred)|/(12×avg_vol) [months 12-23, accumulated]

Final = (2/n₁)×Σ(PE_bucket1) + (1/n₂)×Σ(PE_bucket2)
```

**Metric 2 (Scenario 2)**:
```
PE = 0.2 × Σ|actual-pred|/(18×avg_vol)      [months 6-23, monthly]
   + 0.5 × |Σ(actual)-Σ(pred)|/(6×avg_vol)  [months 6-11, accumulated]
   + 0.3 × |Σ(actual)-Σ(pred)|/(12×avg_vol) [months 12-23, accumulated]

Final = (2/n₁)×Σ(PE_bucket1) + (1/n₂)×Σ(PE_bucket2)
```

---

## Hyperparameter Optimization

### Using Optuna

```bash
# CLI
python -m src.train --scenario 1 --model catboost --hpo --hpo-trials 100

# Python API
from src.train import run_hyperparameter_optimization

results = run_hyperparameter_optimization(
    X_train, y_train, meta_train,
    X_val, y_val, meta_val,
    scenario=1,
    model_type='catboost',
    n_trials=100,
    timeout=3600,
    artifacts_dir='artifacts/hpo'
)

print(f"Best params: {results['best_params']}")
print(f"Best score: {results['best_value']:.4f}")
```

### Default Search Spaces

Defined in `configs/model_*.yaml` under `tuning.search_space`:

```yaml
# CatBoost example
tuning:
  search_space:
    depth: [4, 8]
    learning_rate: [0.01, 0.1]
    l2_leaf_reg: [1.0, 10.0]
```

---

## Testing

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_smoke.py -v

# Run tests matching pattern
pytest tests/ -k "test_metric" -v
```

### Test Requirements

- **All 198 tests must pass**
- **No skipped tests** (all features implemented)
- **No warnings** (clean output)

### Test Categories

| Category | Description |
|----------|-------------|
| Data tests | Loading, panel creation, missing values |
| Feature tests | Feature engineering, leakage prevention |
| Model tests | Training, prediction, serialization |
| Metric tests | Official metric calculation |
| Validation tests | CV splits, series-level validation |
| Integration tests | End-to-end pipeline |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_eda.ipynb` | Exploratory data analysis: distributions, correlations, visualizations |
| `01_feature_prototype.ipynb` | Feature engineering experiments and validation |
| `01_train.ipynb` | Interactive model training and evaluation |
| `02_model_sanity.ipynb` | Model sanity checks: predictions, residuals, feature importance |
| `colab/main.ipynb` | Complete pipeline for Google Colab execution |

---

## Reproducibility

### Random Seed Control

All randomness is controlled via configuration:

```yaml
# configs/run_defaults.yaml
reproducibility:
  seed: 42
  deterministic: true
```

Seeds are applied to:
- NumPy
- Python random
- PyTorch (if used)
- Scikit-learn
- CatBoost/LightGBM/XGBoost

### Reproducing Results

```bash
# 1. Use exact dependencies
pip install -r requirements.txt

# 2. Use the same seed
python -m src.train --scenario 1 --model catboost --seed 42

# 3. Config snapshot is saved automatically
ls artifacts/my_run/configs/
# → data.yaml, features.yaml, run_defaults.yaml, model_cat.yaml, config_hash.txt
```

### Hardware Requirements

- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU
- **GPU**: Optional (CUDA 11+ for neural networks)

### Run Script for Full Reproduction

```bash
#!/bin/bash
# reproduce.sh - Full reproduction from raw data to submission

set -e

# Activate environment
source .venv/bin/activate

# Train Scenario 1
python -m src.train \
    --scenario 1 \
    --model catboost \
    --seed 42 \
    --output artifacts/catboost_s1

# Train Scenario 2
python -m src.train \
    --scenario 2 \
    --model catboost \
    --seed 42 \
    --output artifacts/catboost_s2

# Generate submission
python -m src.inference \
    --model-s1 artifacts/catboost_s1/model.bin \
    --model-s2 artifacts/catboost_s2/model.bin \
    --output submissions/final_submission.csv \
    --save-auxiliary

echo "Submission generated at submissions/final_submission.csv"
```

---

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the project root
cd novartis_datathon_2025
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Memory Errors
```python
# Use batch prediction for large datasets
from src.inference import predict_batch
predictions = predict_batch(model, X_test, batch_size=5000)
```

#### Missing Data Files
```
Ensure data is placed in data/raw/TRAIN/ and data/raw/TEST/
```

#### CatBoost Categorical Errors
```python
# Ensure categorical columns are string type
df['ther_area'] = df['ther_area'].astype(str)
```

### Getting Help

1. Check `TODO.md` for known issues and planned features
2. Run tests to verify installation: `pytest tests/ -v`
3. Check logs in `artifacts/<run_name>/train.log`

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Novartis for hosting the datathon
- Competition organizers for providing the dataset and evaluation framework
