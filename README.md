# Novartis Datathon 2025 – Generic Erosion Forecasting

---

**Repository:** `novartis_datathon_2025`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb)

This project implements an end-to-end, **config-driven forecasting pipeline** for the **Novartis Datathon 2025** generic erosion challenge hosted by the Barcelona Digital Finance Hub.

> **Goal:** Forecast monthly sales volume erosion after generic entry for each (country, brand) under two official scenarios, using the competition's custom error metrics and submission format.

**Key Features:**

- ✅ **Fully Implemented Pipeline** – Data loading, feature engineering, training, inference, and submission generation
- ✅ **Multiple Model Support** – CatBoost (hero), LightGBM, XGBoost, Linear models, and baselines
- ✅ **Scenario-Aware Features** – Strict leakage prevention with proper feature cutoffs
- ✅ **Sample Weights** – Aligned with official metric weighting (bucket & time-window)
- ✅ **Series-Level Validation** – Never mix months from the same series across train/val
- ✅ **Config-Driven** – YAML configurations for data, features, models, and runs
- ✅ **Reproducibility** – Deterministic seeding, config snapshots, structured artifacts

---

## Table of Contents

1. [Competition Problem](#1-competition-problem)
2. [Data Overview](#2-data-overview)
3. [Architecture](#3-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Implemented Modules](#5-implemented-modules)
6. [Feature Engineering](#6-feature-engineering)
7. [Models](#7-models)
8. [Quick Start](#8-quick-start)
9. [CLI Usage](#9-cli-usage)
10. [Configuration Files](#10-configuration-files)
11. [Testing](#11-testing)
12. [Important Technical Details](#12-important-technical-details)
13. [License & Creator](#13-license--creator)

---

## 1. Competition Problem

Novartis wants to anticipate how **branded drug volume erodes after generic (gx) entry**. The datathon provides de-identified commercial data from **50+ countries** and asks participants to:

### Scenarios

| Scenario | Forecast Range | Available Data | Official Metric |
|----------|---------------|----------------|-----------------|
| **Scenario 1** | Months 0–23 | Pre-entry only (months < 0) | Metric 1 (Phase 1A) |
| **Scenario 2** | Months 6–23 | Pre-entry + first 6 months (months < 6) | Metric 2 (Phase 1B) |

### Evaluation Metric

The official **Prediction Error (PE)** metric combines:
- **Monthly errors** weighted by time windows (50% early, 30% mid, 20% late)
- **Cumulative differences** at window endpoints
- **Bucket weighting**: Bucket 1 (high erosion) = **2×**, Bucket 2 = **1×**

Lower PE is better. The competition uses `metric_calculation.py` with `auxiliar_metric_computation.csv` for official scoring.

---

## 2. Data Overview

### Training Data (1,953 series)

| File | Description | Key Columns |
|------|-------------|-------------|
| `df_volume_train.csv` | Time-series with target | `country`, `brand_name`, `month`, `months_postgx`, `volume` |
| `df_generics_train.csv` | Generic competitor count | `country`, `brand_name`, `months_postgx`, `n_gxs` |
| `df_medicine_info_train.csv` | Static drug attributes | `ther_area`, `hospital_rate`, `main_package`, `biological`, `small_molecule` |

### Test Data (340 series: 228 S1 + 112 S2)

Same schema with missing `volume` for post-entry months to predict.

### Metrics & Submission Helpers

| File | Purpose |
|------|---------|
| `auxiliar_metric_computation.csv` | Contains `avg_vol` and `bucket` per series |
| `metric_calculation.py` | Official metric implementation |
| `submission_template.csv` | Required format: `country`, `brand_name`, `month`, `volume` |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL / VS Code                                                │
│  ├─ Edit src/, configs/, notebooks/                             │
│  ├─ Run tests: pytest tests/                                    │
│  └─ Commit & push to GitHub                                     │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXECUTION (Local or Colab)                                     │
│  ├─ python -m src.train --scenario 1 --model catboost           │
│  ├─ python -m src.train --scenario 2 --model lightgbm           │
│  └─ Generate submission CSV                                     │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STORAGE                                                        │
│  ├─ data/raw/           ← Official CSVs (TRAIN/, TEST/)         │
│  ├─ data/interim/       ← Merged panels                         │
│  ├─ data/processed/     ← Feature matrices                      │
│  ├─ artifacts/          ← Model checkpoints, metrics, logs      │
│  └─ submissions/        ← Final submission_*.csv                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Repository Structure

```
novartis_datathon_2025/
├── configs/                    # YAML configuration files
│   ├── data.yaml               # Paths, columns, schema, missing value handling
│   ├── features.yaml           # Feature engineering configuration
│   ├── run_defaults.yaml       # Validation, scenarios, sample weights
│   ├── model_cat.yaml          # CatBoost hyperparameters
│   ├── model_lgbm.yaml         # LightGBM hyperparameters
│   ├── model_xgb.yaml          # XGBoost hyperparameters
│   ├── model_linear.yaml       # Linear model configuration
│   └── model_nn.yaml           # Neural network configuration
│
├── src/                        # Core Python modules
│   ├── __init__.py
│   ├── data.py                 # Data loading & panel construction
│   ├── features.py             # Scenario-aware feature engineering
│   ├── train.py                # Training pipeline with CLI
│   ├── inference.py            # Prediction & submission generation
│   ├── evaluate.py             # Official metric wrappers
│   ├── validation.py           # Series-level stratified validation
│   ├── utils.py                # Seeding, logging, config loading
│   └── models/                 # Model implementations
│       ├── __init__.py
│       ├── base.py             # Abstract BaseModel interface
│       ├── cat_model.py        # CatBoost implementation
│       ├── lgbm_model.py       # LightGBM implementation
│       ├── xgb_model.py        # XGBoost implementation
│       └── linear.py           # Linear + baseline models
│
├── data/
│   ├── raw/                    # Original competition data
│   │   ├── TRAIN/              # Training CSVs
│   │   └── TEST/               # Test CSVs
│   ├── interim/                # Merged/cleaned panels
│   └── processed/              # Feature matrices
│
├── notebooks/                  # Jupyter notebooks
│   ├── 00_eda.ipynb            # Exploratory data analysis
│   ├── 01_feature_prototype.ipynb
│   ├── 01_train.ipynb
│   ├── 02_model_sanity.ipynb
│   └── colab/
│       └── main.ipynb          # Colab-friendly workflow
│
├── docs/                       # Documentation
│   ├── guide/                  # Competition guide files
│   ├── instructions/           # Detailed documentation
│   └── planning/               # Planning & approach docs
│
├── tests/
│   └── test_smoke.py           # Smoke tests for all modules
│
├── submissions/                # Generated submission files
├── env/                        # Environment files
│   ├── requirements.txt
│   ├── environment.yml
│   └── colab_requirements.txt
│
├── requirements.txt            # Main dependencies
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 5. Implemented Modules

### `src/data.py` – Data Loading & Panel Construction

```python
from src.data import load_raw_data, prepare_base_panel, compute_pre_entry_stats

# Load raw CSVs
df_vol, df_gen, df_med = load_raw_data(config, split='train')

# Build panel with merges and computed columns
panel = prepare_base_panel(df_vol, df_gen, df_med, is_train=True)
# Columns: country, brand_name, month, months_postgx, volume, n_gxs, 
#          ther_area, hospital_rate, ..., avg_vol_12m, y_norm, bucket
```

**Key Functions:**
- `load_raw_data()` – Load volume, generics, medicine_info CSVs
- `prepare_base_panel()` – Merge tables, compute `avg_vol_12m`, `y_norm`, `bucket`
- `compute_pre_entry_stats()` – Pre-entry statistics (volume, n_gxs trends)
- `handle_missing_values()` – Strategy-based imputation

### `src/features.py` – Scenario-Aware Feature Engineering

```python
from src.features import make_features, get_feature_columns

# Build features for scenario 1 (strict cutoff at month 0)
panel_s1 = make_features(panel, scenario=1, mode='train', config=features_config)

# Get feature columns (excludes META_COLS)
feature_cols = get_feature_columns(panel_s1, META_COLS)
```

**Feature Categories:**

| Category | Description | Examples |
|----------|-------------|----------|
| Pre-entry | Statistics from months < 0 | `vol_mean_pre`, `vol_std_pre`, `vol_trend_pre` |
| Time | Temporal encodings | `months_postgx`, `month_sin`, `month_cos` |
| Generics | Generic competition | `n_gxs`, `n_gxs_change`, `n_gxs_cumsum` |
| Drug | Static attributes | `ther_area`, `hospital_rate`, `biological` |
| Early erosion | S2 only (months 0-5) | `vol_mean_early`, `erosion_rate_early` |
| Interaction | Combined features | `hospital_x_bio`, `vol_pre_x_ngxs` |

### `src/train.py` – Training Pipeline

```python
from src.train import run_experiment, train_scenario_model

# Full experiment
results = run_experiment(
    scenario=1,
    model_type='catboost',
    config=run_config,
    model_config=model_config
)

# Train single model
model, metrics = train_scenario_model(
    panel, scenario=1, model_cls=CatBoostModel, config=model_config
)
```

**Critical Constants:**
```python
META_COLS = ['country', 'brand_name', 'months_postgx', 'bucket', 
             'avg_vol_12m', 'y_norm', 'volume', 'mean_erosion', 'month']
# NEVER use these as features to prevent leakage
```

### `src/inference.py` – Prediction & Submission

```python
from src.inference import generate_submission, detect_test_scenarios

# Detect which scenario each test series belongs to
scenarios = detect_test_scenarios(test_panel)

# Generate submission file
submission = generate_submission(
    model_s1, model_s2, test_panel, scenarios, config
)
# Inverse transform: volume = y_norm * avg_vol_12m
```

### `src/evaluate.py` – Metric Computation

```python
from src.evaluate import compute_metric1, compute_metric2, compute_bucket_metrics

# Official metric wrappers (require df_actual, df_pred, df_aux)
# df_actual/df_pred: columns [country, brand_name, months_postgx, volume]
# df_aux: columns [country, brand_name, avg_vol, bucket]
metric1 = compute_metric1(df_actual, df_pred, df_aux)
metric2 = compute_metric2(df_actual, df_pred, df_aux)

# Bucket-wise analysis (uses integer scenario)
bucket_metrics = compute_bucket_metrics(df_actual, df_pred, df_aux, scenario=1)
# Returns: {'overall': ..., 'bucket1': ..., 'bucket2': ...}
```

### `src/validation.py` – Series-Level Validation

```python
from src.validation import create_validation_split, simulate_scenario

# Split at series level (critical for time-series)
train_df, val_df = create_validation_split(
    panel, val_fraction=0.2, stratify_by='bucket', random_state=42
)
# Guarantees: no series appears in both train and val

# Simulate scenario for validation (uses integer scenario)
features_df, targets_df = simulate_scenario(val_df, scenario=2)
```

### `src/utils.py` – Utilities

```python
from src.utils import set_seed, load_config, get_project_root, setup_logging

set_seed(42)  # Reproducibility across numpy, torch, random
config = load_config('configs/data.yaml')
root = get_project_root()
logger = setup_logging('experiment')
```

---

## 6. Feature Engineering

### Scenario Feature Cutoffs

| Scenario | Feature Cutoff | Rationale |
|----------|---------------|-----------|
| S1 | `months_postgx < 0` | Only pre-entry data available |
| S2 | `months_postgx < 6` | Pre-entry + first 6 months available |

### Feature Configuration (`configs/features.yaml`)

```yaml
pre_entry:
  vol_mean_pre: true
  vol_std_pre: true
  vol_trend_pre: true
  vol_median_pre: true
  n_gxs_at_entry: true

time_features:
  months_postgx: true
  month_sin: true
  month_cos: true
  quarter: true

generics_features:
  n_gxs: true
  n_gxs_change: true
  n_gxs_cumsum: true

drug_features:
  ther_area: true
  hospital_rate: true
  main_package: true
  biological: true
  small_molecule: true
```

---

## 7. Models

### Implemented Models

| Model | Class | Config File | Notes |
|-------|-------|-------------|-------|
| **CatBoost** | `CatBoostModel` | `model_cat.yaml` | Hero model, native categorical support |
| **LightGBM** | `LGBMModel` | `model_lgbm.yaml` | Fast training, good baseline |
| **XGBoost** | `XGBModel` | `model_xgb.yaml` | Alternative GBM |
| **Linear** | `LinearModel` | `model_linear.yaml` | Ridge/Lasso/ElasticNet |
| **Baselines** | `GlobalMeanBaseline`, `FlatBaseline`, `TrendBaseline` | - | Reference baselines |

### Model Interface (`src/models/base.py`)

All models implement:
```python
class BaseModel(ABC):
    def fit(self, X_train, y_train, X_val, y_val, sample_weight) -> dict
    def predict(self, X) -> np.ndarray
    def save(self, path: Path)
    def load(self, path: Path)
    def get_feature_importance(self) -> pd.DataFrame
```

### CatBoost Default Configuration

```yaml
# configs/model_cat.yaml
model:
  iterations: 1000
  learning_rate: 0.05
  depth: 6
  l2_leaf_reg: 3.0
  early_stopping_rounds: 50
  task_type: CPU
  random_seed: 42
```

---

## 8. Quick Start

### Option A: Colab (Recommended)

Click the badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb)

The notebook will:
1. Install dependencies
2. Mount Google Drive (optional)
3. Load and prepare data
4. Train models for both scenarios
5. Generate submission file

### Option B: Local Development

```bash
# 1. Clone repository
git clone https://github.com/armanfeili/novartis_datathon_2025.git
cd novartis_datathon_2025

# 2. Create environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add competition data
# Copy CSVs to data/raw/TRAIN/ and data/raw/TEST/

# 5. Run tests
pytest tests/test_smoke.py -v

# 6. Train models
python -m src.train --scenario 1 --model catboost
python -m src.train --scenario 2 --model catboost
```

---

## 9. CLI Usage

### Training

```bash
# Train CatBoost for Scenario 1
python -m src.train \
    --scenario 1 \
    --model catboost \
    --data-config configs/data.yaml \
    --features-config configs/features.yaml \
    --run-config configs/run_defaults.yaml \
    --model-config configs/model_cat.yaml

# Train LightGBM for Scenario 2
python -m src.train \
    --scenario 2 \
    --model lightgbm \
    --model-config configs/model_lgbm.yaml
```

### Generating Submissions

```bash
# Generate submission from trained models
python -m src.inference \
    --model-s1 artifacts/scenario1_model.pkl \
    --model-s2 artifacts/scenario2_model.pkl \
    --output submissions/submission.csv
```

---

## 10. Configuration Files

### `configs/data.yaml`
- Raw data paths
- Column definitions and schema
- Missing value handling strategies
- Google Drive paths for Colab

### `configs/features.yaml`
- Feature groups to enable/disable
- Pre-entry feature configuration
- Scenario-specific settings

### `configs/run_defaults.yaml`
- Random seed (42)
- Validation settings (20% series holdout)
- Scenario definitions
- Sample weight configuration
- Logging and output settings

### Model configs (`model_*.yaml`)
- Hyperparameters
- Early stopping
- Regularization

---

## 11. Testing

Run the smoke test suite:

```bash
pytest tests/test_smoke.py -v
```

**Test Coverage:**
- ✅ Module imports
- ✅ Configuration loading
- ✅ Data loading (if data present)
- ✅ Panel construction
- ✅ Feature leakage prevention
- ✅ Model interface
- ✅ Metric computation
- ✅ Validation split (series-level)
- ✅ Submission format
- ✅ Sample weight computation

---

## 12. Important Technical Details

### Target Normalization

Models are trained on **normalized volume**:
```python
y_norm = volume / avg_vol_12m
```

At inference, predictions are inverse-transformed:
```python
predicted_volume = predicted_y_norm * avg_vol_12m
```

### Leakage Prevention

**META_COLS** are strictly excluded from features:
```python
META_COLS = ['country', 'brand_name', 'months_postgx', 'bucket', 
             'avg_vol_12m', 'y_norm', 'volume', 'mean_erosion', 'month']
```

**Feature cutoffs** ensure no future information:
- S1: Only use data from `months_postgx < 0`
- S2: Only use data from `months_postgx < 6`

### Sample Weights

Training loss is aligned with the official metric:
- **Time weights**: Early months weighted higher (3.0 for months 0-5)
- **Bucket weights**: Bucket 1 = 2.0×, Bucket 2 = 1.0×

### Validation Strategy

**Series-level splits** ensure:
- All months of a series stay together
- Stratification by bucket for balanced evaluation
- No data leakage from temporal dependencies

---

## 13. License & Creator

**Created by [Arman Feili](https://github.com/armanfeili)** for the **Novartis Generic Erosion Datathon 2025**.

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

### Dependencies

Core libraries:
- `pandas`, `numpy` – Data manipulation
- `catboost`, `lightgbm`, `xgboost` – Gradient boosting
- `scikit-learn` – ML utilities
- `torch` – Neural networks (optional)
- `pyyaml`, `omegaconf` – Configuration
- `matplotlib`, `seaborn`, `plotly` – Visualization

See `requirements.txt` for the full list.
