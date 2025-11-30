# Novartis Datathon 2025 – Generic Erosion Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-372%20passed-green.svg)](#testing)

Forecast **post–Loss of Exclusivity (LOE)** volume erosion of branded pharmaceutical products after the entry of generic competitors, across multiple countries and scenarios.

> **🚀 Quick Links**
> - **Colab (Recommended):** [Open in Colab](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb)  
> - **Local Jupyter:** `jupyter notebook notebooks/colab/main.ipynb` → [http://localhost:8888](http://localhost:8888)

---

## Table of Contents

1. [Overview & Competition Setup](#overview--competition-setup)  
2. [Repository Highlights](#repository-highlights)  
3. [Getting Started](#getting-started)  
   - [Installation](#installation)  
   - [Quick Start (Local)](#quick-start-local)  
   - [Google Colab](#google-colab)  
4. [Project Structure](#project-structure)  
5. [Data Overview](#data-overview)  
6. [Configuration](#configuration)  
7. [Training & Inference (CLI)](#training--inference-cli)  
8. [Feature Engineering](#feature-engineering)  
9. [Models](#models)  
10. [Validation & Metrics](#validation--metrics)  
    - [Validation Strategy](#validation-strategy)  
    - [Metric Calculation](#metric-calculation)  
11. [Hyperparameter Optimization](#hyperparameter-optimization)  
12. [Notebooks](#notebooks)  
13. [Testing](#testing)  
14. [Reproducibility](#reproducibility)  
15. [Troubleshooting](#troubleshooting)  
16. [License & Acknowledgments](#license--acknowledgments)

---

## Overview & Competition Setup

### Problem Statement

When a drug loses patent protection (**Loss of Exclusivity / LOE**), generic manufacturers enter the market and the branded product typically experiences **volume erosion**.  

The goal is to **forecast 24-month post-LOE volume trajectories** for each `(country, brand_name)` time series, under two visibility scenarios.

### Scenarios

| Scenario | Available History | Forecast Horizon | Metric |
|----------|-------------------|------------------|--------|
| **Scenario 1** | Pre-LOE only (months `< 0`) | Months `0–23` | Metric 1 |
| **Scenario 2** | Pre-LOE + first 6 months post-LOE (`0–5`) | Months `6–23` | Metric 2 |

### Target & Transformation

- **Normalized volume** `y_norm` (training target):  
  `y_norm[t] = volume[t] / avg_vol_12m`
- `avg_vol_12m`: 12-month average pre-LOE volume
- Submission requires **absolute volume**:  
  `volume_pred[t] = y_norm_pred[t] × avg_vol_12m`

### Bucket Classification (Metric Weighting)

Series are grouped by **mean erosion**:

- **Bucket 1 (high erosion)**: `mean_erosion ≤ 0.25` → **weight 2×**
- **Bucket 2 (low erosion)**: `mean_erosion > 0.25` → **weight 1×**

These bucket weights are built into the official metric and our evaluation pipeline.

---

## Repository Highlights

- **End-to-end pipeline** from raw CSVs → panel construction → features → models → validated submissions.
- **Scenario-aware feature pipeline** shared across S1 & S2.
- **Hero model:** CatBoost with scenario-specific sample weights and early stopping.
- Multiple **baseline and advanced models** (Linear, NN, LSTM, CNN-LSTM, Hybrid, ARIHOW, KG-GCN-LSTM).
- **Official metric implementation** (aligned with competition script).
- **Reproducible experiments** via YAML configs, seeds, and config snapshots.
- **Tested codebase** with a dedicated `tests/` suite.

---

## Getting Started

### Installation

#### Prerequisites

- Python **3.10+**
- `pip` or `conda`
- At least **8 GB RAM** recommended
- GPU optional (used for neural network variants)

#### Option 1 – pip (Recommended)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Sanity check
python -c "import pandas, catboost; print('OK')"
````

#### Option 2 – Conda

```bash
conda env create -f environment.yml
conda activate novartis
```

### Quick Start (Local)

```bash
# 0. Clone the repo
git clone <repository-url>
cd novartis_datathon_2025

# 1. (Optional but recommended) run tests
pytest tests/ -v

# 2. Train CatBoost for Scenario 1
python -m src.train --scenario 1 --model catboost --output artifacts/catboost_s1

# 3. Train CatBoost for Scenario 2
python -m src.train --scenario 2 --model catboost --output artifacts/catboost_s2

# 4. Generate submission
python -m src.inference \
  --model-s1 artifacts/catboost_s1/model.bin \
  --model-s2 artifacts/catboost_s2/model.bin \
  --output submissions/submission.csv \
  --save-auxiliary
```

### Google Colab

1. Open the Colab notebook directly:
   [**Open in Colab**](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb)
2. Follow the in-notebook steps to:

   * Mount Google Drive
   * Clone the repository
   * Install dependencies
   * Run the full training and submission pipeline.

---

## Project Structure

```text
novartis_datathon_2025/
├── configs/                    # YAML configs (data, features, runs, models)
│   ├── README.md
│   ├── data.yaml
│   ├── features.yaml
│   ├── run_defaults.yaml
│   ├── run_bonus_all.yaml
│   ├── model_cat.yaml
│   ├── model_lgbm.yaml
│   ├── model_xgb.yaml
│   ├── model_linear.yaml
│   ├── model_nn.yaml
│   ├── model_lstm.yaml
│   ├── model_cnn_lstm.yaml
│   ├── model_hybrid.yaml
│   ├── model_arihow.yaml
│   └── model_kg_gcn_lstm.yaml
│
├── data/
│   ├── raw/                    # Competition TRAIN/TEST data (not in git)
│   ├── interim/                # Intermediate processed data
│   ├── processed/              # Final feature/target/meta parquet files
│   └── external/               # Optional external sources
│
├── src/
│   ├── data.py                 # Load raw CSVs, build panel
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # Training CLI and orchestration
│   ├── inference.py            # Inference & submission generation
│   ├── evaluate.py             # Metric wrappers
│   ├── validation.py           # Series-level CV & splits
│   ├── utils.py                # Logging, config loading, misc utilities
│   ├── config_sweep.py         # Config-based hyperparameter sweeps
│   ├── external_data.py        # External data integration
│   ├── graph_utils.py          # Graph-based utilities
│   ├── scenario_analysis.py    # Scenario-specific analysis tools
│   ├── sequence_builder.py     # Sequence feature construction
│   ├── visibility_sources.py   # Visibility source handling
│   └── models/                 # Model implementations
│       ├── base.py
│       ├── baselines.py
│       ├── cat_model.py
│       ├── lgbm_model.py
│       ├── xgb_model.py
│       ├── linear.py
│       ├── nn.py
│       ├── ensemble.py
│       ├── hybrid_physics_ml.py
│       ├── arihow.py
│       ├── cnn_lstm.py
│       ├── kg_gcn_lstm.py
│       └── gcn_layers.py
│
├── scripts/                    # Helper/automation scripts
│   ├── reproduce.sh
│   ├── train_and_submit_bonus.sh
│   ├── train_and_submit_complete.py
│   ├── train_catboost_bonus_complete.py
│   ├── run_full_training_bonus.py
│   └── generate_submission_from_trained.py
│
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_feature_prototype.ipynb
│   ├── 01_train.ipynb
│   ├── 02_model_sanity.ipynb
│   └── colab/
│       ├── main.ipynb
│       └── colab_requirements.txt
│
├── docs/
│   ├── CODEBASE_DOCUMENTATION.md
│   ├── MODEL_COMPARISON.md
│   ├── guide/
│   │   ├── metric_calculation.py
│   │   ├── submission_template.csv
│   │   ├── submission_example.csv
│   │   ├── auxiliar_metric_computation_example.csv
│   │   ├── Datathon_documentation.pdf
│   │   └── Datathon_presentation.pdf
│   ├── planning/
│   │   ├── TODO.md
│   │   ├── TODO_2.md
│   │   ├── TODO_BONUS.md
│   │   ├── approach.md
│   │   ├── functionality.md
│   │   └── NOVARTIS_DATATHON_2025_COMPLETE_GUIDE_FINAL.md
│   └── results/
│       ├── model_comparison_metrics.csv
│       └── model_comparison_metrics.md
│
├── tests/
│   ├── conftest.py
│   ├── test_smoke.py
│   ├── test_config_sweep.py
│   ├── test_model_fixes.py
│   └── test_research_modules.py
│
├── submissions/                # Generated submissions
├── artifacts/                  # Trained models, configs, metrics
├── logs/                       # Training logs
│
├── requirements.txt
├── environment.yml
├── pytest.ini
├── .gitignore
├── CONTRIBUTING.md
└── LICENSE
```

---

## Data Overview

### Source Files

Located under `data/raw/TRAIN/` and `data/raw/TEST/` (not tracked in git):

| File pattern             | Description                | Key Columns                                                                  |
| ------------------------ | -------------------------- | ---------------------------------------------------------------------------- |
| `df_volume_*.csv`        | Monthly volume time series | `country`, `brand_name`, `month`, `months_postgx`, `volume`                  |
| `df_generics_*.csv`      | Generic competitor counts  | `country`, `brand_name`, `months_postgx`, `n_gxs`                            |
| `df_medicine_info_*.csv` | Product characteristics    | `ther_area`, `hospital_rate`, `main_package`, `biological`, `small_molecule` |

### Core Concepts

* **Series:** unique `(country, brand_name)` pair.
* **months_postgx:** integer month index relative to LOE (`0` = LOE, negative = pre-LOE).
* **n_gxs:** number of generic competitors at time `t`.

### Dataset Size (Approximate)

| Metric     | Train | Test S1 | Test S2 |
| ---------- | ----- | ------- | ------- |
| Series     | 1,953 | 228     | 112     |
| Total rows | ~47k  | ~5.5k   | ~2k     |

---

## Configuration

All configuration lives in `configs/`. See `configs/README.md` for full details.

### Key Files

| File                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `data.yaml`          | Data paths, schema, column names, NA handling    |
| `features.yaml`      | Feature engineering options, leakage guards      |
| `run_defaults.yaml`  | Global run settings (seeds, validation, weights) |
| `run_bonus_all.yaml` | Extended features & bonus experiments            |
| `model_*.yaml`       | Model-specific hyperparameters & HPO spaces      |

Example: change random seed:

```yaml
# configs/run_defaults.yaml
reproducibility:
  seed: 42
  deterministic: true
```

---

## Training & Inference (CLI)

### Training

Basic training:

```bash
python -m src.train --scenario 1 --model catboost
```

Advanced options:

```bash
python -m src.train \
  --scenario 1 \                         # 1 or 2
  --model catboost \                     # catboost, lightgbm, xgboost, linear, nn, ...
  --model-config configs/model_cat.yaml \
  --run-config configs/run_defaults.yaml \
  --data-config configs/data.yaml \
  --features-config configs/features.yaml \
  --output artifacts/my_run \
  --run-name "experiment_v1" \
  --val-fraction 0.2 \
  --seed 42 \
  --use-metric-weights \                # training weights aligned with metric
  --enable-checkpoints \
  --enable-profiling
```

### Cross-Validation

```bash
python -m src.train \
  --scenario 1 \
  --model catboost \
  --cv 5 \
  --stratify bucket
```

### Inference & Submission

```bash
python -m src.inference \
  --model-s1 artifacts/catboost_s1/model.bin \
  --model-s2 artifacts/catboost_s2/model.bin \
  --output submissions/submission.csv \
  --use-versioning \
  --save-auxiliary
```

### Help

```bash
python -m src.train --help
python -m src.inference --help
```

---

## Feature Engineering

### Main Feature Groups

| Category            | Examples                                                                     | Scenarios       |
| ------------------- | ---------------------------------------------------------------------------- | --------------- |
| **Pre-entry stats** | `avg_vol_3m`, `avg_vol_6m`, `avg_vol_12m`, volatility, historical trend      | Both            |
| **Time features**   | `months_postgx`, `months_postgx²`, `time_bucket`, `quarter`                  | Both            |
| **Generics**        | `n_gxs`, `has_generic`, `cummax_n_gxs`, `first_generic_month`                | Both            |
| **Drug attributes** | `ther_area`, `hospital_rate`, `biological`, `small_molecule`, `main_package` | Both            |
| **Early signal**    | `avg_vol_0_5`, `erosion_0_5`, `trend_0_5`                                    | Scenario 2 only |

### Leakage Prevention

The following are **never used as input features**:

* `bucket`
* `y_norm`, `volume`
* `mean_erosion`
* `country`, `brand_name` (kept as meta only)

They are treated as **meta columns** (`META_COLS` in `src/data.py`) used for grouping, filtering, and metric computation.

---

## Models

### Supported Models (CLI Names)

| Model Type        | CLI Name      | Config File              | Notes                               |
| ----------------- | ------------- | ------------------------ | ----------------------------------- |
| CatBoost          | `catboost`    | `model_cat.yaml`         | **Hero model**, native categoricals |
| LightGBM          | `lightgbm`    | `model_lgbm.yaml`        | Fast gradient boosting              |
| XGBoost           | `xgboost`     | `model_xgb.yaml`         | GPU-capable boosting                |
| Linear models     | `linear`      | `model_linear.yaml`      | Ridge/Lasso/ElasticNet + baselines  |
| Neural Net (MLP)  | `nn`          | `model_nn.yaml`          | PyTorch MLP                         |
| LSTM              | `lstm`        | `model_lstm.yaml`        | Sequence model                      |
| CNN-LSTM          | `cnn_lstm`    | `model_cnn_lstm.yaml`    | Conv + LSTM hybrid                  |
| Hybrid physics-ML | `hybrid`      | `model_hybrid.yaml`      | Physics baseline + ML residuals     |
| ARIHOW            | `arihow`      | `model_arihow.yaml`      | ARIMA + Holt-Winters hybrid         |
| KG-GCN-LSTM       | `kg_gcn_lstm` | `model_kg_gcn_lstm.yaml` | Knowledge graph enhanced model      |

### Baseline Models

Provided in `src/models/baselines.py`:

* `FlatBaseline` – constant 1.0 (no erosion)
* `GlobalMeanBaseline` – global average erosion curve
* `TrendBaseline` – extrapolated trend
* `HistoricalCurveBaseline` – KNN curve matching

### Common Interface

All models implement a common `BaseModel` interface:

```python
class BaseModel:
    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None): ...
    def predict(self, X): ...
    def save(self, path): ...
    def load(self, path): ...
    def get_feature_importance(self): ...
```

### Ensembles

```python
from src.models.ensemble import (
    WeightedAveragingEnsemble,
    StackingEnsemble,
    BlendingEnsemble
)

ensemble = WeightedAveragingEnsemble(
    models=[model_cat, model_lgbm],
    weights=[0.6, 0.4]
)
preds = ensemble.predict(X_test)
```

---

## Validation & Metrics

### Validation Strategy

**Key rule:** splits are done at the **series level** to avoid leakage.

```python
from src.validation import create_validation_split

train_df, val_df = create_validation_split(
    df,
    val_fraction=0.2,
    stratify_by='bucket',
    random_state=42
)
```

Cross-validation on series:

```python
from src.validation import get_fold_series

for fold_idx, (train_series, val_series) in enumerate(
    get_fold_series(df, n_folds=5, stratify_by='bucket')
):
    train_df = df[df[['country', 'brand_name']].apply(tuple, axis=1).isin(train_series)]
    val_df   = df[df[['country', 'brand_name']].apply(tuple, axis=1).isin(val_series)]
    # train/eval per fold...
```

### Metric Calculation

Official metric script: `docs/guide/metric_calculation.py`.

```python
import pandas as pd
from docs.guide.metric_calculation import compute_metric1, compute_metric2

# df_actual, df_pred, df_aux prepared as described in the docs
metric1 = compute_metric1(df_actual, df_pred, df_aux)  # Scenario 1
metric2 = compute_metric2(df_actual, df_pred, df_aux)  # Scenario 2
```

Project-level wrappers in `src/evaluate.py`:

```python
from src.evaluate import compute_metric1, compute_metric2, create_aux_file

df_aux = create_aux_file(panel_df)
m1 = compute_metric1(df_actual, df_pred, df_aux)
m2 = compute_metric2(df_actual, df_pred, df_aux)
```

#### Metric Formulas (High Level)

* **Metric 1 (Scenario 1)** and **Metric 2 (Scenario 2)** are based on a **Prediction Error (PE)** that combines:

  * **Monthly absolute errors**, normalized by `avg_vol_12m`
  * **Accumulated errors** over different windows (0–5, 6–11, 12–23)
  * **Bucket-weighted averaging** (Bucket 1 counts 2×, Bucket 2 counts 1×)

Exact formula details match the competition script and are implemented in `docs/guide/metric_calculation.py`.

---

## Hyperparameter Optimization

### CLI (Optuna-based)

```bash
python -m src.train \
  --scenario 1 \
  --model catboost \
  --hpo \
  --hpo-trials 100 \
  --hpo-timeout 3600
```

### Python API

```python
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
print(results['best_params'], results['best_value'])
```

Search spaces are declared in the relevant `model_*.yaml` under `tuning.search_space`.

---

## Notebooks

| Notebook                     | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| `00_eda.ipynb`               | Core EDA: distributions, correlations, plots    |
| `01_feature_prototype.ipynb` | Interactive feature exploration & sanity checks |
| `01_train.ipynb`             | Notebook-based training & diagnostics           |
| `02_model_sanity.ipynb`      | Residuals, feature importance, error analysis   |
| `colab/main.ipynb`           | End-to-end pipeline tailored for Google Colab   |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific tests
pytest tests/test_smoke.py -v
pytest tests/ -k "metric" -v
```

Test categories cover:

* Data loading & panel construction
* Feature engineering & leakage detection
* Model training, prediction & persistence
* Metric correctness (vs official script)
* Validation schemes
* Integration smoke tests

All tests must pass before claiming a reproducible run.

---

## Reproducibility

### Seed Control

```yaml
# configs/run_defaults.yaml
reproducibility:
  seed: 42
  deterministic: true
```

Seeds are applied consistently to:

* Python `random`
* NumPy
* PyTorch (where applicable)
* CatBoost / LightGBM / XGBoost
* scikit-learn components

### Full Reproduction Scripts

* `scripts/reproduce.sh` – raw → features → models → submission
* `scripts/train_and_submit_complete.py` – single entry point for main pipeline
* `scripts/train_and_submit_bonus.sh` – pipeline with bonus features

Manual reproduction (conceptual):

```bash
source .venv/bin/activate

# Train S1
python -m src.train --scenario 1 --model catboost --seed 42 --output artifacts/catboost_s1

# Train S2
python -m src.train --scenario 2 --model catboost --seed 42 --output artifacts/catboost_s2

# Submission
python -m src.inference \
  --model-s1 artifacts/catboost_s1/model.bin \
  --model-s2 artifacts/catboost_s2/model.bin \
  --output submissions/final_submission.csv \
  --save-auxiliary
```

Each run stores a **config snapshot** in `artifacts/<run_name>/configs/` with the exact YAMLs and a config hash.

---

## Troubleshooting

### Import / Module Issues

```bash
# Ensure you're at project root and PYTHONPATH is set
cd novartis_datathon_2025
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Memory Issues

Use batched prediction:

```python
from src.inference import predict_batch
preds = predict_batch(model, X_test, batch_size=5000)
```

### Data Not Found

Ensure the competition data is unpacked to:

```text
data/raw/TRAIN/
data/raw/TEST/
```

with the expected filenames (`df_volume_*.csv`, `df_generics_*.csv`, `df_medicine_info_*.csv`).

### CatBoost Categorical Problems

```python
df['ther_area'] = df['ther_area'].astype(str)
df['main_package'] = df['main_package'].astype(str)
```

### Where to Look Next

1. `docs/planning/TODO.md` for open issues and roadmap.
2. `logs/` and `artifacts/<run_name>/train.log` for run-specific logs.
3. `docs/CODEBASE_DOCUMENTATION.md` for a deeper dive into internals.

---

## License & Acknowledgments

This project is released under the **MIT License** – see [LICENSE](LICENSE).

**Acknowledgments:**

* **Novartis** and the **Barcelona Digital Finance Hub** for organizing the datathon and providing the dataset and evaluation framework.
* All contributors who helped design, implement, and stress-test this pipeline.
