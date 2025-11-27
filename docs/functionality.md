# Novartis Generic Erosion Datathon 2025 – Project Functionality Documentation

> **Last Updated:** November 2025
> **Author:** Arman Feili
> **Repository (example):** `novartis_generic_erosion_2025`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Core Modules](#core-modules)

   * [Utilities & Config](#1-utilities--config-srcutilspy-srccfgpy)
   * [Data Loading & Panel Construction](#2-data-loading--panel-construction-srcdatapy)
   * [Feature Engineering](#3-feature-engineering-srcfeaturespy)
   * [Official Metrics Wrapper](#4-official-metrics-wrapper-srcmetricspy)
   * [Validation & Splitting](#5-validation--splitting-srcvalidationpy)
   * [Scenario-Specific Training Pipelines](#6-scenario-specific-training-pipelines-srctrain_scenario1py-srctrain_scenario2py)
   * [Inference & Submission Generation](#7-inference--submission-generation-srcinferencepy-srcsubmissionpy)
5. [Model Implementations](#model-implementations)
6. [Configuration System](#configuration-system)
7. [Main Notebook Workflow](#main-notebook-workflow)
8. [End-to-End Workflow](#end-to-end-workflow)
9. [Usage Guide](#usage-guide)

---

## Overview

This project is a **competition-ready forecasting pipeline** for the **Novartis Datathon 2025** generic erosion challenge.

### Business Objective

> **Forecast monthly sales volume after generic entry, for each (country, brand), under two scenarios, with special focus on high-erosion brands (Bucket 1), to support revenue forecasting, product planning, and strategic decisions in the post-patent period.**

### Technical Objective

Build a **reproducible, configuration-driven** system that:

* Ingests the **three official datasets** for train/test:

  * `df_volume_*` (monthly volume with months_postgx),
  * `df_generics_*` (number of generics over time),
  * `df_medicine_info_*` (static drug characteristics).
* Constructs unified **panel time-series datasets** keyed by `(country, brand_name, months_postgx)`.
* Trains models for:

  * **Scenario 1** – Forecast 0–23 months after generic entry with **no post-entry actuals**.
  * **Scenario 2** – Forecast 6–23 months after generic entry with **first 6 months of post-entry actuals**.
* Evaluates locally using the **official competition metrics**:

  * Metric 1 (Phase 1A – 0 actuals),
  * Metric 2 (Phase 1B – 6 actuals),
    wrapped around `metric_calculation.py` and `auxiliar_metric_computation.csv`.
* Generates **submission files** in the exact format required by `submission_template.csv`.

---

## Architecture

The architecture is optimized for:

* **Local laptop / Colab** usage (CPU is sufficient; GPU optional but not required).
* Clear separation between:

  * **Raw competition data**,
  * **Processing logic**,
  * **Scenario-specific training and inference**,
  * **Official metric computation**.

High-level flow:

```text
Raw CSVs
(df_volume, df_generics, df_medicine_info, auxiliar_metric_computation)
   │
   ▼
Data Loading & Merge (Panel Construction)
   │
   ▼
Scenario-Specific Datasets
(Scenario 1 panels, Scenario 2 panels)
   │
   ▼
Feature Engineering
(time-series lags, erosion-related features, drug characteristics)
   │
   ▼
Time-Based Validation
   │
   ▼
Model Training & Hyperparameter Tuning
   │
   ▼
Local Evaluation
(Metric 1, Metric 2 with bucket weighting)
   │
   ▼
Inference on Test
   │
   ▼
Submission Files
(submission_scenario1.csv, submission_scenario2.csv)
```

**Principle:** Everything is driven by **config**, and the pipeline can be run end-to-end with a **few clear commands**.

---

## Project Structure

```text
novartis_generic_erosion_2025/
│
├── configs/
│   ├── data.yaml                # Paths and column definitions for official CSVs
│   ├── features.yaml            # Feature engineering settings
│   ├── run_scenario1.yaml       # Scenario 1 run config
│   ├── run_scenario2.yaml       # Scenario 2 run config
│   ├── model_gbm.yaml           # Main GBM hyperparameters
│   └── model_baseline.yaml      # Baseline settings (naive models)
│
├── src/
│   ├── __init__.py
│   ├── cfg.py                   # Simple config loader / resolver
│   ├── utils.py                 # General utilities (seeding, logging, timing)
│   ├── data.py                  # Data loading & panel construction
│   ├── features.py              # Feature engineering for scenarios
│   ├── metrics.py               # Wrapper around metric_calculation.py
│   ├── validation.py            # Validation splits for Scenario 1 & 2
│   ├── train_scenario1.py       # Training pipeline for Scenario 1
│   ├── train_scenario2.py       # Training pipeline for Scenario 2
│   ├── inference.py             # Generic inference helpers
│   ├── submission.py            # Fills submission_template.csv
│   └── models/
│       ├── base.py              # Base model interface
│       ├── gbm.py               # LightGBM/XGBoost/CatBoost wrapper
│       └── baseline.py          # Naive baselines (flat, simple trend)
│
├── notebooks/
│   ├── 00_eda_erosion.ipynb     # EDA on erosion, buckets, patterns
│   ├── 01_scenario1_experiments.ipynb
│   ├── 02_scenario2_experiments.ipynb
│   └── colab_main.ipynb         # End-to-end Colab workflow (optional)
│
├── data/
│   ├── raw/
│   │   ├── df_volume_train.csv
│   │   ├── df_generics_train.csv
│   │   ├── df_medicine_info_train.csv
│   │   ├── df_volume_test.csv
│   │   ├── df_generics_test.csv
│   │   ├── df_medicine_info_test.csv
│   │   ├── auxiliar_metric_computation.csv
│   │   ├── submission_template.csv
│   │   └── submission_example.csv
│   ├── interim/                 # Merged, cleaned panels
│   └── processed/               # Feature matrices per scenario
│
├── artifacts/
│   ├── scenario1_runs/          # Metrics, models, logs for Scenario 1
│   └── scenario2_runs/          # Metrics, models, logs for Scenario 2
│
├── submissions/                 # Final CSVs for upload
├── metric_calculation.py        # Official metric implementation (provided)
├── requirements.txt
└── README.md
```

---

## Core Modules

### 1. Utilities & Config (`src/utils.py`, `src/cfg.py`)

#### `src/utils.py`

Core helpers used everywhere:

* `set_seed(seed: int = 42)`:
  Sets seed for Python and NumPy for reproducibility.
* `setup_logging(level: str = "INFO")`:
  Configures a simple logger for console and optionally a file.
* `timer(name: str)`:
  Context manager to time steps (e.g. data loading, feature build).

Example:

```python
from src.utils import set_seed, setup_logging, timer

set_seed(42)
setup_logging()

with timer("Build Scenario 1 features"):
    df_features = build_scenario1_features(...)
```

#### `src/cfg.py`

Minimal config loader:

* `load_config(path: str) -> dict`: YAML → dict.
* `get_path(config: dict, key: str) -> Path`: resolves paths like `data.raw_dir`.

This keeps everything driven by `configs/*.yaml`.

---

### 2. Data Loading & Panel Construction (`src/data.py`)

This module is responsible for:

1. **Loading the three train CSVs**:

   * `df_volume_train.csv`
   * `df_generics_train.csv`
   * `df_medicine_info_train.csv`
2. **Loading the three test CSVs**:

   * `df_volume_test.csv`
   * `df_generics_test.csv`
   * `df_medicine_info_test.csv`
3. Building a **panel dataset** with one row per:

   * `(country, brand_name, months_postgx)`
     and associated:
   * `volume`, `number_of_gx`, and static drug info.

#### Main functions

```python
def load_train_raw(config: dict) -> dict[str, pd.DataFrame]:
    """Loads raw train CSVs into DataFrames."""

def load_test_raw(config: dict) -> dict[str, pd.DataFrame]:
    """Loads raw test CSVs into DataFrames."""

def build_panel(
    df_volume: pd.DataFrame,
    df_generics: pd.DataFrame,
    df_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns a unified panel with columns:
    [country, brand_name, month, months_postgx,
     volume, n_gxs, ther_area, hospital_rate, main_package,
     biological, small_molecule]
    """
```

Key behavior:

* Merge on `(country, brand_name, months_postgx)` between volume and generics.
* Merge static `df_medicine_info` on `(country, brand_name)`.
* Ensure **no duplicate** `(country, brand_name, months_postgx)` rows.
* Keep both `month` (calendar) and `months_postgx` (relative) fields.

---

### 3. Feature Engineering (`src/features.py`)

This module builds scenario-specific feature matrices.

#### Core ideas

* Common base features:

  * raw `volume` (for training),
  * `months_postgx`,
  * `n_gxs`,
  * `ther_area`, `main_package`, `hospital_rate`,
  * `biological`, `small_molecule`.
* Time-series features:

  * pre-entry **lags** of `volume`,
  * rolling statistics,
  * slopes near entry (trend in last N months),
  * normalized volume / pre-entry average volume.

#### Main interfaces

```python
def make_features_scenario1(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build features for Scenario 1:
    - Only pre-entry volumes used for prediction (months_postgx < 0)
    - Target: volume for months_postgx in [0, 23]
    - Creates one row per (country, brand_name, forecast_month)
    """

def make_features_scenario2(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build features for Scenario 2:
    - Uses pre-entry + months 0–5 actuals as inputs
    - Target: volume for months_postgx in [6, 23]
    """
```

Details driven by `configs/features.yaml`:

* Which lags to compute (e.g. last 3, 6, 12 months pre-entry).
* Whether to include normalized volume (volume / pre-entry mean).
* How to treat categorical variables (encoded later by model or pre-encoded).

---

### 4. Official Metrics Wrapper (`src/metrics.py`)

This module wraps `metric_calculation.py` to:

* Use **exact competition metrics** for local validation,
* Handle:

  * `auxiliar_metric_computation.csv`
    (columns: `country, brand_name, avg_vol, bucket`),
  * The **two phases**:

    * Metric 1 / Scenario 1,
    * Metric 2 / Scenario 2.

#### Public functions

```python
from typing import Tuple

def compute_phase1_metrics(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> Tuple[float, float]:
    """
    Returns (metric1, metric2) for convenience, if both are available,
    or one of them depending on scenario.
    """

def compute_metric1_scenario1(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """Wrapper around compute_metric1 in metric_calculation.py"""

def compute_metric2_scenario2(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """Wrapper around compute_metric2 in metric_calculation.py"""
```

Internal behavior:

* Ensure both `df_actual` and `df_pred` have columns:

  * `country, brand_name, months_postgx, volume`.
* Rename `volume` appropriately to match expectations of `metric_calculation.py`.
* Filter to relevant `start_month` (`0` for Scenario 1, `6` for Scenario 2).

This keeps local validation **exactly aligned** with the leaderboard.

---

### 5. Validation & Splitting (`src/validation.py`)

Because this is **panel time-series**, validation must be **time-aware** and **scenario-aware**.

#### Scenario 1 validation

* Concept: simulate forecasting from LoE date (months_postgx = 0) with no post-entry info.
* Split by `(country, brand_name)` and time:

  * For each series, you can:

    * Train on pre-entry,
    * Validate on a subset of post-entry months (e.g. 0–23, or 0–11).

#### Scenario 2 validation

* Concept: simulate forecasting at 6 months after entry.
* For each series:

  * Train on pre-entry + months 0–5,
  * Validate on months 6–23.

#### Main interface

```python
def get_scenario1_train_val(panel: pd.DataFrame, config: dict):
    """
    Returns (train_df, val_df) for Scenario 1,
    with train_df containing rows whose target belongs to training period,
    val_df for validation period.
    """

def get_scenario2_train_val(panel: pd.DataFrame, config: dict):
    """
    Same idea for Scenario 2: train on pre-entry+0–5, validate on 6–23.
    """
```

Optional: extended version returning **multiple folds** if you want multi-split CV.

---

### 6. Scenario-Specific Training Pipelines (`src/train_scenario1.py`, `src/train_scenario2.py`)

These modules orchestrate the **full training pipeline** for each scenario.

#### `src/train_scenario1.py`

```python
def train_scenario1(run_config_path: str, model_config_path: str) -> dict:
    """
    High-level pipeline:
    1. Load configs
    2. Load train raw data
    3. Build panel
    4. Build Scenario 1 features
    5. Split into train/validation
    6. Train models (baseline + GBM)
    7. Evaluate using compute_metric1_scenario1
    8. Save models, metrics, and predictions to artifacts/scenario1_runs/
    9. Return summary metrics
    """
```

#### `src/train_scenario2.py`

```python
def train_scenario2(run_config_path: str, model_config_path: str) -> dict:
    """
    Similar steps but for Scenario 2 and Metric 2.
    """
```

Each training pipeline:

* Uses `src/models/` abstractions to instantiate models.
* Can support **fold-based training** if desired:

  * For each fold, train a GBM and average predictions.
* Saves in `artifacts/scenarioX_runs/<run_id>/`:

  * `config_used.yaml`,
  * `metrics.json` (including Metric1/Metric2 and maybe MAE/MAE Bucket1),
  * `oof_preds.csv`,
  * model files (e.g. `model_fold_0.pkl`),
  * `logs.txt`.

---

### 7. Inference & Submission Generation (`src/inference.py`, `src/submission.py`)

#### `src/inference.py`

Reusable functions to:

* Load models for a given `run_id`,
* Apply **the same feature engineering** to the test panel,
* Generate predictions for all test `(country, brand_name, months_postgx)` required.

Example:

```python
def predict_scenario1(run_id: str, config: dict) -> pd.DataFrame:
    """
    Returns DataFrame with:
    [country, brand_name, months_postgx, volume_pred]
    for Scenario 1 test subset.
    """
```

#### `src/submission.py`

This module ties predictions to the official `submission_template.csv`.

```python
def build_submission(
    template_path: str,
    preds: pd.DataFrame,
    save_path: str
) -> None:
    """
    - Loads submission_template.csv
    - Merges preds on (country, brand_name, months_postgx)
    - Writes filled CSV to `save_path`
    """
```

You will likely have **two submissions**:

* `submission_scenario1.csv`
* `submission_scenario2.csv`

depending on how the platform expects them (single or separate — you can adapt).

---

## Model Implementations

### Base Model (`src/models/base.py`)

Minimal interface tailored for **regression on tabular features**:

```python
class BaseModel(ABC):
    def __init__(self, config: dict): ...
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None): ...
    @abstractmethod
    def predict(self, X): ...
    @abstractmethod
    def save(self, path: str): ...
    @abstractmethod
    def load(self, path: str): ...
```

### GBM Model (`src/models/gbm.py`)

Single wrapper that can use LightGBM / XGBoost / CatBoost depending on config. For Datathon:

* Start with **one main GBM library** (e.g. LightGBM or XGBoost).
* Include essential hyperparameters in `configs/model_gbm.yaml`.

### Baseline Models (`src/models/baseline.py`)

Collection of simple baselines:

* Scenario 1:

  * Flat: use **pre-entry average volume** as prediction for all months.
  * Generic “average erosion curve” baseline (optional).
* Scenario 2:

  * Trend extrapolation from months 0–5 (e.g. simple linear trend).

These baselines are used to:

* Establish a **reference Metric1/Metric2**,
* Show improvement of the GBM model in your slides.

---

## Configuration System

You keep the config-driven philosophy, but **specialized for this task**.

### `configs/data.yaml`

Defines paths and file names for the eight key CSVs:

```yaml
paths:
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"

files:
  volume_train: "df_volume_train.csv"
  generics_train: "df_generics_train.csv"
  info_train: "df_medicine_info_train.csv"

  volume_test: "df_volume_test.csv"
  generics_test: "df_generics_test.csv"
  info_test: "df_medicine_info_test.csv"

  aux_metric: "auxiliar_metric_computation.csv"
  submission_template: "submission_template.csv"

columns:
  id_keys: ["country", "brand_name"]
  time_keys:
    month: "month"
    rel_time: "months_postgx"
  target: "volume"
```

### `configs/features.yaml`

Scenario-specific feature toggles:

```yaml
scenario1:
  use_lags_pre_entry: true
  lag_months: [1, 3, 6, 12]
  use_rolling_stats: true
  rolling_windows: [3, 6]
  normalize_by_pre_avg: true
  include_calendar_month: true

scenario2:
  use_post_entry_lags: true
  lag_months: [1, 2, 3, 4, 5]
  use_trend_features: true
  trend_window: 6
  normalize_by_pre_avg: true

categorical:
  encode_country: false         # allow model to handle if CatBoost, etc.
  encode_brand_name: false      # mostly for grouping, not as feature
  encode_ther_area: true
  encode_main_package: true
```

### `configs/run_scenario1.yaml` / `run_scenario2.yaml`

Run settings:

```yaml
reproducibility:
  seed: 42

validation:
  # For now, one train/val split based on months_postgx ranges
  scenario1:
    val_months: [0, 23]        # which months to treat as validation in local split
  scenario2:
    val_months: [6, 23]

metrics:
  use_official: true
  aux_metric_file: "auxiliar_metric_computation.csv"

artifacts:
  base_dir: "artifacts/scenario1_runs"
```

---

## Main Notebook Workflow

`notebooks/colab_main.ipynb` (optional but helpful) orchestrates **end-to-end**:

1. **Environment setup**

   * Install dependencies, clone repo (if needed), set paths.
2. **Config & seed**

   * Load `data.yaml`, `run_scenario1.yaml`, `run_scenario2.yaml`.
3. **EDA**

   * Use `build_panel` to inspect:

     * Erosion curves,
     * Bucket distributions,
     * Volume vs `n_gxs`, etc.
4. **Train Scenario 1**

   * Call `train_scenario1(...)`,
   * Inspect local Metric1.
5. **Train Scenario 2**

   * Call `train_scenario2(...)`,
   * Inspect local Metric2.
6. **Generate submissions**

   * Load best run IDs,
   * Use `predict_scenarioX` + `build_submission(...)`.
7. **Export CSVs**

   * Save to `/content/drive/.../submissions/` (if using Drive).

---

## End-to-End Workflow

### High-Level Steps

```text
1. Place official CSVs under data/raw/
2. Configure paths and columns in configs/data.yaml
3. Run training for Scenario 1 (and optionally Scenario 2) locally or in Colab
4. Evaluate locally with official metrics via src/metrics.py
5. Generate final submission CSVs via src/submission.py
6. Upload to competition platform (max 3 submissions per 8 hours)
```

---

## Usage Guide

### Local (Laptop) Workflow

```bash
# 1. Clone repo
git clone https://github.com/armanfeili/novartis_generic_erosion_2025.git
cd novartis_generic_erosion_2025

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure raw data files are in data/raw/

# 4. Train Scenario 1
python -m src.train_scenario1 \
    --run-config configs/run_scenario1.yaml \
    --model-config configs/model_gbm.yaml

# 5. Train Scenario 2
python -m src.train_scenario2 \
    --run-config configs/run_scenario2.yaml \
    --model-config configs/model_gbm.yaml

# 6. Generate submissions (example CLI wrapper)
python -m src.submission \
    --scenario 1 \
    --run-id best_scenario1_run \
    --output submissions/submission_s1.csv
```

### Colab Workflow (if you keep Drive-based pattern)

1. Mount Drive, set project root.
2. Clone repo into `/content/`.
3. Symlink or set `data.raw_dir` to Drive folder.
4. Run training notebooks or call Python modules directly.