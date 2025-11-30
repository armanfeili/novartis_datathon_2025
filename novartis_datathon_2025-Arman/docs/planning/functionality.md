# Novartis Generic Erosion Datathon 2025 – Project Functionality Documentation

> **Last Updated:** November 28, 2025
> **Author:** Arman Feili
> **Repository:** `novartis_datathon_2025`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Core Modules](#core-modules)
   * [Utilities & Config](#1-utilities--config-srcutilspy)
   * [Data Loading & Panel Construction](#2-data-loading--panel-construction-srcdatapy)
   * [Feature Engineering](#3-feature-engineering-srcfeaturespy)
   * [Official Metrics Wrapper](#4-official-metrics-wrapper-srcevaluatepy)
   * [Validation & Splitting](#5-validation--splitting-srcvalidationpy)
   * [Model Training Pipeline](#6-model-training-pipeline-srctrainpy)
   * [Inference & Submission Generation](#7-inference--submission-generation-srcinferencepy)
5. [Model Implementations](#model-implementations)
6. [Configuration System](#configuration-system)
7. [Main Notebook Workflow](#main-notebook-workflow)
8. [End-to-End Workflow](#end-to-end-workflow)
9. [Usage Guide](#usage-guide)
10. [Key Technical Decisions](#key-technical-decisions)
11. [Smoke Test Protocol](#smoke-test-protocol)

---

## Overview

This project is a **competition-ready forecasting pipeline** for the **Novartis Datathon 2025** generic erosion challenge.

### Business Objective

> **Forecast monthly sales volumes for branded drugs after generic entry over a 24-month horizon, with special focus on high-erosion brands (Bucket 1), to help Novartis anticipate revenue loss, plan post-patent strategies, and optimize product and country-level decisions in the post-LoE (loss-of-exclusivity) period.**

### Technical Framing

This is a **panel time-series forecasting problem** with:

* **1,953 training series** (country-brand combinations)
* **340 test series** (228 Scenario 1, 112 Scenario 2)
* **Two forecasting scenarios** with different information availability
* **Weighted evaluation** prioritizing Bucket 1 (high erosion) and early months

### Success Criteria

| Criterion | Target |
|-----------|--------|
| Phase 1-a (Scenario 1) | Top 10 ranking → Advance |
| Phase 1-b (Scenario 2) | Top 5 ranking → Advance to finals |
| Phase 2 (Jury) | Clear methodology + business insights → Top 3 |

### Technical Objective

Build a **reproducible, configuration-driven** system that:

* Ingests the **three official datasets** for train/test:
  * `df_volume_*` (monthly volume with `months_postgx`)
  * `df_generics_*` (number of generics over time: `n_gxs`)
  * `df_medicine_info_*` (static drug characteristics)

* Constructs unified **panel time-series datasets** keyed by `(country, brand_name, months_postgx)`

* Computes **pre-entry statistics** including:
  * `avg_vol_12m`: 12-month pre-entry average volume
  * `bucket`: Erosion classification (1 = high erosion ≤0.25, 2 = medium/low >0.25) - **training only**

* Trains models for:
  * **Scenario 1** – Forecast months 0–23 with **no post-entry actuals**
  * **Scenario 2** – Forecast months 6–23 with **first 6 months of post-entry actuals**

* Evaluates locally using the **official competition metrics** from `metric_calculation.py`:
  * Metric 1 (Phase 1A) with time-window weights: 50% months 0-5, 20% months 6-11, 10% months 12-23, 20% monthly
  * Metric 2 (Phase 1B) with time-window weights: 50% months 6-11, 30% months 12-23, 20% monthly
  * Both metrics apply **2× weight for Bucket 1** (high erosion)

* Generates **submission files** in the exact format required by `submission_template.csv`

---

## Architecture

The architecture is optimized for:

* **Local laptop / Colab** usage (CPU is sufficient; GPU optional but not required)
* Clear separation between:
  * **Raw competition data**
  * **Processing logic**
  * **Scenario-specific training and inference**
  * **Official metric computation**

### High-Level Data Flow

```text
Raw CSVs
(df_volume, df_generics, df_medicine_info)
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  DATA LOADING & PANEL CONSTRUCTION (src/data.py)        │
│  ├─ load_raw_data(split="train"|"test")                 │
│  ├─ prepare_base_panel() - join on keys                 │
│  └─ compute_pre_entry_stats() - avg_vol, bucket (train) │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING (src/features.py)                  │
│  ├─ make_features(panel, scenario)                      │
│  │   ├─ add_pre_entry_features()                        │
│  │   ├─ add_time_features()                             │
│  │   ├─ add_generics_features(cutoff)                   │
│  │   ├─ add_drug_features()                             │
│  │   └─ add_early_erosion_features() [Scenario 2 only]  │
│  └─ select_training_rows(scenario)                      │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  VALIDATION (src/validation.py)                         │
│  ├─ create_validation_split() - series-level, stratified│
│  ├─ simulate_scenario() - mimic true constraints        │
│  └─ adversarial_validation() - detect train/test shift  │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  MODEL TRAINING (src/train.py)                          │
│  ├─ split_features_target_meta() - prevent leakage      │
│  ├─ compute_sample_weights() - align with metric        │
│  └─ train_scenario_model() - with early stopping        │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  LOCAL EVALUATION (src/evaluate.py)                     │
│  ├─ compute_metric1() - Scenario 1 (official wrapper)   │
│  ├─ compute_metric2() - Scenario 2 (official wrapper)   │
│  └─ compute_bucket_metrics() - per-bucket analysis      │
└─────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  INFERENCE & SUBMISSION (src/inference.py)              │
│  ├─ detect_test_scenarios() - identify S1 vs S2 series  │
│  ├─ generate_submission() - inverse transform           │
│  ├─ apply_edge_case_fallback() - handle problematic     │
│  └─ validate_submission_format() - sanity checks        │
└─────────────────────────────────────────────────────────┘
   │
   ▼
Final Submission CSV
(country, brand_name, months_postgx, volume)
```

### Key Architectural Principles

1. **Target is Normalized Volume**: Models train on `y_norm = volume / avg_vol_12m`. At inference, multiply predictions by `avg_vol_12m` to recover actual volume.

2. **Meta Columns Never in Features**: `['country', 'brand_name', 'months_postgx', 'bucket', 'avg_vol_12m', 'y_norm']` are kept separate from model features to prevent leakage.

3. **Scenario-Aware Feature Construction**: Cutoff rules strictly enforced:
   * Scenario 1: `months_postgx < 0` only
   * Scenario 2: `months_postgx < 6` allowed

4. **Sample Weights Approximate Metric**: Training loss weighted to mirror official PE time-window and bucket weights.

---

## Project Structure

```text
novartis_datathon_2025/
│
├── configs/
│   ├── data.yaml                # Paths, column definitions, data schema
│   ├── features.yaml            # Feature engineering settings per scenario
│   ├── model_cat.yaml           # CatBoost hyperparameters
│   ├── model_lgbm.yaml          # LightGBM hyperparameters
│   ├── model_xgb.yaml           # XGBoost hyperparameters
│   ├── model_linear.yaml        # Linear model settings
│   ├── model_nn.yaml            # Neural network settings (optional)
│   └── run_defaults.yaml        # Default run settings (seed, validation)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Seeding, logging, timing utilities
│   ├── data.py                  # Data loading, panel construction, pre-entry stats
│   ├── features.py              # Scenario-aware feature engineering
│   ├── evaluate.py              # Official metric wrappers
│   ├── validation.py            # Series-level splits, adversarial validation
│   ├── train.py                 # Unified training pipeline
│   ├── inference.py             # Prediction generation & submission
│   └── models/
│       ├── base.py              # Abstract base model interface
│       ├── cat_model.py         # CatBoost wrapper
│       ├── lgbm_model.py        # LightGBM wrapper
│       ├── xgb_model.py         # XGBoost wrapper
│       ├── linear.py            # Ridge/Lasso baseline
│       └── nn.py                # Optional neural network
│
├── notebooks/
│   ├── 00_eda.ipynb             # EDA: erosion patterns, bucket analysis
│   ├── 01_feature_prototype.ipynb  # Feature engineering experiments
│   ├── 01_train.ipynb           # Training experiments and tuning
│   ├── 02_model_sanity.ipynb    # Model sanity checks
│   └── colab/
│       └── main.ipynb           # End-to-end Colab workflow
│
├── data/
│   ├── raw/
│   │   ├── TRAIN/
│   │   │   ├── df_volume_train.csv
│   │   │   ├── df_generics_train.csv
│   │   │   └── df_medicine_info_train.csv
│   │   └── TEST/
│   │       ├── df_volume_test.csv
│   │       ├── df_generics_test.csv
│   │       └── df_medicine_info_test.csv
│   ├── interim/                 # Merged panels, cleaned data
│   └── processed/               # Feature matrices ready for training
│
├── docs/
│   ├── functionality.md         # This document
│   ├── planning/
│   │   ├── approach.md          # Implementation strategy
│   │   ├── NOVARTIS_DATATHON_2025_COMPLETE_GUIDE.md
│   │   └── question-set.md      # Q&A reference
│   ├── guide/
│   │   ├── metric_calculation.py
│   │   ├── submission_template.csv
│   │   ├── submission_example.csv
│   │   └── auxiliar_metric_computation_example.csv
│   └── instructions/            # Competition documentation
│
├── tests/
│   └── test_smoke.py            # End-to-end smoke tests
│
├── env/
│   ├── requirements.txt
│   ├── colab_requirements.txt
│   └── environment.yml
│
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Core Modules

### 1. Utilities & Config (`src/utils.py`)

Core helpers used across the codebase:

```python
# src/utils.py

def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and optionally torch for reproducibility."""

def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Configure logging for console and optional file output."""

@contextmanager
def timer(name: str) -> Generator:
    """Context manager to time code blocks."""
    # Usage: with timer("Feature engineering"): ...

def load_config(path: str) -> dict:
    """Load YAML configuration file."""

def get_path(config: dict, key: str) -> Path:
    """Resolve paths from nested config keys like 'paths.raw_dir'."""
```

Example usage:

```python
from src.utils import set_seed, setup_logging, timer, load_config

set_seed(42)
logger = setup_logging()
config = load_config("configs/data.yaml")

with timer("Build features"):
    features = make_features(panel, scenario="scenario1")
```

---

### 2. Data Loading & Panel Construction (`src/data.py`)

This module handles all data ingestion and preprocessing.

#### Key Functions

```python
# src/data.py

def load_raw_data(config: dict, split: str = "train") -> dict[str, pd.DataFrame]:
    """
    Load all three datasets for train or test split using data.yaml config.
    
    Args:
        config: Loaded data.yaml config dict
        split: "train" or "test"
    
    Returns:
        Dictionary with keys: 'volume', 'generics', 'medicine_info'
    
    Example:
        config = load_config('configs/data.yaml')
        train_data = load_raw_data(config, split='train')
    
    Implementation:
        base_dir = config['paths']['raw_dir']
        files = config['files'][split]
        return {
            'volume': pd.read_csv(Path(base_dir) / files['volume']),
            'generics': pd.read_csv(Path(base_dir) / files['generics']),
            'medicine_info': pd.read_csv(Path(base_dir) / files['medicine_info']),
        }
    """

def prepare_base_panel(
    volume_df: pd.DataFrame,
    generics_df: pd.DataFrame,
    medicine_info_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create unified panel with all features joined.
    
    Joins:
    - volume ← generics on (country, brand_name, months_postgx)
    - result ← medicine_info on (country, brand_name)
    
    Returns:
        Panel with columns: country, brand_name, month, months_postgx,
        volume, n_gxs, ther_area, hospital_rate, main_package,
        biological, small_molecule
    """

def compute_pre_entry_stats(
    panel_df: pd.DataFrame,
    is_train: bool = True
) -> pd.DataFrame:
    """
    Compute pre-entry statistics for each series.
    
    Args:
        panel_df: Panel data (train or test)
        is_train: If True, computes target-dependent stats (bucket, mean_erosion)
    
    Always computes:
    - avg_vol_12m: Mean volume over months [-12, -1]
      (For test series with no pre-entry history, this must be handled via fallback or separate logic)
    
    If is_train=True, also computes:
    - mean_erosion: Mean of normalized post-entry volumes
    - bucket: 1 if mean_erosion <= 0.25 else 2
    
    CRITICAL: bucket is NEVER computed on test data and NEVER used as a feature.
    """

# ============================================================================
# TRAIN VS TEST BEHAVIOR DOCUMENTATION
# ============================================================================
#
# The `compute_pre_entry_stats()` function behaves differently based on the
# `is_train` parameter. This is critical for preventing data leakage.
#
# TRAIN MODE (is_train=True):
# ---------------------------
# When processing training data, the function computes:
#   1. avg_vol_12m: Mean volume over months [-12, -1] (pre-entry baseline)
#   2. y_norm: Normalized target = volume / avg_vol_12m (for supervised learning)
#   3. mean_erosion: Mean of y_norm over post-entry months [0, 23]
#   4. bucket: Classification based on mean_erosion
#      - Bucket 1 (high erosion): mean_erosion <= 0.25
#      - Bucket 2 (low/medium erosion): mean_erosion > 0.25
#   5. pre_entry_months_available: Count of months used for avg_vol_12m
#
# These statistics are used for:
#   - Sample weighting (bucket drives 2x weight for Bucket 1)
#   - Stratified validation splits (maintain B1/B2 ratio)
#   - Error analysis by bucket
#
# TEST MODE (is_train=False):
# ---------------------------
# When processing test data, the function ONLY computes:
#   1. avg_vol_12m: Mean volume over pre-entry months
#   2. pre_entry_months_available: Diagnostic count
#
# The function NEVER computes on test data:
#   - y_norm (no target available)
#   - mean_erosion (target-derived)
#   - bucket (target-derived)
#
# This ensures no leakage of target information into test predictions.
#
# FALLBACK HIERARCHY FOR avg_vol_12m:
# -----------------------------------
# Test series may have insufficient pre-entry history. The implementation
# uses a 3-level fallback hierarchy:
#
#   Level 1: Any available pre-entry months (months_postgx < 0)
#            - Uses whatever pre-entry data exists for the series
#            - Falls through if no pre-entry months available
#
#   Level 2: Therapeutic area median
#            - Uses median avg_vol_12m from training series in same ther_area
#            - Provides domain-aware fallback
#            - Falls through if ther_area is unknown/missing
#
#   Level 3: Global median (last resort)
#            - Uses global median avg_vol_12m from all training series
#            - Ensures every series has a valid avg_vol_12m
#
# VALIDATION:
# -----------
# The function logs statistics after computation:
#   - avg_vol_12m range, median, mean
#   - Bucket distribution (train only)
#   - Count of series using each fallback level
#
# ============================================================================

def handle_missing_values(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply missing value strategy. Must be called BEFORE compute_pre_entry_stats.
    
    | Column          | Strategy                              |
    |-----------------|---------------------------------------|
    | volume          | Keep as-is (rare NaN)                 |
    | n_gxs           | Forward-fill per series, then 0       |
    | hospital_rate   | Median by ther_area + flag            |
    | ther_area       | "Unknown" category                    |
    | main_package    | "Unknown" category                    |
    | biological      | False if missing + flag               |
    | small_molecule  | False if missing + flag               |
    """
```

#### Data Schema

| Dataset | Primary Key | Key Columns |
|---------|-------------|-------------|
| `df_volume_*` | `(country, brand_name, months_postgx)` | `month`, `volume` |
| `df_generics_*` | `(country, brand_name, months_postgx)` | `n_gxs` |
| `df_medicine_info_*` | `(country, brand_name)` | `ther_area`, `hospital_rate`, `main_package`, `biological`, `small_molecule` |

---

### 3. Feature Engineering (`src/features.py`)

Scenario-aware feature construction with strict leakage prevention.

#### Feature Categories

```text
┌─────────────────────────────────────────────────────────────────────┐
│  CATEGORY 1: PRE-ENTRY STATISTICS (Both Scenarios)                  │
│  ├─ avg_vol_12m: Mean volume over months [-12, -1]                  │
│  ├─ avg_vol_6m: Mean volume over months [-6, -1]                    │
│  ├─ avg_vol_3m: Mean volume over months [-3, -1]                    │
│  ├─ pre_entry_trend: Linear slope over pre-entry period             │
│  ├─ pre_entry_volatility: std(volume_pre) / avg_vol_12m             │
│  ├─ pre_entry_max: Maximum volume in pre-entry period               │
│  └─ log_avg_vol: log1p(avg_vol_12m) for scale normalization         │
├─────────────────────────────────────────────────────────────────────┤
│  CATEGORY 2: TIME FEATURES                                           │
│  ├─ months_postgx: Direct time index                                │
│  ├─ months_postgx_sq: Squared for non-linear decay                  │
│  ├─ is_post_entry: Binary (months_postgx >= 0)                      │
│  ├─ time_bucket: Categorical (pre, 0-5, 6-11, 12-23)                │
│  └─ month_of_year: Seasonality from calendar month                   │
├─────────────────────────────────────────────────────────────────────┤
│  CATEGORY 3: GENERICS COMPETITION                                    │
│  ├─ n_gxs: Current number of generics                               │
│  ├─ has_generic: Binary (n_gxs > 0)                                 │
│  ├─ multiple_generics: Binary (n_gxs >= 2)                          │
│  ├─ n_gxs_cummax: Maximum n_gxs up to current month                 │
│  └─ first_generic_month: Month when n_gxs first > 0                 │
├─────────────────────────────────────────────────────────────────────┤
│  CATEGORY 4: DRUG CHARACTERISTICS                                    │
│  ├─ ther_area: Therapeutic area (encoded)                           │
│  ├─ biological: Boolean                                              │
│  ├─ small_molecule: Boolean                                          │
│  ├─ hospital_rate: Percentage (0-100)                                │
│  ├─ hospital_rate_bin: Low/Medium/High                               │
│  ├─ main_package: Dosage form (encoded)                              │
│  └─ is_injection: Binary from main_package                           │
├─────────────────────────────────────────────────────────────────────┤
│  CATEGORY 5: SCENARIO 2 ONLY (Early Post-Entry Signal)              │
│  ├─ avg_vol_0_5: Mean volume over months [0, 5]                     │
│  ├─ erosion_0_5: avg_vol_0_5 / avg_vol_12m                          │
│  ├─ trend_0_5: Linear slope over months 0-5                         │
│  └─ drop_month_0: volume[0] / avg_vol_12m (initial drop)            │
├─────────────────────────────────────────────────────────────────────┤
│  CATEGORY 6: INTERACTION FEATURES (Optional)                         │
│  ├─ n_gxs_x_biological: Interaction term                            │
│  ├─ hospital_rate_x_time: Time-varying hospital effect               │
│  └─ ther_area_x_n_gxs: Therapeutic-specific competition effect      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Main Functions

```python
# src/features.py

def make_features(
    panel_df: pd.DataFrame, 
    scenario: str, 
    mode: str = "train"
) -> pd.DataFrame:
    """
    Build features respecting scenario-specific information constraints.
    
    Args:
        panel_df: Base panel with all raw data (must have avg_vol_12m)
        scenario: "scenario1" or "scenario2"
        mode: "train" or "test"
    
    Returns:
        DataFrame with engineered features.
        If mode="train", includes target column (y_norm).
    
    CRITICAL - y_norm Creation:
        if mode == "train":
            df['y_norm'] = df['volume'] / df['avg_vol_12m']
        
        y_norm is what models predict; actual volume is recovered via:
        volume = y_norm * avg_vol_12m
    
    Cutoff Rules:
        - scenario1: cutoff_month = 0 (only months_postgx < 0 for feature derivation)
        - scenario2: cutoff_month = 6 (months_postgx < 6 allowed for feature derivation)
    """

def select_training_rows(panel_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Select only rows that are valid supervised targets for the scenario.
    
    Scenario 1: months_postgx in [0, 23]
    Scenario 2: months_postgx in [6, 23]
    
    ALWAYS call this before splitting into features/target.
    """

def add_pre_entry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-entry statistics (Category 1)."""

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features (Category 2)."""

def add_generics_features(df: pd.DataFrame, cutoff_month: int) -> pd.DataFrame:
    """Add generics competition features respecting cutoff (Category 3)."""

def add_drug_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add static drug characteristics (Category 4)."""

def add_early_erosion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add early post-entry signal features for Scenario 2 (Category 5)."""
```

#### Leakage Prevention Rules

| Rule | Implementation |
|------|----------------|
| **Never use future target-derived info** | Filter `months_postgx < cutoff` for features derived from volume/erosion |
| **Exogenous futures allowed** | If `df_generics_test` provides `n_gxs` for forecast months, these can be used |
| **Never use bucket as feature** | Bucket is computed from target; use only for analysis |
| **Separate train/test pipelines** | Same code, but test has no access to train statistics |
| **No global statistics on test** | Statistics (mean, std) computed only on training data |

---

### 4. Official Metrics Wrapper (`src/evaluate.py`)

Wraps `metric_calculation.py` for local validation.

> **Note:** `docs/guide/metric_calculation.py` is the official script as provided by the organizers (copied verbatim). Do not edit it.

**IMPORTANT:** Both `df_actual` and `df_pred` must contain **actual volume** (not normalized) in a `volume` column. The official metric operates on raw volumes, not `y_norm`.

```python
# src/evaluate.py

def compute_metric1(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """
    Wrapper around official compute_metric1 for Scenario 1.
    
    IMPORTANT: df_actual and df_pred must have columns:
        [country, brand_name, months_postgx, volume]
    where 'volume' is ACTUAL volume (not normalized y_norm).
    
    Phase 1A weights:
    - 20%: Monthly error (months 0-23)
    - 50%: Accumulated error (months 0-5) [CRITICAL]
    - 20%: Accumulated error (months 6-11)
    - 10%: Accumulated error (months 12-23)
    
    Bucket 1 weighted 2×, Bucket 2 weighted 1×.
    """

def compute_metric2(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame
) -> float:
    """
    Wrapper around official compute_metric2 for Scenario 2.
    
    IMPORTANT: df_actual and df_pred must have columns:
        [country, brand_name, months_postgx, volume]
    where 'volume' is ACTUAL volume (not normalized y_norm).
    
    Phase 1B weights:
    - 20%: Monthly error (months 6-23)
    - 50%: Accumulated error (months 6-11) [CRITICAL]
    - 30%: Accumulated error (months 12-23)
    
    Bucket 1 weighted 2×, Bucket 2 weighted 1×.
    """

def compute_bucket_metrics(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame,
    scenario: str
) -> dict:
    """
    Compute metrics separately for Bucket 1 and Bucket 2.
    
    Returns:
        {'overall': float, 'bucket1': float, 'bucket2': float}
    """

def create_aux_file(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create auxiliary file for metric computation.
    
    Returns DataFrame with columns:
    - country, brand_name, avg_vol, bucket
    
    NOTE: Only create from training data. For test, organizers have their own.
    """
```

---

### 5. Validation & Splitting (`src/validation.py`)

Series-level, stratified validation mimicking true scenario constraints.

```python
# src/validation.py

def create_validation_split(
    panel_df: pd.DataFrame,
    val_fraction: float = 0.2,
    stratify_by: str = 'bucket',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split at SERIES level (not row level).
    
    CRITICAL: Never mix months of the same series across train/val.
    
    Args:
        panel_df: Full training panel
        val_fraction: Fraction of series for validation
        stratify_by: Column to stratify by (usually 'bucket')
        random_state: For reproducibility
    
    Returns:
        (train_df, val_df) - both contain full series
    """

def simulate_scenario(
    val_df: pd.DataFrame,
    scenario: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare validation data mimicking true scenario constraints.
    
    Note: In the main pipeline, scenario constraints are typically enforced 
    during feature engineering (make_features) and row selection (select_training_rows).
    This function is useful for explicit history/horizon splitting experiments.
    
    Scenario 1:
        - features_df: months_postgx < 0 only
        - targets_df: months_postgx in [0, 23]
    
    Scenario 2:
        - features_df: months_postgx < 6
        - targets_df: months_postgx in [6, 23]
    """

def adversarial_validation(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame
) -> dict:
    """
    Detect train/test distribution shift.
    
    Train classifier to distinguish train vs test rows.
    
    Returns:
        {
            'mean_auc': float,  # ~0.5 = good, >0.7 = shift detected
            'auc_scores': list,
            'top_shift_features': DataFrame
        }
    
    If AUC > 0.7:
        - Inspect top features driving shift
        - Consider dropping or simplifying those features
        - Increase regularization
    """
```

---

### 6. Model Training Pipeline (`src/train.py`)

Unified training with sample weights aligned to official metric.

```python
# src/train.py

# Define column groups to prevent leakage
META_COLS = ['country', 'brand_name', 'months_postgx', 'bucket', 'avg_vol_12m', 'y_norm']
TARGET_COL = 'y_norm'

def split_features_target_meta(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Separate pure features from target and meta columns.
    
    This GUARANTEES bucket/y_norm never leak into model features.
    
    Returns:
        X: Pure features for model
        y: Target (y_norm)
        meta: Meta columns for weights, grouping, metrics
    """

def get_feature_matrix_and_meta(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For INFERENCE: separate features from meta (no target).
    
    CRITICAL: feature_cols must match training exactly!
    """

def compute_sample_weights(meta_df: pd.DataFrame, scenario: str) -> pd.Series:
    """
    Compute sample weights that approximate official metric weighting.
    
    Scenario 1:
        - Months 0-5: weight 3.0 (highest priority)
        - Months 6-11: weight 1.5
        - Months 12-23: weight 1.0
    
    Scenario 2:
        - Months 6-11: weight 2.5 (highest priority)
        - Months 12-23: weight 1.0
    
    Bucket weights:
        - Bucket 1: multiply by 2.0
        - Bucket 2: multiply by 1.0
    """

def train_scenario_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
    scenario: str,
    model_type: str = 'catboost',
    config: dict = None
) -> Tuple[Model, dict]:
    """
    Train model for specific scenario with early stopping.
    
    Uses sample weights from META to align with official metric.
    
    Returns:
        (trained_model, metrics_dict)
    """
```

---

### 7. Inference & Submission Generation (`src/inference.py`)

Prediction generation with inverse transform and sanity checks.

```python
# src/inference.py

def detect_test_scenarios(test_volume: pd.DataFrame) -> dict:
    """
    Identify which test series belong to Scenario 1 vs 2.
    
    Detection rules (Heuristic):
        - Scenario 1: Series typically missing months 0-5 or has specific length patterns.
        - Scenario 2: Series typically has months 0-5 present.
        - Must validate against expected counts (228 S1, 112 S2).
    
    Returns:
        {'scenario1': list of (country, brand_name) tuples,
         'scenario2': list of (country, brand_name) tuples}
    """

def generate_submission(
    model_scenario1: Model,
    model_scenario2: Model,
    test_panel: pd.DataFrame,
    submission_template: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate final submission file.
    
    Args:
        test_panel: Pre-processed test panel (must have avg_vol_12m computed)
    
    CRITICAL: Models output normalized volume (y_norm).
    Must inverse transform: volume = y_norm * avg_vol_12m
    
    Steps:
    1. Detect scenarios and split test_panel
    2. Build features with make_features(..., mode="test")
    3. Get predictions (normalized)
    4. Inverse transform to actual volume using row-specific avg_vol_12m
    5. Post-process: clip negative to 0
    6. Merge with template
    7. Validate format
    """

def apply_edge_case_fallback(
    predictions: pd.DataFrame,
    edge_case_series: list,
    global_erosion_curve: pd.Series
) -> pd.DataFrame:
    """
    For problematic series, use conservative fallback.
    
    Edge case criteria:
    - Very low baseline (avg_vol_12m < P5)
    - Short pre-entry history (< 6 months)
    - High pre-entry volatility
    
    Fallback: global_erosion_curve * avg_vol_12m
    """

def validate_submission_format(
    submission_df: pd.DataFrame,
    template_df: pd.DataFrame
) -> bool:
    """
    Final sanity checks before submission.
    
    Checks:
    1. Row count matches template
    2. Correct columns: country, brand_name, months_postgx, volume
    3. No missing values in volume
    4. No negative volumes
    5. Keys match template exactly
    6. No duplicate keys
    """
```

---

## Model Implementations

> **Datathon Priority:** Focus on baselines + CatBoost hero model first. Only implement LightGBM/XGBoost/NN if time permits after a strong CatBoost baseline is validated with the official metric.

### Model Hierarchy

```text
┌──────────────────────────────────────────────────────────────────────┐
│  TIER 1: BASELINES (Mandatory - Establish Reference)                 │
│  ├─ Global Mean Baseline: Predict global average erosion curve       │
│  ├─ Flat Baseline: Predict avg_vol_12m for all future months         │
│  └─ Trend Extrapolation: Linear extrapolation from pre-entry         │
├──────────────────────────────────────────────────────────────────────┤
│  TIER 2: HERO MODELS (Primary Focus)                                 │
│  ├─ CatBoost: First choice - native categoricals, robust to overfit  │
│  ├─ LightGBM: Fast training, good for hyperparameter search          │
│  └─ XGBoost: Proven robustness, regularization options               │
├──────────────────────────────────────────────────────────────────────┤
│  TIER 3: OPTIONAL (If Time Permits)                                  │
│  ├─ Linear Model: Ridge/Elastic-Net for interpretability             │
│  ├─ Simple Ensemble: Average of CatBoost + LightGBM                  │
│  └─ Neural Network: MLP for large feature space                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Base Model Interface (`src/models/base.py`)

```python
# src/models/base.py

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Abstract base for all models ensuring consistent interface."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_names: list = []
    
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
        return pd.DataFrame()
```

### GBM Models (`src/models/cat_model.py`, `src/models/lgbm_model.py`)

```python
# src/models/cat_model.py

class CatBoostModel(BaseModel):
    """CatBoost implementation with native categorical support."""
    
    DEFAULT_CONFIG = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'loss_function': 'RMSE',
        'early_stopping_rounds': 50,
        'verbose': 100,
        'cat_features': []
    }
```

```python
# src/models/lgbm_model.py

class LGBMModel(BaseModel):
    """LightGBM implementation optimized for speed."""
    
    DEFAULT_CONFIG = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'early_stopping_rounds': 50
    }
```

### Baseline Models (`src/models/linear.py`)

```python
# src/models/linear.py

class GlobalMeanBaseline(BaseModel):
    """Predict using global average erosion curve from training data.
    
    Note: X_train must contain 'months_postgx' column.
    y_train is y_norm (normalized volume).
    """
    
    def fit(self, X_train, y_train, **kwargs):
        """Compute mean y_norm per months_postgx."""
        # Cleaner groupby using DataFrame
        self.erosion_curve = (
            pd.DataFrame({'months_postgx': X_train['months_postgx'], 'y_norm': y_train})
            .groupby('months_postgx')['y_norm']
            .mean()
            .to_dict()
        )
        return self
    
    def predict(self, X):
        """Apply learned erosion curve. X must have 'months_postgx'."""
        return X['months_postgx'].map(self.erosion_curve).fillna(
            np.mean(list(self.erosion_curve.values()))
        ).values

class FlatBaseline(BaseModel):
    """Predict 1.0 (no erosion) as normalized volume."""
    
    def predict(self, X):
        return np.ones(len(X))
```

---

## Configuration System

### `configs/data.yaml`

Data paths and column specifications:

```yaml
paths:
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"
  artifacts_dir: "artifacts"

files:
  train:
    volume: "TRAIN/df_volume_train.csv"
    generics: "TRAIN/df_generics_train.csv"
    medicine_info: "TRAIN/df_medicine_info_train.csv"
  test:
    volume: "TEST/df_volume_test.csv"
    generics: "TEST/df_generics_test.csv"
    medicine_info: "TEST/df_medicine_info_test.csv"
  
  aux_metric: "docs/guide/auxiliar_metric_computation_example.csv"
  submission_template: "docs/guide/submission_template.csv"

columns:
  id_keys: ["country", "brand_name"]
  time_key: "months_postgx"
  calendar_month: "month"
  raw_target: "volume"
  model_target: "y_norm"
  
  # Meta columns to exclude from features
  meta_cols:
    - "country"
    - "brand_name"
    - "months_postgx"
    - "bucket"
    - "avg_vol_12m"
    - "y_norm"
```

### `configs/features.yaml`

Feature engineering configuration:

```yaml
# Pre-entry statistics (both scenarios)
pre_entry:
  windows: [3, 6, 12]  # months for rolling averages
  compute_trend: true
  compute_volatility: true

# Time features
time:
  include_squared: true
  include_time_bucket: true
  include_month_of_year: true

# Generics features
generics:
  include_cummax: true
  include_first_month: true
  include_binary_flags: true

# Drug characteristics
drug:
  categoricals:
    - "ther_area"
    - "main_package"
  hospital_rate_bins: [30, 70]  # Low/Medium/High cutoffs
  
# Scenario 2 specific
scenario2_early:
  compute_avg_0_5: true
  compute_trend_0_5: true
  compute_drop_month_0: true

# Interaction features (optional)
interactions:
  enabled: false
  pairs:
    - ["n_gxs", "biological"]
    - ["hospital_rate", "months_postgx"]
```

### `configs/model_cat.yaml`

CatBoost hyperparameters:

```yaml
model_type: "catboost"

params:
  iterations: 2000
  learning_rate: 0.03
  depth: 6
  l2_leaf_reg: 3.0
  random_seed: 42
  loss_function: "RMSE"
  
training:
  early_stopping_rounds: 100
  use_sample_weights: true
  
categorical_features:
  - "ther_area"
  - "main_package"
  - "time_bucket"
  - "hospital_rate_bin"
```

### `configs/run_defaults.yaml`

Runtime configuration:

```yaml
reproducibility:
  seed: 42
  
validation:
  val_fraction: 0.2
  stratify_by: "bucket"
  
scenarios:
  scenario1:
    forecast_start: 0
    forecast_end: 23
    feature_cutoff: 0  # Only months_postgx < 0
  scenario2:
    forecast_start: 6
    forecast_end: 23
    feature_cutoff: 6  # months_postgx < 6

sample_weights:
  scenario1:
    months_0_5: 3.0
    months_6_11: 1.5
    months_12_23: 1.0
  scenario2:
    months_6_11: 2.5
    months_12_23: 1.0
  bucket_weights:
    bucket1: 2.0
    bucket2: 1.0

logging:
  level: "INFO"
  log_to_file: true
```

---

## Main Notebook Workflow

The main development notebook (`notebooks/colab/main.ipynb` or `notebooks/01_train.ipynb`) follows this structure.

> **Naming suggestion:** Consider renaming notebooks for clarity:
>
> * `00_eda.ipynb` -> `01_eda.ipynb`
> * `01_feature_prototype.ipynb` -> `02_feature_prototype.ipynb`
> * `01_train.ipynb` -> `03_train.ipynb`
> * `02_model_sanity.ipynb` -> `04_model_sanity.ipynb`

### Cell 1: Setup & Configuration

```python
import sys
sys.path.insert(0, '..')  # If running from notebooks/

from src.utils import set_seed, setup_logging, load_config, timer
from src.data import load_raw_data, prepare_base_panel, compute_pre_entry_stats
from src.features import make_features, select_training_rows
from src.train import split_features_target_meta, compute_sample_weights, train_scenario_model
from src.validation import create_validation_split, simulate_scenario
from src.evaluate import compute_metric1, compute_metric2, create_aux_file
from src.inference import detect_test_scenarios, generate_submission, validate_submission_format

set_seed(42)
logger = setup_logging()

data_config = load_config('configs/data.yaml')
feature_config = load_config('configs/features.yaml')
model_config = load_config('configs/model_cat.yaml')
run_config = load_config('configs/run_defaults.yaml')
```

### Cell 2: Load & Prepare Data

```python
with timer("Load raw data"):
    train_data = load_raw_data(data_config, split='train')
    test_data = load_raw_data(data_config, split='test')

with timer("Build panel"):
    train_panel = prepare_base_panel(
        train_data['volume'],
        train_data['generics'],
        train_data['medicine_info']
    )
    train_panel = handle_missing_values(train_panel)
    train_panel = compute_pre_entry_stats(train_panel, is_train=True)
    
    # Build test panel similarly
    test_panel = prepare_base_panel(
        test_data['volume'],
        test_data['generics'],
        test_data['medicine_info']
    )
    test_panel = handle_missing_values(test_panel)
    test_panel = compute_pre_entry_stats(test_panel, is_train=False)
    
print(f"Training panel: {len(train_panel):,} rows")
print(f"Bucket distribution:\n{train_panel.drop_duplicates('brand_name')['bucket'].value_counts()}")
```

### Cell 3: Train Scenario 1

```python
scenario = 'scenario1'

# Build features
with timer(f"Build features - {scenario}"):
    panel_features = make_features(train_panel, scenario=scenario)
    train_rows = select_training_rows(panel_features, scenario=scenario)

# Split train/validation
train_df, val_df = create_validation_split(
    train_rows, 
    val_fraction=run_config['validation']['val_fraction'],
    stratify_by='bucket'
)

# Prepare data
X_train, y_train, meta_train = split_features_target_meta(train_df)
X_val, y_val, meta_val = split_features_target_meta(val_df)

sample_weights = compute_sample_weights(meta_train, scenario=scenario)

# Train
with timer("Train CatBoost"):
    model_s1, metrics = train_scenario_model(
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        scenario=scenario,
        model_type='catboost',
        config=model_config
    )

# Evaluate with official metric (requires actual volume, not normalized)
yhat_norm = model_s1.predict(X_val)
avg_vol_val = meta_val['avg_vol_12m'].values  # For denormalization
yhat_volume = yhat_norm * avg_vol_val  # Convert to actual volume

# Build DataFrames for metric function (requires 'volume' column with actual values)
df_pred = val_df[['country', 'brand_name', 'months_postgx']].copy()
df_pred['volume'] = yhat_volume

df_actual = val_df[['country', 'brand_name', 'months_postgx', 'volume']].copy()
df_aux = create_aux_file(val_df)

metric1_val = compute_metric1(
    df_actual=df_actual,
    df_pred=df_pred,
    df_aux=df_aux
)
print(f"Scenario 1 Validation Metric: {metric1_val:.4f}")
```

### Cell 4: Train Scenario 2 (Similar Structure)

### Cell 5: Generate Submission

```python
import pandas as pd

# Load template and prepare test data
template = pd.read_csv(data_config['files']['submission_template'])

# Detect test scenarios
scenario_split = detect_test_scenarios(test_data['volume'])
print(f"Scenario 1 series: {len(scenario_split['scenario1'])}")
print(f"Scenario 2 series: {len(scenario_split['scenario2'])}")

# Generate predictions
submission = generate_submission(
    model_scenario1=model_s1,
    model_scenario2=model_s2,
    test_panel=test_panel,
    submission_template=template
)

# Validate and save
assert validate_submission_format(submission, template)
submission.to_csv('submissions/submission.csv', index=False)
print("Submission saved!")
```

---

## End-to-End Workflow

### Complete Pipeline Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA PREPARATION                                           │
│  ├─ Load raw CSVs (volume, generics, medicine_info)                 │
│  ├─ Build unified panel                                             │
│  ├─ Compute pre-entry stats (avg_vol_12m, bucket for train)         │
│  └─ Output: train_panel, test_panel                                 │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 2: FEATURE ENGINEERING (Per Scenario)                         │
│  ├─ Call make_features(panel, scenario)                             │
│  ├─ Respect information cutoff (S1: month<0, S2: month<6)           │
│  ├─ Add S2-specific early erosion features                          │
│  └─ Output: feature DataFrame (includes y_norm target in train mode)│
├─────────────────────────────────────────────────────────────────────┤
│  STEP 3: VALIDATION SETUP                                           │
│  ├─ Series-level stratified split (by bucket)                       │
│  ├─ Simulate scenario constraints on validation set                 │
│  ├─ Compute sample weights                                          │
│  └─ Output: (X_train, y_train, meta_train), (X_val, y_val, meta_val)│
├─────────────────────────────────────────────────────────────────────┤
│  STEP 4: MODEL TRAINING                                             │
│  ├─ Train with sample weights                                       │
│  ├─ Early stopping on validation loss                               │
│  ├─ Log feature importance                                          │
│  └─ Output: trained_model, training_metrics                         │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 5: LOCAL EVALUATION                                           │
│  ├─ Predict on validation set                                       │
│  ├─ Compute official metric (Metric1 or Metric2)                    │
│  ├─ Analyze per-bucket performance                                  │
│  └─ Output: metric scores, error analysis                           │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 6: INFERENCE & SUBMISSION                                     │
│  ├─ Detect scenario assignment for test series                      │
│  ├─ Build features (same pipeline as training)                      │
│  ├─ Predict normalized volume (y_norm)                              │
│  ├─ Inverse transform: volume = y_norm * avg_vol_12m                │
│  ├─ Post-process (clip negatives, handle edge cases)                │
│  ├─ Validate format against template                                │
│  └─ Output: submission.csv                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Smoke Test Protocol

Before any major training run, execute this checklist:

```python
# smoke_test.py or within notebook

def run_smoke_test():
    """Quick validation that entire pipeline works end-to-end."""
    
    # 1. Data loads without error
    train_data = load_raw_data(config, split='train')
    assert len(train_data['volume']) > 0
    
    # 2. Panel builds correctly
    panel = prepare_base_panel(...)
    assert 'months_postgx' in panel.columns
    assert panel[['country', 'brand_name', 'months_postgx']].duplicated().sum() == 0
    
    # 3. Features build without NaN explosion
    features = make_features(panel, scenario='scenario1', mode='train')
    nan_ratio = features.isna().mean().max()
    assert nan_ratio < 0.2, f"Too many NaNs: {nan_ratio}"
    
    # 4. Model trains (on tiny subset)
    mini = features.sample(1000)
    X, y, meta = split_features_target_meta(mini)
    assert 'bucket' not in X.columns, "Bucket leaked into features!"
    assert 'y_norm' not in X.columns, "Target leaked into features!"
    assert X.isna().sum().sum() == 0, "NaNs in model input!"
    
    model = CatBoostModel(config)
    model.fit(X[:800], y[:800], X[800:], y[800:])
    preds = model.predict(X[800:])
    assert len(preds) == 200
    assert not np.any(np.isnan(preds))
    
    # 5. Metric computes
    # Build df_actual, df_pred, df_aux as in the main notebook example
    metric = compute_metric1(...)
    assert 0 < metric < 10  # Sanity range
    
    print("Smoke test passed!")

run_smoke_test()
```

### Implementation Refinements & Known Issues

The following refinements have been identified as critical for correctness and robustness:

1.  **Test Panel Handling**: `compute_pre_entry_stats` must be called for both train and test. Test data may lack pre-entry history, requiring fallback logic for `avg_vol_12m`.
2.  **Submission Generation**: `generate_submission` must use the pre-computed `test_panel` (with its own `avg_vol_12m`) rather than looking up training values.
3.  **Feature Mode**: `make_features` must have an explicit `mode="test"` to prevent `y_norm` calculation on test data.
4.  **Scenario Detection**: `detect_test_scenarios` requires heuristic logic (e.g., checking for presence of months 0-5) rather than simple min/max checks, as test data structure may vary.
5.  **Adversarial Validation**: Should be integrated into the standard workflow to detect train/test distribution shifts.
6.  **CLI Entry Points**: `src/train.py` and `src/inference.py` should expose proper `argparse` CLIs.

---

## Key Technical Decisions

### 1. Normalized Target (y_norm)

**Why**: Raw volume varies by 6+ orders of magnitude across series. Using `y_norm = volume / avg_vol_12m` makes all series comparable and enables the model to learn erosion patterns rather than absolute volumes.

**Inverse Transform**: Final submission requires `volume = y_norm * avg_vol_12m`.

### 2. Sample Weights Approximating Official Metric

**Why**: The official metric heavily weights early months (50% for months 0-5 in Scenario 1) and Bucket 1 (2x weight). Standard RMSE training would under-optimize these critical segments.

**Implementation**: Compute per-row weights based on `months_postgx` and `bucket`, pass to GBM `.fit()`.

### 3. Never Use Bucket as Feature

**Why**: Bucket is computed from the target variable (mean post-entry erosion). Using it as a feature would be direct target leakage.

**Use Cases**: Bucket is only used for:

* Stratified train/val splitting
* Sample weight computation
* Offline error analysis

### 4. Series-Level Validation Split

**Why**: Splitting at row-level would put some months of a series in train and others in validation, allowing the model to "peek" at the series' behavior.

**Implementation**: Group by `(country, brand_name)`, then split at group level.

---

## Usage Guide

### Local Development

```bash
# 1. Clone repository
git clone <repo-url>
cd novartis_datathon_2025

# 2. Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Place data files
# data/raw/TRAIN/df_volume_train.csv
# data/raw/TRAIN/df_generics_train.csv
# data/raw/TRAIN/df_medicine_info_train.csv
# data/raw/TEST/df_volume_test.csv
# data/raw/TEST/df_generics_test.csv
# data/raw/TEST/df_medicine_info_test.csv

# 4. Run smoke test (end-to-end pipeline validation)
pytest tests/test_smoke.py -v
# Or run individual test functions from tests/test_smoke.py

# 5. Train via notebook or CLI
jupyter notebook notebooks/01_train.ipynb
# OR
python -m src.train --config configs/run_defaults.yaml

# 6. Generate submission
python -m src.inference --output submissions/submission.csv
```

### Google Colab Workflow

```python
# Cell 1: Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

!git clone <repo-url> /content/novartis_datathon_2025
%cd /content/novartis_datathon_2025

# Cell 2: Install dependencies
!pip install -q -r env/colab_requirements.txt

# Cell 3: Symlink data from Drive
!ln -s '/content/drive/MyDrive/novartis_data' data/raw

# Cell 4: Run main notebook
%run notebooks/colab/main.ipynb
```

### Submission Protocol

1. **Max 3 submissions per 8-hour window**
2. **Public LB**: 30% of test data (use for sanity check only)
3. **Private LB**: 70% of test data (final ranking)
4. **Strategy**: Reserve submissions for:
   * Initial validation (1)
   * Best single model (1)
   * Final ensemble/refined model (1)

