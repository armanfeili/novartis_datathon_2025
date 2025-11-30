# Novartis Datathon 2025 - Comprehensive Codebase Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Directory Structure](#2-architecture--directory-structure)
3. [Configuration System](#3-configuration-system)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Architecture](#6-model-architecture)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference & Submission](#8-inference--submission)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [API Reference](#10-api-reference)
11. [Best Practices & Guidelines](#11-best-practices--guidelines)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

### 1.1 Competition Background

The Novartis Datathon 2025 focuses on **Generic Erosion Forecasting** - predicting how pharmaceutical drug volumes change after Loss of Exclusivity (LOE), when generic competitors enter the market.

**Key Challenge**: Forecast monthly drug volumes for 24 months post-generic entry, under two scenarios:
- **Scenario 1**: No post-entry actuals available (forecast months 0-23)
- **Scenario 2**: First 6 months of actuals available (forecast months 6-23)

### 1.2 Data Sources

The competition provides three primary data files:

| File | Description | Primary Key |
|------|-------------|-------------|
| `df_volume_{train/test}.csv` | Monthly volume data | (country, brand_name, months_postgx) |
| `df_generics_{train/test}.csv` | Generic competitor counts | (country, brand_name, months_postgx) |
| `df_medicine_info_{train/test}.csv` | Static drug characteristics | (country, brand_name) |

### 1.3 Target Variable

Models predict **normalized volume** (`y_norm`), which is the ratio of actual volume to pre-entry average:

```
y_norm = volume / avg_vol_12m
```

Where `avg_vol_12m` is computed from the 12 months prior to generic entry (months -12 to -1).

**Inverse Transform** for submission:
```
volume = y_norm * avg_vol_12m
```

### 1.4 Key Metrics

The official competition uses weighted Percentage Error (PE) metrics:

| Metric | Scenario | Weights |
|--------|----------|---------|
| Metric 1 | Scenario 1 | 50% months 0-5, 20% months 6-11, 10% months 12-23, 20% monthly |
| Metric 2 | Scenario 2 | 50% months 6-11, 30% months 12-23, 20% monthly |

Bucket weighting:
- **Bucket 1** (high erosion, mean_erosion ≤ 0.25): 2x weight
- **Bucket 2** (low erosion, mean_erosion > 0.25): 1x weight

---

## 2. Architecture & Directory Structure

### 2.1 Project Layout

```
novartis_datathon_2025/
├── configs/                    # YAML configuration files
│   ├── data.yaml              # Data paths, columns, schema
│   ├── features.yaml          # Feature engineering settings
│   ├── run_defaults.yaml      # Training defaults, metrics, weights
│   ├── model_cat.yaml         # CatBoost configuration
│   ├── model_lgbm.yaml        # LightGBM configuration
│   ├── model_xgb.yaml         # XGBoost configuration
│   ├── model_nn.yaml          # Neural network configuration
│   ├── model_linear.yaml      # Linear models configuration
│   ├── model_hybrid.yaml      # Hybrid physics+ML configuration
│   ├── model_arihow.yaml      # ARIMA+HW configuration
│   ├── model_cnn_lstm.yaml    # CNN-LSTM configuration
│   └── model_kg_gcn_lstm.yaml # Knowledge Graph GCN-LSTM configuration
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data.py                # Data loading, panel construction
│   ├── features.py            # Feature engineering
│   ├── train.py               # Training pipeline
│   ├── inference.py           # Submission generation
│   ├── evaluate.py            # Metric computation
│   ├── validation.py          # Cross-validation utilities
│   ├── utils.py               # Helper functions
│   ├── config_sweep.py        # Hyperparameter sweep utilities
│   ├── external_data.py       # External data integration
│   ├── graph_utils.py         # Graph utilities for GCN
│   ├── scenario_analysis.py   # What-if scenario analysis
│   ├── sequence_builder.py    # Sequence data for LSTM
│   ├── visibility_sources.py  # Supply chain visibility
│   │
│   └── models/                 # Model implementations
│       ├── __init__.py
│       ├── base.py            # BaseModel interface
│       ├── cat_model.py       # CatBoost
│       ├── lgbm_model.py      # LightGBM
│       ├── xgb_model.py       # XGBoost
│       ├── nn.py              # Neural Network (MLP)
│       ├── linear.py          # Linear models + baselines
│       ├── ensemble.py        # Ensemble methods
│       ├── hybrid_physics_ml.py # Physics+ML hybrid
│       ├── arihow.py          # ARIMA+Holt-Winters
│       ├── cnn_lstm.py        # CNN-LSTM temporal model
│       ├── kg_gcn_lstm.py     # KG-GCN-LSTM
│       ├── gcn_layers.py      # GCN layer implementations
│       └── baselines.py       # Simple baseline models
│
├── data/                       # Data directory
│   ├── raw/                   # Original CSV files
│   │   ├── TRAIN/
│   │   └── TEST/
│   ├── interim/               # Cached panels (parquet)
│   └── processed/             # Processed features
│
├── artifacts/                  # Training artifacts
│   └── {timestamp}_{model}_{scenario}/
│       ├── model_*.bin        # Saved model
│       ├── metadata.json      # Run metadata
│       ├── metrics.json       # Evaluation metrics
│       ├── feature_importance.csv
│       ├── config_snapshot.yaml
│       └── train.log
│
├── submissions/                # Competition submissions
│   └── {version}/
│       ├── submission.csv
│       ├── auxiliary.csv
│       └── metadata.json
│
├── logs/                       # Training logs
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

### 2.2 Module Dependencies

```
                    ┌─────────────────┐
                    │   configs/      │
                    │   (YAML files)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   src/utils.py  │
                    │   (Core utils)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐     │     ┌────────▼────────┐
     │  src/data.py    │     │     │  src/evaluate.py │
     │  (Data loading) │     │     │  (Metrics)       │
     └────────┬────────┘     │     └────────┬────────┘
              │              │              │
     ┌────────▼────────┐     │              │
     │ src/features.py │     │              │
     │ (Feature eng.)  │     │              │
     └────────┬────────┘     │              │
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  src/train.py   │
                    │  (Training)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ src/inference.py│
                    │ (Submission)    │
                    └─────────────────┘
```

---

## 3. Configuration System

### 3.1 Configuration Files Overview

The project uses a hierarchical YAML-based configuration system with three main config files:

| File | Purpose | Key Settings |
|------|---------|--------------|
| `configs/data.yaml` | Data paths, schema, missing value handling | File paths, column definitions, validation |
| `configs/features.yaml` | Feature engineering settings | Pre-entry stats, time features, interactions |
| `configs/run_defaults.yaml` | Training parameters | Sample weights, metrics, scenarios |

### 3.2 data.yaml - Data Configuration

```yaml
# Path configuration
paths:
  raw_dir: "data/raw"           # Raw CSV files
  interim_dir: "data/interim"   # Cached parquet panels
  processed_dir: "data/processed"
  artifacts_dir: "artifacts"

# File locations per split
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

#### 3.2.1 Column Definitions

The `columns` section defines canonical column names used throughout the codebase:

| Column Type | Columns | Purpose |
|-------------|---------|---------|
| **ID Keys** | `country`, `brand_name` | Series identification |
| **Time Key** | `months_postgx` | Time index (months since generic entry) |
| **Calendar Month** | `month` | Calendar month for seasonality |
| **Raw Target** | `volume` | Actual volume (raw) |
| **Model Target** | `y_norm` | Normalized volume (model predicts this) |

#### 3.2.2 Meta Columns (CRITICAL)

**Meta columns are NEVER used as model features**. They are excluded during feature matrix construction:

```yaml
meta_cols:
  - "country"         # Series identifier
  - "brand_name"      # Series identifier
  - "months_postgx"   # Time index (used for row identification)
  - "bucket"          # Target-derived (training only)
  - "avg_vol_12m"     # Used for target normalization
  - "y_norm"          # The target variable
  - "volume"          # Raw target
  - "mean_erosion"    # Target-derived statistic
  - "month"           # Calendar month (kept in meta)
```

⚠️ **IMPORTANT**: These columns are synchronized across:
- `configs/data.yaml` → `columns.meta_cols`
- `src/data.py` → `META_COLS` constant
- `src/train.py` → Feature/target/meta separation logic

#### 3.2.3 Data Schema

The schema section documents expected columns per data file:

```yaml
schema:
  volume:
    primary_key: ["country", "brand_name", "months_postgx"]
    columns: ["country", "brand_name", "month", "months_postgx", "volume"]
  
  generics:
    primary_key: ["country", "brand_name", "months_postgx"]
    columns: ["country", "brand_name", "months_postgx", "n_gxs"]
  
  medicine_info:
    primary_key: ["country", "brand_name"]
    columns: ["country", "brand_name", "ther_area", "hospital_rate", 
              "main_package", "biological", "small_molecule"]
```

#### 3.2.4 Missing Value Handling

Each column has a defined strategy for handling missing values:

| Column | Strategy | Details |
|--------|----------|---------|
| `volume` | Keep | Rare NaN, keep as-is |
| `n_gxs` | Forward-fill + 0 | Forward-fill per series, then fill 0 |
| `hospital_rate` | Median by group + flag | Median by ther_area, add missing flag |
| `ther_area` | "Unknown" | Fill with "Unknown" category |
| `main_package` | "Unknown" | Fill with "Unknown" category |
| `biological` | False + flag | Default False, add missing flag |
| `small_molecule` | False + flag | Default False, add missing flag |

#### 3.2.5 Validation Settings

```yaml
validation:
  expected_train_series: 1953     # Expected unique (country, brand_name) in train
  expected_test_series_s1: 228    # Scenario 1 test series
  expected_test_series_s2: 112    # Scenario 2 test series
  forecast_horizon: 24            # Months 0-23
```

### 3.3 features.yaml - Feature Engineering Configuration

#### 3.3.1 Pre-Entry Features

Features derived from `months_postgx < 0` (pre-generic entry period):

```yaml
pre_entry:
  windows: [3, 6, 12]      # Rolling average windows
  compute_trend: true       # Linear slope of volume
  compute_volatility: true  # std(volume) / avg_vol_12m
  compute_max: true         # Maximum pre-entry volume
  compute_min: true         # Minimum pre-entry volume
  log_transform: true       # log1p(avg_vol_12m)
  compute_seasonal: true    # Seasonal pattern detection
```

#### 3.3.2 Time Features

```yaml
time:
  include_months_postgx: true      # Direct time index
  include_squared: true            # months_postgx^2
  include_is_post_entry: true      # Binary flag
  include_time_bucket: true        # Categorical (pre, 0-5, 6-11, 12-23)
  include_month_of_year: true      # Calendar month encoding
  include_quarters: true           # Q1-Q4 encoding
  include_decay: true              # exp(-alpha * months)
  decay_alpha: 0.1
```

#### 3.3.3 Generics Competition Features

```yaml
generics:
  include_n_gxs: true              # Current generic count
  include_has_generic: true        # Binary (n_gxs > 0)
  include_multiple_generics: true  # Binary (n_gxs >= 2)
  include_cummax: true             # Max n_gxs up to current month
  include_first_month: true        # First month with generics
  include_entry_speed: true        # Rate of generic entry
  include_future_n_gxs: true       # Future n_gxs (exogenous, not leakage)
```

#### 3.3.4 Drug Characteristics

```yaml
drug:
  categoricals:
    - "ther_area"
    - "main_package"
  numerical:
    - "hospital_rate"
  boolean:
    - "biological"
    - "small_molecule"
  hospital_rate_bins: [30, 70]     # Low/Medium/High
  derive_is_injection: true        # From main_package
  encoding: "label"                # label, onehot, target
```

#### 3.3.5 Scenario 2 Early Erosion Features

**Only computed for Scenario 2** (when months 0-5 actuals are available):

```yaml
scenario2_early:
  compute_avg_0_5: true        # Mean volume months 0-5
  compute_erosion_0_5: true    # avg_vol_0_5 / avg_vol_12m
  compute_trend_0_5: true      # Linear slope months 0-5
  compute_drop_month_0: true   # volume[0] / avg_vol_12m
```

#### 3.3.6 Leakage Prevention Rules

```yaml
leakage_prevention:
  forbidden_features:
    - "bucket"          # Target-derived
    - "y_norm"          # Target
    - "volume"          # Raw target
    - "mean_erosion"    # Target-derived
    - "country"         # Meta key
    - "brand_name"      # Meta key
  
  scenario1:
    feature_cutoff: 0       # Only months_postgx < 0 for features
    target_range: [0, 23]
  
  scenario2:
    feature_cutoff: 6       # months_postgx < 6 allowed
    target_range: [6, 23]
```

### 3.4 run_defaults.yaml - Training Configuration

#### 3.4.1 Scenario Definitions

```yaml
scenarios:
  scenario1:
    name: "Scenario 1 - No Post-Entry Actuals"
    forecast_start: 0
    forecast_end: 23
    feature_cutoff: 0
    target_months: [0, 1, 2, ..., 23]
    
  scenario2:
    name: "Scenario 2 - First 6 Months Available"
    forecast_start: 6
    forecast_end: 23
    feature_cutoff: 6
    target_months: [6, 7, 8, ..., 23]
```

#### 3.4.2 Official Metric Weights

```yaml
official_metric:
  bucket_threshold: 0.25    # Bucket 1 if mean_erosion <= 0.25
  
  bucket_weights:
    bucket1: 2.0            # High erosion - 2x weight
    bucket2: 1.0            # Low erosion - 1x weight
  
  metric1:                  # Scenario 1
    monthly_weight: 0.2
    accumulated_0_5_weight: 0.5    # CRITICAL: 50% weight
    accumulated_6_11_weight: 0.2
    accumulated_12_23_weight: 0.1
  
  metric2:                  # Scenario 2
    monthly_weight: 0.2
    accumulated_6_11_weight: 0.5   # CRITICAL: 50% weight
    accumulated_12_23_weight: 0.3
```

#### 3.4.3 Sample Weights (Training Loss Alignment)

Sample weights align training loss with official metric importance:

```yaml
sample_weights:
  scenario1:
    months_0_5: 3.0     # Highest priority (50% of metric)
    months_6_11: 1.5    # Medium priority (20% of metric)
    months_12_23: 1.0   # Lower priority (10% of metric)
    
  scenario2:
    months_6_11: 2.5    # Highest priority (50% of metric)
    months_12_23: 1.0   # Medium priority (30% of metric)
    
  bucket_weights:
    bucket1: 2.0        # High erosion - 2x weight
    bucket2: 1.0        # Low erosion - 1x weight
```

#### 3.4.4 Validation Configuration

```yaml
validation:
  val_fraction: 0.2          # 20% of series for validation
  stratify_by: "bucket"      # Stratify by bucket
  split_level: "series"      # CRITICAL: split at series level
```

⚠️ **CRITICAL**: `split_level: "series"` ensures that all rows from a series go to either train or validation, never split across. This prevents data leakage.

---

## 4. Data Pipeline

### 4.1 Data Loading Flow

```
Raw CSV Files → load_raw_data() → prepare_base_panel() → compute_pre_entry_stats() 
                                                        → handle_missing_values() → Panel
```

### 4.2 Key Functions in src/data.py

#### 4.2.1 load_raw_data()

```python
def load_raw_data(config: dict, split: str = "train") -> Dict[str, pd.DataFrame]:
    """
    Load all three datasets for train or test split.
    
    Args:
        config: Loaded data.yaml config dict
        split: "train" or "test"
    
    Returns:
        Dictionary with keys: 'volume', 'generics', 'medicine_info'
    
    Example:
        config = load_config('configs/data.yaml')
        train_data = load_raw_data(config, split='train')
    """
```

#### 4.2.2 prepare_base_panel()

```python
def prepare_base_panel(
    volume_df: pd.DataFrame,
    generics_df: pd.DataFrame,
    medicine_info_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create unified panel with all features joined.
    
    Joins:
    - volume <- generics on (country, brand_name, months_postgx)
    - result <- medicine_info on (country, brand_name)
    
    Returns:
        Panel DataFrame with all columns merged
    """
```

#### 4.2.3 compute_pre_entry_stats()

```python
def compute_pre_entry_stats(
    panel_df: pd.DataFrame,
    is_train: bool = True,
    bucket_threshold: float = 0.25
) -> pd.DataFrame:
    """
    Compute pre-entry statistics for each series.
    
    Always computes:
    - avg_vol_12m: Mean volume over months [-12, -1]
    - pre_entry_months_available: Count of pre-entry months
    
    If is_train=True, also computes:
    - y_norm: volume / avg_vol_12m
    - mean_erosion: Mean y_norm over months 0-23
    - bucket: 1 if mean_erosion <= threshold else 2
    """
```

**Fallback Hierarchy for avg_vol_12m**:
1. Mean volume from months [-12, -1]
2. Mean volume from ANY pre-entry months (< 0)
3. Median by therapeutic area
4. Global median

#### 4.2.4 get_panel() - Main Entry Point

```python
def get_panel(
    split: str,
    config: dict,
    use_cache: bool = True,
    force_rebuild: bool = False
) -> pd.DataFrame:
    """
    Get panel DataFrame with caching support.
    
    Primary entry point for obtaining panels.
    
    Cache location: {interim_dir}/panel_{split}.parquet
    
    Example:
        config = load_config('configs/data.yaml')
        train_panel = get_panel('train', config)
        test_panel = get_panel('test', config, force_rebuild=True)
    """
```

### 4.3 Panel Schema

After `get_panel()`, the panel contains:

**Training Panel Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `country` | category | Country identifier |
| `brand_name` | category | Brand identifier |
| `month` | category | Calendar month (Jan-Dec) |
| `months_postgx` | int | Months since generic entry |
| `volume` | float | Raw volume |
| `n_gxs` | int | Number of generic competitors |
| `ther_area` | category | Therapeutic area |
| `hospital_rate` | float | Hospital usage percentage |
| `main_package` | category | Dosage form |
| `biological` | bool | Is biological drug |
| `small_molecule` | bool | Is small molecule |
| `hospital_rate_missing` | int | Missing flag for hospital_rate |
| `biological_missing` | int | Missing flag |
| `small_molecule_missing` | int | Missing flag |
| `avg_vol_12m` | float | Pre-entry average volume |
| `pre_entry_months_available` | int | Count of pre-entry months |
| `y_norm` | float | Normalized target (train only) |
| `mean_erosion` | float | Mean y_norm (train only) |
| `bucket` | int | Bucket 1 or 2 (train only) |

### 4.4 Data Validation Functions

#### validate_panel_schema()
```python
def validate_panel_schema(panel_df, split="train", raise_on_error=True):
    """
    Validate panel has required columns and no duplicate keys.
    
    Checks:
    1. Required columns present
    2. No duplicate (country, brand_name, months_postgx) keys
    3. Train-specific columns (bucket, mean_erosion) for train split
    4. No target columns in test split
    """
```

#### verify_no_future_leakage()
```python
def verify_no_future_leakage(panel_df, scenario, cutoff_column='months_postgx'):
    """
    Verify no features use data beyond scenario cutoff.
    
    Scenario 1: cutoff = 0 (only pre-entry data)
    Scenario 2: cutoff = 6 (pre-entry + first 6 months)
    """
```

#### audit_data_leakage()
```python
def audit_data_leakage(feature_df, scenario, mode="train", strict=True):
    """
    Systematically audit feature DataFrame for data leakage.
    
    Checks:
    1. No target-derived columns (bucket, mean_erosion, y_norm, volume)
    2. No ID columns as features (country, brand_name)
    3. No early erosion features in Scenario 1
    4. No suspicious column names (_test_, _future_, _target_)
    """
```

---

## 5. Feature Engineering

### 5.1 Feature Categories

The feature engineering module (`src/features.py`) creates features in these categories:

| Category | Function | Description |
|----------|----------|-------------|
| Pre-entry | `add_pre_entry_features()` | Statistics from months < 0 |
| Time | `add_time_features()` | Time index and decay features |
| Generics | `add_generics_features()` | Competition features |
| Drug | `add_drug_features()` | Static drug characteristics |
| Early Erosion | `add_early_erosion_features()` | Scenario 2 only: months 0-5 |
| Interactions | `add_interaction_features()` | Feature interactions |
| Target Encoding | `add_target_encoding_features()` | K-fold target encoding |

### 5.2 Scenario-Aware Feature Construction

Features respect **scenario-specific cutoffs**:

| Scenario | Feature Cutoff | Target Range | Allowed Feature Data |
|----------|----------------|--------------|---------------------|
| Scenario 1 | 0 | months 0-23 | Only months_postgx < 0 |
| Scenario 2 | 6 | months 6-23 | months_postgx < 6 |

### 5.3 Key Functions

#### make_features() - Main Entry Point

```python
def make_features(
    panel_df: pd.DataFrame,
    scenario,           # 1, 2, "scenario1", or "scenario2"
    mode: str = "train",# "train" or "test"
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Build features respecting scenario-specific information constraints.
    
    CRITICAL - y_norm Creation:
        if mode == "train":
            df['y_norm'] = df['volume'] / df['avg_vol_12m']
    
    Returns:
        DataFrame with engineered features.
        If mode="train", includes target column (y_norm).
    """
```

#### select_training_rows()

```python
def select_training_rows(panel_df, scenario) -> pd.DataFrame:
    """
    Select only rows that are valid supervised targets.
    
    Scenario 1: months_postgx in [0, 23]
    Scenario 2: months_postgx in [6, 23]
    
    ALWAYS call this before splitting into features/target.
    """
```

#### get_features() - Cached Feature Loading

```python
def get_features(
    split: str,
    scenario: int,
    mode: str = "train",
    data_config: dict = None,
    features_config: dict = None,
    use_cache: bool = True,
    force_rebuild: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Get features with caching support.
    
    Returns:
        (X, y, meta_df) tuple:
        - X: Feature matrix (no meta columns)
        - y: Target series (y_norm) if mode="train", else None
        - meta_df: Meta columns for reconstruction
    """
```

### 5.4 Feature List (Default Configuration)

#### Pre-Entry Features (~15 features)
- `avg_vol_3m`, `avg_vol_6m`, `avg_vol_12m` - Rolling averages
- `log_avg_vol_12m` - Log-transformed average
- `pre_entry_trend` - Linear slope
- `pre_entry_volatility` - Coefficient of variation
- `pre_entry_max`, `pre_entry_min` - Extremes
- `pre_entry_range` - Max - Min
- `pre_entry_months_available` - Data availability

#### Time Features (~12 features)
- `months_postgx` - Direct time index
- `months_postgx_sq` - Squared term
- `is_post_entry` - Binary flag
- `time_bucket` - Categorical (pre, early, mid, late)
- `month_sin`, `month_cos` - Cyclical encoding
- `quarter_1`, `quarter_2`, `quarter_3`, `quarter_4` - Quarter flags
- `time_decay` - exp(-alpha * months)

#### Generics Features (~10 features)
- `n_gxs` - Current generic count
- `has_generic` - Binary (n_gxs > 0)
- `multiple_generics` - Binary (n_gxs >= 2)
- `n_gxs_cummax` - Maximum to date
- `first_gx_month` - Entry timing
- `gx_entry_speed` - Entry rate
- `log_n_gxs` - Log-transformed count
- `n_gxs_bin` - Binned categories

#### Drug Features (~15 features)
- `ther_area_encoded` - Label encoded
- `main_package_encoded` - Label encoded
- `hospital_rate` - Numeric
- `hospital_rate_bin` - Low/Medium/High
- `biological` - Boolean
- `small_molecule` - Boolean
- `is_injection` - Derived from package
- Missing flags for each

#### Scenario 2 Only (~8 features)
- `avg_vol_0_5` - Mean volume months 0-5
- `erosion_0_5` - Early erosion ratio
- `trend_0_5` - Early trend slope
- `drop_month_0` - Initial drop
- `recovery_signal` - Recovery indicator
- `competition_response` - Response to generics
- `erosion_per_generic` - Erosion per competitor

### 5.5 Leakage Prevention

The system has multiple layers of leakage prevention:

1. **Forbidden Features Set**: Columns that can never be features
2. **Cutoff Rules**: Scenario-specific data visibility
3. **Audit Functions**: Pre-training validation
4. **Meta Column Separation**: Features vs. meta split

---

## 6. Model Architecture

### 6.1 BaseModel Interface

All models implement the `BaseModel` abstract class:

```python
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
        """Train the model."""
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance if available."""
        return pd.DataFrame(columns=['feature', 'importance'])
```

### 6.2 Model Implementations

#### 6.2.1 CatBoost (Primary Model)

**File**: `src/models/cat_model.py`

```python
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
    }
```

**Key Features**:
- Native categorical feature handling (no encoding needed)
- Automatic early stopping
- Sample weight support via CatBoost Pool

**Best Hyperparameters** (from sweeps):
- Scenario 1: depth=6, learning_rate=0.03, l2_leaf_reg=3.0
- Scenario 2: depth=6, learning_rate=0.03, l2_leaf_reg=3.0

#### 6.2.2 XGBoost

**File**: `src/models/xgb_model.py`

```python
class XGBModel(BaseModel):
    """XGBoost model with native sample weight support via DMatrix."""
    
    DEFAULT_CONFIG = {
        'params': {
            'objective': 'reg:squarederror',
            'eta': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'training': {
            'num_boost_round': 2000,
            'early_stopping_rounds': 100,
        }
    }
```

**Note**: XGBoost may segfault on Apple Silicon M-series chips.

#### 6.2.3 LightGBM

**File**: `src/models/lgbm_model.py`

```python
class LGBMModel(BaseModel):
    """LightGBM implementation optimized for speed."""
    
    DEFAULT_CONFIG = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 1000,
    }
```

**Note**: LightGBM may segfault on Apple Silicon M-series chips.

#### 6.2.4 Neural Network

**File**: `src/models/nn.py`

```python
class NNModel(BaseModel):
    """PyTorch MLP for regression."""
    
    DEFAULT_CONFIG = {
        'architecture': {
            'hidden_layers': [256, 128, 64],
            'dropout': 0.3,
            'activation': 'relu',
            'batch_norm': True
        },
        'training': {
            'batch_size': 256,
            'epochs': 100,
            'learning_rate': 1e-3,
            'early_stopping_patience': 10,
        }
    }
```

#### 6.2.5 Linear Models

**File**: `src/models/linear.py`

```python
class LinearModel(BaseModel):
    """Ridge/Lasso/ElasticNet with preprocessing."""
    
    DEFAULT_CONFIG = {
        'model_type': 'ridge',  # ridge, lasso, elasticnet, huber
        'alpha': 1.0,
        'use_polynomial': False,
        'polynomial_degree': 2,
    }
```

#### 6.2.6 Baseline Models

**File**: `src/models/linear.py`

| Model | Description |
|-------|-------------|
| `FlatBaseline` | Predict y_norm = 1.0 (no erosion) |
| `TrendBaseline` | Extrapolate pre-entry trend |
| `GlobalMeanBaseline` | Predict global average erosion curve |
| `HistoricalCurveBaseline` | Match to similar historical series |

#### 6.2.7 Hybrid Physics + ML

**File**: `src/models/hybrid_physics_ml.py`

```python
class HybridPhysicsMLModel:
    """
    Combines physics-based decay with ML residual learning.
    
    final_pred = physics_baseline + ML_residual
    physics_baseline = exp(-decay_rate * months_postgx)
    """
```

#### 6.2.8 ARIHOW (ARIMA + Holt-Winters)

**File**: `src/models/arihow.py`

```python
class ARIHOWModel:
    """
    ARIMA + Holt-Winters hybrid for time-series forecasting.
    
    For each series:
    1. Fit ARIMA for trends/autocorrelation
    2. Fit Holt-Winters for smoothing
    3. Learn optimal combination weights
    
    Falls back to exponential decay for short series.
    """
```

### 6.3 Ensemble Methods

**File**: `src/models/ensemble.py`

| Ensemble | Description |
|----------|-------------|
| `AveragingEnsemble` | Simple mean of predictions |
| `WeightedAveragingEnsemble` | Weighted average (optimized weights) |
| `StackingEnsemble` | Two-level stacking with meta-learner |
| `BlendingEnsemble` | Blending with holdout predictions |

---

## 7. Training Pipeline

### 7.1 Training Command

```bash
# Basic training
python -m src.train --scenario 1 --model catboost

# With custom config
python -m src.train --scenario 1 --model catboost --model-config configs/model_cat.yaml

# All models
python -m src.train --scenario 1 --all-models

# Hyperparameter sweep
python -m src.train --scenario 1 --model catboost --sweep
```

### 7.2 CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--scenario` | 1 or 2 | Required |
| `--model` | Model type (catboost, lightgbm, xgboost, nn, linear, hybrid, arihow) | Required |
| `--model-config` | Path to model config YAML | Auto-detected |
| `--data-config` | Path to data config | configs/data.yaml |
| `--run-config` | Path to run config | configs/run_defaults.yaml |
| `--sweep` | Enable hyperparameter sweep | False |
| `--config-id` | Specific config ID from sweep_configs | None |
| `--seed` | Random seed | 42 |
| `--force-rebuild` | Force rebuild cached data | False |

### 7.3 Training Flow

```python
# 1. Load data
train_panel = get_panel('train', data_config)

# 2. Build features
features_df = make_features(train_panel, scenario=1, mode='train')

# 3. Select training rows
train_df = select_training_rows(features_df, scenario=1)

# 4. Split features/target/meta
X, y, meta_df = get_feature_matrix_and_meta(train_df)

# 5. Create train/val split (series-level)
X_train, X_val, y_train, y_val = create_validation_split(...)

# 6. Compute sample weights
weights = compute_sample_weights(meta_df, scenario=1, run_config)

# 7. Train model
model.fit(X_train, y_train, X_val, y_val, sample_weight=weights)

# 8. Evaluate
metrics = evaluate_model(model, X_val, y_val, meta_val)

# 9. Save artifacts
save_model_artifacts(model, metrics, run_id)
```

### 7.4 Sample Weight Computation

```python
def compute_sample_weights(meta_df, scenario, run_config):
    """
    Compute sample weights to align training with official metric.
    
    Weights are product of:
    1. Time-period weights (early months weighted more)
    2. Bucket weights (bucket 1 weighted 2x)
    """
    # Get config weights
    sw_config = run_config['sample_weights'][f'scenario{scenario}']
    bucket_weights = run_config['sample_weights']['bucket_weights']
    
    # Time weights
    if scenario == 1:
        time_weight = meta_df['months_postgx'].apply(
            lambda m: sw_config['months_0_5'] if m <= 5
                      else sw_config['months_6_11'] if m <= 11
                      else sw_config['months_12_23']
        )
    else:  # scenario 2
        time_weight = meta_df['months_postgx'].apply(
            lambda m: sw_config['months_6_11'] if m <= 11
                      else sw_config['months_12_23']
        )
    
    # Bucket weights
    bucket_weight = meta_df['bucket'].map(bucket_weights)
    
    return time_weight * bucket_weight
```

### 7.5 Artifacts Directory Structure

Each training run creates:

```
artifacts/{timestamp}_{model}_{scenario}/
├── model_1.bin              # Saved model (Scenario 1)
│   or model_2.bin           # Saved model (Scenario 2)
├── metadata.json            # Run metadata
│   ├── timestamp
│   ├── git_commit
│   ├── scenario
│   ├── model_type
│   ├── config_hash
│   ├── dataset (sizes)
│   └── configs (full snapshots)
├── metrics.json             # Evaluation metrics
│   ├── official_metric
│   ├── rmse
│   ├── mae
│   └── per_series_metrics
├── feature_importance.csv   # Feature importances
├── config_snapshot.yaml     # Config at training time
└── train.log               # Training log
```

### 7.6 Experiment Tracking

The pipeline supports MLflow and W&B for experiment tracking:

```python
class ExperimentTracker:
    """Unified tracking interface for MLflow and W&B."""
    
    def start_run(self, run_name, tags, config): ...
    def log_params(self, params): ...
    def log_metrics(self, metrics, step): ...
    def log_artifact(self, path): ...
    def end_run(self): ...
```

Enable via `run_defaults.yaml`:
```yaml
experiment_tracking:
  enabled: true
  backend: "mlflow"  # or "wandb"
```

---

## 8. Inference & Submission

### 8.1 Submission Command

```bash
python -m src.inference \
  --model-s1 artifacts/{run_s1}/model_1.bin \
  --model-s2 artifacts/{run_s2}/model_2.bin \
  --use-versioning \
  --run-name catboost_final
```

### 8.2 Submission Format

The competition expects:

```csv
country,brand_name,months_postgx,volume
COUNTRY_9891,BRAND_3C69,0,239066.98
COUNTRY_9891,BRAND_3C69,1,230014.58
...
```

| Column | Description |
|--------|-------------|
| `country` | Country identifier |
| `brand_name` | Brand identifier |
| `months_postgx` | Month (0-23 for S1, 6-23 for S2) |
| `volume` | Predicted volume (inverse-transformed) |

### 8.3 Scenario Detection

```python
def detect_test_scenarios(test_volume):
    """
    Identify which test series belong to Scenario 1 vs 2.
    
    Detection rules:
    - Scenario 1: Series starting at months_postgx = 0
    - Scenario 2: Series starting at months_postgx = 6 (has months 0-5 actuals)
    
    Expected counts: 228 S1, 112 S2
    """
```

### 8.4 Prediction Flow

```python
def generate_submission(model_s1, model_s2, test_panel, template):
    """
    1. Detect which series are S1 vs S2
    2. For each scenario:
       a. Expand panel to include future months
       b. Build features
       c. Generate predictions (y_norm)
       d. Inverse transform: volume = y_norm * avg_vol_12m
    3. Validate against template
    """
```

### 8.5 Post-Processing

```python
# Clip negative predictions
submission['volume'] = submission['volume'].clip(lower=0)

# Handle edge cases
submission = handle_zero_volume_series(submission, test_panel)
submission = handle_extreme_predictions(submission, test_panel, max_ratio=2.0)
```

### 8.6 Submission Validation

```python
def validate_submission_format(submission_df, template_df):
    """
    Checks:
    1. Row count matches template
    2. Correct columns: country, brand_name, months_postgx, volume
    3. No missing values
    4. No negative volumes
    5. Keys match template exactly
    6. No duplicates
    """
```

---

## 9. Evaluation Metrics

### 9.1 Official Metric Computation

**File**: `src/evaluate.py`

#### Metric 1 (Scenario 1)

$$PE_1 = 0.2 \cdot \frac{\sum_{t=0}^{23} |y_t - \hat{y}_t|}{24 \cdot \bar{v}} + 0.5 \cdot \frac{|\sum_{t=0}^{5} (y_t - \hat{y}_t)|}{6 \cdot \bar{v}} + 0.2 \cdot \frac{|\sum_{t=6}^{11} (y_t - \hat{y}_t)|}{6 \cdot \bar{v}} + 0.1 \cdot \frac{|\sum_{t=12}^{23} (y_t - \hat{y}_t)|}{12 \cdot \bar{v}}$$

#### Metric 2 (Scenario 2)

$$PE_2 = 0.2 \cdot \frac{\sum_{t=6}^{23} |y_t - \hat{y}_t|}{18 \cdot \bar{v}} + 0.5 \cdot \frac{|\sum_{t=6}^{11} (y_t - \hat{y}_t)|}{6 \cdot \bar{v}} + 0.3 \cdot \frac{|\sum_{t=12}^{23} (y_t - \hat{y}_t)|}{12 \cdot \bar{v}}$$

#### Bucket-Weighted Final Score

$$Score = \frac{2}{n_1} \sum_{i \in B_1} PE_i + \frac{1}{n_2} \sum_{i \in B_2} PE_i$$

Where $B_1$ is high-erosion bucket (mean_erosion ≤ 0.25), $B_2$ is low-erosion bucket.

### 9.2 Key Functions

```python
def compute_metric1(df_actual, df_pred, df_aux):
    """Compute official Metric 1 for Scenario 1."""

def compute_metric2(df_actual, df_pred, df_aux):
    """Compute official Metric 2 for Scenario 2."""

def create_aux_file(panel_df, output_path=None):
    """Create auxiliary file with avg_vol and bucket per series."""
```

### 9.3 Additional Metrics

| Metric | Description |
|--------|-------------|
| RMSE (y_norm) | Root Mean Squared Error on normalized target |
| MAE (y_norm) | Mean Absolute Error on normalized target |
| MAPE (y_norm) | Mean Absolute Percentage Error |
| Per-series MAE | MAE computed per series |
| Bucket-level metrics | Metrics split by bucket |

---

## 10. API Reference

### 10.1 Data Module (src/data.py)

| Function | Purpose |
|----------|---------|
| `load_raw_data(config, split)` | Load CSV files |
| `prepare_base_panel(vol, gen, med)` | Join datasets |
| `compute_pre_entry_stats(panel, is_train)` | Compute avg_vol, bucket |
| `handle_missing_values(panel)` | Fill missing values |
| `get_panel(split, config)` | Main entry point (cached) |
| `validate_panel_schema(panel)` | Schema validation |
| `audit_data_leakage(features, scenario)` | Leakage detection |

### 10.2 Features Module (src/features.py)

| Function | Purpose |
|----------|---------|
| `make_features(panel, scenario, mode)` | Build features |
| `select_training_rows(panel, scenario)` | Filter target rows |
| `get_features(split, scenario, mode)` | Get features (cached) |
| `add_pre_entry_features(df, config)` | Pre-entry stats |
| `add_time_features(df, config)` | Time features |
| `add_generics_features(df, cutoff, config)` | Competition features |
| `add_drug_features(df, config)` | Drug characteristics |
| `add_early_erosion_features(df, config)` | S2-only features |

### 10.3 Training Module (src/train.py)

| Function | Purpose |
|----------|---------|
| `get_feature_matrix_and_meta(df)` | Split X/y/meta |
| `compute_sample_weights(meta, scenario)` | Weight computation |
| `train_model(X, y, model, config)` | Training wrapper |
| `evaluate_model(model, X, y, meta)` | Compute all metrics |
| `save_model_artifacts(model, metrics, run_id)` | Save to artifacts |

### 10.4 Inference Module (src/inference.py)

| Function | Purpose |
|----------|---------|
| `predict_batch(model, X, batch_size)` | Batch prediction |
| `detect_test_scenarios(test_volume)` | Scenario detection |
| `generate_submission(model_s1, model_s2, panel, template)` | Main submission |
| `validate_submission_format(submission, template)` | Validation |
| `save_submission_with_versioning(submission, output_dir)` | Versioned save |

### 10.5 Model Classes

| Class | Module | Description |
|-------|--------|-------------|
| `BaseModel` | base.py | Abstract interface |
| `CatBoostModel` | cat_model.py | CatBoost wrapper |
| `LGBMModel` | lgbm_model.py | LightGBM wrapper |
| `XGBModel` | xgb_model.py | XGBoost wrapper |
| `NNModel` | nn.py | PyTorch MLP |
| `LinearModel` | linear.py | Ridge/Lasso |
| `FlatBaseline` | linear.py | y_norm = 1 baseline |
| `GlobalMeanBaseline` | linear.py | Global mean curve |
| `HybridPhysicsMLModel` | hybrid_physics_ml.py | Physics + ML |
| `ARIHOWModel` | arihow.py | ARIMA + HW |

---

## 11. Best Practices & Guidelines

### 11.1 Data Leakage Prevention

1. **Never use meta columns as features**: `country`, `brand_name`, `months_postgx`, `bucket`, `y_norm`, `volume`, `mean_erosion`, `month`

2. **Respect scenario cutoffs**:
   - Scenario 1: Only `months_postgx < 0` for features
   - Scenario 2: Only `months_postgx < 6` for features

3. **Series-level validation splits**: Never split rows from the same series across train/val

4. **Run leakage audits**: Call `audit_data_leakage()` before training

### 11.2 Model Training

1. **Use sample weights**: Align training loss with official metric importance
   - Early months (0-5 or 6-11) get higher weights
   - Bucket 1 gets 2x weight

2. **Early stopping**: Prevent overfitting with validation-based stopping

3. **Categorical handling**:
   - CatBoost: Use native Pool with cat_features
   - Others: Use label encoding

4. **Save all artifacts**: Model, config, metrics for reproducibility

### 11.3 Feature Engineering

1. **Pre-entry features are key**: Most predictive features come from pre-generic-entry period

2. **Time features for decay**: Include polynomial and exponential decay terms

3. **Generics count is exogenous**: Future n_gxs is given in test data (not leakage)

4. **Scenario 2 early erosion**: Use months 0-5 actuals when available

### 11.4 Submission

1. **Validate format**: Match template exactly (columns, order, row count)

2. **Clip predictions**: No negative volumes, reasonable upper bounds

3. **Inverse transform correctly**: `volume = y_norm * avg_vol_12m`

4. **Version submissions**: Use timestamped directories with metadata

---

## 12. Troubleshooting

### 12.1 Common Issues

#### Issue: "META_COLS mismatch between code and config"
**Cause**: `data.yaml` and `src/data.py` have different meta_cols lists
**Solution**: Ensure both are synchronized

#### Issue: "Panel schema validation failed"
**Cause**: Missing required columns after panel construction
**Solution**: Check data files exist and have expected columns

#### Issue: "Data leakage detected"
**Cause**: Features include target-derived or future information
**Solution**: Review feature engineering, run `audit_data_leakage()`

#### Issue: "Segmentation fault" (LightGBM/XGBoost on Apple Silicon)
**Cause**: Known issue with GBDT libraries on M-series chips
**Solution**: Use CatBoost, or run on Intel/Linux

#### Issue: "No pre-entry data for series"
**Cause**: Series lacks months -12 to -1
**Solution**: Fallback hierarchy handles this automatically

### 12.2 Performance Optimization

1. **Use cached panels**: Enable `use_cache=True` in `get_panel()`
2. **Parquet format**: Faster than CSV for repeated loads
3. **Batch predictions**: Use `predict_batch()` for large datasets
4. **Optimize dtypes**: Category columns save memory

### 12.3 Debugging Tips

1. **Enable verbose logging**: Set logging level to DEBUG
2. **Check metadata.json**: Contains full config snapshot
3. **Review feature_importance.csv**: Identify key predictors
4. **Validate intermediate outputs**: Check panel shapes at each step

---

## Appendix: Configuration Hash

The `config_hash` in metadata.json is a SHA256 hash of the full configuration, used for reproducibility:

```python
import hashlib
import json

config_str = json.dumps(full_config, sort_keys=True)
config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
```

This ensures that runs with identical configurations can be identified and compared.

---

*Documentation generated for Novartis Datathon 2025 codebase. Last updated: November 2025.*
