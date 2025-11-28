# Novartis Datathon 2025 - Comprehensive TODO List

> **Competition Goal**: Forecast generic erosion (normalized volume) for pharmaceutical drugs  
> **Deadline**: Phase 1A/1B submission deadline (check competition site)  
> **Hero Model**: CatBoost with scenario-specific sample weights  
>
> ⚠️ **When in doubt, complete Priority A items (CatBoost + features + validation + submission) before touching Priority B.**

### 📅 Implementation Phases

| Phase | Focus | Sections |
|-------|-------|----------|
| **Phase 1** (before first stable LB submission) | Core pipeline, baseline model, first submission | 0–7 (Priority A only), 9.1, 9.4, 10.1–10.2, 11.1–11.3, 12.1–12.4 |
| **Phase 2** (after solid LB score) | Advanced features, unified logging, plotting, CV search, ensemble tuning | All remaining unchecked items (6.7–6.9, 8.*, booster models, HPO, TabNet/FT-Transformer) |

### 🔧 Canonical CLI Flags (Locked)

| Flag | Format | Example |
|------|--------|--------|
| `--scenario` | Integer `1` or `2` | `--scenario 1` |
| `--model` | Lowercase string | `--model catboost` |
| `--split` | `train` or `test` | `--split train` |
| `--mode` | `train` or `test` | `--mode train` |
| `--data-config` | Path to YAML | `--data-config configs/data.yaml` |
| `--features-config` | Path to YAML | `--features-config configs/features.yaml` |
| `--run-config` | Path to YAML | `--run-config configs/run_defaults.yaml` |
| `--model-config` | Path to YAML | `--model-config configs/model_cat.yaml` |
| `--force-rebuild` | Flag (no value) | `--force-rebuild` |

---

## Table of Contents
0. [Design & Consistency Checks](#0-design--consistency-checks)
1. [Critical Path (Must Do)](#1-critical-path-must-do)
2. [Data Pipeline](#2-data-pipeline)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Development](#4-model-development)
5. [Training Pipeline](#5-training-pipeline)
6. [Validation & Evaluation](#6-validation--evaluation)
7. [Inference & Submission](#7-inference--submission)
8. [Experimentation & Optimization](#8-experimentation--optimization)
9. [Testing & Quality Assurance](#9-testing--quality-assurance)
10. [Documentation & Presentation](#10-documentation--presentation)
11. [Colab/Production Readiness](#11-colabproduction-readiness)
12. [Competition Strategy](#12-competition-strategy)

---

### 🧭 Recommended Implementation Order (Sections)

0 → 2 → 3 → 6 → 5 → 4 → 7 → 9 → 10 → 11 → 8 → 12

- **0 – Design & Consistency Checks**: configs, paths, CLI & docs alignment.
- **2 – Data Pipeline**: panels, caching, train/test behavior, leakage checks.
- **3 – Feature Engineering**: scenario-aware `make_features`, early-erosion features.
- **6 – Validation & Evaluation**: metrics, scenario simulation, official metric wrapper.
- **5 – Training Pipeline**: `train.py`, sample weights, run metadata, config wiring.
- **4 – Model Development**: baselines + CatBoost hero; other models after that.
- **7 – Inference & Submission**: `inference.py`, template alignment, aux file.
- **9 – Testing & QA**: unit + integration tests tightened around the full pipeline.
- **10 – Documentation & Presentation**: README, approach, methodology notes.
- **11 – Colab/Production Readiness**: Colab notebook, envs, performance tweaks.
- **8 – Experimentation & Optimization**: HPO, ensembles, ablations, smoothing.
- **12 – Competition Strategy**: LB management, final-week playbook, external-data checks.

> **Section 1 – Critical Path (Must Do)** is a *running checklist* (EDA, first LB submission, score iterations) and should be followed in parallel as soon as the minimal 0→2→3→6→5→4 path is usable.


## 0. Design & Consistency Checks

> **Priority Note**: Items directly tied to CatBoost + robust features + solid validation are **Priority A**; purely architectural items (full BaseModel hierarchy, advanced notebooks, stretch goals) are **Priority B / post-competition**.

### 0.0 Global Preparation & Repository Hygiene
- [x] **Read and internalize documentation**:
  - [x] Read `docs/planning/functionality.md` thoroughly ✅
  - [x] Read `docs/planning/approach.md` ✅
  - [x] Skim `docs/instructions/NOVARTIS_DATATHON_2025_COMPLETE_GUIDE_V2.md` and `docs/planning/question-set.md` ✅
- [x] **Ensure directory structure matches the spec**:
  - [x] Verify top-level folders exist: `data/`, `src/`, `configs/`, `docs/`, `notebooks/`, `submissions/`, `tests/`, `artifacts/` ✅
  - [x] Verify `docs/guide/metric_calculation.py` exists ✅
  - [x] Verify `docs/guide/submission_template.csv` exists ✅
  - [x] Verify `docs/guide/auxiliar_metric_computation_example.csv` exists ✅
- [x] **Verify environment files exist and are consistent**:
  - [x] `requirements.txt`, `env/colab_requirements.txt`, and/or `environment.yml` exist ✅
  - [x] Ensure requirements list all needed packages and are tested ✅ (added pyarrow, pytest; all 92 tests pass)

### 0.1 Code-Documentation Alignment
- [x] **Cross-check all items marked as ✅ (`[x]`)** against the actual codebase and revert them to ☐ (`[ ]`) if implementation is partial or differs from `functionality.md` / `README.md` ✅
- [x] **Ensure function signatures and behaviors** in code match the descriptions in `docs/functionality.md` and `README.md` (e.g. `make_features`, `compute_pre_entry_stats`, `compute_metric1`, `generate_submission`, CLI entrypoints) ✅
- [x] **Align CLI examples** in `README.md`, `TODO.md` (Quick Commands) and actual `src/train.py` / `src/inference.py` argument names (`--data-config`, `--features-config`, `--run-config`, `--model-config`, `--scenario`, `--model` etc.) ✅
  - [x] Updated README.md CLI examples to use `--scenario 1` and `--scenario 2` (integer scenarios) ✅
  - [x] Updated `src/train.py` CLI to accept integer scenarios ✅
  - [x] Updated `src/evaluate.py` examples in README to match actual function signatures ✅
- [x] **Confirm all referenced paths exist** and are correctly used in code:
  - [x] `docs/guide/metric_calculation.py` ✅
  - [x] `docs/guide/submission_template.csv` ✅
  - [x] `configs/*.yaml` (all referenced config files) ✅

### 0.2 Configuration System (`configs/`)
- [x] **Verify `configs/data.yaml` contains**:
  - [x] `paths.raw_dir`, `paths.interim_dir`, `paths.processed_dir`, `paths.artifacts_dir` ✅
  - [x] `files.train`, `files.test`, `files.aux_metric`, `files.submission_template` ✅
  - [x] `columns` section: `id_keys`, `time_key`, `calendar_month`, `raw_target`, `model_target`, `meta_cols` ✅
  - [x] Added `mean_erosion` to `meta_cols` list ✅
- [x] **Verify `configs/features.yaml` contains**:
  - [x] Sections: `pre_entry`, `time`, `generics`, `drug`, `scenario2_early`, `interactions` ✅
  - [x] Scenario-specific enable/disable flags for feature groups ✅
- [x] **Verify `configs/model_cat.yaml` contains**:
  - [x] `model_type`, `params`, `training`, `categorical_features` ✅
- [x] **Verify `configs/run_defaults.yaml` contains**:
  - [x] `reproducibility.seed`, `validation.*` settings ✅
  - [x] `scenarios.scenario1/scenario2` with `forecast_start`, `forecast_end`, `feature_cutoff` ✅
  - [x] `sample_weights` definition (time-window and bucket weights) ✅
  - [x] `logging.level`, `logging.log_to_file` ✅
- [x] **Configs as single source of truth**: ✅
  - [x] Added `official_metric` section to `configs/run_defaults.yaml` with all metric weights and bucket threshold ✅
  - [x] Updated `src/data.py compute_pre_entry_stats()` to accept `bucket_threshold` and `run_config` parameters ✅
  - [x] Added `OFFICIAL_*` constants and `validate_config_matches_official()` to `src/evaluate.py` ✅
  - [x] Documented that fallback metrics intentionally match official script (hardcoded values for exact correspondence) ✅
  - [x] Added 5 new tests for config validation (92 tests total, all pass) ✅

### 0.3 External Data & Constraints Rule-Check
- [x] **Explicit rule-check on external data and constraints** ✅
  - [x] Re-read the latest official rules on:
    - [x] Use of **external data**: Not explicitly prohibited but "Modeling Freedom: Any approach/model allowed" ✅
    - [x] Any restrictions on **model types** or **ensembles**: None - "explainability and simplicity valued" ✅
    - [x] Exact requirements for **reproducibility** and **code handover**: Required for finalists ✅
  - [x] Documented in `docs/planning/approach.md` (Appendix D.1):
    - [x] **Decision: NOT using external data** - rationale documented (time investment, ROI uncertainty, complexity) ✅
    - [x] All constraints documented (no test target leakage, deterministic seeds, code reproducibility) ✅

### 0.4 Code Fixes Applied (Implementation Log)
> **Summary**: The following fixes were applied to achieve full Section 0 compliance.

- [x] **`src/data.py`**:
  - [x] Fixed syntax error: removed extra `"""` at line 1 (module docstring) ✅
  - [x] Fixed Unicode arrow character `←` in docstring (replaced with `<-`) ✅
  - [x] Added canonical constants: `ID_COLS`, `TIME_COL`, `CALENDAR_MONTH_COL`, `RAW_TARGET_COL`, `MODEL_TARGET_COL` ✅
- [x] **`src/features.py`**:
  - [x] Changed `SCENARIO_CONFIG` keys from strings (`'scenario1'`, `'scenario2'`) to integers (`1`, `2`) ✅
  - [x] Added `_normalize_scenario()` helper to accept both int and string inputs for backward compatibility ✅
  - [x] Fixed month parsing: replaced `pd.to_datetime(df['month']).dt.month` with explicit month name mapping (`{'Jan': 1, ...}`) ✅
  - [x] Fixed FutureWarning in `groupby.apply()` by adding `include_groups=False` ✅
- [x] **`src/train.py`**:
  - [x] Updated `--scenario` CLI argument to use `type=int` ✅
  - [x] Updated imports to use `META_COLS` from `src/data.py` ✅
- [x] **`src/validation.py`**:
  - [x] Updated `simulate_scenario()` to use `_normalize_scenario()` from features.py ✅
- [x] **`src/evaluate.py`**:
  - [x] Updated `compute_bucket_metrics()` to handle integer scenarios ✅
- [x] **`src/inference.py`**:
  - [x] Updated `detect_test_scenarios()` to return `{1: [...], 2: [...]}` (integer keys) ✅
- [x] **`src/models/__init__.py`**:
  - [x] Implemented lazy imports for native-library models (LightGBM, XGBoost, CatBoost, NN) to prevent import failures when libraries are misconfigured ✅
  - [x] Linear models and baselines import eagerly (no native dependencies) ✅
- [x] **`configs/data.yaml`**:
  - [x] Added `mean_erosion` to `meta_cols` list ✅
- [x] **`tests/test_smoke.py`**:
  - [x] Complete rewrite to match actual function signatures ✅
  - [x] Updated to use integer scenarios (`scenario=1`, `scenario=2`) ✅
  - [x] Added `test_model_imports` using lazy imports (no skip needed) ✅
  - [x] Separated model interface test from import test ✅
  - [x] Added Section 0.2 config validation tests (5 new tests) ✅
  - [x] **Test results: 92 passed, 0 skipped, 0 warnings** ✅
- [x] **`tests/conftest.py`**:
  - [x] Created pytest configuration with warning filters ✅
  - [x] Filters torch/NumPy initialization warnings (external library) ✅
  - [x] Filters DataFrameGroupBy.apply FutureWarnings ✅
- [x] **`README.md`**:
  - [x] Updated CLI examples: `--scenario scenario1` → `--scenario 1` ✅
  - [x] Updated `src/evaluate.py` usage examples to match actual function signatures ✅

---

## 1. Critical Path (Must Do)

### 1.1 Immediate Priorities (Day 1-2)
- [x] **Verify data files are correctly placed** in `data/raw/TRAIN/` and `data/raw/TEST/` ✅
- [x] **Run smoke tests** to verify entire pipeline: `pytest tests/test_smoke.py -v` ✅ (87 passed, 0 skipped, 0 warnings)
- [ ] **Run EDA notebook** (`notebooks/00_eda.ipynb`) to understand data distributions
- [ ] **Establish baseline score** using `FlatBaseline` or `GlobalMeanBaseline`
- [ ] **Verify submission format** matches `docs/guide/submission_template.csv`

### 1.2 First Submission Target (Day 2-3)
- [ ] **Train CatBoost Scenario 1 model** with default hyperparameters
- [ ] **Train CatBoost Scenario 2 model** with default hyperparameters
- [ ] **Generate first submission file** for Phase 1A (Scenario 1)
- [ ] **Generate first submission file** for Phase 1B (Scenario 2)
- [ ] **Validate submission using** `metric_calculation.py` from docs/guide/
- [ ] **Submit to leaderboard** and record baseline score

### 1.3 Score Improvement Iteration (Day 3-7)
- [ ] **Analyze feature importance** to identify key predictors
- [ ] **Tune hyperparameters** using Optuna (focus on CatBoost)
- [ ] **Implement ensemble** of top 2-3 models
- [ ] **Add advanced features** based on EDA insights
- [ ] **Re-submit** and track improvement

---

## 2. Data Pipeline

### 2.1 Data Loading (`src/data.py`)
- [x] `load_raw_data()` - Load train/test CSV files ✅
- [x] `prepare_base_panel()` - Merge volume, generics, medicine info ✅
- [x] `compute_pre_entry_stats()` - Calculate avg_vol_12m, y_norm ✅
- [x] `handle_missing_values()` - Imputation strategies ✅
- [x] **Add validation** for expected column types and ranges ✅
  - [x] Added `validate_dataframe_schema()` for required columns ✅
  - [x] Added `validate_value_ranges()` with `VALUE_RANGES` dict ✅
  - [x] Added `validate_no_duplicates()` for key uniqueness ✅
- [x] **Add logging** for data loading statistics (rows, missing %) ✅
  - [x] Added `log_data_statistics()` helper function ✅
  - [x] Logs shape, missing value percentages, unique series count, time range ✅
- [x] **Handle edge cases**: Series with < 12 pre-entry months ✅
  - [x] Fallback 1: Use any available pre-entry months (months_postgx < 0) ✅
  - [x] Fallback 2: Use ther_area median ✅
  - [x] Fallback 3: Use global median (last resort) ✅
  - [x] Added `pre_entry_months_available` column for diagnostics ✅
- [x] **Add data caching** to speed up repeated loads (pickle/parquet) ✅
  - [x] Added `get_panel()` function with caching support ✅
  - [x] Parquet format if available, else pickle fallback ✅
  - [x] Added `clear_panel_cache()` utility ✅
  - [x] Added `_optimize_dtypes()` for storage efficiency ✅
- [x] **Define canonical constants** for id/time columns consistent with `configs/data.yaml` ✅
  - [x] `ID_COLS`, `TIME_COL`, `CALENDAR_MONTH_COL`, `RAW_TARGET_COL`, `MODEL_TARGET_COL` ✅
  - [x] `EXPECTED_DTYPES` and `VALUE_RANGES` for validation ✅

### 2.2 Train vs Test Behavior for `compute_pre_entry_stats()`
- [x] **Verify `is_train=True` behavior**: computes `y_norm`, `mean_erosion`, and `bucket` ✅
- [x] **Verify `is_train=False` behavior**: does **not** compute any target-dependent statistics (no `bucket`, no `mean_erosion`) and only computes `avg_vol_12m` ✅
- [x] **Implement robust fallback for `avg_vol_12m`** on test series with insufficient pre-entry history (e.g. global/ther_area-level averages) ✅
  - [x] 3-level fallback hierarchy: any pre-entry -> ther_area median -> global median ✅
- [x] **Document the train vs test behavior** in `docs/functionality.md` ✅
- [x] **Log distribution of `avg_vol_12m`** and bucket counts on train for sanity checking ✅
  - [x] Logs range, median, mean for avg_vol_12m ✅
  - [x] Logs bucket distribution with counts and percentages ✅
- [x] **Bucket definition**: Compute `bucket` as defined in `docs/functionality.md` and ensure code + configs implement the same rule ✅
  - [x] Bucket 1: mean_erosion <= 0.25 (high erosion) ✅
  - [x] Bucket 2: mean_erosion > 0.25 (low erosion) ✅

### 2.3 META_COLS Consistency
- [x] **Add a check that `META_COLS`** in `src/data.py` matches `columns.meta_cols` in `configs/data.yaml` ✅
  - [x] Added `verify_meta_cols_consistency()` function ✅
  - [x] Added test `test_meta_cols_consistency_check` ✅
- [x] **Update both files** if there is any mismatch ✅ (aligned in Section 0)

### 2.4 Data Leakage Audits
- [x] **Implement systematic leakage audit** that confirms no feature uses: ✅
  - [x] Future `volume` values beyond the scenario cutoff ✅
  - [x] `bucket`, `mean_erosion`, or any other target-derived statistic ✅
  - [x] Test-set statistics (global means, encodings) computed using test data ✅
- [x] **Add automated leakage check** to run before training ✅
  - [x] `audit_data_leakage()` function in `src/data.py` ✅
  - [x] `run_pre_training_leakage_check()` function in `src/data.py` ✅
  - [x] LEAKAGE_COLUMNS constant for forbidden columns ✅

### 2.5 Data Quality Checks
- [x] **Verify no future leakage**: Features only use data before cutoff ✅
  - [x] `verify_no_future_leakage()` function in `src/data.py` ✅
  - [x] Called during `get_features()` to validate feature matrices ✅
- [x] **Check for duplicate rows** in panel ✅
  - [x] `validate_no_duplicates()` raises ValueError if duplicates detected ✅
  - [x] Called in `prepare_base_panel()` ✅
- [x] **Validate date continuity**: No gaps in months_postgx per series ✅
  - [x] `validate_date_continuity()` function in `src/data.py` ✅
  - [x] `get_series_month_coverage()` function for coverage statistics ✅
- [x] **Profile missing values** per column per scenario ✅ (via `log_data_statistics()`)
- [x] **Check target distribution**: y_norm should be mostly 0-1.5 ✅
  - [x] `VALUE_RANGES['y_norm'] = (0, 5)` with warnings for outliers ✅

### 2.6 Data Splits
- [x] **Implement time-based CV split** for more realistic validation ✅
  - [x] `create_temporal_cv_split()` function in `src/validation.py` ✅
- [x] **Stratify by bucket AND therapeutic area** for balanced folds ✅
  - [x] `create_validation_split()` supports `stratify_by` list of columns ✅
- [x] **Create holdout set** from training data for final validation ✅
  - [x] `create_holdout_set()` function in `src/validation.py` ✅
- [x] **Document split rationale** in approach.md ✅

### 2.7 Utility Functions (`src/utils.py`)
- [x] **`set_seed(seed)`**: Sets Python, NumPy (and torch if available) RNGs for reproducibility ✅
- [x] **`setup_logging(level, log_file)`**: Configures the root logger, avoids duplicate handlers ✅
- [x] **`timer(name)`**: Context manager that logs start/end and elapsed seconds ✅
- [x] **`load_config(path)`**: Loads YAML, raises clear errors if missing/invalid ✅
- [x] **`get_path(config, key)`**: Helper returning `Path` objects from config paths ✅

### 2.8 Persisted Data Build (raw → interim → processed)

> **Goal**: Have a **single, well-defined data build pipeline** that:
> * Reads CSVs from `raw/`
> * Builds cleaned panels and saves them to `interim/`
> * Builds scenario-specific feature matrices and saves them to `processed/`
> * **Without creating extra modules** (only extend `src/data.py`, `src/features.py`, and configs)

#### 2.8.1 Align Paths in `configs/data.yaml`
- [x] **Set `paths.raw_dir`** to the existing `./raw` directory (containing `TRAIN/` and `TEST/`) ✅
- [x] **Set `paths.interim_dir`** to `./interim` ✅
- [x] **Set `paths.processed_dir`** to `./processed` ✅
- [x] **Ensure no new top-level `data/` folder** is created implicitly (reuse the existing `raw/`, `interim/`, `processed/`) ✅

#### 2.8.2 Implement Cached Panel Builder in `src/data.py`
- [x] **Add function `get_panel(split: str, config, use_cache: bool = True, force_rebuild: bool = False) -> pd.DataFrame`** that: ✅
  - [x] Determines the panel path, e.g. `interim/panel_{split}.parquet` (e.g. `panel_train.parquet`, `panel_test.parquet`) ✅
  - [x] If `use_cache=True` and the parquet file exists and `force_rebuild=False`, loads it and returns ✅
  - [x] Otherwise: ✅
    - [x] Calls `load_raw_data(config, split=split)` to read CSVs from `raw/TRAIN` or `raw/TEST` ✅
    - [x] Calls `prepare_base_panel(...)` and `handle_missing_values(...)` ✅
    - [x] Calls `compute_pre_entry_stats(..., is_train=(split == "train"))` ✅
    - [x] Saves the resulting panel to `interim/panel_{split}.parquet` ✅
    - [x] Returns the panel DataFrame ✅
- [x] **Ensure this function is the only entry point** used by training/inference to obtain panels, to avoid duplicated logic ✅

#### 2.8.3 Implement Persisted Feature Matrices in `src/features.py`
- [x] **Add function `get_features(split: str, scenario: int, mode: str, config, use_cache: bool = True, force_rebuild: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]`** that: ✅
  - [x] Computes the output path based on `paths.processed_dir`, e.g.: ✅
    - [x] `processed/features_{split}_scenario{scenario}_{mode}.parquet` ✅
    - [x] `processed/target_{split}_scenario{scenario}_{mode}.parquet` (optional, for y) ✅
    - [x] `processed/meta_{split}_scenario{scenario}_{mode}.parquet` (optional, for META_COLS) ✅
  - [x] If `use_cache=True` and files exist and `force_rebuild=False`, loads and returns `(X, y, meta)` (for `mode="train"`; for `mode="test"` y is `None`) ✅
  - [x] Otherwise: ✅
    - [x] Uses `get_panel(split, config)` to obtain the base panel ✅
    - [x] Calls `make_features(panel_df, scenario=scenario, mode=mode, config=features_config)` ✅
    - [x] Splits into `(X, y, meta)` via `split_features_target_meta` ✅
    - [x] Saves `X` (features) to `processed/features_...parquet` ✅
    - [x] Saves `y` (if not None) and `meta` (if needed) similarly ✅
    - [x] Returns `(X, y, meta)` ✅
- [x] **Add `clear_feature_cache()` function** to clear cached feature files ✅

#### 2.8.4 Wire Training and Inference to Use Persisted Features
- [x] **In `src/train.py`**, modify `train_scenario_model` / `run_experiment` to: ✅
  - [x] Call `get_features(split="train", scenario, mode="train", ...)` instead of manually building panels + features ✅
  - [x] Optionally support a flag `--force-rebuild` in the CLI that sets `force_rebuild=True` for both `get_panel` and `get_features` ✅
  - [x] Added `--no-cache` flag to disable feature caching ✅
- [x] **In `src/inference.py`**, modify the inference pipeline to: ✅
  - [x] Use `get_panel()` for test data loading ✅
  - [x] Added `--force-rebuild` CLI flag ✅

#### 2.8.5 Add Minimal CLI Entrypoints for Data Build (No New Modules)
- [x] **In `src/data.py`**, add a small `if __name__ == "__main__":` block using `argparse` to: ✅
  - [x] Accept arguments like `--split train|test`, `--scenario 1|2`, `--mode train|test`, `--force-rebuild` ✅
  - [x] Accept `--data-config` and `--features-config` paths ✅
  - [x] Call `get_panel(...)` and/or `get_features(...)` to pre-build and cache data ✅
- [x] **Add commands to README.md / TODO.md "Quick Commands"** (see below) ✅
- [x] **Emphasize that no new Python files** (beyond existing modules) should be created for this; all logic stays inside `src/data.py` and `src/features.py` ✅

#### 2.8.6 Implement Basic Schema and Shape Validation at Each Stage
- [x] **After building each panel**, assert that: ✅
  - [x] Required columns (`country`, `brand_name`, `months_postgx`, `volume`, `n_gxs`, drug attributes) are present ✅
  - [x] There are no duplicate keys (`country`, `brand_name`, `months_postgx`) ✅
  - [x] `validate_panel_schema()` function added to `src/data.py` ✅
- [x] **After building each feature matrix**: ✅
  - [x] Assert that feature columns contain **no META_COLS** ✅
  - [x] Assert that `X.shape[0] == len(y)` (for `mode="train"`) ✅
  - [x] Log shapes, e.g. `X_train_s1: (n_rows, n_features)` ✅
  - [x] `validate_feature_matrix()` function added to `src/features.py` ✅

#### 2.8.7 Optimize Storage Formats and Dtypes
- [x] **Use Parquet format** for both panels and features to reduce disk usage and speed up I/O ✅
- [x] **In `prepare_base_panel` and `make_features`**: ✅
  - [x] Cast high-cardinality categoricals (`country`, `brand_name`, `ther_area`, `main_package`) to `category` dtype before saving ✅
  - [x] Optionally downcast numeric types (`int64` → `int32`, `float64` → `float32`) when safe, and log the downcasting ✅
  - [x] `_optimize_dtypes()` function updated in `src/data.py` ✅
  - [x] Preserves precision for critical columns (`y_norm`, `volume`, `avg_vol_12m`) ✅

---

## 3. Feature Engineering

### 3.0 Feature Mode Handling (`make_features`)
- [x] **Ensure `make_features(panel_df, scenario, mode)` behaves correctly**: ✅
  - [x] Creates `y_norm` **only** when `mode="train"` ✅
  - [x] Never touches target-related columns for `mode="test"` (no `y_norm`, no `bucket`) ✅
  - [x] Respects scenario-specific cutoffs internally (`months_postgx < 0` for S1, `< 6` for S2) ✅
- [x] **Ensure `configs/features.yaml`** can enable/disable features per scenario (e.g. Scenario 2 early-erosion features are not accidentally computed for Scenario 1) ✅
- [x] **Confirm `make_features`** loads and applies scenario-specific settings from `features.yaml`, not hardcoded logic ✅
  - [x] Added `_load_feature_config()` helper to merge user config with defaults ✅
- [x] **Implement helper functions** driven by `configs/features.yaml`: ✅
  - [x] `add_pre_entry_features`, `add_time_features`, `add_generics_features`, `add_drug_features`, `add_early_erosion_features` ✅
  - [x] All functions now accept config dict parameter ✅

### 3.1 Pre-Entry Features (`src/features.py`)
- [x] `avg_vol_3m`, `avg_vol_6m`, `avg_vol_12m` - Rolling averages ✅
- [x] `log_avg_vol_12m` - Log-transformed scale normalization ✅
- [x] `pre_entry_trend` - Linear slope of pre-entry volume ✅
- [x] `pre_entry_volatility` - Coefficient of variation ✅
- [x] **Add pre_entry_max** - Maximum volume in pre-entry period ✅
- [x] **Add pre_entry_min** - Minimum volume in pre-entry period ✅
- [x] **Add volume_growth_rate** - (vol_3m - vol_12m) / vol_12m ✅
- [x] **Add pre_entry_trend_norm** - Normalized trend for scale invariance ✅
- [x] **Add pre_entry_max_ratio, pre_entry_min_ratio, pre_entry_range_ratio** ✅
- [ ] **Add seasonal pattern detection** from pre-entry months
  - [ ] Define precisely: e.g., ratios of month-of-year volume to annual average, or cosine/sine seasonal features based on `month`
  - [ ] Document definition in `docs/functionality.md`

### 3.2 Time Features
- [x] `months_postgx` - Direct time index ✅
- [x] `months_postgx_squared` - Quadratic decay ✅
- [x] `months_postgx_cube` - Cubic for flexible decay ✅
- [x] `is_post_entry` - Binary flag ✅
- [x] `time_bucket` - Categorical (pre, early, mid, late) ✅
- [x] `is_early`, `is_mid`, `is_late` - One-hot time bucket encoding ✅
- [x] **Add time decay curve features**: exp(-alpha * months_postgx) ✅
  - [x] `time_decay` (alpha=0.1) and `time_decay_fast` (alpha=0.2) ✅
- [x] **Add month_of_year** with cyclical encoding (month_sin, month_cos) ✅
- [x] **Add quarters** (Q1-Q4) for seasonality ✅
  - [x] `quarter` and `is_q1`, `is_q2`, `is_q3`, `is_q4` one-hot encoding ✅
- [x] **Add is_year_end** flag for December submissions ✅
- [x] **Add is_year_start** flag for January ✅
- [x] **Add sqrt_months_postgx** for slower decay ✅

### 3.3 Generics Competition Features
- [x] `n_gxs` - Current number of generics ✅
- [x] `has_generic` - Binary (n_gxs > 0) ✅
- [x] `multiple_generics` - Binary (n_gxs >= 2) ✅
- [x] `many_generics` - Binary (n_gxs >= 5) ✅
- [x] `n_gxs_pre_cutoff_max` - Maximum generics up to cutoff month ✅
- [x] **Add first_generic_month** - Month of first generic entry ✅
- [x] **Add months_since_first_generic** - Time since first competitor ✅
- [x] **Add had_generic_pre_entry** - Binary flag ✅
- [x] **Add generic_entry_speed** - Rate of new generic entries ✅
- [x] **Add log_n_gxs, log_n_gxs_at_entry** - Log transforms ✅
- [x] **Add n_gxs_bin** - Categorical binning (none/one/few/several/many) ✅
- [ ] **Add expected_future_generics** only if future `n_gxs` for forecast months is provided in `df_generics_*` and is **exogenous** (no use of future observed volume). Otherwise **do not implement** this feature to avoid leakage.

### 3.4 Drug Characteristics (Static Features)
- [x] `ther_area` - Therapeutic area (categorical) ✅
- [x] `main_package` - Dosage form (categorical) ✅
- [x] `hospital_rate` - Hospital percentage (0-100) ✅
- [x] `biological` - Boolean ✅
- [x] `small_molecule` - Boolean ✅
- [x] `hospital_rate_bin` - Binned hospital rate (low/medium/high) ✅
- [x] `hospital_rate_norm` - Normalized 0-1 ✅
- [x] `is_hospital_drug`, `is_retail_drug` - Binary flags ✅
- [x] **Add is_injection** derived from main_package ✅
- [x] **Add is_oral** derived from main_package ✅
- [x] **Add ther_area_encoded, main_package_encoded, country_encoded** - Label encoding ✅
- [x] **Add biological_x_n_gxs** - Interaction feature ✅
- [x] **Add hospital_rate_x_time** - Interaction feature ✅
- [ ] **Add ther_area_erosion_prior** - Historical avg erosion by area
  - [ ] **CRITICAL**: Must be computed **only from training data**, never including test series
- [ ] **Add country_effect** - Country-level erosion patterns
- [ ] **Add ther_area encoding** (target encoding with K-fold)

### 3.5 Scenario 2 Specific Features
- [x] `avg_vol_0_5` - Mean volume over months [0, 5] ✅
- [x] `erosion_0_5` - Early erosion signal ✅
- [x] `trend_0_5` - Linear slope over months 0-5 ✅
- [x] `trend_0_5_norm` - Normalized trend ✅
- [x] `drop_month_0` - Initial volume drop ✅
- [x] **Add month_0_to_3_change** - Short-term erosion rate ✅
- [x] **Add month_3_to_5_change** - Medium-term erosion rate ✅
- [x] **Add avg_vol_0_2, avg_vol_3_5** - Window averages ✅
- [x] **Add erosion_0_2, erosion_3_5** - Window erosion ratios ✅
- [x] **Add recovery_signal** - If volume increases after initial drop ✅
  - [x] Defined: boolean flag if average volume in months [3–5] is higher than in months [0–2] ✅
- [x] **Add recovery_magnitude** - Size of recovery (clipped to [-1, 1]) ✅
- [x] **Add competition_response** - n_gxs change in 0-5 ✅
  - [x] Defined: change in `n_gxs` between months 0 and 5 ✅
- [x] **Add erosion_per_generic** - Erosion per new generic competitor ✅

### 3.6 Feature Scenario & Cutoff Validation
- [x] **Verify that any feature using post-entry volume** (e.g. `avg_vol_0_5`, `erosion_0_5`, `trend_0_5`) is only used: ✅
  - [x] In Scenario 2 ✅
  - [x] And only derived from months strictly before the scenario's forecast start (6–23 for S2) ✅
- [x] **Add automated tests** to enforce these constraints ✅
  - [x] `validate_feature_leakage()` function ✅
  - [x] `validate_feature_cutoffs()` function ✅
  - [x] Test `test_feature_leakage_validation` ✅
  - [x] Test `test_scenario1_no_early_erosion_features` ✅

### 3.7 Interaction Features
- [x] **Implement interaction feature framework** ✅
  - [x] `add_interaction_features()` function ✅
  - [x] Configurable via features.yaml interactions section ✅
- [x] **Add n_gxs × biological** - Competition impact on biologics ✅ (via `biological_x_n_gxs`)
- [x] **Add hospital_rate × months_postgx** - Time-hospital interaction ✅ (via `hospital_rate_x_time`)
- [ ] **Add ther_area × erosion_pattern** - Area-specific curves (requires target encoding)

### 3.8 Feature Selection
- [ ] **Analyze feature correlations** and remove redundant features
- [ ] **Use permutation importance** post-training
- [ ] **Try recursive feature elimination** for linear models
- [ ] **Document final feature set** for each scenario

### 3.9 Code Additions (Implementation Log)
> **Summary**: The following code additions implement Section 3.

- [x] **`src/features.py`**:
  - [x] Added `FORBIDDEN_FEATURES` constant for leakage prevention ✅
  - [x] Added `_load_feature_config()` to merge config with defaults ✅
  - [x] Updated `make_features()` to use config-driven feature generation ✅
  - [x] Updated `add_pre_entry_features()` with config support and new features ✅
  - [x] Updated `add_time_features()` with decay curves, quarters, year-end flags ✅
  - [x] Updated `add_generics_features()` with entry speed, binning, log transforms ✅
  - [x] Updated `add_drug_features()` with is_injection, is_oral, interactions ✅
  - [x] Updated `add_early_erosion_features()` with change windows, recovery, competition ✅
  - [x] Added `add_interaction_features()` for configurable interactions ✅
  - [x] Added `validate_feature_leakage()` for leakage detection ✅
  - [x] Added `validate_feature_cutoffs()` for cutoff validation ✅
  - [x] Added `split_features_target_meta()` for clean feature/target separation ✅
  - [x] Added `get_categorical_feature_names()` and `get_numeric_feature_names()` helpers ✅
- [x] **`tests/test_smoke.py`**:
  - [x] Added `test_feature_config_loading` ✅
  - [x] Added `test_pre_entry_features` ✅
  - [x] Added `test_time_features` ✅
  - [x] Added `test_generics_features` ✅
  - [x] Added `test_scenario2_early_erosion_features` ✅
  - [x] Added `test_feature_leakage_validation` ✅
  - [x] Added `test_scenario1_no_early_erosion_features` ✅
  - [x] Added `test_make_features_mode_train_vs_test` ✅
  - [x] Added `test_split_features_target_meta` ✅
  - [x] Added `test_interaction_features` ✅
  - [x] **Test results: 87 passed, 0 skipped, 0 warnings** ✅

---

## 4. Model Development

### 4.0 Priority Classification & Model Interface
> **Priority A (Must-Have)**: CatBoost + robust features + solid validation + simple ensemble  
> **Priority B (Nice-to-Have)**: Advanced neural architectures, complex augmentation, extensive HPO

- [ ] **Ensure all models implement a common interface** (`fit`, `predict`, `save`, `load`, optional `get_feature_importance`) so that `train_scenario_model` and `inference` can treat them polymorphically
- [ ] **Verify `BaseModel` abstract class** in `src/models/base.py` defines the required interface

### 4.1 Baseline Models (`src/models/linear.py`)
- [x] `FlatBaseline` - Always predicts 1.0 (no erosion) ✅
- [x] `GlobalMeanBaseline` - Predicts mean erosion curve ✅
- [x] `TrendBaseline` - Extrapolates pre-entry trend ✅
- [ ] **Implement HistoricalCurveBaseline** - Matches to similar historical series
- [ ] **Record baseline scores** for both scenarios

### 4.2 Linear Models (`src/models/linear.py`)
- [x] `Ridge` - L2 regularized linear regression ✅
- [x] `Lasso` - L1 regularized (sparse) ✅
- [x] `ElasticNet` - Combined L1+L2 ✅
- [x] `HuberRegressor` - Robust to outliers ✅
- [ ] **Tune regularization strength** (alpha) via CV
- [ ] **Try polynomial features** (degree 2) with linear models
- [ ] **Use linear model coefficients** for interpretability insights

### 4.3 CatBoost (`src/models/cat_model.py`) - Hero Model
- [x] Basic implementation with native categorical support ✅
- [x] Sample weight support via Pool ✅
- [x] Early stopping support ✅
- [x] Feature importance extraction ✅
- [ ] **Tune depth** (try 4, 6, 8)
- [ ] **Tune learning_rate** (try 0.01, 0.03, 0.05)
- [ ] **Tune l2_leaf_reg** (try 1, 3, 5, 10)
- [ ] **Try loss_function = 'MAE'** instead of RMSE
- [ ] **Try loss_function = 'Quantile'** for different quantiles
- [ ] **Implement custom objective** aligned with official metric

### 4.4 LightGBM (`src/models/lgbm_model.py`)
- [x] Basic implementation ✅
- [x] Sample weight support ✅
- [x] Early stopping ✅
- [ ] **Tune num_leaves** (try 15, 31, 63)
- [ ] **Tune min_data_in_leaf** (try 10, 20, 50)
- [ ] **Try dart boosting** for regularization
- [ ] **Compare speed vs CatBoost**

### 4.5 XGBoost (`src/models/xgb_model.py`)
- [x] Basic implementation with DMatrix ✅
- [x] Sample weight support ✅
- [x] Early stopping ✅
- [ ] **Tune max_depth** (try 4, 6, 8)
- [ ] **Tune min_child_weight** (try 1, 5, 10)
- [ ] **Try colsample_bytree** (try 0.6, 0.8, 1.0)
- [ ] **Enable GPU** if available (tree_method='gpu_hist')

### 4.6 Neural Network (`src/models/nn.py`)
- [x] SimpleMLP with configurable layers ✅
- [x] Dropout and batch normalization ✅
- [x] Early stopping ✅
- [x] WeightedRandomSampler for sample weights ✅
- [ ] **Experiment with architecture** (try [128, 64], [512, 256, 128])
- [ ] **Try different activations** (GELU, SiLU instead of ReLU)
- [ ] **Add residual connections** for deeper networks
- [ ] **Try embedding layers** for categorical features

### 4.7 Neural Network - Nice-to-Have / Post-Competition
> **Note**: These are lower priority and should not distract from core datathon-critical work
- [ ] **Implement TabNet** architecture
- [ ] **Implement FT-Transformer** architecture

### 4.8 Ensemble Methods (Priority A)
- [ ] **Implement simple averaging** of predictions
- [ ] **Implement weighted averaging** (tune weights on validation)
- [ ] **Implement stacking** with meta-learner
- [ ] **Implement blending** with hold-out predictions
- [ ] **Try CatBoost + LightGBM + XGBoost ensemble**
- [ ] **Document ensemble weights** for reproducibility

### 4.9 Segment-Specific Models for High-Impact Regions (Priority A)
> **Rationale**: The metric heavily weights bucket 1 and early months. Dedicated models for these segments can be decisive.

- [ ] **Scenario- and segment-specific models for high-impact regions**
  - [ ] Train **separate CatBoost models for bucket 1 vs bucket 2** (using bucket only on train, never as feature). At inference, assign each test series to a "pseudo-bucket" using pre-entry features + early-post-entry dynamics (Scenario 2) via a small classifier, and route its prediction to the corresponding expert model.
  - [ ] Train **early-window-focused models**:
    - [ ] For Scenario 1: a dedicated model optimised only on months 0–5 targets (with higher sample weights), then blend its predictions with the full-horizon model.
    - [ ] For Scenario 2: a dedicated model focused on months 6–11 (the heavy-weight window), blended with the full-horizon one.
  - [ ] Evaluate whether **segment-specific models + blending** beat a single global CatBoost in CV and on the leaderboard proxy.

### 4.10 Hierarchical / Segmented Modelling by Country & Therapeutic Area
- [ ] **Hierarchical / segmented modelling by country and therapeutic area**
  - [ ] Cluster series into **homogeneous groups** (e.g. by country, therapeutic area, hospital_rate_bin, biologic vs small molecule).
  - [ ] For each group with sufficient data, train:
    - [ ] A **group-specific CatBoost model**.
    - [ ] Or a **shared global model + group-specific bias adjustment** (e.g., post-hoc calibration per group).
  - [ ] Compare:
    - [ ] Global-only model vs segmented ensemble on the unified CV scheme.
    - [ ] Pay special attention to segments that contribute most to the metric (e.g., high-erosion bucket 1 series in large markets).
  - [ ] If segmented models improve the metric, integrate them into the main **ensemble/blending step** and document the strategy.

---

## 5. Training Pipeline

### 5.1 Core Training (`src/train.py`)
- [x] `split_features_target_meta()` - Separate columns ✅
- [x] `compute_sample_weights()` - Time + bucket weights ✅
- [x] `train_scenario_model()` - Train single model ✅
- [x] `run_experiment()` - Full experiment loop ✅
- [x] CLI interface with argparse ✅
- [x] **Add cross-validation loop** (`run_cross_validation()` K-fold training with series-level splits) ✅
- [x] **Add OOF prediction saving** (OOF predictions saved in `run_cross_validation()`) ✅
- [ ] **Add experiment tracking** (MLflow/W&B integration)
- [ ] **Add checkpoint saving** for resume training
- [ ] **Add config hashing** for reproducibility
  - [ ] Save an exact copy/snapshot of all config files used for a run into the corresponding `artifacts/run_id/` folder

### 5.2 CLI Consistency & Help
- [x] **Ensure `src/train.py` and `src/inference.py` implement `--help`** with clear argument documentation ✅
- [x] **Update `TODO.md` Quick Commands** to use exactly the same argument names as the code (already consistent: `--data-config`, `--features-config`, `--run-config`, `--model-config`) ✅

### 5.3 Experiment Metadata Logging
- [x] **Log at the start of each training run**: ✅
  - [x] Scenario, model type, config paths ✅
  - [x] Random seed ✅
  - [x] Git commit hash (if available) - `get_git_commit_hash()` ✅
  - [x] Dataset sizes (number of series, number of rows) - `get_experiment_metadata()` ✅
- [x] **Save a small JSON/YAML metadata file** in `artifacts/` per run (`metadata.json`, `config_snapshot.yaml`, `metrics.json` in run_experiment()) ✅

### 5.4 Sample Weights Refinement
- [x] Time-window weights (50/20/10 for S1, 50/30/20 for S2) ✅
- [x] Bucket weights (2× for bucket 1) ✅
- [ ] **Fine-tune time weights** based on official metric formula
- [ ] **Add month-level weights** for 20% monthly component
- [ ] **Experiment with sqrt/log transformations** of weights
- [ ] **Validate weights** correlate with metric improvement
- [x] **Set weights according to `configs/run_defaults.yaml`** (`compute_sample_weights()` reads from config) ✅
- [ ] **Explicit alignment of training loss with official metric**
  - [ ] Derive **exact per-row weights** that reproduce the official metric contribution:
    - [ ] Early windows vs rest of horizon.
    - [ ] Bucket 1 vs bucket 2.
    - [ ] Monthly 20% component.
  - [ ] Implement a **"metric-aligned weight calculator"** that, for any row `(series, month)`, outputs the precise weight implied by the metric.
  - [ ] Plug these weights into CatBoost/GBMs and validate on a small toy example that the **weighted RMSE at row level aggregates to the same series-level metric** (up to numerical noise).

### 5.5 Hyperparameter Optimization
- [ ] **Implement Optuna integration** for CatBoost
- [ ] **Define search space** in configs/model_cat.yaml
- [ ] **Use pruning** for early termination of bad trials
- [ ] **Run for 100+ trials** with timeout
- [ ] **Save best hyperparameters** to configs/
- [ ] **Document tuning results** with visualization

### 5.6 Hyperparameter Optimization - Nice-to-Have / Post-Competition
> **Note**: These are lower priority and should not distract from core datathon-critical work
- [ ] **Bayesian HPO with 100+ trials** across multiple model types
- [ ] **Nested CV** for unbiased model selection

### 5.7 Training Workflow
- [ ] **Create end-to-end training script** (`scripts/train_full.py`)
- [ ] **Add parallel training** for different scenarios
- [ ] **Add memory profiling** for large datasets
- [x] **Add training time logging** (in `train_scenario_model()` metrics) ✅
- [ ] **Create training dashboard** (TensorBoard/W&B)

---

## 6. Validation & Evaluation

### 6.1 Validation Strategy (`src/validation.py`)
- [x] `create_validation_split()` - Series-level split ✅
- [x] `simulate_scenario()` - Create scenario from training data ✅
- [x] `adversarial_validation()` - Train/test distribution check ✅
- [x] `get_fold_series()` - K-fold series-level generation ✅
- [ ] **Implement time-based CV** (temporal cross-validation)
- [ ] **Implement grouped K-fold** by therapeutic area
- [ ] **Add purged CV** with gap between train/val
- [ ] **Implement nested CV** for unbiased model selection
- [ ] **Verify validation respects scenario constraints**: any time-based or purged CV must:
  - [ ] Not leak post-forecast-window information into feature computation
  - [ ] Respect the same history/horizon separation as the real competition scenarios

### 6.2 Scenario Detection & Counts Sanity Check
- [x] **Add test to verify `detect_test_scenarios()`** reproduces expected counts (228 Scenario 1, 112 Scenario 2) ✅
- [x] **Fixed FutureWarning** in detect_test_scenarios() ✅
- [ ] **Raise/log a warning** if detected counts differ from expected

### 6.3 Metrics (`src/evaluate.py`)
- [x] `compute_metric1()` - Scenario 1 official metric ✅
- [x] `compute_metric2()` - Scenario 2 official metric ✅
- [x] `compute_bucket_metrics()` - Per-bucket RMSE ✅
- [x] `create_aux_file()` - Generate auxiliary file ✅
- [x] **Verify metric matches** official implementation exactly (regression test added) ✅
- [x] **Add per-series metrics** for error analysis (`compute_per_series_error()`) ✅
- [ ] **Add metric breakdown** by therapeutic area
- [ ] **Add metric breakdown** by country
- [ ] **Add visualization** of predictions vs actuals

### 6.4 Official Metric Wrapper Regression Test
- [x] **Build a minimal regression test** that: ✅
  - [x] Uses synthetic test data to verify metric calculation ✅
  - [x] Compares the output of `compute_metric1` / `compute_metric2` in `src/evaluate.py` to the official standalone call ✅
  - [x] Asserts equality (or negligible numerical difference) ✅
- [x] **Confirm auxiliary file schema** matches the official example file 1:1: ✅
  - [x] Ensure `create_aux_file` produces a DataFrame with exactly the same schema as `docs/guide/auxiliar_metric_computation_example.csv` (column names and types) ✅

### 6.5 Error Analysis
- [x] **Identify worst-performing series** in validation (`identify_worst_series()`) ✅
- [x] **Analyze errors by bucket** (1 vs 2) (`analyze_errors_by_bucket()`) ✅
- [x] **Analyze errors by time window** (early/mid/late) (`analyze_errors_by_time_window()`) ✅
- [x] **Check for systematic biases** (over/under prediction) (`check_systematic_bias()`) ✅
- [x] **Define canonical metric name constants** (METRIC_NAME_S1, METRIC_NAME_S2, etc.) ✅
- [ ] **Create error distribution plots**
- [ ] **Document insights** for model improvement
- [ ] **Targeted booster models for worst-performing series**
  - [ ] From CV, identify the **top X% series with highest absolute error** (especially in bucket 1 and early windows).
  - [ ] Train a small **"booster" model** on these series only (using the same features + an indicator for "hard series" if needed).
  - [ ] At inference time, use a **"hardness score" predictor** (trained on train data only) to guess if a test series is likely to be "hard"; if so:
    - [ ] Blend the base model prediction with the booster model prediction (e.g. higher weight on booster for hard series).
  - [ ] Validate whether this two-step approach reduces the **tail of the error distribution**, which is often heavily weighted in the official metric.

### 6.6 Cross-Validation Infrastructure
- [x] **Implement CV with reproducible folds** (`get_fold_series()`) ✅
- [ ] **Save fold indices** for reproducibility
- [ ] **Aggregate CV scores** with confidence intervals
- [ ] **Create CV comparison table** for different models
- [ ] **Implement statistical tests** (paired t-test)

### 6.7 Unified Metrics Schema & Logging (Train / Val / Test)

> **Goal**: All phases that touch metrics (training, validation, cross-validation, simulations, offline test) must:
> * Use the **same metric names**
> * Store them in the **same tabular schema**
> * Write them to a **single canonical location** per run (e.g. `artifacts/{run_id}/metrics.csv`)

#### 6.7.1 Define Canonical Metrics Config & Names
- [ ] **Extend `configs/run_defaults.yaml` with a `metrics` section**:
  - [ ] `metrics.primary`: list of main metrics to always log (e.g. `["metric1_official", "metric2_official"]`)
  - [ ] `metrics.secondary`: list of auxiliary metrics (e.g. `["rmse_y_norm", "mae_y_norm", "loss"]`)
  - [ ] `metrics.log_per_series`: `true/false` flag to enable per-series metrics
  - [ ] `metrics.log_dir_pattern`: pattern for metrics dir (e.g. `"artifacts/{run_id}/metrics"`)
- [x] **In `src/evaluate.py` define canonical metric name constants**: ✅
  - [x] e.g. `METRIC_NAME_S1 = "metric1_official"`, `METRIC_NAME_S2 = "metric2_official"`, etc. ✅
  - [ ] e.g. `METRIC_NAME_S1 = "metric1_official"`, `METRIC_NAME_S2 = "metric2_official"`, `METRIC_NAME_RMSE = "rmse_y_norm"`, etc.
  - [ ] Use these constants **everywhere** (training, validation, notebooks) instead of ad-hoc strings

#### 6.7.2 Implement Unified Metric Record Helpers
- [ ] **In `src/evaluate.py` or `src/utils.py`, implement**:
  - [ ] `make_metric_record(phase, split, scenario, model_name, metric_name, value, step=None, bucket=None, series_id=None, extra=None) -> Dict[str, Any]` that always returns a dict with at least:
    - [ ] `run_id`
    - [ ] `phase` (one of: `"train"`, `"val"`, `"cv"`, `"simulation"`, `"test_offline"`, `"test_online"`)
    - [ ] `split` (`"train"`, `"val"`, `"test"`)
    - [ ] `scenario` (`1` or `2`)
    - [ ] `model` (e.g. `"catboost"`, `"lgbm"`)
    - [ ] `metric` (canonical name)
    - [ ] `value` (float, can be `NaN` for metrics without ground truth)
    - [ ] `step` (epoch index, fold index, or `"final"`)
    - [ ] `bucket` (optional, `None` if not applicable)
    - [ ] `series_id` (optional; only used for per-series metrics)
    - [ ] `timestamp` (UTC ISO string)
    - [ ] `extra` (JSON-serializable dict for anything custom; can be `None`)
  - [ ] `save_metric_records(records: List[Dict], path: Path, append: bool = True)` that:
    - [ ] Creates the parent directory (`artifacts/{run_id}/metrics`) if missing
    - [ ] Writes/appends to a **single file per run** (e.g. `metrics.csv` or `metrics.jsonl`)
    - [ ] Ensures column/field names are **stable** across all calls and phases

#### 6.7.3 Wire Unified Logging into Training
- [ ] **In `train_scenario_model` (and any CV loop in `src/train.py`), replace ad-hoc logging with unified records**:
  - [ ] After each major step (epoch or just final fit), compute:
    - [ ] Training loss (from model, if exposed)
    - [ ] Primary/secondary metrics on the training set (if cheap)
    - [ ] Primary/secondary metrics on the validation set (using `compute_metric1` / `compute_metric2`)
  - [ ] Wrap all values using `make_metric_record` with:
    - [ ] `phase="train"` or `phase="val"`
    - [ ] `step` = epoch index or `"final"` if you log only once per run
  - [ ] Persist them via `save_metric_records` into `artifacts/{run_id}/metrics.csv`
- [ ] **For cross-validation** (once implemented):
  - [ ] Log each fold's validation metrics with `phase="cv"`, `step=<fold_index>`
  - [ ] After CV aggregation, log the mean/std as separate records with `phase="cv"`, `step="cv_agg"`

#### 6.7.4 Wire Unified Logging into Validation / Simulation
- [ ] **In `src/validation.py`**:
  - [ ] Wherever `simulate_scenario` or adversarial validation computes metrics, replace any custom logging with:
    - [ ] `make_metric_record(phase="simulation" or "adversarial", split="train"/"val", ...)`
    - [ ] `save_metric_records` to append into the **same** `metrics.csv`
  - [ ] For bucket-level or therapeutic-area-level breakdowns, store them as normal metrics with:
    - [ ] `bucket` set appropriately
    - [ ] Or `extra={"ther_area": "..."}`
    - [ ] No custom schemas or separate CSV for these; just extra columns in the same file

#### 6.7.5 Wire Unified Logging into Inference / Offline Test
- [ ] **For simulated test evaluation (where ground truth exists)**:
  - [ ] Use the same `compute_metric1` / `compute_metric2` functions
  - [ ] Log results via `make_metric_record(phase="test_offline", split="test", ...)`
- [ ] **For real competition test (no ground truth)**:
  - [ ] Define a small set of "diagnostic" metrics computed on predictions only (e.g. `"pred_mean"`, `"pred_std"`, `"pred_min"`, `"pred_max"`)
  - [ ] Log them with `phase="test_online"`, `split="test"`, using the same schema
  - [ ] Keep the structure identical even if there is no error metric (no y); `value` is still a float

#### 6.7.6 Per-Series Metrics in a Consistent Format
- [ ] **Extend error analysis functions in `src/evaluate.py` to optionally return per-series metrics**:
  - [ ] DataFrame with columns:
    - [ ] `series_id` (e.g. synthetic ID or tuple-encoded `(country, brand_name)`)
    - [ ] `scenario`
    - [ ] `bucket`
    - [ ] `metric`
    - [ ] `value`
  - [ ] Save this as `artifacts/{run_id}/metrics_per_series.parquet` with the **same schema** reused for:
    - [ ] validation
    - [ ] cross-validation
    - [ ] simulated test evaluation
- [ ] **Ensure that per-series metrics are optional and controlled via `metrics.log_per_series`** flag from config

#### 6.7.7 Tests for Unified Metrics Logging
- [ ] **Unit test helpers**:
  - [ ] Test that `make_metric_record` always returns all required keys and that their types are consistent
  - [ ] Test that `save_metric_records`:
    - [ ] Creates a new file with the correct header on first call
    - [ ] Preserves the same columns when appending records from another phase (e.g. `train` then `val`)
- [ ] **Integration test**:
  - [ ] Run a tiny end-to-end training + validation flow (on a small subset of series)
  - [ ] Assert that `artifacts/{run_id}/metrics.csv` exists
  - [ ] Assert that it contains at least one row for each phase that was executed (`train`, `val`)
  - [ ] Optionally check that metric names present are exactly those defined in `metrics.primary` and `metrics.secondary`

#### 6.7.8 Documentation of Metrics Flow
- [ ] **In `docs/functionality.md` or `README.md`, add a "Metrics & Logging" section**:
  - [ ] Describe the unified metrics schema (columns of `metrics.csv` and `metrics_per_series.parquet`)
  - [ ] Show a short example (Python snippet) for:
    - [ ] Loading all metrics for a given `run_id`
    - [ ] Pivoting them by `phase`/`scenario`/`metric` to compare different runs or models

### 6.8 Visualization & Plots (Data, Metrics, Predictions)

> **Goal**: Provide a **consistent, scriptable plotting layer** for:
> * Data & EDA
> * Training / validation / CV metrics
> * Prediction vs actual & error analysis
>
> Using the same `run_id`, configs, and metrics files as the rest of the pipeline.

#### 6.8.1 Plot Configuration
- [ ] **Extend `configs/run_defaults.yaml` with a `plots` section**:
  - [ ] `plots.enabled`: global on/off switch (default `true`)
  - [ ] `plots.backend`: e.g. `"matplotlib"` (allow override only if needed)
  - [ ] `plots.dir_pattern`: e.g. `"artifacts/{run_id}/plots"`
  - [ ] `plots.save_format`: e.g. `"png"` (optionally `"pdf"`)
  - [ ] `plots.generate_on_train_end`: bool – whether training automatically generates key plots
  - [ ] `plots.max_series_examples`: number of series to visualize for time-series plots (e.g. 20)
- [ ] **Ensure plotting code uses only this config** (no hardcoded paths or formats)

#### 6.8.2 Core Plotting Module
- [ ] **Create `src/plots.py` (or `src/visualization.py`) with pure plotting functions**:
  - [ ] Functions should be **stateless**, accept dataframes/arrays, and a `save_path`, and return nothing (just save files)
  - [ ] No heavy logic inside plots; they should consume **already prepared data** (panels, features, metrics)

#### 6.8.3 Data & EDA Plots
- [ ] **Implement functions to visualize key data distributions using `panel_df` and raw data**:
  - [ ] `plot_target_distribution(panel_df, save_path)`: distribution of `y_norm` and raw `volume`
  - [ ] `plot_avg_vol_12m_distribution(panel_df, save_path)`: histogram + log-scale option
  - [ ] `plot_bucket_share(panel_df, save_path)`: bar chart of bucket counts (1 vs 2)
  - [ ] `plot_hospital_rate_distribution(panel_df, save_path)`: histogram and binned counts
  - [ ] `plot_missingness_heatmap(panel_df, save_path)`: fraction of missing values per column (train/test)
  - [ ] `plot_erosion_curves_by_bucket(panel_df, save_path)`: mean normalized volume per `months_postgx` by bucket
  - [ ] `plot_erosion_curves_by_ther_area(panel_df, save_path)`: average erosion curves grouped by `ther_area`
- [ ] **Add small helper to select a sample of series** (e.g. 10–20) and:
  - [ ] `plot_series_examples(panel_df, save_dir)`: line plots of `volume` / `y_norm` across `months_postgx` for randomly chosen series, optionally colored by bucket

#### 6.8.4 Training & Validation Curves (Using Unified Metrics)
- [ ] **In `src/plots.py`, implement functions that consume `artifacts/{run_id}/metrics.csv`**:
  - [ ] `plot_training_curves(metrics_df, save_path)`: for each scenario + model:
    - [ ] line plots of metric(s) vs `step` for `phase="train"` and `phase="val"`
    - [ ] optionally separate subplots for each metric in `metrics.primary` / `metrics.secondary`
  - [ ] `plot_cv_scores(metrics_df, save_path)`: if `phase="cv"` exists:
    - [ ] bar or point plots of CV fold scores (with error bars for mean ± std)
- [ ] **Wire these into training**:
  - [ ] At the end of `run_experiment` (or main training flow), if `plots.generate_on_train_end` is `true`, load `metrics.csv` and call the appropriate plotting functions
  - [ ] Save plots to `artifacts/{run_id}/plots/train_val_curves_{scenario}_{model}.png`

#### 6.8.5 Feature Importance & Model Explainability Plots
- [ ] **Add plotting functions for feature importance**:
  - [ ] `plot_feature_importance(importances_df, save_path, top_k=30)`:
    - [ ] bar plot of top-k features by importance for CatBoost / LightGBM / XGBoost
    - [ ] support scenario-specific output names: `feature_importance_s1_catboost.png`, etc.
  - [ ] Ensure `importances_df` schema is standard:
    - [ ] columns: `feature`, `importance`, `model`, `scenario`
- [ ] **Update training code for tree models**:
  - [ ] After training CatBoost / LGBM / XGB:
    - [ ] compute global feature importances as a DataFrame
    - [ ] save raw importances (CSV/Parquet) to `artifacts/{run_id}/feature_importance_{scenario}_{model}.csv`
    - [ ] call `plot_feature_importance` to produce the corresponding plot if `plots.enabled`

#### 6.8.6 Prediction vs Actual & Error Analysis Plots
- [ ] **Extend `src/evaluate.py` to output standard evaluation DataFrames**:
  - [ ] `df_eval` with at least: `series_id`, `scenario`, `bucket`, `months_postgx`, `y_true`, `y_pred`, `error` (`y_pred - y_true`), `abs_error`, etc.
- [ ] **In `src/plots.py`, implement error-analysis plots using `df_eval`**:
  - [ ] `plot_pred_vs_actual_scatter(df_eval, save_path)`:
    - [ ] scatter plot of `y_true` vs `y_pred` (optionally colored by bucket)
  - [ ] `plot_error_distribution(df_eval, save_path)`:
    - [ ] histogram or KDE of `error` and `abs_error`, possibly per-bucket overlay
  - [ ] `plot_error_by_time_bucket(df_eval, save_path)`:
    - [ ] average error / absolute error per `time_bucket` (pre/early/mid/late)
  - [ ] `plot_error_by_ther_area(df_eval, save_path)`:
    - [ ] bar chart of mean absolute error per `ther_area`
- [ ] **Time-series level plots for selected series**:
  - [ ] `plot_series_prediction_curves(df_eval, save_dir, n_examples=10)`:
    - [ ] For a small set of series (random + worst MAE series), plot:
      - [ ] line of `y_true` and `y_pred` across forecast horizon (e.g. 0–23 or 6–23)
      - [ ] Include bucket information in title/legend
    - [ ] Save individual PNGs, e.g. `series_example_{series_id}.png`

#### 6.8.7 Scenario-Specific Diagnostic Plots
- [ ] **Scenario 1 diagnostics**:
  - [ ] `plot_s1_early_vs_late_error(df_eval, save_path)`:
    - [ ] compare metric contributions in early window (0–5) vs later months
- [ ] **Scenario 2 diagnostics**:
  - [ ] `plot_s2_early_erosion_vs_error(df_eval, save_path)`:
    - [ ] scatter of early erosion features (e.g. `erosion_0_5`) vs absolute error
  - [ ] `plot_s2_bucket1_focus(df_eval, save_path)`:
    - [ ] highlight errors for bucket 1 (high erosion) vs bucket 2

#### 6.8.8 CLI Entrypoints for Plot Generation
- [ ] **Add a small CLI in `src/plots.py` (or a thin `scripts/plot_run.py`)**:
  - [ ] Arguments:
    - [ ] `--run-id` (required)
    - [ ] `--data-config`, `--run-config` (to locate paths and enable/disable plot types)
    - [ ] `--phase` in `{eda, train, val, test, all}` to control which plots to generate
  - [ ] Logic:
    - [ ] Load configs, locate `artifacts/{run_id}`
    - [ ] For `phase="eda"`: load panel/features and call EDA plotting functions
    - [ ] For `phase="train"`/`"val"`: load `metrics.csv` and produce training/validation curves
    - [ ] For `phase="test"`: load evaluation DF (if available) and produce prediction/error plots
- [ ] **Add Quick Commands to TODO.md / README** (see Quick Commands section below)

#### 6.8.9 Tests for Plotting Layer (Smoke-Level)
- [ ] **Add basic tests in `tests/test_plots.py`**:
  - [ ] Use tiny synthetic dataframes (few rows) to call each plotting function
  - [ ] Assert that each function runs without error and creates a non-empty file at the expected `save_path`
  - [ ] Clean up temporary plot files after tests if needed
- [ ] **Add a small integration test**:
  - [ ] Run a mini end-to-end training on a handful of series to create `metrics.csv` and `df_eval`
  - [ ] Call the CLI for `phase=train` and `phase=test`
  - [ ] Assert that `artifacts/{run_id}/plots/` contains at least:
    - [ ] one training curve plot
    - [ ] one prediction vs actual / error-distribution plot

#### 6.8.10 Documentation of Plots
- [ ] **In `README.md` or `docs/functionality.md`, add a "Key Plots" subsection**:
  - [ ] Briefly list:
    - [ ] EDA plots (distributions, erosion curves)
    - [ ] Training/validation curves
    - [ ] Feature importance plots
    - [ ] Prediction vs actual and error analysis plots
  - [ ] Show 1–2 example images (or file paths) and the CLI command used to generate them
- [ ] **Ensure Phase 2 slide deck (Section 10.3) reuses these plots**:
  - [ ] Reference run IDs and file names so slides are easily reproducible

### 6.9 Systematic CV Scheme Search for Leaderboard Correlation
> **Rationale**: Top teams systematically search for the CV scheme that best correlates with LB scores. This separates top-10% from podium.

- [ ] **Systematic CV scheme search for leaderboard correlation**
  - [ ] Implement multiple candidate validation schemes:
    - [ ] Time-based split with different cutoffs (e.g. last 4, 6, 8 months).
    - [ ] Grouped CV by country, by therapeutic area, and by (country, ther_area).
    - [ ] Purged time-based CV (gap between train and validation windows).
  - [ ] For each scheme, run the **same model config** and record:
    - [ ] Average CV metric and variance.
    - [ ] Score on the **public leaderboard** for the corresponding submission.
  - [ ] Build a small **"CV vs LB correlation table"** (per scheme) and choose the one with:
    - [ ] Highest correlation to LB movements.
    - [ ] Reasonable variance and stability.
  - [ ] Freeze that CV scheme as the **official one for all further experiments** and document the choice in `docs/planning/approach.md`.
- [ ] **Stress-test robustness across alternative splits**
  - [ ] Even after fixing the primary CV, re-check the final hero model on:
    - [ ] A different temporal holdout.
    - [ ] A different country/therapeutic-area split.
  - [ ] Ensure there is **no catastrophic performance drop** in any realistic split; if yes, iterate feature engineering/modeling for the failing segment.

---

## 7. Inference & Submission

### 7.1 Inference Pipeline (`src/inference.py`)
- [x] `detect_test_scenarios()` - Identify S1 vs S2 series ✅
- [x] `generate_submission()` - Create submission file ✅
- [x] `apply_edge_case_fallback()` - Handle missing predictions ✅
- [x] `validate_submission_format()` - Check format ✅
- [ ] **Add batch prediction** for large datasets
- [ ] **Add confidence intervals** (if using ensemble)
- [ ] **Add prediction clipping** (reasonable bounds)
- [ ] **Add inverse transform verification** (y_norm → volume)
- [ ] **Ensure `generate_submission` always uses the `test_panel`'s own `avg_vol_12m`** and metadata (never look up or merge from training panel)

### 7.2 Template Schema Alignment
- [ ] **Confirm whether the submission template uses**:
  - [ ] `months_postgx` or `month` as the time index column
  - [ ] Standardize naming across code, docs, and TODO (rename references if needed)
- [ ] **Verify key columns in `validate_submission_format`** match the template exactly

### 7.3 Submission File Generation
- [ ] **Verify column order** matches template exactly
- [ ] **Verify date format** (YYYY-MM format)
- [ ] **Verify all required series** are present
- [ ] **Verify no duplicate rows**
- [ ] **Check for NaN/Inf values**
- [ ] **Generate both submission and auxiliary files**
  - [ ] Clarify: is the auxiliary file **only** for local metric calculation or also required for submission?
  - [ ] Check competition rules and update docs to state clearly if `auxiliar_metric_computation.csv` is only for local evaluation

### 7.4 Submission Workflow
- [ ] **Create submission script** (`scripts/submit.py`)
- [ ] **Add automatic validation** before writing file
- [ ] **Add submission versioning** (timestamp + model info)
- [ ] **Create submission log** (model, score, notes)
- [ ] **Add quick sanity check** (mean, std, min, max)

### 7.5 Edge Cases
- [ ] **Handle series with missing pre-entry data**
- [ ] **Handle series with all zeros volume**
- [ ] **Handle extreme predictions** (clip to [0, 2]?)
- [ ] **Document fallback strategies**

---

## 8. Experimentation & Optimization

### 8.0 Priority Tagging
> **Priority A (Must-Have for Datathon)**: CatBoost + robust features + solid validation + simple ensemble  
> **Priority B (Nice-to-Have / Post-Competition)**: Advanced items below marked with [Nice-to-Have]

### 8.1 Feature Experiments (Priority A)
- [ ] **Test each feature group** individually (ablation study)
- [ ] **Compare feature engineering** approaches
- [ ] **Try target encoding** with proper cross-fitting
- [ ] **Try frequency encoding** for categoricals
- [ ] **Test feature scaling** (StandardScaler vs none for GBMs)

### 8.2 Model Experiments (Priority A)
- [ ] **Compare all model types** on same validation
- [ ] **Test ensemble configurations**
- [ ] **Try different loss functions** (MAE, Huber, custom)
- [ ] **Test different learning rates** schedule
- [ ] **Compare native vs sklearn** implementations

### 8.3 Data Augmentation - Nice-to-Have / Post-Competition
> **Note**: These are lower priority and should not distract from core datathon-critical work
- [ ] **[Nice-to-Have] Try noise injection** on features
- [ ] **[Nice-to-Have] Try SMOTE-like augmentation** for rare buckets
- [ ] **[Nice-to-Have] Try mixup** for regression
- [ ] **[Nice-to-Have] Try synthetic series** generation

### 8.4 Post-Processing (Priority A)
- [ ] **Try prediction smoothing** across months
- [ ] **Try prediction adjustment** based on bucket
- [ ] **Try calibration** (isotonic regression)
- [ ] **Try ensemble weights optimization** on validation

### 8.5 Domain-Consistent Erosion Curve Shaping
> **Rationale**: Implausible shapes (e.g. volume increasing sharply after many generics enter) hurt performance on high-weight months and look bad in Phase 2.

- [ ] **Domain-consistent erosion curve shaping**
  - [ ] Add a post-processing step that enforces **soft monotonicity**:
    - [ ] Penalise or smooth out large upward jumps in `y_norm` after LOE unless early empirical data for Scenario 2 clearly supports a recovery.
    - [ ] Ensure curves do not exceed a reasonable cap (e.g. 1.2–1.5 × pre-entry normalisation) except in justified early-LOE artefacts.
  - [ ] Implement a **simple smoothing filter** (e.g. moving average or low-order polynomial fit) on the predicted curve per series:
    - [ ] Run only if smoothing **improves the metric in CV**; otherwise keep raw predictions.
  - [ ] Experiment with **monotonic constraints in GBMs**:
    - [ ] E.g. enforce that higher `months_postgx` and higher `n_gxs` should not increase `y_norm` on average.
    - [ ] Validate whether constrained trees improve robustness in high-erosion segments without hurting overall score.

---

## 9. Testing & Quality Assurance

### 9.1 Unit Tests (`tests/test_smoke.py`)
- [x] Test imports work ✅
- [x] Test set_seed reproducibility ✅
- [x] Test config loading ✅
- [x] Test data loading ✅
- [x] Test panel building ✅
- [x] Test feature leakage prevention ✅
- [x] Test model interface ✅
- [x] Test metric computation ✅
- [x] Test validation split ✅
- [x] Test submission format ✅
- [x] Test sample weights ✅
- [ ] **Add tests for feature engineering correctness**:
  - [ ] `make_features` respects scenario cutoffs (`months_postgx < 0` for S1, `< 6` for S2)
  - [ ] `mode="test"` does not create `y_norm` or modify `volume`
  - [ ] Early-erosion features only appear for Scenario 2
- [ ] **Add test for scenario detection**
- [ ] **Add test for inverse transform** (y_norm → volume)
- [ ] **Add test for edge cases** (empty series, missing data)

### 9.2 CLI Smoke Tests
- [ ] **Add Pytest that calls `python -m src.train --help`** using subprocess
  - [ ] Assert exit code 0
  - [ ] Assert help text includes key arguments (`--scenario`, `--model`, `--data-config`, etc.)
- [ ] **Add Pytest that calls `python -m src.inference --help`** using subprocess
  - [ ] Assert exit code 0
  - [ ] Assert help text includes key arguments (`--model-s1`, `--model-s2`, `--output`, etc.)

### 9.3 Leakage Test Strengthening
- [ ] **Add a test that attempts to include `bucket`, `y_norm`, or `mean_erosion` in features** and confirms that `split_features_target_meta` / leakage checks throw or log errors

### 9.4 Integration Tests
- [ ] **Implement end-to-end smoke test on tiny subset (≈10 series)** that:
  - [ ] Loads configs and data
  - [ ] Builds panel, runs `handle_missing_values` + `compute_pre_entry_stats`
  - [ ] Builds features for S1 (`mode="train"`)
  - [ ] Trains a small CatBoost model (few iterations)
  - [ ] Generates predictions
  - [ ] Constructs `df_actual`, `df_pred`, `df_aux`
  - [ ] Confirms metric is finite and within a reasonable range
- [ ] **Test both scenarios** separately (S1 and S2)
- [ ] **Test Colab notebook** (`notebooks/colab/main.ipynb`) runs without errors

### 9.5 Data Validation Tests
- [ ] **Test for data drift** between train and test
- [ ] **Test for leakage** in features
- [ ] **Test submission file against template**:
  - [ ] Check column order
  - [ ] Check that dtypes are compatible (e.g., `volume` is numeric, no strings)
  - [ ] Confirm no extra columns
- [ ] **Test metric calculation** against provided example

### 9.6 Code Quality
- [ ] **Run pylint/flake8** on all Python files
- [ ] **Add type hints** to all functions
- [ ] **Add docstrings** to all public functions
- [ ] **Review and clean up** unused code

---

## 10. Documentation & Presentation

### 10.1 Code Documentation
- [ ] **Update README.md** with latest instructions
- [ ] **Document all config options** in configs/README.md
- [ ] **Add inline comments** for complex logic
- [ ] **Create API documentation** (sphinx/mkdocs)
- [ ] **Document correct usage of `metric_calculation.py`** in `README.md`:
  - [ ] Write a small wrapper script or document the correct command that imports `metric_calculation.py` and passes `submission`, `auxiliar_metric_computation.csv`, and required arguments correctly

### 10.2 Methodology Documentation
- [ ] **Document feature engineering** rationale
- [ ] **Document model selection** process
- [ ] **Document validation strategy** and results
- [ ] **Document hyperparameter choices**

### 10.3 Phase 2 Presentation
- [ ] **Prepare slide deck** (15-20 slides)
- [ ] **Include problem understanding**
- [ ] **Include methodology overview**
- [ ] **Include key insights from EDA**
- [ ] **Include model performance summary**
- [ ] **Include feature importance analysis**
- [ ] **Include business recommendations**
- [ ] **Prepare for Q&A** (common questions)
- [ ] **Business impact framing for generic erosion**
  - [ ] Quantify, with simple assumptions, how a given improvement in the forecasting metric (e.g. ΔRMSE) translates into:
    - [ ] More accurate planning of **brand defense strategies** (discounts, contracting).
    - [ ] Better **inventory and supply chain management** post-LOE.
    - [ ] Improved **financial forecasting** at portfolio level.
  - [ ] Prepare 1–2 concrete **"what-if" scenarios**:
    - [ ] E.g. "If we underestimated erosion for this blockbuster by 20% in the first 6 months, the revenue miss would be X million; our model reduces that error by Y%."
  - [ ] Highlight 2–3 **actionable insights** derived from feature importance & segmentation:
    - [ ] Countries or therapeutic areas where erosion is systematically faster/slower.
    - [ ] Patterns of competition (number of generics, hospital_rate) that strongly influence brand decline.

### 10.4 Reproducibility
- [ ] **Create requirements.txt** with pinned versions
- [ ] **Document random seeds** used
- [ ] **Document hardware** (CPU/GPU, RAM)
- [ ] **Create run script** for full reproduction
- [ ] **Test on fresh environment**

### 10.5 Notebooks Overview
- [ ] **`notebooks/00_eda.ipynb`**: Basic distributions, erosion curves by bucket / ther_area
- [ ] **`notebooks/01_feature_prototype.ipynb`**: Prototype `make_features` for both scenarios, leakage checks
- [ ] **`notebooks/02_train.ipynb`**: End-to-end training on a subset, metrics (rename from `01_train.ipynb`)
- [ ] **`notebooks/03_model_sanity.ipynb`**: Model sanity checks and validation (rename from `02_model_sanity.ipynb`)
- [ ] **`notebooks/colab/main.ipynb`**: Colab-friendly full workflow

---

## 11. Colab/Production Readiness

### 11.1 Google Colab Setup
- [ ] **Test main.ipynb** in Colab environment
- [ ] **Verify Drive mounting** works
- [ ] **Verify data paths** are correct
- [ ] **Add GPU detection** and utilization
- [ ] **Add memory management** (garbage collection)
- [ ] **Add progress bars** for long operations
- [ ] **Ensure `notebooks/colab/main.ipynb` implements documented end-to-end workflow**:
  - [ ] Clone repo, install `env/colab_requirements.txt`, mount Drive, run training + submission

### 11.2 Environment Management
- [ ] **Update colab_requirements.txt** with exact versions
- [ ] **Test environment.yml** creates working env
- [ ] **Document Python version** requirement (3.8+)
- [ ] **Test on Mac/Linux/Windows**

### 11.3 Performance Optimization
- [ ] **Profile training time** and memory usage
- [ ] **Optimize data loading** (lazy loading, chunking)
- [ ] **Enable GPU** for CatBoost/XGBoost if available
- [ ] **Use parquet** instead of CSV for speed
- [ ] **Add caching** for computed features

---

## 12. Competition Strategy

### 12.1 Leaderboard Management
- [ ] **Track all submissions** with scores and notes
- [ ] **Analyze score variance** between local CV and LB
- [ ] **Identify potential overfitting** to leaderboard
- [ ] **Save submissions** for final selection

### 12.2 Time Management
- [ ] **Allocate time for EDA**: 20%
- [ ] **Allocate time for Feature Engineering**: 25%
- [ ] **Allocate time for Modeling**: 30%
- [ ] **Allocate time for Tuning/Ensemble**: 15%
- [ ] **Allocate time for Documentation**: 10%

### 12.3 Risk Mitigation
- [ ] **Keep simple baseline** as fallback
- [ ] **Save multiple model versions**
- [ ] **Test submission upload** before deadline
- [ ] **Have backup submission ready**

### 12.4 Final Checklist (Pre-Submission)
- [ ] **Validate submission format** one more time
- [ ] **Check all series are predicted**
- [ ] **Check predictions are reasonable** (sanity check)
- [ ] **Record final submission details**
- [ ] **Backup all code and models**

### 12.5 Final-Week Execution Playbook
> **Rationale**: Many teams lose points due to chaos in the final week. A frozen playbook prevents last-minute mistakes.

- [ ] **Final-week execution playbook**
  - [ ] Define a **"frozen best config"** (model type, hyperparameters, features, CV scheme, ensemble weights) at least 48 hours before the deadline.
  - [ ] Reserve the last 24–36 hours for:
    - [ ] Re-running the frozen config with **multiple seeds** (e.g. 3–5 seeds) and ensembling the resulting models.
    - [ ] Generating 3–5 **carefully chosen submissions**:
      - [ ] Best CV score.
      - [ ] Slightly underfitted version (simpler model / fewer trees).
      - [ ] Slightly overfitted version (more trees / more complex ensemble).
      - [ ] A more conservative model focused on bucket 1 / early windows.
    - [ ] Verifying all submissions with the official metric script and format checks.
  - [ ] Strict rule: **no major changes to features, CV, or architecture** in the last 24 hours—only controlled variations around the frozen best config.

### 12.6 External Data & Constraints Check (see also 0.3)
- [ ] **Re-verify external data rules** before final submission
  - [ ] Confirm no prohibited external data sources are used.
  - [ ] Ensure all data used is from official competition sources or explicitly allowed.
  - [ ] Document any external data used in `docs/planning/approach.md`.

---

## Progress Tracking

> **Note**: This table tracks **core functionality** status. Update after each implementation round. Many unchecked items in the TODO are enhancements beyond the working baseline.

| Phase | Status |
|-------|--------|
| Design & Consistency (Section 0) | ✅ Implemented |
| Data Pipeline (Section 2) | ✅ Core Implemented |
| Feature Engineering (Section 3) | ✅ Core Implemented |
| Model Development | ✅ Core Implemented |
| Training Pipeline (Section 5) | ✅ Core Complete (CV, metadata, CLI, config weights) |
| Validation & Evaluation (Section 6) | ✅ Core Implemented |
| Inference & Submission | ✅ Core Implemented |
| Testing | ✅ 87 passed, 0 skipped, 0 warnings |
| Documentation | 🔄 In Progress |
| Optimization | ⏳ Not Started |
| Presentation | ⏳ Not Started |

---

## Quick Commands

```bash
# Run smoke tests
pytest tests/test_smoke.py -v

# ============================================================
# Data Build Commands (pre-build panels and feature matrices)
# ============================================================

# Build train panel (raw → interim)
python -m src.data --split train --data-config configs/data.yaml

# Build test panel (raw → interim)
python -m src.data --split test --data-config configs/data.yaml

# Build train Scenario 1 features (interim → processed)
python -m src.data --split train --scenario 1 --mode train --data-config configs/data.yaml --features-config configs/features.yaml

# Build train Scenario 2 features (interim → processed)
python -m src.data --split train --scenario 2 --mode train --data-config configs/data.yaml --features-config configs/features.yaml

# Build test Scenario 1 features (interim → processed)
python -m src.data --split test --scenario 1 --mode test --data-config configs/data.yaml --features-config configs/features.yaml

# Build test Scenario 2 features (interim → processed)
python -m src.data --split test --scenario 2 --mode test --data-config configs/data.yaml --features-config configs/features.yaml

# Force rebuild (ignore cache)
python -m src.data --split train --scenario 1 --mode train --force-rebuild --data-config configs/data.yaml --features-config configs/features.yaml

# ============================================================
# Training Commands (CLI flags locked — see top of file)
# ============================================================

# Train Scenario 1 model (CatBoost)
python -m src.train --scenario 1 --model catboost --model-config configs/model_cat.yaml --data-config configs/data.yaml --run-config configs/run_defaults.yaml

# Train Scenario 2 model (CatBoost)
python -m src.train --scenario 2 --model catboost --model-config configs/model_cat.yaml --data-config configs/data.yaml --run-config configs/run_defaults.yaml

# Train with forced data rebuild
python -m src.train --scenario 1 --model catboost --force-rebuild --model-config configs/model_cat.yaml --data-config configs/data.yaml --run-config configs/run_defaults.yaml

# ============================================================
# Inference & Submission Commands
# ============================================================

# Generate submission
python -m src.inference --model-s1 artifacts/model_s1.cbm --model-s2 artifacts/model_s2.cbm --output submissions/submission.csv --data-config configs/data.yaml

# Validate submission with official metric (IMPLEMENTATION TASK: create wrapper)
# Current status: metric_calculation.py usage needs a wrapper script.
# Target: python scripts/validate_submission.py --submission submissions/submission.csv --aux submissions/auxiliar_metric_computation.csv
# See section 10.1 for wrapper implementation task.

# ============================================================
# Visualization & Plot Generation Commands
# ============================================================

# Generate all plots for a given run
python -m src.plots --run-id 2025_11_28_001 --data-config configs/data.yaml --run-config configs/run_defaults.yaml --phase all

# Generate only EDA plots (distributions, erosion curves)
python -m src.plots --run-id 2025_11_28_001 --data-config configs/data.yaml --run-config configs/run_defaults.yaml --phase eda

# Generate training/validation curves
python -m src.plots --run-id 2025_11_28_001 --data-config configs/data.yaml --run-config configs/run_defaults.yaml --phase train

# Generate prediction vs actual & error analysis plots
python -m src.plots --run-id 2025_11_28_001 --data-config configs/data.yaml --run-config configs/run_defaults.yaml --phase test
```

---

## Notes

### Key Insights from Documentation
1. **Bucket is NEVER a feature** - it's computed from target, using it causes leakage
2. **Official metric weights early months heavily** - 50% of score from months 0-5 (S1) or 6-11 (S2)
3. **Bucket 1 (high erosion) is 2× weighted** - focus on predicting high-erosion series correctly
4. **Series-level validation is critical** - never mix months from same series across train/val
5. **Target is y_norm, not volume** - model predicts normalized value, then inverse transform

### Known Issues
- [ ] NN model may need feature normalization (GBMs don't need it)
- [ ] Sample weights may need fine-tuning to match official metric exactly
- [ ] Some series may have missing pre-entry data

### Contact & Resources
- Competition Platform: [Check competition site]
- Official Metric Calculator: `docs/guide/metric_calculation.py`
- Submission Template: `docs/guide/submission_template.csv`
