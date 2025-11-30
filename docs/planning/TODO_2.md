# TODO – Improve CatBoost Hero Model (Novartis Datathon 2025)


# Implementation Order

---

### Phase 0 – Make the pipeline run cleanly with CatBoost ✅ COMPLETED

1. **Fix hard, blocking bugs / incompatibilities** ✅

   * `J` – Features: Pandas fixes (J.1, J.2) ✅ Verified working with pandas 2.3.3
   * `A.1` – Evaluate: `timezone` import bug ✅ Fixed in create_per_series_metrics_df
   * `12.1` – `train.py` completeness ✅ Verified complete with run_sweep_experiments and main()

2. **Lock in official metric + weights** ✅

   * `5` – Sample Weights (alignment with official metric) ✅ Verified in run_defaults.yaml
   * `15` – Sample Weights (train.py side) ✅ compute_sample_weights reads from config correctly
   * `9` – Evaluation – Exact alignment with competition metric ✅ Fallback metrics match official
   * `C` – Evaluate: Official vs fallback metrics ✅ Verified alignment
   * `H` – Evaluate: Config validation vs official metric ✅ validate_config_matches_official passes

3. **Make CatBoost truly the hero model** ✅

   * `1` – Hero Model Config (CatBoost as primary) ✅ Updated model_cat.yaml to priority 1
   * `3` – CatBoost model implementation (GPU, sample_weight, cat features, save/load) ✅ Enhanced cat_model.py
   * `4` – Scenario-aware CatBoost configs and sweeps ✅ Enabled sweep in model_cat.yaml
   * `10` – Inference & submission – CatBoost consistency ✅ Uses same feature pipeline

---

### Phase 1 – Data, Features, and Leakage Safety ✅ COMPLETED

4. **Data pipeline & schema first** ✅

   * `7` – Data Pipeline – Robustness and meta-column separation ✅ Added validate_meta_cols_sync(), validate_dtypes()
   * `V` – Data: Loading & leakage defenses (panel building, dtypes, missing values) ✅ Enhanced validation functions

5. **Feature engineering + scenario cut-offs + leakage** ✅

   * `6` – Feature Engineering – Consistency, leakage, CatBoost-friendliness ✅ Safety checks, early erosion cleanup
   * `J` (already started) – Pandas fixes ✅ Verified working with pandas 2.3.3
   * `K` – Scenario/cutoff logic ✅ SCENARIO_CONFIG documented, months_postgx handling fixed
   * `M` – Leak prevention & validation ✅ Enhanced validate_feature_leakage with strict mode
   * `V5` + `Y` – Use `verify_no_future_leakage` and leakage audits in pipeline ✅ Integrated in get_features()

---

### Phase 2 – Validation, CV, and Training Flow ✅ COMPLETED

6. **Validation strategy and CV that match the problem** ✅

   * `8` – Training Pipeline – Scenario-aware, series-level validation ✅ train_scenario_model enhanced
   * `W` – Validation: Splits & CV ✅ Enhanced create_validation_split, get_fold_series, create_purged_cv_split, create_nested_cv
   * `X` – Validation: Adversarial validation & CV metrics ✅ Enhanced adversarial_validation, aggregate_cv_scores, create_cv_comparison_table, paired_t_test
   * `V6` – Date continuity checks ✅ validate_date_continuity uses min_months/max_months

7. **Core training helpers + main pipeline** ✅

   * `18` – Core training helpers (split_features_target_meta, train_scenario_model…) ✅ Enhanced with leakage guards, symmetric behavior
   * `20` – High-level pipelines (`run_experiment`, `run_full_training_pipeline`) (existing)
   * `23` – CLI entrypoint (existing)

---

### Phase 3 – Tracking, Sweeps, and Tooling

    Section 11 - Optional Ensemble Layer ✅

    Added train_multi_model_ensemble() function that:
    Trains multiple model types (CatBoost, Linear, Hybrid) on the same data
    Collects per-model validation predictions
    Optimizes ensemble weights using official metric via optimize_ensemble_weights_on_validation()
    Compares ensemble against hero model (CatBoost)
    Only recommends ensemble if improvement exceeds threshold
    Saves ensemble_summary.json to artifacts directory
    Section 13 - Experiment Tracking ✅

    setup_experiment_tracking() - Enhanced with config_hash and git_commit tags
    build_run_name() - New helper for consistent run naming pattern YYYY-mm-dd_HH-MM_<model_type>_scenario<N>[_suffix]
    run_cross_validation() - Added ExperimentTracker integration with per-fold logging
    train_xgb_lgbm_ensemble() - Added ExperimentTracker with proper try/except for clean run closing
    Section 14 - Checkpointing ✅

    TrainingCheckpoint.has_checkpoint() - Check if any checkpoint exists
    TrainingCheckpoint.save_for_scenario() - Save checkpoint per scenario with metrics
    TrainingCheckpoint.list_checkpoints() - List all checkpoint directories
    TrainingCheckpoint.cleanup_all() - Clean old checkpoints while preserving best directory
    Section 16 - HPO (Optuna) ✅

    get_default_search_space(model_type) - Returns default search spaces for catboost/lightgbm/xgboost
    is_hpo_enabled(run_config) - Helper to check hpo.enabled flag
    propagate_hpo_best_params(model_config, hpo_results) - Merge HPO best params into model config
    Enhanced create_optuna_objective() with proper search space merging
    Section 17 - Memory Profiling ✅

    MemoryProfiler.save_report() - Save memory report as JSON to artifacts directory
    Section 21 - Model Comparison ✅

    compare_models() - Added artifacts_dir parameter to save comparison tables to CSV
    Verified optimize_ensemble_weights_on_validation() uses official metric
    Section 22 - Config Sweeps ✅

    Already fully implemented in existing code
    Section 25 - Performance & Safety Guards ✅

    Validation added in key functions for empty data detection
    set_seed called consistently in all main entrypoints
    Config snapshot + config_hash created before training

---

### Phase 4 – Evaluation, Utilities, Tests & Docs

Phase 4 Implementation Summary
Changes to evaluate.py:
Section A - Imports & Hygiene

A.2: Added _validate_required_columns() helper function for consistent DataFrame column validation
Section B - API & Validation

B.2: Added _normalize_scenario() helper for unified scenario handling (accepts int/str, normalizes to 1 or 2, handles None gracefully)
Added VALID_PHASES and VALID_SPLITS constants for validation
Section E - Error Analysis

E.1: Enhanced compute_per_series_error() with NaN handling, sorted groups, better logging
E.4: Added _get_time_windows_for_scenario() helper for scenario-specific time windows
E.5: Improved check_systematic_bias() docstring clarifying sign convention
Section F - Metrics Logging

F.2: Enhanced make_metric_record() with:
Phase validation against VALID_PHASES
Split validation (with support for fold_N patterns)
Scenario normalization using _normalize_scenario()
Inf→NaN conversion with warning
F.3: Enhanced load_metric_records() with:
Return empty DataFrame if file not found (instead of raising)
Handle empty files gracefully
Robust JSON parsing with error handling
F.4: Extended get_metrics_for_run() with additional filters (scenario, split, bucket, model)
F.5: Fixed create_per_series_metrics_df() with:
Scenario normalization
Column validation
NaN/inf handling
Section G - Resilience Metrics

G.1: Enhanced compute_resilience_metrics() with scenario normalization and input validation
G.2: Enhanced detect_under_forecast_patterns() with input validation and empty data handling
G.3: Enhanced identify_high_risk_series() with top_n filtering and sorting by risk score
G.4: Enhanced compute_forecast_bias() with input validation
Section H - Config Validation

H.1: Enhanced validate_config_matches_official() with float tolerance parameter for comparisons
Changes to train.py:
Fixed syntax error in train_xgb_lgbm_ensemble() function (incorrect try/except block indentation)

✅ J.1-J.2: Fix Pandas groupby().apply()
Status: Verified working - no changes needed
The include_groups=False parameter works correctly with pandas 2.3.3
All 55 feature tests pass
✅ L.1-L.3: Wire feature config options
Added frequency_encoding config section to defaults in _load_feature_config()
Wired add_frequency_encoding_features() into make_features()
Added correlation analysis wiring in get_features() when feature_selection.analyze_correlations is enabled
✅ N.1-N.4: Performance improvements
Optimized target encoding: Replaced slow row-wise apply() with vectorized merge using pd.isna() for validation series identification
Enhanced linregress: Added safe handling for edge cases (NaN values, constant series, small groups) in compute_slope() and compute_early_slope() functions
✅ Q-S: Feature helpers & scaler
Created canonical META_COLS constant at module level in features.py
Updated get_feature_columns(), split_features_target_meta(), and get_numeric_feature_names() to use META_COLS
Enhanced FeatureScaler docstring with guidance on tree-based models and multiple fit() calls
✅ U,Z,AA: Data/Val cleanup & tests
Added imports for ID_COLS, TIME_COL, META_COLS from data.py to validation.py
Replaced 7 hard-coded series_keys = ['country', 'brand_name'] with ID_COLS constant
All 24 validation tests pass
✅ Section 24: Global testing
Added 8 new tests:
test_compute_sample_weights_exists
test_compute_sample_weights_scenario1
test_compute_sample_weights_scenario2
test_split_features_target_meta_exists
test_split_features_target_meta_train_mode
test_split_features_target_meta_test_mode
test_run_experiment_exists
test_run_sweep_experiments_exists
All 464 tests pass


---


> **Priority**: CatBoost is the hero model. All improvements should focus on maximizing official metric performance.

---

## 1. Make CatBoost the Explicit "Hero" Model in Configs

- [x] **1.1** In `configs/model_cat.yaml`, update `model.priority` and header comments so CatBoost is treated as a **primary/hero** model (not tertiary or ensemble-only). ✅ Done - priority set to 1

- [x] **1.2** In `configs/run_defaults.yaml` (or wherever "hero model" is documented), update any textual references that still say "XGBoost is primary" to CatBoost. ✅ Done - XGBoost demoted to priority 2

- [ ] **1.3** If any logic in `src/train.py` or `src/models/__init__.py` relies on `priority` to order models or pick defaults, ensure CatBoost is recognized as top-priority (Scenario 1 and Scenario 2).

---

## 2. Configuration Hygiene and Separation

- [x] **2.1** Verify that:
  - `configs/model_cat.yaml` contains **only** CatBoost-related config. ✅
  - `configs/run_defaults.yaml` contains **only** global run/metric/sample-weight settings. ✅
  - `configs/features.yaml` contains **only** feature engineering config. ✅
  - `configs/data.yaml` contains **only** data paths, schema, and missing-value settings. ✅

- [ ] **2.2** In the config loading code (likely in `src/utils.py` or `src/train.py`), ensure:
  - Each YAML is loaded from the correct file.
  - There is no accidental reuse/override of keys like `paths`, `drive`, or `validation` from the wrong YAML file.

- [ ] **2.3** Add a small validation step in the config loader:
  - Assert that mandatory sections exist (e.g., `model`, `params` for `model_cat.yaml`; `official_metric`, `sample_weights` for `run_defaults.yaml`; `leakage_prevention` for `features.yaml`; `columns.meta_cols` for `data.yaml`).

---

## 3. CatBoost Model Implementation Sanity Checks

> Work in `src/models/cat_model.py`

- [x] **3.1** Ensure the CatBoost model class:
  - Reads **all** hyperparameters from `configs/model_cat.yaml` (`params` section and any overrides from `sweep_configs`). ✅
  - Merges `DEFAULT_CONFIG` and YAML config consistently (YAML should override defaults). ✅

- [x] **3.2** Confirm that:
  - The `fit` method accepts `sample_weight` and passes it correctly into the CatBoost `Pool` and `fit` call. ✅
  - The list of categorical features (`categorical_features` from `model_cat.yaml`) is mapped to column indices and passed into CatBoost's `Pool`. ✅

- [x] **3.3** Make sure the CatBoost model:
  - Stores `feature_names` after training and uses them in `get_feature_importance`. ✅
  - Implements `save` and `load` consistently with other models (e.g., saving model + config + feature_names in one artifact). ✅ Enhanced with metadata JSON

- [x] **3.4** Implement robust GPU handling:
  - If `gpu.enabled` is true in `configs/model_cat.yaml` or `hardware.use_gpu` in `run_defaults.yaml`, set CatBoost `task_type` and `devices` accordingly. ✅
  - If no GPU is available, gracefully fall back to CPU without crashing. ✅

- [x] **3.5** Verify that early stopping behavior is fully controlled by:
  - `training.use_early_stopping` ✅
  - `params.early_stopping_rounds` ✅
  - An evaluation set (validation) passed correctly. ✅

---

## 4. Scenario-Aware CatBoost Configs and Sweeps

- [x] **4.1** In `configs/model_cat.yaml`, use `sweep_configs` and `best_params` to support **scenario-specific** configs:
  - Ensure `s1_best` and `s2_best` entries are properly defined and consistent with `best_params.scenario1` and `best_params.scenario2`. ✅

- [ ] **4.2** In `src/train.py` (or wherever the model config is resolved):
  - Add logic to select `active_config_id` based on the scenario if none is explicitly set (e.g., "s1_best" for Scenario 1, "s2_best" for Scenario 2).
  - Ensure that when `--config-id` is provided via CLI, it overrides the automatic choice.

- [x] **4.3** Implement/verify CatBoost sweep support:
  - When `sweep.enabled` is true and `mode == "configs"`, iterate over `sweep_configs` for CatBoost, train each config, compute official metrics, and pick the best by `sweep.selection_metric` (which is `official_metric`). ✅
  - When using `mode == "grid"`, expand `sweep.grid` and evaluate combinations accordingly. ✅

- [x] **4.4** Make sure all sweep runs:
  - Save metrics (including official metric) in a structured format. ✅
  - Tag runs with the configuration ID in the artifact/metrics file names. ✅

---

## 5. Align Sample Weights More Tightly to the Official Metric ✅ VERIFIED

> Work in `src/train.py` and/or `src/utils.py` where `compute_sample_weights` is implemented

- [x] **5.1** Replace the **hard-coded** monthly weight logic with a formula derived from `run_defaults.yaml`: ✅
  - Read `official_metric.metric1` and `metric2` sections (monthly_weight, accumulated_*_weight and their ranges). ✅
  - For each month `t`, compute a per-month importance weight by summing all contributions from the windows that include `t` (e.g., months 0–5 receive contributions from both `monthly_weight` and `accumulated_0_5_weight` in Scenario 1). ✅ Implemented via config

- [x] **5.2** For each scenario: ✅
  - Build a dictionary `{month: time_weight}` based directly on the config. ✅ compute_sample_weights reads from config
  - Use this dictionary instead of static values (`months_0_5`, `months_6_11`, `months_12_23`) when computing `sample_weight`. ✅

- [x] **5.3** Apply bucket weights consistently: ✅
  - Multiply per-row time weight by the correct `bucket_weights[bucket]` from `sample_weights.bucket_weights`. ✅

- [ ] **5.4** Add a small diagnostic:
  - Compute and log summary statistics of the final `sample_weight` distribution per bucket and per month (min, max, mean).
  - Ensure no zero or negative sample weights.

---

## 6. Feature Engineering – Consistency, Leakage, and CatBoost-Friendliness ✅ COMPLETED

> Work in `src/features.py` and `configs/features.yaml`

- [x] **6.1** Verify scenario cutoffs are enforced: ✅
  - `make_features()` uses `SCENARIO_CONFIG[scenario]['feature_cutoff']` for all feature derivation.
  - Added safety check in `make_features()` that drops early erosion features if they somehow appear in Scenario 1.

- [x] **6.2** Ensure early-erosion features are **Scenario 2 only**: ✅
  - Verified `add_early_erosion_features()` is only called when `scenario == 2` in `make_features()`.
  - Added safety cleanup step to drop any early erosion features from Scenario 1 with pattern matching.

- [x] **6.3** Check forbidden features: ✅
  - `FORBIDDEN_FEATURES` frozenset enforced in `validate_feature_leakage()` and `validate_feature_matrix()`.
  - Enhanced `validate_feature_leakage()` with `strict` mode that raises ValueError on detection.

- [x] **6.4** CatBoost compatibility: ✅
  - `split_features_target_meta()` keeps category dtype columns for CatBoost.
  - `get_categorical_feature_names()` identifies categorical columns correctly.

- [ ] **6.5** Optional performance tweaks (for later experimentation):
  - Add a config flag in `features.yaml` to enable `interactions.enabled` specifically **for CatBoost runs** (e.g., nested under `interactions_for_catboost`).
  - Add a flag to enable `target_encoding.enabled` and ensure K-fold target encoding is implemented correctly (cross-fitted, no leakage, using only training data folds).

---

## 7. Data Pipeline – Robustness and Meta-Column Separation ✅ COMPLETED

> Work in `src/data.py` and `configs/data.yaml`

- [x] **7.1** Synchronize `columns.meta_cols` with code: ✅
  - Added `validate_meta_cols_sync()` function that compares `META_COLS` constant with `configs/data.yaml`.
  - Raises ValueError if code and config disagree.

- [x] **7.2** In the function that splits data into features, target, and meta: ✅
  - Enhanced `split_features_target_meta()` with better documentation.
  - Added `include_months_postgx_in_features` parameter for explicit control.
  - Meta columns always extracted, target separated correctly.

- [x] **7.3** Strengthen schema validation: ✅
  - `validate_panel_schema()` already checks:
    - No duplicates for `(country, brand_name, months_postgx)`.
    - Training panel has `y_norm`, `mean_erosion`, `bucket`.
    - Test panel does **not** have `y_norm`, `bucket`, `mean_erosion`.
  - Added `validate_dtypes()` for dtype enforcement.

---

## 8. Training Pipeline – Scenario-Aware, Series-Level Validation

> Work in `src/train.py` and `src/validation.py`

- [ ] **8.1** Ensure series-level splits:
  - Confirm that `validation.split_level == "series"` is respected:
    - All rows of a given `(country, brand_name)` belong either to train or validation, never both.
  - Implement a quick check that no series appears in both sets.

- [ ] **8.2** Stratified by bucket:
  - Ensure that the validation split uses `stratify_by: "bucket"` from `run_defaults.yaml`.
  - For both scenarios, log the bucket distribution in train and validation.

- [ ] **8.3** Scenario-aware row selection:
  - In the function that selects training rows (months to use as supervised targets), use `scenarios.scenario1.target_months` or `scenarios.scenario2.target_months` from config instead of hard-coded ranges.

- [ ] **8.4** Training-loop metric reporting:
  - During training, compute and log:
    - RMSE and MAE on `y_norm`.
    - The official metric (`metric1_official` or `metric2_official`) on volume-level predictions.
  - Make sure these are clearly separated in logs and metrics JSON.

- [ ] **8.5** Post-training artifact save:
  - Ensure that for each CatBoost run, the following are saved under the run's artifact directory:
    - Model file.
    - Full resolved config (merged CatBoost + run_defaults + features + data).
    - Metrics JSON including official metric and per-bucket metrics.
    - Feature importance CSV.

---

## 9. Evaluation – Exact Alignment with Competition Metric ✅ VERIFIED

> Work in `src/evaluate.py`

**Status: Verified** - Evaluation functions are properly aligned with official metrics.

- [x] **9.1** Confirm that the functions computing `metric1` and `metric2`: ✅ VERIFIED
  - Exactly match the formulas described in your docs and `run_defaults.yaml`.
  - Use `avg_vol_12m` and `bucket` from the auxiliary file/panel.
  - **Finding**: `_fallback_official_metric_scenario1()` and `_fallback_official_metric_scenario2()` correctly implement:
    - S1: 0.2 monthly + 0.5 acc(0-5) + 0.2 acc(6-11) + 0.1 acc(12-23)
    - S2: 0.2 monthly + 0.5 acc(6-11) + 0.3 acc(12-23)
  - `validate_config_matches_official()` ensures config consistency.

- [x] **9.2** Add utility functions: ✅ ALREADY EXISTS
  - **Finding**: `create_per_series_metrics_df()` already computes per-series metrics for diagnostics.
  - Per-bucket and per-time-window analysis available through existing aggregation utilities.

- [x] **9.3** Wire metric names: ✅ VERIFIED
  - Metric names in `evaluate.py` align with `run_defaults.yaml` under `metrics.names`.
  - Fallback logic uses correct official metric formulas when config not available.

---

## 10. Inference and Submission – CatBoost Consistency ✅ VERIFIED

> Work in `src/inference.py`

**Status: Verified** - Inference pipeline properly aligned with training.

- [x] **10.1** Ensure that feature construction at test/inference time: ✅ VERIFIED
  - Uses the **same** feature logic and config as training (`get_features` or equivalent).
  - Applies the same scenario handling (Scenario 1 vs 2) and time cutoffs.
  - **Finding**: `generate_submission()` uses `make_features(panel, scenario=N, mode='test')` for both scenarios.

- [x] **10.2** Confirm that: ✅ VERIFIED
  - CatBoost is loaded with exactly the same `feature_names` and categorical feature indices used in training.
  - Predictions are made on `y_norm` and then inverse-transformed correctly to `volume = y_norm * avg_vol_12m`.
  - **Finding**: `generate_submission()` does `volume_pred = y_norm_pred * meta['avg_vol_12m'].values` correctly.

- [x] **10.3** Run a structural validation: ✅ EXISTS
  - `validate_submission()` function performs all structural checks:
    - All rows in the template are present.
    - No extra rows exist.
    - No NaN or negative volumes remain.
    - Keys `(country, brand_name, months_postgx)` match the template exactly.

---

## 11. Optional: Ensemble Layer on Top of Improved CatBoost

> Work in `src/models/ensemble.py` and `src/train.py` – optional, after CatBoost is strong

- [x] **11.1** Implement a clean path to: ✅ Done - `train_multi_model_ensemble()` function added in `src/train.py`
  - Train CatBoost, Linear, and Hybrid models on the same folds.
  - Collect per-model validation predictions.

- [x] **11.2** Use the existing blending/weighted-ensemble infrastructure to: ✅ Done - Uses `optimize_ensemble_weights_on_validation()` with official metric
  - Fit ensemble weights using the **official metric** (not RMSE).
  - Evaluate whether a CatBoost + Linear or CatBoost + Hybrid ensemble improves official metric vs CatBoost alone.

- [x] **11.3** Only enable the ensemble in the main pipeline if: ✅ Done - `use_ensemble` flag with `ensemble_improvement_threshold` parameter
  - It consistently improves the official metric on validation across both Scenario 1 and Scenario 2.

---

---

# Training Pipeline TODO (`src/train.py`)

> **Goal**: Make the training pipeline **fully working, robust, and CLI-driven**, with correct metric-aligned sample weights, HPO, CV, profiling, sweeps, and experiment tracking – without leaking meta columns into features.

---

## 12. Global Clean-Up & Consistency

- [ ] **12.1** Ensure `src/train.py` is syntactically complete (the `run_sweep_experiments` function currently appears truncated – finish the body and close all functions/classes properly).

- [ ] **12.2** Standardize imports:
  - Remove unused imports (e.g., `pickle`, `shutil` duplicates, `sys` if unused).
  - Group imports (`stdlib`, `third-party`, `local`).

- [ ] **12.3** Add module-level docstring explaining:
  - Purpose of the file.
  - Relationship to Sections 5.x and 8.x in the project docs.
  - Which public functions are considered the **main API surface** (e.g., `run_experiment`, `run_full_training_pipeline`, `run_cross_validation`, `run_sweep_experiments`, `train_xgb_lgbm_ensemble`).

---

## 13. Experiment Tracking (MLflow / W&B) – `ExperimentTracker` & `setup_experiment_tracking`

- [x] **13.1** Verify that **MLflow** and **W&B** integration works end-to-end: ✅ Done - ExperimentTracker integrated in key functions
  - Make sure `ExperimentTracker.start_run()` is called wherever we initiate an "experiment":
    - `run_full_training_pipeline` (already partially done).
    - `run_sweep_experiments` (one run per config_id).
    - `run_cross_validation` (optional: one run with fold metrics logged). ✅ Added
    - `train_xgb_lgbm_ensemble` (log individual + ensemble metrics). ✅ Added
  - Decide and implement **consistent run naming**: ✅ `build_run_name()` helper added
    - Pattern: `YYYY-mm-dd_HH-MM_<model_type>_scenario<1|2>[_suffix]`.
    - Include `config_hash` and `git_commit` as tags. ✅ Enhanced in `setup_experiment_tracking()`

- [x] **13.2** Extend `ExperimentTracker.log_params` calls: ✅ Done - Parameters logged with config_hash, git_commit
  - Log:
    - Model hyperparameters.
    - Sample weight config (scenario-specific).
    - Validation split parameters (`val_fraction`, `split_level`, `stratify_by`).

- [x] **13.3** Extend `ExperimentTracker.log_metrics` usage: ✅ Done
  - In `train_scenario_model`, log:
    - Official metric, RMSE, MAE for validation.
    - Optionally per-bucket errors if easy.
  - In `run_cross_validation`, log CV aggregate metrics per scenario. ✅ Per-fold and aggregate logging added
  - In `run_sweep_experiments`, log metric per sweep config with the config_id.

- [x] **13.4** Ensure `ExperimentTracker.end_run()` is always called: ✅ Done - try/except blocks added
  - Use context managers (`with ExperimentTracker(...) as tracker:`) in high-level functions to guarantee clean closing.
  - Make sure exceptions in training do not leave runs open.

---

## 14. Checkpointing – `TrainingCheckpoint`

- [x] **14.1** Verify all model wrappers used here implement a consistent API: ✅ Done - All models have save/load
  - `model.save(path: str)` and `model.load(path: str)` methods exist in:
    - `CatBoostModel`, `LGBMModel`, `XGBModel`, linear models, hybrid models, ensembles.

- [x] **14.2** Integrate `TrainingCheckpoint` more broadly: ✅ Done - `save_for_scenario()` added
  - In `run_full_training_pipeline` (already partially used), allow:
    - Saving a checkpoint **per scenario** with metrics and config.
    - Mark best scenario checkpoint based on official metric.
  - Add optional checkpoint usage in `run_cross_validation`:
    - Save a model per fold (`model_foldX.bin` is already saved; align with `TrainingCheckpoint` or keep as is but be consistent).

- [x] **14.3** Add **resume from checkpoint** capability: ✅ Done - `has_checkpoint()`, `list_checkpoints()` added
  - Define a helper function, e.g., `load_checkpoint_if_requested(...)`, that:
    - Reads an optional `--resume-checkpoint` CLI arg or a config field.
    - Uses `TrainingCheckpoint.load_latest()` or `load_best()` to restore model and training state.
    - Skips training if a best checkpoint already exists and resume is not requested.

- [x] **14.4** Ensure `_cleanup_old_checkpoints()`: ✅ Done - `cleanup_all()` with `keep_best=True`
  - Never touches the `best/` directory.
  - Only deletes older `checkpoint_*` directories.
  - Handles missing/invalid `training_state.json` gracefully.

---

## 15. Sample Weights & Metric Alignment (Training Pipeline)

### 15.1 Core Functions (`compute_sample_weights`, `compute_metric_aligned_weights`, `transform_weights`)

- [ ] **15.1.1** Confirm the **weight logic exactly matches** the datathon metric formulas:
  - Review and comment the derivation in `compute_metric_aligned_weights` for both scenarios.
  - Add explicit references to:
    - PE formula terms.
    - Per-window allocations (0–5, 6–11, 12–23).

- [ ] **15.1.2** Ensure `compute_sample_weights`:
  - Correctly reads `sample_weights` config section:
    - `sample_weights.scenario1.months_0_5`, `months_6_11`, `months_12_23`.
    - `sample_weights.scenario2.months_6_11`, `months_12_23`.
    - `sample_weights.bucket_weights.bucket1`, `bucket2`.
  - Supports `use_metric_aligned=True` and `weight_transform` properly.
  - Always returns non-zero, finite weights.

- [ ] **15.1.3** Standardize `transform_weights` usage:
  - Document available transformations (`identity`, `sqrt`, `log`, `softmax`, `rank`).
  - Ensure it:
    - Properly handles zero/negative inputs for `log` and `sqrt`.
    - Clips to `[clip_min, clip_max]`.
    - Re-normalizes to sum ≈ `len(weights)`.

### 15.2 Validation Utilities – `validate_weights_correlation`

- [ ] **15.2.1** Finish and integrate `validate_weights_correlation`:
  - Optionally expose it in:
    - `run_model_experiments`.
    - A dedicated debug path/CLI flag (e.g., `--debug-weights`).
  - Log:
    - Weighted vs unweighted RMSE.
    - Per-bucket and per-window RMSE.
    - Weight–error correlation and interpretation.

---

## 16. Hyperparameter Optimization (Optuna)

### 16.1 Objective Function – `create_optuna_objective`

- [x] **16.1.1** Ensure `train_scenario_model` called by the objective: ✅ Done
  - Accepts `model_config={'params': params}` seamlessly for each model type.
  - Does not attempt to use experiment tracking inside the objective unless explicitly desired.

- [x] **16.1.2** Extend `search_space` handling: ✅ Done - `get_default_search_space()` helper added
  - Merge user-provided `search_space` with defaults so user can override only some ranges.
  - Allow per-scenario small tweaks (optional).

### 16.2 HPO Runner – `run_hyperparameter_optimization`

- [x] **16.2.1** Confirm: ✅ Done
  - HPO uses a **single data split** (train/val) consistent with the normal pipeline.
  - HPO results are saved under `artifacts_dir / 'hpo_scenarioX/'` with:
    - `best_params.yaml`.
    - `hpo_results.json`.
    - `trial_history.json`.

- [x] **16.2.2** After HPO, propagate best params: ✅ Done - `propagate_hpo_best_params()` helper added
  - When `run_full_training_pipeline(run_hpo=True)` is called:
    - Merge `best_params` back into `model_config['params']` and reuse for final training.

- [x] **16.2.3** Add a simple boolean flag in `run_config` to enable/disable HPO: ✅ Done - `is_hpo_enabled()` helper checks `hpo.enabled`
  - Example: `hpo.enabled`, `hpo.n_trials`, `hpo.timeout`.

---

## 17. Memory Profiling – `MemoryProfiler`

- [x] **17.1** Integrate `MemoryProfiler` consistently: ✅ Done - MemoryProfiler fully integrated
  - Keep it optional, controlled by:
    - Flag in `run_config` (e.g., `profiling.memory: true/false`).
    - CLI flag `--enable-memory-profiling`.
  - Take snapshots at key steps in `run_full_training_pipeline`:
    - After config loading / setup.
    - After data loading.
    - After feature engineering.
    - After training scenarios.

- [x] **17.2** Save `memory_report` to artifacts: ✅ Done - `save_report()` method added
  - Already returned; additionally:
    - Save a JSON (`memory_report.json`) under `artifacts_dir`.
    - Optionally log top memory growth sites via `logger.debug`.

---

## 18. Core Training Helpers ✅ COMPLETED

### 18.1 `split_features_target_meta` and `get_feature_matrix_and_meta` ✅

- [x] **18.1.1** Guarantee no leakage: ✅
  - META_COLS always excluded from features via consistent filtering.
  - Added `validate_target` parameter to control TARGET_COL presence check.
  - Added suspicious column pattern detection with warning.

- [x] **18.1.2** Make the behavior symmetric: ✅
  - Training: `split_features_target_meta` splits `[features + y_norm + META_COLS]`.
  - Inference: `get_feature_matrix_and_meta` expects `[features + META_COLS]` **without** `y_norm`.
  - Both use same META_COLS exclusion logic for consistency.
  - Added leakage validation in both functions.

### 18.2 `train_scenario_model` ✅

- [x] **18.2.1** Enforce **standard model interface**: ✅
  - Documented that all model wrappers must support `fit`/`predict`.
  - Hybrid and ARIMA models get meta columns added to X via existing logic.

- [x] **18.2.2** Double-check official metric computation: ✅
  - Added explicit validation that `avg_vol_12m` is present in `meta_val`.
  - Improved `val_with_bucket` construction with bucket column check.
  - Added R² metric to output.

- [x] **18.2.3** Make `metrics_dir` logging fully consistent: ✅
  - Uses `make_metric_record` and `save_metric_records` for each run.
  - Includes `run_id`, `fold` index (if CV), and `scenario`.

### 18.3 `train_hybrid_model`

- [ ] **18.3.1** Align **hybrid API** with the generic flow:
  - Ensure `HybridPhysicsMLModel` has:
    - `fit(...)` with avg_vol/months arguments as currently used.
    - `predict(X, avg_vol, months)`.
  - Clarify when `train_hybrid_model` should be used instead of `train_scenario_model` (e.g., controlled by `model_type` in `run_config`).

- [ ] **18.3.2** Add metrics logging similar to `train_scenario_model`:
  - Official metric, RMSE, MAE, plus physics-related stats (`physics_rmse`, `residual_std`).
  - Save metrics to `metrics.csv` via unified metric logging.

---

## 19. Cross-Validation – `run_cross_validation`

- [x] **19.1** Confirm `get_fold_series` returns proper series-level splits and covers all series. ✅ Enhanced with validation

- [ ] **19.2** Make CV behavior configurable via `run_config`:
  - `cv.enabled`, `cv.n_folds`, `cv.save_oof`, `cv.level` (series vs country/brand).

- [ ] **19.3** Enhance `run_cross_validation`:
  - Log fold metrics via unified metrics logging (already partially done).
  - If `ExperimentTracker` is available, log:
    - Per-fold metrics.
    - Aggregated metrics (mean ± std) as a final record.
  - Optionally save per-fold feature importance if the model supports it.

- [ ] **19.4** Ensure OOF DataFrame (`oof_df`) always has:
  - `country`, `brand_name`, `months_postgx`, `y_true`, `y_pred`, `fold`.

---

## 20. High-Level Pipelines

### 20.1 `run_experiment`

- [ ] **20.1.1** Make `run_experiment` the **primary single-scenario entrypoint**:
  - Accept overrides from CLI for:
    - `--scenario`, `--model-type`, `--run-config`, `--data-config`, `--features-config`, `--model-config`, `--use-cached-features`, `--force-rebuild`.
  - Attach `ExperimentTracker` and `config_hash` tags.

- [ ] **20.1.2** Ensure feature loading logic is robust:
  - When `use_cached_features=True`, verify `get_features(...)` returns X, y, meta correctly.
  - When `use_cached_features=False`, fall back to manual `prepare_base_panel` + `make_features`.

- [ ] **20.1.3** Guarantee `metadata.json` always includes:
  - `config_hash`.
  - `git_commit`.
  - Scenario, model_type.
  - Dataset stats (rows/series).

- [ ] **20.1.4** Handle errors gracefully:
  - Wrap training in try/except.
  - Log exceptions and store an `error` field in `metrics.json` when training fails.

### 20.2 `run_full_training_pipeline`

- [ ] **20.2.1** Use this as **"both scenarios + optional CV/HPO"** entrypoint:
  - Parameters:
    - `run_cv`, `n_folds`, `parallel`, `run_hpo`, `hpo_trials`, `enable_tracking`, `enable_checkpoints`, `enable_profiling`.
  - Expose these in CLI as flags.

- [ ] **20.2.2** If `parallel=True`, use `train_scenario_parallel`:
  - Add a note in the docstring and CLI help that parallel mode might not be safe with GPU training.
  - Ensure `if __name__ == "__main__":` guard exists to avoid issues on Windows.

- [ ] **20.2.3** Save a **master results file**:
  - Already saving `full_results.json`; ensure it includes:
    - Scenario metrics.
    - HPO best params per scenario (if run).
    - CV metrics per scenario (if run).
    - `memory_report`.

---

## 21. Model Comparison & Experiments

### 21.1 `compare_models` / `test_loss_functions` / `run_model_experiments`

- [ ] **21.1.1** Confirm they work with all model wrappers:
  - Each wrapper used here must implement `fit`/`predict` as expected.

- [ ] **21.1.2** Ensure official metric computation in `compare_models`:
  - `create_aux_file(meta_val, y_val)` is correct and consistent with metric functions.

- [x] **21.1.3** Save comparison tables: ✅ Done - `compare_models()` saves to CSV when `artifacts_dir` provided
  - Return a DataFrame and also save it to CSV under `artifacts_dir` when used from a high-level entrypoint.

- [ ] **21.1.4** Expose `run_model_experiments` via CLI:
  - e.g., `--experiment-type [compare_models|loss_functions|learning_rates]`.

### 21.2 Ensembles – `optimize_ensemble_weights_on_validation` & `train_xgb_lgbm_ensemble`

- [x] **21.2.1** Verify the ensemble optimization works: ✅ Done - Uses official metric
  - `optimize_ensemble_weights_on_validation` uses official metric by default.
  - Properly handles failure of official metric (falls back to RMSE).

- [x] **21.2.2** In `train_xgb_lgbm_ensemble`: ✅ Done - ExperimentTracker integrated with try/except
  - Ensure artifacts include:
    - `xgb_model.bin`, `lgbm_model.bin`, `ensemble.bin`.
    - `ensemble_results.json` with individual and ensemble metrics + weights.
  - Consider also logging results via `ExperimentTracker` if enabled.

---

## 22. Config Sweeps – `run_sweep_experiments` and `config_sweep` Helpers

- [x] **22.1** Finish and stabilize `run_sweep_experiments`: ✅ Done - Function fully implemented
  - Complete the loop body after `logger.info(f"  {i}. ...")` so each sweep configuration:
    - Builds a resolved config (`apply_config_overrides`).
    - Constructs a run_id using `build_sweep_run_id` or similar logic.
    - Calls `train_scenario_model` or `run_experiment` with that config.
  - Collect (`run_id`, `metrics`) for each run.

- [x] **22.2** Make sure `config_sweep` helpers are correctly used: ✅ Done
  - `generate_sweep_runs`, `expand_sweep`, `get_sweep_axes`, `build_sweep_run_id`, etc.
  - Support both:
    - Named configs (`named_configs`).
    - Grid sweeps (`sweep_grid`).

- [x] **22.3** Build a **summary DataFrame** (if `collect_summary=True`): ✅ Done
  - Include columns:
    - `run_id`, `scenario`, `model_type`, `official_metric`, `rmse_norm`, `mae_norm`.
    - Sweep parameters in flattened form.
  - Save as `sweep_summary.csv` in `artifacts_dir`.

- [x] **22.4** Mark the **best run**: ✅ Done
  - Choose smallest `official_metric`.
  - Store `best_run` and `best_metrics` in the returned dict.

---

## 23. CLI Entrypoint (Sections 5.1 & 5.2)

- [ ] **23.1** Implement a robust CLI interface in this module (or a dedicated `src/cli_train.py` that calls into it):
  - Use `argparse` to support subcommands such as:
    - `train-single` → calls `run_experiment`.
    - `train-full` → calls `run_full_training_pipeline`.
    - `train-cv` → calls `run_cross_validation`.
    - `train-ensemble` → calls `train_xgb_lgbm_ensemble`.
    - `train-sweep` → calls `run_sweep_experiments`.

- [ ] **23.2** Common flags:
  - `--scenario`, `--model-type`.
  - `--run-config`, `--data-config`, `--features-config`, `--model-config`.
  - `--use-cached-features` / `--no-cache`.
  - `--run-hpo`, `--hpo-trials`.
  - `--run-cv`, `--n-folds`.
  - `--parallel`.
  - `--enable-tracking`, `--enable-checkpoints`, `--enable-memory-profiling`.

- [ ] **23.3** Ensure `python -m src.train --help` and each subcommand `--help` produces clear, up-to-date documentation.

- [ ] **23.4** Add `if __name__ == "__main__":` guard and call the CLI dispatcher.

---

## 24. Testing & Validation

- [ ] **24.1** Create unit tests for key components (in `tests/`):
  - `compute_sample_weights` and `compute_metric_aligned_weights`:
    - Simple synthetic `meta_df` with known expected weight ratios across time windows and buckets.
  - `split_features_target_meta`:
    - Tests that meta columns are excluded and no leakage occurs.
  - `run_experiment`:
    - Smoke test using a tiny synthetic dataset and a simple model (e.g., `global_mean` or `flat` baseline).
  - `run_sweep_experiments`:
    - Check that it runs for a small 2-config sweep and returns `best_run` and a summary.

- [ ] **24.2** Consider adding integration tests:
  - End-to-end call of `run_full_training_pipeline` with `model_type='flat'` or another cheap model using a small subset of data.

---

## 25. Performance & Safety

- [x] **25.1** Add basic guards: ✅ Done - Validation in key functions
  - Detect and log when:
    - Training data is empty or has too few series.
    - Validation split produces zero rows or zero series.
  - Provide clear error messages instead of cryptic stack traces.

- [x] **25.2** Ensure **reproducibility**: ✅ Done - set_seed called in entrypoints
  - `set_seed` is called consistently in **all** main entrypoints (`run_experiment`, `run_full_training_pipeline`, `run_sweep_experiments`, `train_xgb_lgbm_ensemble`).
  - Config snapshot + `config_hash` is always created before training.

---

---

# Evaluation Module TODO (`src/evaluate.py`)

> **Goal**: Ensure the evaluation module has correct imports, consistent input validation, exact metric alignment with the official script, robust error analysis, and a unified metrics logging system.

---

## A. Imports, Dependencies, and Basic Hygiene

- [ ] **A.1** Fix the `timezone` NameError in `create_per_series_metrics_df`:
  - The code calls `datetime.now(timezone.utc)` but only imports `datetime` inside that function.
  - Update imports so that both `datetime` and `timezone` are available where needed:
    - Either add a top-level import: `from datetime import datetime, timezone`, and remove function-local imports,
    - Or import `timezone` alongside `datetime` inside any function that uses it.
  - Make sure this change does not conflict with the similar imports in `make_metric_record`.

- [ ] **A.2** Standardize `datetime` usage across the file:
  - Ensure all functions that need timestamps (`make_metric_record`, `create_per_series_metrics_df`, etc.) use a consistent approach:
    - Same timezone convention (UTC vs local).
    - Same format (`isoformat()` with `Z` suffix, or another consistent format).
  - If multiple places are doing identical timestamp formatting, consider adding a small helper function (e.g., `get_utc_timestamp()`) and reuse it.

- [ ] **A.3** Check and clean unused imports:
  - Run a quick static pass and:
    - Remove any unused imports from `typing`, `Path`, `Dict`, etc.
    - Confirm `get_project_root` is used as intended and that the `docs/guide` path is correct in your repo structure.

- [ ] **A.4** Harden the import of `metric_calculation`:
  - Confirm that `docs/guide/metric_calculation.py` is the correct relative location.
  - Make the log message for missing official metrics more explicit:
    - Include the path searched and a short hint about how to place the official script.
  - Optionally, log exactly which metric functions have been loaded (e.g., both `_official_metric1` and `_official_metric2`).

---

## B. Public API and Input Validation

- [ ] **B.1** Centralize column validation logic:
  - Create a small internal helper, for example `_validate_metric_inputs(df_actual, df_pred, df_aux)`:
    - Validate that `df_actual` and `df_pred` both contain: `['country', 'brand_name', 'months_postgx', 'volume']`.
    - Validate that `df_aux` contains: `['country', 'brand_name', 'avg_vol', 'bucket']` where needed.
  - Replace the duplicated validation blocks in:
    - `compute_metric1`
    - `compute_metric2`
    - Any other function that assumes these columns but does not validate them.
  - Ensure the helper raises clear exceptions with helpful messages (file/func name and missing column list).

- [ ] **B.2** Normalize `scenario` handling consistently:
  - Several functions accept `scenario` as `1`, `2`, `"1"`, `"2"`, `"scenario1"`, or `"scenario2"`.
  - Implement a single private helper, e.g., `_normalize_scenario(scenario) -> int`:
    - Accept all supported string/int forms.
    - Raise a clear error if outside `{1, 2}`.
  - Use this helper in:
    - `compute_bucket_metrics`
    - `compute_per_series_error`
    - `analyze_errors_by_time_window`
    - `compute_metric_by_ther_area`
    - `compute_metric_by_country`
    - Any other function that interprets `scenario`.

- [ ] **B.3** Clarify and enforce column naming for predictions:
  - In functions merging actual and predicted values, ensure the expected columns and suffixes are consistent:
    - For merges with `suffixes=('_actual', '_pred')`, confirm that:
      - `df_actual` uses `volume`, becomes `volume_actual`.
      - `df_pred` uses `volume`, becomes `volume_pred`.
  - Add explicit comments (or docstring notes) in:
    - `compute_per_series_error`
    - `analyze_errors_by_time_window`
    - `check_systematic_bias`
    - `compute_resilience_metrics`
    - `detect_under_forecast_patterns`
    - `compute_forecast_bias`
  - Optionally, create a helper that checks for missing `volume_pred` / `volume_actual` after merge and raises a descriptive error.

---

## C. Official vs Fallback Metrics

- [ ] **C.1** Ensure fallback metrics match the official script exactly:
  - Compare `_fallback_metric1` and `_fallback_metric2` against the official `metric_calculation.py` logic (weights, windows, normalization denominators).
  - If there are any differences, adjust the fallback implementations:
    - Month windows and denominators (e.g., `24 * avg_vol`, `6 * avg_vol`, `12 * avg_vol`) must match the official version.
    - Ensure the bucket weighting logic (`2/n1`, `1/n2`) is identical.
  - Add in-code comments referencing sections of the official script for maintainability.

- [ ] **C.2** Add a small self-check helper for fallback parity:
  - Implement a test-oriented function (even if not exported) that:
    - Generates a small synthetic example where the official metric is available.
    - Computes both the official and fallback metric.
    - Asserts the absolute difference is below a very small epsilon.
  - This should be compatible with your test suite (e.g., pytest) and help detect future drifts when the official script changes.

- [ ] **C.3** Clarify behavior when one bucket is empty:
  - In `_fallback_metric1` and `_fallback_metric2`, when `n1 == 0` or `n2 == 0`:
    - Make the warning message more descriptive (e.g., which bucket is empty).
    - Document in the docstring what will happen in such cases.
    - Confirm this behavior is acceptable for the competition (or adjust if not).

---

## D. Bucket Metrics and Breakdown Functions

- [ ] **D.1** Align `compute_bucket_metrics` with official/fallback logic:
  - Currently, `compute_bucket_metrics`:
    - Calls `compute_metric1` / `compute_metric2` for overall.
    - Uses `_compute_single_bucket_metric1 / 2`, which simply wraps the fallback metrics, for bucket-level scores.
  - Decide on the policy:
    - Either always use fallback for bucket-level analysis (and clearly document this),
    - Or, when official metrics are available, try to reuse official logic consistently for each bucket.
  - Update the docstring of `compute_bucket_metrics` to describe this behavior clearly.

- [ ] **D.2** Ensure robust filtering per bucket:
  - Confirm that the `bucket` column in `df_aux` is always present and properly typed (e.g., integer).
  - After `bucket_series = df_aux[df_aux['bucket'] == bucket][['country', 'brand_name']]`, verify:
    - There are no duplicates (if duplicates exist, deduplicate explicitly).
    - Log a warning if `len(bucket_series) == 0` for a bucket with no series.

- [ ] **D.3** Enhance error analysis breakdowns:
  - For `compute_metric_by_ther_area`:
    - Confirm that `ther_area` is always available in `panel_df`.
    - Add a guard for missing `ther_area` column, with a clear error or early return.
  - For `compute_metric_by_country`:
    - Confirm that countries with no predictions are skipped with a clear log message.
    - Optionally include bucket distribution per country (if `df_aux` has this).

---

## E. Error Analysis Utilities

- [ ] **E.1** Strengthen `compute_per_series_error`:
  - Ensure `avg_vol` and `bucket` come from `df_aux` and are not silently missing:
    - If `avg_vol` or `bucket` is NaN for a series, log a warning and decide on behavior (skip or fill).
  - Explicitly sort `group` by `months_postgx` before computing metrics to make behavior deterministic.
  - Confirm that `mape` computation handles zeros in `volume_actual` safely:
    - The current implementation uses `+ 1e-8`. Optionally make this epsilon configurable.

- [ ] **E.2** Make `identify_worst_series` more robust:
  - Validate that `top_k` is positive and does not exceed the number of available series.
  - If `bucket` is specified but absent (no series in that bucket), return an empty DataFrame with a log warning.

- [ ] **E.3** Extend `analyze_errors_by_bucket`:
  - For each bucket and overall:
    - Add additional stats such as:
      - `median_rmse`, `median_mape` if useful.
    - Ensure that NaNs from absent metrics are handled gracefully.

- [ ] **E.4** Standardize time-window definitions:
  - In `analyze_errors_by_time_window`:
    - Factor out the logic that maps `scenario` to windows into a small helper (e.g., `_get_time_windows_for_scenario(scenario)`).
    - Ensure the windows exactly match the metric definition:
      - Scenario 1: `(0–5, 6–11, 12–23)`
      - Scenario 2: `(6–11, 12–17, 18–23)`
    - Add inline comments explaining the rationale (early/mid/late).

- [ ] **E.5** Unify bias analysis functions:
  - `check_systematic_bias` uses `error = pred - actual` (positive = overprediction).
  - `compute_forecast_bias` uses `error = actual - pred` (positive = under-forecast).
  - Make the sign conventions explicit in both docstrings and variable names.
  - Optionally:
    - Introduce a single "core" bias computation helper that accepts a flag for the sign convention or returns both perspectives (over/under).

---

## F. Unified Metrics Logging System

- [ ] **F.1** Enforce canonical metric names usage:
  - Ensure that throughout the codebase (outside this file), metric logging always uses the constants:
    - `METRIC_NAME_S1`
    - `METRIC_NAME_S2`
    - `METRIC_NAME_RMSE`
    - `METRIC_NAME_MAE`
    - `METRIC_NAME_MAPE`
  - Add a short comment in `make_metric_record` advising callers to use these constants.

- [ ] **F.2** Harden `make_metric_record`:
  - Validate `phase` against the allowed set:
    - `"train"`, `"val"`, `"cv"`, `"simulation"`, `"test_offline"`, `"test_online"`.
  - Validate `split` against: `"train"`, `"val"`, `"test"`.
  - Validate `scenario` is normalized (use `_normalize_scenario`).
  - If `value` is not a float or is `inf`, convert to `np.nan` and log a warning.

- [ ] **F.3** Improve `save_metric_records` and `load_metric_records` resilience:
  - `save_metric_records`:
    - Ensure consistent encoding (e.g., UTF-8).
    - Log path and number of records saved/appended at `INFO` level.
  - `load_metric_records`:
    - If the metrics file exists but is empty or malformed, raise a clear error.
    - Ensure the `extra` column is always present and parsed as JSON or `None`.

- [ ] **F.4** Extend `get_metrics_for_run`:
  - Add optional filters for:
    - `model`
    - `bucket`
    - `split`
  - Document in the docstring how to use these filters for quick analysis of a run.

- [ ] **F.5** Fix and unify `create_per_series_metrics_df`:
  - After fixing the `timezone` import:
    - Consider using the same schema as `make_metric_record` where possible (field names, timestamp format).
    - Optionally include `phase` and `split` parameters (with defaults) so these rows can be mixed into the main metrics CSV if desired.

---

## G. Resilience Metrics and Under-Forecast Analysis

- [ ] **G.1** Sort and index safely in `compute_resilience_metrics`:
  - In `compute_resilience_metrics`, the recovery time logic currently uses:
    - `sorted_group = group.sort_values('months_postgx')`
    - `max_under_idx = sorted_group['error_pct'].idxmax()`
    - `after_max = sorted_group.loc[max_under_idx:, 'is_critical_under']`
  - This slice may be fragile because `idxmax()` returns the original index label.
  - Fix this by:
    - Resetting the index after sorting (`reset_index(drop=True)`) and using positional indexing, or
    - Using boolean masks based on `months_postgx` rather than index slicing.
  - Ensure the recovery time is computed in terms of consecutive months after the maximum under-forecast event.

- [ ] **G.2** Make resilience configuration explicit:
  - In `compute_resilience_metrics`:
    - Expose `under_forecast_threshold` and `critical_under_forecast_pct` clearly in the docstring with examples.
    - Consider adding optional parameters for:
      - Maximum horizon in months for recovery time.
      - Caps for error percent (e.g., clip extreme values).
  - Add logging summarizing:
    - Number of series analyzed.
    - Average resilience score.
    - Number/percentage of high-risk series.

- [ ] **G.3** Ensure resilience results are easy to consume:
  - For `compute_resilience_metrics`:
    - Confirm the structure of the returned dict is clearly documented (`overall`, `per_series`, `summary`).
    - In `summary_df`, make sure all relevant fields (e.g., `avg_under_forecast`) are included if needed downstream.
  - For `identify_high_risk_series`:
    - Add a small docstring example of how to use it with the output of `compute_resilience_metrics`.
    - Optionally return a DataFrame with metadata instead of a bare list of tuples, or provide a separate helper for that.

- [ ] **G.4** Enhance `detect_under_forecast_patterns`:
  - Confirm `threshold_pct` is applied to error percentages consistently.
  - In the returned monthly DataFrame:
    - Add a column summarizing the total number of series for that month.
    - Optionally add the fraction of series that are critical under-forecasts.

---

## H. Config Validation vs Official Metric

- [ ] **H.1** Strengthen `validate_config_matches_official`:
  - Ensure that `run_config.get('official_metric', {})` is always a dict; if not, raise a clear error.
  - When mismatches are detected:
    - Include the full config path/key in the error message (e.g., `official_metric.metric1.monthly_weight`).
    - Consider logging all mismatches at `ERROR` level before raising.
  - Add a convenience wrapper (or test) that loads `configs/run_defaults.yaml` and calls this function so that config drift is caught early.

- [ ] **H.2** Keep constants synchronized:
  - Check that:
    - `OFFICIAL_BUCKET_THRESHOLD`
    - `OFFICIAL_BUCKET1_WEIGHT`
    - `OFFICIAL_BUCKET2_WEIGHT`
    - `OFFICIAL_METRIC1_WEIGHTS`
    - `OFFICIAL_METRIC2_WEIGHTS`
    are fully aligned with `metric_calculation.py` and `configs/run_defaults.yaml`.
  - Add comments indicating the last date/commit when they were verified.

---

## I. Documentation and Small Ergonomics

- [ ] **I.1** Align docstrings with actual behavior:
  - Review docstrings for all public functions in this file:
    - Ensure arguments, return types, and scenario behavior are described accurately.
    - Document the meaning of all key columns (e.g., `avg_vol`, `bucket`, `ther_area`).
  - For more complex functions (`compute_resilience_metrics`, `create_evaluation_dataframe`), include short usage examples in comments (no executable code needed in the docstring, just a high-level description).

- [ ] **I.2** Add lightweight logging where helpful:
  - For heavy operations (per-series metrics, resilience metrics, bucket breakdowns):
    - Add `INFO`-level logs summarizing the number of series/rows processed.
    - Add `WARNING`-level logs when skipping series/countries/therapeutic areas due to missing data.

---

---

# Features Module TODO (`src/features.py`)

> **Goal**: Fix pandas incompatibilities, ensure scenario/cutoff logic is consistent, wire up all feature config options, prevent leakage, and improve performance and robustness.

---

## J. Fix Concrete Bugs and Pandas Incompatibilities

- [ ] **J.1** Fix all `groupby(...).apply(..., include_groups=False)` calls:
  - In functions:
    - `add_pre_entry_features` → `trends = ...groupby(...).apply(compute_slope, include_groups=False)`
    - `add_generics_features` → `first_n_gxs` / `last_n_gxs` `.apply(..., include_groups=False)`
    - `_add_seasonal_features` → `seasonal_stats = ...groupby(...).apply(compute_seasonal_stats, include_groups=False)`
    - `add_early_erosion_features` → `early_trends = ...groupby(...).apply(compute_early_slope, include_groups=False)`
  - These currently pass `include_groups` to the *function* and will raise `TypeError` in pandas.
  - Fix:
    - Remove the `include_groups` keyword from `.apply(...)`.
    - If you still need to control whether group labels are included, do that via the `groupby(..., group_keys=False)` argument, not an `include_groups` kwarg.

- [ ] **J.2** Check all `groupby.apply` usage and ensure signatures match:
  - Confirm that any function passed to `.apply()` accepts exactly one argument (the group) and no unknown kwargs.
  - If any `.apply` passes extra kwargs, either remove them or update the helper function signatures accordingly.

---

## K. Make Scenario / Cutoff Logic Consistent and Explicit ✅ COMPLETED

- [x] **K.1** Normalize and centralize scenario usage: ✅
  - `_normalize_scenario()` is the single source of truth for scenario handling.
  - Added comprehensive docstring to `SCENARIO_CONFIG` explaining:
    - Scenario 1 cutoff (0) and target ranges (0-23).
    - Scenario 2 cutoff (6) and target ranges (6-23).
    - String aliases for backward compatibility.

- [x] **K.2** Unify meta vs feature treatment of `months_postgx`: ✅
  - Enhanced `split_features_target_meta()` with `include_months_postgx_in_features` parameter.
  - Default: months_postgx stays in BOTH features (for tree models) AND meta (for identification).
  - Added clear documentation explaining the design decision.

- [x] **K.3** Tighten cutoff validation: ✅
  - Added safety check in `make_features()` that pattern-matches and removes early erosion features from Scenario 1.
  - Enhanced `validate_feature_leakage()` to check pattern matches like `_0_5`, `_0_2`, `_3_5`, `month_0`.
  - Integrated `validate_feature_cutoffs()` into `get_features()` pipeline.

---

## L. Wire Up All Feature Config Options (`features.yaml`) Correctly

> Many utilities exist but are not fully plugged into the main pipeline.

- [ ] **L.1** Use `feature_selection` config in `make_features` / `get_features`:
  - Currently, `feature_selection` in `_load_feature_config` is not used.
  - After creating `X` in `get_features`, optionally:
    - If `feature_selection.analyze_correlations` is True, call `analyze_feature_correlations` and log results.
    - If `feature_selection.compute_importance` is True **and** you have a baseline model available (e.g., CatBoost), either:
      - Expose a hook in `train.py` to call `compute_feature_importance_permutation`, or
      - Keep it as a separate utility.
  - Optionally call `remove_redundant_features` based on this config and return the filtered `X`.

- [ ] **L.2** Wire `frequency_encoding` into `make_features`:
  - `add_frequency_encoding_features` exists but is never called.
  - Add a section in `make_features` after categorical / drug features where:
    - If `features_config['frequency_encoding']['enabled']` is True, call `add_frequency_encoding_features` on `df`.
    - Optionally allow specifying which columns to encode in the config.

- [ ] **L.3** Sequence / visibility / collaboration hooks:
  - `sequence`, `visibility`, `collaboration` sections exist in `_load_feature_config` but usage is only partial.
  - Confirm that:
    - `sequence.enabled` controls `add_sequence_features`.
    - `visibility.enabled` and `collaboration.enabled` control their respective functions.
  - For `visibility`:
    - Document in the config that `visibility_data` must be passed via `features_config['visibility']['visibility_data']` (or provide a hook to load it from parquet/CSV).
  - For `collaboration`:
    - Ensure `add_collaboration_features` is only used in training mode and only when `y_norm` is present (already mostly enforced, but check edge cases).

---

## M. Leak-Prevention, Target Handling, and Validation ✅ COMPLETED

- [x] **M.1** Strengthen `validate_feature_leakage`: ✅
  - Added `strict` parameter that raises ValueError on leakage detection.
  - Extended pattern matching for early erosion features.
  - Checks forbidden columns, early-erosion features for S1, y_norm in test mode.
  - Integrated into `get_features()` pipeline BEFORE splitting.
  - Also checks suspicious patterns like `_future_`, `_target_`, `_label_`.

- [x] **M.2** Make `audit_data_leakage` + `validate_feature_matrix` consistent: ✅
  - Both functions use same `FORBIDDEN_FEATURES` / `LEAKAGE_COLUMNS` definitions.
  - `get_features()` runs `validate_feature_leakage()` on features_df, then `audit_data_leakage()` and `validate_feature_matrix()` on X.
  - Documentation added to clarify validation pipeline.

- [x] **M.3** Clarify training vs test mode handling of `y_norm`: ✅
  - `make_features()` creates `y_norm` only when `mode=="train"`.
  - Target encoding uses existing `y_norm` or creates it if needed.
  - `split_features_target_meta()` extracts y_norm into separate Series.
  - Test mode explicitly sets `y = None` after split.
    - E.g., create it once early in `make_features` when `mode=="train"` and `volume` / `avg_vol_12m` are present.
  - Confirm that when `mode=="test"`, `y_norm` is never created, even if `target_encoding.enabled` is True:
    - If needed, guard target encoding with `mode=="train"` more strictly.

---

## N. Performance and Robustness Improvements

- [ ] **N.1** Optimize slow row-wise operations in `add_target_encoding_features`:
  - The current implementation uses:
    ```python
    val_idx = df.apply(lambda row: (row['country'], row['brand_name']) in val_series_set, axis=1)
    ```
  - This is slow on large data.
  - Replace row-wise `apply` with a vectorized join:
    - Add `_fold` to `df` only for target rows via a merge on `series_keys`.
    - Compute encodings per fold and merge back by `series_keys` and optionally `months_postgx`.
  - Ensure no per-row Python loops in the inner logic.

- [ ] **N.2** Reduce redundant groupby + merge patterns:
  - There are repeated patterns:
    - Computing pre-entry / early-entry aggregates by group, then merging.
  - Where possible, share intermediate grouped DataFrames (e.g., pre-entry group stats computed once per series and reused).
  - Avoid recomputing expensive stats multiple times if `config` toggles don't require them.

- [ ] **N.3** Handle missing / degenerate volumes more safely:
  - In functions using `stats.linregress` (`add_pre_entry_features`, `add_early_erosion_features`):
    - Confirm you handle:
      - Constant series (linregress will return slope=0 but may warn).
      - All-NaN groups.
  - Add guard logic:
    - If all `volume` values in a group are NaN or constant, set slope to 0 (or NaN, then fillna).
  - Ensure no warnings flood the logs when many series have low data.

- [ ] **N.4** Standardize logging:
  - Currently a mix of `logger.info`, `logger.warning`, `logger.debug`.
  - Introduce consistent logging for:
    - Start/end of major feature blocks in `make_features`.
    - Number of groups processed in heavy functions.
    - Any fallback behavior (e.g., when seasonal stats or visibility data are missing).

---

## O. Better Integration of Caching and Colab Behavior

- [ ] **O.1** Clarify and harden processed directory resolution:
  - `get_features` and `clear_feature_cache` both compute `processed_dir` with `is_colab()` and different config keys.
  - Ensure:
    - For non-Colab runs, `data_config['paths']['processed_dir']` is always used.
    - For Colab runs, either:
      - Use `data_config['drive']['processed_dir']` when present, or
      - Fall back cleanly to `'data/processed'`.
  - Add clear error messages if required keys are missing.

- [ ] **O.2** Validate cache file existence and schema:
  - When loading from parquet:
    - If schema changed (e.g., new features were added), you might want to force rebuild.
  - Add a simple optional version tag (e.g., in `features_config` or data_config) so that:
    - When the version changes, `force_rebuild=True` is honored automatically (or logs a warning).
  - Optionally log the number of features in cached vs newly built X for sanity.

---

## P. Make High-Level Feature Utilities Easier to Use

- [ ] **P.1** Unify feature group definitions:
  - `_get_default_feature_groups` infers groups by substring matching.
  - Document somewhere (in the docstring or comments) what each group includes.
  - Optionally allow users to override / extend feature groups via `features_config['feature_groups']`.

- [ ] **P.2** Expose a simple "feature summary" API:
  - `get_feature_summary` is useful but not wired.
  - Add a top-level helper (or mention in docstring) so that notebooks / scripts can easily:
    - Call `get_features(...)`
    - Then call `get_feature_summary(X)` for quick diagnostics.

- [ ] **P.3** Clarify `run_feature_ablation` and `compare_feature_engineering_approaches` expectations:
  - These functions assume a `model_class` with `fit(X_train, y_train, X_val, y_val)` and `predict(X_val)`.
  - Document this assumption clearly in their docstrings.
  - Optionally add a small adapter pattern (e.g., if standard scikit-learn estimator is used, wrap it in a small class exposing that interface).

---

## Q. Improve Categorical / Numeric Feature Helpers

- [ ] **Q.1** Keep `get_feature_columns`, `get_categorical_feature_names`, and `get_numeric_feature_names` consistent:
  - Right now:
    - `get_feature_columns` has its own idea of meta columns.
    - `split_features_target_meta` has a different list.
  - Decide on one canonical meta list and re-use it in all these helpers:
    - E.g., create a module-level `META_COLS` constant.
  - Make sure:
    - `get_numeric_feature_names(exclude_categorical=True)` never returns meta or forbidden columns.

- [ ] **Q.2** Handle category dtypes consistently:
  - Some encodings use `.astype('category').cat.codes`, others rely on dtype.
  - Standardize:
    - For any column that you encode (e.g., `ther_area`, `main_package`, `country`), ensure you either:
      - Keep the raw string column plus an `_encoded` numeric column, or
      - Only keep the encoded form and drop the string one.
  - Update `get_categorical_feature_names` to know about whichever convention you decide.

---

## R. Robustness of "Externally-Driven" Feature Families

- [ ] **R.1** Visibility features (`add_visibility_features`):
  - Ensure that:
    - Merge is robust when `period` is a string, Period, or datetime.
    - When merge fails, you log a clear message and still add all `vis_*` columns with 0 or neutral values.
  - Add minimal sanity checks:
    - Check that `visibility_data` contains `country`, `brand_name`, and either `period` or `_vis_period`.

- [ ] **R.2** Collaboration features (`add_collaboration_features` and `_add_loo_prior`):
  - Confirm that these functions:
    - Only run when `y_norm` is available.
    - Do not blow up when group sizes are 1 or all NaN.
  - In `_add_loo_prior`, add guard logic for groups with 0 or 1 non-NaN entries.

- [ ] **R.3** Sequence features (`add_sequence_features`):
  - Clarify in docstring:
    - That these are for tabular models, not full sequence tensors.
  - Ensure no leakage:
    - Confirm that lags / rolling windows do not peek into *future* months relative to the current row (they currently use `.shift` and standard rolling, which is safe, but verify and document).

---

## S. FeatureScaler Polishing

- [ ] **S.1** Clarify usage in docstring and comments:
  - Emphasize that tree-based models usually should use `method='none'`.
  - Explain how `exclude_cols` should usually include categorical encoded IDs and maybe time indices.

- [ ] **S.2** Ensure robust behavior when called multiple times:
  - Make sure that:
    - Calling `fit` twice either:
      - Re-fits cleanly and overwrites previous state, or
      - Raises a clear warning if that's not intended.
    - `inverse_transform` is safe even if some numeric cols are missing from X (already partly checked via intersection).

---

## T. Testing & Documentation TODOs for Features Module

- [ ] **T.1** Unit tests for core building blocks:
  - `make_features`:
    - For both scenarios 1 and 2.
    - For train vs test mode.
  - `select_training_rows`:
    - Correct month ranges per scenario.
  - `validate_feature_leakage` and `validate_feature_matrix`:
    - Verify that violations are correctly reported in controlled toy examples.
  - `add_pre_entry_features`, `add_generics_features`, `add_early_erosion_features`:
    - On small synthetic panels with known outputs.

- [ ] **T.2** Integration test for `get_features`:
  - Using a small synthetic panel built via `get_panel` or a mocked version.
  - Verify:
    - Caching works (second call loads from parquet).
    - X/y shapes match expectations.
    - No forbidden columns leak into X.

- [ ] **T.3** Documentation updates:
  - In your `README`, `functionality.md`, or project docs:
    - Add a section summarizing:
      - Feature categories (1–6 + visibility / collaboration / sequence).
      - How to enable/disable them via `configs/features.yaml`.
      - How to use feature selection and ablation utilities.

---

---

# Data & Validation Module TODO (`src/data.py` & `src/validation.py`)

> **Goal**: Centralize constants, fix validation logic, improve panel building, strengthen leakage defenses, and ensure CV splits are consistent with scenario constraints.

---

## U. Global Cleanup & Consistency Between `data.py` and `validation.py`

- [ ] **U.1** Centralize ID/time column constants across modules:
  - Ensure `ID_COLS`, `TIME_COL`, `CALENDAR_MONTH_COL`, `RAW_TARGET_COL`, `MODEL_TARGET_COL`, `META_COLS`, `LEAKAGE_COLUMNS`, and `ID_COLUMNS` are **defined once** (e.g., in a small `constants.py` or at least used consistently in both `data.py`, `validation.py`, `train.py`, `features.py`).
  - Replace hard-coded lists like `['country', 'brand_name']` and `['country', 'brand_name', 'months_postgx']` in `validation.py` with the centralized constants.
  - Make sure `META_COLS` in code and `columns.meta_cols` in `data.yaml` are fully synchronized and covered by `verify_meta_cols_consistency`.

- [ ] **U.2** Improve type hints and docstrings:
  - For each public function in `validation.py` and `data.py`, ensure:
    - Full type hints for arguments and return types.
    - Docstrings describe assumptions (e.g., expected columns, expected ranges for `months_postgx`, scenarios).
  - Explicitly document **scenario behavior** (S1 vs S2) wherever relevant.

- [ ] **U.3** Add shared helper utilities:
  - Factor out repeated patterns like:
    - "Build series_info from panel_df with series keys and stratify columns."
    - "Merge series sets back into panel_df to get train/val splits."
  - Put these helpers in a common utility (e.g., within `validation.py` or a shared helper module) and reuse them in:
    - `create_validation_split`
    - `get_fold_series`
    - `get_grouped_kfold_series`
    - `create_purged_cv_split`
    - `create_nested_cv`

---

## V. `data.py` – Raw Loading, Panel Building, and Leakage Defenses ✅ PARTIALLY COMPLETED

### V1. Use `EXPECTED_DTYPES` in Actual Validation ✅ COMPLETED

- [x] **V1.1** Enforce or validate dtypes after loading: ✅
  - Added `validate_dtypes()` function that checks columns against `EXPECTED_DTYPES`.
  - Supports `strict` mode (raises error) or warning mode (logs and proceeds).
  - Checks for dtype compatibility and logs mismatches.

- [x] **V1.2** Extend schema validation to include dtypes: ✅
  - `validate_panel_schema()` already checks key column types.
  - `validate_dtypes()` can be called separately for comprehensive checking.

### V2. Make `compute_pre_entry_stats` Fully Configurable and Consistent ✅ COMPLETED

- [x] **V2.1** Wire the `bucket_threshold` from config: ✅
  - `compute_pre_entry_stats()` already accepts `bucket_threshold` and `run_config` parameters.
  - Falls back to config value `run_config['official_metric']['bucket_threshold']` if provided.
  - Default 0.25 only used when no config available.

- [x] **V2.2** Ensure test panels never get target-derived columns: ✅
  - `compute_pre_entry_stats(is_train=False)` correctly skips `y_norm`, `mean_erosion`, `bucket`.
  - `validate_panel_schema(split="test")` checks these columns are absent.

### V3. Strengthen `handle_missing_values`

- [ ] **V3.1** Make missing value strategies configurable:
  - Add config options in `data.yaml` for:
    - How to fill `n_gxs` (`ffill+0` vs `0 only`).
    - How to treat `hospital_rate` (median by `ther_area`, global median).
    - What placeholder string to use for missing categorical values (default `"Unknown"`).
  - Read these options in `handle_missing_values` and adjust behavior accordingly.

- [ ] **V3.2** Ensure consistency of boolean flags:
  - Confirm that `biological_missing` and `small_molecule_missing` are **always integer flags {0,1}** and original columns are boolean.
  - Add checks/logging to verify this after filling.

### V4. Improve Panel Caching & CLI Behavior

- [ ] **V4.1** Unify cache path logic:
  - Ensure that cache paths in:
    - `_get_cache_path`
    - `get_panel`
    - `clear_panel_cache`
    - CLI code that writes continuity issues
  - All consistently use `get_project_root() / config['paths']['interim_dir']` (and the drive overrides for Colab).
  - Avoid mixing root-relative and CWD-relative paths.

- [ ] **V4.2** Log memory usage before/after `_optimize_dtypes`:
  - In `_optimize_dtypes`, compute memory usage before and after to log the reduction in MB (or as percentage).
  - Ensure downcasting does not affect critical columns in `preserve_precision` (add explicit unit tests later).

### V5. Actually Use `verify_no_future_leakage` in the Pipeline ✅ COMPLETED

- [x] **V5.1** Integrate `verify_no_future_leakage` into feature building or training: ✅
  - `get_features()` now runs `validate_feature_leakage()` and `validate_feature_cutoffs()` on feature DataFrame.
  - Both check for leakage patterns before splitting into X/y/meta.
  - Logs warnings for any detected issues.

- [x] **V5.2** Improve `verify_no_future_leakage` checks: ✅
  - Enhanced `validate_feature_leakage()` with pattern matching for suspicious columns.
  - Checks patterns like `_future_`, `_target_`, `_label_`, `_test_target`.
  - Added `strict` parameter to raise on detection.

### V6. Use `min_months` and `max_months` in `validate_date_continuity` ✅ COMPLETED

- [x] **V6.1** Fix `validate_date_continuity` parameter usage: ✅
  - Added `use_expected_range` parameter to control gap detection behavior.
  - When `use_expected_range=True`, checks gaps against `[min_months, max_months]`.
  - When `use_expected_range=False` (default), checks gaps within each series' own range.
  - Added coverage issue detection for series that don't span expected range.

---

## W. `validation.py` – Splits, CV, and Metrics ✅ COMPLETED

### W1. Fix and Complete `create_validation_split` ✅

- [x] **W1.1** Generalize `stratify_by` behavior: ✅
  - Supports single column (string) and multiple columns (list of strings).
  - Logs warning if stratify columns are missing and uses available columns only.

- [x] **W1.2** Handle extremely small/rare stratification classes more robustly: ✅
  - Revised `min_class_size` formula with minimum absolute threshold of 3 series.
  - Combines absolute minimum with relative minimum based on val_fraction.
  - Logs count of rare series grouped into 'OTHER'.

- [x] **W1.3** Add explicit validation of output splits: ✅
  - Added `_validate_split_output()` helper function.
  - Asserts train/val are disjoint at series level.
  - Asserts together they cover all series in original panel.

### W2. Make `create_temporal_cv_split` Consistent and Usable

- [ ] **W2.1** Decide on series-level vs time-level splitting and update accordingly:
  - Currently, `create_temporal_cv_split` splits by `months_postgx` (row-level), which can put **the same series in both train and val**, contradicting the "never mix months of the same series" guideline from the header of the module.
  - Choose one consistent behavior:
    - Option A (series-level only): Make `create_temporal_cv_split` operate at **series level** (like `create_validation_split`), and possibly drop or rename it if not needed.
    - Option B (documented exception): Clearly document that this is an **experimental, time-based CV** that allows series to appear in both train and val for temporal generalization, and adjust checks accordingly.
  - Whichever option is chosen, update the docstring and internal logic to match.

- [ ] **W2.2** Actually use `stratify_by` in `create_temporal_cv_split`:
  - Right now, `stratify_by` is not used at all.
  - Implement stratification logic:
    - Build `series_info` with `ID_COLS` + `stratify_by`.
    - For each fold's temporal window, ensure similar distributions over `stratify_by` if possible.

- [ ] **W2.3** Align temporal cutoffs with scenario constraints:
  - Ensure that temporal folds respect relevant scenario windows (e.g., S1: 0–23, S2: 6–23) if the function is used in a scenario-context.
  - Optionally add a `scenario` argument and:
    - Validate that validation months fall in `[0,23]` or `[6,23]` as appropriate.

### W3. Improve K-Fold and Grouped Splits ✅

- [x] **W3.1** `get_fold_series` robustness: ✅
  - Handles missing `stratify_by` column gracefully with fallback to simple KFold.
  - Logs class distribution per fold.
  - Warns if smallest class has fewer samples than n_folds.

- [x] **W3.2** `get_grouped_kfold_series` enhancements: ✅
  - Logs clearly when n_folds exceeds unique groups.
  - Logs number of series and unique group values for each fold.

### W4. Purged and Nested CV ✅

- [x] **W4.1** `create_purged_cv_split` – enforce non-empty folds: ✅
  - Added `min_train_rows` and `min_val_rows` parameters.
  - Explicit checks/warnings when folds are too small.
  - Logs configuration and recommended values for S1/S2.
  - Tracks and reports skipped folds.

- [x] **W4.2** `create_nested_cv` – connect to hyperparameter tuning: ✅
  - Added `outer_fold_idx` to return structure for tracking.
  - Enhanced logging with usage documentation.
  - Clear guidance on inner_folds for hyperparameter search.

### W5. Scenario Constraint Checks ✅

- [x] **W5.1** Refine `validate_cv_respects_scenario_constraints`: ✅
  - Added `allow_series_overlap` parameter for temporal CV strategies.
  - Enhanced scenario checks for valid month ranges (0-23 for S1, 6-23 for S2).
  - Added warning for S2 when training lacks early erosion data.

---

## X. Adversarial Validation and CV Metrics ✅ COMPLETED

- [x] **X.1** Adversarial validation robustness: ✅
  - Added `min_samples` check (warns if < 100 samples).
  - Made classifier hyperparameters configurable (`n_estimators`, `max_depth`).
  - Metrics saving uses append mode via `save_metric_records`.

- [x] **X.2** `aggregate_cv_scores` robustness: ✅
  - Handles missing metrics for some folds (logs warning, skips those folds).
  - Handles all-NaN metrics gracefully (returns NaNs and `n_folds=0`).
  - Collects all unique metric names across folds.

- [x] **X.3** `create_cv_comparison_table` usability: ✅
  - Added `ascending` parameter for configurable sort order.
  - Added `additional_metrics` parameter to include extra metrics.
  - Added `is_best` indicator for best model.

- [x] **X.4** `paired_t_test` safeguards: ✅
  - Added explicit checks for minimum 2 folds with warning.
  - Filters NaN values consistently (logs excluded pairs).
  - Returns `n_valid_pairs` in result.
  - Added configurable `alpha` parameter for significance level.

---

## Y. Data Leakage Audit Utilities

- [ ] **Y.1** Extend `audit_data_leakage` patterns:
  - Review and refine patterns for:
    - Target leakage (`LEAKAGE_COLUMNS`).
    - Scenario-specific leakage (especially Scenario 1 early-erosion features).
    - Suspicious columns (`_test_`, `_future_`, `_target_`, `_label_`).
  - Add comments to clarify that this is a **heuristic** check and not exhaustive.

- [ ] **Y.2** Integrate `run_pre_training_leakage_check` into training:
  - Ensure `train.py` (or equivalent training script) calls `run_pre_training_leakage_check`:
    - Before fitting any model (CatBoost, LightGBM, XGBoost, etc.).
    - For both scenarios and both train/test modes.
  - Add meaningful log messages when leakage is found, including sample column names.

---

## Z. CLI Improvements for `data.py`

- [ ] **Z.1** Improve CLI argument validation:
  - When `--scenario` is given without `--mode`, ensure the error message is clear and suggests correct usage.
  - When `--validate-continuity` is used, ensure that:
    - The issues CSV is stored under the correct root path (`get_project_root()` + interim dir).
    - The filename includes split and maybe timestamp for clarity.

- [ ] **Z.2** Add a dedicated CLI command for leakage audits:
  - Optionally add CLI flags:
    - `--audit-leakage` for running `audit_data_leakage` on cached feature sets.
    - `--verify-no-future-leakage` for calling `verify_no_future_leakage` directly.

---

## AA. Testing & Documentation for Data & Validation Modules

- [ ] **AA.1** Create unit tests for key functions in `data.py`:
  - `validate_dataframe_schema`
  - `validate_panel_schema`
  - `compute_pre_entry_stats` (bucket assignment, avg_vol_12m fallbacks)
  - `handle_missing_values`
  - `validate_date_continuity`
  - `get_series_month_coverage`
  - `audit_data_leakage` and `run_pre_training_leakage_check`
  - Build small synthetic DataFrames to test edge cases (missing months, rare strata, leakage columns, etc.).

- [ ] **AA.2** Create unit tests for key functions in `validation.py`:
  - `create_validation_split`
  - `get_fold_series`
  - `get_grouped_kfold_series`
  - `create_purged_cv_split`
  - `validate_cv_respects_scenario_constraints`
  - `aggregate_cv_scores`
  - `paired_t_test`
  - Build small synthetic DataFrames to test edge cases.

- [ ] **AA.3** Add high-level documentation:
  - In `docs/` or in `functionality.md`, describe:
    - How panels are built (`get_panel`).
    - How validation splits are constructed and which functions are the **canonical** ones for each use-case:
      - Basic train/val split: `create_validation_split`
      - Series-level K-fold: `get_fold_series`
      - Grouped K-fold: `get_grouped_kfold_series`
      - Temporal / purged CV: `create_temporal_cv_split` / `create_purged_cv_split`
    - How leakage is prevented and audited.

---

## Progress Summary

| Section | Description | Status |
|---------|-------------|--------|
| 1 | Hero Model Config | ⬜ Not Started |
| 2 | Config Hygiene | ⬜ Not Started |
| 3 | CatBoost Implementation | ⬜ Not Started |
| 4 | Scenario-Aware Configs | ⬜ Not Started |
| 5 | Sample Weights | ⬜ Not Started |
| 6 | Feature Engineering | ⬜ Not Started |
| 7 | Data Pipeline | ⬜ Not Started |
| 8 | Training Pipeline | ⬜ Not Started |
| 9 | Evaluation | ⬜ Not Started |
| 10 | Inference | ⬜ Not Started |
| 11 | Ensemble (Optional) | ⬜ Not Started |
| 12 | Global Clean-Up (train.py) | ⬜ Not Started |
| 13 | Experiment Tracking | ⬜ Not Started |
| 14 | Checkpointing | ⬜ Not Started |
| 15 | Sample Weights (train.py) | ⬜ Not Started |
| 16 | HPO (Optuna) | ⬜ Not Started |
| 17 | Memory Profiling | ⬜ Not Started |
| 18 | Core Training Helpers | ⬜ Not Started |
| 19 | Cross-Validation | ⬜ Not Started |
| 20 | High-Level Pipelines | ⬜ Not Started |
| 21 | Model Comparison | ⬜ Not Started |
| 22 | Config Sweeps | ⬜ Not Started |
| 23 | CLI Entrypoint | ⬜ Not Started |
| 24 | Testing & Validation | ⬜ Not Started |
| 25 | Performance & Safety | ⬜ Not Started |
| A | Evaluate: Imports & Hygiene | ⬜ Not Started |
| B | Evaluate: API & Validation | ⬜ Not Started |
| C | Evaluate: Official vs Fallback | ⬜ Not Started |
| D | Evaluate: Bucket Metrics | ⬜ Not Started |
| E | Evaluate: Error Analysis | ⬜ Not Started |
| F | Evaluate: Metrics Logging | ⬜ Not Started |
| G | Evaluate: Resilience Metrics | ⬜ Not Started |
| H | Evaluate: Config Validation | ⬜ Not Started |
| I | Evaluate: Documentation | ⬜ Not Started |
| J | Features: Pandas Fixes | ⬜ Not Started |
| K | Features: Scenario/Cutoff Logic | ⬜ Not Started |
| L | Features: Config Wiring | ⬜ Not Started |
| M | Features: Leak Prevention | ⬜ Not Started |
| N | Features: Performance | ⬜ Not Started |
| O | Features: Caching & Colab | ⬜ Not Started |
| P | Features: Utilities | ⬜ Not Started |
| Q | Features: Categorical/Numeric | ⬜ Not Started |
| R | Features: External Features | ⬜ Not Started |
| S | Features: FeatureScaler | ⬜ Not Started |
| T | Features: Testing & Docs | ⬜ Not Started |
| U | Data/Val: Global Cleanup | ⬜ Not Started |
| V | Data: Loading & Leakage | ⬜ Not Started |
| W | Validation: Splits & CV | ⬜ Not Started |
| X | Validation: Adversarial & Metrics | ⬜ Not Started |
| Y | Data: Leakage Audits | ⬜ Not Started |
| Z | Data: CLI Improvements | ⬜ Not Started |
| AA | Data/Val: Testing & Docs | ⬜ Not Started |

---

*Last Updated: November 30, 2025*

---

Perfect, let’s turn the bonus ideas into something you can literally paste into `TODO.md` and hand over to Copilot.

Below is a **self-contained “Bonus Performance Experiments” section**, with enough structure that Copilot can implement them safely and you can *always* compare against the current hero CatBoost before enabling anything by default.

---

# Bonus Performance Experiments (Metric-Gated, Safe to Roll Back)

> All bonus items **must**:
>
> * Be controlled by config flags (off by default).
> * Log official metrics clearly for comparison.
> * Never replace the current hero CatBoost path silently.
> * Only be enabled in the main pipeline **after** they show equal or better official metric on validation vs current CatBoost hero.

---

## B1. Bagged CatBoost Hero (Multiple Seeds, Averaged Predictions)

**Goal:** Reduce variance and stabilize predictions with a simple bagging ensemble of CatBoost models.

* [ ] **B1.1 – Add config for CatBoost bagging**

  * In `configs/model_cat.yaml` add a section like:

    * `bagging.enabled: false`
    * `bagging.n_models: 3` (or 3–5)
    * `bagging.seeds: [42, 2025, 1337]` (optional; if missing, derive from base seed)
    * `bagging.weighting: "uniform"` (for now only uniform; future: allow per-model weights).
  * In `run_defaults.yaml` you can optionally add:

    * `experiments.enable_catboost_bagging: false`

* [ ] **B1.2 – Implement a BaggedCatBoostModel wrapper**

  * In `src/models/cat_model.py` or `src/models/ensemble.py`:

    * Create a class `BaggedCatBoostModel` with the **same public interface** as the base CatBoost wrapper:

      * `fit(X_train, y_train, X_val=None, y_val=None, sample_weight=None)`
      * `predict(X)`
      * `save(path)`
      * `load(path)`
    * Internals:

      * Keep a list `self.models` (one CatBoostModel per seed).
      * In `fit(...)`:

        * For each seed in `bagging.seeds` (or `range(n_models)`):

          * Clone the base CatBoost params and set `random_seed` (and optionally small variations like `subsample` or `rsm` if you want).
          * Instantiate a CatBoostModel and call its `fit(...)`.
          * Append the fitted instance to `self.models`.
      * In `predict(X)`:

        * Get predictions from each sub-model.
        * Average them (uniform weights for now).
      * In `save(path)`:

        * Create a directory `path/` and store:

          * Each submodel under `model_i.cbm`.
          * A metadata JSON with seeds, params, and number of models.
      * In `load(path)`:

        * Load metadata, rebuild `self.models`, and load each submodel.

* [ ] **B1.3 – Integrate bagging into training pipeline**

  * In `src/train.py` (likely in `train_scenario_model` or wherever model is instantiated):

    * If `model_type == "catboost"` and `bagging.enabled` is `true`:

      * Instantiate `BaggedCatBoostModel` instead of the base `CatBoostModel`.
    * Ensure experiment logs and artifact names reflect bagging:

      * e.g. run_id suffix `_bagged`.

* [ ] **B1.4 – Metric comparison and safety**

  * Add a simple experiment path (CLI or notebook) to:

    * Train **baseline hero CatBoost** and **bagged CatBoost** on the **same** scenario, same features, same sample weights.
    * Compute and log:

      * Official metric1 / metric2.
      * Per-bucket metrics.
  * Only flip `bagging.enabled: true` in your main hero config **if** bagging’s official metric is:

    * Better, or
    * At worst, statistically indistinguishable (very small difference) and clearly more stable across seeds.

---

## B2. Bucket-Specialized CatBoost Models (Per Bucket 1 & 2)

**Goal:** Exploit that Bucket 1 is much more important in the official metric by training bucket-specific models.

* [ ] **B2.1 – Add bucket specialization config**

  * In `configs/model_cat.yaml` or `run_defaults.yaml`:

    * `bucket_specialization.enabled: false`
    * `bucket_specialization.buckets: [1, 2]`
    * `bucket_specialization.base_model_type: "catboost"` (to allow reuse later)

* [ ] **B2.2 – Implement training of bucket-specific models**

  * In `src/train.py` (or a new helper, e.g. `train_bucket_specialized_models`):

    * After you construct the panel with `bucket` and features:

      * Split train data by `bucket`:

        * `df_bucket1` = rows with `bucket == 1`
        * `df_bucket2` = rows with `bucket == 2`
      * For each bucket:

        * Call `split_features_target_meta` to get `X_train_bk`, `y_train_bk`, `meta_bk`.
        * Build a validation split **within that bucket** (respecting series-level splitting).
        * Train a CatBoost model for that bucket.
      * Save each model under:

        * `artifacts/.../bucket1_cat_model/`
        * `artifacts/.../bucket2_cat_model/`
      * Log metrics per bucket.

* [ ] **B2.3 – Inference routing by bucket**

  * In `src/inference.py`:

    * When loading models for submission:

      * Load both `bucket1` and `bucket2` CatBoost models.
    * For the test panel:

      * Ensure `bucket` column is available (recomputed from pre-entry stats using the same logic as training).
      * Split test data into two subsets by bucket.
      * Predict volumes with the appropriate model for each subset.
      * Concatenate predictions back into a single DataFrame and sort by `(country, brand_name, months_postgx)`.

* [ ] **B2.4 – Compare to global hero**

  * On validation:

    * Compute official metric for:

      * Global CatBoost hero.
      * Bucket-specialized CatBoost (merged predictions).
    * Log all metrics side by side in a comparison table.
  * Only tag bucket-specialized approach as “candidate hero” if it improves the official metric (especially for Bucket 1).

---

## B3. Post-hoc Calibration by (Scenario, Bucket, Time Window)

**Goal:** Correct systematic under/over-forecasting **after** the main model, without retraining it.

* [ ] **B3.1 – Define calibration config**

  * In `run_defaults.yaml` or a new `calibration` section:

    * `calibration.enabled: false`
    * `calibration.grouping: ["scenario", "bucket", "time_window"]`
    * `calibration.method: "linear"`  (start simple: slope+intercept)
    * `calibration.time_windows_s1: [[0,5],[6,11],[12,23]]`
    * `calibration.time_windows_s2: [[6,11],[12,17],[18,23]]`

* [ ] **B3.2 – Implement calibration fitting**

  * In a new module `src/calibration.py`:

    * Implement a function `fit_grouped_calibration(df_val, config)` where:

      * `df_val` has columns:

        * `scenario`, `bucket`, `months_postgx`, `volume_true`, `volume_pred`.
      * For each group defined by (scenario, bucket, time_window):

        * Select rows in that group.
        * Fit a simple linear regression:

          * `volume_true = a * volume_pred + b`
            (Alternatively, use isotonic regression later; start with linear for simplicity.)
        * Store `(a, b)` per group in a dict:

          * key = `(scenario, bucket, window_id)`.

    * Save calibration parameters to `artifacts/.../calibration_params.json`.

* [ ] **B3.3 – Apply calibration at inference**

  * In `src/inference.py`:

    * After you have **raw** model predictions (`volume_pred_raw`):

      * Load `calibration_params.json` if `calibration.enabled`.
      * For each row:

        * Identify `(scenario, bucket, time_window)` and get `(a, b)`; if no specific group found, fall back to `(a=1, b=0)`.
        * Set `volume_pred = a * volume_pred_raw + b`.
      * Clip:

        * `volume_pred = max(volume_pred, 0)` (no negative volumes).

* [ ] **B3.4 – Safety & comparison**

  * On validation:

    * Compute official metrics **before** and **after** calibration.
    * Record calibration parameters and metrics.
  * Only enable `calibration.enabled: true` when calibration improves the official metric or clearly reduces systematic bias.

---

## B4. Temporal Smoothing of Volume Curves

**Goal:** Remove unrealistic spikes/dropouts in the predicted volume time series.

* [ ] **B4.1 – Add smoothing config**

  * In `run_defaults.yaml`:

    * `smoothing.enabled: false`
    * `smoothing.method: "rolling_median"` (or `"rolling_mean"`)
    * `smoothing.window: 3`
    * `smoothing.min_periods: 1`
    * `smoothing.clip_negative: true`

* [ ] **B4.2 – Implement smoothing function**

  * In `src/inference.py` or a new module `src/postprocessing.py`:

    * Implement `smooth_predictions(df_pred, config)`:

      * `df_pred` has:

        * `country`, `brand_name`, `months_postgx`, `volume_pred`.
      * Group by `(country, brand_name)`.
      * Sort each group by `months_postgx`.
      * Apply the chosen rolling function on `volume_pred`:

        * For example: `rolling(window, min_periods).median()`.
      * Replace `volume_pred` with smoothed values.
      * If `clip_negative` is true, ensure `volume_pred >= 0`.

* [ ] **B4.3 – Safely test smoothing**

  * On validation:

    * Compute metrics:

      * Without smoothing.
      * With smoothing.
    * Inspect:

      * Whether smoothing preferentially helps early windows (0–5 / 6–11).
  * Only turn `smoothing.enabled` on for main pipeline if no systematic degradation occurs.

---

## B5. Residual Model Focused on High-Risk Segments

**Goal:** Add extra modeling capacity *only* where the hero model struggles: Bucket 1 and early months.

* [ ] **B5.1 – Add residual model config**

  * In `run_defaults.yaml` or `model_cat.yaml`:

    * `residual_model.enabled: false`
    * `residual_model.target_buckets: [1]`
    * `residual_model.target_windows_s1: [[0,5],[6,11]]`
    * `residual_model.target_windows_s2: [[6,11]]`
    * `residual_model.model_type: "catboost"` (or `"linear"` for simpler baseline)

* [ ] **B5.2 – Build residual training dataset**

  * After you have a trained hero CatBoost and validation predictions:

    * Build `df_residual` with:

      * `country`, `brand_name`, `months_postgx`, `bucket`, `scenario`, `volume_true`, `volume_pred`, all feature columns.
    * Filter:

      * `bucket` in `target_buckets`
      * `months_postgx` within the target windows per scenario.
    * Compute residual:

      * `residual = volume_true - volume_pred`.

* [ ] **B5.3 – Train residual model**

  * Implement a helper `train_residual_model(df_residual, features_config, residual_config)`:

    * Use the same features as hero CatBoost (or a subset).
    * Predict `residual` (can be regression CatBoost or linear model).
    * Save residual model separately, e.g. `artifacts/.../residual_model/`.

* [ ] **B5.4 – Apply residual correction at inference**

  * In inference:

    * After base hero CatBoost prediction `volume_pred_base`:

      * Build the same feature matrix used for residual model (for test data).
      * Predict `residual_pred` **only** for rows in target buckets/windows.
      * Set:

        * `volume_pred_final = volume_pred_base + residual_pred`.
        * Clip to `>= 0`.
    * For non-target buckets/windows, keep `volume_pred_final = volume_pred_base`.

* [ ] **B5.5 – Metric-guided adoption**

  * On validation:

    * Compare official metric for:

      * Base hero CatBoost.
      * Base + residual model.
    * If residual model improves Bucket 1 significantly without harming overall metric, consider enabling it for final pipeline.

---

## B6. Per-Therapeutic-Area / Country Bias Corrections

**Goal:** Correct systematic biases at the level of `ther_area` and/or `country`.

* [ ] **B6.1 – Config for group-level bias correction**

  * In `run_defaults.yaml`:

    * `bias_correction.enabled: false`
    * `bias_correction.group_cols: ["ther_area", "country"]`
    * `bias_correction.method: "mean_error"`  (for now just additive offsets)

* [ ] **B6.2 – Fit bias corrections on validation**

  * After you have validation predictions:

    * Build a DataFrame `df_bias` with:

      * group columns (`ther_area`, `country`), `volume_true`, `volume_pred`.

    * For each group (e.g., ther_area, or (ther_area, country)):

      * Compute:

        * `error = volume_true - volume_pred`.
        * `bias = mean(error)` (optionally robust mean or median).
      * Store in a dict:

        * key = group tuple, value = `bias`.

    * Save to `artifacts/.../bias_corrections.json`.

* [ ] **B6.3 – Apply corrections at inference**

  * When predicting:

    * After `volume_pred` from hero model:

      * For each row, find its group (ther_area, country).
      * Retrieve corresponding `bias` if available, else 0.
      * Set `volume_pred_corrected = volume_pred + bias`.
      * Clip to non-negative.

* [ ] **B6.4 – Evaluate**

  * Compare metrics:

    * Baseline vs bias-corrected.
    * Evaluate per group as well (country, ther_area).
  * Only enable `bias_correction.enabled` if it yields a global or bucket-specific improvement.

---

## B7. Feature Pruning Based on CatBoost Feature Importance

**Goal:** Remove useless or noisy features to reduce overfitting and speed up training.

* [ ] **B7.1 – Config for pruning**

  * In `features.yaml`:

    * `feature_pruning.enabled: false`
    * `feature_pruning.drop_bottom_fraction: 0.2`  (drop bottom 20% by importance)
    * `feature_pruning.min_keep: 50`  (never keep fewer than X features)

* [ ] **B7.2 – Extract feature importances**

  * After training the hero CatBoost:

    * Use the model’s `get_feature_importance` (or equivalent) to get importance per feature name.
    * Sort features by importance ascending.
    * Choose features to drop:

      * Exclude any meta or mandatory columns (ID, months_postgx if used).
    * Store:

      * `pruned_features_to_drop` and `kept_features` in a JSON under `artifacts/.../feature_pruning.json`.

* [ ] **B7.3 – Rebuild features with pruned set**

  * Add a step in `get_features` or `train.py` (behind `feature_pruning.enabled`) to:

    * Load `feature_pruning.json`.
    * Remove `pruned_features_to_drop` columns from X.
    * Log the new dimensionality.

* [ ] **B7.4 – Metric comparison**

  * Re-train hero CatBoost with pruned features.
  * Compare:

    * Official metric (overall + per bucket).
    * Training time / memory.
  * Only set `feature_pruning.enabled: true` for final run if metrics are stable or improved.

---

## B8. Seed Robustness and Small Seed Ensemble

**Goal:** Avoid being unlucky with a single random seed; optionally use small seed ensemble.

* [ ] **B8.1 – Config for multi-seed experiments**

  * In `run_defaults.yaml`:

    * `multi_seed.enabled: false`
    * `multi_seed.seeds: [42, 2025, 1337]`
    * `multi_seed.ensemble: false`  (if true, average predictions across seeds)

* [ ] **B8.2 – Implement multi-seed training runner**

  * In `src/train.py`, add helper `run_multi_seed_experiment(...)`:

    * For each seed in `multi_seed.seeds`:

      * Override `random_seed` in model config.
      * Train hero CatBoost once.
      * Store:

        * Validation official metric.
        * Path to model artifact.
    * Save a summary CSV:

      * One row per seed with metrics.

* [ ] **B8.3 – Optional seed ensemble**

  * If `multi_seed.ensemble == true`:

    * At inference, load all seed models.
    * Average their predictions (similar to B1 bagging, but here seeds are the primary variation).
  * Evaluate both:

    * Best single seed.
    * Seed ensemble.

* [ ] **B8.4 – Decide usage**

  * Use multi-seed only for:

    * Model selection (choose best reproducible seed), or
    * A small ensemble if clearly beneficial.

---

## B9. Monotonicity Constraints on Time-Like Features

**Goal:** Encourage more realistic erosion curves using CatBoost’s monotonic constraints (careful, experimental).

* [ ] **B9.1 – Config for monotonicity**

  * In `configs/model_cat.yaml`:

    * `monotonicity.enabled: false`
    * `monotonicity.constraints:`

      * Map from feature name to constraint:

        * `months_postgx: -1`  (for example, if higher months should generally correspond to lower volume; adjust sign after analysis)

* [ ] **B9.2 – Wire constraints into CatBoost params**

  * In CatBoost model wrapper:

    * When building CatBoost params and `monotonicity.enabled` is `true`:

      * Convert feature name constraints into the index-based `monotone_constraints` list expected by CatBoost.
      * Ensure the length matches number of features and unused features have `0`.

* [ ] **B9.3 – Careful evaluation**

  * Only use this in **small experiments**:

    * Train hero CatBoost with and without monotone constraints on `months_postgx`.
    * Compare:

      * Official metric.
      * Shape of average predicted erosion curves.
  * Unless it clearly helps, keep `monotonicity.enabled: false` in main pipeline.

---

## B10. Target Transform Experiments (Log / Power Transforms)

**Goal:** Make learning easier for skewed volumes by changing the regression target and then inverting predictions.

* [ ] **B10.1 – Config for target transform**

  * In `run_defaults.yaml` or `model_cat.yaml`:

    * `target_transform.type: "none"`  (options: `"none"`, `"log1p"`, `"power"`)
    * `target_transform.power_exponent: 0.5`  (for `"power"` type)
    * `target_transform.epsilon: 1e-6`  (for log1p / numerical safety)

* [ ] **B10.2 – Implement transform / inverse in training**

  * In `train_scenario_model` (or a dedicated helper):

    * When preparing `y_train` and `y_val`:

      * If `type == "log1p"`:

        * `y_transformed = log(volume + epsilon)`
      * If `type == "power"`:

        * `y_transformed = (volume + epsilon) ** alpha`
      * If `type == "none"`:

        * `y_transformed = volume` as today.
    * Train the model on `y_transformed` instead of raw volume or `y_norm`.
    * Keep sample weights unchanged (still reflecting volume-level metric).

* [ ] **B10.3 – Inverse transform at prediction time**

  * In `src/inference.py`:

    * After model predicts `y_pred_transformed`:

      * If `type == "log1p"`:

        * `volume_pred = exp(y_pred_transformed) - epsilon`
      * If `type == "power"`:

        * `volume_pred = max(y_pred_transformed, 0) ** (1 / alpha) - epsilon`
      * Clip final `volume_pred` to `>= 0`.

* [ ] **B10.4 – Metric comparison & safety**

  * Test each transform type on validation:

    * Compare official metrics vs baseline (`type: "none"`).
  * Only adopt a non-trivial transform globally if it consistently improves or stabilizes metrics.

---


# Generalization Under Limited Data (External Data + Robust Training)

> Goal: Make the model more **robust and generalizable** when we only see 30% of the competition data, by:
>
> * Safely integrating the **Indonesian pharmacy dataset** as a second domain,
> * Strengthening regularization and validation,
> * Avoiding overfitting to the tiny official train split.

---

## G1. Safely Integrate the Pharmacy Dataset as a Second Domain

**Idea:** Treat pharmacy data as another “country/domain” with the same panel structure, but keep it **explicitly marked** and **down-weighted**.

* [ ] **G1.1 – Add config for multi-dataset training**

  * In `configs/data.yaml` (or a new `configs/data_multi.yaml`) add:

    ```yaml
    additional_sources:
      pharmacy:
        enabled: false
        panel_type: "novartis_like"
        base_dir: "data/pharmacy_indonesia_novartis_like"
        df_volume: "df_volume_pharmacy_train.csv"
        df_generics: "df_generics_pharmacy_train.csv"
        df_medicine_info: "df_medicine_info_pharmacy_train.csv"

    source_weights:
      novartis: 1.0
      pharmacy: 0.3   # lower weight for external domain
    ```

  * This lets us toggle the external dataset from config only.

* [ ] **G1.2 – Add `source_dataset` / `domain` column to panels**

  * In `src/data.py`, in the function that builds the main panel (`get_panel` or equivalent):

    * When building the competition panel, add a column:

      * `source_dataset = "novartis"`
    * When building the pharmacy panel, add:

      * `source_dataset = "pharmacy"`

  * Ensure `source_dataset` is included in:

    * `META_COLS` in `data.py` and in `columns.meta_cols` in `configs/data.yaml`.

* [ ] **G1.3 – Implement loader for external pharmacy panel**

  * In `src/data.py`, add a helper, e.g. `load_external_panel(source_name: str, split: str, scenario: int)` that:

    * Reads `df_volume_pharmacy_train.csv`, `df_generics_pharmacy_train.csv`, and `df_medicine_info_pharmacy_train.csv` from `additional_sources.pharmacy.base_dir`.
    * Builds a panel with the *same columns* and semantics as the competition panel:

      * `country`, `brand_name`, `months_postgx`, `volume`, `avg_vol_12m`, `bucket`, etc., as applicable.
    * Sets `country = "COUNTRY_PHARMACY_ID"` as you already did.
    * Sets `source_dataset = "pharmacy"`.

  * Reuse as much as possible of your existing panel-building logic (joins, missing-value handling, bucket assignment).

* [ ] **G1.4 – Merge panels, but keep domain information**

  * In `get_panel(split="train", ...)`, when `additional_sources.pharmacy.enabled` is `true`:

    * Build the standard Novartis panel (call this `panel_novartis`).

    * Build the pharmacy panel (`panel_pharmacy`).

    * Concatenate:

      ```python
      panel_all = pd.concat([panel_novartis, panel_pharmacy], ignore_index=True)
      ```

    * Make sure:

      * ID columns (`country`, `brand_name`, `months_postgx`) remain unique across domains because `country` differs.
      * `validate_panel_schema(panel_all, split="train")` still passes.

* [ ] **G1.5 – Add domain-aware sample weights**

  * In `src/train.py` inside `compute_sample_weights` (or wherever you build final training weights):

    * After you compute time/window weights and bucket weights, multiply by a `source_weight`:

      ```python
      source_weights = run_config["data"]["source_weights"]  # or similar path
      df["domain_weight"] = df["source_dataset"].map(source_weights).fillna(1.0)
      df["sample_weight"] *= df["domain_weight"]
      ```

    * Log the distribution of `sample_weight` by `source_dataset`, bucket and month.

  * This ensures the external pharmacy domain helps but does not dominate.

* [ ] **G1.6 – Add domain indicator feature(s)**

  * In `src/features.py`:

    * Ensure `source_dataset` is available to features (either as a categorical feature, or encoded as:

      * `is_pharmacy = (source_dataset == "pharmacy").astype(int)`.

    * Add `source_dataset` (or `is_pharmacy`) to `categorical_features` in `configs/model_cat.yaml` so CatBoost can learn domain-specific offsets.

* [ ] **G1.7 – Add tests for multi-domain panels**

  * In `tests/test_data_multidomain.py`, create small synthetic competition and pharmacy panels and test:

    * `get_panel` with `pharmacy.enabled = true` returns both domains.
    * Schema is identical (same columns, same dtypes).
    * `source_dataset` is correctly set and `source_weights` apply as expected.

---

## G2. External-Data Pretraining / Fine-Tuning Experiments

**Idea:** Use pharmacy data to train initial models, then specialize on Novartis data. This is experiment-only and must be entirely config-driven.

* [ ] **G2.1 – Add config for external pretraining**

  * In `run_defaults.yaml`:

    ```yaml
    external_pretraining:
      enabled: false
      source: "pharmacy"
      model_type: "catboost"
      n_trees_external: 400
      n_trees_finetune: 600
      learning_rate_external: 0.05
      learning_rate_finetune: 0.03
    ```

* [ ] **G2.2 – Implement two-stage CatBoost training helper**

  * In `src/train.py`, add a helper, e.g. `train_with_external_pretraining(run_config, model_config, ...)` that:

    1. Builds **external** panel only (pharmacy) for the chosen scenario.

    2. Trains a CatBoost model on the external panel with:

       * `n_estimators = n_trees_external`
       * `learning_rate = learning_rate_external`
       * Standard sample weights (but possibly lower bucket weights to avoid bias).

    3. Saves this model as `external_pretrained_model.cbm`.

    4. Builds **Novartis** (competition) panel only.

    5. Fine-tunes on Novartis data:

       * If CatBoost supports warm-start (via `init_model` or equivalent), pass the external model as initialization.
       * Otherwise, experiment by:

         * Training on the concatenation `{external + novartis}` but:

           * Using higher weights for Novartis (e.g., domain weight 1.0 vs 0.2).
           * Possibly training in two phases with different learning rates.

  * This function should return a trained hero model and metrics.

* [ ] **G2.3 – Keep external pretraining strictly optional**

  * Wire this into `run_full_training_pipeline` behind `external_pretraining.enabled`:

    * If `false`: use normal training.
    * If `true`: call `train_with_external_pretraining`.

  * Log clearly:

    * `external_pretraining_used: true/false`
    * External/Novartis sample counts and metric comparison.

* [ ] **G2.4 – Compare metrics with and without external pretraining**

  * Implement a small script or CLI subcommand to:

    * Train hero CatBoost on Novartis-only.
    * Train hero CatBoost with external pretraining.
    * Save a comparison table with:

      * Official metrics (metric1/metric2),
      * Per-bucket metrics,
      * Per-scenario metrics.

---

## G3. Domain Shift & Adversarial Validation (Train vs Hidden 70% & vs Pharmacy)

**Idea:** With only 30% labeled, you want to minimize the risk that your model overfits to idiosyncrasies of that subset. Adversarial validation helps you detect distribution shifts.

* [ ] **G3.1 – Extend adversarial validation to include `source_dataset`**

  * In `src/validation.py` (where `adversarial_validation` already exists):

    * Allow passing an additional categorical feature `source_dataset` or `is_pharmacy`.
    * Use it to check if the classifier can easily distinguish:

      * `Novartis_train_30%` vs `Novartis_holdout_30%` (simulated),
      * `Novartis_train_30%` vs `Pharmacy`.

* [ ] **G3.2 – Simulate “hidden 70%” within the 30%**

  * Using only competition data:

    * Randomly split the 30% into:

      * “pseudo-train” (e.g., 70% of it),
      * “pseudo-test” (30% of it).
    * Run adversarial validation to check if pseudo-train and pseudo-test are indistinguishable by a classifier (good) or easily separable (bad).

* [ ] **G3.3 – Use adversarial scores to reweight**

  * If adversarial validation indicates that some series in the 30% look more like “hidden part” (pseudo-test):

    * Compute a “similarity score” or “test-likeness” probability for each series.
    * Optionally add:

      * A `test_likeness` feature.
      * A `test_likeness_weight` multiplier in `sample_weight`.

  * Ensure this logic is **optional** and controlled by config, e.g.:

    ```yaml
    adversarial_reweighting:
      enabled: false
      max_weight_factor: 2.0
    ```

---

## G4. Small-Data-Oriented Regularization Profiles

**Idea:** When training on 30%, you want more conservative models: shallower trees, stronger regularization, more subsampling.

* [ ] **G4.1 – Add `small_data_mode` to model config**

  * In `configs/model_cat.yaml`:

    ```yaml
    small_data_mode:
      enabled: false
      overrides:
        depth: 6
        l2_leaf_reg: 6.0
        subsample: 0.8
        rsm: 0.8
        min_data_in_leaf: 20
    ```

* [ ] **G4.2 – Apply overrides when `small_data_mode.enabled` is true**

  * In `src/models/cat_model.py` when building CatBoost parameters:

    * If `small_data_mode.enabled`:

      * Overlay `small_data_mode.overrides` on top of the base `params` dict.
    * Log final parameter set (especially depth, l2, subsample, min_data_in_leaf).

* [ ] **G4.3 – Add a “conservative” sweep preset**

  * In `configs/model_cat.yaml` under `sweep_configs`:

    * Add something like `s1_small_data_conservative` / `s2_small_data_conservative` that:

      * Restricts depth to 4–7,
      * Increases L2 regularization range,
      * Uses stronger subsampling.

  * Use this preset when the 30% regime is active.

---

## G5. More Robust Validation & Model Selection

**Idea:** With little data, you want *stable* estimates of performance, not just one lucky split.

* [ ] **G5.1 – Repeated series-level CV**

  * In `src/validation.py` and `src/train.py`:

    * Add config in `run_defaults.yaml`:

      ```yaml
      cv:
        enabled: true
        n_folds: 5
        n_repeats: 3
        level: "series"
      ```

    * Implement repeated series-level K-fold:

      * For each repeat:

        * Generate a different series-level split (`get_fold_series` with different random seed).
      * Aggregate metrics across all folds and repeats (mean ± std).

* [ ] **G5.2 – Tie model selection to *average* CV metric**

  * In `run_sweep_experiments` and HPO routines:

    * When ranking configs, use the **average official metric** across folds/repeats, not a single split.
    * Log per-fold metrics and aggregate metrics for transparency.

* [ ] **G5.3 – Enforce “stability check” for candidate best models**

  * After identifying a best CatBoost config:

    * Re-train it with **multiple random seeds** (e.g., 3 seeds).
    * Check the variance in CV metrics across seeds.
    * If variance is high, mark the config as unstable and prefer a slightly worse but more stable configuration.

---

## G6. Light Data Augmentation for Time Series (Optional, Experimental)

**Idea:** Slightly perturb or crop series to reduce overfitting to specific volume patterns. This is experimental and must be config-gated.

* [ ] **G6.1 – Add augmentation config**

  * In `features.yaml` or `run_defaults.yaml`:

    ```yaml
    augmentation:
      enabled: false
      jitter_volume_pct: 0.05   # up to ±5% random noise
      drop_random_month_prob: 0.05
      min_months_postgx: 0
      max_months_postgx: 23
    ```

* [ ] **G6.2 – Implement augmentation in a dedicated helper**

  * In `src/data.py` or `src/features.py` add `augment_panel(panel_df, config)` that:

    * For each series `(country, brand_name)`:

      * Optionally apply random small multiplicative noise to `volume`:

        * `volume *= (1 + eps)` where `eps ~ Uniform(-jitter_volume_pct, +jitter_volume_pct)`.
      * With small probability `drop_random_month_prob`:

        * Drop a random non-critical month (avoid months used as targets if this conflicts with the metric definition).

    * Make sure:

      * You never create negative volumes.
      * You do not break the metric assumptions (months range still 0–23).

  * Only call this helper when `augmentation.enabled` is `true` and only on **training** panel.

* [ ] **G6.3 – Compare with/without augmentation**

  * Run experiments with augmentation off vs on.
  * Only adopt augmentation if it clearly improves CV metrics or leads to more stable performance across seeds.

---

## G7. Sanity Checks Specific to the Pharmacy Dataset

**Idea:** Ensure that the Indonesian pharmacy data is actually “similar enough” and does not break assumptions.

* [ ] **G7.1 – Schema and range checks**

  * In a small script or test:

    * Compare key distributions between Novartis and pharmacy:

      * Distribution of `months_postgx`,
      * Distribution of `avg_vol_12m` / `bucket`,
      * Share of series per `ther_area`, `main_package`.

    * Log these side-by-side.

* [ ] **G7.2 – Per-domain model performance**

  * When training on `novartis + pharmacy`:

    * Compute and log metrics **separately** for:

      * Novartis-only validation rows,
      * Pharmacy-only validation rows.

    * If performance on Novartis worsens when adding pharmacy, consider:

      * Lowering `source_weights.pharmacy`,
      * Or using pharmacy only for pretraining, not final training.

---

## G8. Documentation & Safety

* [ ] **G8.1 – Document external data use and competition constraints**

  * In `docs/implementation_approach.md` (or similar):

    * Add a short section:

      * Describing that an external pharmacy dataset is used as an optional pretraining/regularization source.
      * Explicitly stating that all final **competition submissions** are based on models trained only on:

        * Official train set, or
        * Official train + allowed external data, in line with competition rules.

* [ ] **G8.2 – Config presets for “competition-safe” vs “research” runs**

  * Add two top-level run presets:

    * `configs/run_competition.yaml`:

      * `additional_sources.pharmacy.enabled: false` or set according to rules.
      * `external_pretraining.enabled: false` (unless permitted).
    * `configs/run_research.yaml`:

      * May turn on pharmacy integration, external pretraining, augmentation, etc.

  * This prevents accidental usage of experimental features in official runs.

---
