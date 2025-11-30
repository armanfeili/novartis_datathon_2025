# =============================================================================
# File: src/config.py
# Description: Central configuration for all project settings
# 
# ⚠️  IMPORTANT: This is the ONLY file you need to edit to customize the pipeline!
#     All parameters that affect model training and evaluation are defined here.
#
# 📋 SECTIONS:
#    1. RUN MODE & TOGGLES - What to run and how
#    2. MODEL TOGGLES - Enable/disable specific models
#    3. PATHS - Directory structure
#    4. DATA FILES - Input file locations
#    5. COMPETITION CONSTANTS - Rules from the datathon
#    6. SCORING WEIGHTS - How predictions are evaluated
#    7. BASELINE MODEL PARAMETERS - Exponential decay settings
#    8. LIGHTGBM PARAMETERS - Gradient boosting (LightGBM)
#    9. XGBOOST PARAMETERS - Gradient boosting (XGBoost)
#   10. HYBRID MODEL PARAMETERS - Physics + ML combined
#   11. ARHOW MODEL PARAMETERS - Time-series (ARIMA + Holt-Winters)
#   12. FEATURE ENGINEERING PARAMETERS - Lag, rolling windows, etc.
#   13. COLUMN DEFINITIONS - Data schema
#
# =============================================================================

from pathlib import Path
import os

# =============================================================================
# 1. RUN MODE & TOGGLES
# =============================================================================
# Control what the pipeline does without changing CLI arguments.
# Just change these values and run the scripts!
#
# 🎮 WORKFLOW TOGGLES:
#    RUN_SCENARIO: Which scenario to run (1 or 2, or [1,2] for both)
#    TEST_MODE: If True, use only 50 brands for fast testing
#    TRAIN_MODE: "separate" = S1 & S2 trained separately, "unified" = single model
#
# 🔄 PIPELINE STEPS:
#    RUN_EDA: Run EDA visualization and export data
#    RUN_TRAINING: Train models
#    RUN_SUBMISSION: Generate final submission files
#    RUN_VALIDATION: Validate submission files

# Which scenario(s) to run
# Options: 1, 2, or [1, 2] for both
RUN_SCENARIO = [1, 2]                          # Set to 1, 2, or [1, 2]

# Training mode - controls how models are trained
# Options:
#   "separate" - Train S1 and S2 pipelines separately (current behavior)
#   "unified"  - Train a single global model that handles both scenarios
#                Uses scenario_flag feature and months_postgx to differentiate
TRAIN_MODE = "separate"                 # "separate" or "unified"

# Test mode - use only 50 brands for quick testing
# Set to False for full training before final submission
TEST_MODE = False                       # True = fast (50 brands), False = full

# Number of brands to use in test mode
TEST_MODE_BRANDS = 50                 # Number of brands for testing

# Which models to generate submissions for
# Set to True for each model you want submission files for
# 💡 TIP: Enable multiple models to compare submissions
#
# ⚠️ NOTE: Model must be trained first (via train_models.py)
#    before generating its submission
SUBMISSIONS_ENABLED = {
    'baseline': True,              # Exponential decay (best) 🏆
    'hybrid': True,                # Physics + LightGBM hybrid
    'hybrid_xgboost': True,        # Physics + XGBoost hybrid
    'lightgbm': True,              # Pure LightGBM
    'xgboost': True,               # Pure XGBoost
    'catboost': True,              # CatBoost
    'linear': True,                # Linear models (ridge/lasso/elasticnet)
    'nn': True,                    # Simple MLP
    'arihow': True,                # SARIMAX + Holt-Winters
}

# Quick preset: Generate only best model submission
# SUBMISSIONS_ENABLED = {'baseline': True, 'hybrid': False, 'lightgbm': False, 'xgboost': False, 'arihow': False}

# Pipeline step toggles - enable/disable individual steps
RUN_EDA = False                       # Run EDA visualization
RUN_TRAINING = True                   # Run model training
RUN_SUBMISSION = True                # Generate submission files
RUN_VALIDATION = True                # Validate submission files

# =============================================================================
# 2. MODEL TOGGLES
# =============================================================================
# Enable/disable specific models in training.
# Disabled models will be skipped, saving time.
#
# 💡 TIP: Disable slow models during experimentation, enable all for final run
#
# ⏱️ APPROXIMATE TRAINING TIME (test mode, 50 brands):
#    Baseline No Erosion:    ~1 sec
#    Baseline Exp Decay:     ~2 sec (tunes decay rate)
#    LightGBM:               ~5 sec
#    XGBoost:                ~10 sec
#    Hybrid LightGBM:        ~8 sec
#    Hybrid XGBoost:         ~12 sec
#    ARHOW:                  ~30 sec (fits per-brand time series)

MODELS_ENABLED = {
    # Baseline models (fast, simple)
    'baseline_no_erosion': True,      # Predicts avg_vol (no decline)
    'baseline_exp_decay': True,       # Exponential decay
    
    # Gradient boosting models (ML-based)
    'lightgbm': True,                 # LightGBM gradient boosting
    'xgboost': True,                  # XGBoost gradient boosting
    'catboost': True,                # CatBoost gradient boosting
    'linear': True,                  # Linear models (ridge/lasso/elasticnet)
    'nn': True,                      # Simple neural network (MLP)
    
    # Hybrid models (Physics + ML)
    'hybrid_lightgbm': True,          # Decay baseline + LightGBM residual
    'hybrid_xgboost': True,           # Decay baseline + XGBoost residual
    
    # Time-series models
    'arihow': True,                   # SARIMAX + Holt-Winters ensemble
}

# Quick presets - uncomment one to use
# MODELS_ENABLED = {'baseline_exp_decay': True}  #  baseline only
# MODELS_ENABLED = {k: True for k in ['baseline_no_erosion', 'baseline_exp_decay', 'lightgbm']}  # Quick
# MODELS_ENABLED = {k: True for k in MODELS_ENABLED}  # All models - OVERWRITES individual settings!
# Use the individual settings above instead

# =============================================================================
# 2b. MULTI-CONFIG MODE (Grid Search over Hyperparameters)
# =============================================================================
# Enable this to run multiple configurations and compare results.
# When enabled, the pipeline will iterate over all config combinations.
#
# 💡 USE CASES:
#    - Hyperparameter tuning (try different learning rates, depths)
#    - Compare model variants (different decay rates)
#    - Find optimal settings before final submission
#
# ⚠️ WARNING: Multi-config can be slow! Each config trains all enabled models.
#    With 3 configs × 7 models = 21 training runs!

MULTI_CONFIG_MODE = True           # Toggle: True = run all configs, False = single config

# Define multiple configurations to try
# Each config is a dict that overrides the default parameters
# The pipeline will run each config and save results with config ID

MULTI_CONFIGS = [
    # Config 0: Default (baseline)
    {
        'id': 'default',
        'description': 'Default parameters',
        # No overrides - uses defaults
    },
    
    # Config 1: Lower learning rate, more trees
    {
        'id': 'low_lr',
        'description': 'Low learning rate with more trees',
        'LGBM_PARAMS': {
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'num_leaves': 31,
        },
        'XGB_PARAMS': {
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'max_depth': 6,
        },
    },
    
    # Config 2: Higher complexity
    {
        'id': 'high_complexity',
        'description': 'More complex trees',
        'LGBM_PARAMS': {
            'learning_rate': 0.03,
            'n_estimators': 500,
            'num_leaves': 63,          # More leaves = more complex
        },
        'XGB_PARAMS': {
            'learning_rate': 0.03,
            'n_estimators': 500,
            'max_depth': 8,            # Deeper trees
        },
    },
    
    # Config 3: More regularization
    {
        'id': 'regularized',
        'description': 'Strong regularization to prevent overfitting',
        'LGBM_PARAMS': {
            'learning_rate': 0.05,
            'n_estimators': 300,
            'num_leaves': 15,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6,
        },
        'XGB_PARAMS': {
            'learning_rate': 0.05,
            'n_estimators': 300,
            'max_depth': 4,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
        },
    },
    
    # Config 4: Different decay rates
    {
        'id': 'slow_decay',
        'description': 'Slower baseline decay rate',
        'DEFAULT_DECAY_RATE': 0.01,
    },
    
    # Config 5: Fast decay
    {
        'id': 'fast_decay',
        'description': 'Faster baseline decay rate',
        'DEFAULT_DECAY_RATE': 0.05,
    },
]

# Currently active config (set by pipeline when running multi-config)
ACTIVE_CONFIG_ID = 'default'


def get_config_by_id(config_id: str) -> dict:
    """Get a specific config by ID."""
    for cfg in MULTI_CONFIGS:
        if cfg['id'] == config_id:
            return cfg
    return {}


def apply_config(config: dict) -> None:
    """
    Apply a config's overrides to global settings.
    Call this at the start of training to switch configs.
    
    Usage:
        from config import apply_config, MULTI_CONFIGS
        apply_config(MULTI_CONFIGS[1])  # Apply config 1
    """
    global LGBM_PARAMS, XGB_PARAMS, HYBRID_PARAMS, ARIHOW_PARAMS
    global DEFAULT_DECAY_RATE, ACTIVE_CONFIG_ID
    
    if 'id' in config:
        ACTIVE_CONFIG_ID = config['id']
    
    if 'LGBM_PARAMS' in config:
        LGBM_PARAMS.update(config['LGBM_PARAMS'])
    
    if 'XGB_PARAMS' in config:
        XGB_PARAMS.update(config['XGB_PARAMS'])
    
    if 'HYBRID_PARAMS' in config:
        HYBRID_PARAMS.update(config['HYBRID_PARAMS'])
    
    if 'ARIHOW_PARAMS' in config:
        ARIHOW_PARAMS.update(config['ARIHOW_PARAMS'])
    
    if 'DEFAULT_DECAY_RATE' in config:
        DEFAULT_DECAY_RATE = config['DEFAULT_DECAY_RATE']


def get_all_config_ids() -> list:
    """Get list of all config IDs."""
    return [cfg['id'] for cfg in MULTI_CONFIGS]


def get_model_filename(base_name: str) -> str:
    """
    Get model filename with config prefix when in multi-config mode.
    
    Args:
        base_name: Base model name, e.g., "scenario1_lightgbm"
        
    Returns:
        If MULTI_CONFIG_MODE and ACTIVE_CONFIG_ID != 'default':
            "{config_id}_{base_name}" e.g., "low_lr_scenario1_lightgbm"
        Otherwise:
            "{base_name}" e.g., "scenario1_lightgbm"
    """
    if MULTI_CONFIG_MODE and ACTIVE_CONFIG_ID != 'default':
        return f"{ACTIVE_CONFIG_ID}_{base_name}"
    return base_name


def get_submission_filename(model_type: str, suffix: str = "final") -> str:
    """
    Get submission filename with config prefix when in multi-config mode.
    
    Args:
        model_type: Model type, e.g., "baseline", "lightgbm"
        suffix: Filename suffix, e.g., "final" or timestamp
        
    Returns:
        If MULTI_CONFIG_MODE and ACTIVE_CONFIG_ID != 'default':
            "submission_{config_id}_{model_type}_{suffix}.csv"
        Otherwise:
            "submission_{model_type}_{suffix}.csv"
    """
    if MULTI_CONFIG_MODE and ACTIVE_CONFIG_ID != 'default':
        return f"submission_{ACTIVE_CONFIG_ID}_{model_type}_{suffix}.csv"
    return f"submission_{model_type}_{suffix}.csv"


def get_current_config_snapshot() -> dict:
    """
    Get a snapshot of ALL current config settings for saving with models/submissions.
    This ensures full reproducibility - you can see exactly what config was used.
    
    Returns:
        Dictionary with all relevant config values at time of call
    """
    return {
        "config_id": ACTIVE_CONFIG_ID,
        "multi_config_mode": MULTI_CONFIG_MODE,
        "train_mode": TRAIN_MODE,
        "test_mode": TEST_MODE,
        "test_mode_brands": TEST_MODE_BRANDS,
        "run_scenario": RUN_SCENARIO,
        "models_enabled": MODELS_ENABLED.copy(),
        "submissions_enabled": SUBMISSIONS_ENABLED.copy(),
        "competition_constants": {
            "pre_entry_months": PRE_ENTRY_MONTHS,
            "post_entry_months": POST_ENTRY_MONTHS,
            "bucket_1_threshold": BUCKET_1_THRESHOLD,
            "bucket_1_weight": BUCKET_1_WEIGHT,
            "bucket_2_weight": BUCKET_2_WEIGHT,
        },
        "baseline_params": {
            "default_decay_rate": DEFAULT_DECAY_RATE,
            "decay_rate_range": DECAY_RATE_RANGE,
            "decay_rate_steps": DECAY_RATE_STEPS,
        },
        "lgbm_params": LGBM_PARAMS.copy(),
        "xgb_params": XGB_PARAMS.copy(),
        "hybrid_params": HYBRID_PARAMS.copy() if 'HYBRID_PARAMS' in dir() else {},
        "feature_params": {
            "lag_windows": LAG_WINDOWS,
            "rolling_windows": ROLLING_WINDOWS,
            "pct_change_windows": PCT_CHANGE_WINDOWS,
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_splits_cv": N_SPLITS_CV,
    }


def reset_to_defaults() -> None:
    """Reset all parameters to their default values."""
    global LGBM_PARAMS, XGB_PARAMS, DEFAULT_DECAY_RATE, ACTIVE_CONFIG_ID
    
    ACTIVE_CONFIG_ID = 'default'
    
    # Reset LGBM
    LGBM_PARAMS.update({
        'learning_rate': 0.03,
        'n_estimators': 500,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
    })
    
    # Reset XGB
    XGB_PARAMS.update({
        'learning_rate': 0.03,
        'n_estimators': 500,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    })
    
    DEFAULT_DECAY_RATE = 0.02


# =============================================================================
# 3. PATHS
# =============================================================================
# These define the project directory structure. Usually no need to change.

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"           # Raw input CSVs
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # Processed data
MODELS_DIR = PROJECT_ROOT / "models"               # Saved model files (.joblib)
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"     # Final submission CSVs
REPORTS_DIR = PROJECT_ROOT / "reports"             # Comparison reports, summaries
FIGURES_DIR = REPORTS_DIR / "figures"              # Visualization images
EDA_DATA_DIR = REPORTS_DIR / "eda_data"            # EDA JSON/CSV exports
EXTERNAL_DATA_DIR = DATA_RAW / "external"          # Optional external/context data

# Create directories if they don't exist
for dir_path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, SUBMISSIONS_DIR, 
                 REPORTS_DIR, FIGURES_DIR, EDA_DATA_DIR, EXTERNAL_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 4. DATA FILES
# =============================================================================
# Input data file paths. Update if your file names are different.

VOLUME_TRAIN = DATA_RAW / "df_volume_train.csv"
GENERICS_TRAIN = DATA_RAW / "df_generics_train.csv"
MEDICINE_INFO_TRAIN = DATA_RAW / "df_medicine_info_train.csv"
VOLUME_TEST = DATA_RAW / "df_volume_test.csv"
GENERICS_TEST = DATA_RAW / "df_generics_test.csv"
MEDICINE_INFO_TEST = DATA_RAW / "df_medicine_info_test.csv"

# =============================================================================
# 5. COMPETITION CONSTANTS
# =============================================================================
# These are defined by the Novartis Datathon rules. DO NOT CHANGE unless the
# competition rules change.
#
# PRE_ENTRY_MONTHS: Number of months BEFORE generic entry used to calculate
#                   the baseline average (Avg_j). The competition uses 12 months.
#
# POST_ENTRY_MONTHS: Number of months AFTER generic entry to predict.
#                    Scenario 1: Predict months 0-23 (24 months)
#                    Scenario 2: Predict months 6-23 (18 months, given 0-5 actuals)
#
# BUCKET_1_THRESHOLD: If mean erosion ≤ 0.25, the brand is Bucket 1 (high erosion)
#                     Bucket 1 predictions are weighted 2x in scoring!
#
# BUCKET_X_WEIGHT: Scoring weights for each bucket. Bucket 1 counts double.

PRE_ENTRY_MONTHS = 12       # Months -12 to -1 (before generic entry)
POST_ENTRY_MONTHS = 24      # Months 0 to 23 (after generic entry)
BUCKET_1_THRESHOLD = 0.25   # Mean erosion ≤ 0.25 → Bucket 1 (severe erosion)
BUCKET_1_WEIGHT = 2         # Bucket 1 errors weighted 2× (CRITICAL!)
BUCKET_2_WEIGHT = 1         # Bucket 2 errors weighted 1× (normal)

# =============================================================================
# 5b. SAMPLE WEIGHTS FOR CLASS IMBALANCE (From EDA Report)
# =============================================================================
# The EDA report found severe class imbalance:
#   - Bucket 1 (High Erosion): Only 6.7% of brands (130 of 1,953)
#   - Bucket 2 (Low Erosion): 93.3% of brands (1,823 of 1,953)
#   - Imbalance Ratio: 14:1
#
# Since Bucket 1 has 2× weight in the competition metric, we need to handle
# this imbalance in training. These weights are applied to training samples.
#
# 💡 TIP: Higher weight for Bucket 1 = model focuses more on high-erosion brands

USE_SAMPLE_WEIGHTS = True       # Apply bucket-based sample weights during training
BUCKET_1_SAMPLE_WEIGHT = 4.0    # Weight for Bucket 1 samples (high erosion)
BUCKET_2_SAMPLE_WEIGHT = 1.0    # Weight for Bucket 2 samples (normal)

# =============================================================================
# 5c. TIME-WINDOW SAMPLE WEIGHTS (From Todo Section 3.4)
# =============================================================================
# Align loss with competition metric by weighting time periods differently.
#
# From Competition Scoring:
#   Scenario 1: sum_0-5 = 50%, sum_6-11 = 20%, sum_12-23 = 10%
#   Scenario 2: sum_6-11 = 50%, sum_12-23 = 30%
#
# Higher weights for early months since they contribute more to the metric.
#
# 💡 TIP: These weights are multiplied with bucket weights for final sample weight

USE_TIME_WINDOW_WEIGHTS = True  # Apply time-window based sample weights

# Scenario 1 time window weights (months 0-5 most important)
S1_TIME_WINDOW_WEIGHTS = {
    'months_0_5': 2.5,      # 50% of metric → weight 2.5×
    'months_6_11': 1.0,     # 20% of metric → weight 1.0×
    'months_12_23': 0.5,    # 10% of metric → weight 0.5×
    'pre_entry': 0.1,       # Pre-entry data less important for prediction
}

# Scenario 2 time window weights (months 6-11 most important)
S2_TIME_WINDOW_WEIGHTS = {
    'months_6_11': 2.5,     # 50% of metric → weight 2.5×
    'months_12_23': 1.5,    # 30% of metric → weight 1.5×
}

# =============================================================================
# 6. SCORING WEIGHTS (PE - Prediction Error)
# =============================================================================
# The competition uses a weighted scoring formula. These weights determine
# how much each time period contributes to the final PE score.
#
# 📊 SCENARIO 1 (Predict months 0-23, no actuals given):
#    - Monthly individual errors: 20%
#    - Sum of months 0-5: 50%   ← MOST IMPORTANT! Half your score!
#    - Sum of months 6-11: 20%
#    - Sum of months 12-23: 10%
#
# 📊 SCENARIO 2 (Predict months 6-23, given 0-5 actuals):
#    - Monthly individual errors: 20%
#    - Sum of months 6-11: 50%  ← MOST IMPORTANT!
#    - Sum of months 12-23: 30%

# Scenario 1 weights
S1_MONTHLY_WEIGHT = 0.2      # Weight for individual monthly absolute errors
S1_SUM_0_5_WEIGHT = 0.5      # Weight for accumulated error months 0-5 (50%!)
S1_SUM_6_11_WEIGHT = 0.2     # Weight for accumulated error months 6-11
S1_SUM_12_23_WEIGHT = 0.1    # Weight for accumulated error months 12-23

# Scenario 2 weights  
S2_MONTHLY_WEIGHT = 0.2      # Weight for individual monthly absolute errors
S2_SUM_6_11_WEIGHT = 0.5     # Weight for accumulated error months 6-11 (50%!)
S2_SUM_12_23_WEIGHT = 0.3    # Weight for accumulated error months 12-23

# =============================================================================
# 7. BASELINE MODEL PARAMETERS
# =============================================================================
# The exponential decay baseline: volume(t) = avg_vol × exp(-λ × t)
#
# DECAY_RATE_RANGE: Range of λ values to search during tuning
#                   Lower λ = slower decay, Higher λ = faster decay
#                   Typical best values: 0.01 - 0.05
#
# DECAY_RATE_STEPS: Number of values to try in the range (grid search)
#
# DEFAULT_DECAY_RATE: Fallback if tuning fails or for quick tests

DECAY_RATE_RANGE = (0.005, 0.10)  # Min and max decay rates to try
DECAY_RATE_STEPS = 20             # Number of steps in grid search
DEFAULT_DECAY_RATE = 0.02         # Default if tuning not performed

# =============================================================================
# 8. LIGHTGBM PARAMETERS
# =============================================================================
# LightGBM gradient boosting parameters. These control model complexity and
# learning speed.
#
# 🎛️ KEY TUNING PARAMETERS:
#    - n_estimators: Number of trees (more = better but slower)
#    - learning_rate: Step size (lower = more trees needed, but better)
#    - num_leaves: Complexity per tree (higher = more complex, risk overfit)
#    - feature_fraction: % of features per tree (lower = more regularization)
#    - bagging_fraction: % of data per tree (lower = more regularization)
#
# 💡 TIPS:
#    - If overfitting: Reduce num_leaves, increase feature_fraction/bagging_fraction
#    - If underfitting: Increase n_estimators, num_leaves
#    - For faster training: Reduce n_estimators, increase learning_rate

RANDOM_STATE = 42  # Random seed for reproducibility
TEST_SIZE = 0.2    # Validation split (20% of data for validation)
N_SPLITS_CV = 5    # Cross-validation folds

LGBM_PARAMS = {
    # Objective and metrics
    'objective': 'regression',      # Regression task (predict continuous volume)
    'metric': 'rmse',               # Root Mean Square Error for evaluation
    'boosting_type': 'gbdt',        # Gradient Boosting Decision Tree
    
    # Tree structure
    'num_leaves': 31,               # Max leaves per tree (2^5 - 1, default)
                                    # ↑ Higher = more complex, risk overfit
                                    # ↓ Lower = simpler, may underfit
    
    # Learning parameters
    'learning_rate': 0.03,          # Step size for each iteration
                                    # ↑ Higher = faster learning, less trees needed
                                    # ↓ Lower = slower but often better final result
    
    'n_estimators': 500,            # Number of boosting rounds (trees)
                                    # ↑ Higher = better but slower, may overfit
                                    # ↓ Lower = faster, may underfit
    
    # Regularization (prevent overfitting)
    'feature_fraction': 0.8,        # Use 80% of features per tree
    'bagging_fraction': 0.8,        # Use 80% of data per tree
    'bagging_freq': 5,              # Bagging every 5 iterations
    
    # Other
    'verbose': -1,                  # Suppress output (-1 = silent)
    'random_state': RANDOM_STATE    # Reproducibility
}

# =============================================================================
# 9. XGBOOST PARAMETERS
# =============================================================================
# XGBoost gradient boosting parameters. Similar to LightGBM but different API.
#
# 🎛️ KEY TUNING PARAMETERS:
#    - n_estimators: Number of trees
#    - learning_rate: Step size (eta in XGBoost terms)
#    - max_depth: Maximum depth per tree (instead of num_leaves)
#    - subsample: Row sampling (like bagging_fraction)
#    - colsample_bytree: Feature sampling (like feature_fraction)
#
# 💡 LIGHTGBM VS XGBOOST:
#    - LightGBM uses num_leaves (leaf-wise growth)
#    - XGBoost uses max_depth (depth-wise growth)
#    - LightGBM is usually faster on large datasets

XGB_PARAMS = {
    # Objective and metrics
    'objective': 'reg:squarederror',  # Regression with squared error
    'eval_metric': 'rmse',            # RMSE for evaluation
    
    # Tree structure
    'max_depth': 6,                   # Maximum depth per tree
                                      # ↑ Higher = more complex, risk overfit
                                      # ↓ Lower = simpler, may underfit
    
    # Learning parameters
    'learning_rate': 0.03,            # Step size (eta)
    'n_estimators': 500,              # Number of boosting rounds
    
    # Regularization
    'subsample': 0.8,                 # Use 80% of data per tree
    'colsample_bytree': 0.8,          # Use 80% of features per tree
    
    # Other
    'random_state': RANDOM_STATE      # Reproducibility
}

# =============================================================================
# 10. HYBRID MODEL PARAMETERS
# =============================================================================
# The hybrid model combines physics-based exponential decay with ML correction:
#    prediction = decay_baseline + ml_residual
#
# This often outperforms pure ML because the decay captures the known physics
# of generic erosion, and ML learns the brand-specific deviations.
#
# HYBRID_ML_TYPE: Which ML model to use for residual learning
#                 Options: 'lightgbm' or 'xgboost'

HYBRID_PARAMS = {
    'ml_type': 'lightgbm',            # 'lightgbm' or 'xgboost'
    'use_decay_features': True,       # Include decay-based features
    'residual_clip': None,            # Optional: clip extreme residuals
}

# =============================================================================
# 11. ARHOW MODEL PARAMETERS (ARIMA + Holt-Winters)
# =============================================================================
# The ARHOW model is a time-series ensemble that combines:
#    1. SARIMAX (Seasonal ARIMA with eXogenous variables)
#    2. Holt-Winters Exponential Smoothing
#
# It learns optimal weights: y_hat = β₀ × ARIMA + β₁ × HW
#
# 📊 ARIMA ORDER (p, d, q):
#    p = AutoRegressive terms (how many past values to use)
#    d = Differencing order (0 = stationary, 1 = first difference)
#    q = Moving Average terms (how many past errors to use)
#    Common choices: (1,1,1), (2,1,2), (1,0,1)
#
# 📊 SEASONAL ORDER (P, D, Q, s):
#    Same as ARIMA but for seasonal component
#    s = seasonal period (12 for monthly data with yearly seasonality)
#    (0,0,0,0) = no seasonality
#
# 📊 HOLT-WINTERS:
#    hw_trend: 'add' (additive), 'mul' (multiplicative), or None
#    hw_seasonal: 'add', 'mul', or None
#    hw_seasonal_periods: Seasonal period (12 for monthly)
#
# 📊 WEIGHT LEARNING:
#    weight_window: How many observations to use for learning β₀, β₁
#                   Larger = more stable but less adaptive
#                   Smaller = more adaptive but noisier

ARIHOW_PARAMS = {
    # SARIMAX parameters
    'arima_order': (2, 1, 2),         # ARIMA(1,1,1) - simple and robust
                                       # Try (2,1,2) for more complex patterns
    
    'seasonal_order': (0, 0, 0, 0),   # No seasonality (our data is monthly sales)
                                       # Try (1,0,1,12) if yearly patterns exist
    
    # Holt-Winters parameters  
    'hw_trend': 'add',                 # Additive trend (linear decline)
                                       # 'mul' for multiplicative (% decline)
                                       # None for no trend component
    
    'hw_seasonal': None,               # No seasonality in HW
                                       # 'add' or 'mul' if seasonal patterns
    
    'hw_seasonal_periods': 12,         # Seasonal period (12 = yearly for monthly data)
    
    # Weight learning
    'weight_window': 12,               # Use last 12 months to learn weights
                                       # ↑ Higher = more stable weights
                                       # ↓ Lower = more adaptive to recent data
    
    # Fitting parameters
    'min_history_months': 6,           # Minimum months needed to fit
    'suppress_warnings': True,         # Suppress statsmodels warnings
}

# =============================================================================
# 12. FEATURE ENGINEERING PARAMETERS (Enhanced from EDA Report)
# =============================================================================
# These control what features are created from the raw data.
#
# From EDA Report Section 12 - Feature Engineering Recommendations:
#   - Lag features: Critical for time-series (vol_lag1, lag3, lag6)
#   - Rolling statistics: 3m and 6m windows most important
#   - Competition: Log-transform n_gxs, cap at 15
#   - Time: Period indicators, squared/sqrt transforms
#
# LAG_WINDOWS: Create features from N months ago
#              e.g., [1, 3, 6, 12] creates volume_lag_1, volume_lag_3, etc.
#
# ROLLING_WINDOWS: Create rolling statistics over N months
#                  e.g., [3, 6, 12] creates rolling_mean_3, rolling_std_3, etc.
#
# 💡 MORE WINDOWS = MORE FEATURES = POTENTIALLY BETTER BUT SLOWER
#    Start with defaults, add more if model underfits

LAG_WINDOWS = [1, 3, 6, 12]           # Create lag features for these months
ROLLING_WINDOWS = [3, 6, 12]          # Create rolling mean/std/min/max for these windows
PCT_CHANGE_WINDOWS = [1, 3, 6]        # Create percent change for these periods

# Competition feature settings (from EDA Section 5 & 11)
N_GXS_CAP = 15                        # Cap n_gxs outliers at 99th percentile
HIGH_COMPETITION_THRESHOLD = 5        # Binary threshold: volume plateaus after 5-6 competitors

# Pre-entry feature aggregations (computed from months before generic entry)
PRE_ENTRY_FEATURES = {
    'compute_slope': True,            # Pre-entry trend (slope of linear fit)
    'compute_volatility': True,       # Pre-entry volatility (std / mean)
    'compute_momentum': True,         # Recent vs older average ratio
}

# Feature engineering toggles
FEATURE_ENGINEERING_SETTINGS = {
    # Which feature groups to include
    'include_lag_features': True,           # Lag features (volume_lag_1, etc.)
    'include_rolling_features': True,       # Rolling statistics
    'include_competition_features': True,   # n_gxs transformations
    'include_time_features': True,          # Time period indicators
    'include_hospital_features': True,      # Hospital rate bucketing
    'include_therapeutic_features': True,   # Therapeutic area encoding
    'include_interaction_features': True,   # Cross-feature interactions
    'include_pre_entry_features': True,     # Pre-entry aggregates
    
    # Outlier handling
    'cap_n_gxs': True,                      # Cap n_gxs at N_GXS_CAP
    'clip_vol_norm': True,                  # Clip vol_norm to [0, 1.5]
}

# Optional visibility/collaboration/context feature toggles (default off)
VISIBILITY_FEATURES = {
    'use_visibility_features': False,        # Visibility-style flags (drops/spikes, data gaps)
    'use_collaboration_features': False,     # Peer/cross-country aggregates
    'max_event_lag': 24,                     # Max months to look back for events
    'sources': {
        'holidays': False,
        'epidemics': False,
        'macro': False,
        'promotions': False,
        'stockout_flags': False,
    }
}

# =============================================================================
# 13. COLUMN DEFINITIONS
# =============================================================================
# Data schema - column names in the input/output data.
# Update if your data has different column names.

ID_COLS = ['country', 'brand_name']   # Unique identifier columns
TIME_COLS = ['month', 'months_postgx'] # Time-related columns
TARGET_COL = 'volume'                  # Target variable to predict

# Feature types (for encoding and processing)
CATEGORICAL_COLS = ['country', 'ther_area', 'main_package']
BINARY_COLS = ['biological', 'small_molecule']
NUMERIC_COLS = ['hospital_rate', 'n_gxs']


# =============================================================================
# TESTING & VALIDATION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\n🎮 RUN MODE:")
    print(f"   Scenario: {RUN_SCENARIO}")
    print(f"   Test mode: {TEST_MODE} ({TEST_MODE_BRANDS} brands)")
    
    print(f"\n🔄 PIPELINE STEPS:")
    print(f"   Run EDA: {RUN_EDA}")
    print(f"   Run Training: {RUN_TRAINING}")
    print(f"   Run Submission: {RUN_SUBMISSION}")
    print(f"   Run Validation: {RUN_VALIDATION}")
    
    print(f"\n🤖 MODELS ENABLED (Training):")
    for model, enabled in MODELS_ENABLED.items():
        status = "✅" if enabled else "❌"
        print(f"   {status} {model}")
    
    print(f"\n📤 SUBMISSIONS ENABLED:")
    for model, enabled in SUBMISSIONS_ENABLED.items():
        status = "✅" if enabled else "❌"
        print(f"   {status} {model}")
    
    print(f"\n📁 Project Paths:")
    print(f"   Root: {PROJECT_ROOT}")
    print(f"   Data: {DATA_RAW}")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Reports: {REPORTS_DIR}")
    
    print(f"\n📊 Competition Constants:")
    print(f"   Pre-entry months: {PRE_ENTRY_MONTHS}")
    print(f"   Post-entry months: {POST_ENTRY_MONTHS}")
    print(f"   Bucket 1 threshold: {BUCKET_1_THRESHOLD}")
    print(f"   Bucket weights: {BUCKET_1_WEIGHT}× (B1), {BUCKET_2_WEIGHT}× (B2)")
    
    print(f"\n⚖️ Scoring Weights:")
    print(f"   Scenario 1: monthly={S1_MONTHLY_WEIGHT}, sum_0-5={S1_SUM_0_5_WEIGHT}, "
          f"sum_6-11={S1_SUM_6_11_WEIGHT}, sum_12-23={S1_SUM_12_23_WEIGHT}")
    print(f"   Scenario 2: monthly={S2_MONTHLY_WEIGHT}, sum_6-11={S2_SUM_6_11_WEIGHT}, "
          f"sum_12-23={S2_SUM_12_23_WEIGHT}")
    
    print(f"\n🎛️ Model Parameters:")
    print(f"   Random state: {RANDOM_STATE}")
    print(f"   Test size: {TEST_SIZE}")
    print(f"   CV folds: {N_SPLITS_CV}")
    print(f"   Decay rate range: {DECAY_RATE_RANGE}")
    print(f"   Default decay rate: {DEFAULT_DECAY_RATE}")
    
    print(f"\n🌲 LightGBM:")
    print(f"   n_estimators: {LGBM_PARAMS['n_estimators']}")
    print(f"   learning_rate: {LGBM_PARAMS['learning_rate']}")
    print(f"   num_leaves: {LGBM_PARAMS['num_leaves']}")
    
    print(f"\n🌲 XGBoost:")
    print(f"   n_estimators: {XGB_PARAMS['n_estimators']}")
    print(f"   learning_rate: {XGB_PARAMS['learning_rate']}")
    print(f"   max_depth: {XGB_PARAMS['max_depth']}")
    
    print(f"\n📈 ARHOW:")
    print(f"   ARIMA order: {ARIHOW_PARAMS['arima_order']}")
    print(f"   Seasonal order: {ARIHOW_PARAMS['seasonal_order']}")
    print(f"   HW trend: {ARIHOW_PARAMS['hw_trend']}")
    print(f"   Weight window: {ARIHOW_PARAMS['weight_window']}")
    
    print(f"\n🔧 Feature Engineering:")
    print(f"   Lag windows: {LAG_WINDOWS}")
    print(f"   Rolling windows: {ROLLING_WINDOWS}")
    print(f"   Pct change windows: {PCT_CHANGE_WINDOWS}")
    
    print("\n✅ Checking data files:")
    for f in [VOLUME_TRAIN, GENERICS_TRAIN, MEDICINE_INFO_TRAIN, 
              VOLUME_TEST, GENERICS_TEST, MEDICINE_INFO_TEST]:
        status = "✅" if f.exists() else "❌"
        print(f"   {status} {f.name}")
    
    print("\n" + "=" * 60)
