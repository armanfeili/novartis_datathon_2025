# =============================================================================
# File: src/config.py
# Description: All project paths, constants, and configuration
# =============================================================================

from pathlib import Path
import os

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, SUBMISSIONS_DIR, REPORTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA FILES
# =============================================================================
VOLUME_TRAIN = DATA_RAW / "df_volume_train.csv"
GENERICS_TRAIN = DATA_RAW / "df_generics_train.csv"
MEDICINE_INFO_TRAIN = DATA_RAW / "df_medicine_info_train.csv"
VOLUME_TEST = DATA_RAW / "df_volume_test.csv"
GENERICS_TEST = DATA_RAW / "df_generics_test.csv"
MEDICINE_INFO_TEST = DATA_RAW / "df_medicine_info_test.csv"

# =============================================================================
# CONSTANTS
# =============================================================================
PRE_ENTRY_MONTHS = 12      # Months before generic entry for Avg_j calculation
POST_ENTRY_MONTHS = 24     # Months to forecast (0-23)
BUCKET_1_THRESHOLD = 0.25  # Mean erosion <= 0.25 = Bucket 1 (high erosion)
BUCKET_1_WEIGHT = 2        # Bucket 1 weighted 2x in metric
BUCKET_2_WEIGHT = 1        # Bucket 2 weighted 1x in metric

# =============================================================================
# METRIC WEIGHTS - SCENARIO 1 (Predict months 0-23)
# =============================================================================
S1_MONTHLY_WEIGHT = 0.2      # Weight for monthly absolute errors
S1_SUM_0_5_WEIGHT = 0.5      # Weight for accumulated error months 0-5
S1_SUM_6_11_WEIGHT = 0.2     # Weight for accumulated error months 6-11
S1_SUM_12_23_WEIGHT = 0.1    # Weight for accumulated error months 12-23

# =============================================================================
# METRIC WEIGHTS - SCENARIO 2 (Predict months 6-23, given 0-5 actuals)
# =============================================================================
S2_MONTHLY_WEIGHT = 0.2      # Weight for monthly absolute errors
S2_SUM_6_11_WEIGHT = 0.5     # Weight for accumulated error months 6-11
S2_SUM_12_23_WEIGHT = 0.3    # Weight for accumulated error months 12-23

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5

# LightGBM default parameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 500,
    'random_state': RANDOM_STATE
}

# XGBoost default parameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE
}

# =============================================================================
# FEATURE COLUMNS
# =============================================================================
ID_COLS = ['country', 'brand_name']
TIME_COLS = ['month', 'months_postgx']
TARGET_COL = 'volume'

CATEGORICAL_COLS = ['country', 'ther_area', 'main_package']
BINARY_COLS = ['biological', 'small_molecule']
NUMERIC_COLS = ['hospital_rate', 'n_gxs']


if __name__ == "__main__":
    # Test configuration
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data raw: {DATA_RAW}")
    print(f"All directories created successfully!")
    
    # Check if data files exist
    print("\nChecking data files:")
    for f in [VOLUME_TRAIN, GENERICS_TRAIN, MEDICINE_INFO_TRAIN, 
              VOLUME_TEST, GENERICS_TEST, MEDICINE_INFO_TEST]:
        status = "✅" if f.exists() else "❌"
        print(f"  {status} {f.name}")
