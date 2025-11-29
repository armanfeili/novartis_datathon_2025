# =============================================================================
# src/__init__.py - Novartis Datathon 2025 Main Package
# =============================================================================

from .config import *
from .data_loader import (
    load_all_data, load_volume_data, load_generics_data, load_medicine_info,
    merge_datasets, validate_data, split_train_validation,
    # New data cleaning functions
    remove_duplicates, verify_months_postgx_range, check_multiple_rows_per_month,
    create_time_to_50pct_features, impute_avg_vol_regression, create_vol_norm_gt1_flag,
    clean_data
)
from .bucket_calculator import (
    compute_avg_j, compute_normalized_volume, compute_mean_erosion,
    assign_buckets, create_auxiliary_file
)
from .feature_engineering import (
    create_all_features, get_feature_columns,
    create_lag_features, create_rolling_features, create_competition_features,
    create_time_features, create_pre_entry_features,
    # New feature engineering functions
    create_max_n_gxs_post_feature, create_early_postloe_features,
    create_horizon_as_row_dataset, create_brand_static_features,
    compute_sample_weights, compute_time_window_weights, compute_combined_sample_weights,
    # Scaling and target encoding (Section 1.4, 1.5)
    FeatureScaler, scale_features, target_encode_cv, create_target_encoded_features,
    # Additional brand-level features (Section 2.4)
    create_log_avg_vol, create_pre_loe_growth_flag
)
from .models import (
    BaselineModels, GradientBoostingModel,
    prepare_training_data, train_and_evaluate,
    # New ensemble blending
    EnsembleBlender, optimize_ensemble_weights
)
from .evaluation import (
    evaluate_model, compute_pe_scenario1, compute_pe_scenario2,
    compute_final_metric, analyze_worst_predictions, compare_models
)
from .submission import (
    generate_submission, validate_submission, save_submission
)

__version__ = "1.1.0"  # Updated with Todo implementations
__author__ = "Novartis Datathon Team"
