# =============================================================================
# src/__init__.py - Novartis Datathon 2025 Main Package
# =============================================================================

from .config import *
from .data_loader import (
    load_all_data, load_volume_data, load_generics_data, load_medicine_info,
    merge_datasets, validate_data, split_train_validation
)
from .bucket_calculator import (
    compute_avg_j, compute_normalized_volume, compute_mean_erosion,
    assign_buckets, create_auxiliary_file
)
from .feature_engineering import (
    create_all_features, get_feature_columns,
    create_lag_features, create_rolling_features, create_competition_features,
    create_time_features, create_pre_entry_features
)
from .models import (
    BaselineModels, GradientBoostingModel,
    prepare_training_data, train_and_evaluate
)
from .evaluation import (
    evaluate_model, compute_pe_scenario1, compute_pe_scenario2,
    compute_final_metric, analyze_worst_predictions, compare_models
)
from .submission import (
    generate_submission, validate_submission, save_submission
)

__version__ = "1.0.0"
__author__ = "Novartis Datathon Team"
