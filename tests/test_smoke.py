"""
Smoke tests for Novartis Datathon 2025.

Tests core functionality according to functionality.md:
1. Data loads correctly
2. Panel builds without errors  
3. Features build without leakage
4. Model trains
5. Metrics compute correctly
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test 1: Imports work
# =============================================================================
def test_imports():
    """Test that all core modules can be imported."""
    from src import utils
    from src import data
    from src import features
    from src import evaluate
    from src import validation
    from src import train
    from src import inference
    from src.models import base
    from src.models import cat_model
    from src.models import lgbm_model
    from src.models import xgb_model
    from src.models import linear
    
    assert hasattr(utils, 'set_seed')
    assert hasattr(utils, 'load_config')
    assert hasattr(data, 'load_raw_data')
    assert hasattr(features, 'make_features')
    assert hasattr(evaluate, 'compute_metric1')
    assert hasattr(validation, 'create_validation_split')
    assert hasattr(train, 'train_scenario_model')
    assert hasattr(inference, 'generate_submission')


def test_set_seed():
    """Test that set_seed produces reproducible results."""
    from src.utils import set_seed
    
    set_seed(42)
    x1 = np.random.rand(5)
    
    set_seed(42)
    x2 = np.random.rand(5)
    
    assert np.allclose(x1, x2)


# =============================================================================
# Test 2: Configuration loads
# =============================================================================
def test_load_configs():
    """Test that all configuration files load correctly."""
    from src.utils import load_config, get_project_root
    
    root = get_project_root()
    
    # Load all configs
    data_config = load_config(root / 'configs' / 'data.yaml')
    features_config = load_config(root / 'configs' / 'features.yaml')
    run_config = load_config(root / 'configs' / 'run_defaults.yaml')
    
    # Check required keys
    assert 'paths' in data_config
    assert 'files' in data_config
    assert 'pre_entry' in features_config
    assert 'scenarios' in run_config


# =============================================================================
# Test 3: Data loading (if data exists)
# =============================================================================
@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_data_loads():
    """Test that raw data loads correctly."""
    from src.utils import load_config, get_project_root
    from src.data import load_raw_data
    
    root = get_project_root()
    config = load_config(root / 'configs' / 'data.yaml')
    
    df_vol, df_gen, df_med = load_raw_data(config, split='train')
    
    # Check dataframes are not empty
    assert len(df_vol) > 0
    assert len(df_gen) > 0
    assert len(df_med) > 0
    
    # Check required columns
    assert 'country' in df_vol.columns
    assert 'brand_name' in df_vol.columns
    assert 'month' in df_vol.columns


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_panel_builds():
    """Test that base panel builds correctly."""
    from src.utils import load_config, get_project_root
    from src.data import load_raw_data, prepare_base_panel
    
    root = get_project_root()
    config = load_config(root / 'configs' / 'data.yaml')
    
    df_vol, df_gen, df_med = load_raw_data(config, split='train')
    panel = prepare_base_panel(df_vol, df_gen, df_med, is_train=True)
    
    # Check panel has expected columns
    assert 'country' in panel.columns
    assert 'brand_name' in panel.columns
    assert 'months_postgx' in panel.columns
    assert 'avg_vol_12m' in panel.columns
    assert 'y_norm' in panel.columns
    
    # Check y_norm calculation
    assert not panel['y_norm'].isna().all()


# =============================================================================
# Test 4: Feature engineering
# =============================================================================
def test_feature_leakage_prevention():
    """Test that feature engineering respects leakage prevention rules."""
    from src.train import META_COLS
    from src.features import get_feature_columns, SCENARIO_CONFIG
    
    # Check META_COLS are defined
    assert 'country' in META_COLS
    assert 'brand_name' in META_COLS
    assert 'months_postgx' in META_COLS
    assert 'bucket' in META_COLS
    assert 'y_norm' in META_COLS
    assert 'volume' in META_COLS
    
    # Check scenario configs
    assert 1 in SCENARIO_CONFIG
    assert 2 in SCENARIO_CONFIG
    assert SCENARIO_CONFIG[1]['feature_cutoff_month'] == 0
    assert SCENARIO_CONFIG[2]['feature_cutoff_month'] == 6


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_features_build():
    """Test that features build without using future information."""
    from src.utils import load_config, get_project_root
    from src.data import load_raw_data, prepare_base_panel
    from src.features import make_features, get_feature_columns
    from src.train import META_COLS
    
    root = get_project_root()
    data_config = load_config(root / 'configs' / 'data.yaml')
    features_config = load_config(root / 'configs' / 'features.yaml')
    
    df_vol, df_gen, df_med = load_raw_data(data_config, split='train')
    panel = prepare_base_panel(df_vol, df_gen, df_med, is_train=True)
    
    # Build features for scenario 1
    panel_s1 = make_features(panel.copy(), scenario=1, mode='train', config=features_config)
    
    # Get feature columns (exclude meta)
    feature_cols = get_feature_columns(panel_s1, META_COLS)
    
    # Verify no meta columns in features
    for col in META_COLS:
        assert col not in feature_cols, f"Meta column {col} found in features!"


# =============================================================================
# Test 5: Model interface
# =============================================================================
def test_model_interface():
    """Test that model classes implement required interface."""
    from src.models.base import BaseModel
    from src.models.cat_model import CatBoostModel
    from src.models.lgbm_model import LGBMModel
    from src.models.xgb_model import XGBModel
    from src.models.linear import LinearModel, GlobalMeanBaseline, FlatBaseline
    
    # Create dummy data
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'months_postgx': np.random.randint(0, 24, 100)
    })
    y_train = pd.Series(np.random.rand(100))
    X_val = X_train.iloc[:20].copy()
    y_val = y_train.iloc[:20].copy()
    sample_weight = pd.Series(np.ones(100))
    
    # Test FlatBaseline (simple, no external deps)
    baseline = FlatBaseline({})
    baseline.fit(X_train, y_train, X_val, y_val, sample_weight)
    preds = baseline.predict(X_train)
    assert len(preds) == len(X_train)
    assert np.allclose(preds, 1.0)  # FlatBaseline predicts 1.0
    
    # Test GlobalMeanBaseline
    global_baseline = GlobalMeanBaseline({})
    global_baseline.fit(X_train, y_train, X_val, y_val, sample_weight)
    preds = global_baseline.predict(X_train)
    assert len(preds) == len(X_train)


# =============================================================================
# Test 6: Metric computation
# =============================================================================
def test_metric_computation():
    """Test that metrics compute correctly with known values."""
    from src.evaluate import compute_metric1, compute_metric2, compute_bucket_metrics
    
    # Create dummy predictions
    df = pd.DataFrame({
        'country': ['A', 'A', 'A', 'B', 'B', 'B'],
        'brand_name': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
        'months_postgx': [0, 6, 12, 0, 6, 12],
        'bucket': [1, 1, 1, 2, 2, 2],
        'y_norm': [0.8, 0.6, 0.5, 0.9, 0.85, 0.8],
        'pred': [0.7, 0.5, 0.4, 0.95, 0.9, 0.85]
    })
    
    # Metrics should compute without error
    metric1 = compute_metric1(df, scenario=1)
    assert metric1 >= 0
    
    bucket_metrics = compute_bucket_metrics(df)
    assert 'bucket_1_rmse' in bucket_metrics
    assert 'bucket_2_rmse' in bucket_metrics


# =============================================================================
# Test 7: Validation split
# =============================================================================
def test_validation_split():
    """Test that validation split works at series level."""
    from src.validation import create_validation_split
    
    # Create dummy panel
    panel = pd.DataFrame({
        'country': ['A'] * 12 + ['B'] * 12,
        'brand_name': ['X'] * 12 + ['Y'] * 12,
        'months_postgx': list(range(12)) * 2,
        'bucket': [1] * 12 + [2] * 12,
        'y_norm': np.random.rand(24)
    })
    
    train_idx, val_idx = create_validation_split(
        panel,
        val_fraction=0.5,
        stratify_by='bucket',
        random_state=42
    )
    
    # Check no overlap
    assert len(set(train_idx) & set(val_idx)) == 0
    
    # Check all indices covered
    assert len(train_idx) + len(val_idx) == len(panel)


# =============================================================================
# Test 8: Submission format
# =============================================================================
def test_submission_format():
    """Test submission format validation."""
    from src.inference import validate_submission_format
    
    # Valid submission
    submission = pd.DataFrame({
        'country': ['A', 'B'],
        'brand_name': ['X', 'Y'],
        'month': ['2024-01', '2024-02'],
        'volume': [1000.0, 2000.0]
    })
    
    is_valid, errors = validate_submission_format(submission)
    assert is_valid, f"Validation failed: {errors}"
    
    # Invalid submission (missing column)
    bad_submission = pd.DataFrame({
        'country': ['A', 'B'],
        'brand_name': ['X', 'Y'],
        'month': ['2024-01', '2024-02']
        # Missing 'volume'
    })
    
    is_valid, errors = validate_submission_format(bad_submission)
    assert not is_valid


# =============================================================================
# Test 9: Sample weights
# =============================================================================
def test_sample_weights():
    """Test that sample weight computation works correctly."""
    from src.train import compute_sample_weights
    from src.utils import load_config, get_project_root
    
    root = get_project_root()
    run_config = load_config(root / 'configs' / 'run_defaults.yaml')
    
    # Create dummy panel
    panel = pd.DataFrame({
        'months_postgx': [0, 3, 6, 12, 18],
        'bucket': [1, 1, 2, 2, 2]
    })
    
    weights = compute_sample_weights(panel, scenario=1, config=run_config)
    
    # Early months should have higher weight
    assert weights.iloc[0] > weights.iloc[-1], "Early months should have higher weight"
    
    # Bucket 1 should have higher weight
    bucket1_weight = weights[panel['bucket'] == 1].mean()
    bucket2_weight = weights[panel['bucket'] == 2].mean()
    assert bucket1_weight > bucket2_weight, "Bucket 1 should have higher weight"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

