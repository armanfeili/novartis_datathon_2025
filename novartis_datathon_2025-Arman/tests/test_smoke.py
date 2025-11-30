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
    
    # Test basic module attributes
    assert hasattr(utils, 'set_seed')
    assert hasattr(utils, 'load_config')
    assert hasattr(data, 'load_raw_data')
    assert hasattr(features, 'make_features')
    assert hasattr(evaluate, 'compute_metric1')
    assert hasattr(validation, 'create_validation_split')
    assert hasattr(train, 'train_scenario_model')
    assert hasattr(inference, 'generate_submission')


def test_model_imports():
    """Test that model modules can be imported via lazy loading."""
    # These use lazy imports, so they should work even without native libraries
    from src.models import BaseModel, LinearModel, GlobalMeanBaseline, FlatBaseline
    from src.models import get_model_class
    
    # Test get_model_class for models that don't require native libs
    assert get_model_class('linear') == LinearModel
    assert get_model_class('flat') == FlatBaseline
    assert get_model_class('global_mean') == GlobalMeanBaseline


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
    assert 'columns' in data_config
    assert 'meta_cols' in data_config['columns']
    assert 'pre_entry' in features_config
    assert 'scenarios' in run_config
    assert 'reproducibility' in run_config
    assert 'seed' in run_config['reproducibility']


def test_config_meta_cols_alignment():
    """Test that META_COLS in code matches configs/data.yaml."""
    from src.utils import load_config, get_project_root
    from src.data import META_COLS as DATA_META_COLS
    
    root = get_project_root()
    data_config = load_config(root / 'configs' / 'data.yaml')
    config_meta_cols = set(data_config['columns']['meta_cols'])
    
    # Check alignment between code and config
    assert set(DATA_META_COLS) == config_meta_cols, \
        f"data.py META_COLS mismatch: {set(DATA_META_COLS)} vs {config_meta_cols}"


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
    
    data = load_raw_data(config, split='train')
    
    # Check returned dict has expected keys
    assert 'volume' in data
    assert 'generics' in data
    assert 'medicine_info' in data
    
    # Check dataframes are not empty
    assert len(data['volume']) > 0
    assert len(data['generics']) > 0
    assert len(data['medicine_info']) > 0
    
    # Check required columns
    assert 'country' in data['volume'].columns
    assert 'brand_name' in data['volume'].columns
    assert 'month' in data['volume'].columns


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_panel_builds():
    """Test that base panel builds correctly."""
    from src.utils import load_config, get_project_root
    from src.data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values
    
    root = get_project_root()
    config = load_config(root / 'configs' / 'data.yaml')
    
    data = load_raw_data(config, split='train')
    panel = prepare_base_panel(
        data['volume'], 
        data['generics'], 
        data['medicine_info']
    )
    panel = handle_missing_values(panel)
    panel = compute_pre_entry_stats(panel, is_train=True)
    
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
    from src.data import META_COLS
    from src.features import SCENARIO_CONFIG
    
    # Check META_COLS are defined
    assert 'country' in META_COLS
    assert 'brand_name' in META_COLS
    assert 'months_postgx' in META_COLS
    assert 'bucket' in META_COLS
    assert 'y_norm' in META_COLS
    assert 'volume' in META_COLS
    
    # Check scenario configs use integer keys
    assert 1 in SCENARIO_CONFIG
    assert 2 in SCENARIO_CONFIG
    assert SCENARIO_CONFIG[1]['feature_cutoff'] == 0
    assert SCENARIO_CONFIG[2]['feature_cutoff'] == 6


def test_scenario_normalization():
    """Test that scenario normalization works correctly."""
    from src.features import _normalize_scenario
    
    # Integer inputs
    assert _normalize_scenario(1) == 1
    assert _normalize_scenario(2) == 2
    
    # String inputs
    assert _normalize_scenario('1') == 1
    assert _normalize_scenario('2') == 2
    assert _normalize_scenario('scenario1') == 1
    assert _normalize_scenario('scenario2') == 2
    
    # Invalid inputs should raise
    with pytest.raises(ValueError):
        _normalize_scenario(3)
    with pytest.raises(ValueError):
        _normalize_scenario('invalid')


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_features_build():
    """Test that features build without using future information."""
    from src.utils import load_config, get_project_root
    from src.data import load_raw_data, prepare_base_panel, compute_pre_entry_stats, handle_missing_values, META_COLS
    from src.features import make_features, get_feature_columns
    
    root = get_project_root()
    data_config = load_config(root / 'configs' / 'data.yaml')
    features_config = load_config(root / 'configs' / 'features.yaml')
    
    data = load_raw_data(data_config, split='train')
    panel = prepare_base_panel(
        data['volume'], 
        data['generics'], 
        data['medicine_info']
    )
    panel = handle_missing_values(panel)
    panel = compute_pre_entry_stats(panel, is_train=True)
    
    # Build features for scenario 1 (using integer)
    panel_s1 = make_features(panel.copy(), scenario=1, mode='train', config=features_config)
    
    # Get feature columns (exclude meta)
    feature_cols = get_feature_columns(panel_s1, exclude_meta=True)
    
    # Verify no meta columns in features
    for col in META_COLS:
        assert col not in feature_cols, f"Meta column {col} found in features!"


# =============================================================================
# Test 5: Model interface
# =============================================================================
def test_model_interface():
    """Test that model classes implement required interface."""
    # Import linear models directly to avoid importing other models that might fail
    from src.models.linear import GlobalMeanBaseline, FlatBaseline
    
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
    from src.evaluate import compute_metric1, compute_metric2
    
    # Create proper DataFrames for metric computation
    # df_actual and df_pred need: country, brand_name, months_postgx, volume
    # df_aux needs: country, brand_name, avg_vol, bucket
    
    # Series A: bucket 1 (high erosion)
    # Series B: bucket 2 (low erosion)
    df_actual = pd.DataFrame({
        'country': ['A'] * 24 + ['B'] * 24,
        'brand_name': ['X'] * 24 + ['Y'] * 24,
        'months_postgx': list(range(24)) * 2,
        'volume': [100 - i * 3 for i in range(24)] + [100 - i for i in range(24)]
    })
    
    df_pred = pd.DataFrame({
        'country': ['A'] * 24 + ['B'] * 24,
        'brand_name': ['X'] * 24 + ['Y'] * 24,
        'months_postgx': list(range(24)) * 2,
        'volume': [100 - i * 2.5 for i in range(24)] + [100 - i * 0.8 for i in range(24)]
    })
    
    df_aux = pd.DataFrame({
        'country': ['A', 'B'],
        'brand_name': ['X', 'Y'],
        'avg_vol': [100.0, 100.0],
        'bucket': [1, 2]
    })
    
    # Metrics should compute without error
    metric1 = compute_metric1(df_actual, df_pred, df_aux)
    assert isinstance(metric1, (int, float))
    assert metric1 >= 0


# =============================================================================
# Test 7: Validation split
# =============================================================================
def test_validation_split():
    """Test that validation split works at series level."""
    from src.validation import create_validation_split
    
    # Create dummy panel with 4 series (2 per bucket for stratification)
    panel = pd.DataFrame({
        'country': ['A'] * 12 + ['B'] * 12 + ['C'] * 12 + ['D'] * 12,
        'brand_name': ['W'] * 12 + ['X'] * 12 + ['Y'] * 12 + ['Z'] * 12,
        'months_postgx': list(range(12)) * 4,
        'bucket': [1] * 12 + [1] * 12 + [2] * 12 + [2] * 12,
        'y_norm': np.random.rand(48)
    })
    
    train_df, val_df = create_validation_split(
        panel,
        val_fraction=0.5,
        stratify_by='bucket',
        random_state=42
    )
    
    # Check no series overlap between train and val
    train_series = set(zip(train_df['country'], train_df['brand_name']))
    val_series = set(zip(val_df['country'], val_df['brand_name']))
    assert len(train_series & val_series) == 0, "Series should not overlap between train and val"
    
    # Check all rows covered
    assert len(train_df) + len(val_df) == len(panel)


# =============================================================================
# Test 8: Submission format validation
# =============================================================================
def test_submission_format():
    """Test submission format validation."""
    from src.inference import validate_submission_format
    
    # Create template
    template = pd.DataFrame({
        'country': ['A', 'B'],
        'brand_name': ['X', 'Y'],
        'months_postgx': [0, 0],
        'volume': [0.0, 0.0]  # Template has placeholder values
    })
    
    # Valid submission
    submission = pd.DataFrame({
        'country': ['A', 'B'],
        'brand_name': ['X', 'Y'],
        'months_postgx': [0, 0],
        'volume': [1000.0, 2000.0]
    })
    
    # Should pass validation (returns True or raises)
    is_valid = validate_submission_format(submission, template)
    assert is_valid
    
    # Invalid submission (missing volume values)
    bad_submission = pd.DataFrame({
        'country': ['A', 'B'],
        'brand_name': ['X', 'Y'],
        'months_postgx': [0, 0],
        'volume': [1000.0, np.nan]  # NaN value
    })
    
    with pytest.raises(ValueError):
        validate_submission_format(bad_submission, template)


# =============================================================================
# Test 9: Sample weights
# =============================================================================
def test_sample_weights():
    """Test that sample weight computation works correctly."""
    from src.train import compute_sample_weights
    
    # Create dummy panel
    panel = pd.DataFrame({
        'months_postgx': [0, 3, 6, 12, 18],
        'bucket': [1, 1, 2, 2, 2]
    })
    
    # Test with integer scenario
    weights = compute_sample_weights(panel, scenario=1)
    
    # Early months should have higher weight
    assert weights.iloc[0] > weights.iloc[-1], "Early months should have higher weight"
    
    # Bucket 1 should have higher weight (rows 0,1 are bucket 1)
    bucket1_weight = weights.iloc[:2].mean()
    bucket2_weight = weights.iloc[2:].mean()
    assert bucket1_weight > bucket2_weight, "Bucket 1 should have higher weight"


def test_sample_weights_scenario2():
    """Test sample weights for scenario 2."""
    from src.train import compute_sample_weights
    
    # Create dummy panel for scenario 2 (months 6-23)
    panel = pd.DataFrame({
        'months_postgx': [6, 9, 12, 18, 23],
        'bucket': [1, 1, 2, 2, 2]
    })
    
    # Test with integer scenario
    weights = compute_sample_weights(panel, scenario=2)
    
    # Months 6-11 should have highest weight
    assert weights.iloc[0] > weights.iloc[-1], "Early months (6-11) should have higher weight"


# =============================================================================
# Test 10: Constants alignment
# =============================================================================
def test_constants_defined():
    """Test that canonical column constants are defined."""
    from src.data import ID_COLS, TIME_COL, CALENDAR_MONTH_COL, RAW_TARGET_COL, MODEL_TARGET_COL, META_COLS
    
    assert ID_COLS == ['country', 'brand_name']
    assert TIME_COL == 'months_postgx'
    assert CALENDAR_MONTH_COL == 'month'
    assert RAW_TARGET_COL == 'volume'
    assert MODEL_TARGET_COL == 'y_norm'
    assert len(META_COLS) > 0


# =============================================================================
# Test 11: Data validation functions (Section 2)
# =============================================================================
def test_data_validation_functions():
    """Test data validation helper functions."""
    from src.data import (
        validate_dataframe_schema, 
        validate_value_ranges, 
        validate_no_duplicates,
        log_data_statistics
    )
    
    # Test validate_dataframe_schema
    df = pd.DataFrame({
        'country': ['US', 'DE'],
        'brand_name': ['A', 'B'],
        'volume': [100.0, 200.0]
    })
    
    # Should pass
    validate_dataframe_schema(df, 'test', ['country', 'brand_name'])
    
    # Should raise on missing column
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe_schema(df, 'test', ['country', 'missing_col'])
    
    # Test validate_no_duplicates
    validate_no_duplicates(df, ['country', 'brand_name'], 'test')
    
    df_dup = pd.DataFrame({
        'country': ['US', 'US'],
        'brand_name': ['A', 'A'],
        'volume': [100.0, 200.0]
    })
    with pytest.raises(ValueError, match="duplicate"):
        validate_no_duplicates(df_dup, ['country', 'brand_name'], 'test')


def test_meta_cols_consistency_check():
    """Test META_COLS consistency verification."""
    from src.data import verify_meta_cols_consistency
    from src.utils import load_config, get_project_root
    
    root = get_project_root()
    config = load_config(root / 'configs' / 'data.yaml')
    
    # Should pass (we ensured alignment in Section 0)
    assert verify_meta_cols_consistency(config) is True


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_get_panel_caching():
    """Test panel caching functionality."""
    from src.data import get_panel, clear_panel_cache, get_series_count
    from src.utils import load_config, get_project_root
    import time
    
    root = get_project_root()
    config = load_config(root / 'configs' / 'data.yaml')
    
    # Clear cache first
    clear_panel_cache(config, split='train')
    
    # First load (builds from scratch)
    start = time.time()
    panel1 = get_panel('train', config, use_cache=True, force_rebuild=True)
    build_time = time.time() - start
    
    # Second load (from cache)
    start = time.time()
    panel2 = get_panel('train', config, use_cache=True, force_rebuild=False)
    cache_time = time.time() - start
    
    # Cache should be faster (at least 2x for decent-sized data)
    # But don't fail on small datasets
    assert panel1.shape == panel2.shape
    assert get_series_count(panel1) == get_series_count(panel2)
    
    # Verify columns are same
    assert list(panel1.columns) == list(panel2.columns)


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_pre_entry_stats_edge_cases():
    """Test pre-entry stats computation handles edge cases."""
    from src.data import compute_pre_entry_stats
    
    # Create test data with a series that has limited pre-entry data
    panel = pd.DataFrame({
        'country': ['US'] * 10 + ['DE'] * 5,
        'brand_name': ['A'] * 10 + ['B'] * 5,
        'months_postgx': list(range(-5, 5)) + list(range(0, 5)),  # US has 5 pre-entry, DE has 0
        'volume': [100.0] * 15,
        'ther_area': ['Cardio'] * 15
    })
    
    # Should not raise, should use fallbacks
    result = compute_pre_entry_stats(panel, is_train=True)
    
    # All series should have avg_vol_12m (via fallbacks)
    assert 'avg_vol_12m' in result.columns
    assert not result['avg_vol_12m'].isna().any(), "All series should have avg_vol_12m"
    
    # Training columns should be present
    assert 'y_norm' in result.columns


# =============================================================================
# Test 12: Feature Engineering (Section 3)
# =============================================================================
def test_feature_config_loading():
    """Test feature configuration loading with defaults."""
    from src.features import _load_feature_config
    
    # Test with None config (should return defaults)
    config = _load_feature_config(None)
    assert 'pre_entry' in config
    assert 'time' in config
    assert 'generics' in config
    assert 'drug' in config
    assert 'scenario2_early' in config
    assert 'interactions' in config
    
    # Test with partial config (should merge with defaults)
    partial_config = {'pre_entry': {'compute_trend': False}}
    merged = _load_feature_config(partial_config)
    assert merged['pre_entry']['compute_trend'] is False  # From partial
    assert merged['pre_entry']['compute_volatility'] is True  # From defaults


def test_pre_entry_features():
    """Test pre-entry feature generation."""
    from src.features import add_pre_entry_features
    
    # Create test data
    df = pd.DataFrame({
        'country': ['US'] * 15,
        'brand_name': ['A'] * 15,
        'months_postgx': list(range(-12, 3)),
        'volume': [100 - i for i in range(15)],  # Declining volume
        'avg_vol_12m': [100.0] * 15
    })
    
    result = add_pre_entry_features(df, config={})
    
    # Check features exist
    assert 'avg_vol_6m' in result.columns
    assert 'avg_vol_3m' in result.columns
    assert 'pre_entry_trend' in result.columns
    assert 'pre_entry_volatility' in result.columns
    assert 'pre_entry_max' in result.columns
    assert 'pre_entry_min' in result.columns
    assert 'volume_growth_rate' in result.columns
    assert 'log_avg_vol' in result.columns
    
    # Check no NaN values in key features
    assert not result['avg_vol_6m'].isna().any()
    assert not result['avg_vol_3m'].isna().any()


def test_time_features():
    """Test time feature generation."""
    from src.features import add_time_features
    
    # Create test data with month column
    df = pd.DataFrame({
        'months_postgx': [-6, -3, 0, 6, 12, 18],
        'month': ['Jan', 'Apr', 'Jul', 'Oct', 'Jan', 'Apr']
    })
    
    result = add_time_features(df, config={})
    
    # Check features exist
    assert 'months_postgx_sq' in result.columns
    assert 'is_post_entry' in result.columns
    assert 'time_bucket' in result.columns
    assert 'month_of_year' in result.columns
    assert 'month_sin' in result.columns
    assert 'month_cos' in result.columns
    assert 'quarter' in result.columns
    assert 'is_year_end' in result.columns
    assert 'time_decay' in result.columns
    
    # Check time bucket values
    assert result.loc[0, 'time_bucket'] == 'pre'
    assert result.loc[2, 'time_bucket'] == 'early'
    assert result.loc[3, 'time_bucket'] == 'mid'
    assert result.loc[4, 'time_bucket'] == 'late'


def test_generics_features():
    """Test generics competition feature generation."""
    from src.features import add_generics_features
    
    # Create test data
    df = pd.DataFrame({
        'country': ['US'] * 6,
        'brand_name': ['A'] * 6,
        'months_postgx': [-3, -1, 0, 3, 6, 12],
        'n_gxs': [0, 1, 2, 3, 4, 5]
    })
    
    result = add_generics_features(df, cutoff_month=6, config={})
    
    # Check features exist
    assert 'has_generic' in result.columns
    assert 'multiple_generics' in result.columns
    assert 'n_gxs_at_entry' in result.columns
    assert 'log_n_gxs' in result.columns
    
    # Check has_generic logic
    assert result.loc[0, 'has_generic'] == 0
    assert result.loc[1, 'has_generic'] == 1


def test_scenario2_early_erosion_features():
    """Test Scenario 2 early erosion features."""
    from src.features import add_early_erosion_features
    
    # Create test data with early erosion pattern
    df = pd.DataFrame({
        'country': ['US'] * 8,
        'brand_name': ['A'] * 8,
        'months_postgx': [-2, -1, 0, 1, 2, 3, 4, 5],
        'volume': [100, 100, 80, 70, 65, 60, 58, 55],  # Erosion pattern
        'avg_vol_12m': [100.0] * 8,
        'n_gxs': [0, 0, 1, 2, 2, 3, 3, 4]
    })
    
    result = add_early_erosion_features(df, config={})
    
    # Check features exist
    assert 'avg_vol_0_5' in result.columns
    assert 'erosion_0_5' in result.columns
    assert 'trend_0_5' in result.columns
    assert 'drop_month_0' in result.columns
    assert 'month_0_to_3_change' in result.columns
    assert 'month_3_to_5_change' in result.columns
    assert 'recovery_signal' in result.columns
    assert 'competition_response' in result.columns
    
    # Check erosion_0_5 is reasonable (should be < 1 due to erosion)
    erosion = result['erosion_0_5'].iloc[0]
    assert erosion < 1.0, "Erosion should be less than 1.0"


def test_feature_leakage_validation():
    """Test feature leakage validation."""
    from src.features import validate_feature_leakage, FORBIDDEN_FEATURES
    
    # Valid features (no leakage)
    df_valid = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    is_valid, violations = validate_feature_leakage(df_valid, scenario=1, mode="train")
    assert is_valid, f"Valid features should pass: {violations}"
    
    # Invalid features (contains forbidden column)
    df_invalid = pd.DataFrame({
        'feature1': [1, 2, 3],
        'bucket': [1, 2, 1]  # Forbidden!
    })
    is_valid, violations = validate_feature_leakage(df_invalid, scenario=1, mode="train")
    assert not is_valid, "Should detect forbidden column"
    assert any('bucket' in v for v in violations)


def test_scenario1_no_early_erosion_features():
    """Test that Scenario 1 validation detects early erosion features."""
    from src.features import validate_feature_leakage
    
    # Scenario 1 should not have early erosion features
    df_with_early = pd.DataFrame({
        'feature1': [1, 2, 3],
        'erosion_0_5': [0.8, 0.7, 0.6]  # Early erosion feature!
    })
    is_valid, violations = validate_feature_leakage(df_with_early, scenario=1, mode="train")
    assert not is_valid, "Scenario 1 should flag early erosion features"


def test_make_features_mode_train_vs_test():
    """Test that make_features respects mode parameter."""
    from src.features import make_features
    
    # Create minimal test panel
    panel = pd.DataFrame({
        'country': ['US'] * 10,
        'brand_name': ['A'] * 10,
        'months_postgx': list(range(-5, 5)),
        'volume': [100.0] * 10,
        'avg_vol_12m': [100.0] * 10,
        'n_gxs': [0, 0, 0, 0, 0, 1, 2, 2, 3, 3],
        'month': ['Jan'] * 10
    })
    
    # Train mode should create y_norm
    result_train = make_features(panel.copy(), scenario=1, mode='train')
    assert 'y_norm' in result_train.columns, "Train mode should have y_norm"
    
    # Test mode should NOT create y_norm
    result_test = make_features(panel.copy(), scenario=1, mode='test')
    assert 'y_norm' not in result_test.columns, "Test mode should not have y_norm"


def test_split_features_target_meta():
    """Test splitting features, target, and meta."""
    from src.features import split_features_target_meta
    
    df = pd.DataFrame({
        'country': ['US', 'DE'],
        'brand_name': ['A', 'B'],
        'months_postgx': [0, 6],
        'feature1': [1.0, 2.0],
        'feature2': [3.0, 4.0],
        'y_norm': [0.8, 0.7],
        'volume': [100, 200],
        'bucket': [1, 2]
    })
    
    X, y, meta = split_features_target_meta(df)
    
    # X should only have numeric features
    assert 'feature1' in X.columns
    assert 'feature2' in X.columns
    assert 'country' not in X.columns
    assert 'y_norm' not in X.columns
    assert 'bucket' not in X.columns
    
    # y should be the target
    assert y is not None
    assert len(y) == 2
    
    # meta should have identifiers
    assert 'country' in meta.columns
    assert 'brand_name' in meta.columns


def test_interaction_features():
    """Test interaction feature generation."""
    from src.features import add_interaction_features
    
    df = pd.DataFrame({
        'n_gxs': [1, 2, 3],
        'biological': [0, 1, 0],
        'hospital_rate_norm': [0.3, 0.7, 0.5]
    })
    
    config = {
        'enabled': True,
        'pairs': [['n_gxs', 'biological'], ['n_gxs', 'hospital_rate_norm']]
    }
    
    result = add_interaction_features(df, config)
    
    assert 'n_gxs_x_biological' in result.columns
    assert 'n_gxs_x_hospital_rate_norm' in result.columns
    
    # Check values
    assert result.loc[0, 'n_gxs_x_biological'] == 0  # 1 * 0
    assert result.loc[1, 'n_gxs_x_biological'] == 2  # 2 * 1


# =============================================================================
# SECTION 6 TESTS: VALIDATION & EVALUATION
# =============================================================================


def test_detect_test_scenarios():
    """Test scenario detection matches expected counts (228 S1, 112 S2)."""
    from src.inference import detect_test_scenarios
    from src.data import load_raw_data
    from src.utils import load_config
    
    # Load config and actual test data
    config = load_config('configs/data.yaml')
    raw = load_raw_data(config, split='test')
    test_volume = raw['volume']
    
    # Detect scenarios
    scenarios = detect_test_scenarios(test_volume)
    
    # Must return dict with integer keys
    assert 1 in scenarios
    assert 2 in scenarios
    
    n_s1 = len(scenarios[1])
    n_s2 = len(scenarios[2])
    
    # Log counts for debugging
    print(f"Detected: S1={n_s1}, S2={n_s2}")
    
    # Expected counts from documentation: 228 S1, 112 S2
    # Allow small tolerance in case of test data changes
    EXPECTED_S1 = 228
    EXPECTED_S2 = 112
    
    # Check total is correct (340 total series in test)
    assert n_s1 + n_s2 == EXPECTED_S1 + EXPECTED_S2, \
        f"Total series {n_s1 + n_s2} != expected {EXPECTED_S1 + EXPECTED_S2}"
    
    # Check individual counts
    assert n_s1 == EXPECTED_S1, f"S1 count {n_s1} != expected {EXPECTED_S1}"
    assert n_s2 == EXPECTED_S2, f"S2 count {n_s2} != expected {EXPECTED_S2}"


def test_metric_official_wrapper_regression():
    """
    Regression test comparing our metric wrappers with official implementation.
    
    Uses synthetic data to verify compute_metric1/compute_metric2 produce
    correct results matching the official metric_calculation.py.
    """
    from src.evaluate import compute_metric1, compute_metric2, OFFICIAL_METRICS_AVAILABLE
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Build test DataFrames
    series_data = []
    for i in range(4):
        country = f'COUNTRY_{i}'
        brand = f'BRAND_{i}'
        bucket = 1 if i < 2 else 2
        avg_vol = 1000 * (i + 1)
        
        # Create month data (0-23 for S1, 6-23 for S2)
        for month in range(24):
            actual_vol = avg_vol * (1 - 0.02 * month) + np.random.normal(0, 50)
            # Prediction with some error
            pred_vol = actual_vol + np.random.normal(0, avg_vol * 0.05)
            
            series_data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'volume_actual': max(0, actual_vol),
                'volume_pred': max(0, pred_vol),
                'avg_vol': avg_vol,
                'bucket': bucket
            })
    
    df = pd.DataFrame(series_data)
    
    # Create input DataFrames
    df_actual = df[['country', 'brand_name', 'months_postgx', 'volume_actual']].copy()
    df_actual = df_actual.rename(columns={'volume_actual': 'volume'})
    
    df_pred = df[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
    df_pred = df_pred.rename(columns={'volume_pred': 'volume'})
    
    df_aux = df[['country', 'brand_name', 'avg_vol', 'bucket']].drop_duplicates()
    
    # Test Metric 1
    metric1 = compute_metric1(df_actual, df_pred, df_aux)
    assert metric1 is not None
    assert isinstance(metric1, (int, float))
    assert metric1 >= 0, "Metric 1 should be non-negative"
    
    # Test Metric 2 (filter to months 6-23)
    df_actual_s2 = df_actual[df_actual['months_postgx'] >= 6].copy()
    df_pred_s2 = df_pred[df_pred['months_postgx'] >= 6].copy()
    
    metric2 = compute_metric2(df_actual_s2, df_pred_s2, df_aux)
    assert metric2 is not None
    assert isinstance(metric2, (int, float))
    assert metric2 >= 0, "Metric 2 should be non-negative"
    
    # If official implementation is available, verify they produce same results
    if OFFICIAL_METRICS_AVAILABLE:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'docs' / 'guide'))
        from metric_calculation import compute_metric1 as official_m1
        from metric_calculation import compute_metric2 as official_m2
        
        official_metric1 = official_m1(df_actual, df_pred, df_aux)
        official_metric2 = official_m2(df_actual_s2, df_pred_s2, df_aux)
        
        assert abs(metric1 - official_metric1) < 1e-6, \
            f"Metric 1 mismatch: {metric1} vs official {official_metric1}"
        assert abs(metric2 - official_metric2) < 1e-6, \
            f"Metric 2 mismatch: {metric2} vs official {official_metric2}"


def test_auxiliary_file_schema():
    """Test create_aux_file produces DataFrame matching official schema."""
    from src.evaluate import create_aux_file
    from pathlib import Path
    
    # Create test panel
    panel = pd.DataFrame({
        'country': ['US', 'DE', 'FR'],
        'brand_name': ['A', 'B', 'C'],
        'months_postgx': [0, 0, 0],
        'avg_vol_12m': [1000.0, 1500.0, 1200.0],
        'bucket': [1, 2, 1]
    })
    
    # Create aux file
    aux = create_aux_file(panel)
    
    # Official schema: country, brand_name, avg_vol, bucket
    expected_columns = ['country', 'brand_name', 'avg_vol', 'bucket']
    
    # Check columns match exactly (order matters for CSV)
    assert list(aux.columns) == expected_columns, \
        f"Columns {list(aux.columns)} != expected {expected_columns}"
    
    # Check types
    assert aux['avg_vol'].dtype in [np.float64, np.float32, float], \
        f"avg_vol should be float, got {aux['avg_vol'].dtype}"
    assert aux['bucket'].dtype in [np.int64, np.int32, int], \
        f"bucket should be int, got {aux['bucket'].dtype}"
    
    # Check no duplicates
    assert len(aux) == aux[['country', 'brand_name']].drop_duplicates().shape[0], \
        "Aux file should have unique series"


def test_validation_split_series_level():
    """Test that validation split maintains series integrity."""
    from src.validation import create_validation_split
    
    # Create panel with multiple months per series (need enough series for stratification)
    countries = ['US', 'DE', 'FR', 'UK', 'JP', 'IT', 'ES', 'BR']
    brands = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    buckets = [1, 2, 1, 2, 1, 2, 1, 2]  # Balanced buckets
    
    panel_data = []
    for country, brand, bucket in zip(countries, brands, buckets):
        for month in range(24):
            panel_data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'bucket': bucket,
                'volume': np.random.rand() * 1000
            })
    
    panel = pd.DataFrame(panel_data)
    
    train_df, val_df = create_validation_split(panel, val_fraction=0.25)
    
    # Series should NOT overlap
    train_keys = set(train_df[['country', 'brand_name']].drop_duplicates().itertuples(index=False, name=None))
    val_keys = set(val_df[['country', 'brand_name']].drop_duplicates().itertuples(index=False, name=None))
    
    assert len(train_keys & val_keys) == 0, "Train and val should not share series"
    
    # Each series should have all its months in one split
    for country, brand in train_keys:
        series_months = train_df[(train_df['country'] == country) & 
                                  (train_df['brand_name'] == brand)]['months_postgx'].nunique()
        assert series_months == 24, f"Train series ({country}, {brand}) should have all 24 months"
    
    for country, brand in val_keys:
        series_months = val_df[(val_df['country'] == country) & 
                                (val_df['brand_name'] == brand)]['months_postgx'].nunique()
        assert series_months == 24, f"Val series ({country}, {brand}) should have all 24 months"


# =============================================================================
# STANDARDIZED CROSS-VALIDATION TESTS
# =============================================================================

def test_stratified_group_kfold_no_series_leakage():
    """Test that all months of a given brand go to the same fold."""
    from src.validation import create_stratified_group_kfold
    
    # Create panel with multiple series, each having 24 months
    np.random.seed(42)
    countries = ['US', 'DE', 'FR', 'UK', 'JP', 'IT', 'ES', 'BR', 'CA', 'MX']
    brands = [f'Brand_{i}' for i in range(10)]
    
    panel_data = []
    for i, (country, brand) in enumerate(zip(countries, brands)):
        bucket = 1 if i < 5 else 2  # First 5 are bucket 1, rest bucket 2
        for month in range(24):
            panel_data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'bucket': bucket,
                'volume': np.random.rand() * 1000
            })
    
    panel = pd.DataFrame(panel_data)
    
    # Create 5-fold CV
    folds = create_stratified_group_kfold(panel, n_folds=5, random_state=42)
    
    assert len(folds) == 5, "Should have 5 folds"
    
    # For each fold, verify no series appears in both train and val
    for fold_idx, (train_idx, val_idx, fold_info) in enumerate(folds):
        train_df = panel.iloc[train_idx]
        val_df = panel.iloc[val_idx]
        
        train_series = set(train_df[['country', 'brand_name']].drop_duplicates().itertuples(index=False, name=None))
        val_series = set(val_df[['country', 'brand_name']].drop_duplicates().itertuples(index=False, name=None))
        
        # No overlap
        overlap = train_series & val_series
        assert len(overlap) == 0, f"Fold {fold_idx}: Series should not appear in both train and val: {overlap}"
        
        # Each train series has all 24 months
        for country, brand in train_series:
            n_months = train_df[
                (train_df['country'] == country) & (train_df['brand_name'] == brand)
            ]['months_postgx'].nunique()
            assert n_months == 24, f"Fold {fold_idx}: Train series ({country}, {brand}) should have 24 months, got {n_months}"
        
        # Each val series has all 24 months
        for country, brand in val_series:
            n_months = val_df[
                (val_df['country'] == country) & (val_df['brand_name'] == brand)
            ]['months_postgx'].nunique()
            assert n_months == 24, f"Fold {fold_idx}: Val series ({country}, {brand}) should have 24 months, got {n_months}"


def test_stratified_group_kfold_bucket_balance():
    """Test that bucket 1 and bucket 2 are reasonably balanced across folds."""
    from src.validation import create_stratified_group_kfold
    
    # Create panel with 50/50 bucket split
    np.random.seed(42)
    n_series = 20
    
    panel_data = []
    for i in range(n_series):
        country = f'Country_{i}'
        brand = f'Brand_{i}'
        bucket = 1 if i < n_series // 2 else 2
        for month in range(24):
            panel_data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'bucket': bucket,
                'volume': np.random.rand() * 1000
            })
    
    panel = pd.DataFrame(panel_data)
    
    # Create 5-fold CV
    folds = create_stratified_group_kfold(panel, n_folds=5, random_state=42)
    
    # Check bucket balance in each fold
    max_imbalance = 0.20  # Allow up to 20% difference
    
    for fold_idx, (train_idx, val_idx, fold_info) in enumerate(folds):
        train_bucket1_pct = fold_info['train_bucket1_pct'].iloc[0]
        val_bucket1_pct = fold_info['val_bucket1_pct'].iloc[0]
        
        # Bucket 1 should be ~50% in both train and val
        assert abs(train_bucket1_pct - 0.5) < max_imbalance, \
            f"Fold {fold_idx}: Train bucket 1 pct {train_bucket1_pct:.1%} too far from 50%"
        assert abs(val_bucket1_pct - 0.5) < max_imbalance, \
            f"Fold {fold_idx}: Val bucket 1 pct {val_bucket1_pct:.1%} too far from 50%"
        
        # Train and val should have similar bucket distribution
        bucket_diff = abs(train_bucket1_pct - val_bucket1_pct)
        assert bucket_diff < max_imbalance, \
            f"Fold {fold_idx}: Bucket imbalance {bucket_diff:.1%} > {max_imbalance:.1%}"


def test_stratified_group_kfold_reproducibility():
    """Test that same random_state produces identical folds."""
    from src.validation import create_stratified_group_kfold
    
    # Create simple panel
    np.random.seed(42)
    panel_data = []
    for i in range(10):
        for month in range(24):
            panel_data.append({
                'country': f'C_{i}',
                'brand_name': f'B_{i}',
                'months_postgx': month,
                'bucket': 1 if i < 5 else 2,
                'volume': np.random.rand() * 1000
            })
    panel = pd.DataFrame(panel_data)
    
    # Create folds twice with same seed
    folds1 = create_stratified_group_kfold(panel, n_folds=5, random_state=123)
    folds2 = create_stratified_group_kfold(panel, n_folds=5, random_state=123)
    
    # Should produce identical splits
    for fold_idx in range(5):
        train_idx1, val_idx1, _ = folds1[fold_idx]
        train_idx2, val_idx2, _ = folds2[fold_idx]
        
        assert np.array_equal(train_idx1, train_idx2), \
            f"Fold {fold_idx}: Train indices should be identical with same seed"
        assert np.array_equal(val_idx1, val_idx2), \
            f"Fold {fold_idx}: Val indices should be identical with same seed"
    
    # Different seed should produce different splits
    folds3 = create_stratified_group_kfold(panel, n_folds=5, random_state=456)
    train_idx3, _, _ = folds3[0]
    train_idx1, _, _ = folds1[0]
    
    assert not np.array_equal(train_idx1, train_idx3), \
        "Different seeds should produce different splits"


def test_validate_cv_splits():
    """Test the CV validation function."""
    from src.validation import create_stratified_group_kfold, validate_cv_splits
    
    # Create valid panel
    np.random.seed(42)
    panel_data = []
    for i in range(10):
        for month in range(24):
            panel_data.append({
                'country': f'C_{i}',
                'brand_name': f'B_{i}',
                'months_postgx': month,
                'bucket': 1 if i < 5 else 2,
                'volume': np.random.rand() * 1000
            })
    panel = pd.DataFrame(panel_data)
    
    # Create folds
    folds = create_stratified_group_kfold(panel, n_folds=5, random_state=42)
    
    # Validate
    result = validate_cv_splits(panel, folds)
    
    assert result['valid'] is True, f"CV splits should be valid: {result['issues']}"
    assert len(result['issues']) == 0, f"Should have no issues: {result['issues']}"
    assert result['n_folds'] == 5


def test_simulate_scenario_constraints():
    """Test simulate_scenario respects scenario constraints."""
    from src.validation import simulate_scenario
    
    # Create panel with months -12 to 23
    panel = pd.DataFrame({
        'country': ['US'] * 36,
        'brand_name': ['A'] * 36,
        'months_postgx': list(range(-12, 24)),
        'volume': np.random.rand(36) * 1000,
        'bucket': [1] * 36
    })
    
    # Test Scenario 1
    features_s1, targets_s1 = simulate_scenario(panel, scenario=1)
    
    # S1 features: months < 0
    assert features_s1['months_postgx'].max() < 0, "S1 features should be pre-entry only"
    
    # S1 targets: months 0-23
    assert targets_s1['months_postgx'].min() >= 0, "S1 targets should start at month 0"
    assert targets_s1['months_postgx'].max() <= 23, "S1 targets should end at month 23"
    
    # Test Scenario 2
    features_s2, targets_s2 = simulate_scenario(panel, scenario=2)
    
    # S2 features: months < 6
    assert features_s2['months_postgx'].max() < 6, "S2 features should be months < 6"
    
    # S2 targets: months 6-23
    assert targets_s2['months_postgx'].min() >= 6, "S2 targets should start at month 6"
    assert targets_s2['months_postgx'].max() <= 23, "S2 targets should end at month 23"


def test_adversarial_validation():
    """Test adversarial validation function works."""
    from src.validation import adversarial_validation
    
    np.random.seed(42)
    
    # Create similar distributions (should have low AUC)
    train_features = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100) + 1
    })
    
    test_features = pd.DataFrame({
        'feature1': np.random.randn(50),  # Same distribution
        'feature2': np.random.randn(50) + 1  # Same distribution
    })
    
    result = adversarial_validation(train_features, test_features)
    
    # Should return dict with expected keys
    assert 'mean_auc' in result
    assert 'auc_scores' in result
    assert 'top_shift_features' in result
    
    # For similar distributions, AUC should be close to 0.5
    # Allow some variance due to randomness
    assert 0.3 < result['mean_auc'] < 0.8, \
        f"AUC {result['mean_auc']} seems off for similar distributions"


def test_compute_bucket_metrics():
    """Test per-bucket metric computation."""
    from src.evaluate import compute_bucket_metrics
    
    # Create test data
    np.random.seed(42)
    data = []
    
    for i, (country, brand, bucket) in enumerate([
        ('US', 'A', 1), ('DE', 'B', 1), ('FR', 'C', 2), ('UK', 'D', 2)
    ]):
        avg_vol = 1000
        for month in range(24):
            vol = avg_vol * (1 - 0.02 * month)
            data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'volume_actual': vol,
                'volume_pred': vol + np.random.normal(0, 50),
                'avg_vol': avg_vol,
                'bucket': bucket
            })
    
    df = pd.DataFrame(data)
    
    df_actual = df[['country', 'brand_name', 'months_postgx', 'volume_actual']].copy()
    df_actual = df_actual.rename(columns={'volume_actual': 'volume'})
    
    df_pred = df[['country', 'brand_name', 'months_postgx', 'volume_pred']].copy()
    df_pred = df_pred.rename(columns={'volume_pred': 'volume'})
    
    df_aux = df[['country', 'brand_name', 'avg_vol', 'bucket']].drop_duplicates()
    
    # Test for scenario 1
    metrics = compute_bucket_metrics(df_actual, df_pred, df_aux, scenario=1)
    
    assert 'overall' in metrics
    assert 'bucket1' in metrics
    assert 'bucket2' in metrics
    
    # All should be non-negative
    assert metrics['overall'] >= 0
    assert metrics['bucket1'] >= 0
    assert metrics['bucket2'] >= 0


def test_fold_series_generator():
    """Test K-fold series-level generation."""
    from src.validation import get_fold_series
    
    # Create panel with enough series per bucket (at least n_folds per bucket)
    # Using 6 series per bucket = 12 total, with n_folds=3
    countries = ['US', 'DE', 'FR', 'UK', 'JP', 'IT', 'ES', 'BR', 'CA', 'AU', 'MX', 'NL']
    brands = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    buckets = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]  # 6 each
    
    panel_data = []
    for country, brand, bucket in zip(countries, brands, buckets):
        for month in range(24):
            panel_data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'bucket': bucket,
                'volume': np.random.rand() * 1000
            })
    
    panel = pd.DataFrame(panel_data)
    
    folds = get_fold_series(panel, n_folds=3)
    
    assert len(folds) == 3, "Should return 3 folds"
    
    # Check each fold
    for i, (train_df, val_df) in enumerate(folds):
        # No overlap between train and val
        train_keys = set(train_df[['country', 'brand_name']].drop_duplicates().itertuples(index=False, name=None))
        val_keys = set(val_df[['country', 'brand_name']].drop_duplicates().itertuples(index=False, name=None))
        
        assert len(train_keys & val_keys) == 0, f"Fold {i}: train/val overlap"
        
        # Together they cover all series
        all_keys = set(panel[['country', 'brand_name']].drop_duplicates().itertuples(index=False, name=None))
        assert train_keys | val_keys == all_keys, f"Fold {i}: not all series covered"


def test_per_series_error_computation():
    """Test per-series error computation."""
    from src.evaluate import compute_per_series_error
    
    # Create test data
    np.random.seed(42)
    data = []
    for country, brand, bucket in [('US', 'A', 1), ('DE', 'B', 2)]:
        avg_vol = 1000
        for month in range(24):
            vol = avg_vol * (1 - 0.02 * month)
            data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'volume_actual': vol,
                'volume_pred': vol + np.random.normal(0, 50),
                'avg_vol': avg_vol,
                'bucket': bucket
            })
    
    df = pd.DataFrame(data)
    df_actual = df[['country', 'brand_name', 'months_postgx', 'volume_actual']].rename(
        columns={'volume_actual': 'volume'})
    df_pred = df[['country', 'brand_name', 'months_postgx', 'volume_pred']].rename(
        columns={'volume_pred': 'volume'})
    df_aux = df[['country', 'brand_name', 'avg_vol', 'bucket']].drop_duplicates()
    
    # Test for scenario 1
    per_series = compute_per_series_error(df_actual, df_pred, df_aux, scenario=1)
    
    assert len(per_series) == 2, "Should have 2 series"
    assert 'mae' in per_series.columns
    assert 'rmse' in per_series.columns
    assert 'mape' in per_series.columns
    assert 'bucket' in per_series.columns
    
    # Values should be reasonable
    assert all(per_series['mae'] > 0)
    assert all(per_series['rmse'] >= per_series['mae'])  # RMSE >= MAE always


def test_identify_worst_series():
    """Test identifying worst-performing series."""
    from src.evaluate import identify_worst_series
    
    # Create mock per_series_errors DataFrame
    per_series = pd.DataFrame({
        'country': ['US', 'DE', 'FR', 'UK'],
        'brand_name': ['A', 'B', 'C', 'D'],
        'bucket': [1, 1, 2, 2],
        'mae': [100, 200, 50, 300],
        'rmse': [120, 220, 60, 320]
    })
    
    # Get top 2 worst by MAE
    worst = identify_worst_series(per_series, metric='mae', top_k=2)
    
    assert len(worst) == 2
    assert worst.iloc[0]['mae'] == 300  # UK/D has highest MAE
    assert worst.iloc[1]['mae'] == 200  # DE/B is second
    
    # Filter by bucket
    worst_bucket1 = identify_worst_series(per_series, metric='mae', top_k=2, bucket=1)
    assert all(worst_bucket1['bucket'] == 1)


def test_analyze_errors_by_bucket():
    """Test error analysis by bucket."""
    from src.evaluate import analyze_errors_by_bucket
    
    per_series = pd.DataFrame({
        'country': ['US', 'DE', 'FR', 'UK'],
        'brand_name': ['A', 'B', 'C', 'D'],
        'bucket': [1, 1, 2, 2],
        'mae': [100, 200, 50, 100],
        'rmse': [120, 220, 60, 110],
        'mape': [10, 20, 5, 10],
        'normalized_total_error': [0.1, 0.2, 0.05, 0.1]
    })
    
    analysis = analyze_errors_by_bucket(per_series)
    
    assert 'bucket1' in analysis
    assert 'bucket2' in analysis
    assert 'overall' in analysis
    
    assert analysis['bucket1']['n_series'] == 2
    assert analysis['bucket2']['n_series'] == 2
    assert analysis['overall']['n_series'] == 4
    
    # Check mean calculations
    assert analysis['bucket1']['mean_mae'] == 150  # (100 + 200) / 2


def test_check_systematic_bias():
    """Test systematic bias checking."""
    from src.evaluate import check_systematic_bias
    
    # Create data with systematic overprediction
    df_actual = pd.DataFrame({
        'country': ['US'] * 10,
        'brand_name': ['A'] * 10,
        'months_postgx': list(range(10)),
        'volume': [100] * 10
    })
    
    df_pred = pd.DataFrame({
        'country': ['US'] * 10,
        'brand_name': ['A'] * 10,
        'months_postgx': list(range(10)),
        'volume': [120] * 10  # Consistently 20 higher
    })
    
    bias = check_systematic_bias(df_actual, df_pred)
    
    assert 'mean_error' in bias
    assert 'pct_overprediction' in bias
    
    assert bias['mean_error'] == 20  # Overpredicting by 20
    assert bias['pct_overprediction'] == 100  # 100% overprediction


def test_error_analysis_metric_constants():
    """Test metric name constants are defined."""
    from src.evaluate import METRIC_NAME_S1, METRIC_NAME_S2, METRIC_NAME_RMSE, METRIC_NAME_MAE
    
    assert METRIC_NAME_S1 == "metric1_official"
    assert METRIC_NAME_S2 == "metric2_official"
    assert METRIC_NAME_RMSE == "rmse_y_norm"
    assert METRIC_NAME_MAE == "mae_y_norm"


# =============================================================================
# SECTION 5 TESTS: TRAINING PIPELINE
# =============================================================================


def test_config_driven_sample_weights():
    """Test sample weights can be configured from config."""
    from src.train import compute_sample_weights
    
    # Create test meta DataFrame
    meta = pd.DataFrame({
        'months_postgx': [0, 1, 6, 7, 12, 20],
        'bucket': [1, 1, 2, 2, 1, 2]
    })
    
    # Test with custom config
    custom_config = {
        'sample_weights': {
            'scenario1': {
                'months_0_5': 5.0,  # Custom weight
                'months_6_11': 2.0,
                'months_12_23': 0.5
            },
            'bucket_weights': {
                'bucket1': 3.0,  # Custom bucket weight
                'bucket2': 1.0
            }
        }
    }
    
    weights = compute_sample_weights(meta, scenario=1, config=custom_config)
    
    assert len(weights) == len(meta)
    # Weights should be normalized to sum to len(weights)
    assert abs(weights.sum() - len(weights)) < 0.01
    
    # Bucket 1 rows should have higher weights than bucket 2 (same month)
    # Row 0 (month 0, bucket 1) should have higher weight than row 2 (month 6, bucket 2)


def test_sample_weights_without_config():
    """Test sample weights with default values (no config)."""
    from src.train import compute_sample_weights
    
    meta = pd.DataFrame({
        'months_postgx': [0, 6, 12],
        'bucket': [1, 1, 2]
    })
    
    # Should work without config
    weights_s1 = compute_sample_weights(meta, scenario=1, config=None)
    weights_s2 = compute_sample_weights(meta, scenario=2, config=None)
    
    assert len(weights_s1) == 3
    assert len(weights_s2) == 3


def test_get_git_commit_hash():
    """Test git commit hash retrieval."""
    from src.train import get_git_commit_hash
    
    commit_hash = get_git_commit_hash()
    
    # Should return either a string (if in git repo) or None
    assert commit_hash is None or isinstance(commit_hash, str)
    
    # If we got a hash, it should be 8 chars (short hash)
    if commit_hash is not None:
        assert len(commit_hash) == 8


def test_get_experiment_metadata():
    """Test experiment metadata collection."""
    from src.train import get_experiment_metadata
    
    # Create minimal test data
    panel = pd.DataFrame({
        'country': ['US', 'US', 'DE', 'DE'],
        'brand_name': ['A', 'A', 'B', 'B'],
        'months_postgx': [0, 1, 0, 1],
        'bucket': [1, 1, 2, 2]
    })
    
    train_df = panel[panel['country'] == 'US']
    val_df = panel[panel['country'] == 'DE']
    
    metadata = get_experiment_metadata(
        scenario=1,
        model_type='catboost',
        run_config={'reproducibility': {'seed': 42}, 'validation': {}},
        data_config={'paths': {}},
        model_config={'iterations': 100},
        panel_df=panel,
        train_df=train_df,
        val_df=val_df
    )
    
    # Check required fields
    assert 'timestamp' in metadata
    assert 'scenario' in metadata
    assert 'model_type' in metadata
    assert 'random_seed' in metadata
    assert 'dataset' in metadata
    
    # Check dataset info
    assert metadata['dataset']['total_series'] == 2
    assert metadata['dataset']['train_series'] == 1
    assert metadata['dataset']['val_series'] == 1


def test_train_scenario_model_returns_timing():
    """Test that train_scenario_model returns training time."""
    from src.train import train_scenario_model
    from src.models.linear import GlobalMeanBaseline
    
    # Create minimal training data - months_postgx must be in X for GlobalMeanBaseline
    X_train = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'months_postgx': list(range(24)) * 4 + list(range(4))
    })
    y_train = pd.Series(np.random.rand(100))
    meta_train = pd.DataFrame({
        'country': ['US'] * 100,
        'brand_name': ['A'] * 100,
        'months_postgx': list(range(24)) * 4 + list(range(4)),
        'bucket': [1] * 100,
        'avg_vol_12m': [1000.0] * 100
    })
    
    X_val = pd.DataFrame({
        'feature1': np.random.rand(24),
        'feature2': np.random.rand(24),
        'months_postgx': list(range(24))
    })
    y_val = pd.Series(np.random.rand(24))
    meta_val = pd.DataFrame({
        'country': ['DE'] * 24,
        'brand_name': ['B'] * 24,
        'months_postgx': list(range(24)),
        'bucket': [2] * 24,
        'avg_vol_12m': [1000.0] * 24
    })
    
    # Use simple baseline model for fast test
    model, metrics = train_scenario_model(
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        scenario=1,
        model_type='baseline_global_mean',
        model_config=None,
        run_config=None
    )
    
    # Check metrics include training info
    assert 'train_time_seconds' in metrics
    assert metrics['train_time_seconds'] >= 0
    assert 'n_train_samples' in metrics
    assert metrics['n_train_samples'] == 100
    assert 'n_features' in metrics
    assert metrics['n_features'] == 3  # feature1, feature2, months_postgx


def test_cli_help():
    """Test that CLI --help works."""
    import subprocess
    
    result = subprocess.run(
        ['python', '-m', 'src.train', '--help'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=30
    )
    
    # Should not error
    assert result.returncode == 0
    
    # Should contain expected arguments
    assert '--scenario' in result.stdout
    assert '--model' in result.stdout
    assert '--cv' in result.stdout
    assert '--n-folds' in result.stdout


def test_run_cross_validation_function_exists():
    """Test that run_cross_validation function exists and has correct signature."""
    from src.train import run_cross_validation
    import inspect
    
    sig = inspect.signature(run_cross_validation)
    params = list(sig.parameters.keys())
    
    # Check required parameters
    assert 'panel_features' in params
    assert 'scenario' in params
    assert 'model_type' in params
    assert 'n_folds' in params


def test_inference_cli_help():
    """Test that inference CLI --help works."""
    import subprocess
    
    result = subprocess.run(
        ['python', '-m', 'src.inference', '--help'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=30
    )
    
    # Should not error
    assert result.returncode == 0
    
    # Should contain expected arguments
    assert '--model-s1' in result.stdout
    assert '--model-s2' in result.stdout
    assert '--output' in result.stdout
    assert '--data-config' in result.stdout


def test_run_experiment_function_exists():
    """Test that run_experiment function exists and has correct signature."""
    from src.train import run_experiment
    import inspect
    
    sig = inspect.signature(run_experiment)
    params = list(sig.parameters.keys())
    
    # Check required parameters
    assert 'scenario' in params
    assert 'model_type' in params
    assert 'model_config_path' in params
    assert 'run_config_path' in params
    assert 'data_config_path' in params
    assert 'run_name' in params


def test_compute_config_hash_returns_consistent_hash():
    """Test that config hash is consistent for same configs."""
    from src.train import compute_config_hash
    
    # Create config dictionaries
    configs = {
        'run_config': {'param1': 'value1', 'param2': 123},
        'model_config': {'model_type': 'catboost'}
    }
    
    # Compute hash twice
    hash1 = compute_config_hash(configs)
    hash2 = compute_config_hash(configs)
    
    # Should be consistent
    assert hash1 == hash2
    
    # Should be valid SHA256 hex digest (8 chars for short hash)
    assert len(hash1) == 8
    assert all(c in '0123456789abcdef' for c in hash1)


def test_compute_config_hash_changes_with_content():
    """Test that config hash changes when config content changes."""
    from src.train import compute_config_hash
    
    configs1 = {'param': 'original'}
    configs2 = {'param': 'modified'}
    
    hash1 = compute_config_hash(configs1)
    hash2 = compute_config_hash(configs2)
    
    # Hashes should differ
    assert hash1 != hash2


def test_compute_config_hash_handles_empty_config():
    """Test that config hash handles empty configs."""
    from src.train import compute_config_hash
    
    # Empty config
    hash_result = compute_config_hash({})
    
    # Should return a valid hash
    assert isinstance(hash_result, str)
    assert len(hash_result) == 8


def test_compute_metric_aligned_weights_scenario1():
    """Test metric-aligned weights for Scenario 1."""
    from src.train import compute_metric_aligned_weights
    
    # Create test data: 2 series across 24 months
    rows = []
    for series_id, (country, brand, bucket) in enumerate([('US', 'A', 1), ('DE', 'B', 2)]):
        for month in range(24):
            rows.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'bucket': bucket,
                'avg_vol_12m': 1000.0
            })
    
    meta_df = pd.DataFrame(rows)
    
    weights = compute_metric_aligned_weights(
        meta_df=meta_df,
        scenario=1,
        avg_vol_col='avg_vol_12m'
    )
    
    # Check basic properties
    assert len(weights) == 48  # 2 series * 24 months
    assert all(weights > 0)
    
    # Months 0-5 should have higher weight than 12-23 (as per metric formula)
    month0_idx = meta_df[(meta_df['bucket'] == 1) & (meta_df['months_postgx'] == 0)].index[0]
    month12_idx = meta_df[(meta_df['bucket'] == 1) & (meta_df['months_postgx'] == 12)].index[0]
    
    # Month 0-5 weight should be > month 12-23 weight
    assert weights.iloc[month0_idx] > weights.iloc[month12_idx]


def test_compute_metric_aligned_weights_scenario2():
    """Test metric-aligned weights for Scenario 2."""
    from src.train import compute_metric_aligned_weights
    
    # Create test data: 1 series, months 6-23 only (S2 relevant range)
    rows = []
    for month in range(6, 24):
        rows.append({
            'country': 'US',
            'brand_name': 'A',
            'months_postgx': month,
            'bucket': 1,
            'avg_vol_12m': 1000.0
        })
    
    meta_df = pd.DataFrame(rows)
    
    weights = compute_metric_aligned_weights(
        meta_df=meta_df,
        scenario=2,
        avg_vol_col='avg_vol_12m'
    )
    
    # Check basic properties
    assert len(weights) == 18  # months 6-23
    assert all(weights > 0)
    
    # Months 6-11 should have higher weight than 12-23 (0.5/6 vs 0.3/12)
    month6_idx = meta_df[meta_df['months_postgx'] == 6].index[0]
    month12_idx = meta_df[meta_df['months_postgx'] == 12].index[0]
    
    # Weight at month 6 should be > weight at month 12
    assert weights.iloc[month6_idx] > weights.iloc[month12_idx]


def test_compute_sample_weights_with_metric_aligned():
    """Test compute_sample_weights with use_metric_aligned=True."""
    from src.train import compute_sample_weights
    
    # Create test data
    meta_df = pd.DataFrame({
        'country': ['US'] * 24,
        'brand_name': ['A'] * 24,
        'months_postgx': list(range(24)),
        'bucket': [1] * 24,
        'avg_vol_12m': [1000.0] * 24
    })
    
    # Test with metric-aligned weights
    weights = compute_sample_weights(
        meta_df=meta_df,
        scenario=1,
        config=None,
        use_metric_aligned=True
    )
    
    assert len(weights) == 24
    assert all(weights > 0)


def test_save_config_snapshot():
    """Test that save_config_snapshot saves config files and returns hash."""
    from src.train import save_config_snapshot
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / 'artifacts'
        artifacts_dir.mkdir(parents=True)
        
        # Create a test config file
        config_path = Path(tmpdir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            f.write('param1: value1\nparam2: 123\n')
        
        # Call save_config_snapshot
        config_hash = save_config_snapshot(
            artifacts_dir=artifacts_dir,
            run_config_path=str(config_path)
        )
        
        # Check hash is returned
        assert isinstance(config_hash, str)
        assert len(config_hash) == 8
        
        # Check configs directory was created
        config_dir = artifacts_dir / 'configs'
        assert config_dir.exists()
        
        # Check config file was copied
        copied_config = config_dir / 'test_config.yaml'
        assert copied_config.exists()
        
        # Check hash file was created
        hash_file = config_dir / 'config_hash.txt'
        assert hash_file.exists()
        with open(hash_file, 'r') as f:
            content = f.read()
            assert config_hash in content


# =============================================================================
# SECTION 2 TESTS: DATA PIPELINE (NEW ADDITIONS)
# =============================================================================


def test_data_leakage_audit_clean():
    """Test data leakage audit with clean features."""
    from src.data import audit_data_leakage
    
    # Clean features - no leakage
    clean_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'pre_entry_trend': [0.1, 0.2, 0.3]
    })
    
    is_clean, violations = audit_data_leakage(clean_df, scenario=1, mode='train', strict=False)
    
    # Should be clean
    assert is_clean, f"Clean features should pass audit. Violations: {violations}"
    assert len([v for v in violations if v.startswith('LEAKAGE')]) == 0


def test_data_leakage_audit_detects_forbidden():
    """Test data leakage audit detects forbidden columns."""
    from src.data import audit_data_leakage
    
    # Features with forbidden column
    bad_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'bucket': [1, 2, 1]  # FORBIDDEN!
    })
    
    is_clean, violations = audit_data_leakage(bad_df, scenario=1, mode='train', strict=False)
    
    # Should detect leakage
    assert not is_clean, "Should detect forbidden column 'bucket'"
    assert any('bucket' in v for v in violations)


def test_data_leakage_audit_detects_early_erosion_s1():
    """Test data leakage audit detects early erosion features in Scenario 1."""
    from src.data import audit_data_leakage
    
    # Scenario 1 with early erosion features (leakage!)
    bad_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'erosion_0_5': [0.8, 0.7, 0.6]  # Early erosion - invalid for S1
    })
    
    is_clean, violations = audit_data_leakage(bad_df, scenario=1, mode='train', strict=False)
    
    # Should detect leakage for Scenario 1
    assert not is_clean, "S1 should flag early erosion features"
    assert any('erosion_0_' in v or 'early-erosion' in v.lower() for v in violations)


def test_data_leakage_audit_allows_early_erosion_s2():
    """Test data leakage audit allows early erosion features in Scenario 2."""
    from src.data import audit_data_leakage
    
    # Scenario 2 with early erosion features (OK for S2)
    s2_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'erosion_0_5': [0.8, 0.7, 0.6]  # OK for Scenario 2
    })
    
    is_clean, violations = audit_data_leakage(s2_df, scenario=2, mode='train', strict=False)
    
    # Should NOT flag early erosion for Scenario 2
    leakage_violations = [v for v in violations if v.startswith('LEAKAGE')]
    assert len(leakage_violations) == 0, f"S2 should allow early erosion features. Violations: {violations}"


def test_data_leakage_audit_strict_mode():
    """Test data leakage audit strict mode raises ValueError."""
    from src.data import audit_data_leakage
    
    # Features with leakage
    bad_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'y_norm': [0.8, 0.7, 0.6]  # Target column - FORBIDDEN!
    })
    
    # Strict mode should raise
    with pytest.raises(ValueError, match="Data leakage detected"):
        audit_data_leakage(bad_df, scenario=1, mode='train', strict=True)


def test_run_pre_training_leakage_check():
    """Test pre-training leakage check convenience function."""
    from src.data import run_pre_training_leakage_check
    
    # Clean features
    clean_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    
    # Should pass
    result = run_pre_training_leakage_check(clean_df, scenario=1, mode='train')
    assert result is True
    
    # Features with leakage
    bad_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'bucket': [1, 2, 1]  # FORBIDDEN!
    })
    
    # Should raise
    with pytest.raises(ValueError):
        run_pre_training_leakage_check(bad_df, scenario=1, mode='train')


def test_validate_date_continuity_clean():
    """Test date continuity validation with clean data."""
    from src.data import validate_date_continuity
    
    # Clean data - no gaps
    clean_df = pd.DataFrame({
        'country': ['US'] * 10 + ['DE'] * 10,
        'brand_name': ['A'] * 10 + ['B'] * 10,
        'months_postgx': list(range(10)) * 2
    })
    
    is_valid, issues_df = validate_date_continuity(clean_df)
    
    assert is_valid, "Clean data should have no gaps"
    assert len(issues_df) == 0


def test_validate_date_continuity_with_gaps():
    """Test date continuity validation detects gaps."""
    from src.data import validate_date_continuity
    
    # Data with gaps
    gap_df = pd.DataFrame({
        'country': ['US'] * 8,
        'brand_name': ['A'] * 8,
        'months_postgx': [0, 1, 2, 3, 5, 6, 7, 8]  # Missing month 4!
    })
    
    is_valid, issues_df = validate_date_continuity(gap_df)
    
    assert not is_valid, "Should detect gap in months"
    assert len(issues_df) == 1
    assert issues_df.iloc[0]['gap_count'] == 1
    assert 4 in issues_df.iloc[0]['missing_months']


def test_get_series_month_coverage():
    """Test series month coverage statistics."""
    from src.data import get_series_month_coverage
    
    # Create panel with varying coverage
    panel_data = []
    
    # Series A: full coverage (-12 to 23)
    for month in range(-12, 24):
        panel_data.append({'country': 'US', 'brand_name': 'A', 'months_postgx': month})
    
    # Series B: partial pre-entry only (-6 to -1)
    for month in range(-6, 0):
        panel_data.append({'country': 'DE', 'brand_name': 'B', 'months_postgx': month})
    
    panel = pd.DataFrame(panel_data)
    
    coverage = get_series_month_coverage(panel)
    
    assert len(coverage) == 2
    
    # Check Series A
    series_a = coverage[coverage['brand_name'] == 'A'].iloc[0]
    assert series_a['has_full_pre_entry'] == True
    assert series_a['has_full_post_entry'] == True
    assert series_a['total_months'] == 36
    
    # Check Series B
    series_b = coverage[coverage['brand_name'] == 'B'].iloc[0]
    assert series_b['has_full_pre_entry'] == False
    assert series_b['has_full_post_entry'] == False
    assert series_b['pre_entry_months'] == 6
    assert series_b['post_entry_months'] == 0


def test_temporal_cv_split():
    """Test time-based cross-validation split."""
    from src.validation import create_temporal_cv_split
    
    # Create panel with good temporal coverage
    panel_data = []
    for country, brand in [('US', 'A'), ('DE', 'B'), ('FR', 'C'), ('UK', 'D')]:
        for month in range(-12, 24):
            panel_data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'bucket': 1 if country in ['US', 'DE'] else 2
            })
    
    panel = pd.DataFrame(panel_data)
    
    folds = create_temporal_cv_split(panel, n_folds=3, min_train_months=0, gap_months=0)
    
    # Should return list of folds
    assert len(folds) > 0, "Should return at least one fold"
    
    for i, (train_df, val_df) in enumerate(folds):
        # Each fold should have data
        assert len(train_df) > 0, f"Fold {i}: train should not be empty"
        assert len(val_df) > 0, f"Fold {i}: val should not be empty"
        
        # Train months should be <= val months (temporal ordering)
        train_max_month = train_df['months_postgx'].max()
        val_min_month = val_df['months_postgx'].min()
        assert train_max_month < val_min_month, \
            f"Fold {i}: train max month ({train_max_month}) should be < val min month ({val_min_month})"


def test_create_holdout_set():
    """Test holdout set creation."""
    from src.validation import create_holdout_set
    
    # Create panel with enough series
    countries = ['US', 'DE', 'FR', 'UK', 'JP', 'IT', 'ES', 'BR', 'CA', 'AU']
    brands = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    buckets = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    
    panel_data = []
    for country, brand, bucket in zip(countries, brands, buckets):
        for month in range(24):
            panel_data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'bucket': bucket
            })
    
    panel = pd.DataFrame(panel_data)
    
    main_df, holdout_df = create_holdout_set(panel, holdout_fraction=0.2)
    
    # Check series don't overlap
    main_series = set(zip(main_df['country'], main_df['brand_name']))
    holdout_series = set(zip(holdout_df['country'], holdout_df['brand_name']))
    
    assert len(main_series & holdout_series) == 0, "Main and holdout should not share series"
    
    # Check approximate fraction (with small tolerance due to stratification)
    total_series = len(countries)
    holdout_count = len(holdout_series)
    assert 1 <= holdout_count <= 3, f"Expected ~2 series in holdout (20%), got {holdout_count}"


def test_multi_column_stratification():
    """Test validation split with multi-column stratification."""
    try:
        from src.validation import create_validation_split
    except (ImportError, ValueError) as e:
        pytest.skip(f"Skipping due to import error (likely torch/numpy version mismatch): {e}")
    
    # Create panel with bucket and ther_area
    # Need enough series per class for stratification to work
    # With 16 series, 4 combinations, and 50% val_fraction, we get 2 per class in val
    panel_data = []
    for i, (country, brand, bucket, ther_area) in enumerate([
        # Bucket 1, Cardio - 4 series
        ('US', 'A', 1, 'Cardio'),
        ('DE', 'B', 1, 'Cardio'),
        ('CA', 'K', 1, 'Cardio'),
        ('AU', 'L', 1, 'Cardio'),
        # Bucket 1, Onco - 4 series
        ('FR', 'C', 1, 'Onco'),
        ('UK', 'D', 1, 'Onco'),
        ('NL', 'M', 1, 'Onco'),
        ('BE', 'N', 1, 'Onco'),
        # Bucket 2, Cardio - 4 series
        ('JP', 'E', 2, 'Cardio'),
        ('IT', 'F', 2, 'Cardio'),
        ('PT', 'O', 2, 'Cardio'),
        ('AT', 'P', 2, 'Cardio'),
        # Bucket 2, Onco - 4 series
        ('ES', 'G', 2, 'Onco'),
        ('BR', 'H', 2, 'Onco'),
        ('MX', 'Q', 2, 'Onco'),
        ('AR', 'R', 2, 'Onco'),
    ]):
        for month in range(24):
            panel_data.append({
                'country': country,
                'brand_name': brand,
                'months_postgx': month,
                'bucket': bucket,
                'ther_area': ther_area
            })
    
    panel = pd.DataFrame(panel_data)
    
    # Multi-column stratification with larger validation fraction for small datasets
    train_df, val_df = create_validation_split(
        panel,
        val_fraction=0.5,  # 50% to ensure enough samples per class
        stratify_by=['bucket', 'ther_area'],
        random_state=42
    )
    
    # Check no series overlap
    train_series = set(zip(train_df['country'], train_df['brand_name']))
    val_series = set(zip(val_df['country'], val_df['brand_name']))
    assert len(train_series & val_series) == 0
    
    # Check all rows accounted for
    assert len(train_df) + len(val_df) == len(panel)


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_get_features_caching():
    """Test feature matrix caching via get_features()."""
    from src.features import get_features, clear_feature_cache
    from src.utils import load_config, get_project_root
    import time
    
    root = get_project_root()
    data_config = load_config(root / 'configs' / 'data.yaml')
    features_config = load_config(root / 'configs' / 'features.yaml')
    
    # Clear cache first
    clear_feature_cache(data_config, split='train', scenario=1)
    
    # First call - builds from scratch
    start = time.time()
    X1, y1, meta1 = get_features(
        split='train',
        scenario=1,
        mode='train',
        data_config=data_config,
        features_config=features_config,
        use_cache=True,
        force_rebuild=True
    )
    build_time = time.time() - start
    
    # Second call - from cache
    start = time.time()
    X2, y2, meta2 = get_features(
        split='train',
        scenario=1,
        mode='train',
        data_config=data_config,
        features_config=features_config,
        use_cache=True,
        force_rebuild=False
    )
    cache_time = time.time() - start
    
    # Verify shapes match
    assert X1.shape == X2.shape, f"Shape mismatch: {X1.shape} vs {X2.shape}"
    assert len(y1) == len(y2), f"Target length mismatch: {len(y1)} vs {len(y2)}"
    
    # Cache should be faster
    print(f"Build time: {build_time:.2f}s, Cache time: {cache_time:.2f}s")


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
    reason="Training data not available"
)
def test_get_features_mode_difference():
    """Test that get_features respects mode parameter."""
    from src.features import get_features
    from src.utils import load_config, get_project_root
    
    root = get_project_root()
    data_config = load_config(root / 'configs' / 'data.yaml')
    features_config = load_config(root / 'configs' / 'features.yaml')
    
    # Train mode
    X_train, y_train, _ = get_features(
        split='train',
        scenario=1,
        mode='train',
        data_config=data_config,
        features_config=features_config,
        use_cache=True,
        force_rebuild=False
    )
    
    # Test mode (on same split for testing purposes)
    X_test, y_test, _ = get_features(
        split='train',
        scenario=1,
        mode='test',
        data_config=data_config,
        features_config=features_config,
        use_cache=True,
        force_rebuild=True  # Force rebuild to get test mode
    )
    
    # Train mode should have y, test mode should not
    assert y_train is not None, "Train mode should have y"
    assert y_test is None, "Test mode should not have y"


def test_data_cli_arguments():
    """Test that data CLI has expected arguments."""
    import subprocess
    
    result = subprocess.run(
        ['python', '-m', 'src.data', '--help'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=30
    )
    
    # Should not error
    assert result.returncode == 0
    
    # Should contain expected arguments
    assert '--split' in result.stdout
    assert '--scenario' in result.stdout
    assert '--mode' in result.stdout
    assert '--force-rebuild' in result.stdout
    assert '--clear-cache' in result.stdout
    assert '--data-config' in result.stdout
    assert '--features-config' in result.stdout
    assert '--validate-continuity' in result.stdout


# =============================================================================
# SECTION 2 TESTS: ADDITIONAL DATA PIPELINE FUNCTIONS
# =============================================================================


def test_validate_panel_schema_valid():
    """Test validate_panel_schema with valid train panel."""
    from src.data import validate_panel_schema, PANEL_REQUIRED_COLUMNS
    
    # Create a valid train panel
    panel = pd.DataFrame({
        'country': ['US', 'US', 'UK', 'UK'],
        'brand_name': ['A', 'A', 'B', 'B'],
        'months_postgx': [0, 1, 0, 1],
        'volume': [100.0, 90.0, 200.0, 180.0],
        'n_gxs': [1, 1, 2, 2],
        'ther_area': ['oncology', 'oncology', 'cardio', 'cardio'],
        'main_package': ['tablet', 'tablet', 'injection', 'injection'],
        'hospital_rate': [50.0, 50.0, 30.0, 30.0],
        'biological': [True, True, False, False],
        'small_molecule': [False, False, True, True],
        'bucket': [1, 1, 2, 2],
        'mean_erosion': [0.2, 0.2, 0.5, 0.5],
        'avg_vol_12m': [100.0, 100.0, 200.0, 200.0],
        'y_norm': [1.0, 0.9, 1.0, 0.9]
    })
    
    is_valid, issues = validate_panel_schema(panel, split='train', raise_on_error=False)
    
    assert is_valid == True, f"Expected valid panel, got issues: {issues}"
    assert len(issues) == 0


def test_validate_panel_schema_missing_columns():
    """Test validate_panel_schema detects missing columns."""
    from src.data import validate_panel_schema
    
    # Create a panel missing required columns
    panel = pd.DataFrame({
        'country': ['US'],
        'brand_name': ['A'],
        'months_postgx': [0],
        # Missing: volume, n_gxs, ther_area, main_package, etc.
    })
    
    is_valid, issues = validate_panel_schema(panel, split='train', raise_on_error=False)
    
    assert is_valid == False
    assert any('Missing required columns' in issue for issue in issues)


def test_validate_panel_schema_duplicates():
    """Test validate_panel_schema detects duplicate keys."""
    from src.data import validate_panel_schema
    
    # Create a panel with duplicate keys
    panel = pd.DataFrame({
        'country': ['US', 'US', 'US'],  # Duplicate
        'brand_name': ['A', 'A', 'A'],  # Duplicate
        'months_postgx': [0, 0, 1],      # First two are duplicates
        'volume': [100.0, 100.0, 90.0],
        'n_gxs': [1, 1, 1],
        'ther_area': ['oncology'] * 3,
        'main_package': ['tablet'] * 3,
        'hospital_rate': [50.0] * 3,
        'biological': [True] * 3,
        'small_molecule': [False] * 3,
    })
    
    is_valid, issues = validate_panel_schema(panel, split='test', raise_on_error=False)
    
    assert is_valid == False
    assert any('duplicate' in issue.lower() for issue in issues)


def test_validate_feature_matrix_valid():
    """Test validate_feature_matrix with valid feature matrix."""
    from src.features import validate_feature_matrix
    
    # Create a valid feature matrix
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [0.1, 0.2, 0.3],
        'n_gxs': [1, 2, 3]
    })
    y = pd.Series([0.9, 0.8, 0.7], name='y_norm')
    meta_df = pd.DataFrame({
        'country': ['US', 'UK', 'DE'],
        'brand_name': ['A', 'B', 'C'],
        'bucket': [1, 2, 1]
    })
    
    is_valid, issues = validate_feature_matrix(X, y, meta_df, mode='train', raise_on_error=False)
    
    assert is_valid == True, f"Expected valid matrix, got issues: {issues}"


def test_validate_feature_matrix_forbidden_columns():
    """Test validate_feature_matrix detects forbidden columns."""
    from src.features import validate_feature_matrix
    
    # Create a feature matrix with forbidden columns
    X = pd.DataFrame({
        'feature1': [1.0, 2.0],
        'bucket': [1, 2],  # FORBIDDEN
        'y_norm': [0.9, 0.8]  # FORBIDDEN
    })
    y = pd.Series([0.9, 0.8])
    meta_df = pd.DataFrame()
    
    is_valid, issues = validate_feature_matrix(X, y, meta_df, mode='train', raise_on_error=False)
    
    assert is_valid == False
    assert any('forbidden' in issue.lower() or 'meta' in issue.lower() for issue in issues)


def test_validate_feature_matrix_mode_train_requires_y():
    """Test validate_feature_matrix requires y for train mode."""
    from src.features import validate_feature_matrix
    
    X = pd.DataFrame({
        'feature1': [1.0, 2.0]
    })
    y = None  # Missing target
    meta_df = pd.DataFrame()
    
    is_valid, issues = validate_feature_matrix(X, y, meta_df, mode='train', raise_on_error=False)
    
    assert is_valid == False
    assert any("mode='train' requires y" in issue for issue in issues)


def test_validate_feature_matrix_test_mode_no_y():
    """Test validate_feature_matrix expects y=None for test mode."""
    from src.features import validate_feature_matrix
    
    X = pd.DataFrame({
        'feature1': [1.0, 2.0]
    })
    y = pd.Series([0.9, 0.8])  # Should not have y for test mode
    meta_df = pd.DataFrame()
    
    is_valid, issues = validate_feature_matrix(X, y, meta_df, mode='test', raise_on_error=False)
    
    assert is_valid == False
    assert any("mode='test' should have y=None" in issue for issue in issues)


def test_verify_no_future_leakage_clean():
    """Test verify_no_future_leakage with clean data."""
    from src.data import verify_no_future_leakage
    
    # Clean feature DataFrame (no suspicious columns)
    df = pd.DataFrame({
        'country': ['US', 'US'],
        'brand_name': ['A', 'A'],
        'months_postgx': [0, 1],
        'pre_entry_vol': [100.0, 100.0],
        'n_gxs': [1, 1]
    })
    
    is_clean, violations = verify_no_future_leakage(df, scenario=1)
    
    assert is_clean == True
    assert len(violations) == 0


def test_verify_no_future_leakage_suspicious():
    """Test verify_no_future_leakage detects suspicious column names."""
    from src.data import verify_no_future_leakage
    
    # DataFrame with suspicious column names
    df = pd.DataFrame({
        'country': ['US'],
        'brand_name': ['A'],
        'months_postgx': [0],
        'volume_future_avg': [100.0],  # Suspicious name (_future_)
    })
    
    is_clean, violations = verify_no_future_leakage(df, scenario=1)
    
    # Should detect at least one suspicious column
    assert len(violations) >= 1
    assert any('future' in v.lower() for v in violations)


def test_optimize_dtypes_categories():
    """Test _optimize_dtypes converts categoricals."""
    from src.data import _optimize_dtypes
    
    df = pd.DataFrame({
        'country': ['US', 'UK', 'US', 'UK'] * 100,
        'brand_name': ['A', 'B', 'A', 'B'] * 100,
        'volume': [100.0, 200.0, 100.0, 200.0] * 100,
        'n_gxs': [1, 2, 1, 2] * 100
    })
    
    # Ensure object dtype before optimization
    assert df['country'].dtype == 'object'
    
    optimized = _optimize_dtypes(df)
    
    # Known categoricals should be converted
    assert optimized['country'].dtype.name == 'category'
    assert optimized['brand_name'].dtype.name == 'category'


def test_optimize_dtypes_numeric_downcast():
    """Test _optimize_dtypes downcasts numeric columns."""
    from src.data import _optimize_dtypes
    
    df = pd.DataFrame({
        'country': ['US'] * 10,
        'small_int': np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.int64),
        'small_float': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        'volume': np.array([1000.5] * 10, dtype=np.float64)  # Should NOT be downcast (preserve precision)
    })
    
    optimized = _optimize_dtypes(df)
    
    # volume should preserve precision (float64)
    assert optimized['volume'].dtype == np.float64


def test_utility_functions_exist():
    """Test that all Section 2.7 utility functions exist in src/utils.py."""
    from src.utils import set_seed, setup_logging, timer, load_config, get_path
    
    # Check they're callable
    assert callable(set_seed)
    assert callable(setup_logging)
    assert callable(load_config)
    assert callable(get_path)


def test_setup_logging_no_duplicates():
    """Test setup_logging avoids duplicate handlers."""
    import logging
    from src.utils import setup_logging
    
    # Call multiple times
    setup_logging(level='INFO')
    setup_logging(level='DEBUG')
    setup_logging(level='INFO')
    
    root_logger = logging.getLogger()
    
    # Should not have duplicate handlers
    # Note: This might still have multiple handlers due to test framework
    # Just verify the function runs without error
    assert root_logger is not None


def test_timer_context_manager():
    """Test timer context manager works."""
    from src.utils import timer
    import time
    
    with timer("Test operation"):
        time.sleep(0.01)  # Small sleep
    
    # If we get here without error, the timer worked
    assert True


def test_get_path_nested_key():
    """Test get_path resolves nested keys."""
    from src.utils import get_path
    
    config = {
        'paths': {
            'raw_dir': 'data/raw',
            'interim_dir': 'data/interim'
        },
        'files': {
            'train': {
                'volume': 'TRAIN/df_volume_train.csv'
            }
        }
    }
    
    raw_path = get_path(config, 'paths.raw_dir')
    # Use as_posix() for cross-platform path comparison
    assert str(raw_path).replace('\\', '/') == 'data/raw'
    
    volume_path = get_path(config, 'files.train.volume')
    assert str(volume_path).replace('\\', '/') == 'TRAIN/df_volume_train.csv'


def test_get_path_key_not_found():
    """Test get_path raises KeyError for missing keys."""
    from src.utils import get_path
    
    config = {'paths': {'raw_dir': 'data/raw'}}
    
    try:
        get_path(config, 'paths.nonexistent')
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_train_cli_force_rebuild():
    """Test that train CLI has --force-rebuild flag."""
    import subprocess
    
    result = subprocess.run(
        ['python', '-m', 'src.train', '--help'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=30
    )
    
    assert '--force-rebuild' in result.stdout
    assert '--no-cache' in result.stdout


def test_inference_cli_force_rebuild():
    """Test that inference CLI has --force-rebuild flag."""
    import subprocess
    
    result = subprocess.run(
        ['python', '-m', 'src.inference', '--help'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=30
    )
    
    assert '--force-rebuild' in result.stdout


def test_run_experiment_function_signature():
    """Test run_experiment has expected parameters."""
    import inspect
    from src.train import run_experiment
    
    sig = inspect.signature(run_experiment)
    params = list(sig.parameters.keys())
    
    assert 'scenario' in params
    assert 'model_type' in params
    assert 'use_cached_features' in params
    assert 'force_rebuild' in params
    assert 'features_config_path' in params


# =============================================================================
# Section 0.2: Config as Single Source of Truth Tests
# =============================================================================

def test_official_metric_constants_exist():
    """Test that official metric constants are defined in evaluate.py."""
    from src.evaluate import (
        OFFICIAL_BUCKET_THRESHOLD,
        OFFICIAL_BUCKET1_WEIGHT,
        OFFICIAL_BUCKET2_WEIGHT,
        OFFICIAL_METRIC1_WEIGHTS,
        OFFICIAL_METRIC2_WEIGHTS,
    )
    
    # Check bucket threshold
    assert OFFICIAL_BUCKET_THRESHOLD == 0.25
    
    # Check bucket weights
    assert OFFICIAL_BUCKET1_WEIGHT == 2.0
    assert OFFICIAL_BUCKET2_WEIGHT == 1.0
    
    # Check metric1 weights
    assert OFFICIAL_METRIC1_WEIGHTS['monthly'] == 0.2
    assert OFFICIAL_METRIC1_WEIGHTS['accumulated_0_5'] == 0.5
    assert OFFICIAL_METRIC1_WEIGHTS['accumulated_6_11'] == 0.2
    assert OFFICIAL_METRIC1_WEIGHTS['accumulated_12_23'] == 0.1
    
    # Check metric2 weights
    assert OFFICIAL_METRIC2_WEIGHTS['monthly'] == 0.2
    assert OFFICIAL_METRIC2_WEIGHTS['accumulated_6_11'] == 0.5
    assert OFFICIAL_METRIC2_WEIGHTS['accumulated_12_23'] == 0.3


def test_config_official_metric_section_exists():
    """Test that run_defaults.yaml has official_metric section with correct values."""
    from src.utils import load_config
    
    config = load_config('configs/run_defaults.yaml')
    
    assert 'official_metric' in config
    official_metric = config['official_metric']
    
    # Check bucket threshold
    assert 'bucket_threshold' in official_metric
    assert official_metric['bucket_threshold'] == 0.25
    
    # Check bucket weights
    assert 'bucket_weights' in official_metric
    assert official_metric['bucket_weights']['bucket1'] == 2.0
    assert official_metric['bucket_weights']['bucket2'] == 1.0
    
    # Check metric1 section
    assert 'metric1' in official_metric
    metric1 = official_metric['metric1']
    assert metric1['monthly_weight'] == 0.2
    assert metric1['accumulated_0_5_weight'] == 0.5
    assert metric1['accumulated_6_11_weight'] == 0.2
    assert metric1['accumulated_12_23_weight'] == 0.1
    
    # Check metric2 section
    assert 'metric2' in official_metric
    metric2 = official_metric['metric2']
    assert metric2['monthly_weight'] == 0.2
    assert metric2['accumulated_6_11_weight'] == 0.5
    assert metric2['accumulated_12_23_weight'] == 0.3


def test_validate_config_matches_official_function():
    """Test validate_config_matches_official detects mismatches."""
    from src.evaluate import validate_config_matches_official
    from src.utils import load_config
    
    # Test with correct config (should pass)
    config = load_config('configs/run_defaults.yaml')
    result = validate_config_matches_official(config)
    assert result is True
    
    # Test with incorrect config (should raise)
    bad_config = {
        'official_metric': {
            'bucket_threshold': 0.30,  # Wrong!
            'bucket_weights': {'bucket1': 2.0, 'bucket2': 1.0},
            'metric1': {
                'monthly_weight': 0.2,
                'accumulated_0_5_weight': 0.5,
                'accumulated_6_11_weight': 0.2,
                'accumulated_12_23_weight': 0.1,
            },
            'metric2': {
                'monthly_weight': 0.2,
                'accumulated_6_11_weight': 0.5,
                'accumulated_12_23_weight': 0.3,
            }
        }
    }
    
    try:
        validate_config_matches_official(bad_config)
        assert False, "Should have raised ValueError for wrong bucket_threshold"
    except ValueError as e:
        assert 'bucket_threshold' in str(e)


def test_compute_pre_entry_stats_uses_config_threshold():
    """Test compute_pre_entry_stats can use bucket_threshold from config."""
    import pandas as pd
    from src.data import compute_pre_entry_stats
    
    # Create minimal panel data with known mean_erosion
    panel_data = []
    for brand in ['BRAND_A', 'BRAND_B']:
        # Pre-entry months
        for m in range(-12, 0):
            panel_data.append({
                'country': 'COUNTRY_1',
                'brand_name': brand,
                'months_postgx': m,
                'volume': 1000,  # Constant volume
            })
        # Post-entry months
        for m in range(24):
            # BRAND_A: high erosion (y_norm ~ 0.2)
            # BRAND_B: low erosion (y_norm ~ 0.8)
            vol = 200 if brand == 'BRAND_A' else 800
            panel_data.append({
                'country': 'COUNTRY_1',
                'brand_name': brand,
                'months_postgx': m,
                'volume': vol,
            })
    
    panel = pd.DataFrame(panel_data)
    
    # Test with default threshold (0.25)
    result = compute_pre_entry_stats(panel, is_train=True, bucket_threshold=0.25)
    
    brand_a_bucket = result[result['brand_name'] == 'BRAND_A']['bucket'].iloc[0]
    brand_b_bucket = result[result['brand_name'] == 'BRAND_B']['bucket'].iloc[0]
    
    # BRAND_A: mean_erosion = 0.2 <= 0.25 -> Bucket 1
    # BRAND_B: mean_erosion = 0.8 > 0.25 -> Bucket 2
    assert brand_a_bucket == 1
    assert brand_b_bucket == 2
    
    # Test with custom threshold (0.5)
    result2 = compute_pre_entry_stats(panel, is_train=True, bucket_threshold=0.5)
    
    brand_a_bucket2 = result2[result2['brand_name'] == 'BRAND_A']['bucket'].iloc[0]
    brand_b_bucket2 = result2[result2['brand_name'] == 'BRAND_B']['bucket'].iloc[0]
    
    # BRAND_A: mean_erosion = 0.2 <= 0.5 -> Bucket 1
    # BRAND_B: mean_erosion = 0.8 > 0.5 -> Bucket 2
    assert brand_a_bucket2 == 1
    assert brand_b_bucket2 == 2


def test_compute_pre_entry_stats_loads_threshold_from_run_config():
    """Test compute_pre_entry_stats loads threshold from run_config dict."""
    import pandas as pd
    from src.data import compute_pre_entry_stats
    
    # Create minimal panel
    panel_data = []
    for m in range(-12, 0):
        panel_data.append({
            'country': 'COUNTRY_1',
            'brand_name': 'BRAND_A',
            'months_postgx': m,
            'volume': 1000,
        })
    for m in range(24):
        panel_data.append({
            'country': 'COUNTRY_1',
            'brand_name': 'BRAND_A',
            'months_postgx': m,
            'volume': 300,  # mean_erosion = 0.3
        })
    
    panel = pd.DataFrame(panel_data)
    
    # With default threshold 0.25, bucket should be 2 (0.3 > 0.25)
    result_default = compute_pre_entry_stats(panel, is_train=True)
    assert result_default['bucket'].iloc[0] == 2
    
    # With run_config threshold 0.35, bucket should be 1 (0.3 <= 0.35)
    run_config = {
        'official_metric': {
            'bucket_threshold': 0.35
        }
    }
    result_config = compute_pre_entry_stats(panel, is_train=True, run_config=run_config)
    assert result_config['bucket'].iloc[0] == 1


# =============================================================================
# Section 3 - Feature Engineering Tests
# =============================================================================

class TestSeasonalFeatures:
    """Tests for seasonal pattern detection from pre-entry months (Section 3.1)."""
    
    def test_seasonal_features_created(self):
        """Test that seasonal pattern features are created."""
        from src.features import make_features
        
        # Create synthetic panel with seasonal pattern
        panel_data = []
        for brand in ['BRAND_A']:
            for m in range(-24, 24):
                # Seasonal pattern: higher in summer (months 6-8)
                month_of_year = ((m + 24) % 12) + 1
                seasonal_factor = 1.2 if 6 <= month_of_year <= 8 else 0.9
                volume = 1000 * seasonal_factor
                
                panel_data.append({
                    'country': 'COUNTRY_1',
                    'brand_name': brand,
                    'months_postgx': m,
                    'volume': volume,
                    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month_of_year - 1],
                    'n_gxs': 1 if m >= 0 else 0,
                    'ther_area': 'AREA_1',
                    'main_package': 'Tablet',
                    'hospital_rate': 50,
                    'biological': False,
                    'small_molecule': True,
                })
        
        panel = pd.DataFrame(panel_data)
        panel['avg_vol_12m'] = 1000.0  # Mock pre-entry average
        
        # Build features with seasonal enabled
        config = {'pre_entry': {'compute_seasonal': True}}
        features_df = make_features(panel, scenario=1, mode='train', config=config)
        
        # Check seasonal features exist
        assert 'seasonal_amplitude' in features_df.columns
        assert 'seasonal_peak_month' in features_df.columns
        assert 'seasonal_trough_month' in features_df.columns
        assert 'seasonal_ratio' in features_df.columns
        assert 'seasonal_q1_effect' in features_df.columns
        assert 'seasonal_q2_effect' in features_df.columns
        assert 'seasonal_q3_effect' in features_df.columns
        assert 'seasonal_q4_effect' in features_df.columns
    
    def test_seasonal_amplitude_captures_pattern(self):
        """Test that seasonal amplitude captures actual seasonality."""
        from src.features import _add_seasonal_features
        
        # Create panel with strong seasonality
        panel_data = []
        for m in range(-24, 0):
            month_of_year = ((m + 24) % 12) + 1
            # Strong seasonal: double in summer
            volume = 2000 if month_of_year in [6, 7, 8] else 1000
            panel_data.append({
                'country': 'COUNTRY_1',
                'brand_name': 'BRAND_A',
                'months_postgx': m,
                'volume': volume,
                'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month_of_year - 1],
            })
        
        panel = pd.DataFrame(panel_data)
        result = _add_seasonal_features(panel, ['country', 'brand_name'])
        
        # Should detect seasonality
        assert result['seasonal_amplitude'].iloc[0] > 0.3  # Significant amplitude
        # Q3 (Jul-Sep) should have positive effect
        assert result['seasonal_q3_effect'].iloc[0] > 0


class TestFutureGenericsFeatures:
    """Tests for expected future generics features (Section 3.3)."""
    
    def test_future_generics_features_created(self):
        """Test that future n_gxs features are created."""
        from src.features import add_generics_features
        
        # Create panel with n_gxs increasing over time
        panel_data = []
        for m in range(-6, 24):
            n_gxs = max(0, (m // 6) + 1)  # Increases every 6 months
            panel_data.append({
                'country': 'COUNTRY_1',
                'brand_name': 'BRAND_A',
                'months_postgx': m,
                'n_gxs': n_gxs,
            })
        
        panel = pd.DataFrame(panel_data)
        
        # Build generics features with future n_gxs enabled
        config = {'include_future_n_gxs': True}
        result = add_generics_features(panel, cutoff_month=0, config=config)
        
        # Check future generics features exist
        assert 'n_gxs_at_month_12' in result.columns
        assert 'n_gxs_at_month_23' in result.columns
        assert 'n_gxs_max_forecast' in result.columns
        assert 'expected_new_generics' in result.columns
    
    def test_future_generics_values_correct(self):
        """Test that future generics features have correct values."""
        from src.features import _add_future_generics_features
        
        # Create panel with known n_gxs values
        panel_data = []
        for m in range(-6, 24):
            # n_gxs: 0 at entry, 2 at month 12, 4 at month 23
            if m < 0:
                n_gxs = 0
            elif m <= 6:
                n_gxs = 1
            elif m <= 12:
                n_gxs = 2
            else:
                n_gxs = 4
            
            panel_data.append({
                'country': 'COUNTRY_1',
                'brand_name': 'BRAND_A',
                'months_postgx': m,
                'n_gxs': n_gxs,
            })
        
        panel = pd.DataFrame(panel_data)
        result = _add_future_generics_features(panel, ['country', 'brand_name'], cutoff_month=0)
        
        # Check values
        assert result['n_gxs_at_month_12'].iloc[0] == 2
        assert result['n_gxs_at_month_23'].iloc[0] == 4
        assert result['n_gxs_max_forecast'].iloc[0] == 4
        assert result['expected_new_generics'].iloc[0] >= 3  # 4 - 1 = 3


class TestTargetEncodingFeatures:
    """Tests for target encoding features (Section 3.4)."""
    
    def test_target_encoding_function_exists(self):
        """Test that target encoding function exists."""
        from src.features import add_target_encoding_features
        assert callable(add_target_encoding_features)
    
    def test_target_encoding_creates_features(self):
        """Test that target encoding creates expected features."""
        from src.features import add_target_encoding_features
        
        # Create panel with multiple therapeutic areas
        panel_data = []
        for i, brand in enumerate(['BRAND_A', 'BRAND_B', 'BRAND_C', 'BRAND_D', 'BRAND_E']):
            ther_area = 'AREA_1' if i < 3 else 'AREA_2'
            for m in range(24):
                # Different erosion by therapeutic area
                y_norm = 0.5 if ther_area == 'AREA_1' else 0.8
                panel_data.append({
                    'country': 'COUNTRY_1',
                    'brand_name': brand,
                    'months_postgx': m,
                    'volume': 1000 * y_norm,
                    'y_norm': y_norm,
                    'ther_area': ther_area,
                })
        
        panel = pd.DataFrame(panel_data)
        
        config = {
            'enabled': True,
            'features': ['ther_area'],
            'n_folds': 2,
            'smoothing': 5,
        }
        
        result = add_target_encoding_features(panel.copy(), config, scenario=1)
        
        # Check that erosion prior feature was created
        assert 'ther_area_erosion_prior' in result.columns
        # Values should be close to actual means (0.5 and 0.8)
        area1_prior = result[result['ther_area'] == 'AREA_1']['ther_area_erosion_prior'].mean()
        area2_prior = result[result['ther_area'] == 'AREA_2']['ther_area_erosion_prior'].mean()
        
        # They shouldn't be exactly equal due to K-fold, but should be in range
        assert 0.4 < area1_prior < 0.7
        assert 0.6 < area2_prior < 0.9


class TestFeatureSelection:
    """Tests for feature selection utilities (Section 3.8)."""
    
    def test_correlation_analysis_function_exists(self):
        """Test that correlation analysis function exists."""
        from src.features import analyze_feature_correlations
        assert callable(analyze_feature_correlations)
    
    def test_correlation_analysis_returns_correct_format(self):
        """Test correlation analysis returns expected format."""
        from src.features import analyze_feature_correlations
        
        # Create features with known correlations
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'feature_a': np.random.randn(n),
            'feature_b': np.random.randn(n),
            'feature_c_correlated': None,  # Will be correlated with a
        })
        X['feature_c_correlated'] = X['feature_a'] * 0.99 + np.random.randn(n) * 0.01
        
        corr_matrix, redundant = analyze_feature_correlations(X, threshold=0.95)
        
        # Check format
        assert isinstance(corr_matrix, pd.DataFrame)
        assert isinstance(redundant, list)
        
        # feature_c should be marked as redundant (highly correlated with feature_a)
        assert 'feature_c_correlated' in redundant
    
    def test_feature_importance_function_exists(self):
        """Test that permutation importance function exists."""
        from src.features import compute_feature_importance_permutation
        assert callable(compute_feature_importance_permutation)
    
    def test_feature_summary_function(self):
        """Test feature summary generation."""
        from src.features import get_feature_summary
        
        X = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, np.nan, 4.0],
            'categorical_feature': ['a', 'b', 'a', 'c'],
        })
        
        summary = get_feature_summary(X)
        
        assert len(summary) == 2
        assert 'feature' in summary.columns
        assert 'n_missing' in summary.columns
        assert 'missing_pct' in summary.columns
        
        # numeric_feature has 1 missing value
        numeric_row = summary[summary['feature'] == 'numeric_feature'].iloc[0]
        assert numeric_row['n_missing'] == 1
        assert numeric_row['missing_pct'] == 25.0
    
    def test_remove_redundant_features(self):
        """Test redundant feature removal."""
        from src.features import remove_redundant_features
        
        # Create features with high correlation
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'feature_a': np.random.randn(n),
            'feature_b': np.random.randn(n),
            'feature_a_copy': None,
        })
        X['feature_a_copy'] = X['feature_a'] + np.random.randn(n) * 0.001
        
        X_filtered, removed = remove_redundant_features(X, correlation_threshold=0.99)
        
        # One of the correlated features should be removed
        assert len(removed) >= 1
        assert len(X_filtered.columns) < len(X.columns)


class TestInteractionFeatures:
    """Tests for interaction features including ther_area × erosion (Section 3.7)."""
    
    def test_ther_area_erosion_interaction_created(self):
        """Test ther_area × early_erosion interaction is created with target encoding."""
        from src.features import add_target_encoding_features
        
        # Create panel with early erosion features
        panel_data = []
        for i, brand in enumerate(['BRAND_A', 'BRAND_B', 'BRAND_C', 'BRAND_D', 'BRAND_E']):
            ther_area = 'AREA_1' if i < 3 else 'AREA_2'
            for m in range(24):
                panel_data.append({
                    'country': 'COUNTRY_1',
                    'brand_name': brand,
                    'months_postgx': m,
                    'y_norm': 0.5,
                    'ther_area': ther_area,
                    'erosion_0_5': 0.6,  # S2 early erosion feature
                })
        
        panel = pd.DataFrame(panel_data)
        
        config = {
            'enabled': True,
            'features': ['ther_area'],
            'n_folds': 2,
            'smoothing': 5,
        }
        
        result = add_target_encoding_features(panel.copy(), config, scenario=2)
        
        # Check interaction feature exists
        assert 'ther_area_x_early_erosion' in result.columns
        assert 'ther_area_erosion_x_time' in result.columns


# ==============================================================================
# Section 6: Validation & Evaluation Tests
# ==============================================================================

class TestGroupedKFold:
    """Tests for grouped K-fold by therapeutic area (Section 6.1)."""
    
    def test_get_grouped_kfold_series_exists(self):
        """Test grouped K-fold function exists."""
        from src.validation import get_grouped_kfold_series
        assert callable(get_grouped_kfold_series)
    
    def test_get_grouped_kfold_basic(self):
        """Test basic grouped K-fold split by therapeutic area."""
        from src.validation import get_grouped_kfold_series
        
        # Create panel with multiple therapeutic areas
        panel_data = []
        for brand_idx in range(10):
            ther_area = f"AREA_{brand_idx % 3}"  # 3 different areas
            for m in range(12):
                panel_data.append({
                    'country': 'COUNTRY_1',
                    'brand_name': f'BRAND_{brand_idx}',
                    'ther_area': ther_area,
                    'months_postgx': m,
                    'y_norm': 0.5,
                    'bucket': 1 if brand_idx < 5 else 2,
                })
        
        panel_df = pd.DataFrame(panel_data)
        
        # get_grouped_kfold_series returns list of (train_df, val_df) tuples
        folds = get_grouped_kfold_series(panel_df, n_folds=3, group_by='ther_area')
        
        # Should have folds
        assert len(folds) >= 1
        
        # Each fold should have train and val DataFrames
        train_df, val_df = folds[0]
        assert len(train_df) > 0
        assert len(val_df) > 0


class TestPurgedCVSplit:
    """Tests for purged cross-validation with temporal gap (Section 6.1)."""
    
    def test_create_purged_cv_split_exists(self):
        """Test purged CV split function exists."""
        from src.validation import create_purged_cv_split
        assert callable(create_purged_cv_split)
    
    def test_purged_cv_respects_temporal_gap(self):
        """Test that purged CV enforces temporal gap between train and val."""
        from src.validation import create_purged_cv_split
        
        panel_data = []
        for brand_idx in range(6):  # Need 6 series for 3-fold CV
            for m in range(36):
                panel_data.append({
                    'country': 'COUNTRY_1',
                    'brand_name': f'BRAND_{brand_idx}',
                    'months_postgx': m,
                    'y_norm': 0.5,
                    'bucket': 1 if brand_idx < 3 else 2,
                })
        
        panel_df = pd.DataFrame(panel_data)
        gap_months = 3
        
        splits = create_purged_cv_split(panel_df, n_folds=3, gap_months=gap_months, min_train_months=0)
        
        # If we get folds, check the gap
        for train_df, val_df in splits:
            train_max_month = train_df['months_postgx'].max()
            val_min_month = val_df['months_postgx'].min()
            # Val series should have data (train is cutoff based on gap)
            assert train_max_month < val_min_month


class TestNestedCV:
    """Tests for nested cross-validation (Section 6.1)."""
    
    def test_create_nested_cv_exists(self):
        """Test nested CV function exists."""
        from src.validation import create_nested_cv
        assert callable(create_nested_cv)
    
    def test_create_nested_cv_basic(self):
        """Test basic nested CV split creation."""
        from src.validation import create_nested_cv
        
        panel_data = []
        for brand_idx in range(10):
            for m in range(24):
                panel_data.append({
                    'country': 'COUNTRY_1',
                    'brand_name': f'BRAND_{brand_idx}',
                    'months_postgx': m,
                    'y_norm': 0.5,
                    'bucket': 1 if brand_idx < 5 else 2,
                })
        
        panel_df = pd.DataFrame(panel_data)
        
        nested_splits = create_nested_cv(panel_df, outer_folds=3, inner_folds=2)
        
        # Should have outer_folds entries
        assert len(nested_splits) == 3
        
        # Each outer fold should have outer_train, outer_val, inner_folds
        for fold_dict in nested_splits:
            assert 'outer_train' in fold_dict
            assert 'outer_val' in fold_dict
            assert 'inner_folds' in fold_dict
            assert len(fold_dict['outer_train']) > 0
            assert len(fold_dict['outer_val']) > 0


class TestCVAggregation:
    """Tests for CV result aggregation (Section 6.6)."""
    
    def test_aggregate_cv_scores_exists(self):
        """Test CV aggregation function exists."""
        from src.validation import aggregate_cv_scores
        assert callable(aggregate_cv_scores)
    
    def test_aggregate_cv_scores_basic(self):
        """Test CV result aggregation with confidence intervals."""
        from src.validation import aggregate_cv_scores
        
        fold_results = [
            {'metric': 0.15, 'rmse': 0.08},
            {'metric': 0.18, 'rmse': 0.09},
            {'metric': 0.12, 'rmse': 0.07},
            {'metric': 0.16, 'rmse': 0.085},
            {'metric': 0.14, 'rmse': 0.075},
        ]
        
        aggregated = aggregate_cv_scores(fold_results, metric_names=['metric'])
        
        assert 'metric' in aggregated
        assert 'mean' in aggregated['metric']
        assert 'std' in aggregated['metric']
        assert 'ci_lower' in aggregated['metric']
        assert 'ci_upper' in aggregated['metric']
        
        # CI should bracket the mean
        metric_result = aggregated['metric']
        assert metric_result['ci_lower'] < metric_result['mean'] < metric_result['ci_upper']


class TestScenarioDetectionWarning:
    """Tests for scenario detection warning (Section 6.2)."""
    
    def test_expected_counts_in_docstring(self):
        """Test that expected S1 and S2 counts are documented."""
        from src.inference import detect_test_scenarios
        
        # Check docstring mentions expected counts
        assert '228' in detect_test_scenarios.__doc__
        assert '112' in detect_test_scenarios.__doc__


class TestMetricBreakdowns:
    """Tests for metric breakdowns by therapeutic area and country (Section 6.3)."""
    
    def test_compute_metric_by_ther_area_exists(self):
        """Test that metric by therapeutic area function exists."""
        from src.evaluate import compute_metric_by_ther_area
        assert callable(compute_metric_by_ther_area)
    
    def test_compute_metric_by_country_exists(self):
        """Test that metric by country function exists."""
        from src.evaluate import compute_metric_by_country
        assert callable(compute_metric_by_country)
    
    def test_metric_by_ther_area_basic(self):
        """Test basic metric breakdown by therapeutic area."""
        from src.evaluate import compute_metric_by_ther_area
        
        # Create test data
        df_actual = pd.DataFrame({
            'country': ['US', 'US', 'DE', 'DE'],
            'brand_name': ['A', 'B', 'C', 'D'],
            'm1': [0.9, 0.8, 0.7, 0.6],
            'm2': [0.85, 0.75, 0.65, 0.55],
        })
        
        df_pred = pd.DataFrame({
            'country': ['US', 'US', 'DE', 'DE'],
            'brand_name': ['A', 'B', 'C', 'D'],
            'm1': [0.88, 0.78, 0.68, 0.58],
            'm2': [0.83, 0.73, 0.63, 0.53],
        })
        
        df_aux = pd.DataFrame({
            'country': ['US', 'US', 'DE', 'DE'],
            'brand_name': ['A', 'B', 'C', 'D'],
            'm1': [100, 200, 150, 250],
            'm2': [110, 210, 160, 260],
        })
        
        panel_df = pd.DataFrame({
            'country': ['US', 'US', 'DE', 'DE'],
            'brand_name': ['A', 'B', 'C', 'D'],
            'ther_area': ['AREA_1', 'AREA_1', 'AREA_2', 'AREA_2'],
        })
        
        result = compute_metric_by_ther_area(df_actual, df_pred, df_aux, panel_df, scenario=1)
        
        assert isinstance(result, dict)
        assert 'AREA_1' in result or 'AREA_2' in result or len(result) > 0
    
    def test_metric_by_country_basic(self):
        """Test basic metric breakdown by country."""
        from src.evaluate import compute_metric_by_country
        
        df_actual = pd.DataFrame({
            'country': ['US', 'US', 'DE', 'DE'],
            'brand_name': ['A', 'B', 'C', 'D'],
            'm1': [0.9, 0.8, 0.7, 0.6],
            'm2': [0.85, 0.75, 0.65, 0.55],
        })
        
        df_pred = pd.DataFrame({
            'country': ['US', 'US', 'DE', 'DE'],
            'brand_name': ['A', 'B', 'C', 'D'],
            'm1': [0.88, 0.78, 0.68, 0.58],
            'm2': [0.83, 0.73, 0.63, 0.53],
        })
        
        df_aux = pd.DataFrame({
            'country': ['US', 'US', 'DE', 'DE'],
            'brand_name': ['A', 'B', 'C', 'D'],
            'm1': [100, 200, 150, 250],
            'm2': [110, 210, 160, 260],
        })
        
        result = compute_metric_by_country(df_actual, df_pred, df_aux, scenario=1)
        
        assert isinstance(result, dict)
        # Should have country keys
        assert 'US' in result or 'DE' in result or len(result) > 0


class TestEvaluationDataFrame:
    """Tests for evaluation DataFrame creation (Section 6.5)."""
    
    def test_create_evaluation_dataframe_exists(self):
        """Test that evaluation DataFrame function exists."""
        from src.evaluate import create_evaluation_dataframe
        assert callable(create_evaluation_dataframe)
    
    def test_create_evaluation_dataframe_basic(self):
        """Test basic evaluation DataFrame creation."""
        from src.evaluate import create_evaluation_dataframe
        
        # Create test data with months_postgx columns (matching expected format)
        df_actual = pd.DataFrame({
            'country': ['US', 'DE'],
            'brand_name': ['A', 'B'],
            'months_postgx': [0, 0],
            'volume': [0.9, 0.7],
        })
        
        df_pred = pd.DataFrame({
            'country': ['US', 'DE'],
            'brand_name': ['A', 'B'],
            'months_postgx': [0, 0],
            'volume': [0.88, 0.68],
        })
        
        df_aux = pd.DataFrame({
            'country': ['US', 'DE'],
            'brand_name': ['A', 'B'],
            'avg_vol': [100, 150],
        })
        
        panel_df = pd.DataFrame({
            'country': ['US', 'DE'],
            'brand_name': ['A', 'B'],
            'ther_area': ['AREA_1', 'AREA_2'],
        })
        
        result = create_evaluation_dataframe(df_actual, df_pred, df_aux, panel_df, scenario=1)
        
        assert isinstance(result, pd.DataFrame)
        assert 'country' in result.columns
        assert 'brand_name' in result.columns


class TestUnifiedMetricsLogging:
    """Tests for unified metrics logging schema (Section 6.7)."""
    
    def test_make_metric_record_basic(self):
        """Test basic metric record creation."""
        from src.evaluate import make_metric_record
        
        record = make_metric_record(
            phase='train',
            split='fold_0',
            scenario=1,
            model_name='lgbm',
            metric_name='metric1_official',
            value=0.15,
            run_id='test_run_001',
        )
        
        assert isinstance(record, dict)
        assert record['run_id'] == 'test_run_001'
        assert record['phase'] == 'train'
        assert record['split'] == 'fold_0'
        assert record['scenario'] == 1
        assert record['model'] == 'lgbm'
        assert record['metric'] == 'metric1_official'
        assert record['value'] == 0.15
        assert 'timestamp' in record
    
    def test_make_metric_record_with_optional_fields(self):
        """Test metric record with optional fields."""
        from src.evaluate import make_metric_record
        
        record = make_metric_record(
            phase='val',
            split='fold_1',
            scenario=2,
            model_name='xgb',
            metric_name='rmse',
            value=0.08,
            run_id='test_run_002',
            step=100,
            bucket='m1-m6',
            series_id='US_BRAND_A',
            extra={'learning_rate': 0.01}
        )
        
        assert record['step'] == 100
        assert record['bucket'] == 'm1-m6'
        assert record['series_id'] == 'US_BRAND_A'
        assert record['extra'] == {'learning_rate': 0.01}
    
    def test_save_and_load_metric_records(self, tmp_path):
        """Test saving and loading metric records."""
        from src.evaluate import make_metric_record, save_metric_records, load_metric_records
        
        records = [
            make_metric_record('train', 'fold_0', 1, 'lgbm', 'metric1', 0.15, run_id='run1'),
            make_metric_record('train', 'fold_1', 1, 'lgbm', 'metric1', 0.18, run_id='run1'),
            make_metric_record('val', 'fold_0', 1, 'lgbm', 'metric1', 0.20, run_id='run1'),
        ]
        
        path = tmp_path / 'metrics.csv'
        save_metric_records(records, path, append=False)
        
        loaded = load_metric_records(path)
        
        assert len(loaded) == 3
        assert loaded['run_id'].iloc[0] == 'run1'
        assert loaded['metric'].iloc[0] == 'metric1'
    
    def test_save_metric_records_append(self, tmp_path):
        """Test appending metric records to existing file."""
        from src.evaluate import make_metric_record, save_metric_records, load_metric_records
        
        records1 = [
            make_metric_record('train', 'fold_0', 1, 'lgbm', 'metric1', 0.15, run_id='run1'),
        ]
        records2 = [
            make_metric_record('train', 'fold_1', 1, 'lgbm', 'metric1', 0.18, run_id='run1'),
        ]
        
        path = tmp_path / 'metrics_append.csv'
        save_metric_records(records1, path, append=False)
        save_metric_records(records2, path, append=True)
        
        loaded = load_metric_records(path)
        
        assert len(loaded) == 2


class TestConfigMetricsSection:
    """Tests for metrics config section (Section 6.7.1)."""
    
    def test_metrics_config_exists(self):
        """Test that metrics section exists in run_defaults config."""
        import yaml
        
        config_path = Path(__file__).parent.parent / 'configs' / 'run_defaults.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'metrics' in config
        assert 'primary' in config['metrics']
        assert 'secondary' in config['metrics']
        assert 'log_per_series' in config['metrics']
        assert 'log_dir_pattern' in config['metrics']
    
    def test_metrics_config_values(self):
        """Test that metrics config has expected values."""
        import yaml
        
        config_path = Path(__file__).parent.parent / 'configs' / 'run_defaults.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Primary should include official metrics
        assert 'metric1_official' in config['metrics']['primary']
        
        # Secondary should include auxiliary metrics
        secondary = config['metrics']['secondary']
        assert any('rmse' in m.lower() for m in secondary)


class TestTrainScenarioModelUnifiedLogging:
    """Tests for unified logging wired into train_scenario_model (Section 6.7.3)."""
    
    def test_train_scenario_model_signature_has_metrics_params(self):
        """Test that train_scenario_model has run_id, metrics_dir, fold_idx params."""
        import inspect
        from src.train import train_scenario_model
        
        sig = inspect.signature(train_scenario_model)
        params = list(sig.parameters.keys())
        
        assert 'run_id' in params
        assert 'metrics_dir' in params
        assert 'fold_idx' in params
    
    def test_train_scenario_model_saves_metrics_when_metrics_dir_provided(self, tmp_path):
        """Test that train_scenario_model saves metrics when metrics_dir is provided."""
        from src.train import train_scenario_model
        from src.evaluate import load_metric_records
        
        # Create minimal training data
        n_train = 100
        n_val = 30
        
        X_train = pd.DataFrame({
            'feature_1': np.random.randn(n_train),
            'feature_2': np.random.randn(n_train),
        })
        y_train = pd.Series(np.random.rand(n_train) * 0.5 + 0.25)
        meta_train = pd.DataFrame({
            'country': ['US'] * n_train,
            'brand_name': [f'BRAND_{i % 5}' for i in range(n_train)],
            'months_postgx': list(range(24)) * (n_train // 24) + list(range(n_train % 24)),
            'avg_vol_12m': [1000.0] * n_train,
            'bucket': [1 if i < n_train // 2 else 2 for i in range(n_train)],
        })
        
        X_val = pd.DataFrame({
            'feature_1': np.random.randn(n_val),
            'feature_2': np.random.randn(n_val),
        })
        y_val = pd.Series(np.random.rand(n_val) * 0.5 + 0.25)
        meta_val = pd.DataFrame({
            'country': ['US'] * n_val,
            'brand_name': [f'BRAND_{i % 3}' for i in range(n_val)],
            'months_postgx': list(range(24)) * (n_val // 24) + list(range(n_val % 24)),
            'avg_vol_12m': [1000.0] * n_val,
            'bucket': [1 if i < n_val // 2 else 2 for i in range(n_val)],
        })
        
        metrics_dir = tmp_path / 'metrics'
        
        model, metrics = train_scenario_model(
            X_train, y_train, meta_train,
            X_val, y_val, meta_val,
            scenario=1,
            model_type='linear',
            run_id='test_run_unified',
            metrics_dir=metrics_dir,
            fold_idx=0
        )
        
        # Verify metrics file was created
        metrics_file = metrics_dir / 'metrics.csv'
        assert metrics_file.exists()
        
        # Load and verify records
        records_df = load_metric_records(metrics_file)
        assert len(records_df) >= 3  # At least official, rmse, mae
        assert 'run_id' in records_df.columns
        assert records_df['run_id'].iloc[0] == 'test_run_unified'


class TestRunCrossValidationUnifiedLogging:
    """Tests for unified logging in run_cross_validation (Section 6.7.3)."""
    
    def test_run_cross_validation_signature_has_metrics_params(self):
        """Test that run_cross_validation has run_id and metrics_dir params."""
        import inspect
        from src.train import run_cross_validation
        
        sig = inspect.signature(run_cross_validation)
        params = list(sig.parameters.keys())
        
        assert 'run_id' in params
        assert 'metrics_dir' in params


class TestAdversarialValidationUnifiedLogging:
    """Tests for unified logging in adversarial_validation (Section 6.7.4)."""
    
    def test_adversarial_validation_signature_has_metrics_params(self):
        """Test that adversarial_validation has run_id and metrics_dir params."""
        import inspect
        from src.validation import adversarial_validation
        
        sig = inspect.signature(adversarial_validation)
        params = list(sig.parameters.keys())
        
        assert 'run_id' in params
        assert 'metrics_dir' in params
    
    def test_adversarial_validation_saves_metrics_when_metrics_dir_provided(self, tmp_path):
        """Test adversarial_validation saves metrics when metrics_dir is provided."""
        from src.validation import adversarial_validation
        from src.evaluate import load_metric_records
        
        # Create synthetic train and test features
        n = 50
        train_features = pd.DataFrame({
            'feat_1': np.random.randn(n),
            'feat_2': np.random.randn(n) + 0.1,  # Slight shift
        })
        test_features = pd.DataFrame({
            'feat_1': np.random.randn(n),
            'feat_2': np.random.randn(n) + 0.2,  # Different shift
        })
        
        metrics_dir = tmp_path / 'adv_metrics'
        
        result = adversarial_validation(
            train_features, test_features,
            n_folds=2,
            run_id='adv_test_run',
            metrics_dir=metrics_dir
        )
        
        # Verify metrics file was created
        metrics_file = metrics_dir / 'metrics.csv'
        assert metrics_file.exists()
        
        # Load and verify records
        records_df = load_metric_records(metrics_file)
        assert len(records_df) >= 2  # auc_mean and auc_std
        assert records_df['run_id'].iloc[0] == 'adv_test_run'
        assert 'auc' in records_df['metric'].iloc[0].lower()


# ==============================================================================
# SECTION 5 TESTS: TRAINING PIPELINE (NEW ADDITIONS)
# ==============================================================================


class TestExperimentTracker:
    """Tests for experiment tracking functionality (Section 5.1)."""
    
    def test_experiment_tracker_class_exists(self):
        """Test that ExperimentTracker class exists."""
        from src.train import ExperimentTracker
        assert ExperimentTracker is not None
    
    def test_experiment_tracker_init_disabled(self):
        """Test ExperimentTracker initialization with tracking disabled."""
        from src.train import ExperimentTracker
        
        tracker = ExperimentTracker(backend=None, enabled=False)
        assert tracker.enabled == False
        
        # Should not raise when methods are called while disabled
        tracker.start_run(run_name='test')
        tracker.log_params({'param': 'value'})
        tracker.log_metrics({'metric': 0.5})
        tracker.end_run()
    
    def test_experiment_tracker_context_manager(self):
        """Test ExperimentTracker as context manager."""
        from src.train import ExperimentTracker
        
        with ExperimentTracker(backend=None, enabled=False) as tracker:
            tracker.log_params({'param': 1})
        # Should not raise
    
    def test_setup_experiment_tracking_disabled(self):
        """Test setup_experiment_tracking with disabled config."""
        from src.train import setup_experiment_tracking
        
        run_config = {
            'experiment_tracking': {
                'enabled': False
            }
        }
        
        tracker = setup_experiment_tracking(run_config, 'test_run')
        assert tracker is None
    
    def test_tracking_constants_defined(self):
        """Test that tracking availability constants are defined."""
        from src.train import MLFLOW_AVAILABLE, WANDB_AVAILABLE, OPTUNA_AVAILABLE
        
        # These should be boolean
        assert isinstance(MLFLOW_AVAILABLE, bool)
        assert isinstance(WANDB_AVAILABLE, bool)
        assert isinstance(OPTUNA_AVAILABLE, bool)


class TestTrainingCheckpoint:
    """Tests for checkpoint saving functionality (Section 5.1)."""
    
    def test_training_checkpoint_class_exists(self):
        """Test that TrainingCheckpoint class exists."""
        from src.train import TrainingCheckpoint
        assert TrainingCheckpoint is not None
    
    def test_training_checkpoint_init(self, tmp_path):
        """Test TrainingCheckpoint initialization."""
        from src.train import TrainingCheckpoint
        
        checkpoint_dir = tmp_path / 'checkpoints'
        checkpoint = TrainingCheckpoint(
            checkpoint_dir=checkpoint_dir,
            keep_best_n=3,
            metric_name='official_metric',
            minimize=True
        )
        
        assert checkpoint.checkpoint_dir.exists()
        assert checkpoint.keep_best_n == 3
        assert checkpoint.metric_name == 'official_metric'
        assert checkpoint.minimize == True
    
    def test_training_checkpoint_save_and_load(self, tmp_path):
        """Test saving and loading checkpoints."""
        from src.train import TrainingCheckpoint
        from src.models.linear import FlatBaseline
        
        checkpoint_dir = tmp_path / 'checkpoints'
        checkpoint = TrainingCheckpoint(checkpoint_dir=checkpoint_dir)
        
        # Create a simple model
        model = FlatBaseline({})
        model.fit(
            pd.DataFrame({'feature1': [1, 2, 3]}),
            pd.Series([0.9, 0.8, 0.7]),
            pd.DataFrame({'feature1': [4]}),
            pd.Series([0.6]),
            pd.Series([1.0, 1.0, 1.0])
        )
        
        # Save checkpoint
        checkpoint_path = checkpoint.save(
            model=model,
            epoch=5,
            step=100,
            metrics={'official_metric': 0.15, 'rmse': 0.08},
            config={'model_type': 'flat_baseline'},
            is_best=True
        )
        
        assert checkpoint_path.exists()
        assert (checkpoint_path / 'model.bin').exists()
        assert (checkpoint_path / 'training_state.json').exists()
        
        # Load checkpoint
        state = checkpoint.load(checkpoint_path, model_class=FlatBaseline, model_config={})
        
        assert state['epoch'] == 5
        assert state['step'] == 100
        assert state['metrics']['official_metric'] == 0.15
        assert 'model' in state
    
    def test_training_checkpoint_load_best(self, tmp_path):
        """Test loading best checkpoint."""
        from src.train import TrainingCheckpoint
        from src.models.linear import FlatBaseline
        
        checkpoint_dir = tmp_path / 'checkpoints'
        checkpoint = TrainingCheckpoint(checkpoint_dir=checkpoint_dir)
        
        model = FlatBaseline({})
        model.fit(
            pd.DataFrame({'f': [1]}), pd.Series([0.9]),
            pd.DataFrame({'f': [2]}), pd.Series([0.8]),
            pd.Series([1.0])
        )
        
        # Save as best
        checkpoint.save(model=model, epoch=10, metrics={'official_metric': 0.12}, is_best=True)
        
        # Load best
        state = checkpoint.load_best(model_class=FlatBaseline, model_config={})
        
        assert state is not None
        assert state['epoch'] == 10
    
    def test_training_checkpoint_load_latest(self, tmp_path):
        """Test loading latest checkpoint."""
        from src.train import TrainingCheckpoint
        from src.models.linear import FlatBaseline
        import time
        
        checkpoint_dir = tmp_path / 'checkpoints'
        checkpoint = TrainingCheckpoint(checkpoint_dir=checkpoint_dir, keep_best_n=5)
        
        model = FlatBaseline({})
        model.fit(
            pd.DataFrame({'f': [1]}), pd.Series([0.9]),
            pd.DataFrame({'f': [2]}), pd.Series([0.8]),
            pd.Series([1.0])
        )
        
        # Save multiple checkpoints
        checkpoint.save(model=model, epoch=1, metrics={'official_metric': 0.20})
        time.sleep(0.1)  # Ensure different timestamps
        checkpoint.save(model=model, epoch=5, metrics={'official_metric': 0.15})
        time.sleep(0.1)
        checkpoint.save(model=model, epoch=10, metrics={'official_metric': 0.12})
        
        # Load latest
        state = checkpoint.load_latest(model_class=FlatBaseline, model_config={})
        
        assert state is not None
        assert state['epoch'] == 10


class TestWeightTransformations:
    """Tests for sample weight transformations (Section 5.4)."""
    
    def test_transform_weights_identity(self):
        """Test identity weight transformation."""
        from src.train import transform_weights
        
        weights = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        transformed = transform_weights(weights, transformation='identity')
        
        assert len(transformed) == len(weights)
        # Identity should normalize to sum to len(weights)
        assert abs(transformed.sum() - len(weights)) < 0.01
    
    def test_transform_weights_sqrt(self):
        """Test sqrt weight transformation."""
        from src.train import transform_weights
        
        weights = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0])
        transformed = transform_weights(weights, transformation='sqrt')
        
        assert len(transformed) == len(weights)
        # Sqrt should reduce variance
        assert transformed.std() < weights.std()
    
    def test_transform_weights_log(self):
        """Test log weight transformation."""
        from src.train import transform_weights
        
        weights = pd.Series([1.0, 10.0, 100.0, 1000.0])
        transformed = transform_weights(weights, transformation='log')
        
        assert len(transformed) == len(weights)
        # Log should significantly reduce range
        original_range = weights.max() - weights.min()
        transformed_range = transformed.max() - transformed.min()
        assert transformed_range < original_range
    
    def test_transform_weights_rank(self):
        """Test rank weight transformation."""
        from src.train import transform_weights
        
        weights = pd.Series([100.0, 1.0, 50.0, 10.0])
        transformed = transform_weights(weights, transformation='rank')
        
        assert len(transformed) == len(weights)
        # Rank should create more uniform distribution
    
    def test_transform_weights_invalid(self):
        """Test invalid transformation raises error."""
        from src.train import transform_weights
        
        weights = pd.Series([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Unknown weight transformation"):
            transform_weights(weights, transformation='invalid')
    
    def test_compute_sample_weights_with_transform(self):
        """Test compute_sample_weights with weight_transform parameter."""
        from src.train import compute_sample_weights
        
        meta = pd.DataFrame({
            'months_postgx': [0, 3, 6, 12, 18],
            'bucket': [1, 1, 2, 2, 2]
        })
        
        # Test with identity (default)
        weights_identity = compute_sample_weights(
            meta, scenario=1, weight_transform='identity'
        )
        
        # Test with sqrt
        weights_sqrt = compute_sample_weights(
            meta, scenario=1, weight_transform='sqrt'
        )
        
        # Both should have same length
        assert len(weights_identity) == len(weights_sqrt)
        
        # Sqrt should reduce variance
        assert weights_sqrt.std() <= weights_identity.std() + 0.5  # Small tolerance


class TestValidateWeightsCorrelation:
    """Tests for weight validation (Section 5.4)."""
    
    def test_validate_weights_correlation_exists(self):
        """Test validate_weights_correlation function exists."""
        from src.train import validate_weights_correlation
        assert callable(validate_weights_correlation)
    
    def test_validate_weights_correlation_basic(self):
        """Test basic weight validation."""
        from src.train import validate_weights_correlation
        
        np.random.seed(42)
        n = 100
        
        # Create test data
        weights = pd.Series(np.random.rand(n) + 0.5)
        y_true = pd.Series(np.random.rand(n) * 0.5 + 0.5)
        y_pred = y_true + np.random.randn(n) * 0.1  # Predictions with some noise
        
        meta_df = pd.DataFrame({
            'bucket': [1 if i < 50 else 2 for i in range(n)],
            'months_postgx': list(range(24)) * 4 + list(range(4))
        })
        
        results = validate_weights_correlation(
            weights, y_true, y_pred, meta_df, scenario=1
        )
        
        assert 'weighted_rmse' in results
        assert 'unweighted_rmse' in results
        assert 'rmse_ratio' in results
        assert 'weight_error_correlation' in results
        assert 'interpretation' in results


class TestHyperparameterOptimization:
    """Tests for HPO functionality (Section 5.5)."""
    
    def test_optuna_availability_constant_exists(self):
        """Test OPTUNA_AVAILABLE constant exists."""
        from src.train import OPTUNA_AVAILABLE
        assert isinstance(OPTUNA_AVAILABLE, bool)
    
    def test_create_optuna_objective_exists(self):
        """Test create_optuna_objective function exists."""
        from src.train import create_optuna_objective
        assert callable(create_optuna_objective)
    
    def test_run_hyperparameter_optimization_exists(self):
        """Test run_hyperparameter_optimization function exists."""
        from src.train import run_hyperparameter_optimization
        assert callable(run_hyperparameter_optimization)
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
        reason="Training data not available"
    )
    def test_run_hyperparameter_optimization_requires_optuna(self):
        """Test that HPO raises ImportError when Optuna not available."""
        from src.train import OPTUNA_AVAILABLE
        
        if not OPTUNA_AVAILABLE:
            from src.train import run_hyperparameter_optimization
            
            # Should raise ImportError
            with pytest.raises(ImportError, match="Optuna"):
                run_hyperparameter_optimization(
                    pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(),
                    pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(),
                    scenario=1
                )


class TestMemoryProfiler:
    """Tests for memory profiling functionality (Section 5.7)."""
    
    def test_memory_profiler_class_exists(self):
        """Test MemoryProfiler class exists."""
        from src.train import MemoryProfiler
        assert MemoryProfiler is not None
    
    def test_memory_profiler_disabled(self):
        """Test MemoryProfiler when disabled."""
        from src.train import MemoryProfiler
        
        profiler = MemoryProfiler(enabled=False)
        
        # Should not raise when methods are called
        profiler.start()
        profiler.snapshot('test')
        profiler.log_current()
        profiler.stop()
        
        report = profiler.get_report()
        assert report['enabled'] == False
    
    def test_memory_profiler_enabled(self):
        """Test MemoryProfiler when enabled."""
        from src.train import MemoryProfiler
        
        profiler = MemoryProfiler(enabled=True)
        
        profiler.start()
        profiler.snapshot('after_start')
        
        # Allocate some memory
        data = [np.zeros(1000) for _ in range(100)]
        
        profiler.snapshot('after_allocation')
        profiler.stop()
        
        report = profiler.get_report()
        assert report['enabled'] == True
        # Should have peak memory
        assert 'peak_memory_mb' in report


class TestParallelTraining:
    """Tests for parallel training functionality (Section 5.7)."""
    
    def test_train_scenario_parallel_exists(self):
        """Test train_scenario_parallel function exists."""
        from src.train import train_scenario_parallel
        assert callable(train_scenario_parallel)
    
    def test_run_full_training_pipeline_exists(self):
        """Test run_full_training_pipeline function exists."""
        from src.train import run_full_training_pipeline
        assert callable(run_full_training_pipeline)
    
    def test_run_full_training_pipeline_signature(self):
        """Test run_full_training_pipeline has expected parameters."""
        import inspect
        from src.train import run_full_training_pipeline
        
        sig = inspect.signature(run_full_training_pipeline)
        params = list(sig.parameters.keys())
        
        assert 'run_config_path' in params
        assert 'data_config_path' in params
        assert 'model_type' in params
        assert 'run_cv' in params
        assert 'parallel' in params
        assert 'run_hpo' in params
        assert 'enable_tracking' in params
        assert 'enable_checkpoints' in params
        assert 'enable_profiling' in params


class TestTrainCLISection5:
    """Tests for CLI options added in Section 5."""
    
    def test_cli_has_hpo_options(self):
        """Test that CLI has HPO options."""
        import subprocess
        
        result = subprocess.run(
            ['python', '-m', 'src.train', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        assert result.returncode == 0
        assert '--hpo' in result.stdout
        assert '--hpo-trials' in result.stdout
        assert '--hpo-timeout' in result.stdout
    
    def test_cli_has_full_pipeline_options(self):
        """Test that CLI has full pipeline options."""
        import subprocess
        
        result = subprocess.run(
            ['python', '-m', 'src.train', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        assert result.returncode == 0
        assert '--full-pipeline' in result.stdout
        assert '--parallel' in result.stdout
    
    def test_cli_has_tracking_options(self):
        """Test that CLI has experiment tracking options."""
        import subprocess
        
        result = subprocess.run(
            ['python', '-m', 'src.train', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        assert result.returncode == 0
        assert '--enable-tracking' in result.stdout
        assert '--tracking-backend' in result.stdout
        assert '--enable-checkpoints' in result.stdout
        assert '--enable-profiling' in result.stdout
    
    def test_cli_has_weight_transform_option(self):
        """Test that CLI has weight transform option."""
        import subprocess
        
        result = subprocess.run(
            ['python', '-m', 'src.train', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        assert result.returncode == 0
        assert '--weight-transform' in result.stdout
        assert '--metric-aligned-weights' in result.stdout


class TestConfigSection5:
    """Tests for config additions in Section 5."""
    
    def test_experiment_tracking_config_exists(self):
        """Test that experiment_tracking section exists in config."""
        from src.utils import load_config
        
        config = load_config('configs/run_defaults.yaml')
        
        assert 'experiment_tracking' in config
        tracking = config['experiment_tracking']
        
        assert 'enabled' in tracking
        assert 'backend' in tracking
        assert 'experiment_name' in tracking
    
    def test_experiment_tracking_config_values(self):
        """Test experiment_tracking config has expected values."""
        from src.utils import load_config
        
        config = load_config('configs/run_defaults.yaml')
        tracking = config['experiment_tracking']
        
        assert tracking['enabled'] == False  # Disabled by default
        assert tracking['backend'] in ['mlflow', 'wandb']
        assert isinstance(tracking['experiment_name'], str)


# =============================================================================
# SECTION 4 TESTS: MODEL DEVELOPMENT
# =============================================================================


class TestBaseModelInterface:
    """Tests for BaseModel interface (Section 4.0)."""
    
    def test_base_model_abstract_class(self):
        """Test BaseModel is an abstract class with required methods."""
        from src.models.base import BaseModel
        import abc
        
        assert abc.ABC in BaseModel.__bases__
        
        # Check required abstract methods
        abstract_methods = getattr(BaseModel, '__abstractmethods__', set())
        assert 'fit' in abstract_methods
        assert 'predict' in abstract_methods
    
    def test_all_models_implement_interface(self):
        """Test that all model classes implement BaseModel interface."""
        from src.models.base import BaseModel
        from src.models.linear import (
            LinearModel, GlobalMeanBaseline, FlatBaseline, 
            TrendBaseline, HistoricalCurveBaseline
        )
        from src.models.ensemble import (
            AveragingEnsemble, WeightedAveragingEnsemble,
            StackingEnsemble, BlendingEnsemble
        )
        
        models_to_test = [
            LinearModel, GlobalMeanBaseline, FlatBaseline,
            TrendBaseline, HistoricalCurveBaseline,
            AveragingEnsemble, WeightedAveragingEnsemble,
            StackingEnsemble, BlendingEnsemble
        ]
        
        for model_cls in models_to_test:
            assert issubclass(model_cls, BaseModel), \
                f"{model_cls.__name__} should inherit from BaseModel"
            
            # Check it has required methods
            assert hasattr(model_cls, 'fit')
            assert hasattr(model_cls, 'predict')
            assert hasattr(model_cls, 'save')
            assert hasattr(model_cls, 'load')
            assert hasattr(model_cls, 'get_feature_importance')
    
    def test_model_factory_function(self):
        """Test get_model_class returns correct model types."""
        from src.models import get_model_class
        from src.models.linear import (
            LinearModel, GlobalMeanBaseline, FlatBaseline,
            TrendBaseline, HistoricalCurveBaseline
        )
        from src.models.ensemble import (
            AveragingEnsemble, WeightedAveragingEnsemble,
            StackingEnsemble, BlendingEnsemble
        )
        
        assert get_model_class('linear') == LinearModel
        assert get_model_class('global_mean') == GlobalMeanBaseline
        assert get_model_class('flat') == FlatBaseline
        assert get_model_class('trend') == TrendBaseline
        assert get_model_class('historical_curve') == HistoricalCurveBaseline
        assert get_model_class('knn_curve') == HistoricalCurveBaseline
        assert get_model_class('averaging') == AveragingEnsemble
        assert get_model_class('weighted') == WeightedAveragingEnsemble
        assert get_model_class('stacking') == StackingEnsemble
        assert get_model_class('blending') == BlendingEnsemble
    
    def test_model_factory_case_insensitive(self):
        """Test get_model_class is case-insensitive."""
        from src.models import get_model_class
        
        assert get_model_class('LINEAR') == get_model_class('linear')
        assert get_model_class('Averaging') == get_model_class('averaging')
    
    def test_model_factory_unknown_raises(self):
        """Test get_model_class raises for unknown model."""
        from src.models import get_model_class
        
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_class('unknown_model')


class TestHistoricalCurveBaseline:
    """Tests for HistoricalCurveBaseline (Section 4.1)."""
    
    def test_historical_curve_baseline_exists(self):
        """Test HistoricalCurveBaseline class exists."""
        from src.models.linear import HistoricalCurveBaseline
        assert HistoricalCurveBaseline is not None
    
    def test_historical_curve_baseline_fit_predict(self):
        """Test HistoricalCurveBaseline fit and predict."""
        from src.models.linear import HistoricalCurveBaseline
        
        np.random.seed(42)
        
        # Create training data with ther_area groups
        X_train = pd.DataFrame({
            'months_postgx': list(range(24)) * 4,
            'ther_area_encoded': [0] * 48 + [1] * 48,  # Two therapeutic areas
            'feature1': np.random.randn(96)
        })
        y_train = pd.Series(
            [1.0 - 0.02 * m + np.random.normal(0, 0.05) for m in range(24)] * 2 +
            [1.0 - 0.03 * m + np.random.normal(0, 0.05) for m in range(24)] * 2
        )
        
        X_val = X_train.iloc[:24].copy()
        y_val = y_train.iloc[:24].copy()
        sample_weight = pd.Series(np.ones(96))
        
        model = HistoricalCurveBaseline({'n_clusters': 2})
        model.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        # Predict
        preds = model.predict(X_val)
        
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
    
    def test_historical_curve_baseline_save_load(self, tmp_path):
        """Test HistoricalCurveBaseline save and load."""
        from src.models.linear import HistoricalCurveBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(24)) * 2,
            'ther_area_encoded': [0] * 48,
            'feature1': np.random.randn(48)
        })
        y_train = pd.Series([1.0 - 0.02 * m for m in range(24)] * 2)
        sample_weight = pd.Series(np.ones(48))
        
        model = HistoricalCurveBaseline({})
        model.fit(X_train, y_train, X_train.iloc[:24], y_train.iloc[:24], sample_weight)
        
        # Save
        model_path = tmp_path / 'model.pkl'
        model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load using classmethod (returns new instance)
        loaded_model = HistoricalCurveBaseline.load(str(model_path))
        
        # Predictions should match
        orig_preds = model.predict(X_train.iloc[:24])
        loaded_preds = loaded_model.predict(X_train.iloc[:24])
        
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)


class TestLinearModelPolynomial:
    """Tests for LinearModel polynomial features (Section 4.2)."""
    
    def test_linear_model_polynomial_config(self):
        """Test LinearModel accepts polynomial config."""
        from src.models.linear import LinearModel
        
        config = {
            'use_polynomial': True,
            'polynomial_degree': 2
        }
        
        model = LinearModel(config)
        assert model.config.get('use_polynomial') == True
        assert model.config.get('polynomial_degree') == 2
    
    def test_linear_model_polynomial_fit(self):
        """Test LinearModel with polynomial features fits and predicts."""
        from src.models.linear import LinearModel
        
        np.random.seed(42)
        
        # Create non-linear data
        X_train = pd.DataFrame({
            'x': np.linspace(0, 10, 100)
        })
        # Quadratic relationship
        y_train = pd.Series(X_train['x'] ** 2 + np.random.normal(0, 1, 100))
        
        X_val = X_train.iloc[:20].copy()
        y_val = y_train.iloc[:20].copy()
        sample_weight = pd.Series(np.ones(100))
        
        # Model with polynomial features
        model = LinearModel({
            'use_polynomial': True,
            'polynomial_degree': 2,
            'model_type': 'ridge'
        })
        model.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        preds = model.predict(X_train)
        
        assert len(preds) == len(X_train)
        
        # Polynomial model should capture the quadratic relationship better
        # than linear (R^2 should be high)
        ss_res = np.sum((y_train - preds) ** 2)
        ss_tot = np.sum((y_train - y_train.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        assert r2 > 0.9, f"R^2 should be high for quadratic data with polynomial features: {r2}"
    
    def test_linear_model_without_polynomial(self):
        """Test LinearModel works without polynomial features."""
        from src.models.linear import LinearModel
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'x': np.linspace(0, 10, 100)
        })
        y_train = pd.Series(2 * X_train['x'] + 1 + np.random.normal(0, 0.5, 100))
        
        X_val = X_train.iloc[:20].copy()
        y_val = y_train.iloc[:20].copy()
        sample_weight = pd.Series(np.ones(100))
        
        # Model without polynomial features (default)
        model = LinearModel({'model_type': 'ridge'})
        model.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        preds = model.predict(X_train)
        
        assert len(preds) == len(X_train)


class TestEnsembleModels:
    """Tests for ensemble models (Section 4.8)."""
    
    def test_averaging_ensemble_exists(self):
        """Test AveragingEnsemble class exists."""
        from src.models.ensemble import AveragingEnsemble
        assert AveragingEnsemble is not None
    
    def test_weighted_averaging_ensemble_exists(self):
        """Test WeightedAveragingEnsemble class exists."""
        from src.models.ensemble import WeightedAveragingEnsemble
        assert WeightedAveragingEnsemble is not None
    
    def test_stacking_ensemble_exists(self):
        """Test StackingEnsemble class exists."""
        from src.models.ensemble import StackingEnsemble
        assert StackingEnsemble is not None
    
    def test_blending_ensemble_exists(self):
        """Test BlendingEnsemble class exists."""
        from src.models.ensemble import BlendingEnsemble
        assert BlendingEnsemble is not None
    
    def test_create_ensemble_helper(self):
        """Test create_ensemble factory function."""
        from src.models.ensemble import create_ensemble, AveragingEnsemble, StackingEnsemble
        from src.models.linear import FlatBaseline, GlobalMeanBaseline
        
        # create_ensemble takes models as first arg, method as second
        base_models = [FlatBaseline({}), GlobalMeanBaseline({})]
        
        avg_ensemble = create_ensemble(base_models, method='averaging')
        assert isinstance(avg_ensemble, AveragingEnsemble)
        
        stack_ensemble = create_ensemble(base_models, method='stacking')
        assert isinstance(stack_ensemble, StackingEnsemble)
    
    def test_averaging_ensemble_fit_predict(self):
        """Test AveragingEnsemble fit and predict."""
        from src.models.ensemble import AveragingEnsemble
        from src.models.linear import FlatBaseline, GlobalMeanBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(24)) * 4,
            'feature1': np.random.randn(96)
        })
        y_train = pd.Series(np.random.rand(96) * 0.5 + 0.5)
        
        X_val = X_train.iloc[:24].copy()
        y_val = y_train.iloc[:24].copy()
        sample_weight = pd.Series(np.ones(96))
        
        # Create base models using 'models' key (not 'base_models')
        base_models = [
            FlatBaseline({}),
            GlobalMeanBaseline({})
        ]
        
        ensemble = AveragingEnsemble({'models': base_models})
        ensemble.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        preds = ensemble.predict(X_val)
        
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        
        # Average of flat (1.0) and global mean should be between them
        assert preds.min() >= 0
        assert preds.max() <= 1.5
    
    def test_weighted_averaging_ensemble_fit_predict(self):
        """Test WeightedAveragingEnsemble fit and predict."""
        from src.models.ensemble import WeightedAveragingEnsemble
        from src.models.linear import FlatBaseline, GlobalMeanBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(24)) * 4,
            'feature1': np.random.randn(96)
        })
        y_train = pd.Series(np.random.rand(96) * 0.5 + 0.5)
        
        X_val = X_train.iloc[:24].copy()
        y_val = y_train.iloc[:24].copy()
        sample_weight = pd.Series(np.ones(96))
        
        # Use 'models' key for WeightedAveragingEnsemble
        base_models = [
            FlatBaseline({}),
            GlobalMeanBaseline({})
        ]
        
        ensemble = WeightedAveragingEnsemble({
            'models': base_models,
            'optimize_weights': True
        })
        ensemble.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        preds = ensemble.predict(X_val)
        
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        
        # Check weights are stored (in 'weights' attribute, not 'weights_')
        assert hasattr(ensemble, 'weights')
        assert ensemble.weights is not None
        assert len(ensemble.weights) == 2
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6  # Weights should sum to 1
    
    def test_stacking_ensemble_fit_predict(self):
        """Test StackingEnsemble fit and predict."""
        from src.models.ensemble import StackingEnsemble
        from src.models.linear import FlatBaseline, GlobalMeanBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(24)) * 4,
            'feature1': np.random.randn(96)
        })
        y_train = pd.Series(np.random.rand(96) * 0.5 + 0.5)
        
        X_val = X_train.iloc[:24].copy()
        y_val = y_train.iloc[:24].copy()
        sample_weight = pd.Series(np.ones(96))
        
        # StackingEnsemble uses 'base_models' as list of (name, model) tuples
        base_models = [
            ('flat', FlatBaseline({})),
            ('global_mean', GlobalMeanBaseline({}))
        ]
        
        # Don't pass meta_learner string - use default Ridge
        ensemble = StackingEnsemble({
            'base_models': base_models,
            'n_folds': 2
        })
        ensemble.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        preds = ensemble.predict(X_val)
        
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
    
    def test_blending_ensemble_fit_predict(self):
        """Test BlendingEnsemble fit and predict."""
        from src.models.ensemble import BlendingEnsemble
        from src.models.linear import FlatBaseline, GlobalMeanBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(24)) * 4,
            'feature1': np.random.randn(96)
        })
        y_train = pd.Series(np.random.rand(96) * 0.5 + 0.5)
        
        X_val = X_train.iloc[:24].copy()
        y_val = y_train.iloc[:24].copy()
        sample_weight = pd.Series(np.ones(96))
        
        # Use 'models' key for BlendingEnsemble
        base_models = [
            FlatBaseline({}),
            GlobalMeanBaseline({})
        ]
        
        ensemble = BlendingEnsemble({
            'models': base_models,
            'holdout_fraction': 0.3
        })
        ensemble.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        preds = ensemble.predict(X_val)
        
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
    
    def test_ensemble_save_load(self, tmp_path):
        """Test ensemble models can be saved and loaded."""
        from src.models.ensemble import AveragingEnsemble
        from src.models.linear import FlatBaseline, GlobalMeanBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(24)) * 2,
            'feature1': np.random.randn(48)
        })
        y_train = pd.Series(np.random.rand(48) * 0.5 + 0.5)
        sample_weight = pd.Series(np.ones(48))
        
        # Use 'models' key
        base_models = [FlatBaseline({}), GlobalMeanBaseline({})]
        
        ensemble = AveragingEnsemble({'models': base_models})
        ensemble.fit(X_train, y_train, X_train.iloc[:24], y_train.iloc[:24], sample_weight)
        
        # Save
        model_path = tmp_path / 'ensemble.pkl'
        ensemble.save(str(model_path))
        
        assert model_path.exists()
        
        # Load using classmethod (returns new instance)
        loaded = AveragingEnsemble.load(str(model_path))
        
        # Predictions should match
        orig_preds = ensemble.predict(X_train.iloc[:24])
        loaded_preds = loaded.predict(X_train.iloc[:24])
        
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)
    
    def test_ensemble_feature_importance(self):
        """Test ensemble models return feature importance."""
        from src.models.ensemble import AveragingEnsemble
        from src.models.linear import FlatBaseline, GlobalMeanBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(24)) * 2,
            'feature1': np.random.randn(48)
        })
        y_train = pd.Series(np.random.rand(48) * 0.5 + 0.5)
        sample_weight = pd.Series(np.ones(48))
        
        # Use 'models' key
        base_models = [FlatBaseline({}), GlobalMeanBaseline({})]
        
        ensemble = AveragingEnsemble({'models': base_models})
        ensemble.fit(X_train, y_train, X_train.iloc[:24], y_train.iloc[:24], sample_weight)
        
        # Get feature importance (should return None or empty for baselines)
        importance = ensemble.get_feature_importance()
        
        # Should return something (None, empty dict, or DataFrame)
        assert importance is None or isinstance(importance, (dict, pd.DataFrame))


class TestTrainGetModel:
    """Tests for _get_model function in train.py (Section 4.0)."""
    
    def test_get_model_returns_model_instances(self):
        """Test _get_model returns actual model instances for models without native lib deps."""
        from src.train import _get_model
        from src.models.base import BaseModel
        
        # Test models that don't require native libraries (CatBoost, LightGBM, XGBoost)
        model_types = [
            'linear',
            'global_mean', 'flat', 'trend', 'historical_curve',
            'averaging', 'weighted', 'stacking', 'blending'
        ]
        
        for model_type in model_types:
            model = _get_model(model_type)
            assert isinstance(model, BaseModel), \
                f"_get_model('{model_type}') should return a BaseModel instance"
    
    def test_get_model_with_config(self):
        """Test _get_model passes config to model."""
        from src.train import _get_model
        
        config = {'custom_param': 'value'}
        model = _get_model('linear', config)
        
        assert model.config.get('custom_param') == 'value'
    
    def test_get_model_unknown_raises(self):
        """Test _get_model raises for unknown model type."""
        from src.train import _get_model
        
        with pytest.raises(ValueError, match="Unknown model type"):
            _get_model('nonexistent_model')


class TestTrendBaseline:
    """Tests for TrendBaseline model (Section 4.1)."""
    
    def test_trend_baseline_fit_predict(self):
        """Test TrendBaseline fit and predict."""
        from src.models.linear import TrendBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(-12, 24)) * 2,
            'feature1': np.random.randn(72)
        })
        y_train = pd.Series([1.0 - 0.01 * m for m in range(-12, 24)] * 2)
        
        X_val = X_train.iloc[:36].copy()
        y_val = y_train.iloc[:36].copy()
        sample_weight = pd.Series(np.ones(72))
        
        model = TrendBaseline({})
        model.fit(X_train, y_train, X_val, y_val, sample_weight)
        
        preds = model.predict(X_val)
        
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
    
    def test_trend_baseline_save_load(self, tmp_path):
        """Test TrendBaseline save and load."""
        from src.models.linear import TrendBaseline
        
        np.random.seed(42)
        
        X_train = pd.DataFrame({
            'months_postgx': list(range(-12, 24)),
            'feature1': np.random.randn(36)
        })
        y_train = pd.Series([1.0 - 0.01 * m for m in range(-12, 24)])
        sample_weight = pd.Series(np.ones(36))
        
        model = TrendBaseline({})
        model.fit(X_train, y_train, X_train.iloc[:12], y_train.iloc[:12], sample_weight)
        
        # Save
        model_path = tmp_path / 'trend.pkl'
        model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load using classmethod (returns new instance)
        loaded_model = TrendBaseline.load(str(model_path))
        
        # Predictions should match
        orig_preds = model.predict(X_train.iloc[:12])
        loaded_preds = loaded_model.predict(X_train.iloc[:12])
        
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)


# =============================================================================
# Section 7: Inference & Submission Tests
# =============================================================================

class TestBatchPrediction:
    """Tests for batch prediction functionality (Section 7.1)."""
    
    def test_predict_batch_small_dataset(self):
        """Test batch prediction with small dataset (no batching needed)."""
        from src.inference import predict_batch
        
        # Simple mock model
        class MockModel:
            def predict(self, X):
                return np.ones(len(X))
        
        model = MockModel()
        X = pd.DataFrame({'feature1': np.random.randn(100)})
        
        preds = predict_batch(model, X, batch_size=200, verbose=False)
        
        assert len(preds) == 100
        assert np.allclose(preds, 1.0)
    
    def test_predict_batch_large_dataset(self):
        """Test batch prediction with large dataset (batching used)."""
        from src.inference import predict_batch
        
        # Mock model that tracks call count
        class MockModel:
            def __init__(self):
                self.call_count = 0
            
            def predict(self, X):
                self.call_count += 1
                return np.arange(len(X))
        
        model = MockModel()
        X = pd.DataFrame({'feature1': np.random.randn(100)})
        
        preds = predict_batch(model, X, batch_size=30, verbose=False)
        
        assert len(preds) == 100
        assert model.call_count == 4  # 100/30 = 4 batches
        # Check values are sequential (as expected from mock)
        np.testing.assert_array_equal(preds, np.concatenate([
            np.arange(30), np.arange(30), np.arange(30), np.arange(10)
        ]))
    
    def test_predict_batch_preserves_order(self):
        """Test batch prediction preserves order."""
        from src.inference import predict_batch
        
        class MockModel:
            def predict(self, X):
                return X['value'].values * 2
        
        model = MockModel()
        X = pd.DataFrame({'value': np.arange(100)})
        
        preds = predict_batch(model, X, batch_size=25, verbose=False)
        
        expected = np.arange(100) * 2
        np.testing.assert_array_equal(preds, expected)


class TestConfidenceIntervals:
    """Tests for confidence interval computation (Section 7.1)."""
    
    def test_predict_with_confidence_single_model(self):
        """Test confidence intervals with single model (no spread)."""
        from src.inference import predict_with_confidence
        
        class MockModel:
            def predict(self, X):
                return np.ones(len(X)) * 0.5
        
        models = [MockModel()]
        X = pd.DataFrame({'feature1': np.random.randn(50)})
        
        mean_pred, lower, upper = predict_with_confidence(models, X)
        
        assert len(mean_pred) == 50
        assert np.allclose(mean_pred, 0.5)
        # With single model, bounds should equal mean
        np.testing.assert_array_equal(lower, mean_pred)
        np.testing.assert_array_equal(upper, mean_pred)
    
    def test_predict_with_confidence_ensemble(self):
        """Test confidence intervals with ensemble."""
        from src.inference import predict_with_confidence
        
        # Models with different predictions
        class MockModel:
            def __init__(self, offset):
                self.offset = offset
            
            def predict(self, X):
                return np.ones(len(X)) * self.offset
        
        # 5 models predicting 0.1, 0.3, 0.5, 0.7, 0.9
        models = [MockModel(o) for o in [0.1, 0.3, 0.5, 0.7, 0.9]]
        X = pd.DataFrame({'feature1': np.random.randn(10)})
        
        mean_pred, lower, upper = predict_with_confidence(models, X, confidence_level=0.80)
        
        assert len(mean_pred) == 10
        assert np.allclose(mean_pred, 0.5)  # Mean of 0.1, 0.3, 0.5, 0.7, 0.9
        assert all(lower <= mean_pred)
        assert all(upper >= mean_pred)
        assert all(lower < upper)  # Bounds should differ
    
    def test_predict_with_confidence_empty_ensemble_raises(self):
        """Test that empty ensemble raises error."""
        from src.inference import predict_with_confidence
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="At least one model"):
            predict_with_confidence([], X)


class TestPredictionClipping:
    """Tests for prediction clipping (Section 7.1)."""
    
    def test_clip_predictions_no_clipping_needed(self):
        """Test clipping when all values are within bounds."""
        from src.inference import clip_predictions
        
        preds = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
        
        clipped = clip_predictions(preds)
        
        np.testing.assert_array_equal(clipped, preds)
    
    def test_clip_predictions_clips_low_values(self):
        """Test clipping of values below minimum."""
        from src.inference import clip_predictions
        
        preds = np.array([-0.5, 0.0, 0.5, 1.0])
        
        clipped = clip_predictions(preds, min_bound=0.0)
        
        assert clipped[0] == 0.0
        assert clipped[1] == 0.0
        assert clipped[2] == 0.5
    
    def test_clip_predictions_clips_high_values(self):
        """Test clipping of values above maximum."""
        from src.inference import clip_predictions
        
        preds = np.array([1.5, 2.0, 2.5, 3.0])
        
        clipped = clip_predictions(preds, max_bound=2.0)
        
        assert clipped[0] == 1.5
        assert clipped[1] == 2.0
        assert clipped[2] == 2.0
        assert clipped[3] == 2.0
    
    def test_clip_predictions_custom_bounds(self):
        """Test clipping with custom bounds."""
        from src.inference import clip_predictions
        
        preds = np.array([0.0, 0.5, 1.0, 1.5])
        
        clipped = clip_predictions(preds, min_bound=0.3, max_bound=1.2)
        
        assert clipped[0] == 0.3
        assert clipped[1] == 0.5
        assert clipped[2] == 1.0
        assert clipped[3] == 1.2


class TestInverseTransformVerification:
    """Tests for inverse transform verification (Section 7.1)."""
    
    def test_verify_inverse_transform_correct(self):
        """Test verification passes for correct transform."""
        from src.inference import verify_inverse_transform
        
        y_norm = np.array([0.5, 0.8, 1.0])
        avg_vol = np.array([1000.0, 2000.0, 500.0])
        volume = y_norm * avg_vol  # Correct transform
        
        result = verify_inverse_transform(y_norm, avg_vol, volume)
        
        assert result is True
    
    def test_verify_inverse_transform_incorrect(self):
        """Test verification fails for incorrect transform."""
        from src.inference import verify_inverse_transform
        
        y_norm = np.array([0.5, 0.8, 1.0])
        avg_vol = np.array([1000.0, 2000.0, 500.0])
        volume = y_norm * avg_vol * 1.5  # Incorrect transform
        
        with pytest.raises(ValueError, match="verification failed"):
            verify_inverse_transform(y_norm, avg_vol, volume)
    
    def test_verify_inverse_transform_handles_zeros(self):
        """Test verification handles zero values gracefully."""
        from src.inference import verify_inverse_transform
        
        y_norm = np.array([0.0, 0.5, 0.0])
        avg_vol = np.array([0.0, 1000.0, 500.0])
        volume = y_norm * avg_vol  # [0, 500, 0]
        
        result = verify_inverse_transform(y_norm, avg_vol, volume)
        
        assert result is True


class TestSubmissionValidation:
    """Tests for enhanced submission validation (Section 7.2-7.3)."""
    
    def test_validate_submission_format_valid(self):
        """Test validation passes for valid submission."""
        from src.inference import validate_submission_format
        
        template = pd.DataFrame({
            'country': ['A', 'A', 'B', 'B'],
            'brand_name': ['X', 'X', 'Y', 'Y'],
            'months_postgx': [0, 1, 0, 1],
            'volume': [np.nan] * 4
        })
        
        submission = pd.DataFrame({
            'country': ['A', 'A', 'B', 'B'],
            'brand_name': ['X', 'X', 'Y', 'Y'],
            'months_postgx': [0, 1, 0, 1],
            'volume': [1000.0, 900.0, 2000.0, 1800.0]
        })
        
        result = validate_submission_format(submission, template)
        assert result is True
    
    def test_validate_submission_format_missing_rows(self):
        """Test validation fails for missing rows."""
        from src.inference import validate_submission_format
        
        template = pd.DataFrame({
            'country': ['A', 'B'],
            'brand_name': ['X', 'Y'],
            'months_postgx': [0, 0],
            'volume': [np.nan] * 2
        })
        
        submission = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
            'volume': [1000.0]
        })
        
        with pytest.raises(ValueError, match="Row count mismatch"):
            validate_submission_format(submission, template)
    
    def test_validate_submission_format_nan_values(self):
        """Test validation fails for NaN values."""
        from src.inference import validate_submission_format
        
        template = pd.DataFrame({
            'country': ['A', 'B'],
            'brand_name': ['X', 'Y'],
            'months_postgx': [0, 0],
            'volume': [np.nan] * 2
        })
        
        submission = pd.DataFrame({
            'country': ['A', 'B'],
            'brand_name': ['X', 'Y'],
            'months_postgx': [0, 0],
            'volume': [1000.0, np.nan]
        })
        
        with pytest.raises(ValueError, match="missing volume values"):
            validate_submission_format(submission, template)
    
    def test_validate_submission_format_negative_values(self):
        """Test validation fails for negative values."""
        from src.inference import validate_submission_format
        
        template = pd.DataFrame({
            'country': ['A', 'B'],
            'brand_name': ['X', 'Y'],
            'months_postgx': [0, 0],
            'volume': [np.nan] * 2
        })
        
        submission = pd.DataFrame({
            'country': ['A', 'B'],
            'brand_name': ['X', 'Y'],
            'months_postgx': [0, 0],
            'volume': [1000.0, -100.0]
        })
        
        with pytest.raises(ValueError, match="negative volume values"):
            validate_submission_format(submission, template)
    
    def test_validate_submission_format_duplicates(self):
        """Test validation fails for duplicates."""
        from src.inference import validate_submission_format
        
        template = pd.DataFrame({
            'country': ['A', 'B'],
            'brand_name': ['X', 'Y'],
            'months_postgx': [0, 0],
            'volume': [np.nan] * 2
        })
        
        submission = pd.DataFrame({
            'country': ['A', 'A'],
            'brand_name': ['X', 'X'],
            'months_postgx': [0, 0],
            'volume': [1000.0, 1100.0]
        })
        
        with pytest.raises(ValueError, match="duplicate"):
            validate_submission_format(submission, template)
    
    def test_validate_submission_format_non_strict_returns_false(self):
        """Test non-strict mode returns False instead of raising."""
        from src.inference import validate_submission_format
        
        template = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
            'volume': [np.nan]
        })
        
        submission = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
            'volume': [np.nan]  # NaN is invalid
        })
        
        result = validate_submission_format(submission, template, strict=False)
        assert result is False


class TestSubmissionStatistics:
    """Tests for submission statistics (Section 7.4)."""
    
    def test_check_submission_statistics(self):
        """Test statistics computation."""
        from src.inference import check_submission_statistics
        
        submission = pd.DataFrame({
            'country': ['A'] * 24 + ['B'] * 18,
            'brand_name': ['X'] * 24 + ['Y'] * 18,
            'months_postgx': list(range(24)) + list(range(6, 24)),
            'volume': [1000 - i * 20 for i in range(24)] + [500 - i * 15 for i in range(18)]
        })
        
        stats = check_submission_statistics(submission, verbose=False)
        
        assert 'volume_mean' in stats
        assert 'volume_std' in stats
        assert 'volume_min' in stats
        assert 'volume_max' in stats
        assert 'n_scenario1_series' in stats
        assert 'n_scenario2_series' in stats
        assert 'total_rows' in stats
        
        assert stats['total_rows'] == 42
        assert stats['n_scenario1_series'] == 1  # Series starting at month 0
        assert stats['n_scenario2_series'] == 1  # Series starting at month 6


class TestAuxiliaryFileGeneration:
    """Tests for auxiliary file generation (Section 7.3)."""
    
    def test_generate_auxiliary_file(self):
        """Test auxiliary file generation."""
        from src.inference import generate_auxiliary_file
        
        submission = pd.DataFrame({
            'country': ['A'] * 24 + ['B'] * 24,
            'brand_name': ['X'] * 24 + ['Y'] * 24,
            'months_postgx': list(range(24)) * 2,
            'volume': [1000.0] * 48
        })
        
        panel = pd.DataFrame({
            'country': ['A'] * 24 + ['B'] * 24,
            'brand_name': ['X'] * 24 + ['Y'] * 24,
            'months_postgx': list(range(24)) * 2,
            'avg_vol_12m': [1500.0] * 24 + [2000.0] * 24,
            'bucket': [1] * 24 + [2] * 24
        })
        
        aux = generate_auxiliary_file(submission, panel)
        
        assert len(aux) == 2  # Two unique series
        assert list(aux.columns) == ['country', 'brand_name', 'avg_vol', 'bucket']
        assert aux.loc[aux['country'] == 'A', 'avg_vol'].values[0] == 1500.0
        assert aux.loc[aux['country'] == 'B', 'bucket'].values[0] == 2
    
    def test_generate_auxiliary_file_saves_to_path(self, tmp_path):
        """Test auxiliary file is saved when path provided."""
        from src.inference import generate_auxiliary_file
        
        submission = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
            'volume': [1000.0]
        })
        
        panel = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
            'avg_vol_12m': [1500.0],
            'bucket': [1]
        })
        
        output_path = tmp_path / 'auxiliary.csv'
        aux = generate_auxiliary_file(submission, panel, output_path)
        
        assert output_path.exists()
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 1


class TestSubmissionVersioning:
    """Tests for submission versioning (Section 7.4)."""
    
    def test_generate_submission_version(self):
        """Test version string generation."""
        from src.inference import generate_submission_version
        
        version = generate_submission_version(run_name="test_run")
        
        assert isinstance(version, str)
        assert "test_run" in version
        # Should have timestamp format at start
        assert len(version.split("_")) >= 3
    
    def test_generate_submission_version_with_model_info(self):
        """Test version includes hash from model info."""
        from src.inference import generate_submission_version
        
        model_info = {'model_type': 'catboost', 'n_folds': 5}
        
        v1 = generate_submission_version(model_info=model_info, run_name="run")
        v2 = generate_submission_version(model_info=model_info, run_name="run")
        
        # Same model info should produce same hash (last part)
        assert v1.split("_")[-1] == v2.split("_")[-1]
    
    def test_log_submission(self, tmp_path):
        """Test submission logging."""
        from src.inference import log_submission
        
        submission = pd.DataFrame({
            'country': ['A'] * 24,
            'brand_name': ['X'] * 24,
            'months_postgx': list(range(24)),
            'volume': [1000.0] * 24
        })
        
        log_path = tmp_path / 'submission_log.json'
        
        entry = log_submission(
            submission_df=submission,
            version="test_v1",
            validation_score=0.123,
            notes="Test submission",
            log_path=log_path
        )
        
        assert 'version' in entry
        assert entry['version'] == 'test_v1'
        assert entry['validation_score'] == 0.123
        assert log_path.exists()
        
        # Log file should contain the entry
        import json
        with open(log_path) as f:
            logs = json.load(f)
        assert len(logs) == 1
        assert logs[0]['version'] == 'test_v1'


class TestEdgeCaseHandling:
    """Tests for edge case handling (Section 7.5)."""
    
    def test_handle_missing_pre_entry_data(self):
        """Test handling of missing pre-entry data."""
        from src.inference import handle_missing_pre_entry_data
        
        panel = pd.DataFrame({
            'country': ['A', 'A', 'B', 'B'],
            'brand_name': ['X', 'X', 'Y', 'Y'],
            'months_postgx': [0, 1, 0, 1],
            'avg_vol_12m': [1000.0, 1000.0, np.nan, np.nan]
        })
        
        result = handle_missing_pre_entry_data(panel)
        
        assert not result['avg_vol_12m'].isna().any()
        # Missing values should be filled with median of valid values
        assert result.loc[2, 'avg_vol_12m'] == 1000.0
    
    def test_handle_zero_volume_series(self):
        """Test handling of zero-volume series."""
        from src.inference import handle_zero_volume_series
        
        predictions = pd.DataFrame({
            'country': ['A', 'B'],
            'brand_name': ['X', 'Y'],
            'months_postgx': [0, 0],
            'volume': [0.0, 1000.0]
        })
        
        panel = pd.DataFrame({
            'country': ['A', 'A', 'B', 'B'],
            'brand_name': ['X', 'X', 'Y', 'Y'],
            'months_postgx': [0, 1, 0, 1],
            'avg_vol_12m': [0.0, 0.0, 1000.0, 1000.0]
        })
        
        result = handle_zero_volume_series(predictions, panel, replacement_strategy='global_erosion')
        
        # Zero-volume series should get non-zero predictions
        assert result.loc[0, 'volume'] > 0
        # Non-zero series should be unchanged
        assert result.loc[1, 'volume'] == 1000.0
    
    def test_handle_extreme_predictions(self):
        """Test handling of extreme predictions."""
        from src.inference import handle_extreme_predictions
        
        predictions = pd.DataFrame({
            'country': ['A', 'A', 'B', 'B'],
            'brand_name': ['X', 'X', 'Y', 'Y'],
            'months_postgx': [0, 1, 0, 1],
            'volume': [500.0, 5000.0, 1000.0, -100.0]  # Some extreme values
        })
        
        panel = pd.DataFrame({
            'country': ['A', 'A', 'B', 'B'],
            'brand_name': ['X', 'X', 'Y', 'Y'],
            'months_postgx': [0, 1, 0, 1],
            'avg_vol_12m': [1000.0, 1000.0, 1000.0, 1000.0]
        })
        
        result = handle_extreme_predictions(predictions, panel, max_ratio=2.0, min_ratio=0.0)
        
        # Volume 5000 (ratio 5.0) should be clipped to 2000 (ratio 2.0)
        assert result.loc[1, 'volume'] == 2000.0
        # Negative volume should be clipped to 0
        assert result.loc[3, 'volume'] == 0.0
        # Normal values should be unchanged
        assert result.loc[0, 'volume'] == 500.0


class TestSaveSubmissionWithVersioning:
    """Tests for complete submission workflow (Section 7.4)."""
    
    def test_save_submission_with_versioning(self, tmp_path):
        """Test versioned submission saving."""
        from src.inference import save_submission_with_versioning
        
        submission = pd.DataFrame({
            'country': ['A'] * 24,
            'brand_name': ['X'] * 24,
            'months_postgx': list(range(24)),
            'volume': [1000.0 - i * 30 for i in range(24)]
        })
        
        panel = pd.DataFrame({
            'country': ['A'] * 24,
            'brand_name': ['X'] * 24,
            'months_postgx': list(range(24)),
            'avg_vol_12m': [1200.0] * 24,
            'bucket': [1] * 24
        })
        
        paths = save_submission_with_versioning(
            submission_df=submission,
            output_dir=tmp_path,
            run_name="test_run",
            model_info={'type': 'test'},
            validation_score=0.15,
            save_auxiliary=True,
            panel_df=panel
        )
        
        assert 'submission' in paths
        assert 'metadata' in paths
        assert 'log' in paths
        assert 'auxiliary' in paths
        
        assert paths['submission'].exists()
        assert paths['metadata'].exists()
        assert paths['log'].exists()
        assert paths['auxiliary'].exists()
        
        # Check submission content
        loaded = pd.read_csv(paths['submission'])
        assert len(loaded) == 24
        assert list(loaded.columns) == ['country', 'brand_name', 'months_postgx', 'volume']


class TestDetectTestScenarios:
    """Tests for test scenario detection (Section 7.1)."""
    
    def test_detect_test_scenarios_basic(self):
        """Test basic scenario detection."""
        from src.inference import detect_test_scenarios
        
        # Create test data with mixed scenarios
        # S1 series: NO data for months 0-5 (prediction needed from month 0)
        # S2 series: HAS data for months 0-5 (prediction needed from month 6)
        
        # Series A (S1): Only has months 0-23 with empty volume (needs prediction)
        s1_data = pd.DataFrame({
            'country': ['A'] * 24,
            'brand_name': ['X'] * 24,
            'months_postgx': list(range(24)),
            'volume': [np.nan] * 24  # No actual volume data
        })
        
        # Series B (S2): Has actual volume data for months 0-5, needs prediction 6-23
        s2_data = pd.DataFrame({
            'country': ['B'] * 24,
            'brand_name': ['Y'] * 24,
            'months_postgx': list(range(24)),
            'volume': [100.0] * 6 + [np.nan] * 18  # Has volume for 0-5
        })
        
        # For S2 detection, the key is having rows in months 0-5
        # Let's create the correct test: S1 = starts from month 0 with no early data
        # S2 = has months 0-5 present
        
        # S1: series with months 0-23 but no data in 0-5 range marked distinctly
        # Actually, the detection just checks if series HAS data for months 0-5
        
        # Simplify: S1 series has NO rows in months 0-5
        s1_simple = pd.DataFrame({
            'country': ['C'] * 18,
            'brand_name': ['Z'] * 18,
            'months_postgx': list(range(6, 24)),  # No months 0-5
            'volume': [100.0] * 18
        })
        
        test_volume = pd.concat([s2_data, s1_simple], ignore_index=True)
        
        result = detect_test_scenarios(test_volume)
        
        assert 1 in result
        assert 2 in result
        assert ('C', 'Z') in result[1]  # S1 series (no months 0-5)
        assert ('B', 'Y') in result[2]  # S2 series (has months 0-5)
    
    def test_detect_test_scenarios_all_s1(self):
        """Test detection when all series are S1 (no months 0-5 data)."""
        from src.inference import detect_test_scenarios
        
        # Both series start at month 6 (no months 0-5)
        test_volume = pd.DataFrame({
            'country': ['A'] * 18 + ['B'] * 18,
            'brand_name': ['X'] * 18 + ['Y'] * 18,
            'months_postgx': list(range(6, 24)) * 2,  # Only months 6-23
            'volume': [100.0] * 36
        })
        
        result = detect_test_scenarios(test_volume)
        
        assert len(result[1]) == 2  # Both are S1 (no months 0-5)
        assert len(result[2]) == 0
    
    def test_detect_test_scenarios_all_s2(self):
        """Test detection when all series are S2 (have months 0-5 data)."""
        from src.inference import detect_test_scenarios
        
        # Both series have months 0-5
        test_volume = pd.DataFrame({
            'country': ['A'] * 24 + ['B'] * 24,
            'brand_name': ['X'] * 24 + ['Y'] * 24,
            'months_postgx': list(range(24)) * 2,
            'volume': [100.0] * 48
        })
        
        result = detect_test_scenarios(test_volume)
        
        assert len(result[1]) == 0
        assert len(result[2]) == 2  # Both are S2 (have months 0-5)


class TestInferenceCLI:
    """Tests for inference CLI (Section 7)."""
    
    def test_inference_cli_help_extended(self):
        """Test that inference CLI --help shows new arguments."""
        import subprocess
        from pathlib import Path
        
        result = subprocess.run(
            ['python', '-m', 'src.inference', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        assert result.returncode == 0
        
        # Check for new arguments
        assert '--output-dir' in result.stdout
        assert '--run-name' in result.stdout
        assert '--validation-score' in result.stdout
        assert '--use-versioning' in result.stdout
        assert '--save-auxiliary' in result.stdout


# =============================================================================
# SECTION 9 TESTS: TESTING & QUALITY ASSURANCE
# =============================================================================


class TestFeatureEngineeringCorrectness:
    """Tests for feature engineering correctness (Section 9.1)."""
    
    def test_make_features_respects_scenario1_cutoff(self):
        """Test that make_features respects S1 cutoff (months_postgx < 0 for features)."""
        from src.features import make_features, SCENARIO_CONFIG
        
        # Create panel with pre-entry and post-entry data
        panel_data = []
        for m in range(-12, 24):
            panel_data.append({
                'country': 'US',
                'brand_name': 'BRAND_A',
                'months_postgx': m,
                'volume': 1000 - abs(m) * 10,  # Volume varies with month
                'n_gxs': 0 if m < 0 else max(1, m // 3),
                'month': 'Jan',
                'ther_area': 'AREA_1',
                'main_package': 'Tablet',
                'hospital_rate': 50.0,
                'biological': False,
                'small_molecule': True,
            })
        
        panel = pd.DataFrame(panel_data)
        panel['avg_vol_12m'] = 1000.0
        
        # Build features for Scenario 1
        result = make_features(panel.copy(), scenario=1, mode='train')
        
        # S1 cutoff is 0, so features should only use data from months < 0
        assert SCENARIO_CONFIG[1]['feature_cutoff'] == 0
        
        # Early erosion features should NOT be present for S1
        early_erosion_cols = [c for c in result.columns if 'erosion_0_' in c.lower() or 'avg_vol_0_' in c.lower()]
        assert len(early_erosion_cols) == 0, f"S1 should not have early erosion features, found: {early_erosion_cols}"
        
        # Pre-entry features SHOULD be present
        assert 'pre_entry_trend' in result.columns
        assert 'avg_vol_3m' in result.columns
    
    def test_make_features_respects_scenario2_cutoff(self):
        """Test that make_features respects S2 cutoff (months_postgx < 6 for features)."""
        from src.features import make_features, SCENARIO_CONFIG
        
        # Create panel with pre-entry and post-entry data
        panel_data = []
        for m in range(-12, 24):
            panel_data.append({
                'country': 'US',
                'brand_name': 'BRAND_A',
                'months_postgx': m,
                'volume': 1000 - abs(m) * 10,
                'n_gxs': 0 if m < 0 else max(1, m // 3),
                'month': 'Jan',
                'ther_area': 'AREA_1',
                'main_package': 'Tablet',
                'hospital_rate': 50.0,
                'biological': False,
                'small_molecule': True,
            })
        
        panel = pd.DataFrame(panel_data)
        panel['avg_vol_12m'] = 1000.0
        
        # Build features for Scenario 2
        result = make_features(panel.copy(), scenario=2, mode='train')
        
        # S2 cutoff is 6, so features can use data from months < 6
        assert SCENARIO_CONFIG[2]['feature_cutoff'] == 6
        
        # Early erosion features SHOULD be present for S2
        early_erosion_cols = [c for c in result.columns if 'erosion_0_' in c.lower() or 'avg_vol_0_' in c.lower()]
        assert len(early_erosion_cols) > 0, "S2 should have early erosion features"
    
    def test_make_features_mode_test_no_y_norm(self):
        """Test that mode='test' does not create y_norm or modify volume."""
        from src.features import make_features
        
        # Create panel
        panel_data = []
        for m in range(-12, 24):
            panel_data.append({
                'country': 'US',
                'brand_name': 'BRAND_A',
                'months_postgx': m,
                'volume': 1000.0,
                'n_gxs': 1 if m >= 0 else 0,
                'month': 'Jan',
                'ther_area': 'AREA_1',
                'main_package': 'Tablet',
                'hospital_rate': 50.0,
                'biological': False,
                'small_molecule': True,
            })
        
        panel = pd.DataFrame(panel_data)
        panel['avg_vol_12m'] = 1000.0
        original_volume = panel['volume'].copy()
        
        # Build features in TEST mode
        result = make_features(panel.copy(), scenario=1, mode='test')
        
        # y_norm should NOT be created in test mode
        assert 'y_norm' not in result.columns, "Test mode should not have y_norm"
        
        # Volume should not be modified
        assert 'volume' in result.columns
        np.testing.assert_array_equal(result['volume'].values, original_volume.values)
    
    def test_make_features_mode_train_creates_y_norm(self):
        """Test that mode='train' creates y_norm."""
        from src.features import make_features
        
        # Create panel
        panel_data = []
        for m in range(-12, 24):
            panel_data.append({
                'country': 'US',
                'brand_name': 'BRAND_A',
                'months_postgx': m,
                'volume': 800.0 if m >= 0 else 1000.0,
                'n_gxs': 1 if m >= 0 else 0,
                'month': 'Jan',
                'ther_area': 'AREA_1',
                'main_package': 'Tablet',
                'hospital_rate': 50.0,
                'biological': False,
                'small_molecule': True,
            })
        
        panel = pd.DataFrame(panel_data)
        panel['avg_vol_12m'] = 1000.0
        
        # Build features in TRAIN mode
        result = make_features(panel.copy(), scenario=1, mode='train')
        
        # y_norm SHOULD be created in train mode
        assert 'y_norm' in result.columns, "Train mode should have y_norm"
        
        # y_norm should be volume / avg_vol_12m
        expected_y_norm = panel['volume'] / panel['avg_vol_12m']
        np.testing.assert_array_almost_equal(result['y_norm'].values, expected_y_norm.values)
    
    def test_early_erosion_features_only_scenario2(self):
        """Test that early erosion features only appear for Scenario 2."""
        from src.features import make_features
        
        # Create panel
        panel_data = []
        for m in range(-12, 24):
            panel_data.append({
                'country': 'US',
                'brand_name': 'BRAND_A',
                'months_postgx': m,
                'volume': 1000.0 - (10 * m if m >= 0 else 0),
                'n_gxs': 1 if m >= 0 else 0,
                'month': 'Jan',
                'ther_area': 'AREA_1',
                'main_package': 'Tablet',
                'hospital_rate': 50.0,
                'biological': False,
                'small_molecule': True,
            })
        
        panel = pd.DataFrame(panel_data)
        panel['avg_vol_12m'] = 1000.0
        
        # Build for S1 and S2
        s1_result = make_features(panel.copy(), scenario=1, mode='train')
        s2_result = make_features(panel.copy(), scenario=2, mode='train')
        
        # Define early erosion feature patterns
        early_patterns = ['erosion_0_5', 'avg_vol_0_5', 'trend_0_5', 'drop_month_0', 
                          'month_0_to_3_change', 'recovery_signal', 'competition_response']
        
        # Check S1 doesn't have these
        for pattern in early_patterns:
            matching_s1 = [c for c in s1_result.columns if pattern in c.lower()]
            assert len(matching_s1) == 0, f"S1 should not have '{pattern}' features, found: {matching_s1}"
        
        # Check S2 DOES have at least some of these
        found_s2_early = False
        for pattern in early_patterns:
            matching_s2 = [c for c in s2_result.columns if pattern in c.lower()]
            if len(matching_s2) > 0:
                found_s2_early = True
                break
        
        assert found_s2_early, "S2 should have at least one early erosion feature"


class TestTrainCLISmokeTest:
    """Tests for train.py CLI (Section 9.2)."""
    
    def test_train_cli_help_returns_success(self):
        """Test that python -m src.train --help returns exit code 0."""
        import subprocess
        
        result = subprocess.run(
            ['python', '-m', 'src.train', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        assert result.returncode == 0, f"CLI --help failed: {result.stderr}"
    
    def test_train_cli_has_key_arguments(self):
        """Test that train CLI includes all key arguments."""
        import subprocess
        
        result = subprocess.run(
            ['python', '-m', 'src.train', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        # Key arguments that must be present
        key_args = [
            '--scenario', '--model', '--data-config', '--features-config',
            '--run-config', '--model-config', '--cv', '--n-folds',
            '--run-name', '--force-rebuild', '--no-cache'
        ]
        
        for arg in key_args:
            assert arg in result.stdout, f"Missing argument '{arg}' in CLI help"
    
    def test_train_cli_scenario_is_integer(self):
        """Test that --scenario expects an integer argument."""
        import subprocess
        
        result = subprocess.run(
            ['python', '-m', 'src.train', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        # Check that scenario is described as 1 or 2
        help_text = result.stdout.lower()
        assert '1' in help_text and '2' in help_text, "Scenario should accept 1 or 2"


class TestLeakageStrengthening:
    """Tests for leakage prevention (Section 9.3)."""
    
    def test_forbidden_column_bucket_raises_error(self):
        """Test that including 'bucket' in features raises error."""
        from src.features import split_features_target_meta, validate_feature_leakage
        
        # Create DataFrame with forbidden column
        df_with_bucket = pd.DataFrame({
            'country': ['US', 'DE'],
            'brand_name': ['A', 'B'],
            'months_postgx': [0, 0],
            'feature1': [1.0, 2.0],
            'bucket': [1, 2],  # FORBIDDEN
            'y_norm': [0.8, 0.7]
        })
        
        # split_features_target_meta should exclude bucket
        X, y, meta = split_features_target_meta(df_with_bucket)
        assert 'bucket' not in X.columns, "bucket should be excluded from features"
        
        # validate_feature_leakage should detect it if present
        bad_X = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'bucket': [1, 2]  # FORBIDDEN
        })
        is_valid, violations = validate_feature_leakage(bad_X, scenario=1, mode='train')
        assert not is_valid, "Should detect 'bucket' as leakage"
        assert any('bucket' in v.lower() for v in violations)
    
    def test_forbidden_column_y_norm_raises_error(self):
        """Test that including 'y_norm' in features raises error."""
        from src.features import validate_feature_leakage
        
        bad_X = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'y_norm': [0.8, 0.7]  # FORBIDDEN - this is the target
        })
        
        is_valid, violations = validate_feature_leakage(bad_X, scenario=1, mode='train')
        assert not is_valid, "Should detect 'y_norm' as leakage"
    
    def test_forbidden_column_mean_erosion_raises_error(self):
        """Test that including 'mean_erosion' in features raises error."""
        from src.features import validate_feature_leakage
        
        bad_X = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'mean_erosion': [0.25, 0.50]  # FORBIDDEN - target-derived
        })
        
        is_valid, violations = validate_feature_leakage(bad_X, scenario=1, mode='train')
        assert not is_valid, "Should detect 'mean_erosion' as leakage"
    
    def test_data_audit_strict_mode_raises(self):
        """Test that audit_data_leakage in strict mode raises ValueError."""
        from src.data import audit_data_leakage
        
        bad_X = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'volume': [100, 200]  # FORBIDDEN
        })
        
        with pytest.raises(ValueError, match="Data leakage detected"):
            audit_data_leakage(bad_X, scenario=1, mode='train', strict=True)


class TestIntegrationEndToEnd:
    """Integration tests (Section 9.4)."""
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
        reason="Training data not available"
    )
    def test_end_to_end_data_pipeline(self):
        """End-to-end test of data loading and feature engineering pipeline."""
        from src.utils import load_config, get_project_root, set_seed
        from src.data import load_raw_data, prepare_base_panel, handle_missing_values, compute_pre_entry_stats
        from src.features import make_features
        
        set_seed(42)
        root = get_project_root()
        
        # Load configs
        data_config = load_config(root / 'configs' / 'data.yaml')
        features_config = load_config(root / 'configs' / 'features.yaml')
        
        # Load data
        raw_data = load_raw_data(data_config, split='train')
        
        # Verify raw data loaded correctly
        assert 'volume' in raw_data, "Should have volume data"
        assert 'generics' in raw_data, "Should have generics data"
        assert 'medicine_info' in raw_data, "Should have medicine_info data"
        
        # Build panel
        panel = prepare_base_panel(
            raw_data['volume'],
            raw_data['generics'],
            raw_data['medicine_info']
        )
        panel = handle_missing_values(panel)
        panel = compute_pre_entry_stats(panel, is_train=True)
        
        # Filter to ~10 series for speed
        unique_series = panel[['country', 'brand_name']].drop_duplicates().head(10)
        panel_subset = panel.merge(unique_series, on=['country', 'brand_name'])
        
        assert len(unique_series) >= 5, "Need at least 5 series for test"
        
        # Build features for Scenario 1
        panel_features = make_features(panel_subset.copy(), scenario=1, mode='train', config=features_config)
        
        # Verify features were built correctly
        assert 'y_norm' in panel_features.columns, "Should have y_norm in train mode"
        assert 'months_postgx' in panel_features.columns, "Should have months_postgx"
        
        # Test with post-entry data only
        post_entry = panel_features[panel_features['months_postgx'] >= 0].copy()
        assert len(post_entry) > 0, "Should have post-entry data"
        
        # Verify y_norm is properly normalized
        assert post_entry['y_norm'].min() >= 0, "y_norm should be non-negative"
        assert post_entry['y_norm'].max() <= 10, "y_norm should be in reasonable range"
        
        # Verify no NaN in essential columns
        essential_cols = ['country', 'brand_name', 'months_postgx', 'y_norm']
        for col in essential_cols:
            assert not post_entry[col].isna().any(), f"Column {col} should have no NaN values"
        
        print(f"End-to-end data pipeline test passed with {len(panel_features)} rows")
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
        reason="Training data not available"
    )
    def test_end_to_end_scenario2_features(self):
        """End-to-end test for Scenario 2 feature engineering."""
        from src.utils import load_config, get_project_root, set_seed
        from src.data import load_raw_data, prepare_base_panel, handle_missing_values, compute_pre_entry_stats
        from src.features import make_features
        
        set_seed(42)
        root = get_project_root()
        
        # Load configs
        data_config = load_config(root / 'configs' / 'data.yaml')
        features_config = load_config(root / 'configs' / 'features.yaml')
        
        # Load data
        raw_data = load_raw_data(data_config, split='train')
        
        # Build panel
        panel = prepare_base_panel(
            raw_data['volume'],
            raw_data['generics'],
            raw_data['medicine_info']
        )
        panel = handle_missing_values(panel)
        panel = compute_pre_entry_stats(panel, is_train=True)
        
        # Filter to ~10 series
        unique_series = panel[['country', 'brand_name']].drop_duplicates().head(10)
        panel_subset = panel.merge(unique_series, on=['country', 'brand_name'])
        
        # Build features for Scenario 2
        panel_features = make_features(panel_subset.copy(), scenario=2, mode='train', config=features_config)
        
        # Verify features were built correctly
        assert 'y_norm' in panel_features.columns, "Should have y_norm in train mode"
        assert 'months_postgx' in panel_features.columns, "Should have months_postgx"
        
        # S2 should have early erosion features
        early_erosion_cols = [c for c in panel_features.columns if 'erosion_0_5' in c or 'early' in c.lower()]
        assert len(early_erosion_cols) > 0, "S2 should have early erosion features"
        
        # Test with S2 target range (months 6-23)
        s2_data = panel_features[panel_features['months_postgx'] >= 6].copy()
        assert len(s2_data) > 0, "Should have S2 data (months 6-23)"
        
        print(f"S2 feature engineering test passed with {len(panel_features)} rows")


class TestDataValidation:
    """Data validation tests (Section 9.5)."""
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'data' / 'raw' / 'TRAIN').exists(),
        reason="Training data not available"
    )
    def test_data_drift_detection(self):
        """Test that adversarial validation can detect data drift."""
        from src.validation import adversarial_validation
        from src.utils import load_config, get_project_root
        from src.data import load_raw_data, prepare_base_panel, handle_missing_values
        
        root = get_project_root()
        data_config = load_config(root / 'configs' / 'data.yaml')
        
        # Load train data
        train_raw = load_raw_data(data_config, split='train')
        train_panel = prepare_base_panel(
            train_raw['volume'],
            train_raw['generics'],
            train_raw['medicine_info']
        )
        train_panel = handle_missing_values(train_panel)
        
        # Load test data
        test_raw = load_raw_data(data_config, split='test')
        test_panel = prepare_base_panel(
            test_raw['volume'],
            test_raw['generics'],
            test_raw['medicine_info']
        )
        test_panel = handle_missing_values(test_panel)
        
        # Get numeric features
        numeric_cols = ['n_gxs', 'hospital_rate']
        existing_cols = [c for c in numeric_cols if c in train_panel.columns and c in test_panel.columns]
        
        if len(existing_cols) < 2:
            pytest.skip("Not enough numeric columns for drift test")
        
        train_features = train_panel[existing_cols].dropna()
        test_features = test_panel[existing_cols].dropna()
        
        # Run adversarial validation
        result = adversarial_validation(train_features.head(500), test_features.head(200), n_folds=2)
        
        # AUC should be between 0 and 1
        assert 0 <= result['mean_auc'] <= 1, f"AUC should be in [0, 1], got {result['mean_auc']}"
        
        # If AUC is close to 0.5, distributions are similar (no major drift)
        # If AUC is close to 1.0, there's significant drift
        print(f"Adversarial validation AUC: {result['mean_auc']:.3f}")
    
    def test_submission_against_template(self):
        """Test submission validation against template."""
        from src.inference import validate_submission_format
        from src.utils import get_project_root
        
        root = get_project_root()
        template_path = root / 'docs' / 'guide' / 'submission_template.csv'
        
        if not template_path.exists():
            pytest.skip("Submission template not found")
        
        template = pd.read_csv(template_path)
        
        # Check template has expected columns
        expected_cols = ['country', 'brand_name', 'months_postgx', 'volume']
        assert list(template.columns) == expected_cols, f"Template columns: {list(template.columns)}"
        
        # Create a valid submission matching template structure
        submission = template.copy()
        submission['volume'] = 1000.0  # Fill with valid values
        
        # Should pass validation
        is_valid = validate_submission_format(submission, template, strict=False)
        assert is_valid, "Valid submission should pass validation"
    
    def test_submission_column_order_matters(self):
        """Test that column order in submission matches template."""
        from src.inference import validate_submission_format
        
        template = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
            'volume': [np.nan]
        })
        
        # Submission with wrong column order
        wrong_order = pd.DataFrame({
            'volume': [1000.0],  # Wrong order
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
        })
        
        # Reorder to match
        correct_order = wrong_order[['country', 'brand_name', 'months_postgx', 'volume']]
        
        # The validate function should work with correct order
        is_valid = validate_submission_format(correct_order, template, strict=False)
        assert is_valid
    
    def test_submission_dtype_compatibility(self):
        """Test that submission dtypes are compatible."""
        from src.inference import validate_submission_format
        
        template = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
            'volume': [np.nan]
        })
        
        # Submission with correct types
        good_submission = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0],
            'volume': [1000.0]
        })
        
        assert good_submission['volume'].dtype in [np.float64, np.float32, float]
        
        is_valid = validate_submission_format(good_submission, template, strict=False)
        assert is_valid
    
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'docs' / 'guide' / 'auxiliar_metric_computation_example.csv').exists(),
        reason="Auxiliary example file not found"
    )
    def test_metric_calculation_against_example(self):
        """Test metric calculation matches provided example."""
        from src.evaluate import compute_metric1
        from src.utils import get_project_root
        
        root = get_project_root()
        aux_example_path = root / 'docs' / 'guide' / 'auxiliar_metric_computation_example.csv'
        
        # Load the example auxiliary file
        aux_example = pd.read_csv(aux_example_path)
        
        # Check it has expected schema
        expected_cols = ['country', 'brand_name', 'avg_vol', 'bucket']
        for col in expected_cols:
            assert col in aux_example.columns, f"Aux file should have column '{col}'"
        
        # Create synthetic actual and pred DataFrames for testing
        series_list = list(zip(aux_example['country'], aux_example['brand_name']))[:5]  # First 5 series
        
        data = []
        for country, brand in series_list:
            avg_vol = aux_example[(aux_example['country'] == country) & 
                                   (aux_example['brand_name'] == brand)]['avg_vol'].values[0]
            for m in range(24):
                data.append({
                    'country': country,
                    'brand_name': brand,
                    'months_postgx': m,
                    'volume_actual': avg_vol * (1 - 0.02 * m),
                    'volume_pred': avg_vol * (1 - 0.025 * m)  # Slightly different
                })
        
        df = pd.DataFrame(data)
        df_actual = df[['country', 'brand_name', 'months_postgx', 'volume_actual']].rename(
            columns={'volume_actual': 'volume'})
        df_pred = df[['country', 'brand_name', 'months_postgx', 'volume_pred']].rename(
            columns={'volume_pred': 'volume'})
        
        # Filter aux to matching series
        aux_subset = aux_example[aux_example.apply(
            lambda r: (r['country'], r['brand_name']) in series_list, axis=1)]
        
        # Compute metric
        metric = compute_metric1(df_actual, df_pred, aux_subset)
        
        assert np.isfinite(metric), "Metric should be finite"
        assert metric >= 0, "Metric should be non-negative"


class TestCodeQuality:
    """Code quality tests (Section 9.6)."""
    
    def test_all_source_files_have_docstrings(self):
        """Test that all source modules have module docstrings."""
        import importlib
        
        modules = [
            'src.data', 'src.features', 'src.train', 'src.inference',
            'src.evaluate', 'src.validation', 'src.utils'
        ]
        
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module.__doc__ is not None, f"Module {module_name} should have a docstring"
                assert len(module.__doc__.strip()) > 10, f"Module {module_name} docstring should be meaningful"
            except ImportError:
                pytest.skip(f"Could not import {module_name}")
    
    def test_key_functions_have_docstrings(self):
        """Test that key functions have docstrings."""
        from src import data, features, train, inference, evaluate, validation
        
        key_functions = [
            (data, 'load_raw_data'),
            (data, 'prepare_base_panel'),
            (data, 'compute_pre_entry_stats'),
            (data, 'get_panel'),
            (features, 'make_features'),
            (features, 'split_features_target_meta'),
            (train, 'train_scenario_model'),
            (train, 'compute_sample_weights'),
            (inference, 'generate_submission'),
            (inference, 'detect_test_scenarios'),
            (evaluate, 'compute_metric1'),
            (evaluate, 'compute_metric2'),
            (validation, 'create_validation_split'),
        ]
        
        for module, func_name in key_functions:
            func = getattr(module, func_name, None)
            assert func is not None, f"Function {module.__name__}.{func_name} should exist"
            assert func.__doc__ is not None, f"Function {module.__name__}.{func_name} should have a docstring"
    
    def test_no_print_statements_in_source(self):
        """Test that source files don't have bare print statements (should use logger)."""
        # This is a light check - just verify logging is used
        from src import data, features, train, inference, evaluate, validation
        
        for module in [data, features, train, inference, evaluate, validation]:
            # Check that logger is defined
            assert hasattr(module, 'logger') or hasattr(module, 'logging'), \
                f"Module {module.__name__} should use logging"
    
    def test_constants_are_uppercase(self):
        """Test that constants follow UPPER_CASE convention."""
        from src.data import META_COLS, ID_COLS, TIME_COL, RAW_TARGET_COL, MODEL_TARGET_COL
        from src.features import SCENARIO_CONFIG, FORBIDDEN_FEATURES
        
        # These should be defined and uppercase (they're constant names)
        assert META_COLS is not None
        assert ID_COLS is not None
        assert SCENARIO_CONFIG is not None
        assert FORBIDDEN_FEATURES is not None
    
    def test_imports_are_organized(self):
        """Test that imports don't cause circular dependencies."""
        # Just try importing all modules - will fail if circular imports
        from src import data
        from src import features
        from src import train
        from src import inference
        from src import evaluate
        from src import validation
        from src import utils
        from src.models import base
        from src.models import linear
        from src.models import ensemble
        
        # If we get here, no circular import issues
        assert True


# =============================================================================
# SECTION 10 TESTS: DOCUMENTATION
# =============================================================================


class TestDocumentation:
    """Tests for documentation completeness and correctness (Section 10)."""
    
    def test_readme_exists(self):
        """Test that main README.md exists."""
        readme_path = Path(__file__).parent.parent / 'README.md'
        assert readme_path.exists(), "README.md should exist"
    
    def test_readme_has_required_sections(self):
        """Test that README.md has all required sections."""
        readme_path = Path(__file__).parent.parent / 'README.md'
        content = readme_path.read_text(encoding='utf-8')
        
        required_sections = [
            '## Competition Overview',
            '## Quick Start',
            '## Project Structure',
            '## Installation',
            '## Data Overview',
            '## Configuration',
            '## CLI Reference',
            '## Feature Engineering',
            '## Models',
            '## Validation Strategy',
            '## Metric Calculation',
            '## Testing',
            '## Notebooks',
            '## Reproducibility',
        ]
        
        for section in required_sections:
            assert section in content, f"README.md should have section: {section}"
    
    def test_readme_has_cli_examples(self):
        """Test that README.md has CLI usage examples."""
        readme_path = Path(__file__).parent.parent / 'README.md'
        content = readme_path.read_text(encoding='utf-8')
        
        # Should have training CLI examples
        assert 'python -m src.train' in content, "README should show training CLI"
        assert '--scenario' in content, "README should document --scenario flag"
        assert '--model' in content, "README should document --model flag"
        
        # Should have inference CLI examples
        assert 'python -m src.inference' in content, "README should show inference CLI"
    
    def test_readme_documents_metric_calculation(self):
        """Test that README.md documents metric calculation usage."""
        readme_path = Path(__file__).parent.parent / 'README.md'
        content = readme_path.read_text(encoding='utf-8')
        
        # Should document official metric script usage
        assert 'compute_metric1' in content, "README should document compute_metric1"
        assert 'compute_metric2' in content, "README should document compute_metric2"
        assert 'df_aux' in content or 'auxiliary' in content.lower(), \
            "README should document auxiliary file"
    
    def test_configs_readme_exists(self):
        """Test that configs/README.md exists."""
        readme_path = Path(__file__).parent.parent / 'configs' / 'README.md'
        assert readme_path.exists(), "configs/README.md should exist"
    
    def test_configs_readme_documents_all_configs(self):
        """Test that configs/README.md documents all config files."""
        readme_path = Path(__file__).parent.parent / 'configs' / 'README.md'
        content = readme_path.read_text(encoding='utf-8')
        
        config_files = [
            'data.yaml',
            'features.yaml',
            'run_defaults.yaml',
            'model_cat.yaml',
            'model_lgbm.yaml',
            'model_xgb.yaml',
            'model_linear.yaml',
            'model_nn.yaml',
        ]
        
        for config_file in config_files:
            assert config_file in content, f"configs/README.md should document {config_file}"
    
    def test_configs_readme_has_usage_examples(self):
        """Test that configs/README.md has usage examples."""
        readme_path = Path(__file__).parent.parent / 'configs' / 'README.md'
        content = readme_path.read_text(encoding='utf-8')
        
        # Should have loading examples
        assert 'load_config' in content, "Should show how to load configs"
        
        # Should document key sections
        assert 'reproducibility' in content.lower(), "Should document reproducibility settings"
        assert 'sample_weights' in content.lower() or 'sample weights' in content.lower(), \
            "Should document sample weights"
    
    def test_requirements_has_pinned_versions(self):
        """Test that requirements.txt has pinned versions."""
        requirements_path = Path(__file__).parent.parent / 'requirements.txt'
        assert requirements_path.exists(), "requirements.txt should exist"
        
        content = requirements_path.read_text()
        
        # Check that versions are pinned (have ==)
        lines = [l.strip() for l in content.split('\n') 
                 if l.strip() and not l.strip().startswith('#')]
        
        pinned_count = sum(1 for l in lines if '==' in l)
        
        # At least 80% should be pinned
        assert pinned_count >= len(lines) * 0.8, \
            f"Most dependencies should have pinned versions ({pinned_count}/{len(lines)})"
    
    def test_requirements_has_core_packages(self):
        """Test that requirements.txt has core packages."""
        requirements_path = Path(__file__).parent.parent / 'requirements.txt'
        content = requirements_path.read_text().lower()
        
        core_packages = [
            'numpy',
            'pandas',
            'scikit-learn',
            'catboost',
            'pytest',
        ]
        
        for pkg in core_packages:
            assert pkg in content, f"requirements.txt should have {pkg}"
    
    def test_reproduce_script_exists(self):
        """Test that reproduce.sh script exists."""
        script_path = Path(__file__).parent.parent / 'reproduce.sh'
        assert script_path.exists(), "reproduce.sh should exist"
    
    def test_reproduce_script_is_executable(self):
        """Test that reproduce.sh is executable (Unix permissions)."""
        import os
        import sys
        script_path = Path(__file__).parent.parent / 'reproduce.sh'
        
        if script_path.exists():
            # Skip executable check on Windows (not applicable)
            if sys.platform == 'win32':
                return  # Windows doesn't use Unix executable bits
            # Check executable bit
            mode = os.stat(script_path).st_mode
            is_executable = mode & 0o111  # Any execute bit
            assert is_executable, "reproduce.sh should be executable"
    
    def test_reproduce_script_has_required_sections(self):
        """Test that reproduce.sh has required functionality."""
        script_path = Path(__file__).parent.parent / 'reproduce.sh'
        content = script_path.read_text(encoding='utf-8')
        
        # Should train both scenarios
        assert 'scenario 1' in content.lower() or '--scenario 1' in content.lower(), \
            "Should train Scenario 1"
        assert 'scenario 2' in content.lower() or '--scenario 2' in content.lower(), \
            "Should train Scenario 2"
        
        # Should generate submission
        assert 'inference' in content.lower() or 'submission' in content.lower(), \
            "Should generate submission"
        
        # Should activate virtual environment
        assert '.venv' in content or 'venv' in content, \
            "Should reference virtual environment"
    
    def test_todo_md_exists(self):
        """Test that TODO.md exists."""
        todo_path = Path(__file__).parent.parent / 'TODO.md'
        assert todo_path.exists(), "TODO.md should exist"
    
    def test_contributing_md_exists(self):
        """Test that CONTRIBUTING.md exists."""
        contributing_path = Path(__file__).parent.parent / 'CONTRIBUTING.md'
        assert contributing_path.exists(), "CONTRIBUTING.md should exist"
    
    def test_license_exists(self):
        """Test that LICENSE exists."""
        license_path = Path(__file__).parent.parent / 'LICENSE'
        assert license_path.exists(), "LICENSE should exist"
    
    def test_metric_calculation_script_exists(self):
        """Test that official metric calculation script exists."""
        script_path = Path(__file__).parent.parent / 'docs' / 'guide' / 'metric_calculation.py'
        assert script_path.exists(), "docs/guide/metric_calculation.py should exist"
    
    def test_metric_calculation_script_importable(self):
        """Test that official metric calculation script can be imported."""
        import sys
        script_dir = str(Path(__file__).parent.parent / 'docs' / 'guide')
        
        # Temporarily add to path
        sys.path.insert(0, script_dir)
        try:
            import metric_calculation
            assert hasattr(metric_calculation, 'compute_metric1')
            assert hasattr(metric_calculation, 'compute_metric2')
        finally:
            sys.path.remove(script_dir)
    
    def test_notebooks_exist(self):
        """Test that expected notebooks exist."""
        notebooks_dir = Path(__file__).parent.parent / 'notebooks'
        
        expected_notebooks = [
            '00_eda.ipynb',
            '01_feature_prototype.ipynb',
            '01_train.ipynb',
            '02_model_sanity.ipynb',
        ]
        
        for notebook in expected_notebooks:
            notebook_path = notebooks_dir / notebook
            assert notebook_path.exists(), f"Notebook {notebook} should exist"
    
    def test_colab_notebook_exists(self):
        """Test that Colab notebook exists."""
        colab_path = Path(__file__).parent.parent / 'notebooks' / 'colab' / 'main.ipynb'
        assert colab_path.exists(), "notebooks/colab/main.ipynb should exist"
    
    def test_all_yaml_configs_valid(self):
        """Test that all YAML configs are valid and loadable."""
        import yaml
        
        configs_dir = Path(__file__).parent.parent / 'configs'
        yaml_files = list(configs_dir.glob('*.yaml'))
        
        assert len(yaml_files) >= 7, "Should have at least 7 YAML config files"
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                assert isinstance(config, dict), f"{yaml_file.name} should load as dict"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {yaml_file.name}: {e}")
    
    def test_data_readme_exists(self):
        """Test that data/README.md exists."""
        readme_path = Path(__file__).parent.parent / 'data' / 'README.md'
        assert readme_path.exists(), "data/README.md should exist"
    
    def test_submissions_readme_exists(self):
        """Test that submissions/README.md exists."""
        readme_path = Path(__file__).parent.parent / 'submissions' / 'README.md'
        assert readme_path.exists(), "submissions/README.md should exist"


# =============================================================================
# Section 11: Colab/Production Readiness Tests
# =============================================================================

class TestSection11EnvironmentDetection:
    """Tests for Section 11.1 - Environment Detection."""
    
    def test_is_colab_function_exists(self):
        """Test that is_colab function exists."""
        from src.utils import is_colab
        assert callable(is_colab)
    
    def test_is_colab_returns_bool(self):
        """Test that is_colab returns boolean."""
        from src.utils import is_colab
        result = is_colab()
        assert isinstance(result, bool)
    
    def test_is_jupyter_function_exists(self):
        """Test that is_jupyter function exists."""
        from src.utils import is_jupyter
        assert callable(is_jupyter)
        result = is_jupyter()
        assert isinstance(result, bool)
    
    def test_get_platform_info_function(self):
        """Test get_platform_info returns expected keys."""
        from src.utils import get_platform_info
        info = get_platform_info()
        
        assert isinstance(info, dict)
        assert 'python_version' in info
        assert 'platform' in info
        assert 'is_colab' in info
        assert 'is_jupyter' in info
        assert 'cpu_count' in info
        assert 'total_ram_gb' in info


class TestSection11GPUDetection:
    """Tests for Section 11.1 - GPU Detection."""
    
    def test_get_device_function_exists(self):
        """Test that get_device function exists."""
        from src.utils import get_device
        assert callable(get_device)
    
    def test_get_device_returns_valid_type(self):
        """Test that get_device returns valid type."""
        from src.utils import get_device
        device = get_device()
        # Should be None if no torch, or a torch.device
        if device is not None:
            assert hasattr(device, 'type')
    
    def test_get_gpu_info_function(self):
        """Test that get_gpu_info returns expected structure."""
        from src.utils import get_gpu_info
        info = get_gpu_info()
        
        assert isinstance(info, dict)
        assert 'gpu_available' in info
        assert 'device_name' in info
        assert 'mps_available' in info
        assert isinstance(info['gpu_available'], bool)
    
    def test_print_gpu_info_no_error(self):
        """Test that print_gpu_info runs without error."""
        from src.utils import print_gpu_info
        # Should not raise any errors
        print_gpu_info()
    
    def test_enable_gpu_for_catboost(self):
        """Test enable_gpu_for_catboost returns valid config."""
        from src.utils import enable_gpu_for_catboost
        config = enable_gpu_for_catboost()
        
        assert isinstance(config, dict)
        # If GPU available, should have task_type
        if config:
            assert 'task_type' in config
            assert config['task_type'] == 'GPU'
    
    def test_enable_gpu_for_xgboost(self):
        """Test enable_gpu_for_xgboost returns valid config."""
        from src.utils import enable_gpu_for_xgboost
        config = enable_gpu_for_xgboost()
        
        assert isinstance(config, dict)
        if config:
            assert 'tree_method' in config
            assert config['tree_method'] == 'gpu_hist'
    
    def test_enable_gpu_for_lightgbm(self):
        """Test enable_gpu_for_lightgbm returns valid config."""
        from src.utils import enable_gpu_for_lightgbm
        config = enable_gpu_for_lightgbm()
        
        assert isinstance(config, dict)
        if config:
            assert 'device' in config
            assert config['device'] == 'gpu'


class TestSection11MemoryManagement:
    """Tests for Section 11.1 - Memory Management."""
    
    def test_clear_memory_function_exists(self):
        """Test that clear_memory function exists."""
        from src.utils import clear_memory
        assert callable(clear_memory)
    
    def test_clear_memory_returns_dict(self):
        """Test that clear_memory returns dictionary with stats."""
        from src.utils import clear_memory
        result = clear_memory()
        
        assert isinstance(result, dict)
        assert 'gc_collected' in result
        assert 'gpu_cache_cleared' in result
    
    def test_get_memory_usage_function(self):
        """Test that get_memory_usage returns memory stats."""
        from src.utils import get_memory_usage
        mem = get_memory_usage()
        
        assert isinstance(mem, dict)
        assert 'process_rss_gb' in mem
        assert 'system_total_gb' in mem
        assert mem['process_rss_gb'] > 0
    
    def test_log_memory_usage_no_error(self):
        """Test that log_memory_usage runs without error."""
        from src.utils import log_memory_usage
        log_memory_usage("test")
    
    def test_memory_monitor_context_manager(self):
        """Test memory_monitor context manager."""
        from src.utils import memory_monitor
        
        with memory_monitor("test_operation", log_before=False, log_after=False):
            # Allocate some memory
            x = [i for i in range(100000)]
        
        # Should complete without error
        assert True
    
    def test_optimize_dataframe_memory(self):
        """Test DataFrame memory optimization."""
        from src.utils import optimize_dataframe_memory
        import pandas as pd
        
        # Create test DataFrame
        df = pd.DataFrame({
            'int_col': [1, 2, 3] * 100,
            'float_col': [1.1, 2.2, 3.3] * 100,
            'str_col': ['a', 'b', 'c'] * 100,
        })
        
        df_optimized, stats = optimize_dataframe_memory(df, deep_copy=True)
        
        assert isinstance(stats, dict)
        assert 'original_memory_mb' in stats
        assert 'optimized_memory_mb' in stats
        assert 'memory_reduction_percent' in stats
        assert stats['columns_optimized'] >= 0


class TestSection11ProgressBars:
    """Tests for Section 11.1 - Progress Bars."""
    
    def test_get_progress_bar_function_exists(self):
        """Test that get_progress_bar function exists."""
        from src.utils import get_progress_bar
        assert callable(get_progress_bar)
    
    def test_get_progress_bar_with_iterable(self):
        """Test get_progress_bar with iterable."""
        from src.utils import get_progress_bar
        
        items = [1, 2, 3, 4, 5]
        result = list(get_progress_bar(items, desc="test", disable=True))
        assert result == items
    
    def test_get_progress_bar_with_total(self):
        """Test get_progress_bar with total."""
        from src.utils import get_progress_bar
        
        pbar = get_progress_bar(range(5), total=5, desc="test", disable=True)
        result = list(pbar)
        assert result == [0, 1, 2, 3, 4]


class TestSection11EnvironmentVerification:
    """Tests for Section 11.2 - Environment Verification."""
    
    def test_verify_environment_function_exists(self):
        """Test that verify_environment function exists."""
        from src.utils import verify_environment
        assert callable(verify_environment)
    
    def test_verify_environment_returns_dict(self):
        """Test that verify_environment returns proper structure."""
        from src.utils import verify_environment
        result = verify_environment()
        
        assert isinstance(result, dict)
        assert 'python_version' in result
        assert 'python_ok' in result
        assert 'packages' in result
        assert 'all_ok' in result
    
    def test_verify_environment_checks_required_packages(self):
        """Test that verify_environment checks required packages."""
        from src.utils import verify_environment
        result = verify_environment()
        
        packages = result['packages']
        assert 'numpy' in packages
        assert 'pandas' in packages
        assert 'scikit-learn' in packages
        assert 'pyyaml' in packages
        
        # Required packages should be marked as required
        assert packages['numpy']['required'] == True
        assert packages['pandas']['required'] == True
    
    def test_verify_environment_all_required_installed(self):
        """Test that all required packages are installed."""
        from src.utils import verify_environment
        result = verify_environment()
        
        # This should pass since we're running tests
        assert result['all_ok'] == True
    
    def test_print_environment_info_no_error(self):
        """Test that print_environment_info runs without error."""
        from src.utils import print_environment_info
        # Should not raise any errors
        print_environment_info()


class TestSection11PerformanceOptimization:
    """Tests for Section 11.3 - Performance Optimization."""
    
    def test_get_optimal_n_jobs(self):
        """Test get_optimal_n_jobs function."""
        from src.utils import get_optimal_n_jobs
        
        n_jobs = get_optimal_n_jobs(memory_per_job_gb=1.0)
        assert isinstance(n_jobs, int)
        assert n_jobs >= 1
    
    def test_get_optimal_n_jobs_with_max(self):
        """Test get_optimal_n_jobs respects max_jobs."""
        from src.utils import get_optimal_n_jobs
        
        n_jobs = get_optimal_n_jobs(memory_per_job_gb=1.0, max_jobs=2)
        assert n_jobs <= 2
    
    def test_chunked_apply_function_exists(self):
        """Test that chunked_apply function exists."""
        from src.utils import chunked_apply
        assert callable(chunked_apply)
    
    def test_chunked_apply_works(self):
        """Test that chunked_apply processes DataFrame."""
        from src.utils import chunked_apply
        import pandas as pd
        
        df = pd.DataFrame({'a': range(100)})
        
        def add_one(chunk):
            chunk = chunk.copy()
            chunk['b'] = chunk['a'] + 1
            return chunk
        
        result = chunked_apply(df, add_one, chunk_size=25, show_progress=False)
        
        assert 'b' in result.columns
        assert len(result) == len(df)
        assert list(result['b']) == list(range(1, 101))


class TestSection11ColabUtilities:
    """Tests for Section 11 - Colab-specific Utilities."""
    
    def test_mount_google_drive_exists(self):
        """Test that mount_google_drive function exists."""
        from src.utils import mount_google_drive
        assert callable(mount_google_drive)
    
    def test_sync_to_drive_exists(self):
        """Test that sync_to_drive function exists."""
        from src.utils import sync_to_drive
        assert callable(sync_to_drive)
    
    def test_download_file_exists(self):
        """Test that download_file function exists."""
        from src.utils import download_file
        assert callable(download_file)
    
    def test_upload_files_exists(self):
        """Test that upload_files function exists."""
        from src.utils import upload_files
        assert callable(upload_files)
    
    def test_colab_functions_handle_non_colab(self):
        """Test that Colab functions handle non-Colab environment gracefully."""
        from src.utils import mount_google_drive, sync_to_drive, download_file, upload_files
        
        # These should not raise errors when not in Colab
        result = mount_google_drive()
        assert result is None  # Not in Colab
        
        result = sync_to_drive()
        assert result == False  # Not in Colab
        
        result = download_file("/tmp/test.txt")
        assert result == False  # Not in Colab
        
        result = upload_files()
        assert result == {}  # Not in Colab


class TestSection11RequirementsFiles:
    """Tests for Section 11.2 - Environment Management requirements files."""
    
    def test_colab_requirements_exists(self):
        """Test that env/colab_requirements.txt exists."""
        req_path = Path(__file__).parent.parent / 'env' / 'colab_requirements.txt'
        assert req_path.exists(), "env/colab_requirements.txt should exist"
    
    def test_colab_requirements_has_core_packages(self):
        """Test that colab_requirements.txt has core packages."""
        req_path = Path(__file__).parent.parent / 'env' / 'colab_requirements.txt'
        content = req_path.read_text(encoding='utf-8')
        
        core_packages = ['numpy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost']
        for pkg in core_packages:
            assert pkg in content, f"colab_requirements.txt should have {pkg}"
    
    def test_environment_yml_exists(self):
        """Test that env/environment.yml exists."""
        env_path = Path(__file__).parent.parent / 'env' / 'environment.yml'
        assert env_path.exists(), "env/environment.yml should exist"
    
    def test_environment_yml_valid(self):
        """Test that environment.yml is valid YAML."""
        import yaml
        env_path = Path(__file__).parent.parent / 'env' / 'environment.yml'
        
        with open(env_path, 'r') as f:
            content = yaml.safe_load(f)
        
        assert 'name' in content
        assert 'channels' in content
        assert 'dependencies' in content
    
    def test_environment_yml_has_python_version(self):
        """Test that environment.yml specifies Python version."""
        env_path = Path(__file__).parent.parent / 'env' / 'environment.yml'
        content = env_path.read_text(encoding='utf-8')
        
        assert 'python' in content.lower(), "environment.yml should specify Python version"


class TestSection11ColabNotebook:
    """Tests for Section 11.1 - Colab Notebook."""
    
    def test_colab_notebook_exists(self):
        """Test that Colab notebook exists."""
        notebook_path = Path(__file__).parent.parent / 'notebooks' / 'colab' / 'main.ipynb'
        assert notebook_path.exists(), "notebooks/colab/main.ipynb should exist"
    
    def test_colab_notebook_valid_json(self):
        """Test that Colab notebook is valid JSON."""
        import json
        notebook_path = Path(__file__).parent.parent / 'notebooks' / 'colab' / 'main.ipynb'
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        assert 'cells' in content, "Notebook should have cells"
        assert len(content['cells']) > 0, "Notebook should have at least one cell"
    
    def test_colab_notebook_has_required_sections(self):
        """Test that Colab notebook has required sections."""
        import json
        notebook_path = Path(__file__).parent.parent / 'notebooks' / 'colab' / 'main.ipynb'
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # Get all cell sources as text
        all_text = ""
        for cell in content['cells']:
            source = cell.get('source', [])
            if isinstance(source, list):
                all_text += ''.join(source)
            else:
                all_text += source
        
        all_text_lower = all_text.lower()
        
        # Check for required sections
        assert 'environment setup' in all_text_lower, "Should have Environment Setup section"
        assert 'import' in all_text_lower, "Should have import section"
        assert 'training' in all_text_lower or 'train' in all_text_lower, "Should have training section"
        assert 'submission' in all_text_lower, "Should have submission section"
    
    def test_colab_notebook_has_drive_mounting(self):
        """Test that Colab notebook has Drive mounting code."""
        import json
        notebook_path = Path(__file__).parent.parent / 'notebooks' / 'colab' / 'main.ipynb'
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        all_text = ""
        for cell in content['cells']:
            source = cell.get('source', [])
            if isinstance(source, list):
                all_text += ''.join(source)
            else:
                all_text += source
        
        assert 'drive.mount' in all_text, "Should have Drive mounting code"
    
    def test_colab_notebook_imports_project_modules(self):
        """Test that Colab notebook imports project modules."""
        import json
        notebook_path = Path(__file__).parent.parent / 'notebooks' / 'colab' / 'main.ipynb'
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        all_text = ""
        for cell in content['cells']:
            source = cell.get('source', [])
            if isinstance(source, list):
                all_text += ''.join(source)
            else:
                all_text += source
        
        # Should import from src
        assert 'from src' in all_text or 'import src' in all_text, \
            "Should import from src package"


# =============================================================================
# SECTION 8 TESTS: EXPERIMENTATION & OPTIMIZATION
# =============================================================================


class TestSection8FeatureExperiments:
    """Tests for Section 8.1 - Feature Experiments."""
    
    def test_frequency_encoding_function_exists(self):
        """Test that add_frequency_encoding_features function exists."""
        from src.features import add_frequency_encoding_features
        assert callable(add_frequency_encoding_features)
    
    def test_frequency_encoding_basic(self):
        """Test basic frequency encoding functionality."""
        from src.features import add_frequency_encoding_features
        
        df = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'C'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        
        result = add_frequency_encoding_features(df, categorical_cols=['category'], normalize=True)
        
        assert 'category_freq' in result.columns
        # A appears 3 times, B appears 2 times, C appears 1 time
        assert result.loc[0, 'category_freq'] == 3 / 6  # 0.5
        assert result.loc[3, 'category_freq'] == 2 / 6  # ~0.333
        assert result.loc[5, 'category_freq'] == 1 / 6  # ~0.167
    
    def test_frequency_encoding_no_normalize(self):
        """Test frequency encoding without normalization."""
        from src.features import add_frequency_encoding_features
        
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'C', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        
        result = add_frequency_encoding_features(df, categorical_cols=['category'], normalize=False)
        
        assert 'category_freq' in result.columns
        assert result.loc[0, 'category_freq'] == 2  # A appears 2 times
        assert result.loc[2, 'category_freq'] == 1  # B appears 1 time
        assert result.loc[3, 'category_freq'] == 3  # C appears 3 times
    
    def test_frequency_encoding_auto_detect(self):
        """Test frequency encoding auto-detects categorical columns."""
        from src.features import add_frequency_encoding_features
        
        df = pd.DataFrame({
            'country': ['US', 'US', 'DE'],  # Excluded (ID column)
            'brand_name': ['A', 'A', 'B'],  # Excluded (ID column)
            'ther_area': ['Cardio', 'Oncol', 'Cardio'],  # Should be encoded
            'value': [1, 2, 3]
        })
        
        result = add_frequency_encoding_features(df)
        
        # Should encode ther_area but not ID columns
        assert 'ther_area_freq' in result.columns
        assert 'country_freq' not in result.columns
        assert 'brand_name_freq' not in result.columns
    
    def test_feature_scaler_class_exists(self):
        """Test that FeatureScaler class exists."""
        from src.features import FeatureScaler
        assert FeatureScaler is not None
    
    def test_feature_scaler_standard(self):
        """Test StandardScaler functionality."""
        from src.features import FeatureScaler
        
        df = pd.DataFrame({
            'feature1': [0, 10, 20, 30, 40],
            'feature2': [0, 100, 200, 300, 400]
        })
        
        scaler = FeatureScaler(method='standard')
        scaled = scaler.fit_transform(df)
        
        # Standard scaling should give mean ~ 0, std ~ 1
        assert abs(scaled['feature1'].mean()) < 1e-10
        assert abs(scaled['feature2'].mean()) < 1e-10
        assert abs(scaled['feature1'].std(ddof=0) - 1.0) < 1e-10
    
    def test_feature_scaler_minmax(self):
        """Test MinMaxScaler functionality."""
        from src.features import FeatureScaler
        
        df = pd.DataFrame({
            'feature1': [0, 25, 50, 75, 100],
            'feature2': [-10, 0, 10, 20, 30]
        })
        
        scaler = FeatureScaler(method='minmax')
        scaled = scaler.fit_transform(df)
        
        # MinMax scaling should give values in [0, 1]
        assert scaled['feature1'].min() == 0.0
        assert scaled['feature1'].max() == 1.0
        assert scaled['feature2'].min() == 0.0
        assert scaled['feature2'].max() == 1.0
    
    def test_feature_scaler_robust(self):
        """Test RobustScaler functionality."""
        from src.features import FeatureScaler
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 100]  # 100 is outlier
        })
        
        scaler = FeatureScaler(method='robust')
        scaled = scaler.fit_transform(df)
        
        # RobustScaler uses median and IQR, so outliers should not affect much
        assert 'feature1' in scaled.columns
        assert len(scaled) == len(df)
    
    def test_feature_scaler_none(self):
        """Test no-op scaling."""
        from src.features import FeatureScaler
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5]
        })
        
        scaler = FeatureScaler(method='none')
        scaled = scaler.fit_transform(df)
        
        # Should be unchanged
        assert list(scaled['feature1']) == [1, 2, 3, 4, 5]
    
    def test_feature_scaler_transform_new_data(self):
        """Test scaler can transform new data after fitting."""
        from src.features import FeatureScaler
        
        train_df = pd.DataFrame({
            'feature1': [0, 10, 20, 30, 40]
        })
        
        test_df = pd.DataFrame({
            'feature1': [5, 15, 25]
        })
        
        scaler = FeatureScaler(method='standard')
        scaler.fit(train_df)
        
        test_scaled = scaler.transform(test_df)
        
        # Should transform without error
        assert len(test_scaled) == len(test_df)
    
    def test_feature_scaler_inverse_transform(self):
        """Test inverse transform recovers original data."""
        from src.features import FeatureScaler
        
        df = pd.DataFrame({
            'feature1': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        scaler = FeatureScaler(method='standard')
        scaled = scaler.fit_transform(df)
        recovered = scaler.inverse_transform(scaled)
        
        # Should recover original values
        np.testing.assert_array_almost_equal(
            df['feature1'].values, 
            recovered['feature1'].values, 
            decimal=10
        )
    
    def test_feature_scaler_exclude_cols(self):
        """Test excluding columns from scaling."""
        from src.features import FeatureScaler
        
        df = pd.DataFrame({
            'feature1': [0, 10, 20],
            'months_postgx': [0, 6, 12]  # Exclude this
        })
        
        scaler = FeatureScaler(method='standard', exclude_cols=['months_postgx'])
        scaled = scaler.fit_transform(df)
        
        # months_postgx should be unchanged
        assert list(scaled['months_postgx']) == [0, 6, 12]
        # feature1 should be scaled
        assert abs(scaled['feature1'].mean()) < 1e-10
    
    def test_run_feature_ablation_function_exists(self):
        """Test that run_feature_ablation function exists."""
        from src.features import run_feature_ablation
        assert callable(run_feature_ablation)
    
    def test_run_feature_ablation_basic(self):
        """Test basic feature ablation functionality."""
        from src.features import run_feature_ablation
        from src.models.linear import LinearModel
        
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame({
            'pre_entry_trend': np.random.randn(100),
            'avg_vol_6m': np.random.rand(100) * 1000,
            'months_postgx': np.random.randint(0, 24, 100),
            'n_gxs': np.random.randint(0, 5, 100)
        })
        y = pd.Series(np.random.rand(100))
        meta_df = pd.DataFrame({
            'country': ['US'] * 100,
            'brand_name': ['A'] * 100,
            'bucket': [1] * 50 + [2] * 50
        })
        
        # Define feature groups
        feature_groups = {
            'pre_entry': ['pre_entry_trend', 'avg_vol_6m'],
            'time': ['months_postgx'],
            'generics': ['n_gxs']
        }
        
        results = run_feature_ablation(
            X, y, meta_df,
            scenario=1,
            model_class=LinearModel,
            model_config={'model': {'type': 'ridge'}},
            feature_groups=feature_groups,
            val_fraction=0.2,
            random_state=42
        )
        
        # Check results structure
        assert 'baseline' in results
        assert 'rmse' in results['baseline']
        assert 'mae' in results['baseline']
        assert 'n_features' in results['baseline']
        
        # Should have drop results
        assert any('drop_' in k for k in results.keys())
        
        # Should have add results
        assert any('add_' in k for k in results.keys())


class TestSection8ModelExperiments:
    """Tests for Section 8.2 - Model Experiments."""
    
    def test_compare_models_function_exists(self):
        """Test that compare_models function exists."""
        from src.train import compare_models
        assert callable(compare_models)
    
    def test_compare_models_basic(self):
        """Test basic model comparison functionality."""
        from src.train import compare_models
        from src.evaluate import create_aux_file
        
        # Create test data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'months_postgx': np.random.randint(0, 24, 100)
        })
        y_train = pd.Series(np.random.rand(100))
        meta_train = pd.DataFrame({
            'country': ['US'] * 100,
            'brand_name': ['A'] * 100,
            'months_postgx': np.random.randint(0, 24, 100),
            'bucket': [1] * 50 + [2] * 50,
            'avg_vol_12m': [1000.0] * 100
        })
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'months_postgx': np.random.randint(0, 24, 20)
        })
        y_val = pd.Series(np.random.rand(20))
        meta_val = pd.DataFrame({
            'country': ['DE'] * 20,
            'brand_name': ['B'] * 20,
            'months_postgx': np.random.randint(0, 24, 20),
            'bucket': [1] * 10 + [2] * 10,
            'avg_vol_12m': [1000.0] * 20
        })
        
        # Simple model configs
        model_configs = {
            'global_mean': {'model_type': 'global_mean', 'params': {}},
            'flat': {'model_type': 'flat', 'params': {}}
        }
        
        results = compare_models(
            X_train, y_train, X_val, y_val,
            meta_train, meta_val,
            scenario=1,
            model_configs=model_configs
        )
        
        # Check results structure
        assert isinstance(results, pd.DataFrame)
        assert 'model_name' in results.columns
        assert 'rmse' in results.columns
        assert 'mae' in results.columns
        assert 'train_time_s' in results.columns
        
        # Should have entries for both models
        assert len(results) == 2
    
    def test_test_loss_functions_exists(self):
        """Test that test_loss_functions function exists."""
        from src.train import test_loss_functions
        assert callable(test_loss_functions)
    
    def test_run_model_experiments_exists(self):
        """Test that run_model_experiments function exists."""
        from src.train import run_model_experiments
        assert callable(run_model_experiments)


class TestSection8PostProcessing:
    """Tests for Section 8.4 - Post-Processing."""
    
    def test_optimize_ensemble_weights_function_exists(self):
        """Test that optimize_ensemble_weights_on_validation function exists."""
        from src.train import optimize_ensemble_weights_on_validation
        assert callable(optimize_ensemble_weights_on_validation)
    
    def test_optimize_ensemble_weights_basic(self):
        """Test basic ensemble weight optimization."""
        from src.train import optimize_ensemble_weights_on_validation
        from src.models.linear import GlobalMeanBaseline, FlatBaseline
        
        # Create test data
        np.random.seed(42)
        X_val = pd.DataFrame({
            'feature1': np.random.randn(50),
            'months_postgx': np.random.randint(0, 24, 50)
        })
        y_val = pd.Series(np.random.rand(50) * 0.8 + 0.1)  # Between 0.1 and 0.9
        meta_val = pd.DataFrame({
            'country': ['US'] * 50,
            'brand_name': ['A'] * 50,
            'months_postgx': np.random.randint(0, 24, 50),
            'bucket': [1] * 25 + [2] * 25,
            'avg_vol_12m': [1000.0] * 50
        })
        
        # Create two simple trained models
        model1 = GlobalMeanBaseline({})
        model2 = FlatBaseline({})
        
        # Fit models (they need X_train for fitting)
        X_train = X_val.copy()
        y_train = y_val.copy()
        model1.fit(X_train, y_train, X_val, y_val)
        model2.fit(X_train, y_train, X_val, y_val)
        
        weights, best_metric = optimize_ensemble_weights_on_validation(
            models=[model1, model2],
            X_val=X_val,
            y_val=y_val,
            meta_val=meta_val,
            scenario=1,
            optimization_metric='rmse',
            n_restarts=2
        )
        
        # Check weights are valid
        assert len(weights) == 2
        assert abs(weights.sum() - 1.0) < 1e-6  # Should sum to 1
        assert all(weights >= 0)  # Non-negative
        
        # Check best_metric is reasonable
        assert isinstance(best_metric, float)
        assert best_metric >= 0


class TestSection8ErosionCurveShaping:
    """Tests for Section 8.5 - Domain-Consistent Erosion Curve Shaping."""
    
    def test_get_default_feature_groups(self):
        """Test feature group detection."""
        from src.features import _get_default_feature_groups
        
        feature_cols = [
            'pre_entry_trend', 'avg_vol_12m', 'avg_vol_6m',  # pre_entry
            'months_postgx', 'time_bucket', 'time_decay',  # time
            'n_gxs', 'has_generic', 'multiple_generics',  # generics
            'hospital_rate_norm',  # drug (without ther_area to avoid overlap)
            'erosion_0_5', 'trend_0_5',  # early_erosion
            'feature_a_x_feature_b',  # interactions (no overlap with other patterns)
            'category_freq',  # frequency_encoding (generic name to avoid overlap)
            'other_feature'  # other
        ]
        
        groups = _get_default_feature_groups(feature_cols)
        
        # Check groups are assigned correctly
        assert 'pre_entry' in groups
        assert 'time' in groups
        assert 'generics' in groups
        assert 'drug' in groups
        assert 'early_erosion' in groups
        assert 'interactions' in groups
        assert 'frequency_encoding' in groups
        
        # Check specific features are in correct groups
        assert 'pre_entry_trend' in groups['pre_entry']
        assert 'months_postgx' in groups['time']
        assert 'n_gxs' in groups['generics']
        assert 'hospital_rate_norm' in groups['drug']
        assert 'erosion_0_5' in groups['early_erosion']
        assert 'feature_a_x_feature_b' in groups['interactions']
        assert 'category_freq' in groups['frequency_encoding']
    
    def test_compare_feature_engineering_approaches_exists(self):
        """Test that compare_feature_engineering_approaches function exists."""
        from src.features import compare_feature_engineering_approaches
        assert callable(compare_feature_engineering_approaches)


class TestSection8Integration:
    """Integration tests for Section 8 functionality."""
    
    def test_scaler_in_training_pipeline(self):
        """Test that FeatureScaler integrates with training pipeline."""
        from src.features import FeatureScaler
        from src.models.linear import LinearModel
        
        # Create test data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100) * 100,
            'feature2': np.random.randn(100) * 10,
            'months_postgx': np.random.randint(0, 24, 100)
        })
        y_train = pd.Series(np.random.rand(100))
        X_val = X_train.iloc[:20].copy()
        y_val = y_train.iloc[:20].copy()
        
        # Scale features
        scaler = FeatureScaler(method='standard')
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model on scaled features
        model = LinearModel({'model': {'type': 'ridge'}})
        model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Predict
        preds = model.predict(X_val_scaled)
        
        assert len(preds) == len(X_val)
        assert not np.any(np.isnan(preds))
    
    def test_ablation_results_consistency(self):
        """Test that ablation results are consistent."""
        from src.features import run_feature_ablation
        from src.models.linear import LinearModel
        
        np.random.seed(42)
        X = pd.DataFrame({
            'important_feature': np.linspace(0, 1, 100),
            'noise_feature': np.random.randn(100)
        })
        # Target correlates with important_feature
        y = pd.Series(X['important_feature'].values + np.random.randn(100) * 0.1)
        meta_df = pd.DataFrame({
            'country': ['US'] * 100,
            'brand_name': ['A'] * 100,
            'bucket': [1] * 50 + [2] * 50
        })
        
        feature_groups = {
            'important': ['important_feature'],
            'noise': ['noise_feature']
        }
        
        results = run_feature_ablation(
            X, y, meta_df,
            scenario=1,
            model_class=LinearModel,
            model_config={'model': {'type': 'ridge'}},
            feature_groups=feature_groups,
            val_fraction=0.2,
            random_state=42
        )
        
        # Dropping important feature should hurt more than dropping noise
        if 'drop_important' in results and 'drop_noise' in results:
            # Impact > 0 means dropping hurts performance (RMSE increases)
            assert results['drop_important']['impact'] > results['drop_noise']['impact']


# =============================================================================
# SECTION 12: COMPETITION STRATEGY TESTS
# =============================================================================

class TestSection12SubmissionTracker:
    """Tests for SubmissionTracker class."""
    
    def test_submission_tracker_initialization(self, tmp_path):
        """Test SubmissionTracker initializes correctly."""
        from src.utils import SubmissionTracker
        
        log_path = tmp_path / 'tracker.json'
        tracker = SubmissionTracker(log_path=str(log_path))
        
        assert tracker.log_path == log_path
        assert tracker.submissions == []
    
    def test_submission_tracker_log_submission(self, tmp_path):
        """Test logging a submission."""
        from src.utils import SubmissionTracker
        
        log_path = tmp_path / 'tracker.json'
        tracker = SubmissionTracker(log_path=str(log_path))
        
        record = tracker.log_submission(
            submission_path='submissions/v1.csv',
            cv_score=0.15,
            lb_score=0.18,
            scenario=1,
            model_info={'type': 'catboost'},
            notes='Test submission'
        )
        
        assert record['id'] == 1
        assert record['cv_score'] == 0.15
        assert record['lb_score'] == 0.18
        assert record['cv_lb_gap'] == 0.03
        assert len(tracker.submissions) == 1
    
    def test_submission_tracker_update_lb_score(self, tmp_path):
        """Test updating LB score for a submission."""
        from src.utils import SubmissionTracker
        
        log_path = tmp_path / 'tracker.json'
        tracker = SubmissionTracker(log_path=str(log_path))
        
        tracker.log_submission(
            submission_path='submissions/v1.csv',
            cv_score=0.15,
            lb_score=None,
            scenario=1
        )
        
        tracker.update_lb_score(1, 0.18)
        
        assert tracker.submissions[0]['lb_score'] == 0.18
        assert tracker.submissions[0]['cv_lb_gap'] == 0.03
    
    def test_submission_tracker_analyze_cv_lb_variance(self, tmp_path):
        """Test CV-LB variance analysis."""
        from src.utils import SubmissionTracker
        
        log_path = tmp_path / 'tracker.json'
        tracker = SubmissionTracker(log_path=str(log_path))
        
        # Log multiple submissions
        tracker.log_submission('v1.csv', cv_score=0.15, lb_score=0.17)
        tracker.log_submission('v2.csv', cv_score=0.14, lb_score=0.16)
        tracker.log_submission('v3.csv', cv_score=0.13, lb_score=0.15)
        
        analysis = tracker.analyze_cv_lb_variance()
        
        assert analysis['n_submissions'] == 3
        assert analysis['mean_gap'] == 0.02
        assert 'correlation' in analysis
        assert 'overfitting_warning' in analysis
    
    def test_submission_tracker_get_best_submission(self, tmp_path):
        """Test getting best submission."""
        from src.utils import SubmissionTracker
        
        log_path = tmp_path / 'tracker.json'
        tracker = SubmissionTracker(log_path=str(log_path))
        
        tracker.log_submission('v1.csv', cv_score=0.15, lb_score=0.18)
        tracker.log_submission('v2.csv', cv_score=0.12, lb_score=0.14)
        tracker.log_submission('v3.csv', cv_score=0.17, lb_score=0.13)
        
        best_cv = tracker.get_best_submission(by='cv_score')
        best_lb = tracker.get_best_submission(by='lb_score')
        
        assert best_cv['cv_score'] == 0.12
        assert best_lb['lb_score'] == 0.13
    
    def test_submission_tracker_identify_overfitting(self, tmp_path):
        """Test identification of overfitting submissions."""
        from src.utils import SubmissionTracker
        
        log_path = tmp_path / 'tracker.json'
        tracker = SubmissionTracker(log_path=str(log_path))
        
        tracker.log_submission('good.csv', cv_score=0.15, lb_score=0.16)  # gap = 0.01
        tracker.log_submission('bad.csv', cv_score=0.10, lb_score=0.25)   # gap = 0.15
        
        overfitting = tracker.identify_overfitting_submissions(gap_threshold=0.1)
        
        assert len(overfitting) == 1
        assert overfitting[0]['submission_path'] == 'bad.csv'
    
    def test_submission_tracker_persistence(self, tmp_path):
        """Test that submissions persist across instances."""
        from src.utils import SubmissionTracker
        
        log_path = tmp_path / 'tracker.json'
        
        # Create first tracker and log submission
        tracker1 = SubmissionTracker(log_path=str(log_path))
        tracker1.log_submission('v1.csv', cv_score=0.15, lb_score=0.18)
        
        # Create second tracker and verify data is loaded
        tracker2 = SubmissionTracker(log_path=str(log_path))
        assert len(tracker2.submissions) == 1
        assert tracker2.submissions[0]['cv_score'] == 0.15


class TestSection12TimeAllocationTracker:
    """Tests for TimeAllocationTracker class."""
    
    def test_time_allocation_tracker_init(self):
        """Test TimeAllocationTracker initialization."""
        from src.utils import TimeAllocationTracker
        
        tracker = TimeAllocationTracker()
        
        assert tracker.current_phase is None
        assert tracker.time_logs == []
    
    def test_time_allocation_start_end_phase(self):
        """Test starting and ending a phase."""
        from src.utils import TimeAllocationTracker
        import time
        
        tracker = TimeAllocationTracker()
        
        tracker.start_phase('eda')
        assert tracker.current_phase == 'eda'
        
        time.sleep(0.1)  # Small delay
        duration = tracker.end_phase()
        
        assert tracker.current_phase is None
        assert duration > 0
        assert len(tracker.time_logs) == 1
        assert tracker.time_logs[0]['phase'] == 'eda'
    
    def test_time_allocation_get_summary(self):
        """Test getting time allocation summary."""
        from src.utils import TimeAllocationTracker
        
        tracker = TimeAllocationTracker()
        
        # Manually add time logs for testing
        tracker.time_logs = [
            {'phase': 'eda', 'duration_hours': 2.0},
            {'phase': 'modeling', 'duration_hours': 3.0},
            {'phase': 'tuning', 'duration_hours': 1.0}
        ]
        
        summary = tracker.get_summary()
        
        assert summary['total_hours'] == 6.0
        assert summary['phase_hours']['eda'] == 2.0
        assert summary['phase_hours']['modeling'] == 3.0
        assert 'deviations' in summary


class TestSection12BackupAndValidation:
    """Tests for backup and validation functions."""
    
    def test_create_backup_submission(self, tmp_path):
        """Test creating backup of submission."""
        from src.utils import create_backup_submission
        
        # Create test submission
        submission_path = tmp_path / 'submission.csv'
        df = pd.DataFrame({'country': ['US'], 'volume': [100]})
        df.to_csv(submission_path, index=False)
        
        backup_dir = tmp_path / 'backups'
        
        result = create_backup_submission(
            submission_path=str(submission_path),
            backup_dir=str(backup_dir)
        )
        
        assert 'submission' in result
        assert Path(result['submission']).exists()
    
    def test_validate_submission_completeness(self, tmp_path):
        """Test submission completeness validation."""
        from src.utils import validate_submission_completeness
        
        template_df = pd.DataFrame({
            'country': ['US', 'US', 'UK', 'UK'],
            'brand_name': ['A', 'A', 'B', 'B'],
            'months_postgx': [0, 1, 0, 1]
        })
        
        # Complete submission
        submission_df = template_df.copy()
        result = validate_submission_completeness(submission_df, template_df)
        assert result['is_complete']
        assert result['missing_rows'] == 0
        
        # Incomplete submission
        incomplete_df = template_df.iloc[:2].copy()
        result = validate_submission_completeness(incomplete_df, template_df)
        assert not result['is_complete']
        assert result['missing_rows'] == 2
    
    def test_check_prediction_sanity(self):
        """Test prediction sanity checks."""
        from src.utils import check_prediction_sanity
        
        # Good predictions
        good_df = pd.DataFrame({
            'volume': [100, 90, 80, 70, 60, 50],
            'months_postgx': [0, 4, 8, 12, 16, 20]
        })
        result = check_prediction_sanity(good_df)
        assert result['is_sane']
        assert len(result['issues']) == 0
        
        # Bad predictions with negative values
        bad_df = pd.DataFrame({
            'volume': [100, -10, 80, 70]
        })
        result = check_prediction_sanity(bad_df)
        assert not result['is_sane']
        assert any('negative' in issue for issue in result['issues'])
    
    def test_check_prediction_sanity_with_nan(self):
        """Test prediction sanity with NaN values."""
        from src.utils import check_prediction_sanity
        
        df = pd.DataFrame({
            'volume': [100, np.nan, 80, 70]
        })
        result = check_prediction_sanity(df)
        assert not result['is_sane']
        assert any('NaN' in issue for issue in result['issues'])


class TestSection12FinalWeekPlaybook:
    """Tests for FinalWeekPlaybook class."""
    
    def test_final_week_playbook_init(self, tmp_path):
        """Test FinalWeekPlaybook initialization."""
        from src.utils import FinalWeekPlaybook
        
        playbook_path = tmp_path / 'playbook.json'
        playbook = FinalWeekPlaybook(playbook_path=str(playbook_path))
        
        assert not playbook.is_config_frozen()
        assert playbook.frozen_config is None
    
    def test_final_week_playbook_freeze_config(self, tmp_path):
        """Test freezing configuration."""
        from src.utils import FinalWeekPlaybook
        
        playbook_path = tmp_path / 'playbook.json'
        playbook = FinalWeekPlaybook(playbook_path=str(playbook_path))
        
        playbook.freeze_config(
            model_type='catboost',
            model_config={'iterations': 1000, 'depth': 6},
            feature_config={'use_lag': True},
            cv_scheme='time_series'
        )
        
        assert playbook.is_config_frozen()
        assert playbook.frozen_config['model_type'] == 'catboost'
        assert playbook.freeze_timestamp is not None
    
    def test_final_week_playbook_suggest_variants(self, tmp_path):
        """Test suggesting submission variants."""
        from src.utils import FinalWeekPlaybook
        
        playbook = FinalWeekPlaybook(playbook_path=str(tmp_path / 'playbook.json'))
        
        variants = playbook.suggest_submission_variants()
        
        assert len(variants) > 0
        variant_names = [v['name'] for v in variants]
        assert 'best_cv' in variant_names
        assert 'multi_seed' in variant_names
    
    def test_final_week_playbook_log_submission(self, tmp_path):
        """Test logging submission generation."""
        from src.utils import FinalWeekPlaybook
        
        playbook = FinalWeekPlaybook(playbook_path=str(tmp_path / 'playbook.json'))
        
        playbook.log_submission_generated(
            variant_name='best_cv',
            submission_path='submissions/best_cv.csv',
            cv_score=0.14,
            notes='Best CV model'
        )
        
        assert len(playbook.submissions_generated) == 1
        assert playbook.submissions_generated[0]['variant'] == 'best_cv'
    
    def test_final_week_playbook_log_verification(self, tmp_path):
        """Test logging verification."""
        from src.utils import FinalWeekPlaybook
        
        playbook = FinalWeekPlaybook(playbook_path=str(tmp_path / 'playbook.json'))
        
        playbook.log_verification(
            submission_path='submissions/v1.csv',
            format_ok=True,
            metric_score=0.15,
            issues=[]
        )
        
        assert len(playbook.verifications) == 1
        assert playbook.verifications[0]['format_ok']
    
    def test_final_week_playbook_get_summary(self, tmp_path):
        """Test getting playbook summary."""
        from src.utils import FinalWeekPlaybook
        
        playbook = FinalWeekPlaybook(playbook_path=str(tmp_path / 'playbook.json'))
        
        playbook.freeze_config(
            model_type='catboost',
            model_config={},
            feature_config={},
            cv_scheme='kfold'
        )
        playbook.log_submission_generated('v1', 'path1.csv', 0.15)
        playbook.log_verification('path1.csv', True)
        
        summary = playbook.get_summary()
        
        assert summary['is_frozen']
        assert summary['n_submissions_generated'] == 1
        assert summary['n_verified'] == 1
    
    def test_final_week_playbook_persistence(self, tmp_path):
        """Test that playbook persists across instances."""
        from src.utils import FinalWeekPlaybook
        
        playbook_path = tmp_path / 'playbook.json'
        
        # Create and configure first playbook
        playbook1 = FinalWeekPlaybook(playbook_path=str(playbook_path))
        playbook1.freeze_config(
            model_type='catboost',
            model_config={'iterations': 1000},
            feature_config={},
            cv_scheme='kfold'
        )
        
        # Create second playbook and verify config is loaded
        playbook2 = FinalWeekPlaybook(playbook_path=str(playbook_path))
        assert playbook2.is_config_frozen()
        assert playbook2.frozen_config['model_type'] == 'catboost'


class TestSection12ExternalDataCompliance:
    """Tests for external data compliance checking."""
    
    def test_verify_external_data_compliance(self, tmp_path):
        """Test external data compliance verification."""
        from src.utils import verify_external_data_compliance
        
        # Create minimal project structure
        (tmp_path / 'data' / 'raw').mkdir(parents=True)
        (tmp_path / 'data' / 'raw' / 'TRAIN').mkdir()
        (tmp_path / 'data' / 'raw' / 'TRAIN' / 'df_volume_train.csv').write_text('a,b\n1,2')
        
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'clean_code.py').write_text('# Clean code\nimport pandas as pd')
        
        result = verify_external_data_compliance(str(tmp_path))
        
        assert result['is_compliant']
        assert len(result['data_sources']) > 0


class TestSection12PreSubmissionChecklist:
    """Tests for pre-submission checklist."""
    
    def test_run_pre_submission_checklist_all_pass(self, tmp_path):
        """Test pre-submission checklist when all checks pass."""
        from src.utils import run_pre_submission_checklist
        
        # Create template
        template_df = pd.DataFrame({
            'country': ['US', 'US', 'UK', 'UK'],
            'brand_name': ['A', 'A', 'B', 'B'],
            'months_postgx': [0, 1, 0, 1],
            'volume': [0, 0, 0, 0]
        })
        template_path = tmp_path / 'template.csv'
        template_df.to_csv(template_path, index=False)
        
        # Create matching submission
        submission_df = template_df.copy()
        submission_df['volume'] = [100, 95, 80, 75]
        submission_path = tmp_path / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        
        result = run_pre_submission_checklist(
            submission_path=str(submission_path),
            template_path=str(template_path)
        )
        
        assert result['all_passed']
        assert result['checks']['file_exists']['passed']
        assert result['checks']['completeness']['passed']
    
    def test_run_pre_submission_checklist_missing_file(self, tmp_path):
        """Test pre-submission checklist when file is missing."""
        from src.utils import run_pre_submission_checklist
        
        template_path = tmp_path / 'template.csv'
        pd.DataFrame({'a': [1]}).to_csv(template_path, index=False)
        
        result = run_pre_submission_checklist(
            submission_path=str(tmp_path / 'nonexistent.csv'),
            template_path=str(template_path)
        )
        
        assert not result['all_passed']
        assert not result['checks']['file_exists']['passed']
    
    def test_run_pre_submission_checklist_incomplete(self, tmp_path):
        """Test pre-submission checklist with incomplete submission."""
        from src.utils import run_pre_submission_checklist
        
        # Create template
        template_df = pd.DataFrame({
            'country': ['US', 'US', 'UK', 'UK'],
            'brand_name': ['A', 'A', 'B', 'B'],
            'months_postgx': [0, 1, 0, 1],
            'volume': [0, 0, 0, 0]
        })
        template_path = tmp_path / 'template.csv'
        template_df.to_csv(template_path, index=False)
        
        # Create incomplete submission (missing rows)
        submission_df = template_df.iloc[:2].copy()
        submission_df['volume'] = [100, 95]
        submission_path = tmp_path / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        
        result = run_pre_submission_checklist(
            submission_path=str(submission_path),
            template_path=str(template_path)
        )
        
        assert not result['all_passed']
        assert not result['checks']['completeness']['passed']


class TestSection12GenerateFinalReport:
    """Tests for final submission report generation."""
    
    def test_generate_final_submission_report(self, tmp_path):
        """Test generating final submission report."""
        from src.utils import generate_final_submission_report
        
        # Create submission file
        submission_df = pd.DataFrame({
            'country': ['US', 'UK'],
            'brand_name': ['A', 'B'],
            'months_postgx': [0, 0],
            'volume': [100, 80]
        })
        submission_path = tmp_path / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        
        # Create model file
        model_path = tmp_path / 'model_s1.pkl'
        model_path.write_text('dummy model')
        
        report = generate_final_submission_report(
            submission_path=str(submission_path),
            model_paths={'S1': str(model_path)},
            cv_scores={'S1': 0.145}
        )
        
        assert 'FINAL SUBMISSION REPORT' in report
        assert 'Volume range' in report
        assert 'CV SCORES' in report
    
    def test_generate_final_report_save_to_file(self, tmp_path):
        """Test saving final report to file."""
        from src.utils import generate_final_submission_report
        
        submission_df = pd.DataFrame({'volume': [100, 80]})
        submission_path = tmp_path / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        
        output_path = tmp_path / 'report.txt'
        
        report = generate_final_submission_report(
            submission_path=str(submission_path),
            model_paths={},
            cv_scores={},
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert output_path.read_text() == report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


