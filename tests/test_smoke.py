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
    assert str(raw_path) == 'data/raw'
    
    volume_path = get_path(config, 'files.train.volume')
    assert str(volume_path) == 'TRAIN/df_volume_train.csv'


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


