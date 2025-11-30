"""
Tests for model fixes to ensure they work correctly.

These tests verify the following fixes:
1. LinearModel: Handles categorical features correctly (drop/onehot/label)
2. GlobalMeanBaseline: Works without months_postgx in X features
3. HybridPhysicsMLWrapper: Conforms to BaseModel interface
4. ARIHOWWrapper: Conforms to BaseModel interface

Each test validates the BaseModel interface:
- fit(X_train, y_train, X_val=None, y_val=None, sample_weight=None)
- predict(X)
- save(path) / load(path)
"""

import sys
from pathlib import Path
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Helper Functions
# =============================================================================

def create_synthetic_data(
    n_series: int = 5,
    n_months: int = 24,
    include_categoricals: bool = True,
    include_months_postgx: bool = True,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Create synthetic data for model testing.
    
    Returns:
        X: Feature DataFrame
        y: Target Series
        meta: Metadata DataFrame with series identifiers
    """
    np.random.seed(seed)
    
    data = []
    for i in range(n_series):
        country = f'COUNTRY_{i}'
        brand = f'BRAND_{i}'
        bucket = 1 if i < n_series // 2 else 2
        avg_vol = 1000 * (i + 1)
        
        for month in range(n_months):
            # Synthetic erosion pattern
            y_norm = 1.0 - 0.02 * month + np.random.normal(0, 0.05)
            y_norm = max(0.1, min(1.5, y_norm))  # Clip to reasonable range
            
            row = {
                'country': country,
                'brand_name': brand,
                'bucket': bucket,
                'avg_vol_12m': avg_vol,
                'feature1': np.random.randn(),
                'feature2': np.random.randn(),
                'n_gxs': max(0, int(np.random.poisson(month / 4))),
                'y_norm': y_norm,
            }
            
            if include_months_postgx:
                row['months_postgx'] = month
                
            if include_categoricals:
                row['ther_area'] = ['Cardio', 'Oncology', 'Immuno'][i % 3]
                row['biological'] = i % 2
                
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Split into X, y, meta
    meta_cols = ['country', 'brand_name', 'bucket', 'avg_vol_12m']
    if include_months_postgx:
        meta_cols.append('months_postgx')
        
    target_col = 'y_norm'
    
    feature_cols = [c for c in df.columns if c not in meta_cols + [target_col]]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    meta = df[meta_cols].copy()
    
    return X, y, meta


def verify_basemodel_interface(model_class, model_config: dict, **init_kwargs):
    """
    Generic test to verify a model conforms to BaseModel interface.
    
    Tests:
    - fit(X_train, y_train, X_val, y_val, sample_weight)
    - predict(X)
    - save(path) / load(path)
    """
    # Create synthetic data
    X_train, y_train, meta_train = create_synthetic_data(
        n_series=4, n_months=24, seed=42
    )
    X_val, y_val, meta_val = create_synthetic_data(
        n_series=2, n_months=24, seed=123
    )
    
    # Combine X and meta for models that need meta features
    X_train_full = pd.concat([X_train.reset_index(drop=True), 
                              meta_train.reset_index(drop=True)], axis=1)
    X_val_full = pd.concat([X_val.reset_index(drop=True),
                            meta_val.reset_index(drop=True)], axis=1)
    
    sample_weight = pd.Series(np.ones(len(y_train)))
    
    # Instantiate model
    model = model_class(model_config, **init_kwargs)
    
    # Test fit
    model.fit(
        X_train=X_train_full, 
        y_train=y_train.reset_index(drop=True),
        X_val=X_val_full, 
        y_val=y_val.reset_index(drop=True),
        sample_weight=sample_weight
    )
    
    # Test predict
    preds = model.predict(X_val_full)
    assert len(preds) == len(X_val_full), "Predictions length should match input"
    assert not np.isnan(preds).any(), "Predictions should not contain NaN"
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'model'
        model.save(str(save_path))
        
        # Load into new instance
        model_loaded = model_class(model_config, **init_kwargs)
        model_loaded.load(str(save_path))
        
        # Predictions should match
        preds_loaded = model_loaded.predict(X_val_full)
        np.testing.assert_array_almost_equal(
            preds, preds_loaded, decimal=5,
            err_msg="Loaded model predictions should match original"
        )
    
    return model, preds


# =============================================================================
# Test 1: LinearModel with Categorical Features
# =============================================================================

class TestLinearModelCategoricals:
    """Tests for LinearModel categorical feature handling."""
    
    def test_linear_model_handles_categoricals_drop(self):
        """Test LinearModel with handle_categoricals='drop'."""
        from src.models.linear import LinearModel
        
        # Create data with categorical features
        X_train, y_train, meta = create_synthetic_data(
            n_series=4, n_months=12, include_categoricals=True
        )
        X_val, y_val, _ = create_synthetic_data(
            n_series=2, n_months=12, include_categoricals=True, seed=123
        )
        
        # Model should work with categoricals when set to 'drop' in preprocessing config
        config = {
            'model': {'type': 'ridge'},
            'ridge': {'alpha': 1.0},
            'preprocessing': {'handle_categoricals': 'drop'}
        }
        model = LinearModel(config)
        
        # Should not raise
        model.fit(X_train, y_train, X_val, y_val, sample_weight=None)
        
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        
    def test_linear_model_handles_categoricals_onehot(self):
        """Test LinearModel with handle_categoricals='onehot'."""
        from src.models.linear import LinearModel
        
        X_train, y_train, _ = create_synthetic_data(
            n_series=4, n_months=12, include_categoricals=True
        )
        X_val, y_val, _ = create_synthetic_data(
            n_series=2, n_months=12, include_categoricals=True, seed=123
        )
        
        config = {
            'model': {'type': 'ridge'},
            'ridge': {'alpha': 1.0},
            'preprocessing': {'handle_categoricals': 'onehot'}
        }
        model = LinearModel(config)
        
        model.fit(X_train, y_train, X_val, y_val, sample_weight=None)
        
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        
    def test_linear_model_handles_categoricals_label(self):
        """Test LinearModel with handle_categoricals='label'."""
        from src.models.linear import LinearModel
        
        X_train, y_train, _ = create_synthetic_data(
            n_series=4, n_months=12, include_categoricals=True
        )
        X_val, y_val, _ = create_synthetic_data(
            n_series=2, n_months=12, include_categoricals=True, seed=123
        )
        
        config = {
            'model': {'type': 'ridge'},
            'ridge': {'alpha': 1.0},
            'preprocessing': {'handle_categoricals': 'label'}
        }
        model = LinearModel(config)
        
        model.fit(X_train, y_train, X_val, y_val, sample_weight=None)
        
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        
    def test_linear_model_without_categoricals(self):
        """Test LinearModel works normally without categorical features."""
        from src.models.linear import LinearModel
        
        X_train, y_train, _ = create_synthetic_data(
            n_series=4, n_months=12, include_categoricals=False
        )
        X_val, y_val, _ = create_synthetic_data(
            n_series=2, n_months=12, include_categoricals=False, seed=123
        )
        
        config = {
            'model': {'type': 'ridge'},
            'ridge': {'alpha': 1.0}
        }
        model = LinearModel(config)
        
        model.fit(X_train, y_train, X_val, y_val, sample_weight=None)
        
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        
    def test_linear_model_save_load_with_categoricals(self):
        """Test LinearModel save/load preserves categorical encoders."""
        from src.models.linear import LinearModel
        
        X_train, y_train, _ = create_synthetic_data(
            n_series=4, n_months=12, include_categoricals=True
        )
        X_val, y_val, _ = create_synthetic_data(
            n_series=2, n_months=12, include_categoricals=True, seed=123
        )
        
        config = {
            'model': {'type': 'ridge'},
            'ridge': {'alpha': 1.0},
            'preprocessing': {'handle_categoricals': 'onehot'}
        }
        model = LinearModel(config)
        model.fit(X_train, y_train, X_val, y_val, sample_weight=None)
        
        preds_original = model.predict(X_val)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model'
            model.save(str(save_path))
            
            # Load using class method
            model_loaded = LinearModel.load(str(save_path))
            
            preds_loaded = model_loaded.predict(X_val)
            
            np.testing.assert_array_almost_equal(preds_original, preds_loaded)
            
    def test_linear_model_unseen_category(self):
        """Test LinearModel handles unseen categories at prediction time."""
        from src.models.linear import LinearModel
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'ther_area': ['Cardio', 'Oncology'] * 50
        })
        y_train = pd.Series(np.random.rand(100))
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(10),
            'ther_area': ['Cardio', 'NewCategory'] * 5  # NewCategory not in train!
        })
        
        config = {
            'model': {'type': 'ridge'},
            'ridge': {'alpha': 1.0},
            'preprocessing': {'handle_categoricals': 'onehot'}
        }
        model = LinearModel(config)
        model.fit(X_train, y_train, None, None, sample_weight=None)
        
        # Should handle gracefully (encode unknown as zeros)
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()


# =============================================================================
# Test 2: GlobalMeanBaseline without months_postgx in X
# =============================================================================

class TestGlobalMeanBaselineFix:
    """Tests for GlobalMeanBaseline months_postgx fallback."""
    
    def test_global_mean_with_months_in_x(self):
        """Test GlobalMeanBaseline works when months_postgx is in X."""
        from src.models.linear import GlobalMeanBaseline
        
        X_train, y_train, meta = create_synthetic_data(
            n_series=4, n_months=24, include_months_postgx=True
        )
        # Include months_postgx in X
        X_train_with_months = X_train.copy()
        X_train_with_months['months_postgx'] = meta['months_postgx'].values
        
        X_val, y_val, meta_val = create_synthetic_data(
            n_series=2, n_months=24, include_months_postgx=True, seed=123
        )
        X_val_with_months = X_val.copy()
        X_val_with_months['months_postgx'] = meta_val['months_postgx'].values
        
        model = GlobalMeanBaseline({})
        model.fit(X_train_with_months, y_train, X_val_with_months, y_val, None)
        
        preds = model.predict(X_val_with_months)
        assert len(preds) == len(X_val_with_months)
        assert not np.isnan(preds).any()
        
    def test_global_mean_without_months_in_x(self):
        """Test GlobalMeanBaseline works when months_postgx is NOT in X."""
        from src.models.linear import GlobalMeanBaseline
        
        # X does NOT contain months_postgx
        X_train = pd.DataFrame({
            'feature1': np.random.randn(96),  # 4 series * 24 months
            'feature2': np.random.randn(96),
        })
        y_train = pd.Series(np.random.rand(96))
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(48),  # 2 series * 24 months
            'feature2': np.random.randn(48),
        })
        
        model = GlobalMeanBaseline({})
        
        # Should NOT raise, but use fallback
        model.fit(X_train, y_train, X_val, None, None)
        
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        # Fallback should predict average y_norm value
        
    def test_global_mean_save_load(self):
        """Test GlobalMeanBaseline save/load works with months fallback."""
        from src.models.linear import GlobalMeanBaseline
        
        # With months_postgx in X
        n_samples = 96
        X_train = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'months_postgx': list(range(24)) * 4,
        })
        y_train = pd.Series([1.0 - 0.02 * m for m in X_train['months_postgx']])
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(48),
            'months_postgx': list(range(24)) * 2,
        })
        
        model = GlobalMeanBaseline({})
        model.fit(X_train, y_train, X_val, None, None)
        
        preds_original = model.predict(X_val)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model'
            model.save(str(save_path))
            
            # Load using classmethod
            model_loaded = GlobalMeanBaseline.load(str(save_path))
            
            preds_loaded = model_loaded.predict(X_val)
            
            np.testing.assert_array_almost_equal(preds_original, preds_loaded)


# =============================================================================
# Test 3: HybridPhysicsMLWrapper BaseModel Interface
# =============================================================================

class TestHybridPhysicsMLWrapper:
    """Tests for HybridPhysicsMLWrapper conforming to BaseModel interface."""
    
    def test_wrapper_basemodel_interface(self):
        """Test HybridPhysicsMLWrapper has proper BaseModel interface."""
        from src.models.hybrid_physics_ml import HybridPhysicsMLWrapper
        from src.models.base import BaseModel
        
        # Check inheritance
        assert issubclass(HybridPhysicsMLWrapper, BaseModel)
        
        # Check required methods exist
        wrapper = HybridPhysicsMLWrapper({})
        assert hasattr(wrapper, 'fit')
        assert hasattr(wrapper, 'predict')
        assert hasattr(wrapper, 'save')
        assert hasattr(wrapper, 'load')
        assert hasattr(wrapper, 'get_feature_importance')
        
    def test_wrapper_fit_predict(self):
        """Test HybridPhysicsMLWrapper fit and predict."""
        from src.models.hybrid_physics_ml import HybridPhysicsMLWrapper
        
        # Create synthetic data with required columns
        np.random.seed(42)
        n_samples = 120  # 5 series * 24 months
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'months_postgx': list(range(24)) * 5,
            'avg_vol_12m': [1000.0] * n_samples,
            'n_gxs': np.random.randint(0, 5, n_samples),
        })
        y = pd.Series(np.maximum(0.1, 1.0 - 0.02 * X['months_postgx'] + np.random.randn(n_samples) * 0.05))
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(48),
            'feature2': np.random.randn(48),
            'months_postgx': list(range(24)) * 2,
            'avg_vol_12m': [1000.0] * 48,
            'n_gxs': np.random.randint(0, 5, 48),
        })
        y_val = pd.Series(np.maximum(0.1, 1.0 - 0.02 * X_val['months_postgx'] + np.random.randn(48) * 0.05))
        
        sample_weight = pd.Series(np.ones(n_samples))
        
        # Use CatBoost which is available (LightGBM may have libomp issues on macOS)
        config = {
            'decay_type': 'exponential',
            'ml_model_type': 'catboost',
            'ml_params': {'iterations': 10, 'verbose': False, 'random_seed': 42}
        }
        
        model = HybridPhysicsMLWrapper(config)
        
        # Fit should accept BaseModel signature
        model.fit(X, y, X_val, y_val, sample_weight)
        
        # Predict should work
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        
    def test_wrapper_save_load(self):
        """Test HybridPhysicsMLWrapper save and load."""
        from src.models.hybrid_physics_ml import HybridPhysicsMLWrapper
        
        np.random.seed(42)
        n_samples = 72
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'months_postgx': list(range(24)) * 3,
            'avg_vol_12m': [1000.0] * n_samples,
        })
        y = pd.Series(np.maximum(0.1, 1.0 - 0.02 * X['months_postgx']))
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(24),
            'months_postgx': list(range(24)),
            'avg_vol_12m': [1000.0] * 24,
        })
        
        # Use CatBoost which is available (LightGBM may have libomp issues on macOS)
        config = {
            'decay_type': 'linear',
            'ml_model_type': 'catboost',
            'ml_params': {'iterations': 10, 'verbose': False, 'random_seed': 42}
        }
        model = HybridPhysicsMLWrapper(config)
        model.fit(X, y, X_val, None, None)
        
        preds_original = model.predict(X_val)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'hybrid_model'
            model.save(str(save_path))
            
            # Load using classmethod
            model_loaded = HybridPhysicsMLWrapper.load(str(save_path))
            
            preds_loaded = model_loaded.predict(X_val)
            
            np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=5)


# =============================================================================
# Test 4: ARIHOWWrapper BaseModel Interface
# =============================================================================

class TestARIHOWWrapper:
    """Tests for ARIHOWWrapper conforming to BaseModel interface."""
    
    @pytest.fixture
    def check_statsmodels(self):
        """Check if statsmodels is available."""
        try:
            import statsmodels
            return True
        except ImportError:
            pytest.skip("statsmodels not installed, skipping ARIHOW tests")
            return False
            
    def test_wrapper_basemodel_interface(self, check_statsmodels):
        """Test ARIHOWWrapper has proper BaseModel interface."""
        from src.models.arihow import ARIHOWWrapper
        from src.models.base import BaseModel
        
        # Check inheritance
        assert issubclass(ARIHOWWrapper, BaseModel)
        
        # Check required methods exist
        wrapper = ARIHOWWrapper({})
        assert hasattr(wrapper, 'fit')
        assert hasattr(wrapper, 'predict')
        assert hasattr(wrapper, 'save')
        assert hasattr(wrapper, 'load')
        assert hasattr(wrapper, 'get_feature_importance')
        
    def test_wrapper_fit_predict(self, check_statsmodels):
        """Test ARIHOWWrapper fit and predict."""
        from src.models.arihow import ARIHOWWrapper
        
        np.random.seed(42)
        n_series = 3
        n_months = 24
        
        # Create data with series identifiers
        data = []
        for i in range(n_series):
            for m in range(n_months):
                data.append({
                    'country': f'COUNTRY_{i}',
                    'brand_name': f'BRAND_{i}',
                    'months_postgx': m,
                    'y_norm': max(0.1, 1.0 - 0.02 * m + np.random.randn() * 0.05),
                    'feature1': np.random.randn(),
                })
        
        df = pd.DataFrame(data)
        X = df[['feature1', 'months_postgx', 'country', 'brand_name']].copy()
        y = df['y_norm']
        
        X_val = X.iloc[-24:].copy()  # Last series
        y_val = y.iloc[-24:].copy()
        
        config = {
            'arima_order': (1, 0, 1),
            'hw_seasonal_periods': 12,
        }
        
        model = ARIHOWWrapper(config)
        
        # Fit should accept BaseModel signature
        model.fit(X, y, X_val, y_val, sample_weight=None)
        
        # Predict should work
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        
    def test_wrapper_save_load(self, check_statsmodels):
        """Test ARIHOWWrapper save and load."""
        from src.models.arihow import ARIHOWWrapper
        
        np.random.seed(42)
        n_samples = 48
        
        X = pd.DataFrame({
            'country': ['US'] * n_samples,
            'brand_name': ['A'] * n_samples,
            'months_postgx': list(range(24)) * 2,
            'feature1': np.random.randn(n_samples),
        })
        y = pd.Series([max(0.1, 1.0 - 0.02 * m) for m in range(24)] * 2)
        
        X_val = X.iloc[:24].copy()
        
        config = {'arima_order': (1, 0, 0)}
        model = ARIHOWWrapper(config)
        model.fit(X, y, X_val, None, None)
        
        preds_original = model.predict(X_val)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'arihow_model'
            model.save(str(save_path))
            
            # Load using classmethod
            model_loaded = ARIHOWWrapper.load(str(save_path))
            
            preds_loaded = model_loaded.predict(X_val)
            
            np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=5)


# =============================================================================
# Test 5: Model Factory Function
# =============================================================================

class TestModelFactory:
    """Tests for get_model_class factory function."""
    
    def test_get_model_class_linear(self):
        """Test get_model_class returns correct class for linear models."""
        from src.models import get_model_class, LinearModel
        
        assert get_model_class('linear') == LinearModel
        assert get_model_class('ridge') == LinearModel
        assert get_model_class('lasso') == LinearModel
        
    def test_get_model_class_baselines(self):
        """Test get_model_class returns correct class for baselines."""
        from src.models import get_model_class, GlobalMeanBaseline, FlatBaseline, TrendBaseline
        
        assert get_model_class('global_mean') == GlobalMeanBaseline
        assert get_model_class('flat') == FlatBaseline
        assert get_model_class('trend') == TrendBaseline
        
    def test_get_model_class_hybrid_wrapper(self):
        """Test get_model_class returns wrapper for hybrid models."""
        from src.models import get_model_class, HybridPhysicsMLWrapper
        
        # 'hybrid' should return wrapper by default
        assert get_model_class('hybrid') == HybridPhysicsMLWrapper
        assert get_model_class('hybrid_wrapper') == HybridPhysicsMLWrapper
        assert get_model_class('hybrid_physics_ml') == HybridPhysicsMLWrapper
        
    def test_get_model_class_arihow_wrapper(self):
        """Test get_model_class returns wrapper for arihow models."""
        from src.models import get_model_class, ARIHOWWrapper
        
        # 'arihow' should return wrapper by default
        assert get_model_class('arihow') == ARIHOWWrapper
        assert get_model_class('arihow_wrapper') == ARIHOWWrapper
        assert get_model_class('ts_hybrid') == ARIHOWWrapper
        
    def test_get_model_class_raw_models(self):
        """Test get_model_class can return raw models if needed."""
        from src.models import get_model_class, HybridPhysicsMLModel
        
        # Raw hybrid model (for direct use)
        assert get_model_class('hybrid_raw') == HybridPhysicsMLModel
        
        # Raw ARIHOW (lazy import)
        arihow_raw = get_model_class('arihow_raw')
        assert arihow_raw.__name__ == 'ARIHOWModel'
        
    def test_get_model_class_unknown_raises(self):
        """Test get_model_class raises for unknown model types."""
        from src.models import get_model_class
        
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_class('nonexistent_model')


# =============================================================================
# Test 6: Integration Tests - Full Training Pipeline
# =============================================================================

class TestIntegration:
    """Integration tests for fixed models in training pipeline."""
    
    def test_linear_model_full_pipeline(self):
        """Test LinearModel works in full training pipeline with categoricals."""
        from src.models.linear import LinearModel
        from src.train import compute_sample_weights
        
        # Create realistic data
        X, y, meta = create_synthetic_data(
            n_series=5, n_months=24, include_categoricals=True
        )
        
        # Include meta in X for sample weights
        X_full = pd.concat([X.reset_index(drop=True), 
                           meta.reset_index(drop=True)], axis=1)
        
        # Compute sample weights
        weights = compute_sample_weights(
            meta_df=meta.reset_index(drop=True),
            scenario=1,
            config=None
        )
        
        # Split train/val
        train_idx = range(0, len(X) - 48)
        val_idx = range(len(X) - 48, len(X))
        
        X_train = X_full.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X_full.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].reset_index(drop=True)
        weights_train = weights.iloc[train_idx].reset_index(drop=True)
        
        # Train model with proper config format
        config = {
            'model': {'type': 'ridge'},
            'ridge': {'alpha': 1.0},
            'preprocessing': {'handle_categoricals': 'onehot'}
        }
        model = LinearModel(config)
        model.fit(X_train, y_train, X_val, y_val, sample_weight=weights_train)
        
        # Predict
        preds = model.predict(X_val)
        
        # Basic checks
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        
        # Predictions should be in reasonable range for y_norm
        assert all(preds > 0), "y_norm predictions should be positive"
        assert all(preds < 2), "y_norm predictions should be < 2"
        
    def test_global_mean_full_pipeline(self):
        """Test GlobalMeanBaseline works in training pipeline."""
        from src.models.linear import GlobalMeanBaseline
        
        # Create data WITHOUT months_postgx in X
        X = pd.DataFrame({
            'feature1': np.random.randn(120),
            'feature2': np.random.randn(120),
        })
        y = pd.Series([1.0 - 0.02 * (i % 24) + np.random.randn() * 0.05 
                       for i in range(120)])
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(48),
            'feature2': np.random.randn(48),
        })
        
        model = GlobalMeanBaseline({})
        model.fit(X, y, X_val, None, None)
        
        preds = model.predict(X_val)
        
        assert len(preds) == len(X_val)
        assert not np.isnan(preds).any()
        
    def test_models_registered_in_factory(self):
        """Test all fixed models are properly registered in factory."""
        from src.models import get_model_class
        
        # All these should work without raising
        model_names = [
            'linear', 'ridge', 'lasso',
            'global_mean', 'flat', 'trend',
            'hybrid', 'hybrid_wrapper', 'hybrid_physics_ml',
            'arihow', 'arihow_wrapper', 'ts_hybrid',
        ]
        
        for name in model_names:
            cls = get_model_class(name)
            assert cls is not None, f"Model {name} not registered"


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
