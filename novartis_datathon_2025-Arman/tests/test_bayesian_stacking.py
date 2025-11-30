"""
Tests for Bayesian Stacking Module.

Tests cover:
- BayesianStacker correctness
- Weight optimization
- Sample weight effects
- OOF prediction alignment
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stacking.bayesian_stacking import (
    BayesianStacker,
    softmax,
    fit_dirichlet_weighted_ensemble,
    compute_sample_weight,
    compute_sample_weights_vectorized,
    build_meta_dataset_for_scenario
)


class TestSoftmax:
    """Test softmax function."""
    
    def test_softmax_sum_to_one(self):
        """Softmax output should sum to 1."""
        z = np.random.randn(5)
        w = softmax(z)
        assert np.isclose(w.sum(), 1.0)
    
    def test_softmax_positive(self):
        """Softmax output should be positive."""
        z = np.random.randn(5)
        w = softmax(z)
        assert np.all(w > 0)
    
    def test_softmax_numerical_stability(self):
        """Softmax should be stable for large values."""
        z = np.array([1000.0, 1000.0, 1000.0])
        w = softmax(z)
        assert np.isclose(w.sum(), 1.0)
        assert np.allclose(w, 1/3)


class TestComputeSampleWeight:
    """Test sample weight computation."""
    
    def test_scenario1_early_months(self):
        """Early months in S1 should have higher weight."""
        w_early = compute_sample_weight(month=3, bucket=2, scenario=1)
        w_mid = compute_sample_weight(month=9, bucket=2, scenario=1)
        w_late = compute_sample_weight(month=15, bucket=2, scenario=1)
        
        # Mid should be highest, early next, late lowest
        assert w_mid > w_early > w_late
    
    def test_scenario2_mid_months(self):
        """Mid months in S2 should have highest weight."""
        w_mid = compute_sample_weight(month=9, bucket=2, scenario=2)
        w_late = compute_sample_weight(month=15, bucket=2, scenario=2)
        
        assert w_mid > w_late
    
    def test_bucket1_higher_weight(self):
        """Bucket 1 should have 2x weight of Bucket 2."""
        w_b1 = compute_sample_weight(month=6, bucket=1, scenario=1)
        w_b2 = compute_sample_weight(month=6, bucket=2, scenario=1)
        
        assert w_b1 == 2 * w_b2
    
    def test_vectorized_matches_scalar(self):
        """Vectorized version should match scalar version."""
        months = np.array([3, 9, 15])
        buckets = np.array([1, 2, 1])
        scenario = 1
        
        vectorized = compute_sample_weights_vectorized(months, buckets, scenario)
        
        for i in range(len(months)):
            scalar = compute_sample_weight(months[i], buckets[i], scenario)
            assert np.isclose(vectorized[i], scalar)


class TestFitDirichletWeightedEnsemble:
    """Test Dirichlet-weighted ensemble fitting."""
    
    def test_weights_sum_to_one(self):
        """Fitted weights should sum to 1."""
        np.random.seed(42)
        N, M = 100, 3
        X = np.random.randn(N, M)
        y = X[:, 0] + 0.1 * np.random.randn(N)  # Model 0 is best
        sample_weight = np.ones(N)
        
        weights = fit_dirichlet_weighted_ensemble(X, y, sample_weight)
        
        assert np.isclose(weights.sum(), 1.0)
    
    def test_weights_positive(self):
        """All weights should be positive."""
        np.random.seed(42)
        N, M = 100, 3
        X = np.random.randn(N, M)
        y = X[:, 0] + 0.1 * np.random.randn(N)
        sample_weight = np.ones(N)
        
        weights = fit_dirichlet_weighted_ensemble(X, y, sample_weight)
        
        assert np.all(weights > 0)
    
    def test_best_model_gets_higher_weight(self):
        """Best model should get higher weight."""
        np.random.seed(42)
        N, M = 500, 3
        
        # Model 0 is perfect, others are noise
        y = np.random.randn(N)
        X = np.column_stack([
            y,  # Perfect model
            np.random.randn(N),  # Random
            np.random.randn(N),  # Random
        ])
        sample_weight = np.ones(N)
        
        weights = fit_dirichlet_weighted_ensemble(X, y, sample_weight)
        
        # Model 0 should have highest weight
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]
    
    def test_sample_weight_effect(self):
        """Sample weights should affect final weights."""
        np.random.seed(42)
        N, M = 200, 2
        
        # First half: model 0 is better
        # Second half: model 1 is better
        y = np.concatenate([np.ones(N//2), np.zeros(N//2)])
        X = np.column_stack([
            np.concatenate([np.ones(N//2), 0.5*np.ones(N//2)]),  # Model 0
            np.concatenate([0.5*np.ones(N//2), np.zeros(N//2)]),  # Model 1
        ])
        
        # Equal weights -> both models matter
        sample_weight_equal = np.ones(N)
        weights_equal = fit_dirichlet_weighted_ensemble(X, y, sample_weight_equal)
        
        # Weight first half more -> model 0 should dominate
        sample_weight_first = np.concatenate([10*np.ones(N//2), np.ones(N//2)])
        weights_first = fit_dirichlet_weighted_ensemble(X, y, sample_weight_first)
        
        # Model 0 should have higher weight when first half is weighted more
        assert weights_first[0] > weights_equal[0]


class TestBayesianStacker:
    """Test BayesianStacker class."""
    
    def test_fit_and_predict(self):
        """Basic fit and predict workflow."""
        np.random.seed(42)
        N, M = 100, 3
        X = np.random.randn(N, M)
        y = X.mean(axis=1) + 0.1 * np.random.randn(N)
        
        stacker = BayesianStacker()
        stacker.fit(X, y)
        
        predictions = stacker.predict(X)
        
        assert len(predictions) == N
        assert stacker.weights_ is not None
        assert len(stacker.weights_) == M
    
    def test_predict_before_fit_raises(self):
        """Calling predict before fit should raise error."""
        stacker = BayesianStacker()
        X = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="not fitted"):
            stacker.predict(X)
    
    def test_clip_predictions(self):
        """Predictions should be clipped when enabled."""
        np.random.seed(42)
        N, M = 100, 3
        X = 5 * np.random.randn(N, M)  # Large values
        y = X.mean(axis=1)
        
        stacker = BayesianStacker(clip_predictions=True, clip_min=0.0, clip_max=2.0)
        stacker.fit(X, y)
        
        predictions = stacker.predict(X)
        
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 2.0)
    
    def test_get_weights_dict(self):
        """get_weights_dict should return correct mapping."""
        np.random.seed(42)
        N, M = 100, 3
        X = np.random.randn(N, M)
        y = X.mean(axis=1)
        model_names = ['cat', 'lgbm', 'xgb']
        
        stacker = BayesianStacker()
        stacker.fit(X, y, model_names=model_names)
        
        weights_dict = stacker.get_weights_dict()
        
        assert set(weights_dict.keys()) == set(model_names)
        assert np.isclose(sum(weights_dict.values()), 1.0)
    
    def test_save_and_load(self, tmp_path):
        """Save and load should preserve state."""
        np.random.seed(42)
        N, M = 100, 3
        X = np.random.randn(N, M)
        y = X.mean(axis=1)
        
        stacker = BayesianStacker()
        stacker.fit(X, y, model_names=['m1', 'm2', 'm3'])
        
        save_path = tmp_path / 'stacker.joblib'
        stacker.save(str(save_path))
        
        loaded = BayesianStacker.load(str(save_path))
        
        assert np.allclose(loaded.weights_, stacker.weights_)
        assert loaded.model_names_ == stacker.model_names_


class TestMetaDatasetBuilding:
    """Test meta-dataset building."""
    
    def test_oof_alignment(self, tmp_path):
        """OOF predictions should align correctly."""
        # Create mock OOF files
        df1 = pd.DataFrame({
            'country': ['A', 'A', 'B', 'B'],
            'brand_name': ['X', 'X', 'Y', 'Y'],
            'months_postgx': [0, 1, 0, 1],
            'y_true': [0.9, 0.8, 0.7, 0.6],
            'y_pred': [0.85, 0.75, 0.65, 0.55],
            'bucket': [1, 1, 2, 2]
        })
        
        df2 = pd.DataFrame({
            'country': ['A', 'A', 'B', 'B'],
            'brand_name': ['X', 'X', 'Y', 'Y'],
            'months_postgx': [0, 1, 0, 1],
            'y_true': [0.9, 0.8, 0.7, 0.6],
            'y_pred': [0.88, 0.78, 0.68, 0.58],
            'bucket': [1, 1, 2, 2]
        })
        
        # Save to temp files
        path1 = tmp_path / 'model1.parquet'
        path2 = tmp_path / 'model2.parquet'
        df1.to_parquet(path1)
        df2.to_parquet(path2)
        
        # Build meta-dataset
        oof_files = {'model1': str(path1), 'model2': str(path2)}
        df_meta = build_meta_dataset_for_scenario(oof_files, scenario=1)
        
        # Check alignment
        assert len(df_meta) == 4
        assert 'pred_model1' in df_meta.columns
        assert 'pred_model2' in df_meta.columns
        assert 'y_true' in df_meta.columns
        assert 'sample_weight' in df_meta.columns
        
        # y_true should be identical
        assert np.allclose(df_meta['y_true'], df1['y_true'])


class TestHierarchicalBayesianDecay:
    """Test Hierarchical Bayesian Decay model (Section 7)."""
    
    def test_fit_and_predict(self):
        """Basic fit and predict workflow."""
        from src.stacking.bayesian_stacking import HierarchicalBayesianDecay
        
        # Create synthetic decay data
        np.random.seed(42)
        data = []
        for brand in ['Brand_A', 'Brand_B', 'Brand_C']:
            a, b, c = 1.0, 0.05, 0.3
            for t in range(24):
                y = a * np.exp(-b * t) + c + 0.02 * np.random.randn()
                data.append({
                    'country': 'USA',
                    'brand_name': brand,
                    'months_postgx': t,
                    'y_norm': y,
                    'bucket': 1
                })
        
        df = pd.DataFrame(data)
        
        model = HierarchicalBayesianDecay(use_hierarchical_priors=True)
        model.fit(df, target_col='y_norm')
        
        predictions = model.predict_fast(df)
        
        assert len(predictions) == len(df)
        assert np.all(predictions >= 0)
    
    def test_hierarchical_priors_learned(self):
        """Model should learn hierarchical priors from data."""
        from src.stacking.bayesian_stacking import HierarchicalBayesianDecay
        
        np.random.seed(42)
        data = []
        
        # Bucket 1: faster decay
        for brand in ['A', 'B', 'C']:
            a, b, c = 1.0, 0.08, 0.2  # Higher b = faster decay
            for t in range(24):
                y = a * np.exp(-b * t) + c + 0.01 * np.random.randn()
                data.append({
                    'country': 'USA', 'brand_name': f'B1_{brand}',
                    'months_postgx': t, 'y_norm': y, 'bucket': 1
                })
        
        # Bucket 2: slower decay
        for brand in ['X', 'Y', 'Z']:
            a, b, c = 1.0, 0.03, 0.4  # Lower b = slower decay
            for t in range(24):
                y = a * np.exp(-b * t) + c + 0.01 * np.random.randn()
                data.append({
                    'country': 'USA', 'brand_name': f'B2_{brand}',
                    'months_postgx': t, 'y_norm': y, 'bucket': 2
                })
        
        df = pd.DataFrame(data)
        
        model = HierarchicalBayesianDecay(use_hierarchical_priors=True)
        model.fit(df, target_col='y_norm')
        
        # Check that bucket priors were learned
        assert 1 in model.bucket_priors_
        assert 2 in model.bucket_priors_
        
        # Bucket 1 should have higher decay rate (b) prior
        b1_mean = model.bucket_priors_[1]['b'][0]
        b2_mean = model.bucket_priors_[2]['b'][0]
        assert b1_mean > b2_mean
    
    def test_save_and_load(self, tmp_path):
        """Save and load should preserve state."""
        from src.stacking.bayesian_stacking import HierarchicalBayesianDecay
        
        np.random.seed(42)
        data = []
        for brand in ['A', 'B']:
            for t in range(12):
                data.append({
                    'country': 'USA', 'brand_name': brand,
                    'months_postgx': t, 'y_norm': np.exp(-0.05 * t),
                    'bucket': 1
                })
        df = pd.DataFrame(data)
        
        model = HierarchicalBayesianDecay()
        model.fit(df, target_col='y_norm')
        
        save_path = tmp_path / 'decay.joblib'
        model.save(str(save_path))
        
        loaded = HierarchicalBayesianDecay.load(str(save_path))
        
        assert loaded.brand_params_ == model.brand_params_


class TestDiversification:
    """Test submission diversification methods (Section 6)."""
    
    def test_multi_init_returns_diverse_weights(self):
        """Multi-init should return diverse weight configurations."""
        from src.stacking.bayesian_stacking import fit_stacker_multi_init
        
        np.random.seed(42)
        N, M = 200, 4
        X = np.random.randn(N, M)
        y = X.mean(axis=1) + 0.1 * np.random.randn(N)
        sample_weight = np.ones(N)
        
        all_weights = fit_stacker_multi_init(X, y, sample_weight, n_inits=10)
        
        assert len(all_weights) >= 1
        
        # All should sum to 1
        for w in all_weights:
            assert np.isclose(w.sum(), 1.0)
    
    def test_mcmc_sampling(self):
        """MCMC should return multiple weight samples."""
        from src.stacking.bayesian_stacking import mcmc_sample_weights
        
        np.random.seed(42)
        N, M = 100, 3
        X = np.random.randn(N, M)
        y = X.mean(axis=1) + 0.1 * np.random.randn(N)
        sample_weight = np.ones(N)
        
        samples = mcmc_sample_weights(
            X, y, sample_weight,
            n_samples=10,
            burn_in=20,
            thin=2
        )
        
        assert len(samples) == 10
        
        # All samples should be valid probability vectors
        for w in samples:
            assert np.isclose(w.sum(), 1.0)
            assert np.all(w >= 0)
    
    def test_create_blend_of_blends(self):
        """Blend should combine multiple submissions correctly."""
        from src.stacking.bayesian_stacking import create_blend_of_blends
        
        # Create mock submissions
        base_df = pd.DataFrame({
            'country': ['A', 'A', 'B'],
            'brand_name': ['X', 'X', 'Y'],
            'months_postgx': [0, 1, 0]
        })
        
        sub1 = base_df.copy()
        sub1['volume'] = [0.9, 0.8, 0.7]
        
        sub2 = base_df.copy()
        sub2['volume'] = [0.8, 0.7, 0.6]
        
        # Equal weights blend
        blend = create_blend_of_blends([sub1, sub2])
        
        expected = [0.85, 0.75, 0.65]
        assert np.allclose(blend['volume'].values, expected)
    
    def test_create_blend_with_custom_weights(self):
        """Blend should respect custom weights."""
        from src.stacking.bayesian_stacking import create_blend_of_blends
        
        base_df = pd.DataFrame({
            'country': ['A'],
            'brand_name': ['X'],
            'months_postgx': [0]
        })
        
        sub1 = base_df.copy()
        sub1['volume'] = [1.0]
        
        sub2 = base_df.copy()
        sub2['volume'] = [0.0]
        
        # 75% weight on sub1, 25% on sub2
        blend = create_blend_of_blends([sub1, sub2], weights=[0.75, 0.25])
        
        assert np.isclose(blend['volume'].values[0], 0.75)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
