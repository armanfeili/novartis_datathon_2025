"""
Unit tests for config_sweep module.

Tests the hyperparameter sweep functionality including:
- Nested value access (get/set)
- Sweep axis detection
- Configuration expansion (Cartesian product)
- Run ID generation
"""

import pytest
import copy

from src.config_sweep import (
    expand_sweep,
    get_sweep_axes,
    build_sweep_run_id,
    get_nested_value,
    set_nested_value,
    SWEEPABLE_KEYS,
    detect_list_axes,
    generate_sweep_suffix
)


class TestNestedValueAccess:
    """Tests for get_nested_value and set_nested_value."""
    
    def test_get_nested_value_simple(self):
        """Test getting a simple top-level value."""
        config = {'depth': 6, 'learning_rate': 0.03}
        assert get_nested_value(config, 'depth') == 6
        assert get_nested_value(config, 'learning_rate') == 0.03
    
    def test_get_nested_value_dotted_path(self):
        """Test getting a value with dotted path."""
        config = {
            'params': {
                'depth': 6,
                'learning_rate': 0.03
            }
        }
        assert get_nested_value(config, 'params.depth') == 6
        assert get_nested_value(config, 'params.learning_rate') == 0.03
    
    def test_get_nested_value_deep(self):
        """Test getting a deeply nested value."""
        config = {
            'model': {
                'params': {
                    'tree': {
                        'depth': 8
                    }
                }
            }
        }
        assert get_nested_value(config, 'model.params.tree.depth') == 8
    
    def test_get_nested_value_missing(self):
        """Test getting a missing value returns None."""
        config = {'params': {'depth': 6}}
        assert get_nested_value(config, 'params.missing') is None
        assert get_nested_value(config, 'missing.path') is None
    
    def test_get_nested_value_with_default(self):
        """Test getting a missing value with default."""
        config = {'params': {'depth': 6}}
        assert get_nested_value(config, 'params.missing', default=10) == 10
    
    def test_set_nested_value_simple(self):
        """Test setting a simple top-level value."""
        config = {'depth': 6}
        set_nested_value(config, 'depth', 8)
        assert config['depth'] == 8
    
    def test_set_nested_value_dotted_path(self):
        """Test setting a value with dotted path."""
        config = {'params': {'depth': 6}}
        set_nested_value(config, 'params.depth', 8)
        assert config['params']['depth'] == 8
    
    def test_set_nested_value_creates_path(self):
        """Test that set creates intermediate dictionaries."""
        config = {}
        set_nested_value(config, 'params.depth', 8)
        assert config['params']['depth'] == 8
    
    def test_set_nested_value_deep_creates_path(self):
        """Test creating deeply nested path."""
        config = {}
        set_nested_value(config, 'model.params.tree.depth', 10)
        assert config['model']['params']['tree']['depth'] == 10


class TestGetSweepAxes:
    """Tests for get_sweep_axes."""
    
    def test_get_sweep_axes_from_config(self):
        """Test that explicit sweep.axes in config is returned."""
        config = {
            'sweep': {
                'enabled': True,
                'axes': ['params.depth', 'params.learning_rate']
            },
            'params': {
                'depth': [4, 6, 8],
                'learning_rate': 0.03
            }
        }
        axes = get_sweep_axes(config)
        assert 'params.depth' in axes
        assert 'params.learning_rate' in axes
    
    def test_get_sweep_axes_disabled(self):
        """Test that disabled sweep returns empty list."""
        config = {
            'sweep': {
                'enabled': False,
                'axes': ['params.depth']
            },
            'params': {
                'depth': [4, 6, 8]
            }
        }
        axes = get_sweep_axes(config)
        assert len(axes) == 0
    
    def test_sweepable_keys_defined(self):
        """Test that SWEEPABLE_KEYS contains expected params."""
        # Check that we have common parameters
        assert 'params.depth' in SWEEPABLE_KEYS
        assert 'params.learning_rate' in SWEEPABLE_KEYS
        assert 'params.num_leaves' in SWEEPABLE_KEYS
        assert 'params.max_depth' in SWEEPABLE_KEYS


class TestExpandSweep:
    """Tests for expand_sweep."""
    
    def test_expand_sweep_no_lists(self):
        """Test that config without lists returns single config."""
        config = {
            'params': {
                'depth': 6,
                'learning_rate': 0.03
            }
        }
        expanded = expand_sweep(config, ['params.depth', 'params.learning_rate'])
        assert len(expanded) == 1
        assert expanded[0]['params']['depth'] == 6
        assert expanded[0]['params']['learning_rate'] == 0.03
    
    def test_expand_sweep_single_axis(self):
        """Test expansion with single list axis."""
        config = {
            'params': {
                'depth': [4, 6, 8],
                'learning_rate': 0.03
            }
        }
        expanded = expand_sweep(config, ['params.depth'])
        
        assert len(expanded) == 3
        depths = [c['params']['depth'] for c in expanded]
        assert sorted(depths) == [4, 6, 8]
        
        # All should have same learning_rate
        for c in expanded:
            assert c['params']['learning_rate'] == 0.03
    
    def test_expand_sweep_cartesian_product(self):
        """Test Cartesian product of multiple axes."""
        config = {
            'params': {
                'depth': [4, 6],
                'learning_rate': [0.01, 0.03]
            }
        }
        expanded = expand_sweep(config, ['params.depth', 'params.learning_rate'])
        
        # 2 x 2 = 4 combinations
        assert len(expanded) == 4
        
        # Check all combinations present
        combos = [(c['params']['depth'], c['params']['learning_rate']) for c in expanded]
        expected = [(4, 0.01), (4, 0.03), (6, 0.01), (6, 0.03)]
        assert sorted(combos) == sorted(expected)
    
    def test_expand_sweep_three_axes(self):
        """Test expansion with three axes."""
        config = {
            'params': {
                'depth': [4, 6],
                'learning_rate': [0.01, 0.03],
                'l2_leaf_reg': [1.0, 3.0]
            }
        }
        axes = ['params.depth', 'params.learning_rate', 'params.l2_leaf_reg']
        expanded = expand_sweep(config, axes)
        
        # 2 x 2 x 2 = 8 combinations
        assert len(expanded) == 8
    
    def test_expand_sweep_preserves_other_params(self):
        """Test that non-swept params are preserved."""
        config = {
            'model': {'name': 'catboost'},
            'params': {
                'depth': [4, 6],
                'learning_rate': 0.03,
                'iterations': 1000
            }
        }
        expanded = expand_sweep(config, ['params.depth'])
        
        for c in expanded:
            assert c['model']['name'] == 'catboost'
            assert c['params']['learning_rate'] == 0.03
            assert c['params']['iterations'] == 1000
    
    def test_expand_sweep_metadata(self):
        """Test that _sweep_metadata is added."""
        config = {
            'params': {
                'depth': [4, 6]
            }
        }
        expanded = expand_sweep(config, ['params.depth'])
        
        for c in expanded:
            assert '_sweep_metadata' in c
            assert 'axes' in c['_sweep_metadata']
            assert 'params.depth' in c['_sweep_metadata']['axes']
    
    def test_expand_sweep_no_mutation(self):
        """Test that original config is not mutated."""
        config = {
            'params': {
                'depth': [4, 6]
            }
        }
        original = copy.deepcopy(config)
        expand_sweep(config, ['params.depth'])
        
        assert config == original


class TestGenerateSweepRunId:
    """Tests for build_sweep_run_id."""
    
    def test_generate_sweep_run_id_basic(self):
        """Test basic run ID generation."""
        sweep_meta = {
            'is_sweep': True,
            'axes': {
                'params.depth': 6,
                'params.learning_rate': 0.03
            },
            'index': 0,
            'total': 4
        }
        run_id = build_sweep_run_id('base_run', sweep_meta)
        
        assert 'base_run' in run_id
        assert 'depth6' in run_id or 'd6' in run_id
    
    def test_generate_sweep_run_id_empty_meta(self):
        """Test run ID with empty metadata."""
        run_id = build_sweep_run_id('base_run', {'is_sweep': False, 'axes': {}})
        assert run_id == 'base_run'
    
    def test_generate_sweep_run_id_none_meta(self):
        """Test run ID with None metadata."""
        run_id = build_sweep_run_id('base_run', None)
        assert run_id == 'base_run'
    
    def test_generate_sweep_run_id_filesystem_safe(self):
        """Test that generated ID is filesystem safe."""
        sweep_meta = {
            'is_sweep': True,
            'axes': {
                'params.learning_rate': 0.001
            },
            'index': 0,
            'total': 1
        }
        run_id = build_sweep_run_id('base_run', sweep_meta)
        
        # Should not contain problematic characters
        assert '/' not in run_id
        assert '\\' not in run_id
        assert ':' not in run_id
        assert '*' not in run_id
        assert '?' not in run_id


class TestIntegration:
    """Integration tests for the sweep functionality."""
    
    def test_full_sweep_workflow(self):
        """Test the complete sweep workflow."""
        config = {
            'model': {'name': 'catboost'},
            'sweep': {
                'enabled': True,
                'axes': ['params.depth', 'params.learning_rate']
            },
            'params': {
                'depth': [4, 6],
                'learning_rate': [0.01, 0.03],
                'iterations': 1000,
                'random_seed': 42
            }
        }
        
        # Get axes
        axes = get_sweep_axes(config)
        assert 'params.depth' in axes
        assert 'params.learning_rate' in axes
        
        # Expand
        expanded = expand_sweep(config, axes)
        assert len(expanded) == 4
        
        # Generate run IDs
        base_name = 'test_catboost_s1'
        run_ids = []
        for c in expanded:
            meta = c.get('_sweep_metadata', {})
            run_id = build_sweep_run_id(base_name, meta)
            run_ids.append(run_id)
            
            # Verify config is valid (single values)
            assert isinstance(c['params']['depth'], int)
            assert isinstance(c['params']['learning_rate'], float)
        
        # All run IDs should be unique
        assert len(set(run_ids)) == 4
    
    def test_sweep_detects_list_axes(self):
        """Test that detect_list_axes works correctly."""
        config = {
            'params': {
                'depth': [4, 6, 8],
                'learning_rate': 0.03,  # Not a list
                'l2_leaf_reg': [1.0, 3.0]
            }
        }
        axes = ['params.depth', 'params.learning_rate', 'params.l2_leaf_reg']
        list_axes = detect_list_axes(config, axes)
        
        assert 'params.depth' in list_axes
        assert 'params.l2_leaf_reg' in list_axes
        assert 'params.learning_rate' not in list_axes  # Not a list
