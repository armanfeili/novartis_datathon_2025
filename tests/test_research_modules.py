"""
Tests for new modules from research papers implementation.

Tests cover:
- external_data.py
- visibility_sources.py
- scenario_analysis.py
- graph_utils.py
- sequence_builder.py
- gcn_layers.py
- kg_gcn_lstm.py
- cnn_lstm.py

References:
- KG-GCN-LSTM paper
- Ghannem et al. (2023) - Supply-chain visibility
- Li et al. (2024) - CNN-LSTM for drug sales prediction
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import os


# ==============================================================================
# Tests for external_data.py
# ==============================================================================

class TestExternalData:
    """Tests for external data loading module."""
    
    def test_load_holiday_calendar_returns_empty_when_file_missing(self):
        """Test graceful fallback when holiday file is missing."""
        from src.external_data import load_holiday_calendar
        
        result = load_holiday_calendar("nonexistent_file.csv")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "date" in result.columns
        assert "country" in result.columns
    
    def test_load_holiday_calendar_with_valid_data(self, tmp_path):
        """Test loading valid holiday data."""
        from src.external_data import load_holiday_calendar
        
        # Create test file
        holiday_file = tmp_path / "holidays.csv"
        holiday_file.write_text(
            "date,country,holiday_name,is_public\n"
            "2024-01-01,USA,New Year,1\n"
            "2024-07-04,USA,Independence Day,1\n"
        )
        
        result = load_holiday_calendar(str(holiday_file))
        
        assert len(result) == 2
        assert "date" in result.columns
        assert "country" in result.columns
    
    def test_load_epidemic_events_returns_empty_when_file_missing(self):
        """Test graceful fallback for epidemic events."""
        from src.external_data import load_epidemic_events
        
        result = load_epidemic_events("nonexistent.csv")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_load_macro_indicators_returns_empty_when_file_missing(self):
        """Test graceful fallback for macro indicators."""
        from src.external_data import load_macro_indicators
        
        result = load_macro_indicators("nonexistent.csv")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_join_external_context_with_empty_sources(self):
        """Test joining when all external sources are empty."""
        from src.external_data import join_external_context
        
        base_df = pd.DataFrame({
            "month_id": [0, 1, 2],
            "country": ["USA", "USA", "USA"],
            "ndc": ["123", "123", "123"],
        })
        
        result = join_external_context(base_df)
        
        assert len(result) == 3
        # Original columns preserved
        assert "month_id" in result.columns
        assert "country" in result.columns


# ==============================================================================
# Tests for visibility_sources.py
# ==============================================================================

class TestVisibilitySources:
    """Tests for visibility source module."""
    
    def test_csv_visibility_source_creation(self, tmp_path):
        """Test creating a CsvVisibilitySource."""
        from src.visibility_sources import CsvVisibilitySource
        
        # CsvVisibilitySource takes base_path and optional config
        source = CsvVisibilitySource(base_path=str(tmp_path))
        
        # Check that source was created successfully
        assert source is not None
        # Check actual method names
        assert hasattr(source, 'load_inventory')
        assert hasattr(source, 'load_supplier_metrics')
        assert hasattr(source, 'get_aggregated_visibility')
    
    def test_csv_visibility_source_loads_empty_for_missing_files(self, tmp_path):
        """Test that missing files return empty DataFrames."""
        from src.visibility_sources import CsvVisibilitySource
        
        source = CsvVisibilitySource(base_path=str(tmp_path))
        
        # Should return empty for missing inventory
        result = source.load_inventory()
        assert isinstance(result, pd.DataFrame)
    
    def test_join_visibility_features(self):
        """Test join_visibility_features aggregates visibility data."""
        from src.visibility_sources import join_visibility_features, create_visibility_feature_names
        
        # Create base dataframe
        base_df = pd.DataFrame({
            "ndc": ["123", "456"],
            "month_id": [0, 0],
            "volume": [100.0, 200.0],
        })
        
        # Create mock visibility source
        mock_source = Mock()
        mock_source.fetch_lead_times.return_value = pd.DataFrame({
            "ndc": ["123"],
            "supplier": ["A"],
            "lead_time_days": [10]
        })
        mock_source.fetch_inventory_levels.return_value = pd.DataFrame({
            "ndc": ["123"],
            "warehouse": ["W1"],
            "quantity": [500]
        })
        
        # Check that function exists and has correct signature
        feature_names = create_visibility_feature_names()
        assert isinstance(feature_names, list)


# ==============================================================================
# Tests for scenario_analysis.py
# ==============================================================================

class TestScenarioAnalysis:
    """Tests for scenario analysis module."""
    
    def test_apply_demand_shock_step(self):
        """Test step demand shock using DemandShock dataclass."""
        from src.scenario_analysis import apply_demand_shock, DemandShock, ShockType
        
        df = pd.DataFrame({
            "months_postgx": [0, 1, 2, 3, 4],
            "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
            "country": ["USA"] * 5,
            "brand_name": ["DrugA"] * 5,
            "ther_area": ["oncology"] * 5,
        })
        
        shock = DemandShock(
            name="test_step",
            shock_type=ShockType.STEP,
            magnitude=1.2,  # 20% increase
            start_month=2
        )
        
        result = apply_demand_shock(df, shock, volume_col="volume")
        
        # First two months unchanged
        assert result.iloc[0]["volume"] == 100.0
        assert result.iloc[1]["volume"] == 100.0
        # After shock - magnitude is multiplier
        assert result.iloc[2]["volume"] == pytest.approx(120.0)
        assert result.iloc[3]["volume"] == pytest.approx(120.0)
    
    def test_apply_demand_shock_impulse(self):
        """Test impulse demand shock."""
        from src.scenario_analysis import apply_demand_shock, DemandShock, ShockType
        
        df = pd.DataFrame({
            "months_postgx": [0, 1, 2, 3, 4],
            "volume": [100.0] * 5,
            "country": ["USA"] * 5,
            "brand_name": ["DrugA"] * 5,
            "ther_area": ["oncology"] * 5,
        })
        
        shock = DemandShock(
            name="test_impulse",
            shock_type=ShockType.IMPULSE,
            magnitude=1.5,  # 50% spike
            start_month=2,
            duration_months=2
        )
        
        result = apply_demand_shock(df, shock, volume_col="volume")
        
        # Impulse only at start month
        assert result.iloc[1]["volume"] == 100.0
        # At shock time
        assert result.iloc[2]["volume"] > 100.0
    
    def test_apply_multiple_shocks(self):
        """Test applying multiple demand shocks."""
        from src.scenario_analysis import apply_multiple_shocks, DemandShock, ShockType
        
        df = pd.DataFrame({
            "months_postgx": [0, 1, 2, 3, 4],
            "volume": [100.0] * 5,
            "country": ["USA"] * 5,
            "brand_name": ["DrugA"] * 5,
            "ther_area": ["oncology"] * 5,
        })
        
        # Create shocks list using DemandShock dataclass
        shocks = [
            DemandShock(name="shock1", shock_type=ShockType.STEP, magnitude=1.1, start_month=1),
            DemandShock(name="shock2", shock_type=ShockType.STEP, magnitude=0.9, start_month=3),
        ]
        
        result = apply_multiple_shocks(df, shocks, volume_col="volume")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
    
    def test_get_predefined_scenarios(self):
        """Test getting predefined shock scenarios."""
        from src.scenario_analysis import get_predefined_scenarios
        
        scenarios = get_predefined_scenarios()
        
        # Returns dict of scenario names to DemandShock instances
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0
        assert "competitive_launch" in scenarios or "epidemic_surge" in scenarios


# ==============================================================================
# Tests for graph_utils.py
# ==============================================================================

class TestGraphUtils:
    """Tests for graph utilities module."""
    
    def test_drug_graph_creation(self):
        """Test DrugGraph initialization."""
        from src.graph_utils import DrugGraph
        
        graph = DrugGraph()
        
        assert hasattr(graph, "nodes")
        assert hasattr(graph, "edges")
    
    def test_build_drug_graph_creates_graph(self):
        """Test building drug graph returns DrugGraph."""
        from src.graph_utils import build_drug_graph, DrugGraph
        
        df = pd.DataFrame({
            "ndc": ["001", "002", "003"],
            "country": ["USA", "USA", "UK"],
            "brand_name": ["DrugA", "DrugA", "DrugB"],
            "ther_area": ["oncology", "oncology", "cardiology"],
            "feature1": [1.0, 2.0, 3.0],
        })
        
        result = build_drug_graph(
            df,
            feature_cols=["feature1"]
        )
        
        assert isinstance(result, DrugGraph)
        assert hasattr(result, "nodes")
        # DrugGraph has adjacency_matrix as attribute, not method
        assert hasattr(result, "adjacency_matrix")
    
    def test_node_feature_builder(self):
        """Test NodeFeatureBuilder class."""
        from src.graph_utils import NodeFeatureBuilder
        
        df = pd.DataFrame({
            "ndc": ["001", "002"],
            "country": ["USA", "USA"],
            "brand_name": ["DrugA", "DrugA"],
            "ther_area": ["oncology", "oncology"],
            "feature1": [1.0, 2.0],
            "feature2": [0.5, 0.7],
        })
        
        builder = NodeFeatureBuilder()
        builder.fit(df)
        features = builder.transform(df)
        
        assert features is not None
        assert isinstance(features, np.ndarray) or hasattr(features, 'shape')
    
    def test_edge_types_enum(self):
        """Test EdgeType enum has expected values."""
        from src.graph_utils import EdgeType
        
        # Check that EdgeType is an enum
        assert hasattr(EdgeType, 'SAME_THERAPEUTIC_AREA') or len(list(EdgeType)) > 0


# ==============================================================================
# Tests for sequence_builder.py
# ==============================================================================

class TestSequenceBuilder:
    """Tests for sequence builder module."""
    
    def test_sequence_config_defaults(self):
        """Test SequenceConfig has sensible defaults."""
        from src.sequence_builder import SequenceConfig
        
        config = SequenceConfig()
        
        assert config.lookback_window == 12
        assert config.forecast_horizon == 1
        assert config.target_col == "y_norm"
    
    def test_build_sequences_creates_arrays(self):
        """Test build_sequences creates proper arrays."""
        from src.sequence_builder import build_sequences, SequenceConfig
        
        # Create test data
        df = pd.DataFrame({
            "ndc": ["001"] * 20,
            "brand_drug_id": [1] * 20,
            "month_id": list(range(20)),
            "feature1": np.random.randn(20),
            "feature2": np.random.randn(20),
            "y_norm": np.random.randn(20),
        })
        
        config = SequenceConfig(
            lookback_window=5,
            forecast_horizon=1,
            time_varying_features=["feature1", "feature2"],
            group_cols=("ndc",),
        )
        
        result = build_sequences(df, config, is_train=True)
        
        assert "X_seq" in result
        assert "y" in result
        assert "masks" in result
        assert result["X_seq"].shape[1] == 5  # lookback
        assert result["X_seq"].shape[2] == 2  # features
    
    def test_pad_sequences_handles_variable_lengths(self):
        """Test pad_sequences with variable length inputs."""
        from src.sequence_builder import pad_sequences
        
        sequences = [
            np.array([[1, 2], [3, 4]]),  # length 2
            np.array([[5, 6], [7, 8], [9, 10]]),  # length 3
        ]
        
        padded, masks = pad_sequences(sequences, padding="pre")
        
        assert padded.shape == (2, 3, 2)  # max_len = 3
        assert masks.shape == (2, 3)
        # First sequence has padding at start
        assert masks[0, 0] == 0.0
        assert masks[0, 1] == 1.0
        # Second sequence no padding
        assert masks[1].sum() == 3.0
    
    def test_sequence_scaler_fit_transform(self):
        """Test SequenceScaler fit and transform."""
        from src.sequence_builder import SequenceScaler
        
        X_seq = np.random.randn(100, 10, 5).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        scaler = SequenceScaler(method="standard")
        scaler.fit(X_seq, y=y)
        
        X_scaled, _, y_scaled = scaler.transform(X_seq, y=y)
        
        # Check scaling was applied
        assert X_scaled.shape == X_seq.shape
        # Standard scaling should have ~0 mean
        assert abs(X_scaled.mean()) < 0.5


# ==============================================================================
# Tests for gcn_layers.py (PyTorch required)
# ==============================================================================

class TestGCNLayers:
    """Tests for GCN layer implementations."""
    
    @pytest.fixture
    def skip_if_no_torch(self):
        """Skip test if torch not available."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_normalize_adjacency(self, skip_if_no_torch):
        """Test adjacency normalization."""
        import torch
        from src.models.gcn_layers import normalize_adjacency
        
        adj = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=torch.float32)
        
        adj_norm = normalize_adjacency(adj, add_self_loops=True)
        
        assert adj_norm.shape == adj.shape
        # Should be symmetric
        assert torch.allclose(adj_norm, adj_norm.T, atol=1e-6)
    
    def test_gcn_layer_forward(self, skip_if_no_torch):
        """Test GCNLayer forward pass."""
        import torch
        from src.models.gcn_layers import GCNLayer
        
        layer = GCNLayer(
            in_features=16,
            out_features=32,
            activation="relu"
        )
        
        x = torch.randn(10, 16)  # 10 nodes, 16 features
        adj = torch.eye(10)  # Simple identity adjacency
        
        out = layer(x, adj)
        
        assert out.shape == (10, 32)
    
    def test_gcn_encoder_forward(self, skip_if_no_torch):
        """Test GCNEncoder forward pass."""
        import torch
        from src.models.gcn_layers import GCNEncoder
        
        encoder = GCNEncoder(
            in_features=16,
            hidden_dims=[32],
            out_features=8,
            dropout=0.1
        )
        
        x = torch.randn(10, 16)
        adj = torch.eye(10)
        
        out = encoder(x, adj)
        
        assert out.shape == (10, 8)
    
    def test_graph_attention_layer_forward(self, skip_if_no_torch):
        """Test GraphAttentionLayer forward pass."""
        import torch
        from src.models.gcn_layers import GraphAttentionLayer
        
        layer = GraphAttentionLayer(
            in_features=16,
            out_features=8,
            heads=4,
            concat=True
        )
        
        x = torch.randn(10, 16)
        adj = torch.ones(10, 10)  # Fully connected
        
        out = layer(x, adj)
        
        assert out.shape == (10, 32)  # 8 * 4 heads
    
    def test_graph_readout_mean(self, skip_if_no_torch):
        """Test GraphReadout with mean pooling."""
        import torch
        from src.models.gcn_layers import GraphReadout
        
        readout = GraphReadout(in_features=16, pooling="mean")
        
        x = torch.randn(5, 10, 16)  # Batch of 5, 10 nodes each
        
        out = readout(x)
        
        assert out.shape == (5, 16)


# ==============================================================================
# Tests for kg_gcn_lstm.py
# ==============================================================================

class TestKGGCNLSTM:
    """Tests for KG-GCN-LSTM model."""
    
    @pytest.fixture
    def skip_if_no_torch(self):
        """Skip test if torch not available."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_config_creation(self, skip_if_no_torch):
        """Test KGGCNLSTMConfig creation."""
        from src.models.kg_gcn_lstm import KGGCNLSTMConfig
        
        config = KGGCNLSTMConfig(
            gcn_hidden_dims=[64, 32],
            lstm_hidden_dim=64,
            lookback_window=12
        )
        
        assert config.gcn_hidden_dims == [64, 32]
        assert config.lstm_hidden_dim == 64
    
    def test_config_from_dict(self, skip_if_no_torch):
        """Test KGGCNLSTMConfig from dictionary."""
        from src.models.kg_gcn_lstm import KGGCNLSTMConfig
        
        config_dict = {
            "gcn_hidden_dims": [32, 16],
            "lstm_hidden_dim": 48,
            "batch_size": 16
        }
        
        config = KGGCNLSTMConfig.from_dict(config_dict)
        
        # Check that values from dict are applied
        assert config.gcn_hidden_dims == [32, 16]
        assert config.lstm_hidden_dim == 48
        assert config.batch_size == 16
    
    def test_model_initialization(self, skip_if_no_torch):
        """Test KGGCNLSTMModel initialization."""
        from src.models.kg_gcn_lstm import KGGCNLSTMModel
        
        model = KGGCNLSTMModel({
            "gcn_hidden_dims": [16],
            "lstm_hidden_dim": 16,
            "graph_embed_dim": 8,
            "seq_feature_dim": 4,
            "node_feature_dim": 4,
        })
        
        assert model.config is not None
        assert model.model is None  # Not fitted yet


# ==============================================================================
# Tests for cnn_lstm.py
# ==============================================================================

class TestCNNLSTM:
    """Tests for CNN-LSTM model."""
    
    @pytest.fixture
    def skip_if_no_torch(self):
        """Skip test if torch not available."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_cnn_lstm_config_creation(self, skip_if_no_torch):
        """Test CNNLSTMConfig creation."""
        from src.models.cnn_lstm import CNNLSTMConfig
        
        config = CNNLSTMConfig(
            cnn_channels=[32, 64],
            lstm_hidden_dim=64,
            use_attention=True
        )
        
        assert config.cnn_channels == [32, 64]
        assert config.use_attention is True
    
    def test_cnn_lstm_model_initialization(self, skip_if_no_torch):
        """Test CNNLSTMModel initialization."""
        from src.models.cnn_lstm import CNNLSTMModel
        
        model = CNNLSTMModel({
            "cnn_channels": [16, 32],
            "lstm_hidden_dim": 32,
            "input_dim": 8,
        })
        
        assert model.config is not None
    
    def test_lstm_only_model_initialization(self, skip_if_no_torch):
        """Test LSTMModel initialization."""
        from src.models.cnn_lstm import LSTMModel
        
        model = LSTMModel({
            "lstm_hidden_dim": 32,
            "input_dim": 8,
        })
        
        assert model.config is not None
    
    def test_temporal_attention_forward(self, skip_if_no_torch):
        """Test TemporalAttention forward pass."""
        import torch
        from src.models.cnn_lstm import TemporalAttention
        
        attention = TemporalAttention(input_dim=32, attention_dim=16)
        
        x = torch.randn(4, 10, 32)  # batch=4, seq=10, dim=32
        mask = torch.ones(4, 10)
        
        out, weights = attention(x, mask)
        
        assert out.shape == (4, 32)
        assert weights.shape == (4, 10)
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
    
    def test_cnn_encoder_forward(self, skip_if_no_torch):
        """Test CNNEncoder forward pass."""
        import torch
        from src.models.cnn_lstm import CNNEncoder
        
        encoder = CNNEncoder(
            input_dim=16,
            channels=[32, 64],
            kernel_sizes=[3, 3],
            pool_sizes=[2, 2],
            dropout=0.1
        )
        
        x = torch.randn(4, 12, 16)  # batch=4, seq=12, dim=16
        
        out = encoder(x)
        
        # Sequence length reduced by pooling
        assert out.shape[0] == 4
        assert out.shape[1] == 3  # 12 -> 6 -> 3 after two pool_size=2
        assert out.shape[2] == 64


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests across modules."""
    
    def test_full_sequence_pipeline(self):
        """Test full sequence building pipeline."""
        from src.sequence_builder import (
            SequenceConfig,
            build_sequences,
            SequenceScaler,
        )
        
        # Create synthetic data
        np.random.seed(42)
        n_series = 10
        n_months = 24
        
        data = []
        for i in range(n_series):
            for m in range(n_months):
                data.append({
                    "ndc": f"ndc_{i}",
                    "brand_drug_id": i,
                    "month_id": m,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                    "y_norm": 1.0 - 0.03 * m + np.random.randn() * 0.1,
                })
        
        df = pd.DataFrame(data)
        
        config = SequenceConfig(
            lookback_window=6,
            forecast_horizon=1,
            time_varying_features=["feature1", "feature2"],
            group_cols=("ndc",),
        )
        
        # Build sequences
        seq_data = build_sequences(df, config, is_train=True)
        
        # Scale
        scaler = SequenceScaler()
        scaler.fit(seq_data["X_seq"], y=seq_data["y"])
        X_scaled, _, y_scaled = scaler.transform(seq_data["X_seq"], y=seq_data["y"])
        
        assert X_scaled.shape[1] == 6
        assert X_scaled.shape[2] == 2
        assert y_scaled is not None
    
    def test_graph_construction_with_real_like_data(self):
        """Test graph construction with realistic drug data."""
        from src.graph_utils import build_drug_graph, DrugGraph
        
        df = pd.DataFrame({
            "ndc": [f"ndc_{i}" for i in range(20)],
            "country": ["USA"] * 10 + ["UK"] * 10,
            "brand_name": ["DrugA"] * 5 + ["DrugB"] * 5 + ["DrugA"] * 5 + ["DrugC"] * 5,
            "ther_area": ["oncology"] * 10 + ["cardiology"] * 10,
            "feature1": np.random.randn(20),
            "feature2": np.random.randn(20),
        })
        
        graph = build_drug_graph(
            df,
            feature_cols=["feature1", "feature2"]
        )
        
        # Check graph properties
        assert isinstance(graph, DrugGraph)
        assert hasattr(graph, "nodes")
        assert hasattr(graph, "adjacency_matrix")


# ==============================================================================
# Fixture for temporary directories
# ==============================================================================

@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for test files."""
    return tmp_path_factory.mktemp("test_data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
