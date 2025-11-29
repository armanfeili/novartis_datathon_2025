"""
KG-GCN-LSTM Model for pharmaceutical demand forecasting.

This model combines:
1. Knowledge Graph (KG) construction from drug metadata
2. Graph Convolutional Networks (GCN) for drug relationship embeddings
3. LSTM for temporal sequence modeling

Reference: KG-GCN-LSTM paper for pharmaceutical demand forecasting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.models.base import BaseModel

if TORCH_AVAILABLE:
    from src.models.gcn_layers import (
        GCNEncoder,
        GATEncoder,
        GraphReadout,
        create_gcn_encoder,
        normalize_adjacency,
    )
    from src.sequence_builder import (
        SequenceConfig,
        SequenceScaler,
        SequenceDataset,
        GraphSequenceDataset,
        build_sequences,
        create_dataloader,
    )


@dataclass
class KGGCNLSTMConfig:
    """Configuration for KG-GCN-LSTM model."""
    
    # GCN configuration
    gcn_type: str = "gcn"  # "gcn" or "gat"
    gcn_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    gcn_dropout: float = 0.2
    gcn_activation: str = "relu"
    gcn_skip_connection: bool = False
    gcn_layer_norm: bool = True
    gat_heads: int = 4
    
    # LSTM configuration
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = False
    
    # Feature dimensions
    node_feature_dim: int = 16
    seq_feature_dim: int = 32
    static_feature_dim: int = 8
    
    # Graph embedding dimension
    graph_embed_dim: int = 32
    
    # Fusion configuration
    fusion_type: str = "concat"  # "concat", "attention", "gate"
    fusion_hidden_dim: int = 64
    
    # Output configuration
    forecast_horizon: int = 1
    output_dim: int = 1
    
    # Sequence configuration
    lookback_window: int = 12
    
    # Training configuration
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Device
    device: str = "auto"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "KGGCNLSTMConfig":
        """Create config from dictionary."""
        from dataclasses import fields
        valid_field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config_dict.items() if k in valid_field_names})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "gcn_type": self.gcn_type,
            "gcn_hidden_dims": self.gcn_hidden_dims,
            "gcn_dropout": self.gcn_dropout,
            "gcn_activation": self.gcn_activation,
            "gcn_skip_connection": self.gcn_skip_connection,
            "gcn_layer_norm": self.gcn_layer_norm,
            "gat_heads": self.gat_heads,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_num_layers": self.lstm_num_layers,
            "lstm_dropout": self.lstm_dropout,
            "lstm_bidirectional": self.lstm_bidirectional,
            "node_feature_dim": self.node_feature_dim,
            "seq_feature_dim": self.seq_feature_dim,
            "static_feature_dim": self.static_feature_dim,
            "graph_embed_dim": self.graph_embed_dim,
            "fusion_type": self.fusion_type,
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "forecast_horizon": self.forecast_horizon,
            "output_dim": self.output_dim,
            "lookback_window": self.lookback_window,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "device": self.device,
        }


if TORCH_AVAILABLE:
    class LSTMEncoder(nn.Module):
        """
        LSTM encoder for temporal sequences.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = False,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            
            self.output_dim = hidden_dim * self.num_directions
        
        def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:
            """
            Forward pass.
            
            Args:
                x: Input sequence (B, T, input_dim)
                mask: Optional mask (B, T) with 1 for valid, 0 for padded
                
            Returns:
                Tuple of (all_outputs, final_hidden)
                - all_outputs: (B, T, hidden_dim * num_directions)
                - final_hidden: (B, hidden_dim * num_directions)
            """
            if mask is not None:
                # Pack padded sequence for efficiency
                lengths = mask.sum(dim=1).cpu().long()
                lengths = torch.clamp(lengths, min=1)  # Ensure at least length 1
                
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                outputs, (h_n, c_n) = self.lstm(packed)
                outputs, _ = nn.utils.rnn.pad_packed_sequence(
                    outputs, batch_first=True, total_length=x.size(1)
                )
            else:
                outputs, (h_n, c_n) = self.lstm(x)
            
            # Get final hidden state
            if self.bidirectional:
                # Concatenate last layer forward and backward
                final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                final_hidden = h_n[-1]
            
            return outputs, final_hidden
    
    
    class FusionModule(nn.Module):
        """
        Fusion module for combining graph and sequence embeddings.
        """
        
        def __init__(
            self,
            graph_dim: int,
            seq_dim: int,
            static_dim: int,
            hidden_dim: int,
            output_dim: int,
            fusion_type: str = "concat",
        ):
            super().__init__()
            self.fusion_type = fusion_type
            
            if fusion_type == "concat":
                self.fusion = nn.Sequential(
                    nn.Linear(graph_dim + seq_dim + static_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, output_dim),
                )
            elif fusion_type == "attention":
                self.query = nn.Linear(seq_dim, hidden_dim)
                self.key_graph = nn.Linear(graph_dim, hidden_dim)
                self.key_static = nn.Linear(static_dim, hidden_dim)
                self.value_graph = nn.Linear(graph_dim, hidden_dim)
                self.value_static = nn.Linear(static_dim, hidden_dim)
                self.output_proj = nn.Linear(seq_dim + hidden_dim, output_dim)
            elif fusion_type == "gate":
                self.gate_graph = nn.Sequential(
                    nn.Linear(seq_dim + graph_dim, graph_dim),
                    nn.Sigmoid(),
                )
                self.gate_static = nn.Sequential(
                    nn.Linear(seq_dim + static_dim, static_dim),
                    nn.Sigmoid(),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(graph_dim + seq_dim + static_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        def forward(
            self,
            graph_embed: Tensor,
            seq_embed: Tensor,
            static_embed: Tensor,
        ) -> Tensor:
            """
            Fuse graph, sequence, and static embeddings.
            
            Args:
                graph_embed: Graph embedding (B, graph_dim)
                seq_embed: Sequence embedding (B, seq_dim)
                static_embed: Static feature embedding (B, static_dim)
                
            Returns:
                Fused embedding (B, output_dim)
            """
            if self.fusion_type == "concat":
                combined = torch.cat([graph_embed, seq_embed, static_embed], dim=1)
                return self.fusion(combined)
            
            elif self.fusion_type == "attention":
                # Query from sequence
                q = self.query(seq_embed)  # (B, hidden_dim)
                
                # Keys and values from graph and static
                k_g = self.key_graph(graph_embed)  # (B, hidden_dim)
                k_s = self.key_static(static_embed)  # (B, hidden_dim)
                v_g = self.value_graph(graph_embed)  # (B, hidden_dim)
                v_s = self.value_static(static_embed)  # (B, hidden_dim)
                
                # Compute attention weights
                k = torch.stack([k_g, k_s], dim=1)  # (B, 2, hidden_dim)
                v = torch.stack([v_g, v_s], dim=1)  # (B, 2, hidden_dim)
                
                attn_scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)  # (B, 2)
                attn_weights = F.softmax(attn_scores, dim=1)  # (B, 2)
                
                attended = (v * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
                
                return self.output_proj(torch.cat([seq_embed, attended], dim=1))
            
            elif self.fusion_type == "gate":
                # Gated fusion
                g_graph = self.gate_graph(torch.cat([seq_embed, graph_embed], dim=1))
                g_static = self.gate_static(torch.cat([seq_embed, static_embed], dim=1))
                
                gated_graph = graph_embed * g_graph
                gated_static = static_embed * g_static
                
                combined = torch.cat([gated_graph, seq_embed, gated_static], dim=1)
                return self.fusion(combined)
    
    
    class KGGCNLSTMNetwork(nn.Module):
        """
        Full KG-GCN-LSTM network architecture.
        """
        
        def __init__(self, config: KGGCNLSTMConfig):
            super().__init__()
            self.config = config
            
            # GCN encoder for graph embeddings
            gcn_config = {
                "encoder_type": config.gcn_type,
                "in_features": config.node_feature_dim,
                "hidden_dims": config.gcn_hidden_dims,
                "hidden_dim": config.gcn_hidden_dims[0] if config.gcn_hidden_dims else 32,
                "out_features": config.graph_embed_dim,
                "num_layers": len(config.gcn_hidden_dims) + 1,
                "heads": config.gat_heads,
                "dropout": config.gcn_dropout,
                "activation": config.gcn_activation,
                "skip_connection": config.gcn_skip_connection,
                "layer_norm": config.gcn_layer_norm,
            }
            self.gcn_encoder = create_gcn_encoder(gcn_config)
            
            # Graph readout
            self.graph_readout = GraphReadout(
                in_features=config.graph_embed_dim,
                pooling="attention",
            )
            
            # LSTM encoder for sequences
            self.lstm_encoder = LSTMEncoder(
                input_dim=config.seq_feature_dim,
                hidden_dim=config.lstm_hidden_dim,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout,
                bidirectional=config.lstm_bidirectional,
            )
            
            lstm_output_dim = config.lstm_hidden_dim * (2 if config.lstm_bidirectional else 1)
            
            # Static feature embedding
            self.static_embed = nn.Sequential(
                nn.Linear(config.static_feature_dim, config.static_feature_dim * 2),
                nn.ReLU(),
                nn.Linear(config.static_feature_dim * 2, config.static_feature_dim),
            )
            
            # Fusion module
            self.fusion = FusionModule(
                graph_dim=config.graph_embed_dim,
                seq_dim=lstm_output_dim,
                static_dim=config.static_feature_dim,
                hidden_dim=config.fusion_hidden_dim,
                output_dim=config.fusion_hidden_dim,
                fusion_type=config.fusion_type,
            )
            
            # Output layer
            self.output_layer = nn.Sequential(
                nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.fusion_hidden_dim // 2, config.forecast_horizon * config.output_dim),
            )
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights."""
            for name, param in self.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        
        def forward(
            self,
            X_seq: Tensor,
            X_static: Tensor,
            node_features: Tensor,
            adjacency: Tensor,
            node_indices: Tensor,
            mask: Optional[Tensor] = None,
        ) -> Tensor:
            """
            Forward pass.
            
            Args:
                X_seq: Sequence features (B, T, seq_feature_dim)
                X_static: Static features (B, static_feature_dim)
                node_features: Node features (N, node_feature_dim)
                adjacency: Adjacency matrix (N, N)
                node_indices: Node index for each sample (B,)
                mask: Sequence mask (B, T)
                
            Returns:
                Predictions (B, forecast_horizon * output_dim)
            """
            # Graph encoding
            node_embeddings = self.gcn_encoder(node_features, adjacency)  # (N, graph_embed_dim)
            
            # Get embeddings for each sample's node
            graph_embed = node_embeddings[node_indices]  # (B, graph_embed_dim)
            
            # Sequence encoding
            _, seq_embed = self.lstm_encoder(X_seq, mask)  # (B, lstm_output_dim)
            
            # Static feature embedding
            static_embed = self.static_embed(X_static)  # (B, static_feature_dim)
            
            # Fusion
            fused = self.fusion(graph_embed, seq_embed, static_embed)  # (B, fusion_hidden_dim)
            
            # Output
            output = self.output_layer(fused)  # (B, forecast_horizon * output_dim)
            
            return output


class KGGCNLSTMModel(BaseModel):
    """
    KG-GCN-LSTM Model wrapper implementing BaseModel interface.
    
    Combines Knowledge Graph embeddings with LSTM for demand forecasting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.
        
        Args:
            config: Model configuration dictionary
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for KGGCNLSTMModel")
        
        self.config = KGGCNLSTMConfig.from_dict(config or {})
        self.model: Optional[KGGCNLSTMNetwork] = None
        self.scaler: Optional[SequenceScaler] = None
        self.seq_config: Optional[SequenceConfig] = None
        
        # Graph data
        self.node_features: Optional[np.ndarray] = None
        self.adjacency: Optional[np.ndarray] = None
        self.node_id_to_idx: Dict[str, int] = {}
        
        # Feature columns
        self.feature_columns: List[str] = []
        self.static_columns: List[str] = []
        
        # Device
        self._setup_device()
    
    def _setup_device(self):
        """Setup compute device."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
    
    def _build_graph(self, df: pd.DataFrame) -> None:
        """
        Build knowledge graph from data.
        
        Args:
            df: DataFrame with drug metadata
        """
        from src.graph_utils import KnowledgeGraphBuilder, build_drug_graph
        
        # Use default edge types for drug graph
        builder = KnowledgeGraphBuilder()
        
        # Identify unique nodes (NDCs or drug-brand pairs)
        group_cols = list(self.seq_config.group_cols) if self.seq_config else ["ndc"]
        
        if len(group_cols) == 1:
            unique_ids = df[group_cols[0]].unique()
            node_ids = [str(nid) for nid in unique_ids]
        else:
            unique_groups = df[group_cols].drop_duplicates()
            node_ids = ["-".join(str(v) for v in row) for _, row in unique_groups.iterrows()]
        
        # Create node ID mapping
        self.node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Build adjacency based on shared attributes
        n_nodes = len(node_ids)
        adjacency = np.eye(n_nodes, dtype=np.float32)  # Self-loops
        
        # Add edges based on shared therapeutic area
        if "therapeutic_area" in df.columns:
            ta_groups = df.groupby("therapeutic_area")[group_cols[0]].apply(list).to_dict()
            for ta, nodes in ta_groups.items():
                node_list = [str(n) for n in nodes]
                for i, n1 in enumerate(node_list):
                    for n2 in node_list[i+1:]:
                        if n1 in self.node_id_to_idx and n2 in self.node_id_to_idx:
                            i1 = self.node_id_to_idx[n1]
                            i2 = self.node_id_to_idx[n2]
                            adjacency[i1, i2] = 1.0
                            adjacency[i2, i1] = 1.0
        
        # Add edges based on shared manufacturer
        if "manufacturer_name" in df.columns:
            mfr_groups = df.groupby("manufacturer_name")[group_cols[0]].apply(list).to_dict()
            for mfr, nodes in mfr_groups.items():
                node_list = [str(n) for n in nodes]
                for i, n1 in enumerate(node_list):
                    for n2 in node_list[i+1:]:
                        if n1 in self.node_id_to_idx and n2 in self.node_id_to_idx:
                            i1 = self.node_id_to_idx[n1]
                            i2 = self.node_id_to_idx[n2]
                            adjacency[i1, i2] = 0.5  # Weaker edge
                            adjacency[i2, i1] = 0.5
        
        self.adjacency = adjacency
        
        # Build simple node features (can be extended)
        self.node_features = np.random.randn(n_nodes, self.config.node_feature_dim).astype(np.float32)
        
        # If we have categorical data, encode it
        for col in ["therapeutic_area", "route_of_administration", "dosage_form"]:
            if col in df.columns:
                unique_vals = df[col].unique()
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                
                # One-hot style contribution to node features
                for group_id, idx in self.node_id_to_idx.items():
                    group_data = df[df[group_cols[0]].astype(str) == group_id.split("-")[0]]
                    if not group_data.empty:
                        val = group_data[col].iloc[0]
                        if val in val_to_idx:
                            feature_idx = val_to_idx[val] % self.config.node_feature_dim
                            self.node_features[idx, feature_idx] += 1.0
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
    ) -> Tuple[GraphSequenceDataset, Optional[np.ndarray]]:
        """
        Prepare data for training/inference.
        
        Args:
            df: Input DataFrame
            is_train: Whether preparing for training
            
        Returns:
            Tuple of (dataset, node_indices)
        """
        # Build sequences
        seq_data = build_sequences(df, self.seq_config, is_train=is_train)
        
        # Map group IDs to node indices
        node_indices = []
        for group_id in seq_data["group_ids"]:
            if isinstance(group_id, tuple):
                key = "-".join(str(v) for v in group_id)
            else:
                key = str(group_id)
            
            idx = self.node_id_to_idx.get(key, 0)  # Default to first node if not found
            node_indices.append(idx)
        
        node_indices = np.array(node_indices)
        
        # Apply scaling
        if is_train:
            self.scaler = SequenceScaler(method="standard")
            self.scaler.fit(seq_data["X_seq"], seq_data["X_static"], seq_data["y"])
        
        X_seq, X_static, y = self.scaler.transform(
            seq_data["X_seq"],
            seq_data["X_static"],
            seq_data["y"] if is_train else None,
        )
        
        # Create dataset
        dataset = GraphSequenceDataset(
            X_seq=X_seq,
            X_static=X_static if X_static is not None else np.zeros((len(X_seq), 1)),
            y=y,
            masks=seq_data["masks"],
            node_features=self.node_features,
            adjacency=self.adjacency,
            node_indices=node_indices,
        )
        
        return dataset, node_indices
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "KGGCNLSTMModel":
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            self
        """
        # Combine X and y
        train_df = X.copy()
        train_df[self.config.target_col if hasattr(self.config, 'target_col') else "y_norm"] = y.values
        
        # Setup sequence config
        self.seq_config = SequenceConfig(
            lookback_window=self.config.lookback_window,
            forecast_horizon=self.config.forecast_horizon,
            target_col="y_norm",
        )
        
        # Identify feature columns
        self.feature_columns = [c for c in X.columns if c not in ["ndc", "brand_drug_id", "month_id"]]
        self.seq_config.time_varying_features = self.feature_columns[:self.config.seq_feature_dim]
        
        # Build graph
        self._build_graph(train_df)
        
        # Update config with actual dimensions
        actual_seq_dim = len(self.seq_config.time_varying_features) if self.seq_config.time_varying_features else 1
        self.config.seq_feature_dim = max(actual_seq_dim, 1)
        
        # Prepare training data
        train_dataset, _ = self._prepare_data(train_df, is_train=True)
        
        # Prepare validation data
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_df = X_val.copy()
            val_df["y_norm"] = y_val.values
            val_dataset, _ = self._prepare_data(val_df, is_train=False)
        
        # Initialize model
        self.model = KGGCNLSTMNetwork(self.config).to(self.device)
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = create_dataloader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        # Get graph tensors
        node_features_t = torch.from_numpy(self.node_features).float().to(self.device)
        adjacency_t = torch.from_numpy(self.adjacency).float().to(self.device)
        
        for epoch in range(self.config.max_epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                X_seq = batch["X_seq"].to(self.device)
                X_static = batch["X_static"].to(self.device)
                y_batch = batch["y"].to(self.device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                node_idx = batch["node_idx"].to(self.device)
                
                optimizer.zero_grad()
                
                output = self.model(
                    X_seq, X_static, node_features_t, adjacency_t, node_idx, mask
                )
                
                # Reshape output and target for loss
                output = output.view(-1)
                y_batch = y_batch.view(-1)
                
                loss = F.mse_loss(output, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        X_seq = batch["X_seq"].to(self.device)
                        X_static = batch["X_static"].to(self.device)
                        y_batch = batch["y"].to(self.device)
                        mask = batch.get("mask", None)
                        if mask is not None:
                            mask = mask.to(self.device)
                        node_idx = batch["node_idx"].to(self.device)
                        
                        output = self.model(
                            X_seq, X_static, node_features_t, adjacency_t, node_idx, mask
                        )
                        
                        output = output.view(-1)
                        y_batch = y_batch.view(-1)
                        
                        val_loss += F.mse_loss(output, y_batch).item()
                
                val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        # Prepare data
        pred_df = X.copy()
        pred_df["y_norm"] = 0  # Placeholder
        
        dataset, _ = self._prepare_data(pred_df, is_train=False)
        loader = create_dataloader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Graph tensors
        node_features_t = torch.from_numpy(self.node_features).float().to(self.device)
        adjacency_t = torch.from_numpy(self.adjacency).float().to(self.device)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                X_seq = batch["X_seq"].to(self.device)
                X_static = batch["X_static"].to(self.device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                node_idx = batch["node_idx"].to(self.device)
                
                output = self.model(
                    X_seq, X_static, node_features_t, adjacency_t, node_idx, mask
                )
                
                predictions.append(output.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform_target(predictions)
        
        return predictions.flatten()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        if self.model is not None:
            torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save config and metadata
        metadata = {
            "config": self.config.to_dict(),
            "feature_columns": self.feature_columns,
            "static_columns": self.static_columns,
            "node_id_to_idx": self.node_id_to_idx,
            "seq_config": {
                "lookback_window": self.seq_config.lookback_window,
                "forecast_horizon": self.seq_config.forecast_horizon,
                "target_col": self.seq_config.target_col,
                "time_varying_features": self.seq_config.time_varying_features,
                "static_features": self.seq_config.static_features,
            } if self.seq_config else None,
            "scaler_params": self.scaler.get_params() if self.scaler else None,
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save graph data
        if self.node_features is not None:
            np.save(path / "node_features.npy", self.node_features)
        if self.adjacency is not None:
            np.save(path / "adjacency.npy", self.adjacency)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "KGGCNLSTMModel":
        """Load model from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(config=metadata["config"])
        instance.feature_columns = metadata["feature_columns"]
        instance.static_columns = metadata["static_columns"]
        instance.node_id_to_idx = metadata["node_id_to_idx"]
        
        # Load sequence config
        if metadata.get("seq_config"):
            instance.seq_config = SequenceConfig(**metadata["seq_config"])
        
        # Load scaler
        if metadata.get("scaler_params"):
            instance.scaler = SequenceScaler.from_params(metadata["scaler_params"])
        
        # Load graph data
        if (path / "node_features.npy").exists():
            instance.node_features = np.load(path / "node_features.npy")
        if (path / "adjacency.npy").exists():
            instance.adjacency = np.load(path / "adjacency.npy")
        
        # Load model weights
        instance.model = KGGCNLSTMNetwork(instance.config).to(instance.device)
        instance.model.load_state_dict(torch.load(path / "model.pt", map_location=instance.device))
        instance.model.eval()
        
        return instance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        For neural networks, we use gradient-based importance.
        """
        if not self.feature_columns:
            return {}
        
        # Return uniform importance (or implement gradient-based analysis)
        n_features = len(self.feature_columns)
        return {col: 1.0 / n_features for col in self.feature_columns}
