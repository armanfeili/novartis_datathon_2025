"""
CNN-LSTM Model for drug sales prediction.

This module implements the CNN-LSTM architecture from:
Li et al. (2024) "CNN-LSTM for drug sales prediction"

The model uses:
1. 1D Convolutional layers to extract local temporal patterns
2. LSTM layers to capture long-term dependencies
3. Fully connected layers for prediction

Reference: Li et al. (2024) CNN-LSTM for drug sales prediction.
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
    from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.models.base import BaseModel

if TORCH_AVAILABLE:
    from src.sequence_builder import (
        SequenceConfig,
        SequenceScaler,
        SequenceDataset,
        build_sequences,
        build_sequences_for_inference,
        create_dataloader,
    )


@dataclass
class CNNLSTMConfig:
    """Configuration for CNN-LSTM model."""
    
    # CNN configuration
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])
    cnn_pool_sizes: List[int] = field(default_factory=lambda: [2, 2])
    cnn_dropout: float = 0.2
    
    # LSTM configuration
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = False
    
    # Attention configuration
    use_attention: bool = True
    attention_dim: int = 32
    
    # Feature dimensions
    input_dim: int = 32
    static_dim: int = 8
    
    # Output configuration
    fc_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
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
    use_one_cycle: bool = True
    
    # Regularization
    label_smoothing: float = 0.0
    gradient_clip: float = 1.0
    
    # Device
    device: str = "auto"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CNNLSTMConfig":
        """Create config from dictionary."""
        from dataclasses import fields
        valid_field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config_dict.items() if k in valid_field_names})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "cnn_channels": self.cnn_channels,
            "cnn_kernel_sizes": self.cnn_kernel_sizes,
            "cnn_pool_sizes": self.cnn_pool_sizes,
            "cnn_dropout": self.cnn_dropout,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_num_layers": self.lstm_num_layers,
            "lstm_dropout": self.lstm_dropout,
            "lstm_bidirectional": self.lstm_bidirectional,
            "use_attention": self.use_attention,
            "attention_dim": self.attention_dim,
            "input_dim": self.input_dim,
            "static_dim": self.static_dim,
            "fc_hidden_dims": self.fc_hidden_dims,
            "forecast_horizon": self.forecast_horizon,
            "output_dim": self.output_dim,
            "lookback_window": self.lookback_window,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "use_one_cycle": self.use_one_cycle,
            "label_smoothing": self.label_smoothing,
            "gradient_clip": self.gradient_clip,
            "device": self.device,
        }


@dataclass
class LSTMOnlyConfig:
    """Configuration for pure LSTM model (ablation)."""
    
    # LSTM configuration
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True
    
    # Attention
    use_attention: bool = True
    attention_dim: int = 64
    
    # Feature dimensions
    input_dim: int = 32
    static_dim: int = 8
    
    # Output configuration
    fc_hidden_dims: List[int] = field(default_factory=lambda: [64])
    forecast_horizon: int = 1
    output_dim: int = 1
    
    # Sequence configuration
    lookback_window: int = 12
    
    # Training configuration
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 15
    
    # Device
    device: str = "auto"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LSTMOnlyConfig":
        from dataclasses import fields
        valid_field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config_dict.items() if k in valid_field_names})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_num_layers": self.lstm_num_layers,
            "lstm_dropout": self.lstm_dropout,
            "lstm_bidirectional": self.lstm_bidirectional,
            "use_attention": self.use_attention,
            "attention_dim": self.attention_dim,
            "input_dim": self.input_dim,
            "static_dim": self.static_dim,
            "fc_hidden_dims": self.fc_hidden_dims,
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
    class TemporalAttention(nn.Module):
        """
        Temporal attention mechanism for sequence data.
        
        Computes attention weights over timesteps to create a weighted sum.
        """
        
        def __init__(self, input_dim: int, attention_dim: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(input_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1),
            )
        
        def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:
            """
            Apply temporal attention.
            
            Args:
                x: Input tensor (B, T, D)
                mask: Optional mask (B, T) with 1 for valid, 0 for padded
                
            Returns:
                Tuple of (attended_output, attention_weights)
                - attended_output: (B, D)
                - attention_weights: (B, T)
            """
            # Compute attention scores
            scores = self.attention(x).squeeze(-1)  # (B, T)
            
            # Apply mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            
            # Normalize
            weights = F.softmax(scores, dim=-1)  # (B, T)
            
            # Handle all-masked case
            weights = torch.where(
                torch.isnan(weights),
                torch.zeros_like(weights),
                weights
            )
            
            # Weighted sum
            output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
            
            return output, weights
    
    
    class Conv1dBlock(nn.Module):
        """
        1D Convolutional block with BatchNorm and optional pooling.
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            pool_size: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            
            padding = kernel_size // 2  # Same padding
            
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.bn = nn.BatchNorm1d(out_channels)
            self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()
        
        def forward(self, x: Tensor) -> Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor (B, C_in, T)
                
            Returns:
                Output tensor (B, C_out, T // pool_size)
            """
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation(x)
            x = self.pool(x)
            x = self.dropout(x)
            return x
    
    
    class CNNEncoder(nn.Module):
        """
        1D CNN encoder for temporal pattern extraction.
        """
        
        def __init__(
            self,
            input_dim: int,
            channels: List[int],
            kernel_sizes: List[int],
            pool_sizes: List[int],
            dropout: float = 0.2,
        ):
            super().__init__()
            
            assert len(channels) == len(kernel_sizes) == len(pool_sizes)
            
            layers = []
            in_ch = input_dim
            
            for out_ch, ks, ps in zip(channels, kernel_sizes, pool_sizes):
                layers.append(Conv1dBlock(in_ch, out_ch, ks, ps, dropout))
                in_ch = out_ch
            
            self.encoder = nn.Sequential(*layers)
            self.output_channels = channels[-1] if channels else input_dim
        
        def forward(self, x: Tensor) -> Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor (B, T, D)
                
            Returns:
                Encoded tensor (B, T', C_out)
            """
            # Conv1d expects (B, C, T)
            x = x.transpose(1, 2)  # (B, D, T)
            x = self.encoder(x)  # (B, C_out, T')
            x = x.transpose(1, 2)  # (B, T', C_out)
            return x
    
    
    class CNNLSTMNetwork(nn.Module):
        """
        CNN-LSTM network architecture.
        
        1D CNN extracts local patterns, LSTM captures long-term dependencies.
        """
        
        def __init__(self, config: CNNLSTMConfig):
            super().__init__()
            self.config = config
            
            # Input projection
            self.input_proj = nn.Linear(config.input_dim, config.cnn_channels[0])
            
            # CNN encoder
            self.cnn_encoder = CNNEncoder(
                input_dim=config.cnn_channels[0],
                channels=config.cnn_channels,
                kernel_sizes=config.cnn_kernel_sizes,
                pool_sizes=config.cnn_pool_sizes,
                dropout=config.cnn_dropout,
            )
            
            # Calculate CNN output length
            cnn_output_len = config.lookback_window
            for ps in config.cnn_pool_sizes:
                cnn_output_len = cnn_output_len // ps
            cnn_output_len = max(cnn_output_len, 1)
            
            # LSTM encoder
            lstm_input_dim = self.cnn_encoder.output_channels
            self.lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=config.lstm_hidden_dim,
                num_layers=config.lstm_num_layers,
                batch_first=True,
                dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
                bidirectional=config.lstm_bidirectional,
            )
            
            lstm_output_dim = config.lstm_hidden_dim * (2 if config.lstm_bidirectional else 1)
            
            # Attention
            if config.use_attention:
                self.attention = TemporalAttention(lstm_output_dim, config.attention_dim)
            else:
                self.attention = None
            
            # Static feature encoder
            self.static_encoder = nn.Sequential(
                nn.Linear(config.static_dim, config.static_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.static_dim * 2, config.static_dim),
            )
            
            # Fully connected layers
            fc_input_dim = lstm_output_dim + config.static_dim
            
            fc_layers = []
            prev_dim = fc_input_dim
            for hidden_dim in config.fc_hidden_dims:
                fc_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ])
                prev_dim = hidden_dim
            
            fc_layers.append(nn.Linear(prev_dim, config.forecast_horizon * config.output_dim))
            
            self.fc = nn.Sequential(*fc_layers)
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights."""
            for name, param in self.named_parameters():
                if "weight" in name:
                    if "lstm" in name:
                        nn.init.orthogonal_(param)
                    elif param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        
        def forward(
            self,
            X_seq: Tensor,
            X_static: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
        ) -> Tensor:
            """
            Forward pass.
            
            Args:
                X_seq: Sequence features (B, T, input_dim)
                X_static: Static features (B, static_dim)
                mask: Sequence mask (B, T)
                
            Returns:
                Predictions (B, forecast_horizon * output_dim)
            """
            B = X_seq.size(0)
            
            # Input projection
            x = self.input_proj(X_seq)  # (B, T, cnn_channels[0])
            
            # CNN encoding
            x = self.cnn_encoder(x)  # (B, T', cnn_output_channels)
            
            # LSTM encoding
            lstm_out, (h_n, c_n) = self.lstm(x)  # (B, T', lstm_output_dim)
            
            # Apply attention or use final hidden state
            if self.attention is not None:
                seq_embed, _ = self.attention(lstm_out, None)  # (B, lstm_output_dim)
            else:
                # Use last hidden state
                if self.config.lstm_bidirectional:
                    seq_embed = torch.cat([h_n[-2], h_n[-1]], dim=1)
                else:
                    seq_embed = h_n[-1]
            
            # Static features
            if X_static is not None:
                static_embed = self.static_encoder(X_static)  # (B, static_dim)
                combined = torch.cat([seq_embed, static_embed], dim=1)
            else:
                # Pad with zeros if no static features
                static_embed = torch.zeros(B, self.config.static_dim, device=X_seq.device)
                combined = torch.cat([seq_embed, static_embed], dim=1)
            
            # Output
            output = self.fc(combined)  # (B, forecast_horizon * output_dim)
            
            return output
    
    
    class LSTMOnlyNetwork(nn.Module):
        """
        Pure LSTM network (no CNN) for ablation comparison.
        """
        
        def __init__(self, config: LSTMOnlyConfig):
            super().__init__()
            self.config = config
            
            # LSTM encoder
            self.lstm = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=config.lstm_hidden_dim,
                num_layers=config.lstm_num_layers,
                batch_first=True,
                dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
                bidirectional=config.lstm_bidirectional,
            )
            
            lstm_output_dim = config.lstm_hidden_dim * (2 if config.lstm_bidirectional else 1)
            
            # Attention
            if config.use_attention:
                self.attention = TemporalAttention(lstm_output_dim, config.attention_dim)
            else:
                self.attention = None
            
            # Static feature encoder
            self.static_encoder = nn.Sequential(
                nn.Linear(config.static_dim, config.static_dim * 2),
                nn.ReLU(),
                nn.Linear(config.static_dim * 2, config.static_dim),
            )
            
            # FC layers
            fc_input_dim = lstm_output_dim + config.static_dim
            
            fc_layers = []
            prev_dim = fc_input_dim
            for hidden_dim in config.fc_hidden_dims:
                fc_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ])
                prev_dim = hidden_dim
            
            fc_layers.append(nn.Linear(prev_dim, config.forecast_horizon * config.output_dim))
            self.fc = nn.Sequential(*fc_layers)
            
            self._init_weights()
        
        def _init_weights(self):
            for name, param in self.named_parameters():
                if "weight" in name and "lstm" in name:
                    nn.init.orthogonal_(param)
                elif "weight" in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        
        def forward(
            self,
            X_seq: Tensor,
            X_static: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
        ) -> Tensor:
            B = X_seq.size(0)
            
            # LSTM encoding
            lstm_out, (h_n, c_n) = self.lstm(X_seq)
            
            # Apply attention or use final hidden state
            if self.attention is not None:
                seq_embed, _ = self.attention(lstm_out, mask)
            else:
                if self.config.lstm_bidirectional:
                    seq_embed = torch.cat([h_n[-2], h_n[-1]], dim=1)
                else:
                    seq_embed = h_n[-1]
            
            # Static features
            if X_static is not None:
                static_embed = self.static_encoder(X_static)
                combined = torch.cat([seq_embed, static_embed], dim=1)
            else:
                static_embed = torch.zeros(B, self.config.static_dim, device=X_seq.device)
                combined = torch.cat([seq_embed, static_embed], dim=1)
            
            return self.fc(combined)


class _BaseTemporalModel(BaseModel):
    """Base class for temporal models (CNN-LSTM, LSTM)."""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for temporal models")
        
        self.model = None
        self.scaler: Optional[SequenceScaler] = None
        self.seq_config: Optional[SequenceConfig] = None
        self.feature_columns: List[str] = []
        self.static_columns: List[str] = []
        self.device = None
    
    def _setup_device(self, device_config: str):
        if device_config == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
    ) -> SequenceDataset:
        """Prepare data for training/inference."""
        # Build sequences
        seq_data = build_sequences(df, self.seq_config, is_train=is_train)
        
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
        dataset = SequenceDataset(
            X_seq=X_seq,
            X_static=X_static if X_static is not None else np.zeros((len(X_seq), 1)),
            y=y,
            masks=seq_data["masks"],
        )
        
        return dataset
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        gradient_clip: float = 1.0,
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
        for batch in loader:
            X_seq = batch["X_seq"].to(self.device)
            X_static = batch.get("X_static")
            if X_static is not None:
                X_static = X_static.to(self.device)
            y_batch = batch["y"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)
            
            optimizer.zero_grad()
            
            output = model(X_seq, X_static, mask)
            output = output.view(-1)
            y_batch = y_batch.view(-1)
            
            loss = F.mse_loss(output, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()
            
            if scheduler is not None and hasattr(scheduler, 'step') and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> float:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                X_seq = batch["X_seq"].to(self.device)
                X_static = batch.get("X_static")
                if X_static is not None:
                    X_static = X_static.to(self.device)
                y_batch = batch["y"].to(self.device)
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(self.device)
                
                output = model(X_seq, X_static, mask)
                output = output.view(-1)
                y_batch = y_batch.view(-1)
                
                total_loss += F.mse_loss(output, y_batch).item()
        
        return total_loss / len(loader)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        # Prepare data
        pred_df = X.copy()
        pred_df["y_norm"] = 0  # Placeholder
        
        dataset = self._prepare_data(pred_df, is_train=False)
        loader = create_dataloader(dataset, batch_size=64, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                X_seq = batch["X_seq"].to(self.device)
                X_static = batch.get("X_static")
                if X_static is not None:
                    X_static = X_static.to(self.device)
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(self.device)
                
                output = self.model(X_seq, X_static, mask)
                predictions.append(output.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform_target(predictions)
        
        return predictions.flatten()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (uniform for neural networks)."""
        if not self.feature_columns:
            return {}
        n_features = len(self.feature_columns)
        return {col: 1.0 / n_features for col in self.feature_columns}


class CNNLSTMModel(_BaseTemporalModel):
    """
    CNN-LSTM Model wrapper implementing BaseModel interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = CNNLSTMConfig.from_dict(config or {})
        self._setup_device(self.config.device)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "CNNLSTMModel":
        """Train the model."""
        # Combine X and y
        train_df = X.copy()
        train_df["y_norm"] = y.values
        
        # Setup sequence config
        self.seq_config = SequenceConfig(
            lookback_window=self.config.lookback_window,
            forecast_horizon=self.config.forecast_horizon,
            target_col="y_norm",
        )
        
        # Identify feature columns
        exclude_cols = ["ndc", "brand_drug_id", "month_id", "y_norm"]
        self.feature_columns = [c for c in X.columns if c not in exclude_cols]
        self.seq_config.time_varying_features = self.feature_columns[:self.config.input_dim]
        
        # Update config with actual dimensions
        actual_input_dim = len(self.seq_config.time_varying_features) if self.seq_config.time_varying_features else 1
        self.config.input_dim = max(actual_input_dim, 1)
        
        # Prepare training data
        train_dataset = self._prepare_data(train_df, is_train=True)
        
        # Prepare validation data
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_df = X_val.copy()
            val_df["y_norm"] = y_val.values
            val_dataset = self._prepare_data(val_df, is_train=False)
        
        # Initialize model
        self.model = CNNLSTMNetwork(self.config).to(self.device)
        
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
        
        if self.config.use_one_cycle:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate * 10,
                epochs=self.config.max_epochs,
                steps_per_epoch=len(train_loader),
            )
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            train_loss = self._train_epoch(
                self.model, train_loader, optimizer, 
                scheduler if self.config.use_one_cycle else None,
                self.config.gradient_clip,
            )
            
            # Validation
            val_loss = train_loss
            if val_loader is not None:
                val_loss = self._validate(self.model, val_loader)
            
            if not self.config.use_one_cycle:
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
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            torch.save(self.model.state_dict(), path / "model.pt")
        
        metadata = {
            "config": self.config.to_dict(),
            "feature_columns": self.feature_columns,
            "static_columns": self.static_columns,
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
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "CNNLSTMModel":
        """Load model from disk."""
        path = Path(path)
        
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        instance = cls(config=metadata["config"])
        instance.feature_columns = metadata["feature_columns"]
        instance.static_columns = metadata["static_columns"]
        
        if metadata.get("seq_config"):
            instance.seq_config = SequenceConfig(**metadata["seq_config"])
        
        if metadata.get("scaler_params"):
            instance.scaler = SequenceScaler.from_params(metadata["scaler_params"])
        
        instance.model = CNNLSTMNetwork(instance.config).to(instance.device)
        instance.model.load_state_dict(torch.load(path / "model.pt", map_location=instance.device))
        instance.model.eval()
        
        return instance


class LSTMModel(_BaseTemporalModel):
    """
    Pure LSTM Model wrapper implementing BaseModel interface.
    
    For ablation comparison with CNN-LSTM.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = LSTMOnlyConfig.from_dict(config or {})
        self._setup_device(self.config.device)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "LSTMModel":
        """Train the model."""
        # Combine X and y
        train_df = X.copy()
        train_df["y_norm"] = y.values
        
        # Setup sequence config
        self.seq_config = SequenceConfig(
            lookback_window=self.config.lookback_window,
            forecast_horizon=self.config.forecast_horizon,
            target_col="y_norm",
        )
        
        # Identify feature columns
        exclude_cols = ["ndc", "brand_drug_id", "month_id", "y_norm"]
        self.feature_columns = [c for c in X.columns if c not in exclude_cols]
        self.seq_config.time_varying_features = self.feature_columns[:self.config.input_dim]
        
        # Update config
        actual_input_dim = len(self.seq_config.time_varying_features) if self.seq_config.time_varying_features else 1
        self.config.input_dim = max(actual_input_dim, 1)
        
        # Prepare data
        train_dataset = self._prepare_data(train_df, is_train=True)
        
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_df = X_val.copy()
            val_df["y_norm"] = y_val.values
            val_dataset = self._prepare_data(val_df, is_train=False)
        
        # Initialize model
        self.model = LSTMOnlyNetwork(self.config).to(self.device)
        
        # Data loaders
        train_loader = create_dataloader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size=self.config.batch_size, shuffle=False) if val_dataset else None
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        
        # Training
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            train_loss = self._train_epoch(self.model, train_loader, optimizer, gradient_clip=1.0)
            
            val_loss = train_loss
            if val_loader is not None:
                val_loss = self._validate(self.model, val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break
        
        return self
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            torch.save(self.model.state_dict(), path / "model.pt")
        
        metadata = {
            "config": self.config.to_dict(),
            "feature_columns": self.feature_columns,
            "static_columns": self.static_columns,
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
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "LSTMModel":
        """Load model from disk."""
        path = Path(path)
        
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        instance = cls(config=metadata["config"])
        instance.feature_columns = metadata["feature_columns"]
        instance.static_columns = metadata["static_columns"]
        
        if metadata.get("seq_config"):
            instance.seq_config = SequenceConfig(**metadata["seq_config"])
        
        if metadata.get("scaler_params"):
            instance.scaler = SequenceScaler.from_params(metadata["scaler_params"])
        
        instance.model = LSTMOnlyNetwork(instance.config).to(instance.device)
        instance.model.load_state_dict(torch.load(path / "model.pt", map_location=instance.device))
        instance.model.eval()
        
        return instance
