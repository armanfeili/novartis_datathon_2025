"""
Sequence builder utilities for time-series models.

This module provides functions to convert tabular data into sequences
suitable for LSTM, GRU, and CNN-based temporal models.

Reference: Li et al. (2024) CNN-LSTM for drug sales prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SequenceConfig:
    """Configuration for sequence building."""
    
    # Sequence parameters
    lookback_window: int = 12  # Number of historical time steps
    forecast_horizon: int = 1  # Number of steps to predict
    
    # Feature handling
    time_varying_features: Optional[List[str]] = None
    static_features: Optional[List[str]] = None
    target_col: str = "y_norm"
    
    # Grouping
    group_cols: Tuple[str, ...] = ("ndc", "brand_drug_id")
    time_col: str = "month_id"
    
    # Padding
    pad_value: float = 0.0
    min_sequence_length: int = 3
    
    # Scaling
    scale_features: bool = True
    scale_target: bool = True
    
    def __post_init__(self):
        if self.time_varying_features is None:
            self.time_varying_features = []
        if self.static_features is None:
            self.static_features = []


def build_sequences(
    df: pd.DataFrame,
    config: SequenceConfig,
    is_train: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert tabular data to sequences for temporal models.
    
    Args:
        df: DataFrame with columns for group_cols, time_col, features, and target
        config: Sequence configuration
        is_train: Whether this is training data (affects target availability)
        
    Returns:
        Dictionary with:
            - 'X_seq': Time-varying features (B, T, F_tv)
            - 'X_static': Static features (B, F_s)
            - 'y': Targets (B, H) if is_train else None
            - 'masks': Valid timestep masks (B, T)
            - 'group_ids': Group identifiers (B,)
            - 'time_indices': Time indices for each sequence (B, T)
    """
    # Ensure sorted by time within each group
    df = df.sort_values(list(config.group_cols) + [config.time_col])
    
    # Get feature columns
    tv_features = config.time_varying_features or []
    static_features = config.static_features or []
    
    # Auto-detect time-varying features if not specified
    if not tv_features:
        exclude_cols = set(list(config.group_cols) + [config.time_col, config.target_col])
        exclude_cols.update(static_features)
        tv_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    sequences = []
    static_features_list = []
    targets = []
    masks = []
    group_ids = []
    time_indices = []
    
    # Group by entity
    for group_key, group_df in df.groupby(list(config.group_cols)):
        group_df = group_df.reset_index(drop=True)
        n_timesteps = len(group_df)
        
        if n_timesteps < config.min_sequence_length:
            continue
        
        # Extract time-varying features
        if tv_features:
            tv_data = group_df[tv_features].values
        else:
            tv_data = np.zeros((n_timesteps, 1))
        
        # Extract static features (use first row values)
        if static_features:
            static_data = group_df[static_features].iloc[0].values
        else:
            static_data = np.zeros(1)
        
        # Extract target if available
        if is_train and config.target_col in group_df.columns:
            target_data = group_df[config.target_col].values
        else:
            target_data = None
        
        # Create sliding window sequences
        for start_idx in range(n_timesteps - config.lookback_window - config.forecast_horizon + 1):
            end_idx = start_idx + config.lookback_window
            
            # Extract sequence
            seq = tv_data[start_idx:end_idx]
            
            # Create mask (all valid since we only create full sequences here)
            mask = np.ones(config.lookback_window)
            
            # Extract target
            if target_data is not None:
                target_start = end_idx
                target_end = target_start + config.forecast_horizon
                y = target_data[target_start:target_end]
            else:
                y = None
            
            sequences.append(seq)
            static_features_list.append(static_data)
            if y is not None:
                targets.append(y)
            masks.append(mask)
            group_ids.append(group_key)
            time_indices.append(group_df[config.time_col].values[start_idx:end_idx])
    
    # Stack arrays
    result = {
        "X_seq": np.array(sequences, dtype=np.float32),
        "X_static": np.array(static_features_list, dtype=np.float32),
        "masks": np.array(masks, dtype=np.float32),
        "group_ids": np.array(group_ids),
        "time_indices": np.array(time_indices),
    }
    
    if targets:
        result["y"] = np.array(targets, dtype=np.float32)
    else:
        result["y"] = None
    
    return result


def build_sequences_for_inference(
    df: pd.DataFrame,
    config: SequenceConfig,
) -> Dict[str, np.ndarray]:
    """
    Build sequences for inference, using all available history.
    
    Unlike training, we create one sequence per group using the most recent
    lookback_window timesteps.
    
    Args:
        df: DataFrame with historical data
        config: Sequence configuration
        
    Returns:
        Dictionary with sequence arrays
    """
    df = df.sort_values(list(config.group_cols) + [config.time_col])
    
    tv_features = config.time_varying_features or []
    static_features = config.static_features or []
    
    # Auto-detect if not specified
    if not tv_features:
        exclude_cols = set(list(config.group_cols) + [config.time_col, config.target_col])
        exclude_cols.update(static_features)
        tv_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    sequences = []
    static_features_list = []
    masks = []
    group_ids = []
    
    for group_key, group_df in df.groupby(list(config.group_cols)):
        group_df = group_df.reset_index(drop=True)
        n_timesteps = len(group_df)
        
        # Extract time-varying features
        if tv_features:
            tv_data = group_df[tv_features].values
        else:
            tv_data = np.zeros((n_timesteps, 1))
        
        # Extract static features
        if static_features:
            static_data = group_df[static_features].iloc[0].values
        else:
            static_data = np.zeros(1)
        
        # Use last lookback_window timesteps (or pad if shorter)
        if n_timesteps >= config.lookback_window:
            seq = tv_data[-config.lookback_window:]
            mask = np.ones(config.lookback_window)
        else:
            # Pad with config.pad_value
            pad_length = config.lookback_window - n_timesteps
            seq = np.vstack([
                np.full((pad_length, tv_data.shape[1]), config.pad_value),
                tv_data
            ])
            mask = np.concatenate([
                np.zeros(pad_length),
                np.ones(n_timesteps)
            ])
        
        sequences.append(seq)
        static_features_list.append(static_data)
        masks.append(mask)
        group_ids.append(group_key)
    
    return {
        "X_seq": np.array(sequences, dtype=np.float32),
        "X_static": np.array(static_features_list, dtype=np.float32),
        "masks": np.array(masks, dtype=np.float32),
        "group_ids": np.array(group_ids),
        "y": None,
    }


def pad_sequences(
    sequences: List[np.ndarray],
    max_length: Optional[int] = None,
    pad_value: float = 0.0,
    padding: str = "pre",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences to uniform length.
    
    Args:
        sequences: List of arrays of shape (T_i, F)
        max_length: Maximum length to pad to (None = use longest)
        pad_value: Value to use for padding
        padding: 'pre' or 'post' padding
        
    Returns:
        Tuple of (padded_sequences, masks)
            - padded_sequences: (B, max_length, F)
            - masks: (B, max_length) with 1 for valid, 0 for padded
    """
    if not sequences:
        raise ValueError("Empty sequence list")
    
    n_features = sequences[0].shape[1] if sequences[0].ndim > 1 else 1
    lengths = [len(s) for s in sequences]
    
    if max_length is None:
        max_length = max(lengths)
    
    batch_size = len(sequences)
    padded = np.full((batch_size, max_length, n_features), pad_value, dtype=np.float32)
    masks = np.zeros((batch_size, max_length), dtype=np.float32)
    
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        
        actual_length = min(length, max_length)
        
        if padding == "pre":
            start = max_length - actual_length
            padded[i, start:, :] = seq[-actual_length:]
            masks[i, start:] = 1.0
        else:  # post
            padded[i, :actual_length, :] = seq[:actual_length]
            masks[i, :actual_length] = 1.0
    
    return padded, masks


def create_time_features(
    df: pd.DataFrame,
    time_col: str = "month_id",
    base_month: int = 0,
) -> pd.DataFrame:
    """
    Create temporal encoding features.
    
    Args:
        df: DataFrame with time column
        time_col: Name of time column (integer month ID)
        base_month: Reference month for relative encoding
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Relative time
    df["time_relative"] = df[time_col] - base_month
    
    # Cyclical encoding for month-of-year (assuming 12-month cycle)
    month_in_year = df[time_col] % 12
    df["month_sin"] = np.sin(2 * np.pi * month_in_year / 12)
    df["month_cos"] = np.cos(2 * np.pi * month_in_year / 12)
    
    # Quarter encoding
    quarter = (month_in_year // 3) % 4
    df["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)
    
    return df


class SequenceScaler:
    """
    Scaler for sequence data with separate handling of time-varying and static features.
    """
    
    def __init__(
        self,
        method: str = "standard",
        feature_range: Tuple[float, float] = (0, 1),
    ):
        """
        Initialize scaler.
        
        Args:
            method: 'standard' (z-score), 'minmax', or 'robust'
            feature_range: Range for minmax scaling
        """
        self.method = method
        self.feature_range = feature_range
        
        self._seq_params: Optional[Dict[str, np.ndarray]] = None
        self._static_params: Optional[Dict[str, np.ndarray]] = None
        self._target_params: Optional[Dict[str, float]] = None
    
    def fit(
        self,
        X_seq: np.ndarray,
        X_static: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> "SequenceScaler":
        """
        Fit scaler on training data.
        
        Args:
            X_seq: Time-varying features (B, T, F_tv)
            X_static: Static features (B, F_s)
            y: Targets (B, H)
        """
        # Fit on sequence features (flatten time dimension)
        X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
        self._seq_params = self._compute_params(X_seq_flat)
        
        # Fit on static features
        if X_static is not None and X_static.shape[-1] > 0:
            self._static_params = self._compute_params(X_static)
        
        # Fit on target
        if y is not None:
            y_flat = y.flatten()
            self._target_params = {
                "mean": float(np.mean(y_flat)),
                "std": float(np.std(y_flat) + 1e-8),
                "min": float(np.min(y_flat)),
                "max": float(np.max(y_flat)),
            }
        
        return self
    
    def _compute_params(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute scaling parameters."""
        if self.method == "standard":
            return {
                "mean": np.mean(X, axis=0),
                "std": np.std(X, axis=0) + 1e-8,
            }
        elif self.method == "minmax":
            return {
                "min": np.min(X, axis=0),
                "max": np.max(X, axis=0),
            }
        elif self.method == "robust":
            return {
                "median": np.median(X, axis=0),
                "iqr": np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-8,
            }
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
    
    def transform(
        self,
        X_seq: np.ndarray,
        X_static: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Transform data using fitted parameters.
        """
        # Transform sequences
        X_seq_scaled = self._apply_transform(X_seq, self._seq_params)
        
        # Transform static features
        X_static_scaled = None
        if X_static is not None and self._static_params is not None:
            X_static_scaled = self._apply_transform(X_static, self._static_params)
        elif X_static is not None:
            X_static_scaled = X_static
        
        # Transform target
        y_scaled = None
        if y is not None and self._target_params is not None:
            if self.method == "standard":
                y_scaled = (y - self._target_params["mean"]) / self._target_params["std"]
            elif self.method == "minmax":
                range_val = self._target_params["max"] - self._target_params["min"] + 1e-8
                y_scaled = (y - self._target_params["min"]) / range_val
            else:
                y_scaled = y
        
        return X_seq_scaled, X_static_scaled, y_scaled
    
    def _apply_transform(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply transformation to array."""
        if self.method == "standard":
            return (X - params["mean"]) / params["std"]
        elif self.method == "minmax":
            range_val = params["max"] - params["min"] + 1e-8
            scaled = (X - params["min"]) / range_val
            a, b = self.feature_range
            return scaled * (b - a) + a
        elif self.method == "robust":
            return (X - params["median"]) / params["iqr"]
        else:
            return X
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform target values."""
        if self._target_params is None:
            return y_scaled
        
        if self.method == "standard":
            return y_scaled * self._target_params["std"] + self._target_params["mean"]
        elif self.method == "minmax":
            a, b = self.feature_range
            range_val = self._target_params["max"] - self._target_params["min"] + 1e-8
            return (y_scaled - a) / (b - a) * range_val + self._target_params["min"]
        else:
            return y_scaled
    
    def get_params(self) -> Dict[str, Any]:
        """Get all scaling parameters for serialization."""
        return {
            "method": self.method,
            "feature_range": self.feature_range,
            "seq_params": self._seq_params,
            "static_params": self._static_params,
            "target_params": self._target_params,
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "SequenceScaler":
        """Create scaler from saved parameters."""
        scaler = cls(
            method=params["method"],
            feature_range=tuple(params["feature_range"]),
        )
        scaler._seq_params = params.get("seq_params")
        scaler._static_params = params.get("static_params")
        scaler._target_params = params.get("target_params")
        return scaler


if TORCH_AVAILABLE:
    class SequenceDataset(Dataset):
        """
        PyTorch Dataset for sequence data.
        """
        
        def __init__(
            self,
            X_seq: np.ndarray,
            X_static: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            masks: Optional[np.ndarray] = None,
        ):
            """
            Initialize dataset.
            
            Args:
                X_seq: Time-varying features (B, T, F_tv)
                X_static: Static features (B, F_s)
                y: Targets (B, H)
                masks: Valid timestep masks (B, T)
            """
            self.X_seq = torch.from_numpy(X_seq).float()
            self.X_static = torch.from_numpy(X_static).float() if X_static is not None else None
            self.y = torch.from_numpy(y).float() if y is not None else None
            self.masks = torch.from_numpy(masks).float() if masks is not None else None
        
        def __len__(self) -> int:
            return len(self.X_seq)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            item = {"X_seq": self.X_seq[idx]}
            
            if self.X_static is not None:
                item["X_static"] = self.X_static[idx]
            
            if self.y is not None:
                item["y"] = self.y[idx]
            
            if self.masks is not None:
                item["mask"] = self.masks[idx]
            
            return item
    
    
    class GraphSequenceDataset(Dataset):
        """
        PyTorch Dataset for graph-enhanced sequence data.
        
        Combines node features, adjacency matrix, and sequence data.
        """
        
        def __init__(
            self,
            X_seq: np.ndarray,
            X_static: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            masks: Optional[np.ndarray] = None,
            node_features: Optional[np.ndarray] = None,
            adjacency: Optional[np.ndarray] = None,
            node_indices: Optional[np.ndarray] = None,
        ):
            """
            Initialize dataset.
            
            Args:
                X_seq: Time-varying features (B, T, F_tv)
                X_static: Static features (B, F_s)
                y: Targets (B, H)
                masks: Valid timestep masks (B, T)
                node_features: Node features for graph (N, F_node)
                adjacency: Adjacency matrix (N, N)
                node_indices: Index into node_features for each sample (B,)
            """
            self.X_seq = torch.from_numpy(X_seq).float()
            self.X_static = torch.from_numpy(X_static).float() if X_static is not None else None
            self.y = torch.from_numpy(y).float() if y is not None else None
            self.masks = torch.from_numpy(masks).float() if masks is not None else None
            
            self.node_features = torch.from_numpy(node_features).float() if node_features is not None else None
            self.adjacency = torch.from_numpy(adjacency).float() if adjacency is not None else None
            self.node_indices = torch.from_numpy(node_indices).long() if node_indices is not None else None
        
        def __len__(self) -> int:
            return len(self.X_seq)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            item = {"X_seq": self.X_seq[idx]}
            
            if self.X_static is not None:
                item["X_static"] = self.X_static[idx]
            
            if self.y is not None:
                item["y"] = self.y[idx]
            
            if self.masks is not None:
                item["mask"] = self.masks[idx]
            
            if self.node_indices is not None:
                item["node_idx"] = self.node_indices[idx]
            
            return item
        
        def get_graph_data(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            """Return graph data (shared across all samples)."""
            return self.node_features, self.adjacency
    
    
    def create_dataloader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create DataLoader with standard settings."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
else:
    # Stub classes when torch is not available
    class SequenceDataset:  # type: ignore
        """Stub when torch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for SequenceDataset")
    
    class GraphSequenceDataset:  # type: ignore
        """Stub when torch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for GraphSequenceDataset")
    
    def create_dataloader(*args, **kwargs):
        raise ImportError("torch is required for create_dataloader")


def prepare_sequences_from_df(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    config: Optional[SequenceConfig] = None,
    scale: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to prepare all sequence data from DataFrames.
    
    Args:
        train_df: Training DataFrame
        val_df: Optional validation DataFrame
        config: Sequence configuration
        scale: Whether to apply scaling
        
    Returns:
        Dictionary with:
            - 'train_data': Dict with X_seq, X_static, y, masks
            - 'val_data': Dict with X_seq, X_static, y, masks (if val_df provided)
            - 'scaler': Fitted SequenceScaler (if scale=True)
            - 'config': SequenceConfig used
    """
    if config is None:
        config = SequenceConfig()
    
    # Build training sequences
    train_data = build_sequences(train_df, config, is_train=True)
    
    # Build validation sequences
    val_data = None
    if val_df is not None:
        val_data = build_sequences(val_df, config, is_train=True)
    
    # Apply scaling
    scaler = None
    if scale:
        scaler = SequenceScaler(method="standard")
        scaler.fit(
            train_data["X_seq"],
            train_data["X_static"],
            train_data["y"],
        )
        
        train_data["X_seq"], train_data["X_static"], train_data["y"] = scaler.transform(
            train_data["X_seq"],
            train_data["X_static"],
            train_data["y"],
        )
        
        if val_data is not None:
            val_data["X_seq"], val_data["X_static"], val_data["y"] = scaler.transform(
                val_data["X_seq"],
                val_data["X_static"],
                val_data["y"],
            )
    
    return {
        "train_data": train_data,
        "val_data": val_data,
        "scaler": scaler,
        "config": config,
    }
