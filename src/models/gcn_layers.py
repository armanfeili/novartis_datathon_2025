"""
Graph Convolutional Network (GCN) Layers for KG-GCN-LSTM model.

This module implements GCN layers following the spectral graph convolution approach
from Kipf & Welling (2017) "Semi-Supervised Classification with Graph Convolutional Networks".

Reference: KG-GCN-LSTM paper for pharmaceutical demand forecasting.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _init_weights(weight: Tensor, bias: Optional[Tensor] = None) -> None:
    """Initialize layer weights using Xavier uniform initialization."""
    nn.init.xavier_uniform_(weight)
    if bias is not None:
        nn.init.zeros_(bias)


def normalize_adjacency(adj: Tensor, add_self_loops: bool = True) -> Tensor:
    """
    Normalize adjacency matrix using symmetric normalization.
    
    A_norm = D^(-1/2) @ A @ D^(-1/2)
    
    Args:
        adj: Adjacency matrix of shape (N, N)
        add_self_loops: Whether to add self-loops (identity) before normalization
        
    Returns:
        Normalized adjacency matrix
    """
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    
    # Compute degree matrix
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    # D^(-1/2) @ A @ D^(-1/2)
    deg_inv_sqrt_mat = torch.diag(deg_inv_sqrt)
    adj_norm = deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat
    
    return adj_norm


def sparse_normalize_adjacency(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: int = -1,
    add_self_loops: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Normalize adjacency for sparse representation.
    
    Args:
        edge_index: Edge indices of shape (2, E)
        edge_weight: Edge weights of shape (E,)
        num_nodes: Number of nodes
        add_self_loops: Whether to add self-loops
        
    Returns:
        Tuple of (edge_index, normalized_edge_weight)
    """
    if num_nodes < 0:
        num_nodes = int(edge_index.max()) + 1
    
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    
    if add_self_loops:
        # Add self-loops
        loop_index = torch.arange(num_nodes, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        loop_weight = torch.ones(num_nodes, device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight])
    
    # Compute degree
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, edge_weight)
    
    # Symmetric normalization
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    edge_weight_norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    return edge_index, edge_weight_norm


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer.
    
    Implements: H' = σ(A_norm @ H @ W + b)
    
    Where:
        - A_norm is the normalized adjacency matrix
        - H is the input node features
        - W is the learnable weight matrix
        - b is the optional bias
        - σ is the activation function
        
    Args:
        in_features: Size of input features per node
        out_features: Size of output features per node
        bias: Whether to include bias term
        activation: Activation function ('relu', 'leaky_relu', 'elu', 'tanh', 'none')
        dropout: Dropout probability applied to input features
        normalize: Whether to apply symmetric normalization to adjacency
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        _init_weights(self.weight, self.bias)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "none": nn.Identity(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        adj_normalized: bool = False,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features of shape (N, in_features) or (B, N, in_features)
            adj: Adjacency matrix of shape (N, N) or (B, N, N)
            adj_normalized: Whether adjacency is already normalized
            
        Returns:
            Output features of shape (N, out_features) or (B, N, out_features)
        """
        # Apply dropout to input
        x = self.dropout(x)
        
        # Handle batched input
        is_batched = x.dim() == 3
        
        if self.normalize and not adj_normalized:
            if is_batched:
                # Normalize each adjacency matrix in batch
                adj = torch.stack([normalize_adjacency(a) for a in adj])
            else:
                adj = normalize_adjacency(adj)
        
        # Linear transformation: X @ W
        support = torch.matmul(x, self.weight)
        
        # Graph convolution: A @ (X @ W)
        if is_batched:
            output = torch.bmm(adj, support)
        else:
            output = torch.matmul(adj, support)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        # Apply activation
        output = self.activation(output)
        
        return output
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class GCNEncoder(nn.Module):
    """
    Multi-layer GCN Encoder.
    
    Stacks multiple GCN layers with optional skip connections.
    
    Args:
        in_features: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        out_features: Output feature dimension
        dropout: Dropout probability between layers
        activation: Activation function for hidden layers
        skip_connection: Whether to add residual skip connections
        layer_norm: Whether to apply layer normalization
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        out_features: int,
        dropout: float = 0.1,
        activation: str = "relu",
        skip_connection: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        
        # Build layer dimensions
        dims = [in_features] + hidden_dims + [out_features]
        
        # Create GCN layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if layer_norm else None
        self.skip_projections = nn.ModuleList() if skip_connection else None
        
        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            self.layers.append(
                GCNLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    bias=True,
                    activation="none" if is_last else activation,
                    dropout=0.0 if is_last else dropout,
                    normalize=True,
                )
            )
            
            if layer_norm and not is_last:
                self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
            
            if skip_connection and dims[i] != dims[i + 1]:
                self.skip_projections.append(nn.Linear(dims[i], dims[i + 1]))
            elif skip_connection:
                self.skip_projections.append(nn.Identity())
    
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        return_all_layers: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """
        Forward pass through all GCN layers.
        
        Args:
            x: Node features of shape (N, in_features) or (B, N, in_features)
            adj: Adjacency matrix
            return_all_layers: Whether to return intermediate representations
            
        Returns:
            Final node embeddings, optionally with all intermediate outputs
        """
        # Normalize adjacency once
        is_batched = x.dim() == 3
        if is_batched:
            adj_norm = torch.stack([normalize_adjacency(a) for a in adj])
        else:
            adj_norm = normalize_adjacency(adj)
        
        all_outputs = []
        h = x
        
        for i, layer in enumerate(self.layers):
            h_prev = h
            h = layer(h, adj_norm, adj_normalized=True)
            
            # Apply layer normalization
            if self.layer_norms is not None and i < len(self.layer_norms):
                h = self.layer_norms[i](h)
            
            # Apply skip connection
            if self.skip_connection and self.skip_projections is not None:
                h = h + self.skip_projections[i](h_prev)
            
            if return_all_layers:
                all_outputs.append(h)
        
        if return_all_layers:
            return h, all_outputs
        return h


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    
    Implements attention-based graph convolution from Veličković et al. (2018).
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        heads: Number of attention heads
        concat: Whether to concatenate heads (True) or average (False)
        dropout: Dropout on attention weights
        negative_slope: LeakyReLU negative slope for attention
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.empty(heads, in_features, out_features))
        
        # Attention parameters
        self.a_src = nn.Parameter(torch.empty(heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(heads, out_features, 1))
        
        # Bias
        if bias:
            if concat:
                self.bias = nn.Parameter(torch.empty(heads * out_features))
            else:
                self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.xavier_uniform_(self.a_src, gain=gain)
        nn.init.xavier_uniform_(self.a_dst, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features of shape (N, in_features)
            adj: Adjacency matrix of shape (N, N) - used as mask
            
        Returns:
            Output features of shape (N, heads * out_features) if concat
            else (N, out_features)
        """
        N = x.size(0)
        
        # Linear transformation for all heads: (N, in_features) -> (heads, N, out_features)
        # x: (N, in_features), W: (heads, in_features, out_features)
        h = torch.einsum('ni,hio->hno', x, self.W)  # (heads, N, out_features)
        
        # Compute attention scores
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # Split attention: e_ij = a_src^T Wh_i + a_dst^T Wh_j
        attn_src = torch.matmul(h, self.a_src).squeeze(-1)  # (heads, N)
        attn_dst = torch.matmul(h, self.a_dst).squeeze(-1)  # (heads, N)
        
        # Broadcast to get pairwise attention
        # attn_src: (heads, N, 1), attn_dst: (heads, 1, N)
        e = attn_src.unsqueeze(-1) + attn_dst.unsqueeze(-2)  # (heads, N, N)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)
        
        # Mask attention with adjacency (including self-loops)
        adj_with_self = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
        mask = (adj_with_self == 0).unsqueeze(0)  # (1, N, N)
        e = e.masked_fill(mask, float('-inf'))
        
        # Softmax normalization
        alpha = F.softmax(e, dim=-1)  # (heads, N, N)
        alpha = self.dropout(alpha)
        
        # Aggregate neighbor features
        # h: (heads, N, out_features), alpha: (heads, N, N)
        out = torch.bmm(alpha, h)  # (heads, N, out_features)
        
        # Combine heads
        if self.concat:
            out = out.permute(1, 0, 2).reshape(N, -1)  # (N, heads * out_features)
        else:
            out = out.mean(dim=0)  # (N, out_features)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network Encoder.
    
    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden dimension per head
        out_features: Output feature dimension
        num_layers: Number of GAT layers
        heads: Number of attention heads per layer
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer
            self.layers.append(
                GraphAttentionLayer(
                    in_features, out_features, heads=1, concat=False, dropout=dropout
                )
            )
        else:
            # First layer
            self.layers.append(
                GraphAttentionLayer(
                    in_features, hidden_dim, heads=heads, concat=True, dropout=dropout
                )
            )
            
            # Intermediate layers
            for _ in range(num_layers - 2):
                self.layers.append(
                    GraphAttentionLayer(
                        hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout
                    )
                )
            
            # Output layer
            self.layers.append(
                GraphAttentionLayer(
                    hidden_dim * heads, out_features, heads=1, concat=False, dropout=dropout
                )
            )
        
        self.elu = nn.ELU()
    
    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Forward pass through all GAT layers."""
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = self.elu(x)
        return x


class GraphReadout(nn.Module):
    """
    Graph-level readout from node embeddings.
    
    Supports multiple pooling strategies:
        - mean: Average pooling over nodes
        - max: Max pooling over nodes
        - sum: Sum pooling over nodes
        - attention: Attention-weighted pooling
        - set2set: Set2Set pooling (learnable)
        
    Args:
        in_features: Input node embedding dimension
        out_features: Output graph embedding dimension (for attention/set2set)
        pooling: Pooling strategy
        num_iterations: Number of iterations for Set2Set
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        pooling: str = "mean",
        num_iterations: int = 3,
    ):
        super().__init__()
        self.pooling = pooling
        self.in_features = in_features
        self.out_features = out_features or in_features
        
        if pooling == "attention":
            self.attention_mlp = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.Tanh(),
                nn.Linear(in_features // 2, 1),
            )
            self.transform = nn.Linear(in_features, self.out_features)
        elif pooling == "set2set":
            self.num_iterations = num_iterations
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)
            self.transform = nn.Linear(2 * in_features, self.out_features)
    
    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute graph-level embedding.
        
        Args:
            x: Node embeddings of shape (N, in_features) or (B, N, in_features)
            batch: Batch assignment for each node (for non-batched input)
            
        Returns:
            Graph embeddings of shape (B, out_features) or (out_features,)
        """
        is_batched = x.dim() == 3
        
        if self.pooling == "mean":
            if is_batched:
                return x.mean(dim=1)
            elif batch is not None:
                return self._scatter_mean(x, batch)
            else:
                return x.mean(dim=0)
        
        elif self.pooling == "max":
            if is_batched:
                return x.max(dim=1)[0]
            elif batch is not None:
                return self._scatter_max(x, batch)
            else:
                return x.max(dim=0)[0]
        
        elif self.pooling == "sum":
            if is_batched:
                return x.sum(dim=1)
            elif batch is not None:
                return self._scatter_sum(x, batch)
            else:
                return x.sum(dim=0)
        
        elif self.pooling == "attention":
            return self._attention_pooling(x, is_batched, batch)
        
        elif self.pooling == "set2set":
            return self._set2set_pooling(x, is_batched)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def _attention_pooling(
        self,
        x: Tensor,
        is_batched: bool,
        batch: Optional[Tensor],
    ) -> Tensor:
        """Attention-weighted pooling."""
        # Compute attention weights
        attn_scores = self.attention_mlp(x)  # (N, 1) or (B, N, 1)
        
        if is_batched:
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = (x * attn_weights).sum(dim=1)
        else:
            if batch is not None:
                # Apply softmax within each graph
                attn_weights = self._scatter_softmax(attn_scores.squeeze(-1), batch)
                pooled = self._scatter_sum(x * attn_weights.unsqueeze(-1), batch)
            else:
                attn_weights = F.softmax(attn_scores, dim=0)
                pooled = (x * attn_weights).sum(dim=0)
        
        return self.transform(pooled)
    
    def _set2set_pooling(self, x: Tensor, is_batched: bool) -> Tensor:
        """Set2Set pooling with LSTM."""
        if not is_batched:
            x = x.unsqueeze(0)  # Add batch dimension
        
        B, N, D = x.shape
        
        # Initialize query
        q = torch.zeros(B, D, device=x.device)
        h = (torch.zeros(1, B, D, device=x.device), 
             torch.zeros(1, B, D, device=x.device))
        
        for _ in range(self.num_iterations):
            # Attention over nodes
            energy = torch.bmm(x, q.unsqueeze(-1)).squeeze(-1)  # (B, N)
            attn = F.softmax(energy, dim=1)  # (B, N)
            
            # Read from nodes
            r = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # (B, D)
            
            # Update query with LSTM
            lstm_input = torch.cat([q, r], dim=-1).unsqueeze(1)  # (B, 1, 2D)
            # Note: LSTM expects input of same dimension as hidden
            _, h = self.lstm(r.unsqueeze(1), h)
            q = h[0].squeeze(0)  # (B, D)
        
        # Final readout
        final = torch.cat([q, r], dim=-1)  # (B, 2D)
        output = self.transform(final)  # (B, out_features)
        
        if not is_batched:
            output = output.squeeze(0)
        
        return output
    
    def _scatter_mean(self, x: Tensor, batch: Tensor) -> Tensor:
        """Scatter mean operation."""
        num_graphs = int(batch.max()) + 1
        out = torch.zeros(num_graphs, x.size(-1), device=x.device)
        count = torch.zeros(num_graphs, device=x.device)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        count.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        return out / count.unsqueeze(-1).clamp(min=1)
    
    def _scatter_max(self, x: Tensor, batch: Tensor) -> Tensor:
        """Scatter max operation."""
        num_graphs = int(batch.max()) + 1
        out = torch.full((num_graphs, x.size(-1)), float('-inf'), device=x.device)
        out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(x), x, reduce='amax')
        return out
    
    def _scatter_sum(self, x: Tensor, batch: Tensor) -> Tensor:
        """Scatter sum operation."""
        num_graphs = int(batch.max()) + 1
        out = torch.zeros(num_graphs, x.size(-1), device=x.device)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        return out
    
    def _scatter_softmax(self, x: Tensor, batch: Tensor) -> Tensor:
        """Scatter softmax operation."""
        num_graphs = int(batch.max()) + 1
        max_vals = torch.zeros(num_graphs, device=x.device)
        max_vals.scatter_reduce_(0, batch, x, reduce='amax')
        x = x - max_vals[batch]
        exp_x = torch.exp(x)
        sum_exp = torch.zeros(num_graphs, device=x.device)
        sum_exp.scatter_add_(0, batch, exp_x)
        return exp_x / sum_exp[batch]


def create_gcn_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create GCN encoder from config.
    
    Args:
        config: Configuration dictionary with keys:
            - encoder_type: 'gcn' or 'gat'
            - in_features: Input dimension
            - hidden_dims: List of hidden dimensions (for GCN)
            - hidden_dim: Hidden dimension (for GAT)
            - out_features: Output dimension
            - num_layers: Number of layers (for GAT)
            - heads: Number of attention heads (for GAT)
            - dropout: Dropout probability
            - skip_connection: Whether to use skip connections
            - layer_norm: Whether to use layer normalization
            
    Returns:
        GCN or GAT encoder module
    """
    encoder_type = config.get("encoder_type", "gcn")
    
    if encoder_type == "gcn":
        return GCNEncoder(
            in_features=config["in_features"],
            hidden_dims=config.get("hidden_dims", [64]),
            out_features=config["out_features"],
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            skip_connection=config.get("skip_connection", False),
            layer_norm=config.get("layer_norm", False),
        )
    elif encoder_type == "gat":
        return GATEncoder(
            in_features=config["in_features"],
            hidden_dim=config.get("hidden_dim", 32),
            out_features=config["out_features"],
            num_layers=config.get("num_layers", 2),
            heads=config.get("heads", 4),
            dropout=config.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
