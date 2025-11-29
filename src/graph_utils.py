"""
Graph utilities for Knowledge Graph-based modeling.

This module provides utilities for building and manipulating graphs
representing pharmaceutical knowledge, including drug relationships,
therapeutic areas, and competitive dynamics.

Based on research from:
- KG-GCN-LSTM: Knowledge Graph GCN with LSTM for pharma forecasting

Graph structure:
- Nodes: Drugs/brands with feature vectors
- Edges: Relationships (therapeutic_similarity, competitive, same_country, etc.)
- Edge weights: Similarity/strength measures
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


# =============================================================================
# Edge Types
# =============================================================================

class EdgeType(Enum):
    """Types of edges in the drug knowledge graph."""
    THERAPEUTIC_SIMILARITY = "therapeutic_similarity"  # Same therapeutic area
    COMPETITIVE = "competitive"  # Compete in same market
    SAME_COUNTRY = "same_country"  # Available in same country
    SAME_PACKAGE = "same_package"  # Same dosage form
    SIMILAR_EROSION = "similar_erosion"  # Similar erosion patterns
    TEMPORAL = "temporal"  # Temporal sequence within same series


@dataclass
class Edge:
    """Represents an edge in the knowledge graph."""
    source: int  # Source node index
    target: int  # Target node index
    edge_type: EdgeType
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    index: int
    country: str
    brand_name: str
    features: np.ndarray  # Node feature vector
    attributes: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Graph Data Structure
# =============================================================================

@dataclass
class DrugGraph:
    """
    Drug knowledge graph structure.
    
    Attributes:
        nodes: List of Node objects
        edges: List of Edge objects
        node_index: Mapping from (country, brand) to node index
        adjacency_matrix: Sparse adjacency matrix
        edge_index: Edge index tensor (2, num_edges) for GNN
        edge_weight: Edge weight tensor
        node_features: Node feature matrix (num_nodes, feature_dim)
    """
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    node_index: Dict[Tuple[str, str], int] = field(default_factory=dict)
    adjacency_matrix: Optional[sparse.csr_matrix] = None
    edge_index: Optional[np.ndarray] = None  # Shape (2, num_edges)
    edge_weight: Optional[np.ndarray] = None
    node_features: Optional[np.ndarray] = None
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    @property
    def feature_dim(self) -> int:
        if self.node_features is not None:
            return self.node_features.shape[1]
        if self.nodes and self.nodes[0].features is not None:
            return len(self.nodes[0].features)
        return 0
    
    def get_node_index(self, country: str, brand: str) -> Optional[int]:
        """Get node index for a (country, brand) pair."""
        return self.node_index.get((country, brand))
    
    def get_neighbors(self, node_idx: int) -> List[int]:
        """Get neighbor indices for a node."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_idx:
                neighbors.append(edge.target)
            elif edge.target == node_idx:
                neighbors.append(edge.source)
        return list(set(neighbors))
    
    def to_pyg_data(self):
        """
        Convert to PyTorch Geometric Data object.
        
        Returns:
            torch_geometric.data.Data object (if available)
        """
        try:
            import torch
            from torch_geometric.data import Data
            
            x = torch.tensor(self.node_features, dtype=torch.float32)
            edge_index = torch.tensor(self.edge_index, dtype=torch.long)
            
            edge_attr = None
            if self.edge_weight is not None:
                edge_attr = torch.tensor(self.edge_weight, dtype=torch.float32).unsqueeze(1)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except ImportError:
            logger.warning("PyTorch Geometric not available")
            return None


# =============================================================================
# Graph Building Functions
# =============================================================================

def build_drug_graph(
    panel_df: pd.DataFrame,
    feature_cols: List[str],
    edge_types: Optional[List[EdgeType]] = None,
    therapeutic_threshold: float = 0.0,
    erosion_similarity_threshold: float = 0.8,
    add_self_loops: bool = True,
    normalize_features: bool = True,
) -> DrugGraph:
    """
    Build a drug knowledge graph from panel data.
    
    Args:
        panel_df: Panel data with drug information.
                  Must have: country, brand_name, and feature columns.
        feature_cols: List of feature columns for node features.
        edge_types: List of edge types to include. Default: all types.
        therapeutic_threshold: Threshold for therapeutic similarity edges.
        erosion_similarity_threshold: Threshold for erosion similarity edges.
        add_self_loops: Whether to add self-loop edges.
        normalize_features: Whether to normalize node features.
        
    Returns:
        DrugGraph object with nodes, edges, and matrices.
    """
    if edge_types is None:
        edge_types = [EdgeType.THERAPEUTIC_SIMILARITY, EdgeType.SAME_COUNTRY, 
                      EdgeType.SAME_PACKAGE]
    
    # Get unique series (nodes)
    series_df = panel_df.groupby(['country', 'brand_name']).first().reset_index()
    
    graph = DrugGraph()
    
    # Build nodes
    for idx, row in series_df.iterrows():
        country = row['country']
        brand = row['brand_name']
        
        # Extract features
        features = []
        for col in feature_cols:
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    val = 0.0
                elif isinstance(val, (str, bool)):
                    val = 1.0 if val else 0.0
                features.append(float(val))
            else:
                features.append(0.0)
        
        node = Node(
            index=idx,
            country=country,
            brand_name=brand,
            features=np.array(features),
            attributes={
                'ther_area': row.get('ther_area', ''),
                'main_package': row.get('main_package', ''),
            }
        )
        
        graph.nodes.append(node)
        graph.node_index[(country, brand)] = idx
    
    # Build node feature matrix
    feature_matrix = np.array([n.features for n in graph.nodes])
    
    # Normalize features
    if normalize_features and len(feature_matrix) > 0:
        mean = feature_matrix.mean(axis=0, keepdims=True)
        std = feature_matrix.std(axis=0, keepdims=True) + 1e-6
        feature_matrix = (feature_matrix - mean) / std
        
        # Update node features
        for i, node in enumerate(graph.nodes):
            node.features = feature_matrix[i]
    
    graph.node_features = feature_matrix
    
    # Build edges
    n_nodes = len(graph.nodes)
    
    for i, node_i in enumerate(graph.nodes):
        for j, node_j in enumerate(graph.nodes):
            if i >= j and not add_self_loops:
                continue
            if i == j and not add_self_loops:
                continue
            
            for edge_type in edge_types:
                edge = _create_edge_if_valid(
                    node_i, node_j, edge_type, 
                    therapeutic_threshold, erosion_similarity_threshold
                )
                if edge is not None:
                    graph.edges.append(edge)
    
    # Add self-loops
    if add_self_loops:
        for i, node in enumerate(graph.nodes):
            graph.edges.append(Edge(
                source=i,
                target=i,
                edge_type=EdgeType.TEMPORAL,
                weight=1.0
            ))
    
    # Build edge index and weight tensors
    if graph.edges:
        sources = [e.source for e in graph.edges]
        targets = [e.target for e in graph.edges]
        weights = [e.weight for e in graph.edges]
        
        graph.edge_index = np.array([sources, targets])
        graph.edge_weight = np.array(weights)
    else:
        graph.edge_index = np.zeros((2, 0), dtype=np.int64)
        graph.edge_weight = np.array([])
    
    # Build adjacency matrix
    graph.adjacency_matrix = build_adjacency_matrix(graph, normalized=True)
    
    logger.info(f"Built drug graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    return graph


def _create_edge_if_valid(
    node_i: Node,
    node_j: Node,
    edge_type: EdgeType,
    ther_threshold: float,
    erosion_threshold: float,
) -> Optional[Edge]:
    """Create an edge between two nodes if criteria are met."""
    
    if edge_type == EdgeType.THERAPEUTIC_SIMILARITY:
        # Same therapeutic area
        ther_i = node_i.attributes.get('ther_area', '')
        ther_j = node_j.attributes.get('ther_area', '')
        
        if ther_i and ther_j and ther_i == ther_j:
            return Edge(
                source=node_i.index,
                target=node_j.index,
                edge_type=edge_type,
                weight=1.0
            )
    
    elif edge_type == EdgeType.SAME_COUNTRY:
        if node_i.country == node_j.country:
            return Edge(
                source=node_i.index,
                target=node_j.index,
                edge_type=edge_type,
                weight=1.0
            )
    
    elif edge_type == EdgeType.SAME_PACKAGE:
        pkg_i = node_i.attributes.get('main_package', '')
        pkg_j = node_j.attributes.get('main_package', '')
        
        if pkg_i and pkg_j and pkg_i == pkg_j:
            return Edge(
                source=node_i.index,
                target=node_j.index,
                edge_type=edge_type,
                weight=1.0
            )
    
    elif edge_type == EdgeType.COMPETITIVE:
        # Competitive if same country and therapeutic area
        if (node_i.country == node_j.country and 
            node_i.attributes.get('ther_area') == node_j.attributes.get('ther_area')):
            return Edge(
                source=node_i.index,
                target=node_j.index,
                edge_type=edge_type,
                weight=1.0
            )
    
    elif edge_type == EdgeType.SIMILAR_EROSION:
        # Based on feature similarity
        if node_i.features is not None and node_j.features is not None:
            similarity = np.dot(node_i.features, node_j.features) / (
                np.linalg.norm(node_i.features) * np.linalg.norm(node_j.features) + 1e-6
            )
            if similarity > erosion_threshold:
                return Edge(
                    source=node_i.index,
                    target=node_j.index,
                    edge_type=edge_type,
                    weight=float(similarity)
                )
    
    return None


def build_adjacency_matrix(
    graph: DrugGraph,
    normalized: bool = True,
    add_self_loops: bool = False,
) -> sparse.csr_matrix:
    """
    Build adjacency matrix from graph.
    
    Args:
        graph: DrugGraph object.
        normalized: Whether to apply symmetric normalization (D^{-1/2} A D^{-1/2}).
        add_self_loops: Whether to add self-loops (I + A).
        
    Returns:
        Sparse CSR adjacency matrix.
    """
    n = graph.num_nodes
    
    if n == 0:
        return sparse.csr_matrix((0, 0))
    
    # Build COO format data
    rows = []
    cols = []
    data = []
    
    for edge in graph.edges:
        rows.append(edge.source)
        cols.append(edge.target)
        data.append(edge.weight)
        
        # Add reverse edge for undirected graph
        if edge.source != edge.target:
            rows.append(edge.target)
            cols.append(edge.source)
            data.append(edge.weight)
    
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Add self-loops
    if add_self_loops:
        A = A + sparse.eye(n)
    
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    if normalized:
        # Compute degree
        d = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sparse.diags(d_inv_sqrt)
        
        A = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A


# =============================================================================
# Node Feature Builder
# =============================================================================

class NodeFeatureBuilder:
    """
    Builder class for constructing node features from panel data.
    
    Supports multiple feature sources:
    - Static features (drug characteristics)
    - Aggregated time series features
    - Graph-derived features (centrality, etc.)
    """
    
    def __init__(
        self,
        static_cols: Optional[List[str]] = None,
        aggregation_cols: Optional[List[str]] = None,
        include_centrality: bool = False,
    ):
        """
        Initialize feature builder.
        
        Args:
            static_cols: Static feature columns (from medicine_info).
            aggregation_cols: Columns to aggregate over time (mean, std).
            include_centrality: Whether to include graph centrality features.
        """
        self.static_cols = static_cols or [
            'hospital_rate', 'biological', 'small_molecule'
        ]
        self.aggregation_cols = aggregation_cols or [
            'avg_vol_12m', 'pre_entry_trend', 'pre_entry_volatility'
        ]
        self.include_centrality = include_centrality
        
        self._fitted = False
        self._feature_names: List[str] = []
        self._scalers: Dict[str, Tuple[float, float]] = {}  # col -> (mean, std)
    
    def fit(self, panel_df: pd.DataFrame) -> 'NodeFeatureBuilder':
        """
        Fit the feature builder on training data.
        
        Computes normalization statistics for continuous features.
        
        Args:
            panel_df: Panel DataFrame.
            
        Returns:
            self for chaining.
        """
        self._feature_names = []
        
        # Static features
        for col in self.static_cols:
            if col in panel_df.columns:
                self._feature_names.append(col)
                
                # Compute normalization stats for numeric columns
                if panel_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    mean = panel_df[col].mean()
                    std = panel_df[col].std() + 1e-6
                    self._scalers[col] = (mean, std)
        
        # Aggregation features
        for col in self.aggregation_cols:
            if col in panel_df.columns:
                self._feature_names.append(f'{col}_mean')
                self._feature_names.append(f'{col}_std')
                
                # Normalization for mean
                mean_vals = panel_df.groupby(['country', 'brand_name'])[col].mean()
                self._scalers[f'{col}_mean'] = (mean_vals.mean(), mean_vals.std() + 1e-6)
                
                # Normalization for std
                std_vals = panel_df.groupby(['country', 'brand_name'])[col].std().fillna(0)
                self._scalers[f'{col}_std'] = (std_vals.mean(), std_vals.std() + 1e-6)
        
        if self.include_centrality:
            self._feature_names.extend(['degree_centrality', 'betweenness_centrality'])
        
        self._fitted = True
        logger.info(f"NodeFeatureBuilder fitted: {len(self._feature_names)} features")
        
        return self
    
    def transform(
        self,
        panel_df: pd.DataFrame,
        graph: Optional[DrugGraph] = None,
    ) -> np.ndarray:
        """
        Transform panel data to node feature matrix.
        
        Args:
            panel_df: Panel DataFrame.
            graph: Optional graph for centrality features.
            
        Returns:
            Node feature matrix (num_nodes, feature_dim).
        """
        if not self._fitted:
            raise ValueError("NodeFeatureBuilder not fitted. Call fit() first.")
        
        # Get unique series
        series_df = panel_df.groupby(['country', 'brand_name']).first().reset_index()
        n_nodes = len(series_df)
        n_features = len(self._feature_names)
        
        features = np.zeros((n_nodes, n_features))
        
        for i, (_, row) in enumerate(series_df.iterrows()):
            country = row['country']
            brand = row['brand_name']
            
            feat_idx = 0
            
            # Static features
            for col in self.static_cols:
                if col in self._feature_names:
                    val = row.get(col, 0)
                    if pd.isna(val):
                        val = 0
                    elif isinstance(val, (str, bool)):
                        val = 1 if val else 0
                    else:
                        val = float(val)
                        if col in self._scalers:
                            mean, std = self._scalers[col]
                            val = (val - mean) / std
                    
                    features[i, feat_idx] = val
                    feat_idx += 1
            
            # Aggregation features
            series_data = panel_df[
                (panel_df['country'] == country) & 
                (panel_df['brand_name'] == brand)
            ]
            
            for col in self.aggregation_cols:
                if col in panel_df.columns:
                    mean_val = series_data[col].mean()
                    std_val = series_data[col].std()
                    
                    if pd.isna(mean_val):
                        mean_val = 0
                    if pd.isna(std_val):
                        std_val = 0
                    
                    # Normalize
                    if f'{col}_mean' in self._scalers:
                        m, s = self._scalers[f'{col}_mean']
                        mean_val = (mean_val - m) / s
                    if f'{col}_std' in self._scalers:
                        m, s = self._scalers[f'{col}_std']
                        std_val = (std_val - m) / s
                    
                    features[i, feat_idx] = mean_val
                    feat_idx += 1
                    features[i, feat_idx] = std_val
                    feat_idx += 1
        
        # Centrality features
        if self.include_centrality and graph is not None:
            centrality = compute_graph_centrality(graph)
            
            for i, node in enumerate(graph.nodes):
                node_key = (node.country, node.brand_name)
                if node_key in centrality:
                    features[i, -2] = centrality[node_key].get('degree', 0)
                    features[i, -1] = centrality[node_key].get('betweenness', 0)
        
        return features
    
    def fit_transform(
        self,
        panel_df: pd.DataFrame,
        graph: Optional[DrugGraph] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(panel_df)
        return self.transform(panel_df, graph)
    
    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self._feature_names.copy()


# =============================================================================
# Graph Analysis Functions
# =============================================================================

def compute_graph_centrality(graph: DrugGraph) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute centrality metrics for all nodes.
    
    Args:
        graph: DrugGraph object.
        
    Returns:
        Dict mapping (country, brand) to centrality dict.
    """
    results = {}
    n = graph.num_nodes
    
    if n == 0:
        return results
    
    # Degree centrality
    degree = np.zeros(n)
    for edge in graph.edges:
        if edge.source != edge.target:
            degree[edge.source] += 1
            degree[edge.target] += 1
    
    # Normalize
    max_degree = degree.max() if degree.max() > 0 else 1
    degree_normalized = degree / max_degree
    
    # Simple betweenness approximation (for efficiency)
    # Full betweenness is O(n^3), this is a simpler degree-based proxy
    betweenness = degree / (n - 1) if n > 1 else degree
    
    for node in graph.nodes:
        results[(node.country, node.brand_name)] = {
            'degree': float(degree_normalized[node.index]),
            'betweenness': float(betweenness[node.index]),
        }
    
    return results


def get_subgraph(
    graph: DrugGraph,
    node_indices: List[int],
) -> DrugGraph:
    """
    Extract a subgraph containing only specified nodes.
    
    Args:
        graph: Original DrugGraph.
        node_indices: List of node indices to include.
        
    Returns:
        New DrugGraph with only specified nodes and their internal edges.
    """
    node_set = set(node_indices)
    
    # Create new index mapping
    new_index = {old: new for new, old in enumerate(node_indices)}
    
    subgraph = DrugGraph()
    
    # Add nodes
    for old_idx in node_indices:
        old_node = graph.nodes[old_idx]
        new_node = Node(
            index=new_index[old_idx],
            country=old_node.country,
            brand_name=old_node.brand_name,
            features=old_node.features.copy() if old_node.features is not None else None,
            attributes=old_node.attributes.copy()
        )
        subgraph.nodes.append(new_node)
        subgraph.node_index[(new_node.country, new_node.brand_name)] = new_node.index
    
    # Add edges (only internal)
    for edge in graph.edges:
        if edge.source in node_set and edge.target in node_set:
            new_edge = Edge(
                source=new_index[edge.source],
                target=new_index[edge.target],
                edge_type=edge.edge_type,
                weight=edge.weight,
                attributes=edge.attributes.copy()
            )
            subgraph.edges.append(new_edge)
    
    # Rebuild matrices
    if subgraph.nodes:
        subgraph.node_features = np.array([n.features for n in subgraph.nodes])
    
    if subgraph.edges:
        sources = [e.source for e in subgraph.edges]
        targets = [e.target for e in subgraph.edges]
        weights = [e.weight for e in subgraph.edges]
        
        subgraph.edge_index = np.array([sources, targets])
        subgraph.edge_weight = np.array(weights)
        subgraph.adjacency_matrix = build_adjacency_matrix(subgraph, normalized=True)
    
    return subgraph


def filter_graph_by_country(graph: DrugGraph, country: str) -> DrugGraph:
    """
    Get subgraph containing only nodes from a specific country.
    
    Args:
        graph: Original DrugGraph.
        country: Country code to filter by.
        
    Returns:
        Filtered subgraph.
    """
    node_indices = [
        node.index for node in graph.nodes 
        if node.country.upper() == country.upper()
    ]
    return get_subgraph(graph, node_indices)


def filter_graph_by_therapeutic_area(graph: DrugGraph, ther_area: str) -> DrugGraph:
    """
    Get subgraph containing only nodes from a specific therapeutic area.
    
    Args:
        graph: Original DrugGraph.
        ther_area: Therapeutic area to filter by.
        
    Returns:
        Filtered subgraph.
    """
    node_indices = [
        node.index for node in graph.nodes 
        if node.attributes.get('ther_area', '').lower() == ther_area.lower()
    ]
    return get_subgraph(graph, node_indices)
