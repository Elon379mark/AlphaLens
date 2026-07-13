"""
AlphaLens — Graph Attention Network (GAT) for Cross-Asset Modeling
Models inter-asset relationships (correlation, supply chain, sector
co-movement) using multi-head attention over a financial asset graph.

Reference: Veličković et al., "Graph Attention Networks" (2018)
"""

import math
import logging
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GAT will use NumPy fallback.")


# ---------------------------------------------------------------------------
# PyTorch Components
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class GATLayer(nn.Module):
        """
        Single Graph Attention layer with multi-head attention.
        """
        def __init__(self, in_features: int, out_features: int,
                     n_heads: int = 4, dropout: float = 0.1,
                     concat: bool = True):
            super().__init__()
            self.n_heads = n_heads
            self.out_features = out_features
            self.concat = concat

            self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
            self.a_src = nn.Parameter(torch.empty(n_heads, out_features, 1))
            self.a_dst = nn.Parameter(torch.empty(n_heads, out_features, 1))

            self.leaky_relu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(dropout)

            self._reset_parameters()

        def _reset_parameters(self):
            nn.init.xavier_uniform_(self.W)
            nn.init.xavier_uniform_(self.a_src)
            nn.init.xavier_uniform_(self.a_dst)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (n_nodes, in_features)
                adj: (n_nodes, n_nodes) adjacency matrix (1=connected, 0=no edge)
            Returns:
                output: (n_nodes, n_heads * out_features) if concat else (n_nodes, out_features)
                attention_weights: (n_heads, n_nodes, n_nodes)
            """
            n_nodes = x.shape[0]

            # Linear transformation for each head
            # x: (n_nodes, in_features) -> h: (n_heads, n_nodes, out_features)
            h = torch.einsum("ni,hio->hno", x, self.W)

            # Attention coefficients
            # e_src: (n_heads, n_nodes, 1)
            e_src = torch.matmul(h, self.a_src)
            e_dst = torch.matmul(h, self.a_dst)

            # e: (n_heads, n_nodes, n_nodes)
            e = e_src + e_dst.transpose(-2, -1)
            e = self.leaky_relu(e)

            # Mask with adjacency
            mask = adj.unsqueeze(0).expand(self.n_heads, -1, -1)
            e = e.masked_fill(mask == 0, -1e9)

            # Softmax attention
            attn = F.softmax(e, dim=-1)
            attn = self.dropout(attn)

            # Weighted aggregation
            out = torch.matmul(attn, h)  # (n_heads, n_nodes, out_features)

            if self.concat:
                out = out.permute(1, 0, 2).reshape(n_nodes, -1)
            else:
                out = out.mean(dim=0)

            return out, attn


    class GATModel(nn.Module):
        """
        Multi-layer GAT for financial asset graph modeling.
        """
        def __init__(self, n_features: int, hidden_dim: int = 32,
                     output_dim: int = 16, n_heads: int = 4,
                     n_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.layers = nn.ModuleList()
            self.n_layers = n_layers

            # First layer
            self.layers.append(
                GATLayer(n_features, hidden_dim, n_heads, dropout, concat=True)
            )

            # Middle layers
            for _ in range(n_layers - 2):
                self.layers.append(
                    GATLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout, concat=True)
                )

            # Final layer (no concat, average heads)
            if n_layers > 1:
                self.layers.append(
                    GATLayer(hidden_dim * n_heads, output_dim, n_heads, dropout, concat=False)
                )

            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (n_nodes, n_features) node feature matrix
                adj: (n_nodes, n_nodes) adjacency matrix
            Returns:
                embeddings: (n_nodes, output_dim)
                attention: (n_heads, n_nodes, n_nodes) from last layer
            """
            h = x
            attn = None
            for i, layer in enumerate(self.layers):
                h, attn = layer(h, adj)
                if i < self.n_layers - 1:
                    h = F.elu(h)
                    h = self.dropout(h)

            return h, attn


# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------

class CrossAssetGAT:
    """
    High-level API for cross-asset relationship modeling via GAT.
    Builds a correlation-based graph from price data and learns
    node embeddings that capture inter-asset dependencies.
    """
    def __init__(self, hidden_dim: int = 32, output_dim: int = 16,
                 n_heads: int = 4, n_layers: int = 2,
                 correlation_threshold: float = 0.3,
                 learning_rate: float = 1e-3, epochs: int = 100,
                 dropout: float = 0.1):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.correlation_threshold = correlation_threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.is_fitted = False
        self._embeddings = None
        self._attention_weights = None
        self._ticker_names = None
        self._adj_matrix = None

    def build_correlation_graph(self, returns_matrix: np.ndarray,
                                ticker_names: List[str]) -> Tuple[np.ndarray, List[Tuple[str, str, float]]]:
        """
        Builds an adjacency matrix from return correlations.
        
        Args:
            returns_matrix: (n_days, n_tickers) matrix of daily returns
            ticker_names: list of ticker symbols
        
        Returns:
            adj_matrix: (n_tickers, n_tickers) binary adjacency
            edges: list of (src, dst, correlation) tuples
        """
        n_tickers = returns_matrix.shape[1]
        corr = np.corrcoef(returns_matrix.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # Threshold to binary adjacency (including self-loops)
        adj = (np.abs(corr) >= self.correlation_threshold).astype(np.float32)
        np.fill_diagonal(adj, 1.0)

        edges = []
        for i in range(n_tickers):
            for j in range(i + 1, n_tickers):
                if adj[i, j] > 0:
                    edges.append((ticker_names[i], ticker_names[j], float(corr[i, j])))

        self._ticker_names = ticker_names
        self._adj_matrix = adj
        return adj, edges

    def fit(self, returns_matrix: np.ndarray, ticker_names: List[str],
            node_features: Optional[np.ndarray] = None) -> "CrossAssetGAT":
        """
        Trains the GAT model on the asset graph.
        
        Args:
            returns_matrix: (n_days, n_tickers) daily returns
            ticker_names: list of ticker symbols
            node_features: optional (n_tickers, n_features) pre-computed features.
                           If None, uses rolling statistics as features.
        """
        self._ticker_names = ticker_names
        n_tickers = returns_matrix.shape[1]

        # Build graph
        adj, edges = self.build_correlation_graph(returns_matrix, ticker_names)

        # Build node features if not provided
        if node_features is None:
            node_features = self._compute_node_features(returns_matrix)

        if not TORCH_AVAILABLE:
            logger.info("GAT: PyTorch unavailable. Using correlation-based embeddings.")
            self._fit_fallback(returns_matrix, node_features, adj)
            return self

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_features = node_features.shape[1]

        self.model = GATModel(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X = torch.FloatTensor(node_features).to(device)
        A = torch.FloatTensor(adj).to(device)

        # Self-supervised training: reconstruct adjacency from embeddings
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            embeddings, attn = self.model(X, A)

            # Reconstruction loss: embeddings should predict adjacency
            sim = torch.mm(embeddings, embeddings.t())
            sim = torch.sigmoid(sim)
            target = torch.FloatTensor(adj).to(device)
            loss = F.binary_cross_entropy(sim, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"GAT Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.6f}")

        # Extract final embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings, attn = self.model(X, A)
            self._embeddings = embeddings.cpu().numpy()
            self._attention_weights = attn.cpu().numpy()

        self.is_fitted = True
        logger.info("GAT training complete.")
        return self

    def get_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """Returns node embeddings keyed by ticker name."""
        if self._embeddings is None or self._ticker_names is None:
            return None
        return {name: self._embeddings[i] for i, name in enumerate(self._ticker_names)}

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Returns attention weight matrix from last GAT layer."""
        return self._attention_weights

    def get_graph_edges(self) -> List[Tuple[str, str, float]]:
        """Returns the graph edges with correlation weights."""
        if self._adj_matrix is None or self._ticker_names is None:
            return []
        edges = []
        n = len(self._ticker_names)
        for i in range(n):
            for j in range(i + 1, n):
                if self._adj_matrix[i, j] > 0:
                    edges.append((self._ticker_names[i], self._ticker_names[j],
                                  float(self._adj_matrix[i, j])))
        return edges

    # --- Internal helpers ---

    def _compute_node_features(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Computes per-ticker statistical features from returns."""
        n_tickers = returns_matrix.shape[1]
        features = []
        for i in range(n_tickers):
            ret = returns_matrix[:, i]
            features.append([
                np.mean(ret),                          # Mean return
                np.std(ret),                           # Volatility
                np.mean(ret) / max(np.std(ret), 1e-8), # Sharpe proxy
                np.percentile(ret, 5),                 # VaR 5%
                np.percentile(ret, 95),                # Upside 95%
                float(np.corrcoef(ret[:-1], ret[1:])[0, 1]) if len(ret) > 1 else 0.0,  # Autocorrelation
                float((ret > 0).mean()),               # Win rate
                float(np.max(np.maximum.accumulate(np.cumsum(ret)) - np.cumsum(ret))),  # Max drawdown proxy
            ])
        return np.array(features, dtype=np.float32)

    def _fit_fallback(self, returns_matrix: np.ndarray, node_features: np.ndarray,
                      adj: np.ndarray):
        """Fallback using PCA on correlation-weighted features."""
        # Simple SVD-based embedding
        try:
            weighted = adj @ node_features
            U, S, Vt = np.linalg.svd(weighted, full_matrices=False)
            k = min(self.output_dim, U.shape[1])
            self._embeddings = U[:, :k] * S[:k]
        except np.linalg.LinAlgError:
            self._embeddings = node_features[:, :self.output_dim]
        self.is_fitted = True
