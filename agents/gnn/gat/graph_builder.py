import hashlib
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def assign_synthetic_sector(ticker: str) -> str:
    """
    Same deterministic synthetic sector assignment used in the TFT chapter
    (Chapter 4) — kept consistent so sector edges here align conceptually
    with the sector categorical used there.
    """
    sectors = ["Tech", "Healthcare", "Financials", "Energy", "Consumer", "Industrials"]
    hash_val = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
    return sectors[hash_val % len(sectors)]


def build_correlation_graph(
    returns: pd.DataFrame,
    corr_threshold: float = 0.1,  # lower than manual's 0.3 — our synthetic tickers are independent random walks, so correlations are naturally weak; a high threshold would produce zero edges
    window: int = 60,
) -> Data:
    """
    Build stock correlation graph.
    returns: DataFrame (date x ticker) of daily returns.
    Each ticker = node; edge if |corr| > threshold over the trailing window.
    Node features: [mean_return, volatility].
    """
    recent = returns.tail(window)
    corr_matrix = recent.corr().values
    tickers = returns.columns.tolist()
    n = len(tickers)

    mean_ret = recent.mean().values
    vol = recent.std().values
    node_features = np.stack([mean_ret, vol], axis=1)
    x = torch.tensor(node_features, dtype=torch.float)

    edge_index_list = []
    edge_attr_list = []
    for i in range(n):
        for j in range(i + 1, n):
            c = corr_matrix[i, j]
            if not np.isnan(c) and abs(c) >= corr_threshold:
                edge_index_list.extend([[i, j], [j, i]])
                edge_attr_list.extend([c, c])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        # Empty graph fallback: valid shape, zero edges — downstream code
        # must handle this gracefully rather than crash
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.tickers = tickers  # attach for later reference (not a standard PyG field, but convenient)
    return data


def add_sector_edges(graph: Data, ticker_sector_map: Dict[str, str]) -> Data:
    """Add edges between stocks in the same synthetic sector."""
    tickers = graph.tickers
    n = len(tickers)
    existing = set(map(tuple, graph.edge_index.t().tolist())) if graph.edge_index.numel() > 0 else set()
    new_edges, new_attrs = [], []

    for i in range(n):
        for j in range(i + 1, n):
            if ticker_sector_map.get(tickers[i]) == ticker_sector_map.get(tickers[j]):
                if (i, j) not in existing:
                    new_edges.extend([[i, j], [j, i]])
                    new_attrs.extend([1.0, 1.0])

    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
        new_edge_attr = torch.tensor(new_attrs, dtype=torch.float)
        graph.edge_index = torch.cat([graph.edge_index, new_edge_index], dim=1)
        graph.edge_attr = torch.cat([graph.edge_attr, new_edge_attr])

    return graph


if __name__ == "__main__":
    print("Loading sample prices and computing returns...")
    prices = pd.read_parquet("data/processed/sample_prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["daily_return"] = prices.groupby("ticker")["adj_close"].pct_change()

    returns_wide = prices.pivot(index="date", columns="ticker", values="daily_return").dropna()
    print(f"Returns matrix shape: {returns_wide.shape}")

    print("\nBuilding correlation graph...")
    graph = build_correlation_graph(returns_wide)

    print(f"Nodes: {graph.x.shape[0]}")
    print(f"Node feature dim: {graph.x.shape[1]}")
    print(f"Correlation edges: {graph.edge_index.shape[1]}")

    print("\nAssigning synthetic sectors and adding sector edges...")
    ticker_sector_map = {t: assign_synthetic_sector(t) for t in graph.tickers}
    print(f"Sector assignment sample: {dict(list(ticker_sector_map.items())[:5])}")

    graph = add_sector_edges(graph, ticker_sector_map)
    print(f"Total edges after adding sector edges: {graph.edge_index.shape[1]}")

    print(f"\nEdge index shape: {graph.edge_index.shape}")
    print(f"Edge attr shape: {graph.edge_attr.shape}")

    assert graph.x.shape[0] == 20, f"Expected 20 nodes, got {graph.x.shape[0]}"
    assert graph.edge_index.shape[0] == 2, "Edge index must have shape (2, num_edges)"
    assert graph.edge_index.shape[1] == graph.edge_attr.shape[0], "Edge index and edge attr count mismatch"

    print("\nPASS: correlation + sector graph built successfully")