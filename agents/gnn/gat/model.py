import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class StockGAT(nn.Module):
    """
    Graph Attention Network for stock embedding.
    Input: node features (num_nodes x in_channels)
    Output: node embeddings (num_nodes x out_channels)
    """
    def __init__(self, in_channels: int = 2, hidden: int = 16, out_channels: int = 8, heads: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATv2Conv(hidden * heads, out_channels, heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        edge_attr_reshaped = edge_attr.unsqueeze(-1) if edge_attr is not None and edge_attr.numel() > 0 else None

        x = self.conv1(x, edge_index, edge_attr=edge_attr_reshaped)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr_reshaped)
        return x


if __name__ == "__main__":
    from agents.gnn.gat.graph_builder import build_correlation_graph, add_sector_edges, assign_synthetic_sector
    import pandas as pd

    print("Building graph...")
    prices = pd.read_parquet("data/processed/sample_prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["daily_return"] = prices.groupby("ticker")["adj_close"].pct_change()
    returns_wide = prices.pivot(index="date", columns="ticker", values="daily_return").dropna()

    graph = build_correlation_graph(returns_wide)
    ticker_sector_map = {t: assign_synthetic_sector(t) for t in graph.tickers}
    graph = add_sector_edges(graph, ticker_sector_map)

    print(f"Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

    print("\nBuilding GAT model...")
    model = StockGAT(in_channels=graph.x.shape[1])

    print("\nRunning forward pass (untrained, sanity check only)...")
    with torch.no_grad():
        embeddings = model(graph.x, graph.edge_index, graph.edge_attr)

    print(f"Output embeddings shape: {embeddings.shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    assert embeddings.shape == (20, 8), f"Expected (20, 8) embeddings, got {embeddings.shape}"
    print("\nPASS: GAT model built and forward pass verified")