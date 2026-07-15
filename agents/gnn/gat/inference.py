import json
from pathlib import Path

import numpy as np
import torch

from agents.gnn.gat.model import StockGAT

CHECKPOINT_PATH = "models/gat/best_model.pt"


def load_best_gat(checkpoint_path: str = CHECKPOINT_PATH, in_channels: int = 2) -> StockGAT:
    model = StockGAT(in_channels=in_channels)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def extract_embeddings(model: StockGAT, graph_data) -> dict:
    """
    Run inference to get final node embeddings.
    Returns {ticker: embedding_list}.
    """
    with torch.no_grad():
        embeddings = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

    return {ticker: embeddings[i].tolist() for i, ticker in enumerate(graph_data.tickers)}


def save_embeddings(embeddings_dict: dict, path: str = "outputs/gat_embeddings.npy") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tickers = list(embeddings_dict.keys())
    matrix = np.array([embeddings_dict[t] for t in tickers])
    np.save(path, matrix)
    # Also save ticker order alongside, since .npy alone loses the mapping
    with open(path.replace(".npy", "_tickers.json"), "w") as f:
        json.dump(tickers, f, indent=2)
    print(f"Saved embeddings matrix {matrix.shape} to {path}")
    print(f"Saved ticker order to {path.replace('.npy', '_tickers.json')}")


def save_graph_edges(graph_data, path: str = "outputs/graph_edges.json") -> None:
    edges = []
    edge_index = graph_data.edge_index.tolist()
    edge_attr = graph_data.edge_attr.tolist() if graph_data.edge_attr.numel() > 0 else []
    for k in range(len(edge_index[0])):
        src, dst = edge_index[0][k], edge_index[1][k]
        weight = edge_attr[k] if k < len(edge_attr) else None
        edges.append({
            "source": graph_data.tickers[src],
            "target": graph_data.tickers[dst],
            "weight": weight,
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(edges, f, indent=2)
    print(f"Saved {len(edges)} edges to {path}")


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

    print("Loading trained model...")
    model = load_best_gat(in_channels=graph.x.shape[1])

    print("\nExtracting embeddings...")
    embeddings_dict = extract_embeddings(model, graph)

    print(f"\nEmbeddings extracted for {len(embeddings_dict)} tickers")
    sample_ticker = list(embeddings_dict.keys())[0]
    print(f"Sample — {sample_ticker}: {embeddings_dict[sample_ticker]}")

    save_embeddings(embeddings_dict)
    save_graph_edges(graph)

    assert len(embeddings_dict) == 20, f"Expected 20 ticker embeddings, got {len(embeddings_dict)}"
    assert len(embeddings_dict[sample_ticker]) == 8, f"Expected 8-dim embeddings, got {len(embeddings_dict[sample_ticker])}"
    print("\nPASS: GAT inference completed, embeddings and edges saved")