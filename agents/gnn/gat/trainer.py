from pathlib import Path

import torch
from torch.optim import Adam
from torch_geometric.utils import negative_sampling

import mlflow

CHECKPOINT_PATH = "models/gat/best_model.pt"


def train_gat_unsupervised(model, graph_data, epochs: int = 200) -> tuple:
    """
    Train GAT with link prediction loss (positive/negative edge sampling).
    Returns (trained_model, list_of_losses) for verification.
    """
    Path("models/gat").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    graph_data = graph_data.to(device)

    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    best_loss = float("inf")
    loss_history = []

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("alphalens_gat")

    with mlflow.start_run(run_name="GAT_training"):
        mlflow.log_param("epochs", epochs)

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            z = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

            pos_edge = graph_data.edge_index
            neg_edge = negative_sampling(pos_edge, graph_data.num_nodes, num_neg_samples=pos_edge.size(1))

            pos_score = (z[pos_edge[0]] * z[pos_edge[1]]).sum(dim=-1)
            neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=-1)

            loss = (
                -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
                - torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean()
            )

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            mlflow.log_metric("loss", loss.item(), step=epoch)

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), CHECKPOINT_PATH)

            if epoch % 20 == 0:
                print(f"[GAT] Epoch {epoch}: loss={loss.item():.4f}")

        mlflow.log_metric("best_loss", best_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)

    return model, loss_history


if __name__ == "__main__":
    from agents.gnn.gat.graph_builder import build_correlation_graph, add_sector_edges, assign_synthetic_sector
    from agents.gnn.gat.model import StockGAT
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

    print("\nBuilding model...")
    model = StockGAT(in_channels=graph.x.shape[1])

    print("\nTraining GAT with link prediction (200 epochs)...\n")
    model, loss_history = train_gat_unsupervised(model, graph, epochs=200)

    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"First loss: {loss_history[0]:.4f}")
    print(f"Last loss: {loss_history[-1]:.4f}")
    print(f"Min loss: {min(loss_history):.4f}")
    print(f"Checkpoint saved: {CHECKPOINT_PATH}")

    # Verify actual learning happened, not a frozen/flat loss (lesson from PatchTST)
    loss_std = torch.tensor(loss_history).std().item()
    print(f"Loss std across training: {loss_std:.6f}")

    assert Path(CHECKPOINT_PATH).exists(), "No checkpoint saved"
    assert loss_history[-1] < loss_history[0], "Loss did not decrease — check gradient flow"
    print("\nPASS: GAT training completed with verified loss decrease")