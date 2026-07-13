"""
AlphaLens — GNN Agent Node
LangGraph node wrapping the Graph Attention Network for
cross-asset relationship modeling within the pipeline.
"""

import logging
from typing import Dict, Any, List

import numpy as np

from alphalens.agents.gnn.gat import CrossAssetGAT
from alphalens.agents.memory import AgentMemoryEngine
from alphalens.core.utils import run_sync

logger = logging.getLogger(__name__)
_memory_engine = AgentMemoryEngine()


def gnn_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: builds a cross-asset correlation graph and trains
    a GAT model to learn node embeddings.
    
    Reads from state:
        - signal_values, returns_values, close_prices
    
    Writes to state:
        - gat_embeddings, graph_edges
    """
    logger.info("[GNN Agent] Building cross-asset graph...")
    run_id = state.get("run_id", "default_run_id")

    returns_values = np.array(state.get("returns_values", []))
    close_prices = np.array(state.get("close_prices", []))

    if len(returns_values) < 30:
        logger.warning("[GNN Agent] Insufficient data for GAT.")
        return {
            "current_node": "gnn_agent",
            "gat_embeddings": {},
            "graph_edges": [],
            "agent_logs": state.get("agent_logs", []) + [
                "🕸️ GNN Agent: Insufficient data, skipping."
            ],
        }

    # Generate synthetic multi-asset returns for graph construction
    # In production, this would come from a multi-ticker data source
    rng = np.random.default_rng(42)
    n_days = len(returns_values)
    n_assets = 8
    ticker_names = ["MOMENTUM", "VALUE", "QUALITY", "SIZE", "VOL", "GROWTH", "YIELD", "BETA"]

    # Create correlated factor returns based on the primary signal
    base_returns = returns_values.copy()
    factor_returns = np.zeros((n_days, n_assets))
    factor_returns[:, 0] = base_returns  # Momentum
    factor_returns[:, 1] = -base_returns * 0.6 + rng.normal(0, 0.01, n_days)  # Value (anti-correlated)
    factor_returns[:, 2] = base_returns * 0.3 + rng.normal(0, 0.008, n_days)  # Quality
    factor_returns[:, 3] = rng.normal(0, 0.015, n_days)  # Size (independent)
    factor_returns[:, 4] = np.abs(base_returns) * 0.5 + rng.normal(0, 0.01, n_days)  # Vol
    factor_returns[:, 5] = base_returns * 0.7 + rng.normal(0, 0.008, n_days)  # Growth
    factor_returns[:, 6] = -base_returns * 0.2 + rng.normal(0, 0.005, n_days)  # Yield
    factor_returns[:, 7] = base_returns * 0.9 + rng.normal(0, 0.012, n_days)  # Beta

    # Train GAT
    try:
        gat = CrossAssetGAT(
            hidden_dim=16, output_dim=8, n_heads=2, n_layers=2,
            correlation_threshold=0.2, epochs=50
        )
        gat.fit(factor_returns, ticker_names)

        embeddings = gat.get_embeddings()
        edges = gat.get_graph_edges()
        attn_weights = gat.get_attention_weights()

        # Serialize embeddings
        embeddings_serialized = {}
        if embeddings:
            for name, emb in embeddings.items():
                embeddings_serialized[name] = emb.tolist()

        # Serialize edges
        edges_serialized = []
        for src, dst, weight in edges:
            edges_serialized.append({"source": src, "target": dst, "weight": float(weight)})

        logger.info(f"[GNN Agent] GAT trained: {len(ticker_names)} nodes, {len(edges)} edges")

        log_msg = f"GAT complete: {len(ticker_names)} nodes, {len(edges)} edges, embedding_dim={gat.output_dim}"
        run_sync(_memory_engine.add_episode_log(run_id, "gnn_agent", "INFO", log_msg))

        return {
            "gat_embeddings": embeddings_serialized,
            "graph_edges": edges_serialized,
            "current_node": "gnn_agent",
            "agent_logs": state.get("agent_logs", []) + [
                f"🕸️ GNN Agent: GAT trained on {len(ticker_names)} factor nodes, "
                f"{len(edges)} edges discovered",
                f"📊 Factor relationships: {', '.join(f'{e[0]}-{e[1]}' for e in edges[:5])}{'...' if len(edges) > 5 else ''}",
            ],
        }

    except Exception as e:
        logger.error(f"[GNN Agent] GAT training failed: {e}")
        return {
            "gat_embeddings": {},
            "graph_edges": [],
            "current_node": "gnn_agent",
            "agent_logs": state.get("agent_logs", []) + [
                f"🕸️ GNN Agent: Failed — {str(e)[:100]}"
            ],
        }
