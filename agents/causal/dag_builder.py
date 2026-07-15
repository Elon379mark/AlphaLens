import networkx as nx
from typing import List


def build_domain_dag(signals: List[str], outcome: str = "fwd_return") -> nx.DiGraph:
    """
    Domain-knowledge DAG: signals -> outcome; market_regime -> signals.
    This encodes a simple prior structure (not learned from data) as a
    starting point before PC-algorithm discovery is layered on top.
    """
    G = nx.DiGraph()
    G.add_nodes_from(signals + [outcome])

    for s in signals:
        G.add_edge(s, outcome)

    G.add_node("market_regime")
    for s in signals:
        G.add_edge("market_regime", s)

    return G


if __name__ == "__main__":
    import json

    with open("outputs/ranked_signals.json") as f:
        ranked = json.load(f)
    top_signals = ranked[:10]

    print("Building domain DAG...")
    dag = build_domain_dag(top_signals)

    print(f"\nNodes: {dag.number_of_nodes()}")
    print(f"Edges: {dag.number_of_edges()}")
    print(f"Is DAG (acyclic): {nx.is_directed_acyclic_graph(dag)}")

    print(f"\nEdge list:")
    for u, v in dag.edges():
        print(f"  {u} -> {v}")

    assert nx.is_directed_acyclic_graph(dag), "Graph contains cycles — not a valid DAG"
    assert dag.number_of_nodes() == len(top_signals) + 2, "Unexpected node count"  # signals + outcome + market_regime

    print("\nPASS: domain DAG constructed and verified acyclic")