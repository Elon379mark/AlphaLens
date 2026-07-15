import numpy as np
import pandas as pd


def run_pc_discovery(data: pd.DataFrame, alpha: float = 0.05):
    """
    Run PC constraint-based causal discovery on numeric signal + outcome data.
    Returns the causal-learn CausalGraph object.
    """
    from causallearn.search.ConstraintBased.PC import pc

    data_array = data.values.astype(np.float64)
    cg = pc(data_array, alpha=alpha, indep_test="fisherz", stable=True, uc_rule=0, uc_priority=2)
    return cg


if __name__ == "__main__":
    import json
    from agents.causal.data_prep import build_causal_dataset

    print("Loading data...")
    features = pd.read_parquet("data/processed/features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    features = features.set_index(["date", "ticker"]).sort_index()

    prices = pd.read_parquet("data/processed/sample_prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index(["date", "ticker"]).sort_index()
    close = prices["adj_close"].unstack("ticker")
    fwd_returns = (close.shift(-21) / close - 1).stack().rename("fwd_return")

    from agents.causal.data_prep import select_uncorrelated_top_signals

    with open("outputs/ranked_signals.json") as f:
        ranked = json.load(f)
    print("Selecting top 10 signals, skipping near-duplicates (|corr| > 0.95)...")
    top_signals = select_uncorrelated_top_signals(ranked, features, top_n=10, max_corr=0.95)
    print(f"Final selected signals: {top_signals}")
    causal_df = build_causal_dataset(features, fwd_returns, top_signals)

    # PC algorithm runs on raw signal values + outcome, not the binarized treatments
    pc_input = causal_df[top_signals + ["fwd_return"]]
    print(f"PC input shape: {pc_input.shape}")

    print("\nRunning PC algorithm (this may take a minute or two)...")
    cg = run_pc_discovery(pc_input)

    print("\nPC algorithm completed.")
    print(f"Graph object type: {type(cg)}")

    # Extract edges for inspection
    col_names = list(pc_input.columns)
    print(f"\nDiscovered edges (node index -> node index, using column order {col_names}):")
    graph_matrix = cg.G.graph
    n = len(col_names)
    edge_count = 0
    for i in range(n):
        for j in range(n):
            if graph_matrix[i, j] != 0:
                edge_count += 1
    print(f"Non-zero graph matrix entries: {edge_count}")
    print(f"Graph matrix shape: {graph_matrix.shape}")

    assert graph_matrix.shape == (n, n), f"Unexpected graph matrix shape"
    print("\nPASS: PC algorithm discovery completed")