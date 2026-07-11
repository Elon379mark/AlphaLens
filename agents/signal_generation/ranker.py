from typing import Dict, List


def rank_signals(icir_dict: Dict[str, float]) -> List[str]:
    """
    Rank signal names by absolute ICIR, descending.
    Returns an ordered list of feature names, strongest first.
    """
    ranked = sorted(icir_dict.keys(), key=lambda k: abs(icir_dict[k]), reverse=True)
    return ranked


def get_top_signals(ranked: List[str], top_n: int = 50) -> List[str]:
    """Return the top-N signal names from an already-ranked list."""
    return ranked[:top_n]


if __name__ == "__main__":
    import json

    print("Loading ICIR scores...")
    with open("outputs/icir_scores.json") as f:
        icir_dict = json.load(f)

    print(f"Ranking {len(icir_dict)} signals by |ICIR|...")
    ranked = rank_signals(icir_dict)

    print(f"\nTop 10 signals overall (by |ICIR|, regardless of validation pass/fail):")
    for i, name in enumerate(ranked[:10], 1):
        print(f"  {i}. {name}: ICIR={icir_dict[name]:.4f}")

    top_50 = get_top_signals(ranked, top_n=50)
    print(f"\nTop 50 signal names extracted (length check: {len(top_50)})")

    with open("outputs/ranked_signals.json", "w") as f:
        json.dump(ranked, f, indent=2)
    print("\nSaved full ranked list to outputs/ranked_signals.json")

    assert len(ranked) == len(icir_dict), "Ranked list length mismatch"
    assert ranked[0] == max(icir_dict, key=lambda k: abs(icir_dict[k])), "Top signal doesn't match max ICIR"
    print("\nPASS: signal ranking correct and saved")