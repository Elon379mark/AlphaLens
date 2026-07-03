"""
ranker.py
Signal Ranking — AlphaLens Signal Generation Agent

Ranks validated signals by absolute ICIR (descending) so that downstream
agents (TFT, N-BEATS, PatchTST, GAT, Causal) consume features in priority
order, and so the top-N most reliable signals can be selected for causal
inference / portfolio construction.
"""

import json
import os
from typing import Dict, List


def rank_signals(icir_dict: Dict[str, float]) -> List[str]:
    """
    Rank signals by absolute ICIR descending.

    Args:
        icir_dict: feature name -> ICIR value.

    Returns:
        Ordered list of signal names, most reliable first.
    """
    ranked = sorted(
        icir_dict.keys(),
        key=lambda k: abs(icir_dict[k]),
        reverse=True,
    )
    return ranked


def get_top_signals(ranked: List[str], top_n: int = 50) -> List[str]:
    """Return top-N signals by ICIR rank."""
    return ranked[:top_n]


def save_ranked_signals(
    ranked: List[str],
    path: str = "outputs/ranked_signals.json",
) -> None:
    """Persist the ranked signal list as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ranked, f, indent=2)


def build_ranking_report(
    ranked: List[str],
    ic_dict: Dict[str, float],
    icir_dict: Dict[str, float],
    top_n: int = 50,
) -> List[Dict]:
    """
    Build a human-readable ranking report for the top-N signals.

    Returns:
        List of dicts: {rank, signal_name, ic, icir}.
    """
    report = []
    for i, sig in enumerate(ranked[:top_n], start=1):
        report.append({
            "rank": i,
            "signal_name": sig,
            "ic": round(ic_dict.get(sig, 0.0), 5),
            "icir": round(icir_dict.get(sig, 0.0), 5),
        })
    return report
