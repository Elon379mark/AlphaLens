"""
AlphaLens — Signal Generation Agent
Feature factory: 312 alpha signals, IC/ICIR computation, validation, ranking.
"""

from .data_loader import load_ohlcv, load_fundamentals
from .ic_calculator import compute_ic, compute_icir, compute_all_ic_icir, compute_forward_returns
from .validator import validate_features, check_feature_correlation
from .ranker import rank_signals, get_top_signals
from .node import signal_agent_node

__all__ = [
    "load_ohlcv",
    "load_fundamentals",
    "compute_ic",
    "compute_icir",
    "compute_all_ic_icir",
    "compute_forward_returns",
    "validate_features",
    "check_feature_correlation",
    "rank_signals",
    "get_top_signals",
    "signal_agent_node",
]
