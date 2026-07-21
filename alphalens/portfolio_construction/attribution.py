"""
Risk Attribution — Brinson-Hood-Beebower (BHB) Model (Step 10).
Decomposes portfolio excess returns relative to a benchmark into:
  1. Allocation Effect: A_i = (w_{p,i} - w_{b,i}) * (R_{b,i} - R_b)
  2. Selection Effect:  S_i = w_{b,i} * (R_{p,i} - R_{b,i})
  3. Interaction Effect: I_i = (w_{p,i} - w_{b,i}) * (R_{p,i} - R_{b,i})

Total Excess Return = Sum_i (A_i + S_i + I_i)
"""
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BrinsonAttribution:
    """
    Brinson-Hood-Beebower (BHB) performance attribution engine (Step 10).
    Decomposes strategy excess returns across sectors/factors into allocation, selection, and interaction components.
    """
    def __init__(self):
        pass

    def compute_attribution(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Calculates BHB performance attribution.

        Args:
            portfolio_weights: w_p by segment (e.g. sector/asset class)
            benchmark_weights: w_b by segment
            portfolio_returns: R_p by segment
            benchmark_returns: R_b by segment

        Returns:
            Dictionary containing:
                - per_segment: DataFrame with allocation, selection, interaction per segment
                - total_allocation: sum of allocation effects
                - total_selection: sum of selection effects
                - total_interaction: sum of interaction effects
                - total_excess_return: total portfolio return - total benchmark return
        """
        segments = sorted(list(set(portfolio_weights.keys()) | set(benchmark_weights.keys())))

        # Overall benchmark return R_b = sum(w_{b,i} * R_{b,i})
        R_b_total = sum(benchmark_weights.get(seg, 0.0) * benchmark_returns.get(seg, 0.0) for seg in segments)
        R_p_total = sum(portfolio_weights.get(seg, 0.0) * portfolio_returns.get(seg, 0.0) for seg in segments)

        rows = []
        tot_alloc, tot_select, tot_inter = 0.0, 0.0, 0.0

        for seg in segments:
            wp_i = portfolio_weights.get(seg, 0.0)
            wb_i = benchmark_weights.get(seg, 0.0)
            rp_i = portfolio_returns.get(seg, 0.0)
            rb_i = benchmark_returns.get(seg, 0.0)

            # Allocation Effect: A_i = (w_{p,i} - w_{b,i}) * (R_{b,i} - R_b)
            alloc_i = (wp_i - wb_i) * (rb_i - R_b_total)

            # Selection Effect: S_i = w_{b,i} * (R_{p,i} - R_{b,i})
            select_i = wb_i * (rp_i - rb_i)

            # Interaction Effect: I_i = (w_{p,i} - w_{b,i}) * (R_{p,i} - R_{b,i})
            inter_i = (wp_i - wb_i) * (rp_i - rb_i)

            total_i = alloc_i + select_i + inter_i

            tot_alloc += alloc_i
            tot_select += select_i
            tot_inter += inter_i

            rows.append({
                "segment": seg,
                "w_portfolio": wp_i,
                "w_benchmark": wb_i,
                "r_portfolio": rp_i,
                "r_benchmark": rb_i,
                "allocation_effect": alloc_i,
                "selection_effect": select_i,
                "interaction_effect": inter_i,
                "total_effect": total_i,
            })

        df_attr = pd.DataFrame(rows).set_index("segment")
        excess_ret = R_p_total - R_b_total

        logger.info(
            f"[Risk Attribution] BHB Step 10 Complete: Excess Return={excess_ret:.4f} | "
            f"Alloc={tot_alloc:.4f}, Select={tot_select:.4f}, Inter={tot_inter:.4f}"
        )

        return {
            "per_segment": df_attr.to_dict(orient="index"),
            "total_allocation": float(tot_alloc),
            "total_selection": float(tot_select),
            "total_interaction": float(tot_inter),
            "total_excess_return": float(excess_ret),
            "portfolio_return": float(R_p_total),
            "benchmark_return": float(R_b_total),
        }
