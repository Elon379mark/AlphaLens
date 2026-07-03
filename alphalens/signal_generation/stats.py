import math
import logging
from typing import List, Union, Tuple, Optional

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
except ImportError:
    np = None
    pd = None
    spearmanr = None

logger = logging.getLogger(__name__)

def get_ranks(x: List[float]) -> List[float]:
    """
    Computes fractional ranks of a list, handling ties by averaging ranks.
    """
    n = len(x)
    indexed = sorted(enumerate(x), key=lambda item: item[1])
    ranks = [0.0] * n
    
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        # average rank for indices i to j-1 (1-indexed ranks)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks

def compute_spearman_rho(x: List[float], y: List[float]) -> float:
    """
    Computes Spearman rank correlation coefficient between two lists.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    if spearmanr is not None:
        rho, _ = spearmanr(x, y)
        if math.isnan(rho):
            return 0.0
        return float(rho)
        
    rx = get_ranks(x)
    ry = get_ranks(y)
    
    n = len(x)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = sum((rx[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((ry[i] - mean_y) ** 2 for i in range(n))
    
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    
    return num / math.sqrt(den_x * den_y)


class SignalStatistics:
    """
    Computes statistics for financial signals:
    - Information Coefficient (IC)
    - Information Ratio (ICIR)
    - Half-life Decay Analysis
    - Signal Orthogonalisation
    """
    @staticmethod
    def calculate_ic(predictions: List[float], returns: List[float]) -> float:
        """
        Computes the Information Coefficient (IC) as Spearman rank correlation.
        """
        return compute_spearman_rho(predictions, returns)

    @staticmethod
    def calculate_icir(ic_history: List[float], rolling_window: int = 252) -> float:
        """
        Computes the IC Information Ratio: ICIR = mean(IC) / std(IC) * sqrt(rolling_window)
        Note: Multiplying by sqrt(rolling_window) annualizes the ICIR (assuming daily rebalancing).
        Standard (non-annualized) ICIR is typically just mean(IC) / std(IC).
        """
        if len(ic_history) < 2:
            return 0.0
            
        w = min(len(ic_history), rolling_window)
        recent_ic = ic_history[-w:]
        
        n = len(recent_ic)
        mean_ic = sum(recent_ic) / n
        variance = sum((x - mean_ic) ** 2 for x in recent_ic) / (n - 1)
        std_ic = math.sqrt(variance)
        
        if std_ic == 0.0:
            return 0.0
            
        return (mean_ic / std_ic) * math.sqrt(rolling_window)

    @staticmethod
    def estimate_half_life(signal: List[float], max_lag: int = 30) -> Tuple[float, float]:
        """
        Estimates the half-life of a signal using autoregressive decay fitting:
        rho(tau) = rho_0 * e^(-lambda * tau)
        lambda_hat = - ln(rho_hat(tau)) / tau
        Returns (lambda_hat, half_life_in_days).
        """
        if len(signal) < max_lag + 2:
            return 0.0, 0.0

        # Compute autocorr for tau = 1..max_lag
        mean_sig = sum(signal) / len(signal)
        var_sig = sum((x - mean_sig) ** 2 for x in signal)
        
        if var_sig == 0.0:
            return 0.0, 0.0

        autocorrs = []
        for lag in range(1, max_lag + 1):
            num = 0.0
            for t in range(len(signal) - lag):
                num += (signal[t] - mean_sig) * (signal[t + lag] - mean_sig)
            rho_tau = num / var_sig
            autocorrs.append(rho_tau)

        # Autoregressive decay fitting: fit log(rho(tau)) = log(rho_0) - lambda * tau
        # Formulate simple linear regression: y_i = log(autocorrs[i])
        # x_i = i + 1 (tau)
        valid_points = []
        for tau_idx, rho in enumerate(autocorrs):
            if rho > 0.01:  # ignore zero/negative correlations for log
                valid_points.append((tau_idx + 1, math.log(rho)))
        
        if len(valid_points) < 2:
            return 0.0, 0.0

        n = len(valid_points)
        sum_x = sum(pt[0] for pt in valid_points)
        sum_y = sum(pt[1] for pt in valid_points)
        sum_xx = sum(pt[0]**2 for pt in valid_points)
        sum_xy = sum(pt[0]*pt[1] for pt in valid_points)
        
        denom = (n * sum_xx - sum_x**2)
        if denom == 0.0:
            return 0.0, 0.0
            
        # Slope is -lambda
        neg_lambda = (n * sum_xy - sum_x * sum_y) / denom
        lambda_val = -neg_lambda
        
        if lambda_val <= 1e-6:
            # avoid log(2)/0 or negative lambda
            return 0.0, float('inf')
            
        half_life = math.log(2.0) / lambda_val
        return lambda_val, half_life

    @staticmethod
    def orthogonalise_signal(new_signal: List[float], existing_signals: List[List[float]]) -> List[float]:
        """
        Projects the new signal vector onto the orthogonal complement of the existing signals:
        s_tilde = s_i - S(S^T S)^(-1) S^T s_i
        
        new_signal: List of length T
        existing_signals: List of K lists, each of length T
        """
        T = len(new_signal)
        if not existing_signals or len(existing_signals) == 0:
            return new_signal

        # NumPy implementation
        if np is not None:
            s = np.array(new_signal, dtype=np.float64)
            S = np.array(existing_signals, dtype=np.float64).T  # T x K
            
            # Use pseudo-inverse to handle multicollinearity safely
            try:
                # projection matrix P = S (S^T S)^(-1) S^T
                proj_coeffs = np.linalg.pinv(S.T @ S) @ S.T @ s
                s_proj = S @ proj_coeffs
                s_orthogonal = s - s_proj
                return s_orthogonal.tolist()
            except np.linalg.LinAlgError:
                logger.error("Linear algebra error in signal orthogonalisation. Returning raw signal.")
                return new_signal

        # Pure-Python fallback using Gram-Schmidt orthogonalisation
        s = list(new_signal)
        for ref_sig in existing_signals:
            # project s onto ref_sig
            dot_product = sum(a * b for a, b in zip(s, ref_sig))
            norm_sq = sum(a * a for a in ref_sig)
            if norm_sq > 0.0:
                factor = dot_product / norm_sq
                s = [s[idx] - factor * ref_sig[idx] for idx in range(T)]
        return s

    @staticmethod
    def evaluate_gating(ic: float, icir: float) -> bool:
        """
        Returns True if the signal passes the predictive screening thresholds:
        - |IC| >= 0.03 and |ICIR| >= 0.5
        """
        return abs(ic) >= 0.03 and abs(icir) >= 0.5
