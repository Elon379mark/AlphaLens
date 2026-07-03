import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

class CVaROptimizer:
    """
    Mean-CVaR portfolio optimizer implementing the Rockafellar-Uryasev linear programming formulation.
    Minimizes CVaR at a given confidence level alpha subject to:
      - Expected portfolio return >= min_required_return (mu^T w >= bar_mu)
      - Portfolio weights sum to 1 (1^T w = 1)
      - Long-only constraints (w_i >= 0)
      - Maximum weight limits (w_i <= w_max)
    """
    def __init__(self, alpha: float = 0.99, w_max: float = 0.25):
        self.alpha = alpha
        self.w_max = w_max

    def optimize_portfolio(self, 
                           expected_returns: np.ndarray, 
                           historical_returns: np.ndarray, 
                           min_required_return: float) -> Tuple[np.ndarray, float]:
        """
        Solves the LP optimization:
        Let S be the number of historical return scenarios, N be the number of assets.
        Variables: x = [w_1, ..., w_N, gamma, z_1, ..., z_S]
        Size of x: N + 1 + S
        
        Returns:
            Tuple of (optimal_weights, optimal_CVaR)
        """
        S, N = historical_returns.shape
        
        # c coefficient vector: minimize gamma + 1/((1 - alpha)*S) * sum(z_s)
        c = np.zeros(N + 1 + S)
        c[N] = 1.0
        c[N + 1:] = 1.0 / ((1.0 - self.alpha) * S)
        
        # Inequality constraints: A_ub * x <= b_ub
        # 1. z_s >= - r_p_s - gamma => - sum_i(w_i * R_s_i) - gamma - z_s <= 0
        # This gives S constraints.
        A_ub = []
        b_ub = []
        
        for s in range(S):
            row = np.zeros(N + 1 + S)
            row[:N] = -historical_returns[s, :]  # -R_s_i
            row[N] = -1.0                         # -gamma
            row[N + 1 + s] = -1.0                 # -z_s
            A_ub.append(row)
            b_ub.append(0.0)
            
        # 2. Expected return constraint: - mu^T w <= - min_required_return
        return_row = np.zeros(N + 1 + S)
        return_row[:N] = -expected_returns
        A_ub.append(return_row)
        b_ub.append(-min_required_return)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraints: A_eq * x == b_eq
        # 1^T w = 1
        A_eq = np.zeros((1, N + 1 + S))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds on variables
        # w_i in [0, w_max]
        # gamma in [-inf, inf]
        # z_s in [0, inf]
        bounds = []
        for i in range(N):
            bounds.append((0.0, self.w_max))
        bounds.append((None, None))  # gamma
        for s in range(S):
            bounds.append((0.0, None))   # z_s
            
        # Solve using SciPy linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            w_opt = res.x[:N]
            cvar_opt = float(res.fun)
            # Normalise to prevent minor float rounding issues
            w_opt = np.clip(w_opt, 0.0, self.w_max)
            w_opt = w_opt / np.sum(w_opt)
            return w_opt, cvar_opt
        else:
            logger.warning(f"Portfolio optimization failed: {res.message}. Returning equal weights.")
            # Fallback to equal weights
            w_eq = np.ones(N) / N
            # Compute empirical CVaR of equal weights
            eq_returns = historical_returns @ w_eq
            var = np.percentile(eq_returns, (1 - self.alpha) * 100)
            cvar_val = -np.mean(eq_returns[eq_returns <= var])
            return w_eq, float(cvar_val)
