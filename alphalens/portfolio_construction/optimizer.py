"""
Portfolio Optimisation — Mean-CVaR with Rockafellar-Uryasev LP (§10.1-10.2).

Minimises CVaR_α(r_p) subject to:
    μᵀw ≥ μ̄,  1ᵀw = 1,  w ≥ 0,  ‖w‖∞ ≤ w_max        (Eq 27)

CVaR is computed via the Rockafellar-Uryasev linearisation (Eq 29):
    CVaR_α(r_p) = min_{γ} { γ + 1/(1-α) E[max(-r_p - γ, 0)] }

which converts to the LP (Eq 30):
    min_{w,γ,z_s}  γ + 1/((1-α)S) Σ z_s
    s.t.  z_s ≥ -r_{p,s} - γ,   s = 1,...,S
          z_s ≥ 0
          portfolio constraints (Eq 27)

Solver: CVXPY with CLARABEL backend (§10.2).
Falls back to SciPy linprog if CVXPY is unavailable.
"""
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try CVXPY first (paper-specified), fall back to SciPy
try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
    _CLARABEL_AVAILABLE = "CLARABEL" in cp.installed_solvers()
except ImportError:
    _CVXPY_AVAILABLE = False
    _CLARABEL_AVAILABLE = False


class CVaROptimizer:
    """
    Mean-CVaR portfolio optimizer implementing the Rockafellar-Uryasev
    linear programming formulation (§10.1-10.2, Eq 27-30).

    Minimises CVaR at confidence level α subject to:
      - Expected portfolio return ≥ min_required_return (μᵀw ≥ μ̄)
      - Portfolio weights sum to 1 (1ᵀw = 1)
      - Long-only constraints (w ≥ 0)
      - Maximum weight limits (‖w‖∞ ≤ w_max)
    """
    def __init__(
        self,
        alpha: float = 0.99,
        w_max: float = 0.35,
        cvar_max: float = 0.025,  # Step 11: CVaR_0.99 <= 2.5% daily
        max_turnover: float = 0.20,  # Step 11: Maximum 20% turnover per rebalance
        min_net_exposure: float = 0.90,  # Step 11: Net exposure lower bound
        max_net_exposure: float = 1.10,  # Step 11: Net exposure upper bound
    ):
        self.alpha = alpha
        self.w_max = w_max
        self.cvar_max = cvar_max
        self.max_turnover = max_turnover
        self.min_net_exposure = min_net_exposure
        self.max_net_exposure = max_net_exposure

    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        historical_returns: np.ndarray,
        min_required_return: float,
        w_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Solves the mean-CVaR LP (Eq 30, Step 11 constraints).

        Args:
            expected_returns: (N,) vector μ of expected asset returns.
            historical_returns: (S, N) matrix of S historical return scenarios.
            min_required_return: μ̄ — minimum required portfolio return.
            w_prev: Optional (N,) vector of previous portfolio weights for turnover constraint.

        Returns:
            Tuple of (optimal_weights, optimal_CVaR).
        """
        if _CVXPY_AVAILABLE:
            return self._solve_cvxpy(expected_returns, historical_returns, min_required_return, w_prev)
        else:
            return self._solve_scipy(expected_returns, historical_returns, min_required_return, w_prev)

    def _solve_cvxpy(
        self,
        expected_returns: np.ndarray,
        historical_returns: np.ndarray,
        min_required_return: float,
        w_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        §10.2 / Step 11: CVXPY with CLARABEL backend.

        Variables: w (N), γ (1), z (S)
        Objective: min γ + 1/((1-α)S) Σ z_s
        Subject to:
            z_s ≥ -r_{p,s} - γ,  ∀s                  (scenario loss constraint)
            γ + 1/((1-α)S) Σ z_s ≤ cvar_max           (Step 11: CVaR_0.99 ≤ 2.5% daily)
            μᵀw ≥ μ̄                                  (return constraint)
            min_net_exposure ≤ 1ᵀw ≤ max_net_exposure (Step 11: net exposure ∈ [0.90, 1.10])
            0 ≤ w ≤ w_max                             (long-only + concentration limit)
            ‖w - w_prev‖₁ ≤ max_turnover              (Step 11: turnover ≤ 20%)
        """
        S, N = historical_returns.shape

        # Decision variables
        w = cp.Variable(N, name="weights")
        gamma = cp.Variable(name="gamma")
        z = cp.Variable(S, name="z_aux", nonneg=True)

        # Objective (Eq 30): min γ + 1/((1-α)S) Σ z_s
        cvar_expr = gamma + (1.0 / ((1.0 - self.alpha) * S)) * cp.sum(z)
        objective = cp.Minimize(cvar_expr)

        # Constraints
        constraints = [
            # z_s ≥ -r_{p,s} - γ  ⟹  z_s ≥ -(R_s @ w) - γ
            z >= -historical_returns @ w - gamma,
            # Step 11: CVaR_0.99 <= 2.5% daily
            cvar_expr <= self.cvar_max,
            # μᵀw ≥ μ̄
            expected_returns @ w >= min_required_return,
            # Step 11: Net exposure constraint (between 0.90 and 1.10)
            cp.sum(w) >= self.min_net_exposure,
            cp.sum(w) <= self.max_net_exposure,
            # w ≥ 0
            w >= 0,
            # ‖w‖∞ ≤ w_max
            w <= self.w_max,
        ]

        # Step 11: Max turnover constraint of 20% per rebalance if w_prev provided
        if w_prev is not None and len(w_prev) == N:
            constraints.append(cp.norm(w - w_prev, 1) <= self.max_turnover)

        prob = cp.Problem(objective, constraints)

        # Solve with CLARABEL (paper-specified), fallback to default
        solver = cp.CLARABEL if _CLARABEL_AVAILABLE else None
        try:
            prob.solve(solver=solver)
        except cp.SolverError:
            logger.warning("CLARABEL solver failed, trying default CVXPY solver.")
            prob.solve()

        if prob.status in ("optimal", "optimal_inaccurate"):
            w_opt = np.clip(w.value, 0.0, self.w_max)
            w_opt = w_opt / np.sum(w_opt)  # Normalise for float rounding
            cvar_opt = float(prob.value)
            logger.info(
                f"[CVaR Optimizer] §10.2 CVXPY+{solver or 'default'}: "
                f"CVaR={cvar_opt:.6f}, status={prob.status}"
            )
            return w_opt, cvar_opt
        else:
            logger.warning(
                f"CVXPY optimization failed: {prob.status}. Returning equal weights."
            )
            return self._fallback_equal_weights(historical_returns)

    def _solve_scipy(
        self,
        expected_returns: np.ndarray,
        historical_returns: np.ndarray,
        min_required_return: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Fallback: SciPy linprog with HiGHS solver.
        Same LP formulation (Eq 30), different solver backend.
        """
        from scipy.optimize import linprog

        S, N = historical_returns.shape

        # c coefficient vector: minimize γ + 1/((1-α)S) Σ z_s
        c = np.zeros(N + 1 + S)
        c[N] = 1.0
        c[N + 1:] = 1.0 / ((1.0 - self.alpha) * S)

        # Inequality constraints: A_ub * x <= b_ub
        A_ub = []
        b_ub = []

        # z_s ≥ -r_{p,s} - γ  ⟹  -Σ(w_i * R_{s,i}) - γ - z_s ≤ 0
        for s in range(S):
            row = np.zeros(N + 1 + S)
            row[:N] = -historical_returns[s, :]
            row[N] = -1.0
            row[N + 1 + s] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        # μᵀw ≥ μ̄  ⟹  -μᵀw ≤ -μ̄
        return_row = np.zeros(N + 1 + S)
        return_row[:N] = -expected_returns
        A_ub.append(return_row)
        b_ub.append(-min_required_return)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Equality: 1ᵀw = 1
        A_eq = np.zeros((1, N + 1 + S))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w ∈ [0, w_max], γ ∈ ℝ, z ∈ [0, ∞)
        bounds = [(0.0, self.w_max) for _ in range(N)]
        bounds.append((None, None))  # γ
        bounds.extend([(0.0, None) for _ in range(S)])  # z_s

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

        if res.success:
            w_opt = np.clip(res.x[:N], 0.0, self.w_max)
            w_opt = w_opt / np.sum(w_opt)
            return w_opt, float(res.fun)
        else:
            logger.warning(f"SciPy optimization failed: {res.message}. Returning equal weights.")
            return self._fallback_equal_weights(historical_returns)

    def _fallback_equal_weights(self, historical_returns: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fallback to equal-weight portfolio with empirical CVaR."""
        N = historical_returns.shape[1]
        w_eq = np.ones(N) / N
        eq_returns = historical_returns @ w_eq
        var = np.percentile(eq_returns, (1 - self.alpha) * 100)
        cvar_val = -np.mean(eq_returns[eq_returns <= var])
        return w_eq, float(cvar_val)
