import logging
import numpy as np

logger = logging.getLogger(__name__)

class BlackLitterman:
    """
    Implements the Black-Litterman asset allocation view blending framework.
    Combines market equilibrium returns with subjective/model-derived views.
    Equation:
      mu_star = [(tau * Sigma)^(-1) + P^T * Omega^(-1) * P]^(-1) * [(tau * Sigma)^(-1) * Pi + P^T * Omega^(-1) * q]
    """
    def __init__(self, tau: float = 0.05):
        self.tau = tau

    def blend_views(self, 
                    market_equilibrium_returns: np.ndarray, 
                    covariance_matrix: np.ndarray, 
                    pick_matrix: np.ndarray, 
                    view_returns: np.ndarray, 
                    view_uncertainties: np.ndarray) -> np.ndarray:
        """
        Blends views with market equilibrium returns.
        
        Args:
            market_equilibrium_returns (Pi): N vector
            covariance_matrix (Sigma): N x N matrix
            pick_matrix (P): K x N matrix mapping views to assets
            view_returns (q): K vector of view values
            view_uncertainties (Omega): K vector representing diagonal of Omega matrix
        """
        Pi = market_equilibrium_returns
        Sigma = covariance_matrix
        P = pick_matrix
        q = view_returns
        
        # Build Omega diagonal matrix
        Omega = np.diag(view_uncertainties)
        
        try:
            # tau * Sigma
            tau_Sigma = self.tau * Sigma
            
            # Inverses
            inv_tau_Sigma = np.linalg.inv(tau_Sigma)
            inv_Omega = np.linalg.inv(Omega)
            
            # Left term: (tau * Sigma)^(-1) + P^T * Omega^(-1) * P
            left_term = inv_tau_Sigma + P.T @ inv_Omega @ P
            inv_left_term = np.linalg.inv(left_term)
            
            # Right term: (tau * Sigma)^(-1) * Pi + P^T * Omega^(-1) * q
            right_term = inv_tau_Sigma @ Pi + P.T @ inv_Omega @ q
            
            # Blend
            mu_star = inv_left_term @ right_term
            return mu_star
        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion error in Black-Litterman view blending: {e}. Returning raw market equilibrium returns.")
            return Pi
