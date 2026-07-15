"""
Partial R² Sensitivity Analysis for Causal Inference (§5.4).

Quantifies how much of the treatment effect can be attributed to omitted
confounders by comparing the explained variance of models with and without
the treatment variable.

    partial_R² = (R²_full - R²_restricted) / (1 - R²_restricted)

A high partial R² indicates that the treatment adds substantial explanatory
power beyond confounders, making the causal estimate more robust.
"""
import math
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PartialR2Analysis:
    """
    Implements Partial R² sensitivity analysis for assessing the robustness
    of causal estimates to omitted variable bias (§5.4).
    """

    @staticmethod
    def compute_r_squared(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Computes R² using OLS regression: R² = 1 - SS_res / SS_tot.
        X: (n, p) feature matrix
        Y: (n,) outcome vector
        """
        n = len(Y)
        if n < 3 or X.shape[1] == 0:
            return 0.0

        # Add intercept
        ones = np.ones((n, 1))
        X_aug = np.hstack([ones, X])

        # OLS: beta = (X'X)^{-1} X'Y
        try:
            beta = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0

        Y_hat = X_aug @ beta
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1.0 - ss_res / ss_tot)

    def compute_partial_r2(
        self,
        X_confounders: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Computes the Partial R² of the treatment variable:
        
            partial_R² = (R²_full - R²_restricted) / (1 - R²_restricted)

        Args:
            X_confounders: (n, p) matrix of observed confounders
            treatment: (n,) treatment variable (binary or continuous)
            outcome: (n,) outcome variable

        Returns:
            Tuple of (partial_r2, r2_full, r2_restricted)
        """
        n = len(outcome)
        if n < 5:
            logger.warning("Insufficient samples for Partial R² analysis.")
            return 0.0, 0.0, 0.0

        treatment = treatment.reshape(-1, 1) if treatment.ndim == 1 else treatment

        # R²_restricted: outcome ~ confounders only
        r2_restricted = self.compute_r_squared(X_confounders, outcome)

        # R²_full: outcome ~ confounders + treatment
        X_full = np.hstack([X_confounders, treatment])
        r2_full = self.compute_r_squared(X_full, outcome)

        # Partial R²
        denom = 1.0 - r2_restricted
        if denom <= 1e-10:
            # If confounders already explain everything, partial R² is undefined
            return 0.0, r2_full, r2_restricted

        partial_r2 = (r2_full - r2_restricted) / denom

        logger.info(
            f"Partial R² analysis: R²_full={r2_full:.4f}, "
            f"R²_restricted={r2_restricted:.4f}, "
            f"partial_R²={partial_r2:.4f}"
        )

        return float(partial_r2), float(r2_full), float(r2_restricted)

    def robustness_assessment(
        self,
        partial_r2: float,
        threshold: float = 0.05,
    ) -> str:
        """
        Interprets the Partial R² value for robustness reporting.
        
        A partial R² close to 0 means treatment adds little beyond confounders.
        A partial R² > threshold suggests treatment has genuine explanatory power.
        """
        if partial_r2 >= threshold:
            return (
                f"ROBUST: Treatment partial R²={partial_r2:.4f} exceeds threshold "
                f"{threshold}. An omitted confounder would need to explain "
                f">{partial_r2:.1%} of residual outcome variance to nullify the effect."
            )
        else:
            return (
                f"FRAGILE: Treatment partial R²={partial_r2:.4f} is below threshold "
                f"{threshold}. The causal estimate may be sensitive to omitted confounders."
            )
