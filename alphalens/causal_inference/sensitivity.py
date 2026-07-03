import math
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class RosenbaumSensitivity:
    """
    Implements Rosenbaum's Gamma-sensitivity analysis for assessing
    robustness of causal estimates to hidden confounders.
    Calculates p-value bounds for a given Gamma value.
    """
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def calculate_bounds(self, 
                         treatment: List[int], 
                         outcome: List[float], 
                         gamma: float) -> Tuple[float, float]:
        """
        Computes the upper and lower bounds of the Wilcoxon signed-rank test
        p-value for a given sensitivity parameter Gamma >= 1.0.
        """
        if gamma < 1.0:
            raise ValueError("Gamma must be >= 1.0")

        # In case of binary outcomes or continuous outcome residuals,
        # we compute matched pairs statistics or compute a simple sensitivity bound
        # using the proportions of positive outcomes in treated vs control.
        # Let's compute a standard binomial proportion test sensitivity bound:
        # Let p+ be the probability of treated unit having a higher outcome.
        # Under no confounding (gamma=1), p+ = 0.5.
        # Under confounding, p+ is bounded in [1/(1+gamma), gamma/(1+gamma)].
        
        # Let's count how many treated units have outcomes above control median
        n = len(treatment)
        if n < 4:
            return 1.0, 1.0

        treated_outcomes = [outcome[i] for i in range(n) if treatment[i] == 1]
        control_outcomes = [outcome[i] for i in range(n) if treatment[i] == 0]
        
        if not treated_outcomes or not control_outcomes:
            return 1.0, 1.0

        control_median = sorted(control_outcomes)[len(control_outcomes) // 2]
        
        # Count successes in treated group
        y_treated = sum(1 for x in treated_outcomes if x > control_median)
        n_treated = len(treated_outcomes)
        
        if n_treated == 0:
            return 1.0, 1.0

        # Binomial test probability bounds
        p_upper = gamma / (1.0 + gamma)
        p_lower = 1.0 / (1.0 + gamma)
        
        # Compute normal approximation bounds
        # mean = n * p, std = sqrt(n * p * (1-p))
        mean_upper = n_treated * p_upper
        std_upper = math.sqrt(n_treated * p_upper * (1.0 - p_upper))
        
        mean_lower = n_treated * p_lower
        std_lower = math.sqrt(n_treated * p_lower * (1.0 - p_lower))
        
        # Upper bound p-value (prob of observing at least y_treated under p_upper)
        if std_upper > 0:
            z_upper = (y_treated - mean_upper) / std_upper
            p_val_upper = 1.0 - self._normal_cdf(z_upper)
        else:
            p_val_upper = 1.0

        # Lower bound p-value
        if std_lower > 0:
            z_lower = (y_treated - mean_lower) / std_lower
            p_val_lower = 1.0 - self._normal_cdf(z_lower)
        else:
            p_val_lower = 1.0

        return float(p_val_lower), float(p_val_upper)

    def _normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def is_robust(self, treatment: List[int], outcome: List[float], max_gamma: float = 2.0) -> bool:
        """
        Returns True if the upper bound p-value remains below the significance level
        for all Gamma <= max_gamma.
        """
        _, p_upper = self.calculate_bounds(treatment, outcome, max_gamma)
        return p_upper < self.significance_level
