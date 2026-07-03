import math
import logging
from typing import Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

class DoubleMachineLearningATE:
    """
    Double/Debiased Machine Learning (DML) estimator for Average Treatment Effect (ATE).
    Uses Random Forest models for nuisance function estimation and cross-fitting to prevent bias.
    """
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        Estimates the ATE and its p-value using cross-fitted DML.
        Formula:
        ATE = 1/n * sum( (T_i - m_hat(X_i)) * (Y_i - l_hat(X_i)) / (m_hat(X_i) * (1 - m_hat(X_i))) )
        
        Returns:
            Tuple of (ate_magnitude, p_value)
        """
        n = len(Y)
        if n < 10:
            logger.warning("Insufficient samples for DML. Returning defaults.")
            return 0.0, 1.0

        # Validate that T is binary (contains only 0 or 1, or boolean)
        unique_t = np.unique(T)
        if not np.all(np.isin(unique_t, [0, 1, 0.0, 1.0, True, False])):
            raise ValueError("Treatment variable T must be binary (containing only 0, 1, True, or False).")

        # Initialise cross-fitting folds
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Propensity scores m(X) and outcome predictions l(X)
        m_hat = np.zeros(n)
        l_hat = np.zeros(n)
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # 1. Fit Propensity score model: predicting T from X
            m_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            m_model.fit(X_train, T_train)
            
            # Predict probabilities, clip to avoid division by zero
            preds_m = m_model.predict_proba(X_val)[:, 1]
            m_hat[val_idx] = np.clip(preds_m, 0.01, 0.99)
            
            # 2. Fit Outcome regression model: predicting Y from X
            l_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            l_model.fit(X_train, Y_train)
            l_hat[val_idx] = l_model.predict(X_val)

        # Compute DML residuals and individual influence terms
        # influence_i = (T_i - m_hat_i) * (Y_i - l_hat_i) / (m_hat_i * (1 - m_hat_i))
        influence = (T - m_hat) * (Y - l_hat) / (m_hat * (1.0 - m_hat))
        
        ate = float(np.mean(influence))
        
        # Compute standard error of the mean influence
        residuals = influence - ate
        variance = np.mean(residuals ** 2)
        se = math.sqrt(variance / n)
        
        if se == 0:
            return ate, 1.0
            
        # Z-test statistic
        z_stat = ate / se
        # Two-sided p-value
        p_value = 2 * (1.0 - self._normal_cdf(abs(z_stat)))
        
        return ate, float(p_value)

    def _normal_cdf(self, x: float) -> float:
        """
        Cumulative distribution function for standard normal distribution.
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
