"""
AlphaLens — Market Regime Detection
Hidden Markov Model (HMM) with 3 states (Bull, Bear, High-Volatility)
and CUSUM changepoint detection for adaptive architecture switching.
"""

import math
import logging
from typing import List, Tuple, Dict, Optional
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. Regime detection will use rule-based fallback.")


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"


class RegimeDetector:
    """
    Detects market regimes using a 3-state Gaussian HMM fitted on
    returns and volatility features. Falls back to a rule-based
    approach if hmmlearn is not installed.
    """
    def __init__(self, n_regimes: int = 3, lookback_window: int = 252,
                 vol_window: int = 20, random_state: int = 42):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.vol_window = vol_window
        self.random_state = random_state
        self.hmm_model = None
        self.is_fitted = False
        self._regime_map = {}  # Maps HMM state index -> MarketRegime

    def fit(self, returns: np.ndarray) -> "RegimeDetector":
        """
        Fits the HMM on historical returns.
        
        Args:
            returns: (n_days,) array of daily returns
        """
        if len(returns) < self.lookback_window:
            logger.warning("RegimeDetector: Insufficient data. Using rule-based fallback.")
            self.is_fitted = True
            return self

        # Build observation matrix: [returns, rolling_vol]
        obs = self._build_features(returns)

        if HMM_AVAILABLE and len(obs) > 30:
            try:
                self.hmm_model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=200,
                    random_state=self.random_state,
                )
                self.hmm_model.fit(obs)

                # Map HMM states to regime labels based on mean return and vol
                means = self.hmm_model.means_
                self._map_states_to_regimes(means)

                self.is_fitted = True
                logger.info("RegimeDetector: HMM fitted successfully.")
            except Exception as e:
                logger.warning(f"RegimeDetector: HMM fitting failed: {e}. Using fallback.")
                self.hmm_model = None
                self.is_fitted = True
        else:
            self.is_fitted = True

        return self

    def predict(self, returns: np.ndarray) -> MarketRegime:
        """
        Predicts the current market regime.
        
        Args:
            returns: (n_days,) recent return series
        
        Returns:
            Current MarketRegime
        """
        if self.hmm_model is not None:
            obs = self._build_features(returns)
            if len(obs) > 0:
                states = self.hmm_model.predict(obs)
                current_state = int(states[-1])
                return self._regime_map.get(current_state, MarketRegime.BULL)

        # Rule-based fallback
        return self._rule_based_regime(returns)

    def predict_probabilities(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Returns probability distribution over regimes.
        
        Args:
            returns: (n_days,) recent return series
        
        Returns:
            Dict mapping regime name -> probability
        """
        if self.hmm_model is not None:
            obs = self._build_features(returns)
            if len(obs) > 0:
                try:
                    probs = self.hmm_model.predict_proba(obs)
                    last_probs = probs[-1]
                    result = {}
                    for state_idx, regime in self._regime_map.items():
                        if state_idx < len(last_probs):
                            result[regime.value] = float(last_probs[state_idx])
                    # Ensure all regimes present
                    for r in MarketRegime:
                        if r.value not in result:
                            result[r.value] = 0.0
                    return result
                except Exception:
                    pass

        # Rule-based fallback probabilities
        regime = self._rule_based_regime(returns)
        probs = {r.value: 0.1 for r in MarketRegime}
        probs[regime.value] = 0.8
        return probs

    def detect_changepoints(self, returns: np.ndarray,
                            threshold: float = 3.0) -> List[int]:
        """
        Detects structural breaks using CUSUM (Cumulative Sum) algorithm.
        
        Args:
            returns: (n_days,) return series
            threshold: CUSUM threshold in standard deviations
        
        Returns:
            List of changepoint indices
        """
        n = len(returns)
        if n < 10:
            return []

        mean = np.mean(returns)
        std = np.std(returns)
        if std < 1e-10:
            return []

        # Two-sided CUSUM
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        changepoints = []

        for i in range(1, n):
            z = (returns[i] - mean) / std
            s_pos[i] = max(0, s_pos[i-1] + z - 0.5)
            s_neg[i] = max(0, s_neg[i-1] - z - 0.5)

            if s_pos[i] > threshold or s_neg[i] > threshold:
                changepoints.append(i)
                s_pos[i] = 0
                s_neg[i] = 0

        return changepoints

    # --- Internal helpers ---

    def _build_features(self, returns: np.ndarray) -> np.ndarray:
        """Builds [return, rolling_vol] observation matrix."""
        n = len(returns)
        vol = np.zeros(n)
        for i in range(self.vol_window, n):
            vol[i] = np.std(returns[i - self.vol_window:i])

        # Skip initial zero-vol period
        start = self.vol_window
        if start >= n:
            start = max(1, n // 2)

        obs = np.column_stack((returns[start:], vol[start:]))
        return obs

    def _map_states_to_regimes(self, means: np.ndarray):
        """
        Maps HMM states to human-readable regimes based on
        mean return (col 0) and mean volatility (col 1).
        """
        n_states = means.shape[0]
        mean_returns = means[:, 0]
        mean_vols = means[:, 1] if means.shape[1] > 1 else np.zeros(n_states)

        # Sort by volatility
        vol_order = np.argsort(mean_vols)

        if n_states >= 3:
            # Highest vol state = HIGH_VOL
            self._regime_map[int(vol_order[-1])] = MarketRegime.HIGH_VOL

            # Among remaining, highest return = BULL, lowest = BEAR
            remaining = [int(vol_order[i]) for i in range(n_states - 1)]
            ret_vals = [(idx, mean_returns[idx]) for idx in remaining]
            ret_vals.sort(key=lambda x: x[1])
            self._regime_map[ret_vals[0][0]] = MarketRegime.BEAR
            self._regime_map[ret_vals[-1][0]] = MarketRegime.BULL

            # Any remaining states
            for idx in remaining:
                if idx not in self._regime_map:
                    self._regime_map[idx] = MarketRegime.BULL
        elif n_states == 2:
            if mean_returns[0] > mean_returns[1]:
                self._regime_map[0] = MarketRegime.BULL
                self._regime_map[1] = MarketRegime.BEAR
            else:
                self._regime_map[0] = MarketRegime.BEAR
                self._regime_map[1] = MarketRegime.BULL
        else:
            self._regime_map[0] = MarketRegime.BULL

    def _rule_based_regime(self, returns: np.ndarray) -> MarketRegime:
        """Simple rule-based regime classification fallback."""
        if len(returns) < 5:
            return MarketRegime.BULL

        recent = returns[-20:] if len(returns) >= 20 else returns
        mean_ret = np.mean(recent)
        vol = np.std(recent)

        # High volatility threshold (annualized > 25%)
        annualized_vol = vol * math.sqrt(252)
        if annualized_vol > 0.25:
            return MarketRegime.HIGH_VOL
        elif mean_ret < -0.0005:  # Negative drift
            return MarketRegime.BEAR
        else:
            return MarketRegime.BULL
