"""
AlphaLens — Deep Learning Ensemble Combiner
Weighted ensemble of TFT, N-BEATS, and PatchTST predictions
with optional regime-conditional weighting.
"""

import logging
from typing import List, Dict, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Combines predictions from TFT, N-BEATS, and PatchTST using inverse-MSE
    weighting. Supports regime-conditional weight switching.
    """
    def __init__(self, forecast_horizons: List[int] = None):
        self.forecast_horizons = forecast_horizons or [1, 5, 20]
        self.weights = {"tft": 1/3, "nbeats": 1/3, "patchtst": 1/3}
        self.regime_weights = {
            "bull":    {"tft": 0.40, "nbeats": 0.30, "patchtst": 0.30},
            "bear":    {"tft": 0.30, "nbeats": 0.40, "patchtst": 0.30},
            "high_vol": {"tft": 0.35, "nbeats": 0.25, "patchtst": 0.40},
        }

    def calibrate_weights(self, 
                          tft_errors: Dict[int, float],
                          nbeats_errors: Dict[int, float],
                          patchtst_errors: Dict[int, float]) -> Dict[str, float]:
        """
        Calibrate ensemble weights using inverse-MSE weighting.
        
        Args:
            *_errors: dict mapping horizon -> MSE for each model
        
        Returns:
            Updated weight dict
        """
        # Average MSE across horizons for each model
        avg_errors = {}
        for name, errors in [("tft", tft_errors), ("nbeats", nbeats_errors), ("patchtst", patchtst_errors)]:
            if errors:
                avg_errors[name] = np.mean(list(errors.values()))
            else:
                avg_errors[name] = 1.0

        # Inverse-MSE weighting (lower error = higher weight)
        inv_errors = {k: 1.0 / max(v, 1e-10) for k, v in avg_errors.items()}
        total = sum(inv_errors.values())
        self.weights = {k: v / total for k, v in inv_errors.items()}

        logger.info(f"Ensemble weights calibrated: {self.weights}")
        return self.weights

    def combine(self,
                tft_preds: Dict[int, Any],
                nbeats_preds: Dict[int, Any],
                patchtst_preds: Dict[int, Any],
                regime: Optional[str] = None) -> Dict[int, np.ndarray]:
        """
        Combines predictions from all three models.
        
        Args:
            tft_preds: dict mapping horizon -> prediction
            nbeats_preds: dict mapping horizon -> prediction
            patchtst_preds: dict mapping horizon -> prediction
            regime: optional market regime for regime-conditional weighting
        
        Returns:
            dict mapping horizon -> ensemble prediction
        """
        # Select weights based on regime
        if regime and regime in self.regime_weights:
            w = self.regime_weights[regime]
        else:
            w = self.weights

        ensemble = {}
        for h in self.forecast_horizons:
            tft_val = self._to_float(tft_preds.get(h, 0.0))
            nbeats_val = self._to_float(nbeats_preds.get(h, 0.0))
            patchtst_val = self._to_float(patchtst_preds.get(h, 0.0))

            combined = (
                w["tft"] * tft_val +
                w["nbeats"] * nbeats_val +
                w["patchtst"] * patchtst_val
            )
            ensemble[h] = combined

        return ensemble

    def get_model_contributions(self, regime: Optional[str] = None) -> Dict[str, float]:
        """Returns the current weight contribution of each model."""
        if regime and regime in self.regime_weights:
            return dict(self.regime_weights[regime])
        return dict(self.weights)

    @staticmethod
    def _to_float(val: Any) -> float:
        """Convert prediction value to scalar float."""
        if isinstance(val, np.ndarray):
            return float(val.mean()) if val.size > 0 else 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
