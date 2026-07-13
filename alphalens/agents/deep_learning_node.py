"""
AlphaLens — Deep Learning Agent Node
LangGraph node wrapping TFT, N-BEATS, PatchTST, and ensemble
for multi-horizon forecasting within the pipeline.
"""

import logging
from typing import Dict, Any

import numpy as np

from alphalens.agents.deep_learning.tft import TFTForecaster
from alphalens.agents.deep_learning.nbeats import NBeatsForecaster
from alphalens.agents.deep_learning.patchtst import PatchTSTForecaster
from alphalens.agents.deep_learning.ensemble import EnsembleForecaster
from alphalens.core.regime import RegimeDetector
from alphalens.agents.memory import AgentMemoryEngine
from alphalens.core.utils import run_sync

logger = logging.getLogger(__name__)
_memory_engine = AgentMemoryEngine()


def deep_learning_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: trains TFT, N-BEATS, PatchTST on signal data
    and produces ensemble multi-horizon predictions.
    
    Reads from state:
        - signal_values, returns_values, close_prices
    
    Writes to state:
        - tft_predictions, nbeats_predictions, patchtst_predictions
        - ensemble_predictions, current_regime, regime_probabilities
    """
    logger.info("[Deep Learning Agent] Starting multi-model forecasting...")
    run_id = state.get("run_id", "default_run_id")

    signal_values = np.array(state.get("signal_values", []))
    returns_values = np.array(state.get("returns_values", []))
    close_prices = np.array(state.get("close_prices", []))

    if len(signal_values) < 30 or len(returns_values) < 30:
        logger.warning("[Deep Learning Agent] Insufficient data for DL models.")
        return {
            "current_node": "deep_learning_agent",
            "agent_logs": state.get("agent_logs", []) + [
                "🧠 Deep Learning Agent: Insufficient data, skipping."
            ],
        }

    # Build feature matrix (signal + price-derived features)
    n = min(len(signal_values), len(returns_values), len(close_prices))
    signal_values = signal_values[-n:]
    returns_values = returns_values[-n:]
    close_prices = close_prices[-n:]

    # Simple feature matrix: [signal, returns, log_price_change, volatility_proxy]
    log_changes = np.log(close_prices[1:] / close_prices[:-1])
    log_changes = np.concatenate([[0.0], log_changes])
    vol_proxy = np.zeros(n)
    for i in range(20, n):
        vol_proxy[i] = np.std(returns_values[i-20:i])

    features = np.column_stack([signal_values, returns_values, log_changes, vol_proxy])
    n_features = features.shape[1]

    forecast_horizons = [1, 5, 20]

    # --- 1. Regime Detection ---
    regime_detector = RegimeDetector()
    regime_detector.fit(returns_values)
    current_regime = regime_detector.predict(returns_values)
    regime_probs = regime_detector.predict_probabilities(returns_values)
    changepoints = regime_detector.detect_changepoints(returns_values)
    logger.info(f"[Deep Learning Agent] Detected regime: {current_regime.value}")

    # --- 2. TFT ---
    try:
        tft = TFTForecaster(
            n_features=n_features, hidden_dim=32, n_heads=2,
            forecast_horizons=forecast_horizons, epochs=30
        )
        tft.fit(features, returns_values)
        tft_preds = tft.predict(features)
        tft_var_importance = tft.get_variable_importance()
        logger.info(f"[Deep Learning Agent] TFT predictions: {tft_preds}")
    except Exception as e:
        logger.warning(f"[Deep Learning Agent] TFT failed: {e}")
        tft_preds = {h: 0.0 for h in forecast_horizons}
        tft_var_importance = None

    # --- 3. N-BEATS ---
    try:
        nbeats = NBeatsForecaster(
            lookback=min(60, n // 2), hidden_dim=128,
            forecast_horizons=forecast_horizons, epochs=30
        )
        nbeats.fit(close_prices)
        nbeats_preds = nbeats.predict(close_prices)
        logger.info(f"[Deep Learning Agent] N-BEATS predictions: {nbeats_preds}")
    except Exception as e:
        logger.warning(f"[Deep Learning Agent] N-BEATS failed: {e}")
        nbeats_preds = {h: 0.0 for h in forecast_horizons}

    # --- 4. PatchTST ---
    try:
        patchtst = PatchTSTForecaster(
            seq_len=min(60, n // 2), patch_len=8,
            d_model=32, n_heads=2, n_layers=2,
            forecast_horizons=forecast_horizons, epochs=30
        )
        patchtst.fit(features, returns_values)
        patchtst_preds = patchtst.predict(features)
        logger.info(f"[Deep Learning Agent] PatchTST predictions: {patchtst_preds}")
    except Exception as e:
        logger.warning(f"[Deep Learning Agent] PatchTST failed: {e}")
        patchtst_preds = {h: 0.0 for h in forecast_horizons}

    # --- 5. Ensemble ---
    ensemble = EnsembleForecaster(forecast_horizons=forecast_horizons)
    ensemble_preds = ensemble.combine(
        tft_preds, nbeats_preds, patchtst_preds,
        regime=current_regime.value
    )
    model_contributions = ensemble.get_model_contributions(regime=current_regime.value)
    logger.info(f"[Deep Learning Agent] Ensemble predictions: {ensemble_preds}")

    # --- Memory ---
    log_msg = (
        f"DL Agent complete: regime={current_regime.value}, "
        f"ensemble_1d={ensemble_preds.get(1, 0.0):.6f}, "
        f"changepoints={len(changepoints)}"
    )
    run_sync(_memory_engine.add_episode_log(run_id, "deep_learning_agent", "INFO", log_msg))

    # Serialize predictions to JSON-safe format
    def _serialize_preds(preds):
        return {str(k): float(v) if not isinstance(v, np.ndarray) else float(v.mean()) for k, v in preds.items()}

    return {
        "tft_predictions": _serialize_preds(tft_preds),
        "nbeats_predictions": _serialize_preds(nbeats_preds),
        "patchtst_predictions": _serialize_preds(patchtst_preds),
        "ensemble_predictions": _serialize_preds(ensemble_preds),
        "current_regime": current_regime.value,
        "regime_probabilities": regime_probs,
        "model_contributions": model_contributions,
        "changepoints": changepoints[:10],  # Limit for state size
        "current_node": "deep_learning_agent",
        "agent_logs": state.get("agent_logs", []) + [
            f"🧠 Deep Learning Agent: Regime={current_regime.value} | "
            f"Models=[TFT, N-BEATS, PatchTST] | "
            f"Ensemble 1d={ensemble_preds.get(1, 0.0):.6f}, "
            f"5d={ensemble_preds.get(5, 0.0):.6f}, "
            f"20d={ensemble_preds.get(20, 0.0):.6f}",
            f"📊 Model weights: {model_contributions}",
        ],
    }
