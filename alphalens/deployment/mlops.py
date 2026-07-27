"""
MLOps Tracking & Experiment Management Module (Section 12 & 15).
Provides standardized MLflow experiment tracking, metric logging, model registry,
and decision-provenance artifact logging across all 5 agents in the AlphaLens pipeline.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow = None


class MLflowTracker:
    """
    Production MLOps tracker interfacing with MLflow server.
    Logs hypothesis parameters, statistical metrics, causal DAGs, backtest equity curves,
    portfolio risk attribution, and model artifacts.
    """
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "AlphaLens_Research"):
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.experiment_name = experiment_name
        self.active_run = None

        if _MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"[MLflowTracker] Initialized tracking at {self.tracking_uri} for experiment '{self.experiment_name}'")
            except Exception as e:
                logger.warning(f"[MLflowTracker] Failed to connect to MLflow server at {self.tracking_uri}: {e}. Local logging enabled.")

    def start_run(self, run_name: str, nested: bool = True) -> Optional[Any]:
        """Starts a new MLflow run or nested run."""
        if not _MLFLOW_AVAILABLE:
            return None
        try:
            self.active_run = mlflow.start_run(run_name=run_name, nested=nested)
            return self.active_run
        except Exception as e:
            logger.warning(f"[MLflowTracker] start_run failed: {e}")
            return None

    def end_run(self):
        """Ends active MLflow run."""
        if _MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass

    def log_hypothesis(self, hypothesis: Any):
        """Logs Literature Agent hypothesis metadata."""
        if not hypothesis:
            return
        params = {
            "hypothesis_id": getattr(hypothesis, "hypothesis_id", "unknown"),
            "predictor_variable": getattr(hypothesis, "predictor_variable", "unknown"),
            "target_asset_class": getattr(hypothesis, "target_asset_class", "unknown"),
            "predicted_direction": str(getattr(hypothesis, "predicted_direction", "positive")),
            "hypothesis_confidence": float(getattr(hypothesis, "confidence", 0.0)),
        }
        logger.info(f"[MLflowTracker] Hypothesis params: {params}")
        if _MLFLOW_AVAILABLE and mlflow.active_run():
            try:
                mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"[MLflowTracker] log_params failed: {e}")

    def log_signal_metrics(self, ic: float, icir: float, half_life: float, passes_gate: bool):
        """Logs Signal Generation Agent metrics."""
        metrics = {
            "information_coefficient": float(ic),
            "information_ratio": float(icir),
            "signal_half_life_days": float(half_life),
            "signal_passes_gate": 1.0 if passes_gate else 0.0,
        }
        logger.info(f"[MLflowTracker] Signal metrics: {metrics}")
        if _MLFLOW_AVAILABLE and mlflow.active_run():
            try:
                mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"[MLflowTracker] log_metrics failed: {e}")

    def log_causal_results(self, p_value: float, ate: float, rosenbaum_robust: bool, partial_r2: float, dag_path: Optional[str] = None):
        """Logs Causal Validation Agent results (DML ATE, sensitivity, HC3 SEs)."""
        metrics = {
            "causal_p_value": float(p_value),
            "dml_ate_magnitude": float(ate),
            "rosenbaum_robust": 1.0 if rosenbaum_robust else 0.0,
            "partial_r2": float(partial_r2),
            "causal_passes_gate": 1.0 if (p_value < 0.05 and rosenbaum_robust) else 0.0,
        }
        logger.info(f"[MLflowTracker] Causal metrics: {metrics}")
        if _MLFLOW_AVAILABLE and mlflow.active_run():
            try:
                mlflow.log_metrics(metrics)
                if dag_path and os.path.exists(dag_path):
                    mlflow.log_artifact(dag_path, artifact_path="causal_dags")
            except Exception as e:
                logger.warning(f"[MLflowTracker] log_causal_results failed: {e}")

    def log_backtest_results(self, sharpe: float, max_dd: float, total_ret: float, calmar: float = 0.0, info_ratio: float = 0.0):
        """Logs Backtesting Agent metrics."""
        metrics = {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "total_return": float(total_ret),
            "calmar_ratio": float(calmar),
            "information_ratio": float(info_ratio),
            "backtest_passes_gate": 1.0 if sharpe >= 1.0 else 0.0,
        }
        logger.info(f"[MLflowTracker] Backtest metrics: {metrics}")
        if _MLFLOW_AVAILABLE and mlflow.active_run():
            try:
                mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"[MLflowTracker] log_backtest_results failed: {e}")

    def log_portfolio_results(self, weights: List[float], asset_names: List[str], cvar_99: float, bhb_attribution: Optional[Dict[str, Any]] = None):
        """Logs Portfolio Agent weights, CVaR 99%, and BHB risk attribution."""
        metrics = {
            "portfolio_cvar_99": float(cvar_99),
        }
        if bhb_attribution:
            metrics["bhb_allocation_effect"] = float(bhb_attribution.get("total_allocation", 0.0))
            metrics["bhb_selection_effect"] = float(bhb_attribution.get("total_selection", 0.0))
            metrics["bhb_interaction_effect"] = float(bhb_attribution.get("total_interaction", 0.0))

        logger.info(f"[MLflowTracker] Portfolio metrics: {metrics}")
        if _MLFLOW_AVAILABLE and mlflow.active_run():
            try:
                mlflow.log_metrics(metrics)
                for name, w in zip(asset_names, weights):
                    mlflow.log_metric(f"weight_{name.replace(' ', '_')}", float(w))
            except Exception as e:
                logger.warning(f"[MLflowTracker] log_portfolio_results failed: {e}")
