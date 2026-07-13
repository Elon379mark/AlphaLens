"""
AlphaLens — Interpretability & Explainability Layer
SHAP-based feature importance, causal attribution reports,
and human-readable explanations for regulatory compliance.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Explainer will use permutation importance fallback.")


class ExplainabilityReport:
    """
    Structured explainability report for a single pipeline run.
    """
    def __init__(self):
        self.feature_importances: Dict[str, float] = {}
        self.top_features: List[str] = []
        self.causal_attributions: Dict[str, Dict[str, Any]] = {}
        self.decision_narrative: str = ""
        self.regime_context: str = ""
        self.risk_warnings: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_importances": self.feature_importances,
            "top_features": self.top_features,
            "causal_attributions": self.causal_attributions,
            "decision_narrative": self.decision_narrative,
            "regime_context": self.regime_context,
            "risk_warnings": self.risk_warnings,
        }

    def to_narrative(self) -> str:
        """Generates a human-readable explanation."""
        lines = []
        lines.append("=" * 60)
        lines.append("ALPHALENS EXPLAINABILITY REPORT")
        lines.append("=" * 60)

        if self.regime_context:
            lines.append(f"\nMarket Regime: {self.regime_context}")

        if self.top_features:
            lines.append(f"\nTop Contributing Features:")
            for i, feat in enumerate(self.top_features[:10], 1):
                importance = self.feature_importances.get(feat, 0.0)
                lines.append(f"  {i}. {feat}: {importance:.4f}")

        if self.causal_attributions:
            lines.append(f"\nCausal Attribution:")
            for signal, attrs in self.causal_attributions.items():
                lines.append(f"  Signal: {signal}")
                lines.append(f"    ATE: {attrs.get('ate', 'N/A')}")
                lines.append(f"    p-value: {attrs.get('p_value', 'N/A')}")
                lines.append(f"    Causal link: {attrs.get('causal_link', 'N/A')}")

        if self.decision_narrative:
            lines.append(f"\nDecision Narrative:")
            lines.append(f"  {self.decision_narrative}")

        if self.risk_warnings:
            lines.append(f"\nRisk Warnings:")
            for warn in self.risk_warnings:
                lines.append(f"  ⚠️ {warn}")

        lines.append("=" * 60)
        return "\n".join(lines)


class Explainer:
    """
    Generates SHAP-based feature importance and causal attribution
    reports for AlphaLens pipeline decisions.
    """
    def __init__(self, n_background_samples: int = 100):
        self.n_background_samples = n_background_samples

    def compute_shap_importance(self,
                                model_predict_fn,
                                features: np.ndarray,
                                feature_names: List[str]) -> Dict[str, float]:
        """
        Computes SHAP values for a model's predictions.
        
        Args:
            model_predict_fn: callable that takes (n_samples, n_features) -> (n_samples,)
            features: (n_samples, n_features) input data
            feature_names: list of feature name strings
        
        Returns:
            Dict mapping feature_name -> mean absolute SHAP value
        """
        if SHAP_AVAILABLE:
            try:
                # Use KernelExplainer for model-agnostic SHAP
                n_bg = min(self.n_background_samples, len(features))
                background = features[:n_bg]
                explainer = shap.KernelExplainer(model_predict_fn, background)

                n_explain = min(50, len(features))
                shap_values = explainer.shap_values(features[-n_explain:])

                # Mean absolute SHAP values per feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                importance = {}
                for i, name in enumerate(feature_names):
                    if i < len(mean_abs_shap):
                        importance[name] = float(mean_abs_shap[i])

                logger.info("SHAP importance computed successfully.")
                return importance

            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}. Using permutation fallback.")

        # Fallback: permutation importance
        return self._permutation_importance(model_predict_fn, features, feature_names)

    def compute_linear_importance(self,
                                  features: np.ndarray,
                                  targets: np.ndarray,
                                  feature_names: List[str]) -> Dict[str, float]:
        """
        Computes feature importance using correlation-based attribution.
        Model-free approach suitable when no trained model is available.
        
        Args:
            features: (n_samples, n_features)
            targets: (n_samples,) target returns
            feature_names: list of feature name strings
        
        Returns:
            Dict mapping feature_name -> absolute correlation
        """
        importance = {}
        for i, name in enumerate(feature_names):
            if i < features.shape[1]:
                feat = features[:, i]
                # Remove NaN/Inf
                valid = np.isfinite(feat) & np.isfinite(targets)
                if valid.sum() > 5:
                    corr = np.corrcoef(feat[valid], targets[valid])[0, 1]
                    importance[name] = abs(float(corr)) if not np.isnan(corr) else 0.0
                else:
                    importance[name] = 0.0

        return importance

    def build_report(self,
                     feature_importances: Dict[str, float],
                     causal_results: Optional[Dict[str, Any]] = None,
                     regime: Optional[str] = None,
                     portfolio_weights: Optional[Dict[str, float]] = None,
                     sharpe_ratio: Optional[float] = None,
                     p_value: Optional[float] = None) -> ExplainabilityReport:
        """
        Builds a complete ExplainabilityReport from pipeline results.
        
        Args:
            feature_importances: feature -> importance score
            causal_results: causal validation outputs
            regime: current market regime string
            portfolio_weights: asset -> weight mapping
            sharpe_ratio: backtest Sharpe ratio
            p_value: causal p-value
        """
        report = ExplainabilityReport()

        # Feature importances
        report.feature_importances = feature_importances
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        report.top_features = [f[0] for f in sorted_features[:10]]

        # Causal attribution
        if causal_results:
            report.causal_attributions = {"primary_signal": causal_results}

        # Regime context
        if regime:
            regime_descriptions = {
                "bull": "Bull market regime detected — trending conditions favor momentum signals",
                "bear": "Bear market regime detected — defensive positioning recommended",
                "high_vol": "High volatility regime — increased position sizing risk, wider stops advised",
            }
            report.regime_context = regime_descriptions.get(regime, f"Regime: {regime}")

        # Decision narrative
        narrative_parts = []
        if report.top_features:
            top3 = ", ".join(report.top_features[:3])
            narrative_parts.append(f"The primary drivers of this signal are: {top3}.")

        if p_value is not None:
            if p_value < 0.05:
                narrative_parts.append(f"The signal shows statistically significant causal evidence (p={p_value:.4f}).")
            else:
                narrative_parts.append(f"Causal evidence is weak (p={p_value:.4f}), suggesting potential spurious correlation.")

        if sharpe_ratio is not None:
            if sharpe_ratio >= 1.0:
                narrative_parts.append(f"Backtested Sharpe ratio of {sharpe_ratio:.2f} indicates acceptable risk-adjusted returns.")
            else:
                narrative_parts.append(f"Backtested Sharpe ratio of {sharpe_ratio:.2f} is below the 1.0 threshold.")

        if portfolio_weights:
            max_weight_asset = max(portfolio_weights, key=portfolio_weights.get)
            narrative_parts.append(f"Highest allocation goes to {max_weight_asset} ({portfolio_weights[max_weight_asset]:.1%}).")

        report.decision_narrative = " ".join(narrative_parts)

        # Risk warnings
        if p_value is not None and p_value >= 0.05:
            report.risk_warnings.append("Signal failed causal significance test. Higher risk of strategy decay.")

        if sharpe_ratio is not None and sharpe_ratio < 1.0:
            report.risk_warnings.append("Risk-adjusted return below institutional threshold.")

        if regime == "high_vol":
            report.risk_warnings.append("High volatility regime increases drawdown risk and slippage costs.")

        if feature_importances:
            top_importance = max(feature_importances.values())
            if top_importance > 0.5:
                top_feat = max(feature_importances, key=feature_importances.get)
                report.risk_warnings.append(
                    f"Heavy concentration on single feature '{top_feat}' ({top_importance:.2f}). "
                    f"Strategy may lack diversification."
                )

        return report

    # --- Internal helpers ---

    def _permutation_importance(self,
                                model_predict_fn,
                                features: np.ndarray,
                                feature_names: List[str],
                                n_repeats: int = 5) -> Dict[str, float]:
        """Fallback permutation importance when SHAP is unavailable."""
        n_samples, n_features = features.shape
        base_pred = model_predict_fn(features)
        base_metric = float(np.var(base_pred))

        importance = {}
        rng = np.random.default_rng(42)

        for i, name in enumerate(feature_names):
            if i >= n_features:
                importance[name] = 0.0
                continue

            scores = []
            for _ in range(n_repeats):
                permuted = features.copy()
                permuted[:, i] = rng.permutation(permuted[:, i])
                perm_pred = model_predict_fn(permuted)
                perm_metric = float(np.var(perm_pred))
                scores.append(abs(base_metric - perm_metric))

            importance[name] = float(np.mean(scores))

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance
