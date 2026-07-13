"""
AlphaLens — Temporal Fusion Transformer (TFT)
Multi-horizon forecasting with variable selection, interpretable attention,
and gated residual networks for quantitative finance.

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-Horizon Time Series Forecasting" (2021)
"""

import math
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. TFT will use NumPy fallback.")


# ---------------------------------------------------------------------------
# PyTorch Components
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class GatedLinearUnit(nn.Module):
        """GLU activation: sigmoid(Wx + b) ⊙ (Vx + c)"""
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
            self.gate = nn.Linear(input_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.gate(x)) * self.fc(x)


    class GatedResidualNetwork(nn.Module):
        """
        GRN: applies ELU nonlinearity, optional context, GLU gating,
        layer norm, and a residual skip connection.
        """
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     context_dim: int = 0, dropout: float = 0.1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
            self.glu = GatedLinearUnit(hidden_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)
            self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

        def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
            residual = self.skip(x) if self.skip else x
            h = F.elu(self.fc1(x))
            if self.context_proj is not None and context is not None:
                h = h + self.context_proj(context)
            h = F.elu(self.fc2(h))
            h = self.dropout(self.glu(h))
            return self.layer_norm(h + residual)


    class VariableSelectionNetwork(nn.Module):
        """
        VSN: learns which input features matter via softmax variable weights.
        Produces a weighted combination of per-variable GRN outputs.
        """
        def __init__(self, n_features: int, hidden_dim: int, dropout: float = 0.1):
            super().__init__()
            self.n_features = n_features
            self.hidden_dim = hidden_dim
            self.feature_grns = nn.ModuleList([
                GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(n_features)
            ])
            self.weight_grn = GatedResidualNetwork(
                n_features, hidden_dim, n_features, dropout=dropout
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # x shape: (batch, seq_len, n_features)
            weights = self.softmax(self.weight_grn(x))  # (batch, seq_len, n_features)
            
            processed = []
            for i in range(self.n_features):
                feat = x[..., i:i+1]  # (batch, seq_len, 1)
                processed.append(self.feature_grns[i](feat))
            
            processed = torch.stack(processed, dim=-2)  # (batch, seq_len, n_features, hidden)
            weights_expanded = weights.unsqueeze(-1)      # (batch, seq_len, n_features, 1)
            combined = (processed * weights_expanded).sum(dim=-2)  # (batch, seq_len, hidden)
            
            return combined, weights


    class InterpretableMultiHeadAttention(nn.Module):
        """
        Interpretable multi-head attention that shares value weights across heads
        to produce a single attention matrix for interpretability.
        """
        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
            super().__init__()
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, self.d_k)  # Shared V
            self.W_o = nn.Linear(self.d_k, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size, seq_len, _ = q.shape

            Q = self.W_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(v)  # (batch, seq_len, d_k) — shared across heads

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Average attention across heads for interpretability
            attn_avg = attn_weights.mean(dim=1)  # (batch, seq_len, seq_len)
            context = torch.matmul(attn_avg, V)   # (batch, seq_len, d_k)
            output = self.W_o(context)

            return output, attn_avg


    class TemporalFusionTransformer(nn.Module):
        """
        Full TFT model for multi-horizon time series forecasting.
        """
        def __init__(self, n_features: int, hidden_dim: int = 64,
                     n_heads: int = 4, n_layers: int = 1,
                     forecast_horizons: List[int] = None,
                     dropout: float = 0.1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.forecast_horizons = forecast_horizons or [1, 5, 20]

            # Variable Selection
            self.vsn = VariableSelectionNetwork(n_features, hidden_dim, dropout)

            # Temporal Processing (LSTM encoder-decoder)
            self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

            # Static enrichment GRN
            self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

            # Temporal self-attention
            self.attention = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)

            # Post-attention GRN
            self.post_attention_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            self.post_attention_norm = nn.LayerNorm(hidden_dim)

            # Output heads — one per forecast horizon
            self.output_heads = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in self.forecast_horizons
            ])

        def forward(self, x: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (batch, seq_len, n_features)
            Returns:
                predictions: dict mapping horizon -> (batch, seq_len, 1)
                attention_weights: (batch, seq_len, seq_len)
                variable_weights: (batch, seq_len, n_features)
            """
            # 1. Variable Selection
            selected, var_weights = self.vsn(x)

            # 2. LSTM Encoder
            encoded, (h_n, c_n) = self.encoder_lstm(selected)

            # 3. Static Enrichment
            enriched = self.static_enrichment(encoded)

            # 4. Temporal Self-Attention with causal mask
            seq_len = enriched.shape[1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=enriched.device)).unsqueeze(0)
            attn_out, attn_weights = self.attention(enriched, enriched, enriched, mask=causal_mask)

            # 5. Post-attention processing
            post_attn = self.post_attention_norm(attn_out + enriched)
            post_attn = self.post_attention_grn(post_attn)

            # 6. Multi-horizon output
            predictions = {}
            for i, horizon in enumerate(self.forecast_horizons):
                predictions[horizon] = self.output_heads[i](post_attn)

            return predictions, attn_weights, var_weights


# ---------------------------------------------------------------------------
# High-Level Forecaster API (works with or without PyTorch)
# ---------------------------------------------------------------------------

class TFTForecaster:
    """
    High-level API for Temporal Fusion Transformer forecasting.
    Falls back to exponential smoothing if PyTorch is unavailable.
    """
    def __init__(self, n_features: int = 10, hidden_dim: int = 64,
                 n_heads: int = 4, forecast_horizons: List[int] = None,
                 learning_rate: float = 1e-3, epochs: int = 50,
                 dropout: float = 0.1):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.forecast_horizons = forecast_horizons or [1, 5, 20]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.is_fitted = False
        self._attention_weights = None
        self._variable_weights = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "TFTForecaster":
        """
        Trains the TFT model.
        
        Args:
            features: (n_samples, n_features) or (n_samples, seq_len, n_features)
            targets: (n_samples,) forward returns
        """
        if not TORCH_AVAILABLE:
            logger.info("TFT: PyTorch unavailable. Using exponential smoothing fallback.")
            self._fit_fallback(features, targets)
            return self

        # Reshape to (batch, seq_len, n_features) if needed
        if features.ndim == 2:
            seq_len = min(60, len(features))
            X, y = self._create_sequences(features, targets, seq_len)
        else:
            X, y = features, targets

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_feat = X.shape[-1]

        self.model = TemporalFusionTransformer(
            n_features=n_feat,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            forecast_horizons=self.forecast_horizons,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds, attn_w, var_w = self.model(X_tensor)
            # Use horizon=1 predictions at last timestep
            pred_1d = preds[self.forecast_horizons[0]][:, -1, 0]
            loss = criterion(pred_1d, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"TFT Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.6f}")

        self.is_fitted = True
        self._attention_weights = attn_w.detach().cpu().numpy()
        self._variable_weights = var_w.detach().cpu().numpy()
        logger.info("TFT training complete.")
        return self

    def predict(self, features: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Generates multi-horizon predictions.
        Returns dict mapping horizon -> predictions array.
        """
        if not TORCH_AVAILABLE or self.model is None:
            return self._predict_fallback(features)

        device = next(self.model.parameters()).device

        if features.ndim == 2:
            features = features[np.newaxis, :, :]

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features).to(device)
            preds, attn_w, var_w = self.model(X_tensor)
            self._attention_weights = attn_w.cpu().numpy()
            self._variable_weights = var_w.cpu().numpy()

        result = {}
        for h in self.forecast_horizons:
            result[h] = preds[h].cpu().numpy().squeeze()
        return result

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Returns the last computed temporal attention weights."""
        return self._attention_weights

    def get_variable_importance(self) -> Optional[np.ndarray]:
        """Returns the last computed variable selection weights."""
        return self._variable_weights

    # --- Internal helpers ---

    def _create_sequences(self, features: np.ndarray, targets: np.ndarray,
                          seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(seq_len, len(features)):
            X.append(features[i - seq_len:i])
            y.append(targets[i])
        return np.array(X), np.array(y)

    def _fit_fallback(self, features: np.ndarray, targets: np.ndarray):
        """Simple exponential smoothing fallback when PyTorch is unavailable."""
        self._fallback_alpha = 0.3
        if features.ndim == 2:
            self._fallback_coefs = np.zeros(features.shape[1])
            n = len(targets)
            if n > 10:
                # Simple OLS regression
                X = features[-n:]
                y = targets[-n:]
                try:
                    self._fallback_coefs = np.linalg.lstsq(X, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        self.is_fitted = True

    def _predict_fallback(self, features: np.ndarray) -> Dict[int, np.ndarray]:
        """Fallback prediction using linear regression."""
        if features.ndim == 3:
            features = features[:, -1, :]
        if hasattr(self, "_fallback_coefs"):
            pred = features @ self._fallback_coefs
        else:
            pred = np.zeros(len(features))
        return {h: pred * math.sqrt(h) for h in self.forecast_horizons}
