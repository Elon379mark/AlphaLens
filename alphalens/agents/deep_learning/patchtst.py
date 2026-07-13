"""
AlphaLens — PatchTST (Patch Time Series Transformer)
Channel-independent patching strategy for long-horizon multivariate
time series forecasting in quantitative finance.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers" (2023)
"""

import math
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PatchTST will use NumPy fallback.")


# ---------------------------------------------------------------------------
# PyTorch Components
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class PatchEmbedding(nn.Module):
        """Splits time series into non-overlapping patches and embeds them."""
        def __init__(self, patch_len: int, d_model: int, stride: int = None):
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride or patch_len
            self.projection = nn.Linear(patch_len, d_model)
            self.layer_norm = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len) — single channel
            Returns:
                patches: (batch, n_patches, d_model)
            """
            batch_size, seq_len = x.shape
            # Unfold into patches
            n_patches = (seq_len - self.patch_len) // self.stride + 1
            patches = []
            for i in range(n_patches):
                start = i * self.stride
                patches.append(x[:, start:start + self.patch_len])
            patches = torch.stack(patches, dim=1)  # (batch, n_patches, patch_len)
            embedded = self.layer_norm(self.projection(patches))
            return embedded


    class PositionalEncoding(nn.Module):
        """Learnable positional encoding for patch positions."""
        def __init__(self, d_model: int, max_patches: int = 512):
            super().__init__()
            self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            n_patches = x.shape[1]
            return x + self.pos_embedding[:, :n_patches, :]


    class TransformerEncoderBlock(nn.Module):
        """Standard Transformer encoder block with pre-norm."""
        def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 256,
                     dropout: float = 0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pre-norm self-attention
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + self.dropout(attn_out)
            # Pre-norm feedforward
            x = x + self.ff(self.norm2(x))
            return x


    class PatchTSTModel(nn.Module):
        """
        Full PatchTST model for multivariate time series forecasting.
        Uses channel-independent processing (each feature processed independently).
        """
        def __init__(self, n_features: int, seq_len: int,
                     patch_len: int = 16, d_model: int = 64,
                     n_heads: int = 4, n_layers: int = 3,
                     forecast_len: int = 1, dropout: float = 0.1):
            super().__init__()
            self.n_features = n_features
            self.seq_len = seq_len
            self.forecast_len = forecast_len

            stride = patch_len
            n_patches = (seq_len - patch_len) // stride + 1

            # Per-channel patch embedding
            self.patch_embedding = PatchEmbedding(patch_len, d_model, stride)
            self.pos_encoding = PositionalEncoding(d_model, max_patches=n_patches + 1)

            # Shared Transformer encoder
            self.encoder = nn.Sequential(*[
                TransformerEncoderBlock(d_model, n_heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ])

            # Flatten and project to forecast
            self.head_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(n_patches * d_model, forecast_len)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, n_features)
            Returns:
                forecast: (batch, forecast_len, n_features)
            """
            batch_size = x.shape[0]
            forecasts = []

            # Channel-independent processing
            for c in range(self.n_features):
                channel = x[:, :, c]  # (batch, seq_len)
                patches = self.patch_embedding(channel)
                patches = self.pos_encoding(patches)
                encoded = self.encoder(patches)
                encoded = self.head_norm(encoded)
                flat = encoded.reshape(batch_size, -1)
                flat = self.dropout(flat)
                pred = self.head(flat)  # (batch, forecast_len)
                forecasts.append(pred)

            # Stack channels
            forecast = torch.stack(forecasts, dim=-1)  # (batch, forecast_len, n_features)
            return forecast


# ---------------------------------------------------------------------------
# High-Level Forecaster API
# ---------------------------------------------------------------------------

class PatchTSTForecaster:
    """
    High-level API for PatchTST forecasting.
    Falls back to AR(1) model if PyTorch is unavailable.
    """
    def __init__(self, seq_len: int = 60, patch_len: int = 16,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 forecast_horizons: List[int] = None,
                 learning_rate: float = 1e-3, epochs: int = 50,
                 dropout: float = 0.1):
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.forecast_horizons = forecast_horizons or [1, 5, 20]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.models = {}  # One model per horizon
        self.is_fitted = False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "PatchTSTForecaster":
        """
        Trains PatchTST models for each forecast horizon.
        
        Args:
            features: (n_samples, n_features)
            targets: (n_samples,) forward returns
        """
        if not TORCH_AVAILABLE:
            logger.info("PatchTST: PyTorch unavailable. Using AR fallback.")
            self._fit_fallback(features, targets)
            return self

        n_features = features.shape[1] if features.ndim > 1 else 1
        X, y = self._create_sequences(features, targets, self.seq_len)

        if len(X) < 5:
            logger.warning("PatchTST: Insufficient data. Using fallback.")
            self._fit_fallback(features, targets)
            return self

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train one model for all horizons (using horizon 1 target for simplicity)
        model = PatchTSTModel(
            n_features=n_features,
            seq_len=self.seq_len,
            patch_len=min(self.patch_len, self.seq_len // 2),
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            forecast_len=1,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            forecast = model(X_tensor)  # (batch, 1, n_features)
            pred = forecast[:, 0, 0]    # First feature, first forecast step
            loss = criterion(pred, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"PatchTST Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.6f}")

        self.models["main"] = model
        self.is_fitted = True
        logger.info("PatchTST training complete.")
        return self

    def predict(self, features: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Generates multi-horizon predictions.
        Returns dict mapping horizon -> predictions.
        """
        if not TORCH_AVAILABLE or not self.models:
            return self._predict_fallback(features)

        model = self.models["main"]
        device = next(model.parameters()).device

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # Take last seq_len samples
        x = features[-self.seq_len:]
        if len(x) < self.seq_len:
            pad = np.zeros((self.seq_len - len(x), x.shape[1]))
            x = np.vstack([pad, x])
        x = x[np.newaxis, :, :]  # (1, seq_len, n_features)

        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(x).to(device)
            forecast = model(X_tensor)

        pred = forecast.cpu().numpy().flatten()[0]
        result = {}
        for h in self.forecast_horizons:
            result[h] = pred * math.sqrt(h)
        return result

    # --- Internal helpers ---

    def _create_sequences(self, features: np.ndarray, targets: np.ndarray,
                          seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        X, y = [], []
        for i in range(seq_len, len(features)):
            X.append(features[i - seq_len:i])
            y.append(targets[i])
        return np.array(X), np.array(y)

    def _fit_fallback(self, features: np.ndarray, targets: np.ndarray):
        """AR(1) fallback."""
        if len(targets) > 1:
            self._ar_coef = np.corrcoef(targets[:-1], targets[1:])[0, 1]
            self._last_target = targets[-1]
        else:
            self._ar_coef = 0.0
            self._last_target = 0.0
        self.is_fitted = True

    def _predict_fallback(self, features: np.ndarray) -> Dict[int, np.ndarray]:
        coef = getattr(self, "_ar_coef", 0.0)
        last = getattr(self, "_last_target", 0.0)
        return {h: last * (coef ** h) for h in self.forecast_horizons}
