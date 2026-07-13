"""
AlphaLens — N-BEATS (Neural Basis Expansion Analysis for Time Series)
Stack-based architecture with trend and seasonality blocks for
interpretable multi-horizon financial time series forecasting.

Reference: Oreshkin et al., "N-BEATS: Neural Basis Expansion Analysis
for Interpretable Time Series Forecasting" (2020)
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
    logger.warning("PyTorch not available. N-BEATS will use NumPy fallback.")


# ---------------------------------------------------------------------------
# PyTorch Components
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class NBeatsBlock(nn.Module):
        """
        Basic N-BEATS block: FC stack → basis expansion for backcast/forecast.
        """
        def __init__(self, input_dim: int, theta_dim: int, hidden_dim: int = 256,
                     n_layers: int = 4, basis_type: str = "generic"):
            super().__init__()
            self.basis_type = basis_type

            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            self.fc_stack = nn.Sequential(*layers)

            self.theta_b = nn.Linear(hidden_dim, theta_dim)  # backcast coefficients
            self.theta_f = nn.Linear(hidden_dim, theta_dim)  # forecast coefficients

            if basis_type == "trend":
                # Polynomial basis
                self.backcast_basis = self._make_polynomial_basis(input_dim, theta_dim)
                self.forecast_basis = self._make_polynomial_basis(input_dim, theta_dim)
            elif basis_type == "seasonality":
                # Fourier basis
                self.backcast_basis = self._make_fourier_basis(input_dim, theta_dim)
                self.forecast_basis = self._make_fourier_basis(input_dim, theta_dim)

        def _make_polynomial_basis(self, length: int, degree: int) -> torch.Tensor:
            t = torch.linspace(0, 1, length)
            basis = torch.stack([t ** i for i in range(degree)], dim=0)  # (degree, length)
            return nn.Parameter(basis, requires_grad=False)

        def _make_fourier_basis(self, length: int, n_harmonics: int) -> torch.Tensor:
            t = torch.linspace(0, 2 * math.pi, length)
            basis = []
            for k in range(1, n_harmonics // 2 + 1):
                basis.append(torch.sin(k * t))
                basis.append(torch.cos(k * t))
            if len(basis) < n_harmonics:
                basis.append(torch.ones(length))
            basis = torch.stack(basis[:n_harmonics], dim=0)
            return nn.Parameter(basis, requires_grad=False)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.fc_stack(x)
            theta_b = self.theta_b(h)
            theta_f = self.theta_f(h)

            if self.basis_type == "generic":
                backcast = theta_b
                forecast = theta_f
            else:
                basis_b = self.backcast_basis if hasattr(self, "backcast_basis") else None
                basis_f = self.forecast_basis if hasattr(self, "forecast_basis") else None
                if basis_b is not None:
                    backcast = torch.matmul(theta_b, basis_b)
                    forecast = torch.matmul(theta_f, basis_f)
                else:
                    backcast = theta_b
                    forecast = theta_f

            return backcast, forecast


    class NBeatsStack(nn.Module):
        """Stack of N-BEATS blocks sharing the same basis type."""
        def __init__(self, n_blocks: int, input_dim: int, theta_dim: int,
                     hidden_dim: int = 256, basis_type: str = "generic"):
            super().__init__()
            self.blocks = nn.ModuleList([
                NBeatsBlock(input_dim, theta_dim, hidden_dim, basis_type=basis_type)
                for _ in range(n_blocks)
            ])

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            residual = x
            stack_forecast = torch.zeros_like(x)

            for block in self.blocks:
                backcast, forecast = block(residual)
                # Pad/truncate to match dimensions
                if backcast.shape[-1] != residual.shape[-1]:
                    backcast = F.pad(backcast, (0, residual.shape[-1] - backcast.shape[-1]))
                if forecast.shape[-1] != stack_forecast.shape[-1]:
                    forecast = F.pad(forecast, (0, stack_forecast.shape[-1] - forecast.shape[-1]))
                residual = residual - backcast
                stack_forecast = stack_forecast + forecast

            return residual, stack_forecast


    class NBeatsModel(nn.Module):
        """
        Full N-BEATS model: Trend stack + Seasonality stack + Generic stack.
        """
        def __init__(self, input_dim: int, hidden_dim: int = 256,
                     n_trend_blocks: int = 3, n_seasonal_blocks: int = 3,
                     n_generic_blocks: int = 3, theta_dim: int = 8):
            super().__init__()
            self.trend_stack = NBeatsStack(
                n_trend_blocks, input_dim, theta_dim, hidden_dim, "trend"
            )
            self.seasonality_stack = NBeatsStack(
                n_seasonal_blocks, input_dim, theta_dim, hidden_dim, "seasonality"
            )
            self.generic_stack = NBeatsStack(
                n_generic_blocks, input_dim, theta_dim, hidden_dim, "generic"
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual, trend_forecast = self.trend_stack(x)
            residual, seasonal_forecast = self.seasonality_stack(residual)
            _, generic_forecast = self.generic_stack(residual)

            total_forecast = trend_forecast + seasonal_forecast + generic_forecast
            return total_forecast


# ---------------------------------------------------------------------------
# High-Level Forecaster API
# ---------------------------------------------------------------------------

class NBeatsForecaster:
    """
    High-level API for N-BEATS forecasting.
    Falls back to decomposition-based prediction if PyTorch is unavailable.
    """
    def __init__(self, lookback: int = 60, hidden_dim: int = 256,
                 forecast_horizons: List[int] = None,
                 learning_rate: float = 1e-3, epochs: int = 50):
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.forecast_horizons = forecast_horizons or [1, 5, 20]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.is_fitted = False

    def fit(self, series: np.ndarray, targets: Optional[np.ndarray] = None) -> "NBeatsForecaster":
        """
        Trains the N-BEATS model on a univariate or multivariate time series.
        
        Args:
            series: (n_samples,) univariate or (n_samples, n_features) multivariate
            targets: optional separate target array
        """
        if not TORCH_AVAILABLE:
            logger.info("N-BEATS: PyTorch unavailable. Using decomposition fallback.")
            self._fit_fallback(series, targets)
            return self

        # Univariate handling
        if series.ndim == 1:
            X, y = self._create_windows(series, self.lookback)
        else:
            # Use first column as target, flatten features into lookback window
            flat = series[:, 0] if series.shape[1] > 0 else series.flatten()
            X, y = self._create_windows(flat, self.lookback)

        if len(X) < 5:
            logger.warning("N-BEATS: Insufficient data for training.")
            self._fit_fallback(series, targets)
            return self

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X.shape[1]

        self.model = NBeatsModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            forecast = self.model(X_tensor)
            # Take the last value of forecast as prediction
            pred = forecast[:, -1] if forecast.ndim > 1 else forecast
            loss = criterion(pred, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"N-BEATS Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.6f}")

        self.is_fitted = True
        logger.info("N-BEATS training complete.")
        return self

    def predict(self, series: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Generates multi-horizon point forecasts.
        Returns dict mapping horizon -> predictions.
        """
        if not TORCH_AVAILABLE or self.model is None:
            return self._predict_fallback(series)

        device = next(self.model.parameters()).device

        if series.ndim == 1:
            x = series[-self.lookback:].reshape(1, -1)
        else:
            x = series[-self.lookback:, 0].reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(x).to(device)
            forecast = self.model(X_tensor)

        forecast_np = forecast.cpu().numpy().flatten()
        result = {}
        for h in self.forecast_horizons:
            idx = min(h - 1, len(forecast_np) - 1)
            result[h] = forecast_np[idx] if idx >= 0 else 0.0
        return result

    # --- Internal helpers ---

    def _create_windows(self, series: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(lookback, len(series)):
            X.append(series[i - lookback:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def _fit_fallback(self, series: np.ndarray, targets: Optional[np.ndarray]):
        """Simple moving average fallback."""
        if series.ndim > 1:
            series = series[:, 0]
        self._last_values = series[-self.lookback:] if len(series) >= self.lookback else series
        self.is_fitted = True

    def _predict_fallback(self, series: np.ndarray) -> Dict[int, np.ndarray]:
        if series.ndim > 1:
            series = series[:, 0]
        last_val = series[-1] if len(series) > 0 else 0.0
        # Simple drift model
        if len(series) > 1:
            drift = (series[-1] - series[0]) / len(series)
        else:
            drift = 0.0
        return {h: last_val + drift * h for h in self.forecast_horizons}
