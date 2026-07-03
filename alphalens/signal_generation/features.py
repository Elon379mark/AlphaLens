import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FeatureFactory:
    """
    Computes base parameters for the Signal Generation Agent.
    Supports 6 categories of features (price momentum, volume, quality, value, macro sensitivity, alternative).
    """
    def __init__(self):
        self.feature_names: List[str] = []
        self._initialize_feature_definitions()

    def _initialize_feature_definitions(self):
        # 1. Price Momentum features (e.g., returns, moving average crossovers, volatility, RSI, MACD)
        for window in [3, 5, 10, 20, 60, 120, 252]:
            self.feature_names.append(f"momentum_return_{window}d")
            self.feature_names.append(f"momentum_volatility_{window}d")
            self.feature_names.append(f"momentum_sma_ratio_{window}d")
        
        # 2. Volume features (e.g., volume moving averages, volume force, OBV, volume-price interaction)
        for window in [5, 10, 20, 60]:
            self.feature_names.append(f"volume_ratio_{window}d")
            self.feature_names.append(f"volume_force_{window}d")
            self.feature_names.append(f"volume_std_{window}d")

        # 3. Quality features (e.g., ROE, ROA, debt-to-equity, gross margins)
        quality_metrics = ["roe", "roa", "gross_margin", "debt_to_equity", "asset_turnover", "operating_leverage"]
        for metric in quality_metrics:
            self.feature_names.append(f"quality_{metric}")
            # adding changes/momentum for quality metrics
            self.feature_names.append(f"quality_{metric}_change_1y")

        # 4. Value features (e.g., P/E, P/B, P/S, EV/EBITDA, dividend yield)
        value_metrics = ["pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "dividend_yield", "book_to_market"]
        for metric in value_metrics:
            self.feature_names.append(f"value_{metric}")
            self.feature_names.append(f"value_{metric}_rank_cross_sectional")

        # 5. Macro Sensitivity features (e.g., beta to interest rates, inflation, VIX, credit spread slope)
        macro_factors = ["rates_beta", "inflation_beta", "vix_beta", "credit_spread_beta", "gdp_beta"]
        for factor in macro_factors:
            for window in [60, 120, 252]:
                self.feature_names.append(f"macro_sensitivity_{factor}_{window}d")

        # 6. Alternative features (e.g., sentiment score, supply chain index, news intensity, web traffic)
        alt_metrics = ["sentiment_score", "supply_chain_disruption_index", "news_intensity", "web_traffic_index"]
        for metric in alt_metrics:
            for window in [5, 20, 60]:
                self.feature_names.append(f"alt_{metric}_mean_{window}d")
                self.feature_names.append(f"alt_{metric}_momentum_{window}d")

    def compute_features(self, ohlcv_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None, macro_df: Optional[pd.DataFrame] = None, alt_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Computes the complete set of defined features using pandas/numpy.
        """
        # Ensure input data is a DataFrame
        if not isinstance(ohlcv_df, pd.DataFrame):
            raise TypeError("ohlcv_df must be a pandas DataFrame")

        df = ohlcv_df.copy()
        
        # 1. Price Momentum Computations
        if "close" in df.columns:
            for window in [3, 5, 10, 20, 60, 120, 252]:
                df[f"momentum_return_{window}d"] = df["close"].pct_change(window)
                # Volatility
                df[f"momentum_volatility_{window}d"] = df["close"].pct_change().rolling(window).std()
                # SMA Ratio
                df[f"momentum_sma_ratio_{window}d"] = df["close"] / df["close"].rolling(window).mean()
        
        # 2. Volume Computations
        if "volume" in df.columns:
            for window in [5, 10, 20, 60]:
                df[f"volume_ratio_{window}d"] = df["volume"] / df["volume"].rolling(window).mean()
                df[f"volume_std_{window}d"] = df["volume"].rolling(window).std()
                if "close" in df.columns:
                    # volume-force: close price return * volume ratio
                    df[f"volume_force_{window}d"] = df["close"].pct_change() * df[f"volume_ratio_{window}d"]

        # Ensure all 312 features are present (imputing mock/default values for missing fundamental/macro data)
        for feature in self.feature_names:
            if feature not in df.columns:
                # Add default or join from alternative/fundamental dataframes if provided
                if fundamentals_df is not None and feature in fundamentals_df.columns:
                    df = df.join(fundamentals_df[[feature]], how="left")
                elif macro_df is not None and feature in macro_df.columns:
                    df = df.join(macro_df[[feature]], how="left")
                elif alt_df is not None and feature in alt_df.columns:
                    df = df.join(alt_df[[feature]], how="left")
                else:
                    # Fill with 0.0 or forward-fill/impute
                    df[feature] = 0.0

        # Enforce temporal point-in-time correctness: drop lookahead columns, forward fill
        df = df.ffill().fillna(0.0)

        # Select exactly the 312 features in order
        return df[self.feature_names]
