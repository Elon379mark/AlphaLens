"""
Section 6.3: Preprocessing Pipeline
-----------------------------------
Implements the 5 sequential transformations specified in the research document:
1. Adjustment for Corporate Actions (assumed pre-applied in input adj_close).
2. Outlier Winsorisation: Clip features at {1%, 99%} percentiles on a rolling 252-day window.
3. Cross-Sectional Normalisation: Z-score each feature within each cross-sectional slice.
4. Missing Value Imputation: Forward-fill up to 5 days, then replace with the cross-sectional median.
5. Multicollinearity Screening: Deduplicate features with pairwise correlation |r| > 0.95
                               by retaining the feature with the higher absolute in-sample IC.
"""
import logging
from typing import Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def winsorize_series(s: pd.Series, window: int = 252) -> pd.Series:
    """Clips series values at the 1% and 99% rolling quantiles."""
    # Rolling quantiles computed per ticker to avoid mixing different assets
    q_low = s.rolling(window, min_periods=10).quantile(0.01)
    q_high = s.rolling(window, min_periods=10).quantile(0.99)
    # Fill initial NaN quantiles with the series quantiles
    q_low = q_low.fillna(s.quantile(0.01))
    q_high = q_high.fillna(s.quantile(0.99))
    return s.clip(lower=q_low, upper=q_high)


def winsorize_features(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Applies outlier winsorization on a rolling window per ticker."""
    logger.info("[Preprocessing] Applying outlier winsorization (1%, 99%) on a rolling 252-day window...")
    # Group by ticker and winsorize each feature
    return df.groupby(level="ticker", group_keys=False).apply(
        lambda x: x.apply(lambda col: winsorize_series(col, window))
    )


def cross_sectional_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    """Z-scores each feature cross-sectionally for each date slice."""
    logger.info("[Preprocessing] Applying cross-sectional Z-score normalisation...")
    # Subtract mean and divide by standard deviation for each date group
    def zscore(group):
        mean = group.mean()
        std = group.std().replace(0, 1.0)
        return (group - mean) / std
    return df.groupby(level="date", group_keys=False).apply(zscore)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes NaNs: forward-fills up to 5 days, then fills with cross-sectional median."""
    logger.info("[Preprocessing] Imputing missing values (5-day ffill + cross-sectional median)...")
    # Forward-fill per ticker
    df_filled = df.groupby(level="ticker", group_keys=False).apply(lambda x: x.ffill(limit=5))
    
    # Fill remaining NaNs with the cross-sectional median for that date
    def fill_median(group):
        median = group.median()
        return group.fillna(median)
        
    df_imputed = df_filled.groupby(level="date", group_keys=False).apply(fill_median)
    return df_imputed.fillna(0.0)  # Absolute fallback to zero if all assets have NaN


def screen_multicollinearity(df: pd.DataFrame, ic_dict: Dict[str, float], threshold: float = 0.95) -> pd.DataFrame:
    """Deduplicates collinear features (|r| > 0.95) keeping the one with higher absolute IC."""
    logger.info(f"[Preprocessing] Screening for multicollinearity (correlation threshold: {threshold})...")
    
    # Calculate pairwise correlation matrix
    corr_matrix = df.corr().abs()
    
    # Find pairs with correlation exceeding threshold
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper_tri.columns:
        # Get all features that are collinear with 'col'
        collinear_with_col = upper_tri.index[upper_tri[col] > threshold].tolist()
        if collinear_with_col:
            cluster = collinear_with_col + [col]
            # Only keep active features in this cluster
            active_cluster = [f for f in cluster if f not in to_drop]
            if len(active_cluster) > 1:
                # Keep the one with the highest absolute IC value
                best_feature = max(active_cluster, key=lambda f: abs(ic_dict.get(f, 0.0)))
                for f in active_cluster:
                    if f != best_feature:
                        to_drop.add(f)
                        
    logger.info(f"[Preprocessing] Dropping {len(to_drop)} collinear features. Retaining best performers.")
    return df.drop(columns=list(to_drop))


def run_preprocessing_pipeline(df: pd.DataFrame, ic_dict: Dict[str, float]) -> pd.DataFrame:
    """Runs all 5 preprocessing pipeline transformations in sequential order."""
    logger.info(f"[Preprocessing] Starting pipeline on feature matrix: {df.shape}")
    
    # Step 1: Winsorisation
    df_wins = winsorize_features(df)
    
    # Step 2: Cross-Sectional Normalisation
    df_norm = cross_sectional_normalisation(df_wins)
    
    # Step 3: Imputation
    df_imp = impute_missing_values(df_norm)
    
    # Step 4: Multicollinearity Screening
    df_clean = screen_multicollinearity(df_imp, ic_dict)
    
    logger.info(f"[Preprocessing] Pipeline complete. Clean feature matrix: {df_clean.shape}")
    return df_clean
