"""
Section 7.1: Classical Machine Learning
---------------------------------------
Implements classical machine learning signal discovery:
1. LASSO Regularisation with time-series cross-validation (with a purged gap of 2 days).
2. Random Forest with permutation-based Mean Decrease Accuracy (MDA) feature importance.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


# ===========================================================================
# 7.1.1 LASSO REGULARISATION
# ===========================================================================
def time_series_cv_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge_gap: int = 2
) -> float:
    """
    Computes regularisation path for lambda in [10^-4, 10^0] on logarithmic grid
    using time-series cross-validation with a purged gap to prevent look-ahead bias.
    """
    logger.info("[LASSO] Running time-series cross-validation with purged gap...")
    
    # Lambda grid
    lambdas = np.logspace(-4, 0, 20)
    best_lambda = 1e-4
    best_score = float("inf")
    
    # Ensure sorted index
    X = X.sort_index()
    y = y.reindex(X.index)
    n_samples = len(X)
    
    # Custom Time-Series Split with Purge Gap
    split_size = n_samples // (n_splits + 1)
    
    for lmbd in lambdas:
        mse_scores = []
        for i in range(1, n_splits + 1):
            train_end = i * split_size
            test_start = train_end + purge_gap
            test_end = test_start + split_size
            
            if test_end > n_samples:
                break
                
            X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
            X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
                
            model = Lasso(alpha=lmbd, max_iter=2000, random_state=42)
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse_scores.append(mean_squared_error(y_test, preds))
            except Exception as e:
                logger.warning(f"[LASSO] Fold training failed for alpha={lmbd}: {e}")
                
        if mse_scores:
            mean_mse = np.mean(mse_scores)
            if mean_mse < best_score:
                best_score = mean_mse
                best_lambda = lmbd
                
    logger.info(f"[LASSO] Optimal lambda selected: {best_lambda:.6f} (MSE: {best_score:.6f})")
    return best_lambda


def run_lasso_selection(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Fits LASSO with the optimal cross-validated lambda and returns coefficients."""
    best_lmbd = time_series_cv_lasso(X, y)
    model = Lasso(alpha=best_lmbd, max_iter=3000, random_state=42)
    model.fit(X, y)
    return pd.Series(model.coef_, index=X.columns)


# ===========================================================================
# 7.1.2 RANDOM FOREST WITH MDA FEATURE IMPORTANCE
# ===========================================================================
def calculate_mda_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 50,
    max_depth: int = 5
) -> pd.Series:
    """
    Computes permutation-based Mean Decrease Accuracy (MDA) feature importance
    across all Random Forest trees:
    
    MDA(f_i) = 1/B * sum_{b=1}^B [ L(y_hat_b, y) - L(y_hat_b_permuted, y) ]
    """
    logger.info("[Random Forest] Fitting regressor and computing permutation MDA importances...")
    
    # Fit Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Store MDA importances per feature
    mda_dict = {col: 0.0 for col in X.columns}
    B = len(rf.estimators_)
    
    # Pre-calculate predictions of each individual estimator tree
    original_preds = [tree.predict(X) for tree in rf.estimators_]
    
    # For each feature, permute and calculate the increase in MSE loss per tree
    for col in X.columns:
        # Clone feature matrix and permute column values
        X_permuted = X.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col].values)
        
        loss_diffs = []
        for b, tree in enumerate(rf.estimators_):
            # Original MSE loss for tree b
            orig_loss = mean_squared_error(y, original_preds[b])
            # Permuted MSE loss for tree b
            permuted_pred = tree.predict(X_permuted)
            perm_loss = mean_squared_error(y, permuted_pred)
            
            # Loss increase indicates importance: higher loss when permuted = higher importance
            loss_diffs.append(perm_loss - orig_loss)
            
        mda_dict[col] = float(np.mean(loss_diffs))
        
    mda_series = pd.Series(mda_dict)
    logger.info(f"[Random Forest] MDA feature importances computed. Top feature: {mda_series.idxmax()} ({mda_series.max():.6f})")
    return mda_series
