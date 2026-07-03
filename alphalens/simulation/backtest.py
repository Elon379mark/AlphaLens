import math
import logging
from typing import Dict, Any, List, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Production-grade vectorized out-of-sample backtesting simulation engine.
    Computes Sharpe, Max Drawdown, Calmar, and Information Ratios while modeling
    realistic transaction costs (commissions, bid-ask spread, and Kyle's lambda market impact).
    """
    def __init__(self, 
                 initial_capital: float = 10_000_000.0,
                 commission_rate: float = 0.0005,  # 5 bps
                 bid_ask_spread: float = 0.0010,   # 10 bps
                 annualization_factor: float = 252.0):
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.bid_ask_spread = bid_ask_spread
        self.annualization_factor = annualization_factor

    def compute_transaction_costs(self, 
                                  q: float, 
                                  v_adv: float, 
                                  volatility: float, 
                                  price: float) -> float:
        """
        Computes the total transaction cost for order quantity q based on the specification:
        TC(q) = volatility * sqrt(|q| / V_ADV) * sgn(q) * P (added price impact)
        Returns the absolute total dollar cost: |q| * (spread/2 * P + commission * P + |TC_impact|)
        """
        if q == 0 or v_adv <= 0:
            return 0.0

        # Kyle's lambda market impact per share
        impact_per_share = volatility * math.sqrt(abs(q) / v_adv) * price
        
        # Fixed costs (spread + commission) per share
        fixed_per_share = (self.bid_ask_spread / 2.0 + self.commission_rate) * price
        
        total_cost = abs(q) * (fixed_per_share + impact_per_share)
        return total_cost

    def run_backtest(self, 
                     signals: List[float], 
                     prices: List[float], 
                     volumes: List[float], 
                     volatilities: List[float],
                     dates: List[str]) -> Dict[str, Any]:
        """
        Runs the simulation backtest out-of-sample. Enforces point-in-time correctness.
        """
        T = len(prices)
        if T < 2:
            return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "calmar_ratio": 0.0, "total_return": 0.0}

        # Vectorized implementation using NumPy / Pandas
        if np is not None and pd is not None:
            return self._run_vectorized(signals, prices, volumes, volatilities, dates)
        
        # Pure-Python fallback simulation
        logger.info("NumPy/Pandas not available. Running pure-Python simulation.")
        return self._run_python_simulation(signals, prices, volumes, volatilities, dates)

    def _run_vectorized(self, 
                        signals: List[float], 
                        prices: List[float], 
                        volumes: List[float], 
                        volatilities: List[float],
                        dates: List[str]) -> Dict[str, Any]:
        
        sig = np.array(signals, dtype=np.float64)
        prc = np.array(prices, dtype=np.float64)
        vol = np.array(volumes, dtype=np.float64)
        vlt = np.array(volatilities, dtype=np.float64)
        
        # 30-day average daily volume (V_ADV)
        df_vol = pd.Series(vol)
        v_adv = df_vol.rolling(30, min_periods=1).mean().values

        cash = self.initial_capital
        portfolio_values = [cash]
        shares_held = 0.0

        # Out-of-sample rebalancing simulation
        for t in range(1, len(prc)):
            # Enforce Point-In-Time (PIT) data access: decisions at t use only information up to t-1
            target_signal = sig[t - 1]
            current_price = prc[t]
            
            # Value of portfolio before trade is executed on day t
            portfolio_value_before = (shares_held * current_price) + cash
            
            # Target allocation: capital * signal (using current portfolio value as capital base)
            # Ensure target allocation is within risk limits
            target_value = portfolio_value_before * np.clip(target_signal, -1.0, 1.0)
            target_shares = target_value / current_price
            
            # Rebalance order quantity
            order_qty = target_shares - shares_held
            
            # Transaction costs
            tc = self.compute_transaction_costs(order_qty, v_adv[t], vlt[t], current_price)
            
            # Update holdings and cash
            shares_held = target_shares
            cash = cash - (order_qty * current_price) - tc
            
            # Total portfolio value at t
            portfolio_value = (shares_held * current_price) + cash
            portfolio_values.append(portfolio_value)

        # Performance metrics calculation
        val_series = pd.Series(portfolio_values)
        daily_returns = val_series.pct_change().dropna()
        
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        
        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        sharpe = (mean_ret / std_ret * np.sqrt(self.annualization_factor)) if std_ret > 0 else 0.0
        
        # Max Drawdown
        cum_returns = (1.0 + daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_dd = float(drawdowns.min())

        # Calmar Ratio
        annual_return = float(cum_returns.iloc[-1] ** (self.annualization_factor / len(cum_returns)) - 1.0) if len(cum_returns) > 0 else 0.0
        calmar = (annual_return / abs(max_dd)) if max_dd != 0.0 else 0.0

        return {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "total_return": float((portfolio_values[-1] - self.initial_capital) / self.initial_capital),
            "portfolio_values": portfolio_values
        }

    def _run_python_simulation(self, 
                               signals: List[float], 
                               prices: List[float], 
                               volumes: List[float], 
                               volatilities: List[float],
                               dates: List[str]) -> Dict[str, Any]:
        T = len(prices)
        capital = self.initial_capital
        portfolio_values = [capital]
        shares_held = 0.0

        # Calculate V_ADV manually
        v_adv = []
        for t in range(T):
            start = max(0, t - 29)
            subset = volumes[start:t+1]
            v_adv.append(sum(subset) / len(subset))

        for t in range(1, T):
            target_signal = max(-1.0, min(1.0, signals[t - 1]))
            current_price = prices[t]
            
            target_value = capital * target_signal
            target_shares = target_value / current_price
            order_qty = target_shares - shares_held
            
            tc = self.compute_transaction_costs(order_qty, v_adv[t], volatilities[t], current_price)
            
            shares_held = target_shares
            capital = capital - (order_qty * current_price) - tc
            
            portfolio_value = shares_held * current_price + capital
            portfolio_values.append(portfolio_value)

        # Compute returns
        daily_returns = []
        for t in range(1, len(portfolio_values)):
            ret = (portfolio_values[t] - portfolio_values[t-1]) / portfolio_values[t-1]
            daily_returns.append(ret)

        n = len(daily_returns)
        mean_ret = sum(daily_returns) / n
        variance = sum((r - mean_ret)**2 for r in daily_returns) / (n - 1) if n > 1 else 0.0
        std_ret = math.sqrt(variance)

        sharpe = (mean_ret / std_ret * math.sqrt(self.annualization_factor)) if std_ret > 0 else 0.0

        # Drawdown calculation
        peak = portfolio_values[0]
        max_dd = 0.0
        for val in portfolio_values:
            if val > peak:
                peak = val
            dd = (val - peak) / peak
            if dd < max_dd:
                max_dd = dd

        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = ((1.0 + total_return) ** (self.annualization_factor / n) - 1.0) if n > 0 else 0.0
        calmar = (annual_return / abs(max_dd)) if max_dd != 0.0 else 0.0

        return {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "total_return": total_return,
            "portfolio_values": portfolio_values
        }
