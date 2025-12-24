"""
Portfolio Optimization for MetaQuant Nigeria.
Implements Mean-Variance, Risk Parity, and Maximum Sharpe optimization.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import scipy for optimization
try:
    from scipy.optimize import minimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Portfolio optimization limited.")


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory.
    
    Strategies:
    - Mean-Variance (Markowitz)
    - Maximum Sharpe Ratio
    - Minimum Volatility
    - Risk Parity
    - Equal Weight (benchmark)
    """
    
    RISK_FREE_RATE = 0.12  # 12% T-bill rate in Nigeria
    TRADING_DAYS = 252
    
    def __init__(self, returns_data: pd.DataFrame):
        """
        Initialize the optimizer.
        
        Args:
            returns_data: DataFrame with daily returns, columns = symbols
        """
        self.returns = returns_data.dropna()
        self.symbols = list(self.returns.columns)
        self.n_assets = len(self.symbols)
        
        if self.returns.empty:
            raise ValueError("No valid returns data provided")
        
        # Calculate statistics
        self.mean_returns = self.returns.mean() * self.TRADING_DAYS
        self.cov_matrix = self.returns.cov() * self.TRADING_DAYS
        self.corr_matrix = self.returns.corr()
        self.volatilities = self.returns.std() * np.sqrt(self.TRADING_DAYS)
        
        logger.info(f"Optimizer initialized with {self.n_assets} assets, "
                    f"{len(self.returns)} days of data")
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio expected return."""
        return np.dot(weights, self.mean_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility (std dev)."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio."""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.RISK_FREE_RATE) / vol if vol > 0 else 0
    
    def neg_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe for minimization."""
        return -self.portfolio_sharpe(weights)
    
    def optimize_max_sharpe(self) -> Dict[str, Any]:
        """
        Find portfolio with maximum Sharpe ratio.
        
        Returns:
            Dict with optimal weights and metrics
        """
        if not SCIPY_AVAILABLE:
            return self.equal_weight()
        
        n = self.n_assets
        init_weights = np.array([1/n] * n)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: 0% to 30% per asset
        bounds = tuple((0, 0.3) for _ in range(n))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self.neg_sharpe,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        if not result.success:
            logger.warning("Max Sharpe optimization did not converge")
            return self.equal_weight()
        
        weights = result.x
        
        return self._build_result(weights, 'MAX_SHARPE')
    
    def optimize_min_volatility(self) -> Dict[str, Any]:
        """
        Find portfolio with minimum volatility.
        
        Returns:
            Dict with optimal weights and metrics
        """
        if not SCIPY_AVAILABLE:
            return self.equal_weight()
        
        n = self.n_assets
        init_weights = np.array([1/n] * n)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 0.3) for _ in range(n))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self.portfolio_volatility,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        if not result.success:
            logger.warning("Min volatility optimization did not converge")
            return self.equal_weight()
        
        weights = result.x
        
        return self._build_result(weights, 'MIN_VOLATILITY')
    
    def optimize_risk_parity(self) -> Dict[str, Any]:
        """
        Risk parity: Equal risk contribution from each asset.
        
        Returns:
            Dict with optimal weights and metrics
        """
        if not SCIPY_AVAILABLE:
            # Simple approximation: inverse volatility weighting
            inv_vol = 1 / self.volatilities
            weights = inv_vol / inv_vol.sum()
            return self._build_result(weights.values, 'RISK_PARITY_APPROX')
        
        n = self.n_assets
        init_weights = np.array([1/n] * n)
        
        def risk_contribution_error(weights):
            """Minimize deviation from equal risk contribution."""
            weights = np.array(weights)
            port_vol = self.portfolio_volatility(weights)
            
            # Marginal risk contribution
            mrc = np.dot(self.cov_matrix, weights) / port_vol
            
            # Risk contribution
            rc = weights * mrc
            
            # Target: equal contribution
            target_rc = port_vol / n
            
            # Sum of squared deviations
            return np.sum((rc - target_rc) ** 2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 0.4) for _ in range(n))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                risk_contribution_error,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        if not result.success:
            # Fall back to inverse volatility
            inv_vol = 1 / self.volatilities
            weights = inv_vol / inv_vol.sum()
            return self._build_result(weights.values, 'RISK_PARITY_APPROX')
        
        weights = result.x
        
        return self._build_result(weights, 'RISK_PARITY')
    
    def equal_weight(self) -> Dict[str, Any]:
        """Equal weight portfolio (benchmark)."""
        weights = np.array([1/self.n_assets] * self.n_assets)
        return self._build_result(weights, 'EQUAL_WEIGHT')
    
    def optimize_target_return(self, target_return: float) -> Dict[str, Any]:
        """
        Find minimum volatility portfolio for a target return.
        
        Args:
            target_return: Target annual return (e.g., 0.20 for 20%)
            
        Returns:
            Dict with optimal weights and metrics
        """
        if not SCIPY_AVAILABLE:
            return self.equal_weight()
        
        n = self.n_assets
        init_weights = np.array([1/n] * n)
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - target_return}
        )
        bounds = tuple((0, 0.3) for _ in range(n))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self.portfolio_volatility,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        if not result.success:
            logger.warning(f"Target return {target_return:.1%} optimization failed")
            return self.optimize_max_sharpe()
        
        weights = result.x
        
        return self._build_result(weights, f'TARGET_{target_return:.0%}')
    
    def efficient_frontier(self, n_points: int = 50) -> List[Dict]:
        """
        Calculate the efficient frontier.
        
        Args:
            n_points: Number of points on the frontier
            
        Returns:
            List of portfolio dicts along the frontier
        """
        if not SCIPY_AVAILABLE:
            return [self.equal_weight()]
        
        # Get range of possible returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        
        for target in target_returns:
            try:
                result = self.optimize_target_return(target)
                if result.get('expected_return', 0) > 0:
                    frontier.append({
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe': result['sharpe_ratio'],
                        'weights': result['weights']
                    })
            except:
                continue
        
        return frontier
    
    def _build_result(self, weights: np.ndarray, strategy: str) -> Dict[str, Any]:
        """Build result dictionary from weights."""
        weights = np.array(weights)
        
        # Round very small weights to 0
        weights[weights < 0.01] = 0
        weights = weights / weights.sum()  # Re-normalize
        
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        sharpe = self.portfolio_sharpe(weights)
        
        # Individual contributions
        allocations = []
        for i, sym in enumerate(self.symbols):
            if weights[i] > 0.001:
                allocations.append({
                    'symbol': sym,
                    'weight': round(weights[i] * 100, 2),
                    'expected_return': round(float(self.mean_returns.iloc[i]) * 100, 2),
                    'volatility': round(float(self.volatilities.iloc[i]) * 100, 2)
                })
        
        # Sort by weight descending
        allocations.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'strategy': strategy,
            'weights': dict(zip(self.symbols, [round(w * 100, 2) for w in weights])),
            'allocations': allocations,
            'expected_return': round(ret * 100, 2),
            'volatility': round(vol * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'n_assets': sum(1 for w in weights if w > 0.01),
            'correlation_matrix': self.corr_matrix.to_dict(),
            'risk_free_rate': self.RISK_FREE_RATE * 100
        }
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Return correlation matrix."""
        return self.corr_matrix
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all assets."""
        return {
            'symbols': self.symbols,
            'expected_returns': self.mean_returns.to_dict(),
            'volatilities': self.volatilities.to_dict(),
            'sharpe_ratios': ((self.mean_returns - self.RISK_FREE_RATE) / self.volatilities).to_dict(),
            'data_points': len(self.returns)
        }


def calculate_returns(price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate daily returns from price data.
    
    Args:
        price_data: Dict of symbol -> DataFrame with 'close' column
        
    Returns:
        DataFrame of daily returns
    """
    returns = {}
    
    for symbol, df in price_data.items():
        if df is None or df.empty:
            continue
        
        if 'close' in df.columns:
            prices = df['close']
        elif 'Close' in df.columns:
            prices = df['Close']
        else:
            continue
        
        # Daily returns
        ret = prices.pct_change().dropna()
        returns[symbol] = ret
    
    return pd.DataFrame(returns).dropna()
