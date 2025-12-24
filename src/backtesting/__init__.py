"""
Backtesting module for MetaQuant Nigeria.
"""

from .engine import BacktestEngine
from .signal_scorer import SignalScorer
from .metrics import calculate_metrics
from .optimizer import PortfolioOptimizer, calculate_returns

__all__ = ['BacktestEngine', 'SignalScorer', 'calculate_metrics', 'PortfolioOptimizer', 'calculate_returns']

