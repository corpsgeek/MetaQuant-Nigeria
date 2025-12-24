"""
Performance Metrics for Backtesting.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import statistics

import numpy as np

logger = logging.getLogger(__name__)


def calculate_metrics(trades: List[Dict], initial_capital: float = 1000000) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from trade history.
    
    Args:
        trades: List of trade dicts with entry_price, exit_price, quantity, pnl
        initial_capital: Starting capital
        
    Returns:
        Dict with all performance metrics
    """
    if not trades:
        return {
            'total_return': 0,
            'total_return_pct': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0
        }
    
    # Basic stats
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    winners = [t for t in trades if t.get('pnl', 0) > 0]
    losers = [t for t in trades if t.get('pnl', 0) < 0]
    
    win_count = len(winners)
    loss_count = len(losers)
    total_count = len(trades)
    
    # Win rate
    win_rate = (win_count / total_count * 100) if total_count > 0 else 0
    
    # Average win/loss
    avg_win = statistics.mean([t['pnl'] for t in winners]) if winners else 0
    avg_loss = abs(statistics.mean([t['pnl'] for t in losers])) if losers else 0
    
    # Profit factor
    gross_profit = sum(t['pnl'] for t in winners) if winners else 0
    gross_loss = abs(sum(t['pnl'] for t in losers)) if losers else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # Calculate equity curve
    equity = [initial_capital]
    for t in trades:
        equity.append(equity[-1] + t.get('pnl', 0))
    
    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Daily/trade returns for Sharpe
    returns = [t.get('return_pct', 0) for t in trades if 'return_pct' in t]
    if not returns:
        returns = [(t.get('pnl', 0) / initial_capital * 100) for t in trades]
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(returns) > 1:
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe = 0
    
    # Sortino ratio (only downside deviation)
    negative_returns = [r for r in returns if r < 0]
    if len(negative_returns) > 1:
        downside_std = statistics.stdev(negative_returns)
        avg_return = statistics.mean(returns)
        sortino = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    else:
        sortino = sharpe
    
    # Calmar ratio (return / max drawdown)
    annual_return = (total_pnl / initial_capital) * 100
    calmar = (annual_return / (max_dd * 100)) if max_dd > 0 else 0
    
    return {
        'total_return': round(total_pnl, 2),
        'total_return_pct': round((total_pnl / initial_capital) * 100, 2),
        'total_trades': total_count,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': round(win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
        'max_drawdown': round(max_dd * 100, 2),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'calmar_ratio': round(calmar, 2),
        'final_equity': round(equity[-1], 2),
        'equity_curve': equity
    }


def calculate_position_size(
    capital: float,
    price: float,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float = 0.05
) -> int:
    """
    Calculate position size based on risk management.
    
    Args:
        capital: Available capital
        price: Entry price
        risk_per_trade: Max risk per trade (default 2%)
        stop_loss_pct: Stop loss percentage (default 5%)
        
    Returns:
        Number of shares to buy
    """
    risk_amount = capital * risk_per_trade
    shares = int(risk_amount / (price * stop_loss_pct))
    
    # Ensure we can afford it
    max_shares = int(capital * 0.25 / price)  # Max 25% per position
    
    return min(shares, max_shares)
