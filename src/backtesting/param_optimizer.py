"""
Parameter Optimizer for MetaQuant Nigeria Backtesting.
Finds optimal stop loss/take profit for individual stocks.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimalParams:
    """Optimal parameters for a stock."""
    symbol: str
    stop_loss: float
    take_profit: float
    expected_return: float
    win_rate: float
    profit_factor: float
    sharpe: float
    volatility: float
    atr_ratio: float  # Stop as ratio of ATR


class ParameterOptimizer:
    """
    Find optimal stop loss/take profit per stock using grid search.
    
    Considers:
    - Stock volatility (ATR-based stops work better than fixed %)
    - Historical win rate at each parameter combo
    - Risk-adjusted return (Sharpe)
    """
    
    # Grid search parameters
    STOP_LOSS_GRID = [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]  # 2% to 15%
    TAKE_PROFIT_GRID = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]  # 5% to 30%
    
    def __init__(self, min_trades: int = 10):
        """
        Initialize the parameter optimizer.
        
        Args:
            min_trades: Minimum trades required for valid optimization
        """
        self.min_trades = min_trades
        self.optimal_params: Dict[str, OptimalParams] = {}
    
    def optimize_stock(
        self, 
        symbol: str, 
        price_df: pd.DataFrame,
        signals: Optional[List[Dict]] = None
    ) -> OptimalParams:
        """
        Find optimal stop loss/take profit for a single stock.
        
        Args:
            symbol: Stock symbol
            price_df: DataFrame with OHLCV data
            signals: Optional buy signals (if None, uses all entries)
            
        Returns:
            OptimalParams with best parameters
        """
        if price_df.empty or len(price_df) < 60:
            # Not enough data, use volatility-based defaults
            return self._volatility_based_params(symbol, price_df)
        
        # Calculate stock volatility (ATR and standard deviation)
        volatility_info = self._calculate_volatility(price_df)
        
        # Grid search for optimal parameters
        best_score = -np.inf
        best_params = None
        
        for sl, tp in itertools.product(self.STOP_LOSS_GRID, self.TAKE_PROFIT_GRID):
            # Skip illogical combos (TP should be > SL for positive expectancy)
            if tp <= sl:
                continue
            
            # Simulate trades with these parameters
            result = self._simulate_trades(price_df, sl, tp, signals)
            
            if result['n_trades'] < self.min_trades:
                continue
            
            # Score: Sharpe-like metric (return / volatility) weighted by win rate
            score = result['expectancy'] * result['win_rate'] / max(0.01, volatility_info['daily_vol'])
            
            if score > best_score:
                best_score = score
                best_params = {
                    'stop_loss': sl,
                    'take_profit': tp,
                    **result
                }
        
        if best_params is None:
            return self._volatility_based_params(symbol, price_df)
        
        optimal = OptimalParams(
            symbol=symbol,
            stop_loss=best_params['stop_loss'],
            take_profit=best_params['take_profit'],
            expected_return=best_params['avg_return'],
            win_rate=best_params['win_rate'],
            profit_factor=best_params['profit_factor'],
            sharpe=best_params.get('sharpe', 0),
            volatility=volatility_info['annual_vol'],
            atr_ratio=best_params['stop_loss'] / max(0.01, volatility_info['atr_pct'])
        )
        
        self.optimal_params[symbol] = optimal
        
        logger.info(f"{symbol}: Optimal SL={optimal.stop_loss:.1%}, TP={optimal.take_profit:.1%}, "
                    f"WinRate={optimal.win_rate:.1%}")
        
        return optimal
    
    def optimize_all(
        self, 
        price_data: Dict[str, pd.DataFrame],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, OptimalParams]:
        """
        Optimize parameters for all stocks.
        
        Args:
            price_data: Dict of symbol -> DataFrame
            progress_callback: Optional callback(current, total, symbol)
            
        Returns:
            Dict of symbol -> OptimalParams
        """
        total = len(price_data)
        
        for i, (symbol, df) in enumerate(price_data.items()):
            if progress_callback:
                progress_callback(i + 1, total, symbol)
            
            try:
                self.optimize_stock(symbol, df)
            except Exception as e:
                logger.warning(f"Failed to optimize {symbol}: {e}")
        
        return self.optimal_params
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics for the stock."""
        if 'close' not in df.columns:
            return {'daily_vol': 0.03, 'annual_vol': 0.45, 'atr_pct': 0.02}
        
        # Daily returns volatility
        returns = df['close'].pct_change().dropna()
        daily_vol = returns.std() if len(returns) > 10 else 0.03
        annual_vol = daily_vol * np.sqrt(252)
        
        # ATR (Average True Range) as percentage
        if 'high' in df.columns and 'low' in df.columns:
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            atr_pct = atr / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0.02
        else:
            atr_pct = daily_vol * 1.5
        
        return {
            'daily_vol': daily_vol,
            'annual_vol': annual_vol,
            'atr_pct': atr_pct
        }
    
    def _volatility_based_params(self, symbol: str, df: pd.DataFrame) -> OptimalParams:
        """Create default params based on volatility when not enough data."""
        vol_info = self._calculate_volatility(df)
        
        # Rule of thumb: Stop = 2x ATR, TP = 3x ATR
        atr_pct = vol_info['atr_pct']
        stop_loss = max(0.03, min(0.15, atr_pct * 2))
        take_profit = max(0.08, min(0.30, atr_pct * 3))
        
        return OptimalParams(
            symbol=symbol,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_return=0,
            win_rate=0,
            profit_factor=0,
            sharpe=0,
            volatility=vol_info['annual_vol'],
            atr_ratio=2.0
        )
    
    def _simulate_trades(
        self, 
        df: pd.DataFrame, 
        stop_loss: float, 
        take_profit: float,
        signals: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """Simulate trades with given SL/TP to calculate metrics."""
        if 'close' not in df.columns:
            return {'n_trades': 0, 'win_rate': 0, 'avg_return': 0, 
                    'profit_factor': 0, 'expectancy': 0}
        
        prices = df['close'].values
        n = len(prices)
        
        # Entry points: every 10 days (simple simulation) or from signals
        if signals:
            entries = [s['index'] for s in signals if 'index' in s]
        else:
            entries = list(range(0, n - 20, 10))  # Entry every 10 days
        
        trades = []
        
        for entry_idx in entries:
            if entry_idx >= n - 5:
                continue
            
            entry_price = prices[entry_idx]
            stop_price = entry_price * (1 - stop_loss)
            target_price = entry_price * (1 + take_profit)
            
            # Simulate forward
            exit_price = None
            exit_type = None
            
            for j in range(entry_idx + 1, min(entry_idx + 60, n)):  # Max 60 days hold
                price = prices[j]
                
                if price <= stop_price:
                    exit_price = stop_price  # Hit stop
                    exit_type = 'stop'
                    break
                elif price >= target_price:
                    exit_price = target_price  # Hit target
                    exit_type = 'target'
                    break
            
            if exit_price is None:
                # Time stop - exit at end of window
                exit_price = prices[min(entry_idx + 60, n - 1)]
                exit_type = 'time'
            
            ret = (exit_price - entry_price) / entry_price
            trades.append({
                'return': ret,
                'type': exit_type,
                'win': ret > 0
            })
        
        if not trades:
            return {'n_trades': 0, 'win_rate': 0, 'avg_return': 0, 
                    'profit_factor': 0, 'expectancy': 0}
        
        n_trades = len(trades)
        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        
        win_rate = len(wins) / n_trades
        avg_return = sum(t['return'] for t in trades) / n_trades
        
        gross_profit = sum(t['return'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['return'] for t in losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = (gross_profit / len(wins)) if wins else 0
        avg_loss = (gross_loss / len(losses)) if losses else 0.001
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    def get_params(self, symbol: str) -> Tuple[float, float]:
        """
        Get optimal stop loss and take profit for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (stop_loss, take_profit) percentages
        """
        if symbol in self.optimal_params:
            p = self.optimal_params[symbol]
            return (p.stop_loss, p.take_profit)
        
        # Default if not optimized
        return (0.05, 0.15)
    
    def get_all_params(self) -> Dict[str, Dict]:
        """Get all optimized parameters as dict."""
        return {
            sym: {
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit,
                'win_rate': p.win_rate,
                'volatility': p.volatility
            }
            for sym, p in self.optimal_params.items()
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export optimized parameters as DataFrame."""
        if not self.optimal_params:
            return pd.DataFrame()
        
        data = []
        for sym, p in self.optimal_params.items():
            data.append({
                'Symbol': sym,
                'Stop Loss %': round(p.stop_loss * 100, 1),
                'Take Profit %': round(p.take_profit * 100, 1),
                'Win Rate %': round(p.win_rate * 100, 1),
                'Profit Factor': round(p.profit_factor, 2),
                'Volatility %': round(p.volatility * 100, 1),
                'ATR Ratio': round(p.atr_ratio, 2)
            })
        
        return pd.DataFrame(data)
