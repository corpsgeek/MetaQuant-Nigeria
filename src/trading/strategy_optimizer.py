"""
Strategy Optimizer - Run multi-horizon backtests to find optimal trading parameters.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Runs backtests across multiple time horizons to find optimal 
    per-stock trading parameters.
    
    Horizons: 1M, 3M, 6M, 1Y
    Optimizes: stop_loss, take_profit, buy/sell thresholds
    """
    
    # Time horizons to test (in days)
    HORIZONS = {
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365
    }
    
    # Parameter ranges to test
    STOP_LOSS_RANGE = [0.03, 0.05, 0.07, 0.10]
    TAKE_PROFIT_RANGE = [0.10, 0.15, 0.20, 0.30]
    
    def __init__(self, backtest_engine_class, trading_tables, db, ml_engine=None):
        """
        Initialize the strategy optimizer.
        
        Args:
            backtest_engine_class: BacktestEngine class to instantiate
            trading_tables: TradingTables for storing optimized strategies
            db: Database manager
            ml_engine: ML engine for predictions
        """
        self.BacktestEngine = backtest_engine_class
        self.tables = trading_tables
        self.db = db
        self.ml_engine = ml_engine
    
    def optimize_all_stocks(self, symbols: List[str], price_data: Dict,
                           progress_callback=None) -> Dict[str, Dict]:
        """
        Optimize strategies for all stocks.
        
        Args:
            symbols: List of stock symbols
            price_data: Dict of symbol -> DataFrame with OHLCV
            progress_callback: Optional callback(current, total, symbol)
        
        Returns:
            Dict of symbol -> optimal strategy
        """
        results = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i + 1, total, symbol)
            
            if symbol not in price_data or price_data[symbol].empty:
                continue
            
            try:
                strategy = self.optimize_stock(symbol, price_data)
                if strategy:
                    results[symbol] = strategy
                    # Save to database
                    self.tables.save_stock_strategy(symbol, strategy)
            except Exception as e:
                logger.error(f"Failed to optimize {symbol}: {e}")
        
        logger.info(f"Optimized strategies for {len(results)}/{total} stocks")
        return results
    
    def optimize_stock(self, symbol: str, price_data: Dict) -> Optional[Dict]:
        """
        Find optimal trading parameters for a single stock.
        
        Runs backtests across multiple horizons and parameter combinations,
        then selects the best performing configuration.
        
        Args:
            symbol: Stock symbol
            price_data: Dict of symbol -> DataFrame
        
        Returns:
            Dict with optimal parameters and metrics
        """
        if symbol not in price_data or price_data[symbol].empty:
            return None
        
        df = price_data[symbol]
        
        # Run backtests for each horizon
        horizon_results = []
        
        for horizon_name, horizon_days in self.HORIZONS.items():
            # Get data for this horizon
            if len(df) < horizon_days:
                continue
            
            try:
                result = self._run_horizon_backtest(
                    symbol, price_data, horizon_days
                )
                if result:
                    result['horizon'] = horizon_name
                    horizon_results.append(result)
            except Exception as e:
                logger.debug(f"Horizon {horizon_name} failed for {symbol}: {e}")
        
        if not horizon_results:
            return self._default_strategy(symbol)
        
        # Weight results by recency (more recent horizons weighted higher)
        weights = {'1M': 0.4, '3M': 0.3, '6M': 0.2, '1Y': 0.1}
        
        # Calculate weighted average of parameters
        weighted_stop_loss = 0
        weighted_take_profit = 0
        weighted_buy_thresh = 0
        weighted_sell_thresh = 0
        total_weight = 0
        
        best_result = None
        best_score = -float('inf')
        
        for result in horizon_results:
            weight = weights.get(result['horizon'], 0.1)
            total_weight += weight
            
            weighted_stop_loss += result['stop_loss'] * weight
            weighted_take_profit += result['take_profit'] * weight
            weighted_buy_thresh += result.get('buy_threshold', 0.3) * weight
            weighted_sell_thresh += result.get('sell_threshold', -0.3) * weight
            
            # Track best overall result
            score = result.get('sharpe', 0) * 0.5 + result.get('win_rate', 0) * 0.5
            if score > best_score:
                best_score = score
                best_result = result
        
        if total_weight == 0:
            return self._default_strategy(symbol)
        
        # Calculate average holding days from backtests
        avg_hold_days = sum(r.get('avg_hold_days', 10) for r in horizon_results) // len(horizon_results)
        
        return {
            'symbol': symbol,
            'stop_loss': round(weighted_stop_loss / total_weight, 3),
            'take_profit': round(weighted_take_profit / total_weight, 3),
            'buy_threshold': round(weighted_buy_thresh / total_weight, 3),
            'sell_threshold': round(weighted_sell_thresh / total_weight, 3),
            'avg_hold_days': max(3, avg_hold_days),  # Min 3 days
            'min_hold_days': 3,
            'return': best_result.get('return', 0) if best_result else 0,
            'win_rate': best_result.get('win_rate', 0) if best_result else 0,
            'sharpe': best_result.get('sharpe', 0) if best_result else 0,
            'trades': best_result.get('trades', 0) if best_result else 0,
            'horizons_tested': len(horizon_results)
        }
    
    def _run_horizon_backtest(self, symbol: str, price_data: Dict, 
                              days: int) -> Optional[Dict]:
        """
        Run backtest for a single horizon, testing multiple parameter combinations.
        
        Returns best performing configuration for this horizon.
        """
        best_result = None
        best_score = -float('inf')
        
        # Test different stop loss / take profit combinations
        for stop_loss in self.STOP_LOSS_RANGE:
            for take_profit in self.TAKE_PROFIT_RANGE:
                try:
                    engine = self.BacktestEngine(
                        initial_capital=10_000_000,
                        max_positions=1,  # Single stock optimization
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit,
                        db=self.db,
                        ml_engine=self.ml_engine
                    )
                    
                    # Calculate date range
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                    
                    # Run backtest for single symbol
                    result = engine.run(
                        symbols=[symbol],
                        price_data=price_data,
                        signal_data={},
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not result or result.get('metrics', {}).get('total_trades', 0) < 3:
                        continue
                    
                    metrics = result.get('metrics', {})
                    trades = result.get('trades', [])
                    
                    # Calculate score: combine Sharpe and win rate
                    sharpe = metrics.get('sharpe_ratio', 0)
                    win_rate = metrics.get('win_rate', 0)
                    total_return = metrics.get('total_return_pct', 0)
                    
                    # Score that rewards both good returns and consistency
                    score = (sharpe * 0.4 + win_rate / 100 * 0.4 + 
                            min(total_return, 100) / 100 * 0.2)
                    
                    if score > best_score:
                        best_score = score
                        
                        # Calculate average holding days
                        if trades:
                            avg_hold = sum(t.get('holding_days', 10) for t in trades) / len(trades)
                        else:
                            avg_hold = 10
                        
                        best_result = {
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'return': total_return,
                            'win_rate': win_rate,
                            'sharpe': sharpe,
                            'trades': len(trades),
                            'avg_hold_days': int(avg_hold),
                            'score': score
                        }
                
                except Exception as e:
                    logger.debug(f"Backtest failed: SL={stop_loss}, TP={take_profit}, error={e}")
        
        return best_result
    
    def _default_strategy(self, symbol: str) -> Dict:
        """Return default strategy when optimization fails."""
        return {
            'symbol': symbol,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'buy_threshold': 0.3,
            'sell_threshold': -0.3,
            'avg_hold_days': 10,
            'min_hold_days': 3,
            'return': 0,
            'win_rate': 0,
            'sharpe': 0,
            'trades': 0,
            'horizons_tested': 0
        }
    
    def get_top_strategies(self, n: int = 20) -> List[Dict]:
        """
        Get top N stocks by strategy performance.
        
        Returns:
            List of strategies sorted by Sharpe ratio
        """
        strategies = self.tables.get_active_strategies()
        return sorted(strategies, 
                     key=lambda s: s.get('backtest_sharpe', 0), 
                     reverse=True)[:n]
    
    def refresh_stale_strategies(self, max_age_days: int = 7) -> int:
        """
        Re-optimize strategies that haven't been updated recently.
        
        Returns:
            Number of strategies updated
        """
        strategies = self.tables.get_active_strategies()
        stale_count = 0
        
        for strategy in strategies:
            last_opt = strategy.get('last_optimized')
            if last_opt:
                try:
                    last_dt = datetime.strptime(str(last_opt)[:10], '%Y-%m-%d')
                    age = (datetime.now() - last_dt).days
                    if age > max_age_days:
                        stale_count += 1
                        # Mark for re-optimization (actual re-opt happens in optimize_all_stocks)
                        logger.info(f"Strategy for {strategy['symbol']} is {age} days old")
                except:
                    pass
        
        return stale_count
