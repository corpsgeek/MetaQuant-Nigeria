"""
Strategy Optimizer - Run multi-horizon backtests to find optimal trading parameters.

Features:
- Batched processing (10 stocks at a time)
- Delays between batches (prevent memory pressure)
- Background thread execution
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
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
    
    # Parameter ranges to test (reduced for stability)
    STOP_LOSS_RANGE = [0.04, 0.06, 0.08]
    TAKE_PROFIT_RANGE = [0.10, 0.15, 0.25]
    
    # Batching settings
    BATCH_SIZE = 10
    BATCH_DELAY_SECONDS = 1.0
    
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
        
        # State for background optimization
        self._is_running = False
        self._should_stop = False
        self._progress = {'current': 0, 'total': 0, 'symbol': '', 'status': 'idle'}
    
    def optimize_all_stocks_background(self, symbols: List[str], price_data: Dict,
                                       on_complete: Callable = None) -> threading.Thread:
        """
        Run optimization in background thread with batching and delays.
        
        Args:
            symbols: List of stock symbols
            price_data: Dict of symbol -> DataFrame with OHLCV
            on_complete: Optional callback(results) when done
        
        Returns:
            Thread object (already started)
        """
        def run_optimization():
            self._is_running = True
            self._should_stop = False
            self._progress = {'current': 0, 'total': len(symbols), 'symbol': '', 'status': 'running'}
            
            results = {}
            
            # Process in batches
            for batch_start in range(0, len(symbols), self.BATCH_SIZE):
                if self._should_stop:
                    logger.info("Optimization stopped by user")
                    break
                
                batch = symbols[batch_start:batch_start + self.BATCH_SIZE]
                
                for i, symbol in enumerate(batch):
                    if self._should_stop:
                        break
                    
                    global_idx = batch_start + i + 1
                    self._progress['current'] = global_idx
                    self._progress['symbol'] = symbol
                    
                    if symbol not in price_data or price_data[symbol].empty:
                        continue
                    
                    try:
                        strategy = self._optimize_single_stock_safe(symbol, price_data)
                        if strategy:
                            results[symbol] = strategy
                            self.tables.save_stock_strategy(symbol, strategy)
                    except Exception as e:
                        logger.debug(f"Failed to optimize {symbol}: {e}")
                
                # Delay between batches to reduce memory pressure
                if batch_start + self.BATCH_SIZE < len(symbols):
                    time.sleep(self.BATCH_DELAY_SECONDS)
            
            self._progress['status'] = 'complete'
            self._is_running = False
            
            logger.info(f"Optimized strategies for {len(results)}/{len(symbols)} stocks")
            
            if on_complete:
                on_complete(results)
            
            return results
        
        thread = threading.Thread(target=run_optimization, daemon=True)
        thread.start()
        return thread
    
    def stop_optimization(self):
        """Stop background optimization."""
        self._should_stop = True
    
    def get_progress(self) -> Dict:
        """Get current optimization progress."""
        return self._progress.copy()
    
    def is_running(self) -> bool:
        """Check if optimization is in progress."""
        return self._is_running
    
    def _optimize_single_stock_safe(self, symbol: str, price_data: Dict) -> Optional[Dict]:
        """
        Optimize a single stock with reduced parameters for stability.
        
        Only tests fewer horizons and parameter combinations.
        """
        if symbol not in price_data or price_data[symbol].empty:
            return None
        
        df = price_data[symbol]
        
        # Only use 2 horizons for faster/safer execution
        horizons_to_test = [('3M', 90), ('6M', 180)]
        
        horizon_results = []
        
        for horizon_name, horizon_days in horizons_to_test:
            if len(df) < horizon_days:
                continue
            
            try:
                result = self._run_single_horizon(symbol, price_data, horizon_days)
                if result:
                    result['horizon'] = horizon_name
                    horizon_results.append(result)
            except Exception as e:
                logger.debug(f"Horizon {horizon_name} failed for {symbol}: {e}")
        
        if not horizon_results:
            return self._default_strategy(symbol)
        
        # Average results from horizons
        weights = {'3M': 0.6, '6M': 0.4}
        
        weighted_stop_loss = 0
        weighted_take_profit = 0
        total_weight = 0
        
        best_result = None
        best_score = -float('inf')
        
        for result in horizon_results:
            weight = weights.get(result['horizon'], 0.5)
            total_weight += weight
            
            weighted_stop_loss += result['stop_loss'] * weight
            weighted_take_profit += result['take_profit'] * weight
            
            score = result.get('sharpe', 0) * 0.5 + result.get('win_rate', 0) * 0.5
            if score > best_score:
                best_score = score
                best_result = result
        
        if total_weight == 0:
            return self._default_strategy(symbol)
        
        # Calculate thresholds based on momentum
        close = df['close'].astype(float)
        mom_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
        
        if mom_20 > 0:
            buy_threshold = max(0.15, min(0.35, 0.3 - mom_20 * 0.5))
        else:
            buy_threshold = max(0.25, min(0.45, 0.3 - mom_20 * 0.3))
        
        return {
            'symbol': symbol,
            'stop_loss': round(weighted_stop_loss / total_weight, 3),
            'take_profit': round(weighted_take_profit / total_weight, 3),
            'buy_threshold': round(buy_threshold, 3),
            'sell_threshold': round(-buy_threshold * 0.8, 3),
            'avg_hold_days': 10,
            'min_hold_days': 3,
            'return': best_result.get('return', 0) if best_result else 0,
            'win_rate': best_result.get('win_rate', 0) if best_result else 0,
            'sharpe': best_result.get('sharpe', 0) if best_result else 0,
            'trades': best_result.get('trades', 0) if best_result else 0,
            'horizons_tested': len(horizon_results)
        }
    
    def _run_single_horizon(self, symbol: str, price_data: Dict, days: int) -> Optional[Dict]:
        """
        Run backtest for a single horizon with reduced parameter combinations.
        """
        best_result = None
        best_score = -float('inf')
        
        for stop_loss in self.STOP_LOSS_RANGE:
            for take_profit in self.TAKE_PROFIT_RANGE:
                try:
                    engine = self.BacktestEngine(
                        initial_capital=10_000_000,
                        max_positions=1,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit,
                        db=self.db,
                        ml_engine=self.ml_engine
                    )
                    
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                    
                    result = engine.run(
                        symbols=[symbol],
                        price_data=price_data,
                        signal_data={},
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not result or result.get('metrics', {}).get('total_trades', 0) < 2:
                        continue
                    
                    metrics = result.get('metrics', {})
                    trades = result.get('trades', [])
                    
                    sharpe = metrics.get('sharpe_ratio', 0)
                    win_rate = metrics.get('win_rate', 0)
                    total_return = metrics.get('total_return_pct', 0)
                    
                    score = sharpe * 0.4 + win_rate / 100 * 0.4 + min(total_return, 100) / 100 * 0.2
                    
                    if score > best_score:
                        best_score = score
                        
                        avg_hold = 10
                        if trades:
                            avg_hold = sum(t.get('holding_days', 10) for t in trades) / len(trades)
                        
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
    
    def optimize_stock(self, symbol: str, price_data: Dict) -> Optional[Dict]:
        """Backward compatible method."""
        return self._optimize_single_stock_safe(symbol, price_data)
    
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
        """Get top N stocks by strategy performance."""
        strategies = self.tables.get_active_strategies()
        return sorted(strategies, 
                     key=lambda s: s.get('backtest_sharpe', 0), 
                     reverse=True)[:n]
