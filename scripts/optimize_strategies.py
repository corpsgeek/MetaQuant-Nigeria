#!/usr/bin/env python3
"""
Strategy Optimization Worker Script

Runs backtest-based strategy optimization in a separate process.
This script is called by the GUI to avoid segmentation faults caused by
running too many BacktestEngine instances in the main process.

Usage:
    python scripts/optimize_strategies.py [--symbols SYMBOL1,SYMBOL2,...] [--batch-size 5]
"""

import sys
import os

# Add project root to path BEFORE any other imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import time
import gc
from datetime import datetime, timedelta

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_price_data(db):
    """Load all price data from database."""
    stocks = db.conn.execute("""
        SELECT DISTINCT s.symbol, s.id 
        FROM stocks s 
        JOIN daily_prices dp ON s.id = dp.stock_id
    """).fetchall()
    
    price_data = {}
    
    for symbol, stock_id in stocks:
        prices = db.conn.execute("""
            SELECT date, open, high, low, close, volume
            FROM daily_prices
            WHERE stock_id = ?
            ORDER BY date
        """, [stock_id]).fetchall()
        
        if prices:
            df = pd.DataFrame(prices, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            price_data[symbol] = df
    
    return price_data


def optimize_single_stock(symbol: str, df: pd.DataFrame, BacktestEngine, db, ml_engine=None):
    """
    Optimize a single stock using backtest.
    Returns optimal strategy dict or None.
    """
    if df.empty or len(df) < 90:
        return None
    
    # Parameters to test
    stop_loss_range = [0.04, 0.06, 0.08]
    take_profit_range = [0.10, 0.15, 0.25]
    
    best_result = None
    best_score = -float('inf')
    
    # Create price_data dict with just this symbol
    price_data = {symbol: df}
    
    # Test 3M and 6M horizons
    horizons = [('3M', 90), ('6M', 180)]
    
    for horizon_name, horizon_days in horizons:
        if len(df) < horizon_days:
            continue
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=horizon_days)).strftime('%Y-%m-%d')
        
        for stop_loss in stop_loss_range:
            for take_profit in take_profit_range:
                try:
                    engine = BacktestEngine(
                        initial_capital=10_000_000,
                        max_positions=1,
                        stop_loss_pct=stop_loss,
                        take_profit_pct=take_profit,
                        db=db,
                        ml_engine=ml_engine
                    )
                    
                    result = engine.run(
                        symbols=[symbol],
                        price_data=price_data,
                        signal_data={},
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Clean up immediately
                    del engine
                    
                    if not result or result.get('metrics', {}).get('total_trades', 0) < 2:
                        continue
                    
                    metrics = result.get('metrics', {})
                    
                    sharpe = metrics.get('sharpe_ratio', 0)
                    win_rate = metrics.get('win_rate', 0)
                    total_return = metrics.get('total_return_pct', 0)
                    
                    # Score combining multiple factors
                    score = sharpe * 0.4 + win_rate / 100 * 0.4 + min(total_return, 50) / 100 * 0.2
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'return': total_return,
                            'win_rate': win_rate,
                            'sharpe': sharpe,
                            'trades': len(result.get('trades', [])),
                            'horizon': horizon_name
                        }
                    
                except Exception as e:
                    logger.debug(f"Backtest failed: {e}")
    
    # Calculate thresholds based on momentum
    close = df['close'].astype(float)
    mom_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
    
    if mom_20 > 0:
        buy_threshold = max(0.15, min(0.35, 0.3 - mom_20 * 0.5))
    else:
        buy_threshold = max(0.25, min(0.45, 0.3 - mom_20 * 0.3))
    
    # If backtest didn't find enough trades, use statistical fallback
    if best_result is None:
        return calculate_statistical_strategy(symbol, df, buy_threshold)
    
    return {
        'symbol': symbol,
        'stop_loss': best_result['stop_loss'],
        'take_profit': best_result['take_profit'],
        'buy_threshold': round(buy_threshold, 3),
        'sell_threshold': round(-buy_threshold * 0.8, 3),
        'avg_hold_days': 10,
        'min_hold_days': 3,
        'return': best_result['return'],
        'win_rate': best_result['win_rate'],
        'sharpe': best_result['sharpe'],
        'trades': best_result['trades']
    }


def calculate_statistical_strategy(symbol: str, df: pd.DataFrame, buy_threshold: float) -> dict:
    """
    Calculate strategy parameters using statistical analysis when backtest fails.
    Uses volatility and price behavior to set reasonable parameters.
    """
    close = df['close'].astype(float)
    
    # Calculate volatility-based stop-loss
    daily_returns = close.pct_change().dropna()
    volatility = daily_returns.std()
    
    # Stop-loss: based on 2.5x daily volatility, capped between 3-10%
    stop_loss = max(0.03, min(0.10, volatility * 2.5))
    
    # Calculate typical price swings for take-profit
    high = df['high'].astype(float) if 'high' in df.columns else close
    low = df['low'].astype(float) if 'low' in df.columns else close
    
    # Average true range / price ratio
    atr_ratio = ((high - low) / close).mean()
    
    # Max drawup from recent lows (potential gain)
    rolling_min = close.rolling(20).min()
    max_gain = ((close - rolling_min) / rolling_min).max()
    
    # Take-profit: based on ATR and max observed gains, capped between 8-25%
    take_profit = max(0.08, min(0.25, atr_ratio * 15 + max_gain * 0.3))
    
    # Momentum
    mom_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
    
    return {
        'symbol': symbol,
        'stop_loss': round(stop_loss, 3),
        'take_profit': round(take_profit, 3),
        'buy_threshold': round(buy_threshold, 3),
        'sell_threshold': round(-buy_threshold * 0.8, 3),
        'avg_hold_days': 10,
        'min_hold_days': 3,
        'return': round(float(mom_20 * 100), 2),  # 20-day momentum as proxy
        'win_rate': 50,  # Unknown
        'sharpe': 0,
        'trades': 0  # Indicates statistical method used
    }


def main():
    parser = argparse.ArgumentParser(description='Optimize trading strategies via backtest')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to optimize')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of stocks per batch')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between batches in seconds')
    parser.add_argument('--output', type=str, default='', help='Output JSON file for results')
    args = parser.parse_args()
    
    # Import after argument parsing to avoid import overhead if just checking help
    from src.database import DatabaseManager
    from src.backtesting import BacktestEngine
    from src.trading import TradingTables
    
    logger.info("Initializing database...")
    db = DatabaseManager()
    tables = TradingTables(db)
    
    logger.info("Loading price data...")
    price_data = load_price_data(db)
    logger.info(f"Loaded price data for {len(price_data)} stocks")
    
    # Determine symbols to optimize
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = list(price_data.keys())
    
    logger.info(f"Optimizing {len(symbols)} stocks in batches of {args.batch_size}")
    
    results = {}
    optimized = 0
    failed = 0
    
    for batch_start in range(0, len(symbols), args.batch_size):
        batch = symbols[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(symbols) + args.batch_size - 1) // args.batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches}: {', '.join(batch)}")
        
        for symbol in batch:
            if symbol not in price_data or price_data[symbol].empty:
                continue
            
            try:
                strategy = optimize_single_stock(
                    symbol, 
                    price_data[symbol],
                    BacktestEngine,
                    db
                )
                
                if strategy:
                    results[symbol] = strategy
                    tables.save_stock_strategy(symbol, strategy)
                    optimized += 1
                    logger.info(f"  {symbol}: SL={strategy['stop_loss']:.0%}, TP={strategy['take_profit']:.0%}, "
                               f"Return={strategy['return']:.1f}%, Sharpe={strategy['sharpe']:.2f}")
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                logger.error(f"  {symbol}: FAILED - {e}")
            
            # Force garbage collection after each stock
            gc.collect()
        
        # Delay between batches
        if batch_start + args.batch_size < len(symbols):
            logger.info(f"Batch complete. Waiting {args.delay}s before next batch...")
            gc.collect()
            time.sleep(args.delay)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Optimization complete!")
    logger.info(f"Optimized: {optimized}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {len(symbols)}")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    return results


if __name__ == '__main__':
    main()
