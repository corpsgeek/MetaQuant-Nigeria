#!/usr/bin/env python
"""
Historical Data Backfill Script

Uses tvdatafeed (requires Python 3.12) to fetch real historical price data
from TradingView for all NGX stocks.

USAGE: Run with Python 3.12 venv
    .venv312/bin/python scripts/backfill_historical.py
    
OPTIONS:
    --days N     Number of days to fetch (default: 365)
    --symbol X   Fetch only specific symbol
    --test       Test mode - fetch only 5 stocks
"""

import sys
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tvDatafeed import TvDatafeed, Interval
    TVDATAFEED_AVAILABLE = True
except ImportError:
    TVDATAFEED_AVAILABLE = False
    logger.error("tvdatafeed not available. Install with: pip install git+https://github.com/rongardF/tvdatafeed.git")

import duckdb


# NGX Stock symbols (major ones)
NGX_SYMBOLS = [
    'DANGCEM', 'MTNN', 'GTCO', 'ZENITHBANK', 'ACCESSCORP', 'UBA', 
    'BUACEMENT', 'NESTLE', 'SEPLAT', 'AIRTELAFRI', 'STANBIC', 
    'DANGSUGAR', 'FLOURMILL', 'PRESCO', 'OANDO', 'FIDELITYBK',
    'WAPCO', 'TRANSCORP', 'FIRSTHOLDCO', 'FBNH', 'GEREGU', 'ARADEL',
    'CUSTODIAN', 'ETI', 'GUINNESS', 'NB', 'INTBREW', 'CHAMPION',
    'BUAFOODS', 'NASCON', 'CADBURY', 'NAHCO', 'JBERGER', 'JULIUS',
    'NGXGROUP', 'COURTVILLE', 'FIDSON', 'GLAXOSMITH', 'MAYBAKER',
    'NEIMETH', 'PHARMDEKO', 'AFRIPRUD', 'AIICO', 'AXA', 'CHIPLC',
    'CORNERST', 'LASACO', 'LINKASSURE', 'MANSARD', 'MBENEFIT',
    'NEM', 'NIGERINS', 'PRESTIGE', 'REGALINS', 'ROYALEX', 'SOVEREIGN',
    'VERITASKAP', 'WAPIC', 'ELLAHLAKES', 'LIVESTOCK', 'OKOMUOIL',
    'LEARNAFRCA', 'STUDPRESS', 'UPDC', 'UPDCREIT', 'UNIONDAC',
    'ABBEYBDS', 'ABCTRANS', 'ACADEMY', 'AFROMEDIA', 'ALEX',
]


def get_db_path():
    """Get the database path."""
    return PROJECT_ROOT / 'data' / 'metaquant.db'


def backfill_symbol(tv: TvDatafeed, conn, symbol: str, n_bars: int = 365) -> dict:
    """Fetch and store historical data for a single symbol."""
    result = {'symbol': symbol, 'success': False, 'bars': 0, 'error': None}
    
    try:
        # Fetch historical data from TradingView
        data = tv.get_hist(symbol, 'NSENG', Interval.in_daily, n_bars=n_bars)
        
        if data is None or data.empty:
            result['error'] = 'No data returned'
            return result
        
        # Reset index to get datetime as column
        data = data.reset_index()
        
        # Get stock_id from database
        stock_row = conn.execute(
            "SELECT id FROM stocks WHERE symbol = ?", [symbol]
        ).fetchone()
        
        if stock_row is None:
            result['error'] = 'Stock not in database'
            return result
        
        stock_id = stock_row[0]
        
        # Insert/update daily prices
        for _, row in data.iterrows():
            date_str = row['datetime'].strftime('%Y-%m-%d')
            conn.execute("""
                INSERT INTO daily_prices (stock_id, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, [
                stock_id, date_str,
                float(row['open']), float(row['high']),
                float(row['low']), float(row['close']),
                int(row['volume']) if row['volume'] else 0
            ])
        
        conn.commit()
        
        result['success'] = True
        result['bars'] = len(data)
        logger.info(f"✓ {symbol}: {len(data)} bars stored")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"✗ {symbol}: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Backfill historical NGX data')
    parser.add_argument('--days', type=int, default=365, help='Number of days to fetch')
    parser.add_argument('--symbol', type=str, help='Fetch only specific symbol')
    parser.add_argument('--test', action='store_true', help='Test with 5 stocks only')
    
    args = parser.parse_args()
    
    if not TVDATAFEED_AVAILABLE:
        logger.error("tvdatafeed not installed. Run this with Python 3.12 venv:")
        logger.error("  .venv312/bin/python scripts/backfill_historical.py")
        sys.exit(1)
    
    # Initialize TradingView connection
    logger.info("Connecting to TradingView...")
    tv = TvDatafeed()
    
    # Connect to database
    db_path = get_db_path()
    logger.info(f"Using database: {db_path}")
    conn = duckdb.connect(str(db_path))
    
    # Determine symbols to process
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.test:
        symbols = NGX_SYMBOLS[:5]
        logger.info("Test mode: processing 5 symbols only")
    else:
        symbols = NGX_SYMBOLS
    
    logger.info(f"Backfilling {len(symbols)} symbols with {args.days} days of data...")
    logger.info("=" * 60)
    
    # Process each symbol
    success_count = 0
    error_count = 0
    total_bars = 0
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Fetching {symbol}...")
        result = backfill_symbol(tv, conn, symbol, args.days)
        
        if result['success']:
            success_count += 1
            total_bars += result['bars']
        else:
            error_count += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"COMPLETE: {success_count} symbols, {total_bars} total bars")
    logger.info(f"Errors: {error_count}")
    
    conn.close()


if __name__ == "__main__":
    main()
