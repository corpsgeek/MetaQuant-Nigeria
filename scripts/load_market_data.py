#!/usr/bin/env python
"""
NGX Data Loader Script

Fetches current market data from TradingView and stores it in the database.
Run this script daily after market close (after 2:30pm WAT) to build historical data.

Usage:
    python scripts/load_market_data.py
    
    # Or with scheduling:
    python scripts/load_market_data.py --schedule  # Runs daily at 3:00pm WAT
"""

import sys
import os
import logging
import argparse
from datetime import datetime, date
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.tradingview_collector import TradingViewCollector
from src.database.db_manager import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_market_data(db: DatabaseManager = None) -> dict:
    """
    Load current market data from TradingView and store in database.
    
    Args:
        db: Optional database manager (creates new if not provided)
        
    Returns:
        Dictionary with load statistics
    """
    if db is None:
        db = DatabaseManager()
    
    collector = TradingViewCollector()
    
    logger.info("Fetching NGX market data from TradingView...")
    
    # Get all stocks
    df = collector.get_all_stocks()
    
    if df.empty:
        logger.error("No data returned from TradingView")
        return {'success': False, 'error': 'No data returned'}
    
    logger.info(f"Fetched {len(df)} stocks. Storing in database...")
    
    today = date.today().isoformat()
    success_count = 0
    error_count = 0
    
    for _, row in df.iterrows():
        try:
            symbol = row.get('symbol', '')
            if not symbol:
                continue
            
            # Get existing stock
            stock = db.get_stock(symbol)
            
            if stock is None:
                # New stock not in database - skip for now
                # These can be added via universe seeding
                logger.debug(f"Stock not in database, skipping: {symbol}")
                continue
            
            stock_id = stock['id']
            
            # Update stock price and market cap
            try:
                db.conn.execute("""
                    UPDATE stocks SET 
                        last_price = ?,
                        prev_close = ?,
                        change_percent = ?,
                        volume = ?,
                        market_cap = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, [
                    row.get('close'),
                    row.get('open'),
                    row.get('change'),
                    row.get('volume'),
                    row.get('market_cap_basic'),  # Store market cap from TradingView
                    stock_id
                ])
            except Exception as e:
                logger.debug(f"Could not update stock {symbol}: {e}")
            
            # Insert daily price with TradingView's actual change %
            db.insert_daily_price(stock_id, today, {
                'open': row.get('open'),
                'high': row.get('high'),
                'low': row.get('low'),
                'close': row.get('close'),
                'volume': row.get('volume'),
                'change_pct': row.get('change'),  # Store TV's actual change %
            })
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error storing {symbol}: {e}")
            error_count += 1



    
    db.conn.commit()
    
    result = {
        'success': True,
        'date': today,
        'total_fetched': len(df),
        'stored': success_count,
        'errors': error_count,
        'timestamp': datetime.now().isoformat(),
    }
    
    logger.info(f"Data load complete: {success_count} stocks stored, {error_count} errors")
    
    return result


def show_market_snapshot(collector: TradingViewCollector = None):
    """Display current market snapshot."""
    if collector is None:
        collector = TradingViewCollector()
    
    snapshot = collector.get_market_snapshot()
    
    if 'error' in snapshot:
        print(f"Error: {snapshot['error']}")
        return
    
    print("\n" + "=" * 60)
    print("NGX MARKET SNAPSHOT")
    print("=" * 60)
    print(f"Timestamp: {snapshot['timestamp']}")
    print(f"Total Stocks: {snapshot['total_stocks']}")
    print(f"Gainers: {snapshot['gainers']} | Losers: {snapshot['losers']} | Unchanged: {snapshot['unchanged']}")
    print(f"Total Volume: {snapshot['total_volume']:,.0f}")
    
    print("\n📈 TOP GAINERS:")
    for stock in snapshot.get('top_gainers', []):
        print(f"  {stock['symbol']:<12} {stock.get('name', '')[:20]:<20} ₦{stock['close']:,.2f}  +{stock['change']:.2f}%")
    
    print("\n📉 TOP LOSERS:")
    for stock in snapshot.get('top_losers', []):
        print(f"  {stock['symbol']:<12} {stock.get('name', '')[:20]:<20} ₦{stock['close']:,.2f}  {stock['change']:.2f}%")
    
    print("\n🔥 MOST ACTIVE:")
    for stock in snapshot.get('most_active', []):
        print(f"  {stock['symbol']:<12} {stock.get('name', '')[:20]:<20} ₦{stock['close']:,.2f}  Vol: {stock['volume']:,.0f}")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Load NGX market data from TradingView')
    parser.add_argument('--snapshot', action='store_true', help='Show market snapshot only')
    parser.add_argument('--test', action='store_true', help='Test connection without storing')
    
    args = parser.parse_args()
    
    collector = TradingViewCollector()
    
    if args.snapshot:
        show_market_snapshot(collector)
        return
    
    if args.test:
        print("Testing TradingView connection...")
        df = collector.get_all_stocks(limit=5)
        if not df.empty:
            print(f"✓ Successfully fetched {len(df)} test stocks")
            print(df[['symbol', 'name', 'close', 'volume']].to_string(index=False))
        else:
            print("✗ Failed to fetch data")
        return
    
    # Full data load
    print("Starting NGX data load...")
    result = load_market_data()
    
    if result['success']:
        print(f"\n✓ Successfully loaded {result['stored']} stocks for {result['date']}")
        show_market_snapshot(collector)
    else:
        print(f"\n✗ Failed to load data: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
