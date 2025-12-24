#!/usr/bin/env python3
"""
Clean synthetic price data and re-fetch real prices from TradingView.
"""

import duckdb
import os
import sys
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'metaquant.db')

def main():
    print("=" * 60)
    print("NGX Price Data Cleanup & Re-fetch")
    print("=" * 60)
    
    conn = duckdb.connect(DB_PATH)
    
    # Show current state
    result = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_prices").fetchone()
    print(f"\nCurrent daily_prices:")
    print(f"  Records: {result[0]:,}")
    print(f"  Date range: {result[1]} to {result[2]}")
    
    # Check for suspicious data
    print("\n📊 Checking for suspicious price movements...")
    suspicious = conn.execute("""
        SELECT s.symbol, p.date, p.open, p.close, 
               ((p.close - p.open) / NULLIF(p.open, 0)) * 100 as daily_change_pct
        FROM daily_prices p
        JOIN stocks s ON p.stock_id = s.id
        WHERE ABS((p.close - p.open) / NULLIF(p.open, 0)) > 0.2  -- >20% daily change
        ORDER BY ABS((p.close - p.open) / NULLIF(p.open, 0)) DESC
        LIMIT 20
    """).fetchall()
    
    if suspicious:
        print(f"\n⚠️ Found {len(suspicious)} records with >20% daily change:")
        for row in suspicious[:10]:
            print(f"  {row[0]} on {row[1]}: ₦{row[2]:.2f} → ₦{row[3]:.2f} ({row[4]:.1f}%)")
    
    # Confirm deletion
    print("\n" + "=" * 60)
    print("⚠️  This will DELETE ALL price data and re-fetch from TradingView")
    print("=" * 60)
    confirm = input("\nType 'DELETE' to proceed: ")
    
    if confirm != 'DELETE':
        print("Cancelled.")
        return
    
    # Delete all price data
    print("\n🗑️  Deleting all daily_prices...")
    conn.execute("DELETE FROM daily_prices")
    print("✅ Deleted")
    
    conn.close()
    
    # Now fetch real data
    print("\n📥 Fetching real prices from TradingView...")
    print("This may take several minutes...\n")
    
    from tvDatafeed import TvDatafeed, Interval
    from src.database.db_manager import DatabaseManager
    from datetime import datetime
    import time
    
    db = DatabaseManager(DB_PATH)
    
    # Login to TradingView for better access
    try:
        tv = TvDatafeed(username='ridwan.adetu@comerciopartners.com', password=os.environ.get('TV_PASSWORD', ''))
        print("✅ Logged into TradingView")
    except:
        tv = TvDatafeed()
        print("⚠️ Using TradingView without login (limited data)")
    
    # Get all stocks
    stocks = db.get_all_stocks()
    print(f"Found {len(stocks)} stocks to fetch\n")
    
    success_count = 0
    fail_count = 0
    
    for i, stock in enumerate(stocks):
        symbol = stock.get('symbol')
        stock_id = stock.get('id')
        
        if not symbol or not stock_id:
            continue
        
        try:
            # Fetch daily data from TradingView
            df = tv.get_hist(symbol=symbol, exchange='NSENG', interval=Interval.in_daily, n_bars=500)
            
            if df is not None and not df.empty:
                # Save to database
                for idx, row in df.iterrows():
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
                    db.insert_daily_price(stock_id, date_str, {
                        'open': float(row['open']) if 'open' in row else None,
                        'high': float(row['high']) if 'high' in row else None,
                        'low': float(row['low']) if 'low' in row else None,
                        'close': float(row['close']) if 'close' in row else None,
                        'volume': int(row['volume']) if 'volume' in row else 0
                    })
                
                print(f"✅ [{i+1}/{len(stocks)}] {symbol}: {len(df)} records")
                success_count += 1
            else:
                print(f"⚪ [{i+1}/{len(stocks)}] {symbol}: No data")
                fail_count += 1
            
            # Rate limiting
            time.sleep(0.3)
                
        except Exception as e:
            print(f"❌ [{i+1}/{len(stocks)}] {symbol}: {str(e)[:50]}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {fail_count}")
    
    # Verify
    conn = duckdb.connect(DB_PATH)
    result = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_prices").fetchone()
    print(f"\nNew daily_prices:")
    print(f"  Records: {result[0]:,}")
    print(f"  Date range: {result[1]} to {result[2]}")
    conn.close()

if __name__ == "__main__":
    main()
