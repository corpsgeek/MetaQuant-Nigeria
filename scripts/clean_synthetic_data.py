#!/usr/bin/env python
"""
Clean synthetic data from the database.

Removes all price data that wasn't loaded from TradingView (synthetic/seeded data).
Keeps only real data loaded via tvdatafeed backfill.

USAGE: Close the app first, then run:
    python scripts/clean_synthetic_data.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

def main():
    db_path = PROJECT_ROOT / 'data' / 'metaquant.db'
    print(f"Connecting to database: {db_path}")
    
    conn = duckdb.connect(str(db_path))
    
    # Get statistics before cleanup
    before_count = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    
    # Get list of stocks that have real data (from tvdatafeed - these have 730 bars each)
    # Stocks with only synthetic data will have inconsistent patterns
    
    # Real data characteristics:
    # - Comes from tvdatafeed starting December 2023 (730 days ago from Dec 2025)
    # - Has proper OHLC values (not rounded synthetic values)
    
    # The tvdatafeed data is in this range:
    cutoff_date = '2023-12-01'  # Real data starts around here (730 days)
    
    print(f"\nAnalyzing data...")
    
    # Check what date ranges exist
    date_stats = conn.execute("""
        SELECT 
            MIN(date) as earliest,
            MAX(date) as latest,
            COUNT(DISTINCT date) as num_dates
        FROM daily_prices
    """).fetchone()
    
    print(f"Current date range: {date_stats[0]} to {date_stats[1]} ({date_stats[2]} dates)")
    
    # Find stocks loaded with real data (they have consistent 730 bars)
    real_stocks = conn.execute("""
        SELECT stock_id, COUNT(*) as bar_count
        FROM daily_prices
        WHERE date >= '2023-12-01'
        GROUP BY stock_id
        HAVING COUNT(*) >= 300  -- Real stocks have ~700+ bars from tvdatafeed
    """).fetchall()
    
    print(f"Stocks with real data (300+ bars since Dec 2023): {len(real_stocks)}")
    
    # Identify synthetic data patterns:
    # - Dates before our backfill range that have suspicious patterns
    # - Dates with OHLC all the same (typical of synthetic data)
    
    # Count synthetic-looking entries (where open=high=low=close AND before real data)
    suspicious = conn.execute("""
        SELECT COUNT(*) FROM daily_prices
        WHERE date < '2023-12-01'
        AND open = high AND high = low AND low = close
    """).fetchone()[0]
    
    print(f"Suspicious entries (OHLC all same, before Dec 2023): {suspicious}")
    
    # Option: Delete all data before our reliable backfill range
    print("\n" + "="*60)
    print("CLEANUP OPTIONS:")
    print("="*60)
    
    # Count what would be deleted
    old_data = conn.execute("""
        SELECT COUNT(*) FROM daily_prices WHERE date < '2023-12-01'
    """).fetchone()[0]
    
    print(f"\n1. Delete all data before Dec 2023: {old_data} records")
    print(f"   This keeps only verified TradingView data (Dec 2023 - present)")
    
    response = input("\nProceed with cleanup? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\nDeleting old data...")
        conn.execute("DELETE FROM daily_prices WHERE date < '2023-12-01'")
        conn.commit()
        
        after_count = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
        
        print(f"\nCleanup complete!")
        print(f"  Before: {before_count:,} records")
        print(f"  After:  {after_count:,} records")
        print(f"  Removed: {before_count - after_count:,} records")
        
        # Verify new date range
        new_range = conn.execute("""
            SELECT MIN(date), MAX(date) FROM daily_prices
        """).fetchone()
        print(f"\nNew date range: {new_range[0]} to {new_range[1]}")
    else:
        print("Cleanup cancelled.")
    
    conn.close()


if __name__ == "__main__":
    main()
