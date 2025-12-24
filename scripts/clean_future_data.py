#!/usr/bin/env python3
"""
Clean synthetic future data from the database.
Removes all daily_prices records with dates after today.
"""

import duckdb
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'metaquant.db')

def main():
    print(f"Connecting to database: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)
    
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Today's date: {today}")
    
    # Check what we're dealing with
    result = conn.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM daily_prices").fetchone()
    print(f"\nCurrent daily_prices:")
    print(f"  Date range: {result[0]} to {result[1]}")
    print(f"  Total records: {result[2]:,}")
    
    # Count future records
    future_count = conn.execute(
        "SELECT COUNT(*) FROM daily_prices WHERE date > ?", [today]
    ).fetchone()[0]
    print(f"  Future records (after {today}): {future_count:,}")
    
    if future_count == 0:
        print("\n✅ No future data found. Database is clean.")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  About to DELETE {future_count:,} future records!")
    confirm = input("Type 'yes' to proceed: ")
    
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return
    
    # Delete future data
    print("\nDeleting future data...")
    conn.execute("DELETE FROM daily_prices WHERE date > ?", [today])
    
    # Verify
    result = conn.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM daily_prices").fetchone()
    print(f"\n✅ Cleanup complete!")
    print(f"  Date range: {result[0]} to {result[1]}")
    print(f"  Total records: {result[2]:,}")
    
    conn.close()
    
    conn.close()

if __name__ == "__main__":
    main()
