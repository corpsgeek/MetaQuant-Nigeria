"""
Historical Data Seeder
Generates realistic historical price data for all stocks in the database.
Creates market snapshots for each trading day.
"""

import os
import sys
import random
from datetime import datetime, timedelta
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db_manager import DatabaseManager


def generate_price_series(
    start_price: float, 
    days: int, 
    volatility: float = 0.02,
    trend: float = 0.0001
) -> List[Dict]:
    """
    Generate a realistic price series using random walk with drift.
    
    Args:
        start_price: Starting price
        days: Number of days to generate
        volatility: Daily volatility (std dev of returns)
        trend: Daily trend/drift
    """
    prices = []
    price = start_price
    
    for i in range(days):
        # Random return with drift
        daily_return = random.gauss(trend, volatility)
        price = price * (1 + daily_return)
        
        # Generate OHLC
        open_price = price * (1 + random.uniform(-0.005, 0.005))
        high = max(open_price, price) * (1 + random.uniform(0, 0.015))
        low = min(open_price, price) * (1 - random.uniform(0, 0.015))
        close = price
        volume = int(random.gauss(1000000, 500000))
        if volume < 10000:
            volume = 10000
        
        prices.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
        })
        
    return prices


def seed_historical_data(db: DatabaseManager, days: int = 90):
    """
    Seed historical price data for all stocks.
    
    Args:
        db: Database manager instance
        days: Number of days of history to generate
    """
    stocks = db.get_all_stocks()
    print(f"Seeding {days} days of history for {len(stocks)} stocks...")
    
    start_date = datetime.now() - timedelta(days=days)
    
    # Track daily market stats for snapshots
    daily_stats = {i: {'gainers': [], 'losers': [], 'volumes': []} for i in range(days)}
    
    for stock in stocks:
        stock_id = stock['id']
        symbol = stock['symbol']
        start_price = float(stock.get('last_price') or random.uniform(10, 100))
        
        # Generate price series
        volatility = random.uniform(0.015, 0.035)  # 1.5% to 3.5% daily vol
        trend = random.uniform(-0.0005, 0.001)  # Slight trend
        prices = generate_price_series(start_price, days, volatility, trend)
        
        # Insert into database
        for i, price_data in enumerate(prices):
            date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            
            # Skip weekends
            day_of_week = (start_date + timedelta(days=i)).weekday()
            if day_of_week >= 5:
                continue
            
            db.insert_daily_price(stock_id, date, price_data)
            
            # Track for market snapshot
            if i > 0:
                change = (price_data['close'] - prices[i-1]['close']) / prices[i-1]['close'] * 100
                if change > 0:
                    daily_stats[i]['gainers'].append((symbol, change))
                elif change < 0:
                    daily_stats[i]['losers'].append((symbol, change))
            
            daily_stats[i]['volumes'].append(price_data['volume'])
        
        print(f"  {symbol}: {len(prices)} days")
    
    # Create market snapshots
    print("\nCreating market snapshots...")
    for i in range(days):
        date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        day_of_week = (start_date + timedelta(days=i)).weekday()
        if day_of_week >= 5:
            continue
        
        stats = daily_stats[i]
        gainers = sorted(stats['gainers'], key=lambda x: -x[1])
        losers = sorted(stats['losers'], key=lambda x: x[1])
        
        snapshot = {
            'total_volume': sum(stats['volumes']),
            'gainers_count': len(gainers),
            'losers_count': len(losers),
            'unchanged_count': len(stocks) - len(gainers) - len(losers),
            'top_gainer_symbol': gainers[0][0] if gainers else None,
            'top_gainer_change': gainers[0][1] if gainers else None,
            'top_loser_symbol': losers[0][0] if losers else None,
            'top_loser_change': losers[0][1] if losers else None,
            'market_breadth': (len(gainers) - len(losers)) / max(len(stocks), 1),
            'avg_change_percent': sum(g[1] for g in gainers + losers) / max(len(gainers) + len(losers), 1),
        }
        
        db.save_market_snapshot(date, snapshot)
    
    # Verify
    dates = db.get_available_dates(limit=10)
    print(f"\n✅ Created {len(db.get_available_dates(limit=365))} market snapshots")
    print(f"Recent dates: {dates[:5]}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed historical price data")
    parser.add_argument("--days", type=int, default=90, help="Days of history to generate")
    
    args = parser.parse_args()
    
    db = DatabaseManager()
    db.initialize()
    
    try:
        seed_historical_data(db, days=args.days)
        print("\n✅ Historical data seeding complete!")
    finally:
        db.close()


if __name__ == "__main__":
    main()
