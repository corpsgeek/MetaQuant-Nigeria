"""Fix and populate market_snapshots table."""

import sys
sys.path.insert(0, '/Users/macbook/Documents/MetaLabs/MetaQuantNigeria')
from src.database.db_manager import DatabaseManager

db = DatabaseManager()

# Drop and recreate market_snapshots table
print('Dropping old market_snapshots table...')
db.conn.execute('DROP TABLE IF EXISTS market_snapshots')

# Create sequence
db.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_market_snapshots START 1")

# Create table with proper ID
db.conn.execute("""
    CREATE TABLE IF NOT EXISTS market_snapshots (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_market_snapshots'),
        date DATE NOT NULL UNIQUE,
        total_volume BIGINT,
        total_trades INTEGER,
        gainers_count INTEGER,
        losers_count INTEGER,
        unchanged_count INTEGER,
        top_gainer_symbol VARCHAR,
        top_gainer_change DECIMAL(8, 4),
        top_loser_symbol VARCHAR,
        top_loser_change DECIMAL(8, 4),
        market_breadth DECIMAL(8, 4),
        avg_change_percent DECIMAL(8, 4),
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
print('✅ Table recreated')

# Now generate market snapshots from price history
print('Generating market snapshots from price history...')

dates = db.get_price_history_dates()
print(f'Found {len(dates)} dates with price data')

count = 0
for date in dates:
    market_data = db.get_market_on_date(date)
    if not market_data:
        continue
    
    # Calculate stats
    gainers = []
    losers = []
    total_volume = 0
    
    for stock in market_data:
        open_p = float(stock.get('open') or 0)
        close = float(stock.get('close') or 0)
        volume = int(stock.get('volume') or 0)
        total_volume += volume
        
        if open_p > 0:
            change = (close - open_p) / open_p * 100
            if change > 0:
                gainers.append((stock['symbol'], change))
            elif change < 0:
                losers.append((stock['symbol'], change))
    
    gainers.sort(key=lambda x: -x[1])
    losers.sort(key=lambda x: x[1])
    
    snapshot = {
        'total_volume': total_volume,
        'gainers_count': len(gainers),
        'losers_count': len(losers),
        'unchanged_count': len(market_data) - len(gainers) - len(losers),
        'top_gainer_symbol': gainers[0][0] if gainers else None,
        'top_gainer_change': gainers[0][1] if gainers else None,
        'top_loser_symbol': losers[0][0] if losers else None,
        'top_loser_change': losers[0][1] if losers else None,
        'market_breadth': (len(gainers) - len(losers)) / max(len(market_data), 1),
        'avg_change_percent': sum(g[1] for g in gainers + losers) / max(len(gainers) + len(losers), 1) if gainers or losers else 0,
    }
    
    db.save_market_snapshot(date, snapshot)
    count += 1

print(f'✅ Generated {count} market snapshots!')
db.close()
