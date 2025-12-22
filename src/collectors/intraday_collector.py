"""
Intraday Data Collector for MetaQuant Nigeria.
Fetches and stores historical OHLCV data at multiple timeframes.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time

try:
    from tvDatafeed import TvDatafeed, Interval
    TVDATAFEED_AVAILABLE = True
except ImportError:
    TVDATAFEED_AVAILABLE = False

logger = logging.getLogger(__name__)


# Interval mapping
INTERVALS = {
    '15m': Interval.in_15_minute if TVDATAFEED_AVAILABLE else None,
    '1h': Interval.in_1_hour if TVDATAFEED_AVAILABLE else None,
    '1d': Interval.in_daily if TVDATAFEED_AVAILABLE else None,
}


class IntradayCollector:
    """Collects and stores intraday OHLCV data."""
    
    def __init__(self, db):
        """
        Initialize the collector.
        
        Args:
            db: DatabaseManager instance
        """
        self.db = db
        self.tv = None
        
        if TVDATAFEED_AVAILABLE:
            self.tv = TvDatafeed()
    
    def get_all_symbols(self) -> List[str]:
        """Get all stock symbols from the database."""
        try:
            results = self.db.conn.execute(
                "SELECT symbol FROM stocks WHERE is_active = true ORDER BY symbol"
            ).fetchall()
            return [r[0] for r in results]
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def fetch_history(
        self,
        symbol: str,
        interval: str = '15m',
        n_bars: int = 1000
    ) -> Optional[List[Dict]]:
        """
        Fetch historical OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'DANGCEM')
            interval: Timeframe ('15m', '1h', '1d')
            n_bars: Number of bars to fetch
            
        Returns:
            List of OHLCV dictionaries or None if failed
        """
        if not TVDATAFEED_AVAILABLE or not self.tv:
            logger.error("tvDatafeed not available")
            return None
        
        if interval not in INTERVALS:
            logger.error(f"Invalid interval: {interval}")
            return None
        
        try:
            data = self.tv.get_hist(
                symbol=symbol,
                exchange='NSENG',
                interval=INTERVALS[interval],
                n_bars=n_bars
            )
            
            if data is None or data.empty:
                logger.warning(f"No data for {symbol} at {interval}")
                return None
            
            # Convert to list of dicts
            records = []
            for dt, row in data.iterrows():
                records.append({
                    'symbol': symbol,
                    'interval': interval,
                    'datetime': dt,
                    'open': float(row['open']) if row['open'] else None,
                    'high': float(row['high']) if row['high'] else None,
                    'low': float(row['low']) if row['low'] else None,
                    'close': float(row['close']) if row['close'] else None,
                    'volume': float(row['volume']) if row['volume'] else None,
                })
            
            return records
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} at {interval}: {e}")
            return None
    
    def store_ohlcv(self, records: List[Dict]) -> int:
        """
        Store OHLCV records in the database.
        
        Args:
            records: List of OHLCV dictionaries
            
        Returns:
            Number of records inserted/updated
        """
        if not records:
            return 0
        
        try:
            # Use INSERT OR REPLACE for upsert behavior
            count = 0
            for r in records:
                try:
                    self.db.conn.execute("""
                        INSERT OR REPLACE INTO intraday_ohlcv 
                        (symbol, interval, datetime, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        r['symbol'], r['interval'], r['datetime'],
                        r['open'], r['high'], r['low'], r['close'], r['volume']
                    ])
                    count += 1
                except Exception as e:
                    logger.warning(f"Error inserting record: {e}")
            
            return count
            
        except Exception as e:
            logger.error(f"Error storing OHLCV data: {e}")
            return 0
    
    def get_latest_datetime(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get the latest stored datetime for a symbol/interval."""
        try:
            result = self.db.conn.execute("""
                SELECT MAX(datetime) FROM intraday_ohlcv
                WHERE symbol = ? AND interval = ?
            """, [symbol, interval]).fetchone()
            
            return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"Error getting latest datetime: {e}")
            return None
    
    def backfill_symbol(
        self,
        symbol: str,
        intervals: List[str] = None,
        n_bars: int = 1000
    ) -> Dict[str, int]:
        """
        Backfill historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            intervals: List of intervals to fetch (default: ['15m', '1h', '1d'])
            n_bars: Number of bars per interval
            
        Returns:
            Dict of interval -> records count
        """
        if intervals is None:
            intervals = ['15m', '1h', '1d']
        
        results = {}
        
        for interval in intervals:
            logger.info(f"Fetching {symbol} at {interval}...")
            records = self.fetch_history(symbol, interval, n_bars)
            
            if records:
                count = self.store_ohlcv(records)
                results[interval] = count
                logger.info(f"  Stored {count} bars for {symbol} at {interval}")
            else:
                results[interval] = 0
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def backfill_all(
        self,
        intervals: List[str] = None,
        n_bars: int = 1000,
        limit: int = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Backfill historical data for all symbols.
        
        Args:
            intervals: List of intervals (default: ['15m', '1h', '1d'])
            n_bars: Number of bars per interval
            limit: Max symbols to process (None = all)
            
        Returns:
            Dict of symbol -> interval -> records count
        """
        symbols = self.get_all_symbols()
        
        if limit:
            symbols = symbols[:limit]
        
        logger.info(f"Backfilling {len(symbols)} symbols...")
        
        all_results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Processing {symbol}")
            results = self.backfill_symbol(symbol, intervals, n_bars)
            all_results[symbol] = results
            
            # Slightly longer delay between symbols
            time.sleep(1)
        
        return all_results
    
    def incremental_sync(
        self,
        symbol: str,
        interval: str = '15m',
        n_bars: int = 100
    ) -> int:
        """
        Incrementally sync new data for a symbol.
        Only fetches data newer than what's already stored.
        
        Args:
            symbol: Stock symbol
            interval: Timeframe
            n_bars: Number of recent bars to check
            
        Returns:
            Number of new records added
        """
        # Fetch recent data
        records = self.fetch_history(symbol, interval, n_bars)
        
        if not records:
            return 0
        
        # Get latest stored datetime
        latest = self.get_latest_datetime(symbol, interval)
        
        if latest:
            # Filter to only new records
            records = [r for r in records if r['datetime'] > latest]
        
        if not records:
            logger.debug(f"No new data for {symbol} at {interval}")
            return 0
        
        # Store new records
        count = self.store_ohlcv(records)
        logger.info(f"Added {count} new bars for {symbol} at {interval}")
        
        return count
    
    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Query stored OHLCV data.
        
        Args:
            symbol: Stock symbol
            interval: Timeframe
            start_date: Start datetime filter
            end_date: End datetime filter
            limit: Max records to return
            
        Returns:
            List of OHLCV dictionaries
        """
        try:
            query = """
                SELECT symbol, interval, datetime, open, high, low, close, volume
                FROM intraday_ohlcv
                WHERE symbol = ? AND interval = ?
            """
            params = [symbol, interval]
            
            if start_date:
                query += " AND datetime >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND datetime <= ?"
                params.append(end_date)
            
            query += " ORDER BY datetime DESC LIMIT ?"
            params.append(limit)
            
            results = self.db.conn.execute(query, params).fetchall()
            
            return [
                {
                    'symbol': r[0],
                    'interval': r[1],
                    'datetime': r[2],
                    'open': r[3],
                    'high': r[4],
                    'low': r[5],
                    'close': r[6],
                    'volume': r[7]
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Error querying OHLCV: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics about stored data."""
        try:
            result = self.db.conn.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as symbols,
                    COUNT(*) as total_bars,
                    MIN(datetime) as earliest,
                    MAX(datetime) as latest,
                    interval
                FROM intraday_ohlcv
                GROUP BY interval
            """).fetchall()
            
            stats = {
                'total_symbols': 0,
                'intervals': {}
            }
            
            symbols_set = set()
            for r in result:
                stats['intervals'][r[4]] = {
                    'symbols': r[0],
                    'bars': r[1],
                    'earliest': r[2],
                    'latest': r[3]
                }
            
            # Get total unique symbols
            count_result = self.db.conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM intraday_ohlcv"
            ).fetchone()
            stats['total_symbols'] = count_result[0] if count_result else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
