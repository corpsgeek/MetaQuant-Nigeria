# Database Connection for Daemon

import logging
from typing import Optional
import duckdb

from config import Config

logger = logging.getLogger(__name__)

_connection: Optional[duckdb.DuckDBPyConnection] = None


def get_db_connection(config: Config) -> duckdb.DuckDBPyConnection:
    """Get or create database connection."""
    global _connection
    
    if _connection is None:
        db_path = config.data_dir / 'daemon.db'
        _connection = duckdb.connect(str(db_path))
        _init_tables(_connection)
        logger.info(f"Database connected: {db_path}")
    
    return _connection


def _init_tables(conn: duckdb.DuckDBPyConnection):
    """Initialize database tables."""
    
    # OHLCV cache
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_cache (
            symbol TEXT,
            datetime TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (symbol, datetime)
        )
    """)
    
    # Alerts history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts_history (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            alert_type TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Watchlist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    logger.info("Database tables initialized")
