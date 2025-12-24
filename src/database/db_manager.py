"""
DuckDB Database Manager for MetaQuant Nigeria.
Handles connection management, schema migrations, and core database operations.
"""

import os
import duckdb
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class DatabaseManager:
    """Manages DuckDB database connections and operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the DuckDB database file. Defaults to data/metaquant.db
        """
        if db_path is None:
            # Default to data directory in project root
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "metaquant.db")
        
        self.db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn
    
    def close(self):
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def initialize(self):
        """Initialize the database schema."""
        self._create_tables()
        self._create_indexes()
    
    def _create_tables(self):
        """Create all required database tables."""
        
        # Create sequences for auto-increment
        sequences = [
            'seq_stocks', 'seq_fundamentals', 'seq_daily_prices',
            'seq_orderbook', 'seq_disclosures', 'seq_portfolios',
            'seq_positions', 'seq_transactions', 'seq_watchlists',
            'seq_watchlist_items', 'seq_ai_insights', 'seq_technical',
            'seq_market_snapshots'
        ]
        for seq in sequences:
            self.conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq} START 1")
        
        # Stocks table - core equity information
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_stocks'),
                symbol VARCHAR NOT NULL UNIQUE,
                name VARCHAR NOT NULL,
                sector VARCHAR,
                subsector VARCHAR,
                last_price DECIMAL(18, 4),
                prev_close DECIMAL(18, 4),
                change_percent DECIMAL(8, 4),
                volume BIGINT,
                market_cap DECIMAL(24, 2),
                last_updated TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Fundamentals table - company fundamentals
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                id INTEGER PRIMARY KEY,
                stock_id INTEGER REFERENCES stocks(id),
                pe_ratio DECIMAL(12, 4),
                eps DECIMAL(18, 4),
                dividend_yield DECIMAL(8, 4),
                dividend_per_share DECIMAL(18, 4),
                book_value DECIMAL(18, 4),
                pb_ratio DECIMAL(12, 4),
                revenue DECIMAL(24, 2),
                net_income DECIMAL(24, 2),
                roe DECIMAL(8, 4),
                debt_to_equity DECIMAL(12, 4),
                current_ratio DECIMAL(12, 4),
                fiscal_year_end DATE,
                last_updated TIMESTAMP,
                UNIQUE(stock_id)
            )
        """)
        
        # Daily prices - historical OHLCV data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                id INTEGER PRIMARY KEY,
                stock_id INTEGER REFERENCES stocks(id),
                date DATE NOT NULL,
                open DECIMAL(18, 4),
                high DECIMAL(18, 4),
                low DECIMAL(18, 4),
                close DECIMAL(18, 4),
                volume BIGINT,
                trades INTEGER,
                UNIQUE(stock_id, date)
            )
        """)
        
        # Orderbook snapshots - from IDIA Infoware
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                id INTEGER PRIMARY KEY,
                stock_id INTEGER REFERENCES stocks(id),
                timestamp TIMESTAMP NOT NULL,
                bid_price_1 DECIMAL(18, 4),
                bid_volume_1 BIGINT,
                bid_price_2 DECIMAL(18, 4),
                bid_volume_2 BIGINT,
                bid_price_3 DECIMAL(18, 4),
                bid_volume_3 BIGINT,
                bid_price_4 DECIMAL(18, 4),
                bid_volume_4 BIGINT,
                bid_price_5 DECIMAL(18, 4),
                bid_volume_5 BIGINT,
                ask_price_1 DECIMAL(18, 4),
                ask_volume_1 BIGINT,
                ask_price_2 DECIMAL(18, 4),
                ask_volume_2 BIGINT,
                ask_price_3 DECIMAL(18, 4),
                ask_volume_3 BIGINT,
                ask_price_4 DECIMAL(18, 4),
                ask_volume_4 BIGINT,
                ask_price_5 DECIMAL(18, 4),
                ask_volume_5 BIGINT
            )
        """)
        
        # Corporate disclosures - from NGX
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS corporate_disclosures (
                id INTEGER PRIMARY KEY,
                stock_id INTEGER REFERENCES stocks(id),
                disclosure_date DATE NOT NULL,
                title VARCHAR NOT NULL,
                content TEXT,
                disclosure_type VARCHAR,
                source_url VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Portfolios - user-created portfolios
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Positions - holdings in portfolios
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                portfolio_id INTEGER REFERENCES portfolios(id),
                stock_id INTEGER REFERENCES stocks(id),
                quantity DECIMAL(18, 4) NOT NULL,
                avg_cost DECIMAL(18, 4) NOT NULL,
                date_acquired DATE,
                notes TEXT,
                UNIQUE(portfolio_id, stock_id)
            )
        """)
        
        # Transactions - buy/sell history
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY,
                portfolio_id INTEGER REFERENCES portfolios(id),
                stock_id INTEGER REFERENCES stocks(id),
                transaction_type VARCHAR NOT NULL,  -- 'BUY' or 'SELL'
                quantity DECIMAL(18, 4) NOT NULL,
                price DECIMAL(18, 4) NOT NULL,
                fees DECIMAL(18, 4) DEFAULT 0,
                transaction_date TIMESTAMP NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Watchlists
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Watchlist items
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_items (
                id INTEGER PRIMARY KEY,
                watchlist_id INTEGER REFERENCES watchlists(id),
                stock_id INTEGER REFERENCES stocks(id),
                target_price DECIMAL(18, 4),
                alert_above DECIMAL(18, 4),
                alert_below DECIMAL(18, 4),
                notes TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(watchlist_id, stock_id)
            )
        """)
        
        # AI insights cache
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_insights (
                id INTEGER PRIMARY KEY,
                stock_id INTEGER REFERENCES stocks(id),
                insight_type VARCHAR NOT NULL,
                content TEXT NOT NULL,
                model_used VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # Technical indicators cache
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY,
                stock_id INTEGER REFERENCES stocks(id),
                date DATE NOT NULL,
                sma_20 DECIMAL(18, 4),
                sma_50 DECIMAL(18, 4),
                sma_200 DECIMAL(18, 4),
                rsi_14 DECIMAL(8, 4),
                macd DECIMAL(18, 4),
                macd_signal DECIMAL(18, 4),
                macd_histogram DECIMAL(18, 4),
                bollinger_upper DECIMAL(18, 4),
                bollinger_lower DECIMAL(18, 4),
                atr_14 DECIMAL(18, 4),
                UNIQUE(stock_id, date)
            )
        """)
        
        # Market snapshots - daily market summary for historical memory
        self.conn.execute("""
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
        
        # Intraday OHLCV data for backtesting and analysis
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS intraday_ohlcv (
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,  -- '15m', '1h', '1d'
                datetime TIMESTAMP NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, interval, datetime)
            )
        """)
        
        # Fundamental snapshots - daily P/E, EPS, etc. for historical tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_snapshots (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                price DOUBLE,
                pe_ratio DOUBLE,
                eps DOUBLE,
                pb_ratio DOUBLE,
                ps_ratio DOUBLE,
                dividend_yield DOUBLE,
                market_cap DOUBLE,
                volume DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        # Add ps_ratio column if it doesn't exist (for existing databases)
        try:
            self.conn.execute("ALTER TABLE fundamental_snapshots ADD COLUMN ps_ratio DOUBLE")
        except Exception:
            pass  # Column already exists
        
        # Signal history - for backtesting with historical ML/composite signals
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                ml_score DOUBLE,
                anomaly_score DOUBLE,
                cluster_id INTEGER,
                flow_score DOUBLE,
                fundamental_score DOUBLE,
                intel_score DOUBLE,
                momentum_score DOUBLE,
                composite_score DOUBLE,
                signal VARCHAR,
                components JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        # AI Portfolio Manager - Positions
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_portfolio_positions (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                shares INTEGER NOT NULL,
                entry_price DOUBLE NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                stop_loss DOUBLE,
                take_profit DOUBLE,
                sector VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol)
            )
        """)
        
        # AI Portfolio Manager - Trades
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_portfolio_trades (
                id INTEGER PRIMARY KEY,
                trade_date TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                shares INTEGER NOT NULL,
                price DOUBLE NOT NULL,
                entry_price DOUBLE,
                value DOUBLE,
                pnl DOUBLE,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # AI Portfolio Manager - State
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_portfolio_state (
                id INTEGER PRIMARY KEY DEFAULT 1,
                cash DOUBLE NOT NULL,
                equity DOUBLE NOT NULL,
                config JSON,
                start_date TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_indexes(self):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector)",
            "CREATE INDEX IF NOT EXISTS idx_daily_prices_stock_date ON daily_prices(stock_id, date)",
            "CREATE INDEX IF NOT EXISTS idx_orderbook_stock_time ON orderbook_snapshots(stock_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_disclosures_stock ON corporate_disclosures(stock_id)",
            "CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON positions(portfolio_id)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_portfolio ON transactions(portfolio_id)",
            "CREATE INDEX IF NOT EXISTS idx_watchlist_items ON watchlist_items(watchlist_id)",
        ]
        for idx in indexes:
            self.conn.execute(idx)
    
    # ==================== Stock Operations ====================
    
    def upsert_stock(self, stock_data: Dict[str, Any]) -> int:
        """Insert or update a stock."""
        self.conn.execute("""
            INSERT INTO stocks (symbol, name, sector, subsector, last_price, 
                               prev_close, change_percent, volume, market_cap, 
                               last_updated, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                subsector = EXCLUDED.subsector,
                last_price = EXCLUDED.last_price,
                prev_close = EXCLUDED.prev_close,
                change_percent = EXCLUDED.change_percent,
                volume = EXCLUDED.volume,
                market_cap = EXCLUDED.market_cap,
                last_updated = EXCLUDED.last_updated,
                is_active = EXCLUDED.is_active
        """, [
            stock_data.get('symbol'),
            stock_data.get('name'),
            stock_data.get('sector'),
            stock_data.get('subsector'),
            stock_data.get('last_price'),
            stock_data.get('prev_close'),
            stock_data.get('change_percent'),
            stock_data.get('volume'),
            stock_data.get('market_cap'),
            stock_data.get('last_updated', datetime.now()),
            stock_data.get('is_active', True)
        ])
        
        result = self.conn.execute(
            "SELECT id FROM stocks WHERE symbol = ?", 
            [stock_data.get('symbol')]
        ).fetchone()
        return result[0] if result else -1
    
    def get_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock by symbol."""
        result = self.conn.execute(
            "SELECT * FROM stocks WHERE symbol = ?", [symbol]
        ).fetchone()
        if result:
            columns = [desc[0] for desc in self.conn.description]
            return dict(zip(columns, result))
        return None
    
    def get_all_stocks(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all stocks."""
        query = "SELECT * FROM stocks"
        if active_only:
            query += " WHERE is_active = TRUE"
        query += " ORDER BY symbol"
        
        results = self.conn.execute(query).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    def search_stocks(self, query: str) -> List[Dict[str, Any]]:
        """Search stocks by symbol or name."""
        results = self.conn.execute("""
            SELECT * FROM stocks 
            WHERE symbol ILIKE ? OR name ILIKE ?
            ORDER BY symbol
        """, [f"%{query}%", f"%{query}%"]).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    # ==================== Daily Price Operations ====================
    
    def insert_daily_price(self, stock_id: int, date: str, ohlcv: Dict[str, Any]):
        """Insert daily price data."""
        self.conn.execute("""
            INSERT INTO daily_prices (stock_id, date, open, high, low, close, volume, change_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (stock_id, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                change_pct = EXCLUDED.change_pct
        """, [
            stock_id, date, 
            ohlcv.get('open'), ohlcv.get('high'), 
            ohlcv.get('low'), ohlcv.get('close'),
            ohlcv.get('volume'), ohlcv.get('change_pct')
        ])
    
    def get_price_history(self, stock_id: int, days: int = 365) -> List[Dict[str, Any]]:
        """Get price history for a stock."""
        results = self.conn.execute("""
            SELECT * FROM daily_prices 
            WHERE stock_id = ? 
            ORDER BY date DESC 
            LIMIT ?
        """, [stock_id, days]).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    # ==================== Portfolio Operations ====================
    
    def create_portfolio(self, name: str, description: str = "") -> int:
        """Create a new portfolio."""
        self.conn.execute("""
            INSERT INTO portfolios (name, description)
            VALUES (?, ?)
        """, [name, description])
        result = self.conn.execute(
            "SELECT id FROM portfolios WHERE name = ?", [name]
        ).fetchone()
        return result[0] if result else -1
    
    def get_portfolios(self) -> List[Dict[str, Any]]:
        """Get all portfolios."""
        results = self.conn.execute(
            "SELECT * FROM portfolios ORDER BY name"
        ).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    def get_portfolio_positions(self, portfolio_id: int) -> List[Dict[str, Any]]:
        """Get all positions in a portfolio with current stock data."""
        results = self.conn.execute("""
            SELECT p.*, s.symbol, s.name, s.last_price, s.change_percent,
                   (p.quantity * s.last_price) as market_value,
                   (p.quantity * s.last_price) - (p.quantity * p.avg_cost) as unrealized_pnl,
                   ((s.last_price - p.avg_cost) / p.avg_cost * 100) as return_percent
            FROM positions p
            JOIN stocks s ON p.stock_id = s.id
            WHERE p.portfolio_id = ?
            ORDER BY s.symbol
        """, [portfolio_id]).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    def upsert_position(self, portfolio_id: int, stock_id: int, 
                        quantity: float, avg_cost: float, 
                        date_acquired: Optional[str] = None):
        """Insert or update a position."""
        self.conn.execute("""
            INSERT INTO positions (portfolio_id, stock_id, quantity, avg_cost, date_acquired)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (portfolio_id, stock_id) DO UPDATE SET
                quantity = EXCLUDED.quantity,
                avg_cost = EXCLUDED.avg_cost,
                date_acquired = COALESCE(EXCLUDED.date_acquired, positions.date_acquired)
        """, [portfolio_id, stock_id, quantity, avg_cost, date_acquired])
    
    def delete_position(self, portfolio_id: int, stock_id: int):
        """Delete a position from a portfolio."""
        self.conn.execute("""
            DELETE FROM positions 
            WHERE portfolio_id = ? AND stock_id = ?
        """, [portfolio_id, stock_id])
    
    # ==================== Watchlist Operations ====================
    
    def create_watchlist(self, name: str, description: str = "") -> int:
        """Create a new watchlist."""
        self.conn.execute("""
            INSERT INTO watchlists (name, description)
            VALUES (?, ?)
        """, [name, description])
        result = self.conn.execute(
            "SELECT id FROM watchlists WHERE name = ?", [name]
        ).fetchone()
        return result[0] if result else -1
    
    def get_watchlists(self) -> List[Dict[str, Any]]:
        """Get all watchlists."""
        results = self.conn.execute(
            "SELECT * FROM watchlists ORDER BY name"
        ).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    def add_to_watchlist(self, watchlist_id: int, stock_id: int,
                         target_price: Optional[float] = None,
                         alert_above: Optional[float] = None,
                         alert_below: Optional[float] = None):
        """Add a stock to a watchlist."""
        self.conn.execute("""
            INSERT INTO watchlist_items (watchlist_id, stock_id, target_price, 
                                         alert_above, alert_below)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (watchlist_id, stock_id) DO UPDATE SET
                target_price = EXCLUDED.target_price,
                alert_above = EXCLUDED.alert_above,
                alert_below = EXCLUDED.alert_below
        """, [watchlist_id, stock_id, target_price, alert_above, alert_below])
    
    def get_watchlist_items(self, watchlist_id: int) -> List[Dict[str, Any]]:
        """Get all items in a watchlist with current stock data."""
        results = self.conn.execute("""
            SELECT wi.*, s.symbol, s.name, s.last_price, s.change_percent,
                   s.sector, s.volume
            FROM watchlist_items wi
            JOIN stocks s ON wi.stock_id = s.id
            WHERE wi.watchlist_id = ?
            ORDER BY s.symbol
        """, [watchlist_id]).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    # ==================== Screening Operations ====================
    
    def screen_stocks(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Screen stocks based on fundamental and technical filters.
        
        Args:
            filters: Dictionary of filter conditions like:
                - pe_max: Maximum P/E ratio
                - pe_min: Minimum P/E ratio
                - dividend_min: Minimum dividend yield
                - market_cap_min: Minimum market cap
                - sector: Sector filter
        """
        query = """
            SELECT s.*, f.pe_ratio, f.eps, f.dividend_yield, f.book_value,
                   f.pb_ratio, f.roe
            FROM stocks s
            LEFT JOIN fundamentals f ON s.id = f.stock_id
            WHERE s.is_active = TRUE
        """
        params = []
        
        if filters.get('pe_max') is not None:
            query += " AND f.pe_ratio <= ?"
            params.append(filters['pe_max'])
        
        if filters.get('pe_min') is not None:
            query += " AND f.pe_ratio >= ?"
            params.append(filters['pe_min'])
        
        if filters.get('dividend_min') is not None:
            query += " AND f.dividend_yield >= ?"
            params.append(filters['dividend_min'])
        
        if filters.get('market_cap_min') is not None:
            query += " AND s.market_cap >= ?"
            params.append(filters['market_cap_min'])
        
        if filters.get('market_cap_max') is not None:
            query += " AND s.market_cap <= ?"
            params.append(filters['market_cap_max'])
        
        if filters.get('sector'):
            query += " AND s.sector = ?"
            params.append(filters['sector'])
        
        query += " ORDER BY s.symbol"
        
        results = self.conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    def get_sectors(self) -> List[str]:
        """Get list of unique sectors."""
        results = self.conn.execute("""
            SELECT DISTINCT sector FROM stocks 
            WHERE sector IS NOT NULL 
            ORDER BY sector
        """).fetchall()
        return [row[0] for row in results]
    
    # ===== Market Snapshots / Historical Memory =====
    
    def save_market_snapshot(self, date: str, snapshot_data: Dict[str, Any]) -> int:
        """
        Save a market snapshot for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            snapshot_data: Dictionary with market summary data
        """
        self.conn.execute("""
            INSERT INTO market_snapshots (
                date, total_volume, total_trades, gainers_count, losers_count,
                unchanged_count, top_gainer_symbol, top_gainer_change,
                top_loser_symbol, top_loser_change, market_breadth, avg_change_percent, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (date) DO UPDATE SET
                total_volume = EXCLUDED.total_volume,
                gainers_count = EXCLUDED.gainers_count,
                losers_count = EXCLUDED.losers_count,
                top_gainer_symbol = EXCLUDED.top_gainer_symbol,
                top_gainer_change = EXCLUDED.top_gainer_change,
                top_loser_symbol = EXCLUDED.top_loser_symbol,
                top_loser_change = EXCLUDED.top_loser_change,
                market_breadth = EXCLUDED.market_breadth,
                avg_change_percent = EXCLUDED.avg_change_percent
        """, [
            date,
            snapshot_data.get('total_volume', 0),
            snapshot_data.get('total_trades', 0),
            snapshot_data.get('gainers_count', 0),
            snapshot_data.get('losers_count', 0),
            snapshot_data.get('unchanged_count', 0),
            snapshot_data.get('top_gainer_symbol'),
            snapshot_data.get('top_gainer_change'),
            snapshot_data.get('top_loser_symbol'),
            snapshot_data.get('top_loser_change'),
            snapshot_data.get('market_breadth'),
            snapshot_data.get('avg_change_percent'),
            snapshot_data.get('notes'),
        ])
        return 1
    
    def get_market_snapshot(self, date: str) -> Optional[Dict]:
        """Get market snapshot for a specific date."""
        result = self.conn.execute("""
            SELECT * FROM market_snapshots WHERE date = ?
        """, [date]).fetchone()
        
        if result:
            columns = [desc[0] for desc in self.conn.description]
            return dict(zip(columns, result))
        return None
    
    def get_available_dates(self, limit: int = 100) -> List[str]:
        """Get list of dates with available market snapshots."""
        results = self.conn.execute("""
            SELECT date FROM market_snapshots 
            ORDER BY date DESC 
            LIMIT ?
        """, [limit]).fetchall()
        return [str(row[0]) for row in results]
    
    def get_market_on_date(self, date: str) -> List[Dict]:
        """
        Get all stock prices as they were on a specific date.
        Returns stocks with their prices from daily_prices table for that date,
        including the stored change_pct for accurate day-over-day change.
        """
        results = self.conn.execute("""
            SELECT 
                s.symbol, s.name, s.sector,
                dp.open, dp.high, dp.low, dp.close, dp.volume,
                dp.date, dp.change_pct
            FROM stocks s
            JOIN daily_prices dp ON s.id = dp.stock_id
            WHERE dp.date = ?
            ORDER BY s.symbol
        """, [date]).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    def insert_orderbook_snapshot(self, stock_id: int, orderbook_data: Dict[str, Any]):
        """
        Insert order book snapshot data.
        
        Args:
            stock_id: Stock ID
            orderbook_data: Dictionary with bid/ask levels
        """
        self.conn.execute("""
            INSERT INTO orderbook_snapshots (
                stock_id, timestamp,
                bid_price_1, bid_volume_1, bid_price_2, bid_volume_2,
                bid_price_3, bid_volume_3, bid_price_4, bid_volume_4,
                bid_price_5, bid_volume_5,
                ask_price_1, ask_volume_1, ask_price_2, ask_volume_2,
                ask_price_3, ask_volume_3, ask_price_4, ask_volume_4,
                ask_price_5, ask_volume_5
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            stock_id,
            orderbook_data.get('timestamp', datetime.now()),
            orderbook_data.get('bid_price_1'), orderbook_data.get('bid_volume_1'),
            orderbook_data.get('bid_price_2'), orderbook_data.get('bid_volume_2'),
            orderbook_data.get('bid_price_3'), orderbook_data.get('bid_volume_3'),
            orderbook_data.get('bid_price_4'), orderbook_data.get('bid_volume_4'),
            orderbook_data.get('bid_price_5'), orderbook_data.get('bid_volume_5'),
            orderbook_data.get('ask_price_1'), orderbook_data.get('ask_volume_1'),
            orderbook_data.get('ask_price_2'), orderbook_data.get('ask_volume_2'),
            orderbook_data.get('ask_price_3'), orderbook_data.get('ask_volume_3'),
            orderbook_data.get('ask_price_4'), orderbook_data.get('ask_volume_4'),
            orderbook_data.get('ask_price_5'), orderbook_data.get('ask_volume_5'),
        ])
    
    def get_price_history_dates(self) -> List[str]:
        """Get list of dates with price history data."""
        results = self.conn.execute("""
            SELECT DISTINCT date FROM daily_prices 
            ORDER BY date DESC 
            LIMIT 1000
        """).fetchall()
        return [str(row[0]) for row in results]
    
    # ===== Fundamental Snapshots (Historical P/E, EPS, etc.) =====
    
    def save_fundamental_snapshot(self, symbol: str, date: str, data: Dict[str, Any]) -> int:
        """
        Save a fundamental snapshot for a stock on a specific date.
        
        Args:
            symbol: Stock symbol
            date: Date string in YYYY-MM-DD format
            data: Dictionary with price, pe_ratio, eps, etc.
        """
        self.conn.execute("""
            INSERT INTO fundamental_snapshots (
                symbol, date, price, pe_ratio, eps, pb_ratio, ps_ratio,
                dividend_yield, market_cap, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, date) DO UPDATE SET
                price = EXCLUDED.price,
                pe_ratio = EXCLUDED.pe_ratio,
                eps = EXCLUDED.eps,
                pb_ratio = EXCLUDED.pb_ratio,
                ps_ratio = EXCLUDED.ps_ratio,
                dividend_yield = EXCLUDED.dividend_yield,
                market_cap = EXCLUDED.market_cap,
                volume = EXCLUDED.volume
        """, [
            symbol.upper(),
            date,
            data.get('price'),
            data.get('pe_ratio'),
            data.get('eps'),
            data.get('pb_ratio'),
            data.get('ps_ratio'),
            data.get('dividend_yield'),
            data.get('market_cap'),
            data.get('volume')
        ])
        return 1
    
    def get_fundamental_history(self, symbol: str, limit: int = 365) -> List[Dict]:
        """
        Get historical fundamental snapshots for a stock.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of days to return
            
        Returns:
            List of dictionaries with date and metrics
        """
        results = self.conn.execute("""
            SELECT symbol, date, price, pe_ratio, eps, pb_ratio, ps_ratio,
                   dividend_yield, market_cap, volume, created_at
            FROM fundamental_snapshots 
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
        """, [symbol.upper(), limit]).fetchall()
        
        columns = ['symbol', 'date', 'price', 'pe_ratio', 'eps', 'pb_ratio', 'ps_ratio',
                   'dividend_yield', 'market_cap', 'volume', 'created_at']
        return [dict(zip(columns, row)) for row in results]
    
    def save_all_fundamental_snapshots(self, stocks_data: List[Dict], date: str = None) -> int:
        """
        Save fundamental snapshots for multiple stocks at once.
        
        Args:
            stocks_data: List of stock data dictionaries
            date: Date string (defaults to today)
            
        Returns:
            Number of snapshots saved
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        count = 0
        for stock in stocks_data:
            symbol = stock.get('symbol') or stock.get('ticker', '').replace('NSENG:', '')
            if symbol:
                self.save_fundamental_snapshot(symbol, date, {
                    'price': stock.get('close'),
                    'pe_ratio': stock.get('price_earnings_ttm'),
                    'eps': stock.get('earnings_per_share_basic_ttm'),
                    'pb_ratio': stock.get('price_book_ratio'),
                    'dividend_yield': stock.get('dividend_yield_recent'),
                    'market_cap': stock.get('market_cap_basic'),
                    'volume': stock.get('volume')
                })
                count += 1
        return count
    
    def backfill_fundamental_history(self, symbol: str, current_fundamentals: Dict[str, Any], 
                                      interval: str = '1d', limit: int = 365) -> int:
        """
        Backfill historical fundamental data by deriving P/E, P/S, P/BV from price history.
        
        Uses current EPS, Revenue, Book Value and applies to historical prices.
        Formula:
        - P/E = Price / EPS
        - P/S = Market Cap / Revenue (approximated as Price * Shares / Revenue)
        - P/B = Price / Book Value per Share
        
        Args:
            symbol: Stock symbol
            current_fundamentals: Dict with current eps, book_value, shares_outstanding, revenue
            interval: Price interval to use ('15m', '1h', '1d')
            limit: Maximum number of historical days
            
        Returns:
            Number of snapshots created
        """
        eps = current_fundamentals.get('eps')
        book_value = current_fundamentals.get('book_value')
        shares = current_fundamentals.get('shares_outstanding')
        revenue = current_fundamentals.get('revenue')
        dividend_yield = current_fundamentals.get('dividend_yield')
        
        # Get historical prices - aggregate by date
        results = self.conn.execute("""
            SELECT 
                DATE(datetime) as date,
                AVG(close) as avg_close,
                SUM(volume) as total_volume
            FROM intraday_ohlcv
            WHERE symbol = ? AND interval = ?
            GROUP BY DATE(datetime)
            ORDER BY date DESC
            LIMIT ?
        """, [symbol, interval, limit]).fetchall()
        
        if not results:
            return 0
        
        count = 0
        for row in results:
            date_str = str(row[0])
            price = row[1]
            volume = row[2]
            
            if price is None or price <= 0:
                continue
            
            # Calculate derived ratios
            pe_ratio = price / eps if eps and eps > 0 else None
            pb_ratio = price / book_value if book_value and book_value > 0 else None
            
            # P/S = Market Cap / Revenue = (Price * Shares) / Revenue
            ps_ratio = None
            market_cap = None
            if shares and shares > 0:
                market_cap = price * shares
                if revenue and revenue > 0:
                    ps_ratio = market_cap / revenue
            
            # Save snapshot
            self.save_fundamental_snapshot(symbol, date_str, {
                'price': price,
                'pe_ratio': pe_ratio,
                'eps': eps,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'dividend_yield': dividend_yield,
                'market_cap': market_cap,
                'volume': volume
            })
            count += 1
        
        return count
    
    # ==================== Signal History ====================
    
    def save_signal_history(self, symbol: str, date: str, signals: Dict[str, Any]):
        """Save signal scores for a stock on a date."""
        import json
        self.conn.execute("""
            INSERT INTO signal_history (
                symbol, date, ml_score, anomaly_score, cluster_id,
                flow_score, fundamental_score, intel_score, momentum_score,
                composite_score, signal, components
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, date) DO UPDATE SET
                ml_score = EXCLUDED.ml_score,
                anomaly_score = EXCLUDED.anomaly_score,
                cluster_id = EXCLUDED.cluster_id,
                flow_score = EXCLUDED.flow_score,
                fundamental_score = EXCLUDED.fundamental_score,
                intel_score = EXCLUDED.intel_score,
                momentum_score = EXCLUDED.momentum_score,
                composite_score = EXCLUDED.composite_score,
                signal = EXCLUDED.signal,
                components = EXCLUDED.components
        """, [
            symbol, date,
            signals.get('ml_score'),
            signals.get('anomaly_score'),
            signals.get('cluster_id'),
            signals.get('flow_score'),
            signals.get('fundamental_score'),
            signals.get('intel_score'),
            signals.get('momentum_score'),
            signals.get('composite_score'),
            signals.get('signal'),
            json.dumps(signals.get('components', {}))
        ])
    
    def get_signal_history(
        self, 
        symbol: str, 
        start_date: str = None, 
        end_date: str = None
    ) -> List[Dict]:
        """Get signal history for a stock."""
        query = "SELECT * FROM signal_history WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        results = self.conn.execute(query, params).fetchall()
        if not results:
            return []
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    def get_all_signals_for_date(self, date: str) -> List[Dict]:
        """Get all signals for a specific date."""
        results = self.conn.execute("""
            SELECT * FROM signal_history WHERE date = ? ORDER BY composite_score DESC
        """, [date]).fetchall()
        
        if not results:
            return []
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in results]

