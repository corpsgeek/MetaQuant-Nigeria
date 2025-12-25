"""
Trading database tables for paper trading system.
Adds tables for portfolio books, paper trades, strategies, and signals.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TradingTables:
    """Manages trading-related database tables."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.conn = db_manager.conn
        self._init_tables()
    
    def _init_tables(self):
        """Create trading tables if they don't exist."""
        
        # Portfolio books (multiple virtual portfolios)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_books (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                initial_capital FLOAT DEFAULT 10000000,
                current_capital FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                description TEXT
            )
        """)
        
        # Paper trades
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY,
                book_id INTEGER,
                symbol TEXT NOT NULL,
                direction TEXT DEFAULT 'BUY',
                entry_date DATE,
                entry_price FLOAT,
                quantity INTEGER,
                position_value FLOAT,
                stop_loss_price FLOAT,
                take_profit_price FLOAT,
                exit_date DATE,
                exit_price FLOAT,
                pnl FLOAT,
                return_pct FLOAT,
                holding_days INTEGER,
                status TEXT DEFAULT 'OPEN',
                entry_score FLOAT,
                exit_score FLOAT,
                entry_attribution TEXT,
                exit_attribution TEXT,
                exit_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Stock strategies (per-stock optimized parameters)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_strategies (
                symbol TEXT PRIMARY KEY,
                optimal_stop_loss FLOAT DEFAULT 0.05,
                optimal_take_profit FLOAT DEFAULT 0.15,
                buy_threshold FLOAT DEFAULT 0.3,
                sell_threshold FLOAT DEFAULT -0.3,
                avg_hold_days INTEGER DEFAULT 10,
                min_hold_days INTEGER DEFAULT 3,
                backtest_return FLOAT,
                backtest_win_rate FLOAT,
                backtest_sharpe FLOAT,
                backtest_trades INTEGER,
                last_optimized TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Signal log
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                score FLOAT,
                attribution TEXT,
                current_price FLOAT,
                acted_on BOOLEAN DEFAULT FALSE,
                trade_id INTEGER
            )
        """)
        
        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_book ON paper_trades(book_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_log_date ON signal_log(timestamp)")
        
        logger.info("Trading tables initialized")
    
    # ==================== Portfolio Book Operations ====================
    
    def create_portfolio_book(self, name: str, initial_capital: float = 10_000_000, 
                              description: str = None) -> int:
        """Create a new portfolio book."""
        try:
            # First, get the next available ID
            result = self.conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM portfolio_books"
            ).fetchone()
            next_id = result[0] if result else 1
            
            # Insert with explicit ID
            self.conn.execute("""
                INSERT INTO portfolio_books (id, name, initial_capital, current_capital, description)
                VALUES (?, ?, ?, ?, ?)
            """, [next_id, name, initial_capital, initial_capital, description])
            
            logger.info(f"Created portfolio book: {name} with ₦{initial_capital:,.0f}")
            return next_id
        except Exception as e:
            logger.error(f"Failed to create portfolio book: {e}")
            return None
    
    def get_portfolio_books(self, active_only: bool = True) -> List[Dict]:
        """Get all portfolio books."""
        query = "SELECT * FROM portfolio_books"
        if active_only:
            query += " WHERE is_active = TRUE"
        query += " ORDER BY created_at DESC"
        
        results = self.conn.execute(query).fetchall()
        columns = ['id', 'name', 'initial_capital', 'current_capital', 
                   'created_at', 'is_active', 'description']
        return [dict(zip(columns, row)) for row in results]
    
    def get_portfolio_book(self, book_id: int) -> Optional[Dict]:
        """Get a specific portfolio book."""
        result = self.conn.execute(
            "SELECT * FROM portfolio_books WHERE id = ?", [book_id]
        ).fetchone()
        if result:
            columns = ['id', 'name', 'initial_capital', 'current_capital', 
                       'created_at', 'is_active', 'description']
            return dict(zip(columns, result))
        return None
    
    def update_book_capital(self, book_id: int, new_capital: float):
        """Update the current capital of a book."""
        self.conn.execute("""
            UPDATE portfolio_books SET current_capital = ? WHERE id = ?
        """, [new_capital, book_id])
    
    # ==================== Paper Trade Operations ====================
    
    def open_trade(self, book_id: int, symbol: str, entry_price: float,
                   quantity: int, stop_loss: float, take_profit: float,
                   entry_score: float = 0, entry_attribution: Dict = None) -> int:
        """Open a new paper trade."""
        position_value = quantity * entry_price
        entry_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get next available ID
        result = self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM paper_trades"
        ).fetchone()
        next_id = result[0] if result else 1
        
        self.conn.execute("""
            INSERT INTO paper_trades 
            (id, book_id, symbol, entry_date, entry_price, quantity, position_value,
             stop_loss_price, take_profit_price, entry_score, entry_attribution, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, [
            next_id, book_id, symbol, entry_date, entry_price, quantity, position_value,
            stop_loss, take_profit, entry_score,
            json.dumps(entry_attribution) if entry_attribution else None
        ])
        
        logger.info(f"Opened trade #{next_id}: {symbol} x{quantity} @ ₦{entry_price:,.2f}")
        return next_id
    
    def close_trade(self, trade_id: int, exit_price: float, exit_reason: str,
                    exit_score: float = 0, exit_attribution: Dict = None):
        """Close an existing paper trade."""
        # Get the trade
        trade = self.get_trade(trade_id)
        if not trade or trade['status'] != 'OPEN':
            return False
        
        # Calculate P&L
        pnl = (exit_price - trade['entry_price']) * trade['quantity']
        return_pct = (exit_price - trade['entry_price']) / trade['entry_price'] * 100
        
        # Calculate holding days
        try:
            entry_dt = datetime.strptime(trade['entry_date'], '%Y-%m-%d')
            exit_dt = datetime.now()
            holding_days = (exit_dt - entry_dt).days
        except:
            holding_days = 1
        
        exit_date = datetime.now().strftime('%Y-%m-%d')
        
        self.conn.execute("""
            UPDATE paper_trades SET
                exit_date = ?,
                exit_price = ?,
                pnl = ?,
                return_pct = ?,
                holding_days = ?,
                status = 'CLOSED',
                exit_score = ?,
                exit_attribution = ?,
                exit_reason = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, [
            exit_date, exit_price, pnl, return_pct, holding_days,
            exit_score, json.dumps(exit_attribution) if exit_attribution else None,
            exit_reason, trade_id
        ])
        
        # Update book capital
        book = self.get_portfolio_book(trade['book_id'])
        if book:
            new_capital = book['current_capital'] + pnl
            self.update_book_capital(trade['book_id'], new_capital)
        
        logger.info(f"Closed trade #{trade_id}: {trade['symbol']} @ ₦{exit_price:,.2f} "
                    f"P&L: ₦{pnl:,.0f} ({return_pct:+.2f}%)")
        return True
    
    def get_trade(self, trade_id: int) -> Optional[Dict]:
        """Get a specific trade."""
        result = self.conn.execute(
            "SELECT * FROM paper_trades WHERE id = ?", [trade_id]
        ).fetchone()
        if result:
            return self._trade_to_dict(result)
        return None
    
    def get_open_trades(self, book_id: int = None) -> List[Dict]:
        """Get all open trades, optionally filtered by book."""
        query = "SELECT * FROM paper_trades WHERE status = 'OPEN'"
        params = []
        if book_id:
            query += " AND book_id = ?"
            params.append(book_id)
        query += " ORDER BY entry_date DESC"
        
        results = self.conn.execute(query, params).fetchall()
        return [self._trade_to_dict(row) for row in results]
    
    def get_trade_history(self, book_id: int = None, limit: int = 100) -> List[Dict]:
        """Get closed trades, optionally filtered by book."""
        query = "SELECT * FROM paper_trades WHERE status = 'CLOSED'"
        params = []
        if book_id:
            query += " AND book_id = ?"
            params.append(book_id)
        query += f" ORDER BY exit_date DESC LIMIT {limit}"
        
        results = self.conn.execute(query, params).fetchall()
        return [self._trade_to_dict(row) for row in results]
    
    def _trade_to_dict(self, row) -> Dict:
        """Convert trade row to dictionary."""
        columns = [
            'id', 'book_id', 'symbol', 'direction', 'entry_date', 'entry_price',
            'quantity', 'position_value', 'stop_loss_price', 'take_profit_price',
            'exit_date', 'exit_price', 'pnl', 'return_pct', 'holding_days',
            'status', 'entry_score', 'exit_score', 'entry_attribution',
            'exit_attribution', 'exit_reason', 'created_at', 'updated_at'
        ]
        trade = dict(zip(columns, row))
        
        # Parse JSON fields
        if trade.get('entry_attribution'):
            try:
                trade['entry_attribution'] = json.loads(trade['entry_attribution'])
            except:
                pass
        if trade.get('exit_attribution'):
            try:
                trade['exit_attribution'] = json.loads(trade['exit_attribution'])
            except:
                pass
        
        return trade
    
    # ==================== Stock Strategy Operations ====================
    
    def save_stock_strategy(self, symbol: str, strategy: Dict):
        """Save or update optimized strategy for a stock."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if exists first
        existing = self.conn.execute(
            "SELECT symbol FROM stock_strategies WHERE symbol = ?", [symbol]
        ).fetchone()
        
        if existing:
            # Update existing
            self.conn.execute("""
                UPDATE stock_strategies SET
                    optimal_stop_loss = ?,
                    optimal_take_profit = ?,
                    buy_threshold = ?,
                    sell_threshold = ?,
                    avg_hold_days = ?,
                    min_hold_days = ?,
                    backtest_return = ?,
                    backtest_win_rate = ?,
                    backtest_sharpe = ?,
                    backtest_trades = ?,
                    last_optimized = ?
                WHERE symbol = ?
            """, [
                strategy.get('stop_loss', 0.05),
                strategy.get('take_profit', 0.15),
                strategy.get('buy_threshold', 0.3),
                strategy.get('sell_threshold', -0.3),
                strategy.get('avg_hold_days', 10),
                strategy.get('min_hold_days', 3),
                strategy.get('return', 0),
                strategy.get('win_rate', 0),
                strategy.get('sharpe', 0),
                strategy.get('trades', 0),
                now,
                symbol
            ])
        else:
            # Insert new
            self.conn.execute("""
                INSERT INTO stock_strategies 
                (symbol, optimal_stop_loss, optimal_take_profit, buy_threshold, 
                 sell_threshold, avg_hold_days, min_hold_days, backtest_return,
                 backtest_win_rate, backtest_sharpe, backtest_trades, last_optimized)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                symbol,
                strategy.get('stop_loss', 0.05),
                strategy.get('take_profit', 0.15),
                strategy.get('buy_threshold', 0.3),
                strategy.get('sell_threshold', -0.3),
                strategy.get('avg_hold_days', 10),
                strategy.get('min_hold_days', 3),
                strategy.get('return', 0),
                strategy.get('win_rate', 0),
                strategy.get('sharpe', 0),
                strategy.get('trades', 0),
                now
            ])
    
    def get_stock_strategy(self, symbol: str) -> Optional[Dict]:
        """Get strategy for a specific stock."""
        result = self.conn.execute(
            "SELECT * FROM stock_strategies WHERE symbol = ?", [symbol]
        ).fetchone()
        if result:
            columns = [
                'symbol', 'optimal_stop_loss', 'optimal_take_profit',
                'buy_threshold', 'sell_threshold', 'avg_hold_days', 'min_hold_days',
                'backtest_return', 'backtest_win_rate', 'backtest_sharpe',
                'backtest_trades', 'last_optimized', 'is_active'
            ]
            return dict(zip(columns, result))
        return None
    
    def get_active_strategies(self) -> List[Dict]:
        """Get all active stock strategies."""
        results = self.conn.execute("""
            SELECT * FROM stock_strategies WHERE is_active = TRUE
            ORDER BY backtest_sharpe DESC
        """).fetchall()
        columns = [
            'symbol', 'optimal_stop_loss', 'optimal_take_profit',
            'buy_threshold', 'sell_threshold', 'avg_hold_days', 'min_hold_days',
            'backtest_return', 'backtest_win_rate', 'backtest_sharpe',
            'backtest_trades', 'last_optimized', 'is_active'
        ]
        return [dict(zip(columns, row)) for row in results]
    
    # ==================== Signal Log Operations ====================
    
    def log_signal(self, symbol: str, signal: str, score: float,
                   current_price: float, attribution: Dict = None) -> int:
        """Log a generated signal."""
        result = self.conn.execute("""
            INSERT INTO signal_log (symbol, signal, score, current_price, attribution)
            VALUES (?, ?, ?, ?, ?)
            RETURNING id
        """, [
            symbol, signal, score, current_price,
            json.dumps(attribution) if attribution else None
        ]).fetchone()
        return result[0]
    
    def get_latest_signals(self, limit: int = 50) -> List[Dict]:
        """Get the most recent signals."""
        results = self.conn.execute(f"""
            SELECT * FROM signal_log ORDER BY timestamp DESC LIMIT {limit}
        """).fetchall()
        columns = ['id', 'timestamp', 'symbol', 'signal', 'score', 
                   'attribution', 'current_price', 'acted_on', 'trade_id']
        signals = []
        for row in results:
            sig = dict(zip(columns, row))
            if sig.get('attribution'):
                try:
                    sig['attribution'] = json.loads(sig['attribution'])
                except:
                    pass
            signals.append(sig)
        return signals
    
    def mark_signal_acted(self, signal_id: int, trade_id: int):
        """Mark a signal as acted upon."""
        self.conn.execute("""
            UPDATE signal_log SET acted_on = TRUE, trade_id = ? WHERE id = ?
        """, [trade_id, signal_id])
    
    # ==================== Performance Metrics ====================
    
    def get_book_performance(self, book_id: int) -> Dict:
        """Calculate performance metrics for a portfolio book."""
        book = self.get_portfolio_book(book_id)
        if not book:
            return {}
        
        # Get closed trades
        trades = self.get_trade_history(book_id, limit=1000)
        
        if not trades:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'total_trades': 0,
                'avg_return': 0,
                'total_pnl': 0
            }
        
        # Calculate metrics
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        win_rate = wins / len(trades) * 100 if trades else 0
        avg_return = sum(t.get('return_pct', 0) for t in trades) / len(trades) if trades else 0
        total_return = (book['current_capital'] - book['initial_capital']) / book['initial_capital'] * 100
        
        return {
            'initial_capital': book['initial_capital'],
            'current_capital': book['current_capital'],
            'total_return': total_return,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_return': avg_return,
            'winning_trades': wins,
            'losing_trades': len(trades) - wins
        }
