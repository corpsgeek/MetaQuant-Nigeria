"""
Portfolio Book Manager - Manage multiple paper trading portfolios.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """Represents an open position with live P&L."""
    trade_id: int
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    position_value: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: float
    take_profit: float
    days_held: int
    entry_score: float
    entry_attribution: Dict


class PortfolioBookManager:
    """
    Manages multiple paper trading portfolios (books).
    
    Each book is an independent virtual portfolio with:
    - Its own capital
    - Its own positions
    - Its own trade history
    - Its own performance metrics
    """
    
    def __init__(self, trading_tables, price_provider=None):
        """
        Initialize the portfolio book manager.
        
        Args:
            trading_tables: TradingTables instance for database operations
            price_provider: Optional callable that returns current price for symbol
        """
        self.tables = trading_tables
        self.price_provider = price_provider
        self._active_book_id = None
        self._ensure_default_book()
    
    def _ensure_default_book(self):
        """Create a default portfolio book if none exists."""
        books = self.tables.get_portfolio_books()
        if not books:
            self.create_book("Default Portfolio", 10_000_000, 
                           "Default paper trading portfolio")
    
    # ==================== Book Management ====================
    
    def create_book(self, name: str, initial_capital: float = 10_000_000,
                    description: str = None) -> int:
        """Create a new portfolio book."""
        book_id = self.tables.create_portfolio_book(name, initial_capital, description)
        if book_id and self._active_book_id is None:
            self._active_book_id = book_id
        return book_id
    
    def get_books(self) -> List[Dict]:
        """Get all portfolio books."""
        return self.tables.get_portfolio_books()
    
    def get_book(self, book_id: int) -> Optional[Dict]:
        """Get a specific book."""
        return self.tables.get_portfolio_book(book_id)
    
    def set_active_book(self, book_id: int):
        """Set the active book for trading."""
        book = self.tables.get_portfolio_book(book_id)
        if book:
            self._active_book_id = book_id
            logger.info(f"Active portfolio book: {book['name']}")
        else:
            raise ValueError(f"Book {book_id} not found")
    
    @property
    def active_book_id(self) -> int:
        """Get the active book ID."""
        if self._active_book_id is None:
            books = self.tables.get_portfolio_books()
            if books:
                self._active_book_id = books[0]['id']
        return self._active_book_id
    
    def get_active_book(self) -> Optional[Dict]:
        """Get the currently active book."""
        if self.active_book_id:
            return self.tables.get_portfolio_book(self.active_book_id)
        return None
    
    # ==================== Position Management ====================
    
    def get_open_positions(self, book_id: int = None) -> List[OpenPosition]:
        """
        Get all open positions with current P&L.
        
        Args:
            book_id: Optional book ID, uses active book if not specified
        """
        book_id = book_id or self.active_book_id
        trades = self.tables.get_open_trades(book_id)
        
        positions = []
        for trade in trades:
            # Get current price
            current_price = self._get_current_price(trade['symbol'])
            if current_price is None:
                current_price = float(trade['entry_price'])  # Fallback
            else:
                current_price = float(current_price)
            
            # Explicitly cast to prevent DatetimeArray errors
            quantity = int(trade['quantity'])
            entry_price = float(trade['entry_price'])
            position_value = float(trade['position_value'])
            
            # Calculate current values
            current_value = quantity * current_price
            unrealized_pnl = current_value - position_value
            unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            
            # Calculate days held
            try:
                entry_dt = datetime.strptime(trade['entry_date'], '%Y-%m-%d')
                days_held = (datetime.now() - entry_dt).days
            except:
                days_held = 0
            
            positions.append(OpenPosition(
                trade_id=int(trade['id']),
                symbol=trade['symbol'],
                entry_date=str(trade['entry_date']),
                entry_price=entry_price,
                quantity=quantity,
                position_value=position_value,
                current_price=current_price,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                stop_loss=float(trade['stop_loss_price']) if trade.get('stop_loss_price') else 0.0,
                take_profit=float(trade['take_profit_price']) if trade.get('take_profit_price') else 0.0,
                days_held=days_held,
                entry_score=float(trade.get('entry_score', 0) or 0),
                entry_attribution=trade.get('entry_attribution', {}) or {}
            ))
        
        return positions
    
    def get_position_count(self, book_id: int = None) -> int:
        """Get number of open positions in a book."""
        positions = self.tables.get_open_trades(book_id or self.active_book_id)
        return len(positions)
    
    def has_position(self, symbol: str, book_id: int = None) -> bool:
        """Check if there's an open position for a symbol."""
        positions = self.tables.get_open_trades(book_id or self.active_book_id)
        return any(p['symbol'] == symbol for p in positions)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        if self.price_provider:
            try:
                return self.price_provider(symbol)
            except:
                pass
        return None
    
    # ==================== Trade Execution ====================
    
    def open_position(self, symbol: str, current_price: float,
                      stop_loss_pct: float = 0.05, take_profit_pct: float = 0.15,
                      position_pct: float = 0.10, entry_score: float = 0,
                      entry_attribution: Dict = None, book_id: int = None) -> Optional[int]:
        """
        Open a new position.
        
        Args:
            symbol: Stock symbol
            current_price: Entry price
            stop_loss_pct: Stop loss percentage (default 5%)
            take_profit_pct: Take profit percentage (default 15%)
            position_pct: Position size as % of available capital (default 10%)
            entry_score: Signal score at entry
            entry_attribution: Signal component breakdown
            book_id: Portfolio book ID (uses active if not specified)
        
        Returns:
            Trade ID if successful, None otherwise
        """
        book_id = book_id or self.active_book_id
        book = self.tables.get_portfolio_book(book_id)
        
        if not book:
            logger.error(f"Book {book_id} not found")
            return None
        
        # Check position limit
        open_positions = self.get_position_count(book_id)
        if open_positions >= 15:  # Max positions
            logger.warning(f"Max positions reached ({open_positions}/15)")
            return None
        
        # Check if already have position
        if self.has_position(symbol, book_id):
            logger.warning(f"Already have open position in {symbol}")
            return None
        
        # Calculate position size
        available_capital = book['current_capital']
        position_value = available_capital * position_pct
        quantity = int(position_value / current_price)
        
        if quantity <= 0:
            logger.warning(f"Insufficient capital for {symbol}")
            return None
        
        actual_value = quantity * current_price
        if actual_value > available_capital:
            logger.warning(f"Insufficient capital: need ₦{actual_value:,.0f}, have ₦{available_capital:,.0f}")
            return None
        
        # Calculate stop loss and take profit prices
        stop_loss_price = current_price * (1 - stop_loss_pct)
        take_profit_price = current_price * (1 + take_profit_pct)
        
        # Open the trade
        trade_id = self.tables.open_trade(
            book_id=book_id,
            symbol=symbol,
            entry_price=current_price,
            quantity=quantity,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            entry_score=entry_score,
            entry_attribution=entry_attribution
        )
        
        if trade_id:
            # Deduct from capital
            new_capital = available_capital - actual_value
            self.tables.update_book_capital(book_id, new_capital)
            logger.info(f"Opened position: {symbol} x{quantity} @ ₦{current_price:,.2f} "
                       f"(SL: ₦{stop_loss_price:,.2f}, TP: ₦{take_profit_price:,.2f})")
        
        return trade_id
    
    def close_position(self, trade_id: int, current_price: float,
                       exit_reason: str = "SIGNAL", exit_score: float = 0,
                       exit_attribution: Dict = None) -> bool:
        """
        Close an existing position.
        
        Args:
            trade_id: Trade ID to close
            current_price: Exit price
            exit_reason: SIGNAL, STOP_LOSS, TAKE_PROFIT, or MANUAL
            exit_score: Signal score at exit
            exit_attribution: Signal component breakdown
        
        Returns:
            True if successful
        """
        return self.tables.close_trade(
            trade_id=trade_id,
            exit_price=current_price,
            exit_reason=exit_reason,
            exit_score=exit_score,
            exit_attribution=exit_attribution
        )
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float],
                                    book_id: int = None) -> List[Dict]:
        """
        Check open positions for stop loss or take profit triggers.
        
        Args:
            current_prices: Dict of symbol -> current price
            book_id: Portfolio book ID
        
        Returns:
            List of triggered trades with action taken
        """
        book_id = book_id or self.active_book_id
        positions = self.get_open_positions(book_id)
        triggered = []
        
        for pos in positions:
            if pos.symbol not in current_prices:
                continue
            
            price = current_prices[pos.symbol]
            
            # Check stop loss
            if price <= pos.stop_loss:
                self.close_position(
                    pos.trade_id, price, "STOP_LOSS"
                )
                triggered.append({
                    'trade_id': pos.trade_id,
                    'symbol': pos.symbol,
                    'action': 'STOP_LOSS',
                    'price': price,
                    'pnl_pct': pos.unrealized_pnl_pct
                })
                logger.warning(f"STOP LOSS triggered for {pos.symbol} @ ₦{price:,.2f}")
            
            # Check take profit
            elif price >= pos.take_profit:
                self.close_position(
                    pos.trade_id, price, "TAKE_PROFIT"
                )
                triggered.append({
                    'trade_id': pos.trade_id,
                    'symbol': pos.symbol,
                    'action': 'TAKE_PROFIT',
                    'price': price,
                    'pnl_pct': pos.unrealized_pnl_pct
                })
                logger.info(f"TAKE PROFIT triggered for {pos.symbol} @ ₦{price:,.2f}")
        
        return triggered
    
    # ==================== Performance ====================
    
    def get_portfolio_summary(self, book_id: int = None) -> Dict:
        """
        Get comprehensive portfolio summary.
        
        Returns dict with:
        - Book info (name, capital)
        - Open positions count and value
        - Unrealized P&L
        - Realized P&L
        - Overall performance
        """
        book_id = book_id or self.active_book_id
        book = self.tables.get_portfolio_book(book_id)
        
        if not book:
            return {}
        
        positions = self.get_open_positions(book_id)
        performance = self.tables.get_book_performance(book_id)
        
        # Calculate open positions value
        open_value = sum(p.current_value for p in positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        
        return {
            'book_name': book['name'],
            'initial_capital': book['initial_capital'],
            'cash': book['current_capital'],
            'positions_value': open_value,
            'total_value': book['current_capital'] + open_value,
            'open_positions': len(positions),
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': performance.get('total_pnl', 0),
            'total_return_pct': performance.get('total_return', 0),
            'win_rate': performance.get('win_rate', 0),
            'total_trades': performance.get('total_trades', 0)
        }
    
    def get_trade_history(self, book_id: int = None, limit: int = 100) -> List[Dict]:
        """Get trade history for a book."""
        return self.tables.get_trade_history(book_id or self.active_book_id, limit)
