"""
Portfolio manager for MetaQuant Nigeria.
Handles portfolio CRUD, position tracking, and P&L calculations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
from decimal import Decimal

from src.database.db_manager import DatabaseManager
from src.database.models import Portfolio, Position, Transaction


class PortfolioManager:
    """
    Manages user portfolios, positions, and transactions.
    
    Provides:
    - Portfolio creation and management
    - Position tracking with cost basis
    - Transaction recording
    - P&L calculations
    - Performance analytics
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    # ==================== Portfolio Operations ====================
    
    def create_portfolio(self, name: str, description: str = "") -> int:
        """Create a new portfolio."""
        return self.db.create_portfolio(name, description)
    
    def get_portfolio(self, portfolio_id: int) -> Optional[Portfolio]:
        """Get portfolio by ID with positions."""
        portfolios = self.db.get_portfolios()
        portfolio_data = next((p for p in portfolios if p['id'] == portfolio_id), None)
        
        if portfolio_data:
            portfolio = Portfolio.from_dict(portfolio_data)
            positions_data = self.db.get_portfolio_positions(portfolio_id)
            portfolio.positions = [Position.from_dict(p) for p in positions_data]
            return portfolio
        return None
    
    def get_all_portfolios(self) -> List[Portfolio]:
        """Get all portfolios."""
        portfolios_data = self.db.get_portfolios()
        portfolios = []
        
        for data in portfolios_data:
            portfolio = Portfolio.from_dict(data)
            positions_data = self.db.get_portfolio_positions(portfolio.id)
            portfolio.positions = [Position.from_dict(p) for p in positions_data]
            portfolios.append(portfolio)
        
        return portfolios
    
    def delete_portfolio(self, portfolio_id: int):
        """Delete a portfolio and all its positions."""
        self.db.conn.execute(
            "DELETE FROM positions WHERE portfolio_id = ?", 
            [portfolio_id]
        )
        self.db.conn.execute(
            "DELETE FROM transactions WHERE portfolio_id = ?",
            [portfolio_id]
        )
        self.db.conn.execute(
            "DELETE FROM portfolios WHERE id = ?", 
            [portfolio_id]
        )
    
    # ==================== Position Operations ====================
    
    def add_position(
        self, 
        portfolio_id: int, 
        symbol: str, 
        quantity: float, 
        price: float,
        date_acquired: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Add or update a position in a portfolio.
        
        If position already exists, updates average cost and quantity.
        """
        # Get stock ID
        stock = self.db.get_stock(symbol.upper())
        if not stock:
            return False
        
        stock_id = stock['id']
        
        # Check for existing position
        existing = self.db.conn.execute("""
            SELECT id, quantity, avg_cost FROM positions 
            WHERE portfolio_id = ? AND stock_id = ?
        """, [portfolio_id, stock_id]).fetchone()
        
        if existing:
            # Update existing position with new average cost
            old_qty = float(existing[1])
            old_cost = float(existing[2])
            
            new_qty = old_qty + quantity
            new_avg_cost = ((old_qty * old_cost) + (quantity * price)) / new_qty
            
            self.db.upsert_position(
                portfolio_id, stock_id, new_qty, new_avg_cost, date_acquired
            )
        else:
            # Create new position
            self.db.upsert_position(
                portfolio_id, stock_id, quantity, price, date_acquired
            )
        
        # Record transaction
        self._record_transaction(
            portfolio_id, stock_id, 'BUY', quantity, price, 0, notes
        )
        
        return True
    
    def sell_position(
        self,
        portfolio_id: int,
        symbol: str,
        quantity: float,
        price: float,
        fees: float = 0,
        notes: Optional[str] = None
    ) -> bool:
        """
        Sell from a position.
        
        Reduces quantity or removes position if fully sold.
        """
        stock = self.db.get_stock(symbol.upper())
        if not stock:
            return False
        
        stock_id = stock['id']
        
        # Get existing position
        existing = self.db.conn.execute("""
            SELECT id, quantity, avg_cost FROM positions 
            WHERE portfolio_id = ? AND stock_id = ?
        """, [portfolio_id, stock_id]).fetchone()
        
        if not existing:
            return False
        
        current_qty = float(existing[1])
        
        if quantity > current_qty:
            return False  # Can't sell more than owned
        
        # Record transaction first
        self._record_transaction(
            portfolio_id, stock_id, 'SELL', quantity, price, fees, notes
        )
        
        if quantity >= current_qty:
            # Full sale - remove position
            self.db.delete_position(portfolio_id, stock_id)
        else:
            # Partial sale - keep same avg cost
            avg_cost = float(existing[2])
            new_qty = current_qty - quantity
            self.db.upsert_position(portfolio_id, stock_id, new_qty, avg_cost)
        
        return True
    
    def get_position(self, portfolio_id: int, symbol: str) -> Optional[Position]:
        """Get a specific position."""
        stock = self.db.get_stock(symbol.upper())
        if not stock:
            return None
        
        positions = self.db.get_portfolio_positions(portfolio_id)
        pos_data = next((p for p in positions if p['stock_id'] == stock['id']), None)
        
        return Position.from_dict(pos_data) if pos_data else None
    
    # ==================== Transaction Operations ====================
    
    def _record_transaction(
        self,
        portfolio_id: int,
        stock_id: int,
        transaction_type: str,
        quantity: float,
        price: float,
        fees: float = 0,
        notes: Optional[str] = None
    ):
        """Record a buy/sell transaction."""
        self.db.conn.execute("""
            INSERT INTO transactions (
                portfolio_id, stock_id, transaction_type, 
                quantity, price, fees, transaction_date, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            portfolio_id, stock_id, transaction_type,
            quantity, price, fees, datetime.now(), notes
        ])
    
    def get_transactions(
        self, 
        portfolio_id: int, 
        limit: int = 50
    ) -> List[Transaction]:
        """Get transaction history for a portfolio."""
        results = self.db.conn.execute("""
            SELECT t.*, s.symbol, s.name
            FROM transactions t
            JOIN stocks s ON t.stock_id = s.id
            WHERE t.portfolio_id = ?
            ORDER BY t.transaction_date DESC
            LIMIT ?
        """, [portfolio_id, limit]).fetchall()
        
        columns = [desc[0] for desc in self.db.conn.description]
        return [Transaction.from_dict(dict(zip(columns, row))) for row in results]
    
    # ==================== Analytics ====================
    
    def get_portfolio_summary(self, portfolio_id: int) -> Dict[str, Any]:
        """Get summary analytics for a portfolio."""
        positions = self.db.get_portfolio_positions(portfolio_id)
        
        total_value = Decimal(0)
        total_cost = Decimal(0)
        total_unrealized_pnl = Decimal(0)
        
        sector_allocation = {}
        
        for pos in positions:
            market_value = Decimal(str(pos.get('market_value', 0) or 0))
            cost_basis = Decimal(str(pos.get('quantity', 0))) * Decimal(str(pos.get('avg_cost', 0)))
            pnl = Decimal(str(pos.get('unrealized_pnl', 0) or 0))
            
            total_value += market_value
            total_cost += cost_basis
            total_unrealized_pnl += pnl
            
            # Sector allocation
            sector = pos.get('sector', 'Unknown') or 'Unknown'
            sector_allocation[sector] = sector_allocation.get(sector, Decimal(0)) + market_value
        
        # Calculate percentages
        total_return_percent = (
            ((total_value - total_cost) / total_cost * 100) 
            if total_cost > 0 else Decimal(0)
        )
        
        # Convert sector allocation to percentages
        sector_percentages = {}
        if total_value > 0:
            for sector, value in sector_allocation.items():
                sector_percentages[sector] = float(value / total_value * 100)
        
        return {
            'total_value': float(total_value),
            'total_cost': float(total_cost),
            'unrealized_pnl': float(total_unrealized_pnl),
            'return_percent': float(total_return_percent),
            'position_count': len(positions),
            'sector_allocation': sector_percentages,
        }
    
    def get_top_performers(self, portfolio_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top performing positions by return %."""
        positions = self.db.get_portfolio_positions(portfolio_id)
        
        # Sort by return percentage
        sorted_positions = sorted(
            positions,
            key=lambda p: float(p.get('return_percent', 0) or 0),
            reverse=True
        )
        
        return sorted_positions[:top_n]
    
    def get_worst_performers(self, portfolio_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get worst performing positions by return %."""
        positions = self.db.get_portfolio_positions(portfolio_id)
        
        # Sort by return percentage ascending
        sorted_positions = sorted(
            positions,
            key=lambda p: float(p.get('return_percent', 0) or 0)
        )
        
        return sorted_positions[:top_n]
    
    def get_realized_pnl(
        self, 
        portfolio_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Decimal:
        """Calculate realized P&L from closed transactions."""
        query = """
            SELECT 
                transaction_type,
                quantity,
                price,
                fees,
                stock_id
            FROM transactions
            WHERE portfolio_id = ?
        """
        params = [portfolio_id]
        
        if start_date:
            query += " AND DATE(transaction_date) >= ?"
            params.append(start_date)
        if end_date:
            query += " AND DATE(transaction_date) <= ?"
            params.append(end_date)
        
        query += " ORDER BY transaction_date"
        
        transactions = self.db.conn.execute(query, params).fetchall()
        
        # Track cost basis per stock using FIFO
        stock_costs = {}  # stock_id -> list of (quantity, cost)
        realized_pnl = Decimal(0)
        
        for tx_type, qty, price, fees, stock_id in transactions:
            qty = Decimal(str(qty))
            price = Decimal(str(price))
            fees = Decimal(str(fees or 0))
            
            if tx_type == 'BUY':
                if stock_id not in stock_costs:
                    stock_costs[stock_id] = []
                stock_costs[stock_id].append((qty, price))
            
            elif tx_type == 'SELL':
                if stock_id in stock_costs:
                    # FIFO matching
                    remaining = qty
                    while remaining > 0 and stock_costs[stock_id]:
                        buy_qty, buy_price = stock_costs[stock_id][0]
                        
                        if buy_qty <= remaining:
                            # Use entire lot
                            realized_pnl += buy_qty * (price - buy_price)
                            remaining -= buy_qty
                            stock_costs[stock_id].pop(0)
                        else:
                            # Partial lot
                            realized_pnl += remaining * (price - buy_price)
                            stock_costs[stock_id][0] = (buy_qty - remaining, buy_price)
                            remaining = Decimal(0)
                    
                    realized_pnl -= fees
        
        return realized_pnl
