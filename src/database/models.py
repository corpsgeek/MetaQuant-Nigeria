"""
Data models for MetaQuant Nigeria.
Dataclass representations of database entities for type safety and convenience.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List
from decimal import Decimal


@dataclass
class Stock:
    """Represents a stock/equity on the Nigerian Stock Exchange."""
    id: int
    symbol: str
    name: str
    sector: Optional[str] = None
    subsector: Optional[str] = None
    last_price: Optional[Decimal] = None
    prev_close: Optional[Decimal] = None
    change_percent: Optional[Decimal] = None
    volume: Optional[int] = None
    market_cap: Optional[Decimal] = None
    last_updated: Optional[datetime] = None
    is_active: bool = True
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Stock':
        """Create a Stock from a dictionary."""
        return cls(
            id=data.get('id', 0),
            symbol=data.get('symbol', ''),
            name=data.get('name', ''),
            sector=data.get('sector'),
            subsector=data.get('subsector'),
            last_price=data.get('last_price'),
            prev_close=data.get('prev_close'),
            change_percent=data.get('change_percent'),
            volume=data.get('volume'),
            market_cap=data.get('market_cap'),
            last_updated=data.get('last_updated'),
            is_active=data.get('is_active', True)
        )
    
    @property
    def formatted_price(self) -> str:
        """Return formatted price string."""
        if self.last_price is None:
            return "N/A"
        return f"₦{self.last_price:,.2f}"
    
    @property
    def formatted_change(self) -> str:
        """Return formatted change percentage."""
        if self.change_percent is None:
            return "N/A"
        sign = "+" if self.change_percent >= 0 else ""
        return f"{sign}{self.change_percent:.2f}%"


@dataclass
class Fundamentals:
    """Fundamental analysis data for a stock."""
    id: int
    stock_id: int
    pe_ratio: Optional[Decimal] = None
    eps: Optional[Decimal] = None
    dividend_yield: Optional[Decimal] = None
    dividend_per_share: Optional[Decimal] = None
    book_value: Optional[Decimal] = None
    pb_ratio: Optional[Decimal] = None
    revenue: Optional[Decimal] = None
    net_income: Optional[Decimal] = None
    roe: Optional[Decimal] = None
    debt_to_equity: Optional[Decimal] = None
    current_ratio: Optional[Decimal] = None
    fiscal_year_end: Optional[date] = None
    last_updated: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Fundamentals':
        """Create Fundamentals from a dictionary."""
        return cls(
            id=data.get('id', 0),
            stock_id=data.get('stock_id', 0),
            pe_ratio=data.get('pe_ratio'),
            eps=data.get('eps'),
            dividend_yield=data.get('dividend_yield'),
            dividend_per_share=data.get('dividend_per_share'),
            book_value=data.get('book_value'),
            pb_ratio=data.get('pb_ratio'),
            revenue=data.get('revenue'),
            net_income=data.get('net_income'),
            roe=data.get('roe'),
            debt_to_equity=data.get('debt_to_equity'),
            current_ratio=data.get('current_ratio'),
            fiscal_year_end=data.get('fiscal_year_end'),
            last_updated=data.get('last_updated')
        )


@dataclass
class DailyPrice:
    """Daily OHLCV price data."""
    id: int
    stock_id: int
    date: date
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    volume: Optional[int] = None
    trades: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DailyPrice':
        """Create DailyPrice from a dictionary."""
        return cls(
            id=data.get('id', 0),
            stock_id=data.get('stock_id', 0),
            date=data.get('date'),
            open=data.get('open'),
            high=data.get('high'),
            low=data.get('low'),
            close=data.get('close'),
            volume=data.get('volume'),
            trades=data.get('trades')
        )


@dataclass
class Portfolio:
    """User-created stock portfolio."""
    id: int
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    positions: List['Position'] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Portfolio':
        """Create Portfolio from a dictionary."""
        return cls(
            id=data.get('id', 0),
            name=data.get('name', ''),
            description=data.get('description'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
    
    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value."""
        return sum(p.market_value for p in self.positions if p.market_value)
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost basis."""
        return sum(p.quantity * p.avg_cost for p in self.positions)
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return self.total_value - self.total_cost


@dataclass
class Position:
    """A position (holding) in a portfolio."""
    id: int
    portfolio_id: int
    stock_id: int
    quantity: Decimal
    avg_cost: Decimal
    date_acquired: Optional[date] = None
    notes: Optional[str] = None
    
    # Joined fields
    symbol: Optional[str] = None
    name: Optional[str] = None
    last_price: Optional[Decimal] = None
    change_percent: Optional[Decimal] = None
    market_value: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    return_percent: Optional[Decimal] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        """Create Position from a dictionary."""
        return cls(
            id=data.get('id', 0),
            portfolio_id=data.get('portfolio_id', 0),
            stock_id=data.get('stock_id', 0),
            quantity=data.get('quantity', 0),
            avg_cost=data.get('avg_cost', 0),
            date_acquired=data.get('date_acquired'),
            notes=data.get('notes'),
            symbol=data.get('symbol'),
            name=data.get('name'),
            last_price=data.get('last_price'),
            change_percent=data.get('change_percent'),
            market_value=data.get('market_value'),
            unrealized_pnl=data.get('unrealized_pnl'),
            return_percent=data.get('return_percent')
        )
    
    @property
    def cost_basis(self) -> Decimal:
        """Calculate total cost basis."""
        return self.quantity * self.avg_cost
    
    @property
    def formatted_pnl(self) -> str:
        """Return formatted P&L string."""
        if self.unrealized_pnl is None:
            return "N/A"
        sign = "+" if self.unrealized_pnl >= 0 else ""
        return f"{sign}₦{self.unrealized_pnl:,.2f}"


@dataclass
class Transaction:
    """Buy/sell transaction record."""
    id: int
    portfolio_id: int
    stock_id: int
    transaction_type: str  # 'BUY' or 'SELL'
    quantity: Decimal
    price: Decimal
    fees: Decimal = Decimal(0)
    transaction_date: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Transaction':
        """Create Transaction from a dictionary."""
        return cls(
            id=data.get('id', 0),
            portfolio_id=data.get('portfolio_id', 0),
            stock_id=data.get('stock_id', 0),
            transaction_type=data.get('transaction_type', 'BUY'),
            quantity=data.get('quantity', 0),
            price=data.get('price', 0),
            fees=data.get('fees', 0),
            transaction_date=data.get('transaction_date'),
            notes=data.get('notes'),
            created_at=data.get('created_at')
        )
    
    @property
    def total_value(self) -> Decimal:
        """Calculate total transaction value including fees."""
        base = self.quantity * self.price
        if self.transaction_type == 'BUY':
            return base + self.fees
        return base - self.fees


@dataclass
class Watchlist:
    """User-created stock watchlist."""
    id: int
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    items: List['WatchlistItem'] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Watchlist':
        """Create Watchlist from a dictionary."""
        return cls(
            id=data.get('id', 0),
            name=data.get('name', ''),
            description=data.get('description'),
            created_at=data.get('created_at')
        )


@dataclass
class WatchlistItem:
    """An item in a watchlist."""
    id: int
    watchlist_id: int
    stock_id: int
    target_price: Optional[Decimal] = None
    alert_above: Optional[Decimal] = None
    alert_below: Optional[Decimal] = None
    notes: Optional[str] = None
    added_at: Optional[datetime] = None
    
    # Joined fields
    symbol: Optional[str] = None
    name: Optional[str] = None
    last_price: Optional[Decimal] = None
    change_percent: Optional[Decimal] = None
    sector: Optional[str] = None
    volume: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WatchlistItem':
        """Create WatchlistItem from a dictionary."""
        return cls(
            id=data.get('id', 0),
            watchlist_id=data.get('watchlist_id', 0),
            stock_id=data.get('stock_id', 0),
            target_price=data.get('target_price'),
            alert_above=data.get('alert_above'),
            alert_below=data.get('alert_below'),
            notes=data.get('notes'),
            added_at=data.get('added_at'),
            symbol=data.get('symbol'),
            name=data.get('name'),
            last_price=data.get('last_price'),
            change_percent=data.get('change_percent'),
            sector=data.get('sector'),
            volume=data.get('volume')
        )
    
    @property
    def is_at_target(self) -> bool:
        """Check if stock is at or above target price."""
        if self.target_price is None or self.last_price is None:
            return False
        return self.last_price >= self.target_price
    
    @property
    def should_alert(self) -> bool:
        """Check if price alerts should trigger."""
        if self.last_price is None:
            return False
        if self.alert_above and self.last_price >= self.alert_above:
            return True
        if self.alert_below and self.last_price <= self.alert_below:
            return True
        return False


@dataclass 
class CorporateDisclosure:
    """Corporate disclosure/announcement from NGX."""
    id: int
    stock_id: int
    disclosure_date: date
    title: str
    content: Optional[str] = None
    disclosure_type: Optional[str] = None
    source_url: Optional[str] = None
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CorporateDisclosure':
        """Create CorporateDisclosure from a dictionary."""
        return cls(
            id=data.get('id', 0),
            stock_id=data.get('stock_id', 0),
            disclosure_date=data.get('disclosure_date'),
            title=data.get('title', ''),
            content=data.get('content'),
            disclosure_type=data.get('disclosure_type'),
            source_url=data.get('source_url'),
            created_at=data.get('created_at')
        )
