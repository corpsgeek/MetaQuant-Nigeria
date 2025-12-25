"""
Screening engine for MetaQuant Nigeria.
Provides flexible stock screening based on fundamental and technical filters.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from src.database.db_manager import DatabaseManager


class FilterOperator(Enum):
    """Comparison operators for filters."""
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    BETWEEN = "BETWEEN"
    IN = "IN"
    LIKE = "LIKE"


@dataclass
class FilterCondition:
    """Represents a single filter condition."""
    field: str
    operator: FilterOperator
    value: Any
    value2: Optional[Any] = None  # For BETWEEN operator


class Filter(ABC):
    """Abstract base class for stock filters."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable filter name."""
        pass
    
    @property
    @abstractmethod
    def field(self) -> str:
        """Database field to filter on."""
        pass
    
    @abstractmethod
    def get_condition(self) -> Optional[FilterCondition]:
        """Get the filter condition for SQL query."""
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        """Check if filter has valid values set."""
        pass


class PEFilter(Filter):
    """Filter stocks by P/E ratio."""
    
    def __init__(self, min_pe: Optional[float] = None, max_pe: Optional[float] = None):
        self.min_pe = min_pe
        self.max_pe = max_pe
    
    @property
    def name(self) -> str:
        return "P/E Ratio"
    
    @property
    def field(self) -> str:
        return "f.pe_ratio"
    
    def get_condition(self) -> Optional[FilterCondition]:
        if self.min_pe is not None and self.max_pe is not None:
            return FilterCondition(self.field, FilterOperator.BETWEEN, self.min_pe, self.max_pe)
        elif self.min_pe is not None:
            return FilterCondition(self.field, FilterOperator.GREATER_THAN_OR_EQUAL, self.min_pe)
        elif self.max_pe is not None:
            return FilterCondition(self.field, FilterOperator.LESS_THAN_OR_EQUAL, self.max_pe)
        return None
    
    def is_valid(self) -> bool:
        return self.min_pe is not None or self.max_pe is not None


class MarketCapFilter(Filter):
    """Filter stocks by market capitalization."""
    
    # Market cap categories in Naira
    SMALL_CAP = 10_000_000_000  # < 10B NGN
    MID_CAP = 100_000_000_000   # 10B - 100B NGN
    # LARGE_CAP = > 100B NGN
    
    def __init__(self, min_cap: Optional[float] = None, max_cap: Optional[float] = None):
        self.min_cap = min_cap
        self.max_cap = max_cap
    
    @property
    def name(self) -> str:
        return "Market Cap"
    
    @property
    def field(self) -> str:
        return "s.market_cap"
    
    def get_condition(self) -> Optional[FilterCondition]:
        if self.min_cap is not None and self.max_cap is not None:
            return FilterCondition(self.field, FilterOperator.BETWEEN, self.min_cap, self.max_cap)
        elif self.min_cap is not None:
            return FilterCondition(self.field, FilterOperator.GREATER_THAN_OR_EQUAL, self.min_cap)
        elif self.max_cap is not None:
            return FilterCondition(self.field, FilterOperator.LESS_THAN_OR_EQUAL, self.max_cap)
        return None
    
    def is_valid(self) -> bool:
        return self.min_cap is not None or self.max_cap is not None
    
    @classmethod
    def small_cap(cls) -> 'MarketCapFilter':
        """Factory for small cap filter."""
        return cls(max_cap=cls.SMALL_CAP)
    
    @classmethod
    def mid_cap(cls) -> 'MarketCapFilter':
        """Factory for mid cap filter."""
        return cls(min_cap=cls.SMALL_CAP, max_cap=cls.MID_CAP)
    
    @classmethod
    def large_cap(cls) -> 'MarketCapFilter':
        """Factory for large cap filter."""
        return cls(min_cap=cls.MID_CAP)


class DividendFilter(Filter):
    """Filter stocks by dividend yield."""
    
    def __init__(self, min_yield: Optional[float] = None, max_yield: Optional[float] = None):
        self.min_yield = min_yield
        self.max_yield = max_yield
    
    @property
    def name(self) -> str:
        return "Dividend Yield"
    
    @property
    def field(self) -> str:
        return "f.dividend_yield"
    
    def get_condition(self) -> Optional[FilterCondition]:
        if self.min_yield is not None and self.max_yield is not None:
            return FilterCondition(self.field, FilterOperator.BETWEEN, self.min_yield, self.max_yield)
        elif self.min_yield is not None:
            return FilterCondition(self.field, FilterOperator.GREATER_THAN_OR_EQUAL, self.min_yield)
        elif self.max_yield is not None:
            return FilterCondition(self.field, FilterOperator.LESS_THAN_OR_EQUAL, self.max_yield)
        return None
    
    def is_valid(self) -> bool:
        return self.min_yield is not None or self.max_yield is not None


class SectorFilter(Filter):
    """Filter stocks by sector."""
    
    def __init__(self, sector: Optional[str] = None, sectors: Optional[List[str]] = None):
        self.sector = sector
        self.sectors = sectors or []
    
    @property
    def name(self) -> str:
        return "Sector"
    
    @property
    def field(self) -> str:
        return "s.sector"
    
    def get_condition(self) -> Optional[FilterCondition]:
        if self.sector:
            return FilterCondition(self.field, FilterOperator.EQUALS, self.sector)
        elif self.sectors:
            return FilterCondition(self.field, FilterOperator.IN, self.sectors)
        return None
    
    def is_valid(self) -> bool:
        return bool(self.sector or self.sectors)


class VolumeFilter(Filter):
    """Filter stocks by trading volume."""
    
    def __init__(self, min_volume: Optional[int] = None, max_volume: Optional[int] = None):
        self.min_volume = min_volume
        self.max_volume = max_volume
    
    @property
    def name(self) -> str:
        return "Volume"
    
    @property
    def field(self) -> str:
        return "s.volume"
    
    def get_condition(self) -> Optional[FilterCondition]:
        if self.min_volume is not None and self.max_volume is not None:
            return FilterCondition(self.field, FilterOperator.BETWEEN, self.min_volume, self.max_volume)
        elif self.min_volume is not None:
            return FilterCondition(self.field, FilterOperator.GREATER_THAN_OR_EQUAL, self.min_volume)
        elif self.max_volume is not None:
            return FilterCondition(self.field, FilterOperator.LESS_THAN_OR_EQUAL, self.max_volume)
        return None
    
    def is_valid(self) -> bool:
        return self.min_volume is not None or self.max_volume is not None


class EPSFilter(Filter):
    """Filter stocks by earnings per share."""
    
    def __init__(self, min_eps: Optional[float] = None, max_eps: Optional[float] = None):
        self.min_eps = min_eps
        self.max_eps = max_eps
    
    @property
    def name(self) -> str:
        return "EPS"
    
    @property
    def field(self) -> str:
        return "f.eps"
    
    def get_condition(self) -> Optional[FilterCondition]:
        if self.min_eps is not None and self.max_eps is not None:
            return FilterCondition(self.field, FilterOperator.BETWEEN, self.min_eps, self.max_eps)
        elif self.min_eps is not None:
            return FilterCondition(self.field, FilterOperator.GREATER_THAN_OR_EQUAL, self.min_eps)
        elif self.max_eps is not None:
            return FilterCondition(self.field, FilterOperator.LESS_THAN_OR_EQUAL, self.max_eps)
        return None
    
    def is_valid(self) -> bool:
        return self.min_eps is not None or self.max_eps is not None


class PriceChangeFilter(Filter):
    """Filter stocks by price change percentage."""
    
    def __init__(self, min_change: Optional[float] = None, max_change: Optional[float] = None):
        self.min_change = min_change
        self.max_change = max_change
    
    @property
    def name(self) -> str:
        return "Price Change %"
    
    @property
    def field(self) -> str:
        return "s.change_percent"
    
    def get_condition(self) -> Optional[FilterCondition]:
        if self.min_change is not None and self.max_change is not None:
            return FilterCondition(self.field, FilterOperator.BETWEEN, self.min_change, self.max_change)
        elif self.min_change is not None:
            return FilterCondition(self.field, FilterOperator.GREATER_THAN_OR_EQUAL, self.min_change)
        elif self.max_change is not None:
            return FilterCondition(self.field, FilterOperator.LESS_THAN_OR_EQUAL, self.max_change)
        return None
    
    def is_valid(self) -> bool:
        return self.min_change is not None or self.max_change is not None


class ScreeningEngine:
    """
    Stock screening engine that combines multiple filters.
    
    Usage:
        engine = ScreeningEngine(db)
        engine.add_filter(PEFilter(max_pe=15))
        engine.add_filter(DividendFilter(min_yield=5))
        results = engine.run()
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.filters: List[Filter] = []
        self._sort_field = 's.symbol'
        self._sort_order = 'ASC'
        self._limit = 100
    
    def add_filter(self, filter: Filter) -> 'ScreeningEngine':
        """Add a filter to the screen."""
        if filter.is_valid():
            self.filters.append(filter)
        return self
    
    def remove_filter(self, filter_type: type) -> 'ScreeningEngine':
        """Remove all filters of a specific type."""
        self.filters = [f for f in self.filters if not isinstance(f, filter_type)]
        return self
    
    def clear_filters(self) -> 'ScreeningEngine':
        """Remove all filters."""
        self.filters = []
        return self
    
    def sort_by(self, field: str, ascending: bool = True) -> 'ScreeningEngine':
        """Set sort field and order."""
        self._sort_field = field
        self._sort_order = 'ASC' if ascending else 'DESC'
        return self
    
    def limit(self, count: int) -> 'ScreeningEngine':
        """Set maximum number of results."""
        self._limit = count
        return self
    
    def run(self) -> List[Dict[str, Any]]:
        """Execute the screen and return matching stocks."""
        # Build base query with LEFT JOIN to fundamentals
        query = """
            SELECT 
                s.id, s.symbol, s.name, s.sector, s.subsector,
                s.last_price, s.prev_close, s.change_percent, 
                s.volume, s.market_cap, s.last_updated,
                f.pe_ratio, f.pb_ratio, f.eps, f.dividend_yield, f.roe
            FROM stocks s
            LEFT JOIN fundamentals f ON s.id = f.stock_id
            WHERE s.is_active = TRUE
        """
        
        params = []
        
        # Add filter conditions
        for filter in self.filters:
            condition = filter.get_condition()
            if condition:
                query += f"\n  AND {self._build_condition_sql(condition, params)}"
        
        # Add sorting and limit
        query += f"\nORDER BY {self._sort_field} {self._sort_order}"
        query += f"\nLIMIT {self._limit}"
        
        # Execute query
        results = self.db.conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in self.db.conn.description]
        
        return [dict(zip(columns, row)) for row in results]
    
    def _build_condition_sql(self, condition: FilterCondition, params: list) -> str:
        """Build SQL condition from FilterCondition."""
        field = condition.field
        op = condition.operator
        
        if op == FilterOperator.EQUALS:
            params.append(condition.value)
            return f"{field} = ?"
        
        elif op == FilterOperator.NOT_EQUALS:
            params.append(condition.value)
            return f"{field} != ?"
        
        elif op == FilterOperator.GREATER_THAN:
            params.append(condition.value)
            return f"{field} > ?"
        
        elif op == FilterOperator.GREATER_THAN_OR_EQUAL:
            params.append(condition.value)
            return f"{field} >= ?"
        
        elif op == FilterOperator.LESS_THAN:
            params.append(condition.value)
            return f"{field} < ?"
        
        elif op == FilterOperator.LESS_THAN_OR_EQUAL:
            params.append(condition.value)
            return f"{field} <= ?"
        
        elif op == FilterOperator.BETWEEN:
            params.extend([condition.value, condition.value2])
            return f"{field} BETWEEN ? AND ?"
        
        elif op == FilterOperator.IN:
            placeholders = ', '.join(['?' for _ in condition.value])
            params.extend(condition.value)
            return f"{field} IN ({placeholders})"
        
        elif op == FilterOperator.LIKE:
            params.append(condition.value)
            return f"{field} LIKE ?"
        
        return "1=1"
    
    def get_active_filters(self) -> List[str]:
        """Get list of active filter descriptions."""
        return [f.name for f in self.filters if f.is_valid()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize current filter state."""
        return {
            'filters': [
                {
                    'type': type(f).__name__,
                    'name': f.name,
                }
                for f in self.filters
            ],
            'sort_field': self._sort_field,
            'sort_order': self._sort_order,
            'limit': self._limit,
        }


# Preset screens
class PresetScreens:
    """Factory for common screening presets."""
    
    @staticmethod
    def value_stocks(db: DatabaseManager) -> ScreeningEngine:
        """Stocks with low P/E and high dividend yield."""
        return (ScreeningEngine(db)
                .add_filter(PEFilter(max_pe=15))
                .add_filter(DividendFilter(min_yield=3))
                .sort_by('f.dividend_yield', ascending=False))
    
    @staticmethod
    def growth_stocks(db: DatabaseManager) -> ScreeningEngine:
        """Stocks with high EPS growth."""
        return (ScreeningEngine(db)
                .add_filter(EPSFilter(min_eps=0))
                .add_filter(MarketCapFilter.mid_cap())
                .sort_by('f.eps', ascending=False))
    
    @staticmethod
    def large_caps(db: DatabaseManager) -> ScreeningEngine:
        """Large cap stocks."""
        return (ScreeningEngine(db)
                .add_filter(MarketCapFilter.large_cap())
                .sort_by('s.market_cap', ascending=False))
    
    @staticmethod
    def top_gainers(db: DatabaseManager) -> ScreeningEngine:
        """Today's top gainers."""
        return (ScreeningEngine(db)
                .add_filter(PriceChangeFilter(min_change=0))
                .sort_by('s.change_percent', ascending=False)
                .limit(20))
    
    @staticmethod
    def top_losers(db: DatabaseManager) -> ScreeningEngine:
        """Today's top losers."""
        return (ScreeningEngine(db)
                .add_filter(PriceChangeFilter(max_change=0))
                .sort_by('s.change_percent', ascending=True)
                .limit(20))
    
    @staticmethod
    def most_active(db: DatabaseManager) -> ScreeningEngine:
        """Most actively traded stocks."""
        return (ScreeningEngine(db)
                .add_filter(VolumeFilter(min_volume=100000))
                .sort_by('s.volume', ascending=False)
                .limit(20))
