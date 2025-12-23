"""
TradingView data collector for Nigerian Stock Exchange (NSENG).
Uses tradingview-screener for bulk market data and tradingview-ta for individual analysis.

Works with Python 3.13+
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import pandas as pd

# Import working TradingView packages
try:
    from tradingview_screener import Query, Column
    SCREENER_AVAILABLE = True
except ImportError:
    SCREENER_AVAILABLE = False

try:
    from tradingview_ta import TA_Handler, Interval as TAInterval
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False


logger = logging.getLogger(__name__)


class TradingViewCollector:
    """
    Collects market data from TradingView for Nigerian Stock Exchange (NSENG).
    
    Uses tradingview-screener for bulk data fetching and tradingview-ta for
    individual stock technical analysis.
    """
    
    # Nigerian Stock Exchange on TradingView
    EXCHANGE = "NSENG"
    SCREENER = "nigeria"
    
    # Standard columns to fetch from screener
    # Note: Nigeria market has limited fundamental data available
    STANDARD_COLUMNS = [
        'name', 'close', 'open', 'high', 'low', 'volume', 
        'change', 'change_abs', 'Perf.W', 'Perf.1M', 'Perf.3M', 
        'Perf.6M', 'Perf.YTD', 'Perf.Y', 
        'RSI', 'RSI[1]', 'MACD.macd', 'MACD.signal',
        'SMA20', 'SMA50', 'SMA200', 'EMA20', 'EMA50', 'EMA200',
        'ADX', 'ATR', 'Stoch.K', 'Stoch.D', 'CCI20',
        'BB.upper', 'BB.lower', 'Pivot.M.Classic.Middle',
        'market_cap_basic', 'average_volume_10d_calc',
        'Recommend.All', 'Recommend.MA', 'Recommend.Other',
        # Available fundamental columns for Nigeria
        'price_earnings_ttm', 'earnings_per_share_basic_ttm',
        'dividend_yield_recent'
    ]
    
    def __init__(self):
        """Initialize TradingView collector."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are available."""
        if not SCREENER_AVAILABLE:
            logger.warning("tradingview-screener not available - install with: pip install tradingview-screener")
        if not TRADINGVIEW_TA_AVAILABLE:
            logger.warning("tradingview-ta not available - install with: pip install tradingview-ta")
    
    def get_all_stocks(self, limit: int = 200) -> pd.DataFrame:
        """
        Fetch all NGX stocks with current OHLCV data and technicals.
        
        Args:
            limit: Maximum number of stocks to fetch (default 200)
            
        Returns:
            DataFrame with all stock data
        """
        if not SCREENER_AVAILABLE:
            logger.error("tradingview-screener not available")
            return pd.DataFrame()
        
        try:
            query = (Query()
                .set_markets(self.SCREENER)
                .select(*self.STANDARD_COLUMNS)
                .limit(limit))
            
            count, df = query.get_scanner_data()
            
            if df is not None and not df.empty:
                # Clean up ticker column (remove NSENG: prefix for consistency)
                if 'ticker' in df.columns:
                    df['symbol'] = df['ticker'].str.replace(f'{self.EXCHANGE}:', '', regex=False)
                
                # Add metadata
                df['exchange'] = self.EXCHANGE
                df['last_updated'] = datetime.now()
                
                logger.info(f"Fetched {len(df)} stocks from NSENG (total available: {count})")
                return df
            else:
                logger.warning("No data returned from screener")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching all stocks: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch fundamental data for a specific stock.
        
        Args:
            symbol: Stock symbol (e.g., 'DANGCEM')
            
        Returns:
            Dictionary with fundamental metrics
        """
        df = self.get_all_stocks()
        
        if df.empty:
            return None
        
        # Find the stock in the dataframe
        symbol_clean = symbol.upper().replace('.NS', '').replace('NSENG:', '')
        
        # Try to match by symbol or ticker
        stock_row = None
        if 'symbol' in df.columns:
            matches = df[df['symbol'].str.upper() == symbol_clean]
            if not matches.empty:
                stock_row = matches.iloc[0]
        
        if stock_row is None and 'ticker' in df.columns:
            matches = df[df['ticker'].str.contains(symbol_clean, case=False, na=False)]
            if not matches.empty:
                stock_row = matches.iloc[0]
        
        if stock_row is None:
            logger.warning(f"Stock {symbol} not found in screener data")
            return None
        
        # Extract fundamental data
        return {
            'symbol': symbol,
            'name': stock_row.get('name', symbol),
            'last_updated': datetime.now(),
            'price': {
                'close': stock_row.get('close', 0),
                'change': stock_row.get('change', 0),
                'change_pct': stock_row.get('change_abs', 0)
            },
            'fundamentals': {
                'market_cap': stock_row.get('market_cap_basic', 0),
                'pe_ratio': stock_row.get('price_earnings_ttm', None),
                'pb_ratio': stock_row.get('price_book_ratio', None),
                'ps_ratio': stock_row.get('price_sales_ratio', None),
                'eps': stock_row.get('earnings_per_share_basic_ttm', None),
                'book_value': stock_row.get('book_value_per_share', None),
                'shares_outstanding': stock_row.get('total_shares_outstanding', None),
                'dividend_yield': stock_row.get('dividend_yield_recent', None),
                'roe': stock_row.get('return_on_equity', None),
                'debt_to_equity': stock_row.get('debt_to_equity', None),
                'current_ratio': stock_row.get('current_ratio', None),
                'revenue': stock_row.get('total_revenue', None),
                'net_income': stock_row.get('net_income', None)
            },
            'performance': {
                'week': stock_row.get('Perf.W', None),
                'month': stock_row.get('Perf.1M', None),
                'quarter': stock_row.get('Perf.3M', None),
                'half_year': stock_row.get('Perf.6M', None),
                'ytd': stock_row.get('Perf.YTD', None),
                'year': stock_row.get('Perf.Y', None)
            },
            'volume': {
                'current': stock_row.get('volume', 0),
                'avg_10d': stock_row.get('average_volume_10d_calc', 0)
            }
        }
    
    def get_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data for a single stock.
        
        Args:
            symbol: Stock symbol (e.g., 'DANGCEM')
            
        Returns:
            Dictionary with stock data and technicals
        """
        if not TRADINGVIEW_TA_AVAILABLE:
            logger.error("tradingview-ta not available")
            return None
        
        try:
            handler = TA_Handler(
                symbol=symbol,
                exchange=self.EXCHANGE,
                screener=self.SCREENER,
                interval=TAInterval.INTERVAL_1_DAY
            )
            
            analysis = handler.get_analysis()
            
            return {
                'symbol': symbol,
                'exchange': self.EXCHANGE,
                'last_updated': datetime.now(),
                'price': {
                    'open': analysis.indicators.get('open'),
                    'high': analysis.indicators.get('high'),
                    'low': analysis.indicators.get('low'),
                    'close': analysis.indicators.get('close'),
                    'volume': analysis.indicators.get('volume'),
                    'change': analysis.indicators.get('change'),
                    'change_percent': analysis.indicators.get('change_percent'),
                },
                'summary': {
                    'recommendation': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                    'buy': analysis.summary.get('BUY', 0),
                    'sell': analysis.summary.get('SELL', 0),
                    'neutral': analysis.summary.get('NEUTRAL', 0),
                },
                'oscillators': {
                    'recommendation': analysis.oscillators.get('RECOMMENDATION', 'NEUTRAL'),
                    'rsi': analysis.indicators.get('RSI'),
                    'stoch_k': analysis.indicators.get('Stoch.K'),
                    'stoch_d': analysis.indicators.get('Stoch.D'),
                    'cci': analysis.indicators.get('CCI20'),
                    'macd': analysis.indicators.get('MACD.macd'),
                    'macd_signal': analysis.indicators.get('MACD.signal'),
                },
                'moving_averages': {
                    'recommendation': analysis.moving_averages.get('RECOMMENDATION', 'NEUTRAL'),
                    'sma_20': analysis.indicators.get('SMA20'),
                    'sma_50': analysis.indicators.get('SMA50'),
                    'sma_200': analysis.indicators.get('SMA200'),
                    'ema_20': analysis.indicators.get('EMA20'),
                    'ema_50': analysis.indicators.get('EMA50'),
                    'ema_200': analysis.indicators.get('EMA200'),
                },
                'volatility': {
                    'atr': analysis.indicators.get('ATR'),
                    'adx': analysis.indicators.get('ADX'),
                    'bb_upper': analysis.indicators.get('BB.upper'),
                    'bb_lower': analysis.indicators.get('BB.lower'),
                },
                'pivot': analysis.indicators.get('Pivot.M.Classic.Middle'),
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_technical_analysis(
        self, 
        symbol: str, 
        interval: str = 'D'
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch technical analysis data and recommendations.
        
        Args:
            symbol: Stock symbol
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', 'D', 'W', 'M')
            
        Returns:
            Dictionary with indicators and recommendations
        """
        if not TRADINGVIEW_TA_AVAILABLE:
            logger.warning("tradingview-ta not available")
            return None
        
        # Map interval to TA interval
        ta_intervals = {
            '1m': TAInterval.INTERVAL_1_MINUTE,
            '5m': TAInterval.INTERVAL_5_MINUTES,
            '15m': TAInterval.INTERVAL_15_MINUTES,
            '30m': TAInterval.INTERVAL_30_MINUTES,
            '1h': TAInterval.INTERVAL_1_HOUR,
            '4h': TAInterval.INTERVAL_4_HOURS,
            'D': TAInterval.INTERVAL_1_DAY,
            'W': TAInterval.INTERVAL_1_WEEK,
            'M': TAInterval.INTERVAL_1_MONTH,
        }
        
        ta_interval = ta_intervals.get(interval, TAInterval.INTERVAL_1_DAY)
        
        try:
            handler = TA_Handler(
                symbol=symbol,
                screener=self.SCREENER,
                exchange=self.EXCHANGE,
                interval=ta_interval
            )
            
            analysis = handler.get_analysis()
            
            return {
                'summary': {
                    'recommendation': analysis.summary.get('RECOMMENDATION', 'NEUTRAL'),
                    'buy': analysis.summary.get('BUY', 0),
                    'sell': analysis.summary.get('SELL', 0),
                    'neutral': analysis.summary.get('NEUTRAL', 0),
                },
                'oscillators': {
                    'recommendation': analysis.oscillators.get('RECOMMENDATION', 'NEUTRAL'),
                    'buy': analysis.oscillators.get('BUY', 0),
                    'sell': analysis.oscillators.get('SELL', 0),
                    'neutral': analysis.oscillators.get('NEUTRAL', 0),
                },
                'moving_averages': {
                    'recommendation': analysis.moving_averages.get('RECOMMENDATION', 'NEUTRAL'),
                    'buy': analysis.moving_averages.get('BUY', 0),
                    'sell': analysis.moving_averages.get('SELL', 0),
                    'neutral': analysis.moving_averages.get('NEUTRAL', 0),
                },
                'indicators': {
                    'rsi': analysis.indicators.get('RSI'),
                    'macd': analysis.indicators.get('MACD.macd'),
                    'macd_signal': analysis.indicators.get('MACD.signal'),
                    'stoch_k': analysis.indicators.get('Stoch.K'),
                    'stoch_d': analysis.indicators.get('Stoch.D'),
                    'cci': analysis.indicators.get('CCI20'),
                    'adx': analysis.indicators.get('ADX'),
                    'atr': analysis.indicators.get('ATR'),
                    'sma_20': analysis.indicators.get('SMA20'),
                    'sma_50': analysis.indicators.get('SMA50'),
                    'sma_200': analysis.indicators.get('SMA200'),
                    'ema_20': analysis.indicators.get('EMA20'),
                    'ema_50': analysis.indicators.get('EMA50'),
                    'ema_200': analysis.indicators.get('EMA200'),
                    'bb_upper': analysis.indicators.get('BB.upper'),
                    'bb_lower': analysis.indicators.get('BB.lower'),
                    'pivot': analysis.indicators.get('Pivot.M.Classic.Middle'),
                    'open': analysis.indicators.get('open'),
                    'high': analysis.indicators.get('high'),
                    'low': analysis.indicators.get('low'),
                    'close': analysis.indicators.get('close'),
                    'volume': analysis.indicators.get('volume'),
                    'change': analysis.indicators.get('change'),
                    'change_percent': analysis.indicators.get('change_percent'),
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching technical analysis for {symbol}: {e}")
            return None
    
    def get_market_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the entire NGX market.
        
        Returns:
            Dictionary with market statistics
        """
        df = self.get_all_stocks()
        
        if df.empty:
            return {'error': 'No data available'}
        
        # Calculate market statistics
        gainers = df[df['change'] > 0] if 'change' in df.columns else pd.DataFrame()
        losers = df[df['change'] < 0] if 'change' in df.columns else pd.DataFrame()
        unchanged = df[df['change'] == 0] if 'change' in df.columns else pd.DataFrame()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(df),
            'gainers': len(gainers),
            'losers': len(losers),
            'unchanged': len(unchanged),
            'total_volume': df['volume'].sum() if 'volume' in df.columns else 0,
            'top_gainers': gainers.nlargest(5, 'change')[['symbol', 'name', 'close', 'change']].to_dict('records') if not gainers.empty and 'symbol' in gainers.columns else [],
            'top_losers': losers.nsmallest(5, 'change')[['symbol', 'name', 'close', 'change']].to_dict('records') if not losers.empty and 'symbol' in losers.columns else [],
            'most_active': df.nlargest(5, 'volume')[['symbol', 'name', 'close', 'change', 'volume']].to_dict('records') if 'volume' in df.columns and 'symbol' in df.columns else [],
        }
    
    def get_ohlcv_for_date(self, target_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get OHLCV data for all stocks (for a specific date).
        
        Note: TradingView screener only provides current day data.
        For historical data, use daily snapshots stored in database.
        
        Args:
            target_date: Not used (kept for API compatibility)
            
        Returns:
            DataFrame with OHLCV for all stocks
        """
        df = self.get_all_stocks()
        
        if df.empty:
            return df
        
        # Extract only OHLCV columns
        ohlcv_columns = ['symbol', 'name', 'open', 'high', 'low', 'close', 'volume', 'change', 'last_updated']
        available_columns = [col for col in ohlcv_columns if col in df.columns]
        
        return df[available_columns]


# Convenience functions
def fetch_all_ngx_stocks() -> pd.DataFrame:
    """Quick helper to fetch all NGX stocks."""
    collector = TradingViewCollector()
    return collector.get_all_stocks()


def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Quick helper to fetch single stock data."""
    collector = TradingViewCollector()
    return collector.get_stock_data(symbol)


def get_market_snapshot() -> Dict[str, Any]:
    """Quick helper to get market snapshot."""
    collector = TradingViewCollector()
    return collector.get_market_snapshot()


# For backward compatibility
def fetch_ngse_data(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Backward-compatible function (now returns current day data only).
    
    Note: Historical data is no longer available via this function.
    Use the database for historical data.
    """
    collector = TradingViewCollector()
    data = collector.get_stock_data(symbol)
    
    if data is None:
        return None
    
    # Convert to DataFrame format for compatibility
    return pd.DataFrame([{
        'datetime': datetime.now(),
        'symbol': symbol,
        'open': data['price']['open'],
        'high': data['price']['high'],
        'low': data['price']['low'],
        'close': data['price']['close'],
        'volume': data['price']['volume'],
    }])
