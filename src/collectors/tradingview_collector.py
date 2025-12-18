"""
TradingView data collector using tvdatafeed and tradingview-ta libraries.
Fetches historical price data and technical analysis signals.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

try:
    from tvDatafeed import TvDatafeed, Interval
    TVDATAFEED_AVAILABLE = True
except ImportError:
    TVDATAFEED_AVAILABLE = False

try:
    from tradingview_ta import TA_Handler, Interval as TAInterval, Exchange
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False


logger = logging.getLogger(__name__)


class TradingViewCollector:
    """
    Collects market data from TradingView.
    
    Uses tvdatafeed for historical OHLCV data and tradingview-ta for
    technical analysis indicators and recommendations.
    """
    
    # Nigerian Stock Exchange symbol suffix
    EXCHANGE = "NGSE"
    
    # Interval mappings
    INTERVALS = {
        '1m': Interval.in_1_minute if TVDATAFEED_AVAILABLE else None,
        '5m': Interval.in_5_minute if TVDATAFEED_AVAILABLE else None,
        '15m': Interval.in_15_minute if TVDATAFEED_AVAILABLE else None,
        '30m': Interval.in_30_minute if TVDATAFEED_AVAILABLE else None,
        '1h': Interval.in_1_hour if TVDATAFEED_AVAILABLE else None,
        '4h': Interval.in_4_hour if TVDATAFEED_AVAILABLE else None,
        'D': Interval.in_daily if TVDATAFEED_AVAILABLE else None,
        'W': Interval.in_weekly if TVDATAFEED_AVAILABLE else None,
        'M': Interval.in_monthly if TVDATAFEED_AVAILABLE else None,
    }
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize TradingView collector.
        
        Args:
            username: TradingView username (optional, for more data)
            password: TradingView password (optional)
        """
        self.tv = None
        self.username = username
        self.password = password
        
        if TVDATAFEED_AVAILABLE:
            try:
                if username and password:
                    self.tv = TvDatafeed(username, password)
                else:
                    self.tv = TvDatafeed()
                logger.info("TradingView datafeed initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize TradingView datafeed: {e}")
                self.tv = None
        else:
            logger.warning("tvdatafeed not available - install with: pip install tvdatafeed")
    
    def get_historical_data(
        self, 
        symbol: str, 
        interval: str = 'D',
        n_bars: int = 365,
        exchange: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'DANGCEM')
            interval: Time interval ('1m', '5m', '15m', '30m', '1h', '4h', 'D', 'W', 'M')
            n_bars: Number of bars to fetch
            exchange: Exchange code (defaults to NGSE)
            
        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        if not TVDATAFEED_AVAILABLE or self.tv is None:
            logger.warning("TradingView datafeed not available")
            return None
        
        exchange = exchange or self.EXCHANGE
        tv_interval = self.INTERVALS.get(interval)
        
        if tv_interval is None:
            logger.error(f"Invalid interval: {interval}")
            return None
        
        try:
            data = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=tv_interval,
                n_bars=n_bars
            )
            
            if data is not None and not data.empty:
                # Reset index to make datetime a column
                data = data.reset_index()
                data.columns = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                logger.info(f"Fetched {len(data)} bars for {symbol}")
                return data
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_technical_analysis(
        self, 
        symbol: str, 
        interval: str = 'D',
        exchange: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch technical analysis data and recommendations.
        
        Args:
            symbol: Stock symbol
            interval: Time interval
            exchange: Exchange code
            
        Returns:
            Dictionary with indicators and recommendations
        """
        if not TRADINGVIEW_TA_AVAILABLE:
            logger.warning("tradingview-ta not available - install with: pip install tradingview-ta")
            return None
        
        exchange = exchange or self.EXCHANGE
        
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
                screener="nigeria",
                exchange=exchange,
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
    
    def get_multiple_symbols(
        self, 
        symbols: List[str], 
        interval: str = 'D',
        n_bars: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            interval: Time interval
            n_bars: Number of bars per symbol
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, interval, n_bars)
            if data is not None:
                results[symbol] = data
        return results
    
    def search_symbols(self, query: str, exchange: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Search for symbols matching a query.
        
        Args:
            query: Search term
            exchange: Optional exchange filter
            
        Returns:
            List of matching symbols with details
        """
        if not TVDATAFEED_AVAILABLE or self.tv is None:
            return []
        
        try:
            results = self.tv.search_symbol(query, exchange=exchange or self.EXCHANGE)
            return [
                {
                    'symbol': r.get('symbol', ''),
                    'description': r.get('description', ''),
                    'exchange': r.get('exchange', ''),
                    'type': r.get('type', ''),
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []


# Convenience function for quick data fetch
def fetch_ngse_data(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Quick helper to fetch NGSE stock data.
    
    Args:
        symbol: Stock symbol (e.g., 'DANGCEM')
        days: Number of days of data
        
    Returns:
        DataFrame with OHLCV data
    """
    collector = TradingViewCollector()
    return collector.get_historical_data(symbol, 'D', days)
