# Data Fetcher for Daemon

import logging
from typing import Optional
import pandas as pd
from tvDatafeed import TvDatafeed, Interval

from config import Config

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch market data from TradingView."""
    
    def __init__(self, config: Config):
        self.config = config
        self._tv = None
        
    @property
    def tv(self) -> TvDatafeed:
        """Lazy initialize TradingView connection."""
        if self._tv is None:
            try:
                if self.config.tradingview_user and self.config.tradingview_pass:
                    self._tv = TvDatafeed(
                        username=self.config.tradingview_user,
                        password=self.config.tradingview_pass
                    )
                else:
                    self._tv = TvDatafeed()
                logger.info("TradingView connected")
            except Exception as e:
                logger.error(f"TradingView connection failed: {e}")
                self._tv = TvDatafeed()  # Anonymous
        return self._tv
    
    async def get_ohlcv(self, symbol: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol."""
        try:
            # NGX symbols are on NSENG exchange
            df = self.tv.get_hist(
                symbol=symbol,
                exchange='NSENG',
                interval=Interval.in_15_minute,
                n_bars=bars
            )
            
            if df is not None and not df.empty:
                df = df.reset_index()
                df.columns = ['datetime', 'symbol_col', 'open', 'high', 'low', 'close', 'volume']
                return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        
        return None
    
    async def get_fundamentals(self, symbol: str) -> Optional[dict]:
        """Fetch fundamental data (stub)."""
        # TODO: Implement fundamental data fetching
        return None
