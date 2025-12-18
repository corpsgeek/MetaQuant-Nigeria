"""Data collectors for MetaQuant Nigeria."""

from .tradingview_collector import TradingViewCollector
from .ngx_collector import NGXCollector

__all__ = ['TradingViewCollector', 'NGXCollector']
