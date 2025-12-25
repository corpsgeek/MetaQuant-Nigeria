"""GUI tabs for MetaQuant Nigeria."""

from .screener_tab import ScreenerTab
from .universe_tab import UniverseTab
from .history_tab import HistoryTab
from .flow_tape_tab import FlowTapeTab
from .fundamentals_tab import FundamentalsTab
from .market_intelligence_tab import MarketIntelligenceTab
from .backtest_tab import BacktestTab
from .pca_analysis_tab import PCAAnalysisTab

__all__ = [
    'ScreenerTab',
    'UniverseTab',
    'HistoryTab',
    'FlowTapeTab',
    'FundamentalsTab',
    'MarketIntelligenceTab',
    'BacktestTab',
    'PCAAnalysisTab'
]
