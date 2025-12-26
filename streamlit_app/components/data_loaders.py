"""
Shared Data Loaders for Streamlit App
Cached functions for loading data from TradingView, Database, and ML engines
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# DATABASE
# =============================================================================

@st.cache_resource
def get_db():
    """Get database connection (singleton)."""
    try:
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager()
        db.initialize()
        return db
    except Exception as e:
        st.warning(f"Database unavailable: {e}")
        return None


# =============================================================================
# TRADINGVIEW COLLECTOR
# =============================================================================

@st.cache_resource
def get_collector():
    """Get TradingView collector (singleton)."""
    try:
        from src.collectors.tradingview_collector import TradingViewCollector
        return TradingViewCollector()
    except Exception as e:
        return None


@st.cache_resource
def get_intraday_collector():
    """Get intraday collector for flow tape."""
    try:
        from src.collectors.intraday_collector import IntradayCollector
        db = get_db()
        if db:
            return IntradayCollector(db)
    except:
        pass
    return None


# =============================================================================
# AI ENGINES
# =============================================================================

@st.cache_resource
def get_insight_engine():
    """Get Groq AI Insight Engine."""
    try:
        from src.ai.insight_engine import InsightEngine
        api_key = os.environ.get('GROQ_API_KEY')
        if api_key:
            return InsightEngine(groq_api_key=api_key)
    except:
        pass
    return None


@st.cache_resource
def get_pathway_synthesizer():
    """Get ML Pathway Synthesizer."""
    try:
        from src.ml.pathway_synthesizer import PathwaySynthesizer
        db = get_db()
        if db:
            return PathwaySynthesizer(db)
    except:
        pass
    return None


# =============================================================================
# ML ENGINE
# =============================================================================

@st.cache_resource
def get_ml_engine():
    """Get ML Engine for predictions."""
    try:
        from src.ml import MLEngine
        db = get_db()
        if db:
            engine = MLEngine(db)
            return engine
    except:
        pass
    return None


@st.cache_resource  
def get_pca_engine():
    """Get PCA Factor Engine."""
    try:
        from src.ml.pca_factor_engine import PCAFactorEngine
        return PCAFactorEngine(n_components=5)
    except:
        pass
    return None


@st.cache_resource
def get_smart_money_detector():
    """Get Smart Money Detector."""
    try:
        from src.analysis.smart_money_detector import SmartMoneyDetector
        db = get_db()
        if db:
            return SmartMoneyDetector(db)
    except:
        pass
    return None


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def load_all_stocks() -> pd.DataFrame:
    """Load all stocks from TradingView."""
    collector = get_collector()
    if collector:
        try:
            df = collector.get_all_stocks()
            return df if not df.empty else pd.DataFrame()
        except:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_stock_universe() -> pd.DataFrame:
    """Load stock universe from database."""
    db = get_db()
    if db:
        try:
            result = db.conn.execute("""
                SELECT symbol, name, sector, subsector, market_cap
                FROM stocks
                ORDER BY symbol
            """).fetchall()
            return pd.DataFrame(result, columns=['Symbol', 'Name', 'Sector', 'Subsector', 'Market Cap'])
        except:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_sector_rankings() -> List[Dict]:
    """Load and calculate sector rankings."""
    db = get_db()
    collector = get_collector()
    
    if not db or not collector:
        return []
    
    try:
        all_stocks = collector.get_all_stocks()
        stocks_list = all_stocks.to_dict('records') if not all_stocks.empty else []
        
        # Get sector mapping
        sector_map = {}
        try:
            results = db.conn.execute(
                "SELECT symbol, sector FROM stocks WHERE sector IS NOT NULL AND sector != ''"
            ).fetchall()
            sector_map = {row[0]: row[1] for row in results}
        except:
            pass
        
        # Build sector data
        sector_data = {}
        for s in stocks_list:
            symbol = s.get('symbol', '')
            sector = sector_map.get(symbol, 'Other')
            
            if sector not in sector_data:
                sector_data[sector] = {'stocks': [], 'gainers': 0, 'losers': 0}
            
            chg_1d = s.get('change', 0) or 0
            chg_1w = s.get('Perf.W', 0) or 0
            chg_1m = s.get('Perf.1M', 0) or 0
            
            if not isinstance(chg_1d, (int, float)) or pd.isna(chg_1d):
                chg_1d = 0.0
            if not isinstance(chg_1w, (int, float)) or pd.isna(chg_1w):
                chg_1w = 0.0
            if not isinstance(chg_1m, (int, float)) or pd.isna(chg_1m):
                chg_1m = 0.0
            
            sector_data[sector]['stocks'].append({
                'symbol': symbol,
                'price': s.get('close', 0) or 0,
                'chg_1d': chg_1d,
                'chg_1w': chg_1w,
                'chg_1m': chg_1m,
                'volume': s.get('volume', 0) or 0,
            })
            
            if chg_1d > 0:
                sector_data[sector]['gainers'] += 1
            elif chg_1d < 0:
                sector_data[sector]['losers'] += 1
        
        # Calculate sector rankings
        sector_rankings = []
        for sector, data in sector_data.items():
            stocks = data['stocks']
            if not stocks:
                continue
            
            avg_1d = sum(s['chg_1d'] for s in stocks) / len(stocks)
            avg_1w = sum(s['chg_1w'] for s in stocks) / len(stocks)
            avg_1m = sum(s['chg_1m'] for s in stocks) / len(stocks)
            
            sector_rankings.append({
                'sector': sector,
                'avg_1d': avg_1d,
                'avg_1w': avg_1w,
                'avg_1m': avg_1m,
                'count': len(stocks),
                'gainers': data['gainers'],
                'losers': data['losers'],
                'stocks': stocks,
            })
        
        sector_rankings.sort(key=lambda x: x['avg_1d'], reverse=True)
        return sector_rankings
    except:
        return []


@st.cache_data(ttl=300)
def load_intraday_data(symbol: str, interval: str = '15m', bars: int = 200) -> pd.DataFrame:
    """Load intraday data for a symbol."""
    collector = get_intraday_collector()
    if collector:
        try:
            df = collector.fetch_intraday(symbol, interval=interval, n_bars=bars)
            return df if df is not None else pd.DataFrame()
        except:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_disclosures(limit: int = 50) -> pd.DataFrame:
    """Load corporate disclosures."""
    db = get_db()
    if db:
        try:
            result = db.conn.execute(f"""
                SELECT date, company_symbol, type, url
                FROM corporate_disclosures
                ORDER BY date DESC
                LIMIT {limit}
            """).fetchall()
            return pd.DataFrame(result, columns=['Date', 'Symbol', 'Type', 'URL'])
        except:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_ml_predictions(symbols: List[str] = None) -> List[Dict]:
    """Load ML predictions for stocks."""
    engine = get_ml_engine()
    if not engine:
        return []
    
    try:
        if symbols is None:
            universe = load_stock_universe()
            symbols = universe['Symbol'].tolist() if not universe.empty else []
        
        predictions = []
        for symbol in symbols[:50]:  # Limit to avoid timeout
            try:
                result = engine.predict(symbol)
                if result:
                    predictions.append({
                        'symbol': symbol,
                        'signal': result.get('signal', 'HOLD'),
                        'confidence': result.get('confidence', 0.5),
                        'expected_return': result.get('expected_return', 0),
                    })
            except:
                continue
        
        return predictions
    except:
        return []


@st.cache_data(ttl=300)
def load_pca_factors() -> Dict:
    """Load PCA factor data."""
    pca = get_pca_engine()
    if not pca:
        return {}
    
    try:
        regime = pca.get_market_regime() if hasattr(pca, 'get_market_regime') else {}
        return {
            'regime': regime.get('regime', 'Unknown'),
            'confidence': regime.get('confidence', 0),
            'factors': regime.get('factors', []),
        }
    except:
        return {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_flow_classification(stocks_df: pd.DataFrame) -> Dict:
    """Classify stocks by flow (weekly performance)."""
    if stocks_df.empty or 'Perf.W' not in stocks_df.columns:
        return {'strong_in': 0, 'mod_in': 0, 'neutral': 0, 'mod_out': 0, 'strong_out': 0}
    
    perf = stocks_df['Perf.W'].fillna(0)
    
    return {
        'strong_in': len(stocks_df[perf > 5]),
        'mod_in': len(stocks_df[(perf > 0) & (perf <= 5)]),
        'neutral': len(stocks_df[(perf >= -1) & (perf <= 1)]),
        'mod_out': len(stocks_df[(perf < 0) & (perf >= -5)]),
        'strong_out': len(stocks_df[perf < -5]),
    }


def get_market_breadth(stocks_df: pd.DataFrame) -> Dict:
    """Calculate market breadth metrics."""
    if stocks_df.empty or 'change' not in stocks_df.columns:
        return {'gainers': 0, 'losers': 0, 'unchanged': 0, 'ratio': 1.0}
    
    gainers = len(stocks_df[stocks_df['change'] > 0])
    losers = len(stocks_df[stocks_df['change'] < 0])
    unchanged = len(stocks_df) - gainers - losers
    
    return {
        'gainers': gainers,
        'losers': losers,
        'unchanged': unchanged,
        'ratio': gainers / max(losers, 1),
    }


def get_unusual_volume(stocks_df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """Get stocks with unusual volume."""
    if stocks_df.empty or 'volume' not in stocks_df.columns:
        return pd.DataFrame()
    
    df = stocks_df.copy()
    df['vol_avg'] = df['volume'].mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg']
    
    return df[df['vol_ratio'] > threshold].sort_values('vol_ratio', ascending=False)
