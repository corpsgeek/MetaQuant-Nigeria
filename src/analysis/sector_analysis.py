"""
Sector Index and Relative Strength Analysis for MetaQuant Nigeria.

Provides:
- Custom sector index calculation (market-cap weighted)
- Relative strength (RS) ratio computation
- Flow classification (durable, short-term, one-off)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class SectorAnalysis:
    """
    Sector rotation analysis with custom indices and relative strength.
    """
    
    def __init__(self, db):
        """
        Initialize sector analysis.
        
        Args:
            db: DatabaseManager instance
        """
        self.db = db
    
    def get_sectors(self) -> List[str]:
        """Get list of unique sectors."""
        results = self.db.conn.execute("""
            SELECT DISTINCT sector FROM stocks 
            WHERE sector IS NOT NULL AND sector != ''
            ORDER BY sector
        """).fetchall()
        return [r[0] for r in results]
    
    def calculate_sector_index(self, sector: str, date: str = None) -> Dict[str, Any]:
        """
        Calculate market-cap weighted sector index for a given date.
        
        Index = Σ (Price × MarketCap) / Σ MarketCap
        
        Args:
            sector: Sector name
            date: Date to calculate for (default: latest)
            
        Returns:
            Dict with index value and component details
        """
        if date is None:
            date = datetime.now().date().isoformat()
        
        # Get stocks in sector with prices and market caps
        query = """
            SELECT 
                s.symbol, s.name, s.market_cap,
                dp.close, dp.volume, dp.change_pct
            FROM stocks s
            JOIN daily_prices dp ON s.id = dp.stock_id
            WHERE s.sector = ? AND dp.date = ?
            AND s.market_cap > 0 AND dp.close > 0
        """
        
        results = self.db.conn.execute(query, [sector, date]).fetchall()
        
        if not results:
            return {'error': f'No data for {sector} on {date}'}
        
        total_market_cap = 0
        weighted_sum = 0
        components = []
        
        for row in results:
            symbol, name, market_cap, close, volume, change_pct = row
            market_cap = float(market_cap or 0)
            close = float(close or 0)
            
            if market_cap > 0:
                weighted_sum += close * market_cap
                total_market_cap += market_cap
                
                components.append({
                    'symbol': symbol,
                    'name': name,
                    'price': close,
                    'market_cap': market_cap,
                    'weight': 0,  # Will calculate below
                    'change_pct': float(change_pct or 0),
                    'volume': int(volume or 0)
                })
        
        if total_market_cap == 0:
            return {'error': 'No valid market cap data'}
        
        # Calculate index value (normalized to base 1000)
        index_value = (weighted_sum / total_market_cap)
        
        # Calculate weights
        for comp in components:
            comp['weight'] = (comp['market_cap'] / total_market_cap) * 100
        
        # Sort by weight descending
        components.sort(key=lambda x: -x['weight'])
        
        # Calculate sector change (weighted average of component changes)
        sector_change = sum(c['change_pct'] * c['weight'] / 100 for c in components)
        
        return {
            'sector': sector,
            'date': date,
            'index_value': index_value,
            'change_pct': sector_change,
            'total_market_cap': total_market_cap,
            'component_count': len(components),
            'components': components
        }
    
    def calculate_all_sector_indices(self, date: str = None) -> Dict[str, Dict]:
        """
        Calculate indices for all sectors.
        
        Args:
            date: Date to calculate for
            
        Returns:
            Dict mapping sector name to index data
        """
        sectors = self.get_sectors()
        indices = {}
        
        for sector in sectors:
            idx = self.calculate_sector_index(sector, date)
            if 'error' not in idx:
                indices[sector] = idx
        
        return indices
    
    def calculate_relative_strength(
        self, 
        symbol: str, 
        days: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate relative strength of a stock vs its sector.
        
        RS = Stock_Price / Sector_Index (normalized)
        
        Args:
            symbol: Stock symbol
            days: Lookback period
            
        Returns:
            Dict with RS metrics
        """
        # Get stock info
        stock = self.db.get_stock(symbol)
        if not stock:
            return {'error': f'Stock {symbol} not found'}
        
        sector = stock.get('sector')
        if not sector:
            return {'error': f'{symbol} has no sector assigned'}
        
        stock_id = stock['id']
        
        # Get historical prices for stock
        stock_prices = self.db.conn.execute("""
            SELECT date, close FROM daily_prices
            WHERE stock_id = ?
            ORDER BY date DESC
            LIMIT ?
        """, [stock_id, days]).fetchall()
        
        if len(stock_prices) < 5:
            return {'error': 'Insufficient price history'}
        
        # Get sector peers for same dates
        dates = [p[0] for p in stock_prices]
        
        # Calculate RS series
        rs_series = []
        
        for date, stock_close in stock_prices:
            # Get sector index for this date
            sector_idx = self.calculate_sector_index(sector, str(date))
            
            if 'error' not in sector_idx and sector_idx['index_value'] > 0:
                rs = float(stock_close) / sector_idx['index_value']
                rs_series.append({
                    'date': str(date),
                    'stock_price': float(stock_close),
                    'sector_index': sector_idx['index_value'],
                    'rs_ratio': rs
                })
        
        if len(rs_series) < 3:
            return {'error': 'Could not calculate RS series'}
        
        # Calculate RS metrics
        rs_values = [r['rs_ratio'] for r in rs_series]
        
        current_rs = rs_values[0]
        rs_mean = np.mean(rs_values)
        rs_std = np.std(rs_values)
        
        # RS z-score
        rs_zscore = (current_rs - rs_mean) / rs_std if rs_std > 0 else 0
        
        # RS momentum (% change over period)
        if len(rs_values) >= 2:
            rs_momentum = ((rs_values[0] - rs_values[-1]) / rs_values[-1]) * 100
        else:
            rs_momentum = 0
        
        # RS trend (positive = outperforming, negative = underperforming)
        if rs_momentum > 2:
            rs_trend = 'OUTPERFORMING'
        elif rs_momentum < -2:
            rs_trend = 'UNDERPERFORMING'
        else:
            rs_trend = 'NEUTRAL'
        
        return {
            'symbol': symbol,
            'sector': sector,
            'current_rs': current_rs,
            'rs_mean': rs_mean,
            'rs_zscore': rs_zscore,
            'rs_momentum': rs_momentum,
            'rs_trend': rs_trend,
            'series': rs_series[:10],  # Last 10 data points
            'lookback_days': days
        }
    
    def classify_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Classify the type of flow for a stock.
        
        Flow Types:
        - DURABLE: 20+ days of sustained outperformance with high volume
        - SHORT_TERM: 5-20 days of tactical rotation
        - ONE_OFF: 1-5 days, likely news/event driven
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with flow classification
        """
        # Get RS data
        rs_data = self.calculate_relative_strength(symbol, days=30)
        
        if 'error' in rs_data:
            return rs_data
        
        rs_trend = rs_data['rs_trend']
        rs_momentum = rs_data['rs_momentum']
        rs_zscore = rs_data['rs_zscore']
        
        # Count consecutive days of outperformance/underperformance
        series = rs_data.get('series', [])
        if len(series) < 2:
            return {'error': 'Insufficient data for flow classification'}
        
        # Count trend duration
        consecutive_days = 1
        trend_direction = 'up' if series[0]['rs_ratio'] > series[1]['rs_ratio'] else 'down'
        
        for i in range(1, len(series) - 1):
            current_trend = 'up' if series[i]['rs_ratio'] > series[i+1]['rs_ratio'] else 'down'
            if current_trend == trend_direction:
                consecutive_days += 1
            else:
                break
        
        # Get volume data
        stock = self.db.get_stock(symbol)
        stock_id = stock['id']
        
        vol_data = self.db.conn.execute("""
            SELECT AVG(volume) as avg_vol
            FROM daily_prices
            WHERE stock_id = ?
            ORDER BY date DESC
            LIMIT 20
        """, [stock_id]).fetchone()
        
        recent_vol = self.db.conn.execute("""
            SELECT volume FROM daily_prices
            WHERE stock_id = ?
            ORDER BY date DESC
            LIMIT 1
        """, [stock_id]).fetchone()
        
        avg_volume = float(vol_data[0] or 0)
        current_volume = float(recent_vol[0] or 0) if recent_vol else 0
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Classify flow
        if consecutive_days >= 20 and volume_ratio > 1.2:
            flow_type = 'DURABLE'
            flow_description = 'Sustained institutional flow - likely accumulation/distribution'
        elif consecutive_days >= 5:
            flow_type = 'SHORT_TERM'
            flow_description = 'Tactical rotation - sector rebalancing'
        else:
            flow_type = 'ONE_OFF'
            flow_description = 'Event-driven spike - news or announcement'
        
        # Determine direction
        if rs_momentum > 0:
            flow_direction = 'INFLOW'
            impact = 'Price support likely'
        elif rs_momentum < 0:
            flow_direction = 'OUTFLOW'
            impact = 'Price pressure likely'
        else:
            flow_direction = 'NEUTRAL'
            impact = 'No clear directional bias'
        
        return {
            'symbol': symbol,
            'sector': rs_data['sector'],
            'flow_type': flow_type,
            'flow_direction': flow_direction,
            'flow_description': flow_description,
            'impact': impact,
            'duration_days': consecutive_days,
            'rs_momentum': rs_momentum,
            'rs_zscore': rs_zscore,
            'volume_ratio': volume_ratio,
            'confidence': min(90, 50 + consecutive_days * 2 + abs(rs_zscore) * 10)
        }
    
    def get_sector_rankings(self, date: str = None) -> List[Dict]:
        """
        Get all sectors ranked by performance.
        
        Args:
            date: Date to rank for
            
        Returns:
            List of sectors sorted by change_pct
        """
        indices = self.calculate_all_sector_indices(date)
        
        rankings = []
        for sector, data in indices.items():
            rankings.append({
                'sector': sector,
                'index_value': data['index_value'],
                'change_pct': data['change_pct'],
                'total_market_cap': data['total_market_cap'],
                'component_count': data['component_count']
            })
        
        # Sort by change descending
        rankings.sort(key=lambda x: -x['change_pct'])
        
        return rankings
    
    def get_rs_leaders(self, sector: str = None, top_n: int = 10) -> List[Dict]:
        """
        Get stocks with highest relative strength.
        
        Args:
            sector: Filter by sector (optional)
            top_n: Number of results
            
        Returns:
            List of stocks sorted by RS
        """
        # Get all stocks
        if sector:
            stocks = self.db.conn.execute("""
                SELECT symbol FROM stocks 
                WHERE sector = ? AND market_cap > 0
            """, [sector]).fetchall()
        else:
            stocks = self.db.conn.execute("""
                SELECT symbol FROM stocks 
                WHERE sector IS NOT NULL AND market_cap > 0
            """).fetchall()
        
        leaders = []
        
        for (symbol,) in stocks[:50]:  # Limit for performance
            rs = self.calculate_relative_strength(symbol, days=20)
            if 'error' not in rs:
                leaders.append({
                    'symbol': symbol,
                    'sector': rs['sector'],
                    'rs_momentum': rs['rs_momentum'],
                    'rs_zscore': rs['rs_zscore'],
                    'rs_trend': rs['rs_trend']
                })
        
        # Sort by momentum
        leaders.sort(key=lambda x: -x['rs_momentum'])
        
        return leaders[:top_n]
    
    def get_sector_momentum_matrix(self, stocks_data: List[Dict] = None) -> List[Dict]:
        """
        Get momentum matrix with 1D/1W/1M performance for each sector.
        
        Args:
            stocks_data: Optional pre-fetched stock data from TradingView
            
        Returns:
            List of sector dicts with chg_1d, chg_1w, chg_1m
        """
        # Group stocks by sector
        sector_stocks = {}
        
        if stocks_data:
            # Use pre-fetched TradingView data
            for stock in stocks_data:
                sector = stock.get('sector', 'Unknown')
                if sector and sector != 'Unknown':
                    if sector not in sector_stocks:
                        sector_stocks[sector] = []
                    sector_stocks[sector].append(stock)
        else:
            # Fallback to DB
            sectors = self.get_sectors()
            for sector in sectors:
                idx = self.calculate_sector_index(sector)
                if 'error' not in idx:
                    sector_stocks[sector] = idx.get('components', [])
        
        # Calculate averages per sector
        matrix = []
        for sector, stocks in sector_stocks.items():
            if not stocks:
                continue
            
            chg_1d = np.mean([s.get('chg_1d', s.get('change_pct', 0)) or 0 for s in stocks])
            chg_1w = np.mean([s.get('chg_1w', s.get('perf_1w', 0)) or 0 for s in stocks])
            chg_1m = np.mean([s.get('chg_1m', s.get('perf_1m', 0)) or 0 for s in stocks])
            
            matrix.append({
                'sector': sector,
                'chg_1d': chg_1d,
                'chg_1w': chg_1w,
                'chg_1m': chg_1m,
                'count': len(stocks)
            })
        
        # Sort by 1W performance
        matrix.sort(key=lambda x: -x['chg_1w'])
        return matrix
    
    def calculate_sector_correlations(self, stocks_data: List[Dict] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix between sectors based on stock performance.
        
        Uses 1W performance to determine how sectors move together.
        
        Args:
            stocks_data: Optional pre-fetched stock data
            
        Returns:
            Dict mapping sector -> Dict[sector -> correlation]
        """
        # Group stocks by sector
        sector_perfs = {}
        
        if stocks_data:
            for stock in stocks_data:
                sector = stock.get('sector', 'Unknown')
                if sector and sector != 'Unknown':
                    if sector not in sector_perfs:
                        sector_perfs[sector] = []
                    chg = stock.get('chg_1w', stock.get('perf_1w', 0)) or 0
                    sector_perfs[sector].append(chg)
        
        sectors = list(sector_perfs.keys())
        correlations = {}
        
        for s1 in sectors:
            correlations[s1] = {}
            p1 = sector_perfs[s1]
            
            for s2 in sectors:
                if s1 == s2:
                    correlations[s1][s2] = 1.0
                else:
                    p2 = sector_perfs[s2]
                    # Use mean performance comparison for correlation proxy
                    # Higher correlation if both perform similarly
                    mean1, mean2 = np.mean(p1), np.mean(p2)
                    std1, std2 = np.std(p1) if len(p1) > 1 else 1, np.std(p2) if len(p2) > 1 else 1
                    
                    # Similarity score based on direction and magnitude
                    if mean1 * mean2 > 0:  # Same direction
                        diff = abs(mean1 - mean2)
                        corr = max(0, 1 - diff / 10)  # Closer = higher correlation
                    else:  # Opposite direction
                        corr = -min(1, abs(mean1 - mean2) / 10)
                    
                    correlations[s1][s2] = round(corr, 2)
        
        return correlations
    
    def detect_rotation_phase(self, stocks_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Detect current sector rotation phase based on leadership patterns.
        
        Phases:
        - EARLY (Recovery): Financials, Consumer Discretionary leading
        - MID (Expansion): Technology, Industrials leading
        - LATE (Peak): Energy, Materials leading
        - CONTRACTION: Utilities, Healthcare, Consumer Staples leading
        
        Args:
            stocks_data: Optional pre-fetched stock data
            
        Returns:
            Dict with phase, description, leading/lagging sectors
        """
        # Get momentum matrix
        matrix = self.get_sector_momentum_matrix(stocks_data)
        
        if not matrix:
            return {
                'phase': 'UNKNOWN',
                'description': 'Insufficient data',
                'leading': [],
                'lagging': [],
                'confidence': 0
            }
        
        # Phase detection based on leading sectors
        phase_indicators = {
            'EARLY': ['Financial Services', 'Consumer Goods', 'Insurance'],
            'MID': ['Industrial Goods', 'Services', 'Conglomerates'],
            'LATE': ['Oil & Gas', 'Natural Resources', 'Agriculture'],
            'CONTRACTION': ['Healthcare', 'Consumer Goods', 'Real Estate']
        }
        
        # Get top 3 performing sectors (1W)
        leading = [s['sector'] for s in matrix[:3]]
        lagging = [s['sector'] for s in matrix[-3:]]
        
        # Score each phase
        phase_scores = {}
        for phase, indicators in phase_indicators.items():
            score = sum(1 for s in leading if s in indicators)
            phase_scores[phase] = score
        
        # Determine phase with highest score
        best_phase = max(phase_scores, key=phase_scores.get)
        confidence = min(100, phase_scores[best_phase] * 33)
        
        phase_descriptions = {
            'EARLY': '💹 Recovery - Risk-on rotation beginning',
            'MID': '📈 Expansion - Growth sectors leading',
            'LATE': '⚠️ Peak - Commodities/defensive rotation',
            'CONTRACTION': '🛡️ Defensive - Flight to safety'
        }
        
        return {
            'phase': best_phase,
            'description': phase_descriptions.get(best_phase, 'Unknown phase'),
            'leading': leading,
            'lagging': lagging,
            'confidence': confidence,
            'phase_scores': phase_scores
        }
