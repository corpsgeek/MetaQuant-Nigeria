"""
Pattern Recognition Engine for MetaQuant Nigeria.
Detects common chart patterns in price data.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class PatternRecognitionEngine:
    """
    Detects common chart patterns in stock price data.
    
    Supported Patterns:
    - Double Top / Double Bottom
    - Head and Shoulders / Inverse Head and Shoulders
    - Ascending / Descending Triangle
    - Bull / Bear Flag
    - Support / Resistance Breakout
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        
        # Pattern definitions
        self.patterns = {
            'double_top': {'name': 'Double Top', 'type': 'reversal', 'bias': 'bearish'},
            'double_bottom': {'name': 'Double Bottom', 'type': 'reversal', 'bias': 'bullish'},
            'head_shoulders': {'name': 'Head & Shoulders', 'type': 'reversal', 'bias': 'bearish'},
            'inv_head_shoulders': {'name': 'Inverse H&S', 'type': 'reversal', 'bias': 'bullish'},
            'ascending_triangle': {'name': 'Ascending Triangle', 'type': 'continuation', 'bias': 'bullish'},
            'descending_triangle': {'name': 'Descending Triangle', 'type': 'continuation', 'bias': 'bearish'},
            'bull_flag': {'name': 'Bull Flag', 'type': 'continuation', 'bias': 'bullish'},
            'bear_flag': {'name': 'Bear Flag', 'type': 'continuation', 'bias': 'bearish'},
            'breakout_up': {'name': 'Resistance Breakout', 'type': 'breakout', 'bias': 'bullish'},
            'breakout_down': {'name': 'Support Breakdown', 'type': 'breakout', 'bias': 'bearish'},
        }
    
    def scan_all_stocks(self, min_days: int = 30) -> List[Dict]:
        """Scan all stocks for patterns."""
        results = []
        
        try:
            # Get all active stocks
            stocks = self.db.conn.execute(
                "SELECT symbol FROM stocks WHERE is_active = TRUE"
            ).fetchall()
            
            for (symbol,) in stocks:
                patterns = self.detect_patterns(symbol, min_days)
                if patterns:
                    for pattern in patterns:
                        results.append({
                            'symbol': symbol,
                            **pattern
                        })
            
            # Sort by confidence
            results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to scan patterns: {e}")
        
        return results
    
    def detect_patterns(self, symbol: str, min_days: int = 30) -> List[Dict]:
        """Detect all patterns for a specific stock."""
        patterns_found = []
        
        try:
            # Get price data
            prices = self._get_price_data(symbol, min_days)
            if len(prices) < min_days:
                return []
            
            closes = np.array([p['close'] for p in prices])
            highs = np.array([p['high'] for p in prices])
            lows = np.array([p['low'] for p in prices])
            
            # Detect each pattern type
            if pattern := self._detect_double_top(closes, highs):
                patterns_found.append(pattern)
            
            if pattern := self._detect_double_bottom(closes, lows):
                patterns_found.append(pattern)
            
            if pattern := self._detect_head_shoulders(closes, highs):
                patterns_found.append(pattern)
            
            if pattern := self._detect_ascending_triangle(closes, highs, lows):
                patterns_found.append(pattern)
            
            if pattern := self._detect_descending_triangle(closes, highs, lows):
                patterns_found.append(pattern)
            
            if pattern := self._detect_bull_flag(closes, highs, lows):
                patterns_found.append(pattern)
            
            if pattern := self._detect_bear_flag(closes, highs, lows):
                patterns_found.append(pattern)
            
            if pattern := self._detect_breakout(closes, highs, lows):
                patterns_found.append(pattern)
            
        except Exception as e:
            logger.error(f"Failed to detect patterns for {symbol}: {e}")
        
        return patterns_found
    
    def _get_price_data(self, symbol: str, days: int) -> List[Dict]:
        """Get price data for pattern analysis."""
        try:
            # Use intraday_ohlcv table with daily interval or fallback
            result = self.db.conn.execute("""
                SELECT datetime, open, high, low, close, volume
                FROM intraday_ohlcv
                WHERE symbol = ? AND interval = '1d'
                ORDER BY datetime DESC
                LIMIT ?
            """, [symbol, days]).fetchall()
            
            # If no daily data, try 15m data (grouped by date wouldn't work, just get recent)
            if not result:
                result = self.db.conn.execute("""
                    SELECT datetime, open, high, low, close, volume
                    FROM intraday_ohlcv
                    WHERE symbol = ?
                    ORDER BY datetime DESC
                    LIMIT ?
                """, [symbol, days]).fetchall()
            
            return [
                {
                    'date': r[0],
                    'open': float(r[1]) if r[1] else 0,
                    'high': float(r[2]) if r[2] else 0,
                    'low': float(r[3]) if r[3] else 0,
                    'close': float(r[4]) if r[4] else 0,
                    'volume': int(r[5]) if r[5] else 0,
                }
                for r in result
            ]
        except Exception as e:
            logger.error(f"Failed to get price data: {e}")
            return []
    
    def _find_peaks(self, data: np.ndarray, order: int = 3) -> List[int]:
        """Find local peaks in data."""
        peaks = []
        for i in range(order, len(data) - order):
            if all(data[i] > data[i-j] for j in range(1, order+1)) and \
               all(data[i] > data[i+j] for j in range(1, order+1)):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data: np.ndarray, order: int = 3) -> List[int]:
        """Find local troughs in data."""
        troughs = []
        for i in range(order, len(data) - order):
            if all(data[i] < data[i-j] for j in range(1, order+1)) and \
               all(data[i] < data[i+j] for j in range(1, order+1)):
                troughs.append(i)
        return troughs
    
    def _detect_double_top(self, closes: np.ndarray, highs: np.ndarray) -> Optional[Dict]:
        """Detect double top pattern."""
        peaks = self._find_peaks(highs, order=3)
        
        if len(peaks) >= 2:
            # Check last two peaks
            peak1, peak2 = peaks[-2], peaks[-1]
            price1, price2 = highs[peak1], highs[peak2]
            
            # Peaks should be at similar levels (within 3%)
            if abs(price1 - price2) / price1 < 0.03:
                # Current price should be below peaks
                if closes[-1] < min(price1, price2) * 0.95:
                    confidence = 100 - (abs(price1 - price2) / price1 * 100)
                    return {
                        'pattern': 'double_top',
                        'name': 'Double Top',
                        'type': 'reversal',
                        'bias': 'bearish',
                        'confidence': min(85, confidence),
                        'description': f'Two peaks at ₦{price1:.2f} and ₦{price2:.2f}',
                        'target': closes[-1] * 0.95  # 5% downside target
                    }
        return None
    
    def _detect_double_bottom(self, closes: np.ndarray, lows: np.ndarray) -> Optional[Dict]:
        """Detect double bottom pattern."""
        troughs = self._find_troughs(lows, order=3)
        
        if len(troughs) >= 2:
            trough1, trough2 = troughs[-2], troughs[-1]
            price1, price2 = lows[trough1], lows[trough2]
            
            if abs(price1 - price2) / price1 < 0.03:
                if closes[-1] > max(price1, price2) * 1.02:
                    confidence = 100 - (abs(price1 - price2) / price1 * 100)
                    return {
                        'pattern': 'double_bottom',
                        'name': 'Double Bottom',
                        'type': 'reversal',
                        'bias': 'bullish',
                        'confidence': min(85, confidence),
                        'description': f'Two bottoms at ₦{price1:.2f} and ₦{price2:.2f}',
                        'target': closes[-1] * 1.05
                    }
        return None
    
    def _detect_head_shoulders(self, closes: np.ndarray, highs: np.ndarray) -> Optional[Dict]:
        """Detect head and shoulders pattern."""
        peaks = self._find_peaks(highs, order=3)
        
        if len(peaks) >= 3:
            left, head, right = peaks[-3], peaks[-2], peaks[-1]
            
            # Head should be higher than shoulders
            if highs[head] > highs[left] and highs[head] > highs[right]:
                # Shoulders should be at similar levels
                if abs(highs[left] - highs[right]) / highs[left] < 0.05:
                    # Price breaking below neckline
                    neckline = (closes[left] + closes[right]) / 2
                    if closes[-1] < neckline * 0.98:
                        return {
                            'pattern': 'head_shoulders',
                            'name': 'Head & Shoulders',
                            'type': 'reversal',
                            'bias': 'bearish',
                            'confidence': 75,
                            'description': f'Head at ₦{highs[head]:.2f}, neckline at ₦{neckline:.2f}',
                            'target': neckline - (highs[head] - neckline)
                        }
        return None
    
    def _detect_ascending_triangle(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Optional[Dict]:
        """Detect ascending triangle (flat top, rising lows)."""
        peaks = self._find_peaks(highs, order=2)
        troughs = self._find_troughs(lows, order=2)
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Flat top (peaks at similar levels)
            if abs(highs[peaks[-1]] - highs[peaks[-2]]) / highs[peaks[-1]] < 0.02:
                # Rising lows
                if lows[troughs[-1]] > lows[troughs[-2]] * 1.02:
                    resistance = (highs[peaks[-1]] + highs[peaks[-2]]) / 2
                    return {
                        'pattern': 'ascending_triangle',
                        'name': 'Ascending Triangle',
                        'type': 'continuation',
                        'bias': 'bullish',
                        'confidence': 70,
                        'description': f'Resistance at ₦{resistance:.2f}, rising support',
                        'target': resistance * 1.05
                    }
        return None
    
    def _detect_descending_triangle(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Optional[Dict]:
        """Detect descending triangle (flat bottom, falling highs)."""
        peaks = self._find_peaks(highs, order=2)
        troughs = self._find_troughs(lows, order=2)
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Flat bottom
            if abs(lows[troughs[-1]] - lows[troughs[-2]]) / lows[troughs[-1]] < 0.02:
                # Falling highs
                if highs[peaks[-1]] < highs[peaks[-2]] * 0.98:
                    support = (lows[troughs[-1]] + lows[troughs[-2]]) / 2
                    return {
                        'pattern': 'descending_triangle',
                        'name': 'Descending Triangle',
                        'type': 'continuation',
                        'bias': 'bearish',
                        'confidence': 70,
                        'description': f'Support at ₦{support:.2f}, falling resistance',
                        'target': support * 0.95
                    }
        return None
    
    def _detect_bull_flag(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Optional[Dict]:
        """Detect bull flag (strong rally followed by consolidation)."""
        if len(closes) < 15:
            return None
        
        # Check for prior uptrend (first 10 days)
        early_return = (closes[10] - closes[-1]) / closes[-1]
        
        # Check for consolidation (last 5 days)
        recent_range = (max(highs[:5]) - min(lows[:5])) / closes[0]
        
        if early_return > 0.10 and recent_range < 0.05:
            # Strong uptrend followed by tight consolidation
            return {
                'pattern': 'bull_flag',
                'name': 'Bull Flag',
                'type': 'continuation',
                'bias': 'bullish',
                'confidence': 65,
                'description': f'{early_return*100:.1f}% rally with {recent_range*100:.1f}% consolidation',
                'target': closes[0] * 1.08
            }
        return None
    
    def _detect_bear_flag(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Optional[Dict]:
        """Detect bear flag (strong decline followed by consolidation)."""
        if len(closes) < 15:
            return None
        
        early_return = (closes[10] - closes[-1]) / closes[-1]
        recent_range = (max(highs[:5]) - min(lows[:5])) / closes[0]
        
        if early_return < -0.10 and recent_range < 0.05:
            return {
                'pattern': 'bear_flag',
                'name': 'Bear Flag',
                'type': 'continuation',
                'bias': 'bearish',
                'confidence': 65,
                'description': f'{early_return*100:.1f}% decline with {recent_range*100:.1f}% consolidation',
                'target': closes[0] * 0.92
            }
        return None
    
    def _detect_breakout(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Optional[Dict]:
        """Detect support/resistance breakout."""
        if len(closes) < 20:
            return None
        
        # Calculate 20-day high and low
        high_20 = max(highs[1:20])
        low_20 = min(lows[1:20])
        current = closes[0]
        
        # Breakout above resistance
        if current > high_20 * 1.02:
            return {
                'pattern': 'breakout_up',
                'name': 'Resistance Breakout',
                'type': 'breakout',
                'bias': 'bullish',
                'confidence': 80,
                'description': f'Broke above 20-day high of ₦{high_20:.2f}',
                'target': current * 1.10
            }
        
        # Breakdown below support
        if current < low_20 * 0.98:
            return {
                'pattern': 'breakout_down',
                'name': 'Support Breakdown',
                'type': 'breakout',
                'bias': 'bearish',
                'confidence': 80,
                'description': f'Broke below 20-day low of ₦{low_20:.2f}',
                'target': current * 0.90
            }
        
        return None
