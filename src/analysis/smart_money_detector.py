"""
Smart Money Detector & Anomaly Detection for MetaQuant Nigeria.
Identifies unusual market activity and institutional patterns.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import logging
import math

logger = logging.getLogger(__name__)


class SmartMoneyDetector:
    """
    Detects unusual market activity that may indicate smart money flow:
    - Unusual volume spikes
    - Accumulation/Distribution patterns
    - Block trade indicators
    - Stealth buying/selling
    """
    
    # Thresholds
    UNUSUAL_VOLUME_THRESHOLD = 2.0  # 2x average volume
    HIGH_VOLUME_THRESHOLD = 3.0     # 3x = very unusual
    EXTREME_VOLUME_THRESHOLD = 5.0  # 5x = institutional activity
    
    def __init__(self, db=None):
        self.db = db
    
    def analyze_stocks(self, stocks_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze all stocks for smart money patterns.
        
        Returns comprehensive analysis with:
        - Unusual volume stocks
        - Accumulation stocks
        - Distribution stocks
        - Breakout candidates
        - Anomalies
        """
        if not stocks_data:
            return {'error': 'No data provided'}
        
        # Containers for detected patterns
        unusual_volume = []
        accumulation = []
        distribution = []
        breakouts_up = []
        breakouts_down = []
        anomalies = []
        stealth_accumulation = []  # Quiet buying on low volatility
        block_trades = []  # Large institutional-sized trades
        
        # Market-wide statistics
        total_volume = 0
        gainers = 0
        losers = 0
        total_change = 0
        
        for stock in stocks_data:
            symbol = stock.get('symbol', '')
            volume = stock.get('volume', 0) or 0
            avg_volume = stock.get('average_volume_10d_calc', 0) or 0
            chg_1d = stock.get('change', 0) or 0
            chg_1w = stock.get('Perf.W', 0) or 0
            chg_1m = stock.get('Perf.1M', 0) or 0
            price = stock.get('close', 0) or 0
            high = stock.get('high', 0) or 0
            low = stock.get('low', 0) or 0
            open_price = stock.get('open', 0) or 0
            
            # Handle NaN
            for var in [volume, avg_volume, chg_1d, chg_1w, chg_1m, price, high, low, open_price]:
                if not isinstance(var, (int, float)) or var != var:
                    var = 0
            
            total_volume += volume
            total_change += chg_1d
            if chg_1d > 0:
                gainers += 1
            elif chg_1d < 0:
                losers += 1
            
            # Volume ratio calculation
            vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # ===== UNUSUAL VOLUME DETECTION =====
            if vol_ratio >= self.UNUSUAL_VOLUME_THRESHOLD:
                category = 'extreme' if vol_ratio >= self.EXTREME_VOLUME_THRESHOLD else \
                          ('high' if vol_ratio >= self.HIGH_VOLUME_THRESHOLD else 'unusual')
                
                unusual_volume.append({
                    'symbol': symbol,
                    'price': price,
                    'change': chg_1d,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'vol_ratio': vol_ratio,
                    'category': category,
                    'signal': 'BUY' if chg_1d > 0 else ('SELL' if chg_1d < 0 else 'WATCH'),
                })
            
            # ===== ACCUMULATION/DISTRIBUTION =====
            # Accumulation: Close near high with above-average volume = buying pressure
            # Distribution: Close near low with above-average volume = selling pressure
            
            # Money Flow Multiplier: ranges from -1 (close at low) to +1 (close at high)
            if high > low:
                money_flow_mult = ((price - low) - (high - price)) / (high - low)
                daily_range_pct = ((high - low) / low) * 100 if low > 0 else 0
            else:
                money_flow_mult = 0
                daily_range_pct = 0
            
            # Score calculation:
            # - money_flow_mult: -1 to +1 (converted to 0-50 base)
            # - vol_ratio: multiplier effect (1.5x = weak, 2x = moderate, 3x+ = strong)
            # - Price change direction adds/subtracts points
            
            # Accumulation signals (close in upper half of range + volume above average)
            if money_flow_mult > 0.2 and vol_ratio >= 1.3:
                # Base score from money flow position (0-40)
                base_score = int(money_flow_mult * 40)
                # Volume bonus (0-30)
                vol_bonus = min(30, int((vol_ratio - 1) * 15))
                # Price direction bonus (0-20)
                price_bonus = min(20, max(0, int(chg_1d * 4))) if chg_1d > 0 else 0
                # Weekly trend bonus (0-10)
                trend_bonus = min(10, max(0, int(chg_1w))) if chg_1w > 0 else 0
                
                score = min(100, base_score + vol_bonus + price_bonus + trend_bonus)
                
                # Strength based on score
                if score >= 75:
                    strength = 'STRONG'
                elif score >= 50:
                    strength = 'MODERATE'
                else:
                    strength = 'WEAK'
                
                accumulation.append({
                    'symbol': symbol,
                    'price': price,
                    'change': chg_1d,
                    'chg_1w': chg_1w,
                    'vol_ratio': vol_ratio,
                    'money_flow': money_flow_mult,
                    'daily_range': daily_range_pct,
                    'score': score,
                    'signal_strength': strength,
                })
            
            # Distribution signals (close in lower half of range + volume above average)
            if money_flow_mult < -0.2 and vol_ratio >= 1.3:
                # Base score from money flow position (0-40)
                base_score = int(abs(money_flow_mult) * 40)
                # Volume bonus (0-30)
                vol_bonus = min(30, int((vol_ratio - 1) * 15))
                # Price direction bonus (0-20) - negative change is bad
                price_bonus = min(20, max(0, int(abs(chg_1d) * 4))) if chg_1d < 0 else 0
                # Weekly trend penalty (0-10)
                trend_bonus = min(10, max(0, int(abs(chg_1w)))) if chg_1w < 0 else 0
                
                score = min(100, base_score + vol_bonus + price_bonus + trend_bonus)
                
                # Strength based on score
                if score >= 75:
                    strength = 'STRONG'
                elif score >= 50:
                    strength = 'MODERATE'
                else:
                    strength = 'WEAK'
                
                distribution.append({
                    'symbol': symbol,
                    'price': price,
                    'change': chg_1d,
                    'chg_1w': chg_1w,
                    'vol_ratio': vol_ratio,
                    'money_flow': money_flow_mult,
                    'daily_range': daily_range_pct,
                    'score': score,
                    'signal_strength': strength,
                })
            
            # ===== BREAKOUT DETECTION =====
            # Using RSI, SMA relationships from TradingView data
            # Try both possible field names
            sma20 = stock.get('SMA20') or stock.get('sma_20') or 0
            sma50 = stock.get('SMA50') or stock.get('sma_50') or 0
            rsi = stock.get('RSI') or stock.get('RSI14') or stock.get('rsi') or 50
            
            # Ensure numeric and not NaN
            if not isinstance(sma20, (int, float)) or sma20 != sma20:
                sma20 = 0
            if not isinstance(sma50, (int, float)) or sma50 != sma50:
                sma50 = 0
            if not isinstance(rsi, (int, float)) or rsi != rsi:
                rsi = 50
            
            # Bullish breakout: price > SMA20 > SMA50, RSI > 55, volume above average
            # Also require positive 1D change for confirmation
            if sma20 > 0 and sma50 > 0:
                if price > sma20 > sma50 and rsi > 55 and vol_ratio > 1.3 and chg_1d > 0:
                    breakouts_up.append({
                        'symbol': symbol,
                        'price': price,
                        'change': chg_1d,
                        'rsi': rsi,
                        'vol_ratio': vol_ratio,
                        'sma20': sma20,
                        'sma50': sma50,
                        'type': 'BULLISH_BREAKOUT',
                    })
                
                # Bearish breakdown: price < SMA20 < SMA50, RSI < 45
                # Also require negative 1D change for confirmation
                if price < sma20 < sma50 and rsi < 45 and vol_ratio > 1.2 and chg_1d < 0:
                    breakouts_down.append({
                        'symbol': symbol,
                        'price': price,
                        'change': chg_1d,
                        'rsi': rsi,
                        'vol_ratio': vol_ratio,
                        'sma20': sma20,
                        'sma50': sma50,
                        'type': 'BEARISH_BREAKDOWN',
                    })
            
            # ===== ANOMALY DETECTION =====
            # Statistical outliers in price or volume
            if abs(chg_1d) >= 8.0:  # 8%+ move is unusual
                anomalies.append({
                    'symbol': symbol,
                    'price': price,
                    'change': chg_1d,
                    'type': 'PRICE_SPIKE' if chg_1d > 0 else 'PRICE_CRASH',
                    'severity': 'HIGH' if abs(chg_1d) >= 10 else 'MODERATE',
                })
            
            if vol_ratio >= 5.0:  # 5x volume is anomalous
                anomalies.append({
                    'symbol': symbol,
                    'price': price,
                    'vol_ratio': vol_ratio,
                    'type': 'VOLUME_ANOMALY',
                    'severity': 'HIGH' if vol_ratio >= 8 else 'MODERATE',
                })
            
            # ===== STEALTH ACCUMULATION =====
            # Quiet buying: positive money flow, low volatility, moderate volume
            # Indicates smart money quietly accumulating without drawing attention
            if high > low:
                volatility = ((high - low) / low) * 100 if low > 0 else 0
            else:
                volatility = 0
            
            # Stealth criteria: low volatility (<3%), positive money flow, slight volume increase
            if (money_flow_mult > 0.3 and volatility < 3.0 and 
                vol_ratio >= 1.1 and vol_ratio < 2.0 and 
                abs(chg_1d) < 3.0 and chg_1w > 0):
                
                stealth_score = int((money_flow_mult * 40) + (chg_1w * 3) + ((2 - volatility) * 10))
                stealth_score = min(100, max(0, stealth_score))
                
                stealth_accumulation.append({
                    'symbol': symbol,
                    'price': price,
                    'change': chg_1d,
                    'chg_1w': chg_1w,
                    'volatility': volatility,
                    'vol_ratio': vol_ratio,
                    'score': stealth_score,
                    'signal': 'ACCUMULATING' if stealth_score >= 50 else 'WATCHING',
                })
            
            # ===== BLOCK TRADES =====
            # Large institutional-sized trades: very high volume with price impact
            # Simulated by extreme volume ratio + significant move
            if vol_ratio >= 3.0 and volume > 0:
                # Estimate trade value
                trade_value = price * volume
                
                # Block trade criteria: high volume with meaningful price move
                if abs(chg_1d) >= 1.0:
                    block_trades.append({
                        'symbol': symbol,
                        'price': price,
                        'change': chg_1d,
                        'vol_ratio': vol_ratio,
                        'trade_value': trade_value,
                        'direction': 'BUY' if chg_1d > 0 else 'SELL',
                        'size': 'LARGE' if vol_ratio >= 5.0 else 'MEDIUM',
                    })
        
        # Sort by relevance
        unusual_volume.sort(key=lambda x: x['vol_ratio'], reverse=True)
        accumulation.sort(key=lambda x: x['score'], reverse=True)
        distribution.sort(key=lambda x: x['score'], reverse=True)
        stealth_accumulation.sort(key=lambda x: x['score'], reverse=True)
        block_trades.sort(key=lambda x: x['vol_ratio'], reverse=True)
        
        # Calculate market regime
        market_regime = self._calculate_market_regime(
            gainers, losers, total_change / len(stocks_data) if stocks_data else 0,
            unusual_volume, accumulation, distribution
        )
        
        return {
            'unusual_volume': unusual_volume[:15],
            'accumulation': accumulation[:10],
            'distribution': distribution[:10],
            'breakouts_up': breakouts_up[:5],
            'breakouts_down': breakouts_down[:5],
            'anomalies': anomalies[:10],
            'stealth_accumulation': stealth_accumulation[:10],
            'block_trades': block_trades[:10],
            'market_regime': market_regime,
            'stats': {
                'total_stocks': len(stocks_data),
                'gainers': gainers,
                'losers': losers,
                'unusual_volume_count': len(unusual_volume),
                'accumulation_count': len(accumulation),
                'distribution_count': len(distribution),
                'stealth_count': len(stealth_accumulation),
                'block_trade_count': len(block_trades),
            }
        }
    
    def _calculate_market_regime(self, gainers: int, losers: int, avg_change: float,
                                  unusual_vol: List, accum: List, distrib: List) -> Dict:
        """Calculate overall market regime."""
        total = gainers + losers or 1
        breadth = (gainers - losers) / total * 100
        
        # Health score (0-100)
        health_score = 50
        
        # Breadth contribution (+/- 25)
        health_score += breadth * 0.25
        
        # Volume activity contribution (+/- 15)
        vol_score = min(15, len(unusual_vol) * 1.5)
        net_flow = len(accum) - len(distrib)
        if net_flow > 0:
            health_score += vol_score
        elif net_flow < 0:
            health_score -= vol_score
        
        # Average change contribution (+/- 10)
        health_score += min(10, max(-10, avg_change * 2))
        
        # Clamp to 0-100
        health_score = max(0, min(100, health_score))
        
        # Regime classification
        if health_score >= 70:
            regime = 'BULLISH'
            trend = 'STRONG' if health_score >= 80 else 'MODERATE'
        elif health_score >= 55:
            regime = 'NEUTRAL_BULLISH'
            trend = 'WEAK'
        elif health_score >= 45:
            regime = 'NEUTRAL'
            trend = 'SIDEWAYS'
        elif health_score >= 30:
            regime = 'NEUTRAL_BEARISH'
            trend = 'WEAK'
        else:
            regime = 'BEARISH'
            trend = 'STRONG' if health_score <= 20 else 'MODERATE'
        
        # Risk signal
        if regime in ['BULLISH', 'NEUTRAL_BULLISH']:
            risk_signal = 'RISK_ON'
        elif regime in ['BEARISH', 'NEUTRAL_BEARISH']:
            risk_signal = 'RISK_OFF'
        else:
            risk_signal = 'NEUTRAL'
        
        return {
            'health_score': round(health_score, 1),
            'regime': regime,
            'trend_strength': trend,
            'risk_signal': risk_signal,
            'breadth': round(breadth, 1),
            'net_flow': net_flow,
        }


class AnomalyScanner:
    """
    Scans for market anomalies and generates alerts.
    """
    
    def __init__(self):
        self.alerts = []
    
    def scan(self, stocks_data: List[Dict], sector_data: Dict = None) -> List[Dict]:
        """
        Generate anomaly alerts from stock and sector data.
        
        Returns list of alerts with:
        - type: PRICE_SPIKE, VOLUME_SPIKE, MOMENTUM_DIVERGENCE, RSI_EXTREME, etc.
        - severity: HIGH, MODERATE, LOW
        - symbol: affected stock
        - message: human-readable description
        """
        self.alerts = []
        
        for stock in stocks_data:
            symbol = stock.get('symbol', '')
            chg_1d = stock.get('change', 0) or 0
            chg_1w = stock.get('Perf.W', 0) or 0
            chg_1m = stock.get('Perf.1M', 0) or 0
            volume = stock.get('volume', 0) or 0
            avg_volume = stock.get('average_volume_10d_calc', 0) or 0
            rsi = stock.get('RSI', 50) or 50
            
            # Ensure numeric values
            if not isinstance(chg_1d, (int, float)) or chg_1d != chg_1d:
                chg_1d = 0.0
            if not isinstance(chg_1w, (int, float)) or chg_1w != chg_1w:
                chg_1w = 0.0
            if not isinstance(rsi, (int, float)) or rsi != rsi:
                rsi = 50.0
            
            vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # ===== PRICE SPIKE (5%+ moves) =====
            if abs(chg_1d) >= 5.0:
                self.alerts.append({
                    'type': 'PRICE_SPIKE' if chg_1d > 0 else 'PRICE_DROP',
                    'severity': 'HIGH' if abs(chg_1d) >= 8 else 'MODERATE',
                    'symbol': symbol,
                    'message': f"{symbol} moved {chg_1d:+.1f}% today",
                })
            
            # ===== HIGH VOLUME MOVE (3x+ volume with 3%+ move) =====
            if vol_ratio >= 3.0 and abs(chg_1d) >= 3:
                direction = '↑' if chg_1d > 0 else '↓'
                self.alerts.append({
                    'type': 'HIGH_VOL_MOVE',
                    'severity': 'HIGH',
                    'symbol': symbol,
                    'message': f"{symbol} {direction} {abs(chg_1d):.1f}% on {vol_ratio:.1f}x vol",
                })
            
            # ===== RSI EXTREMES =====
            if rsi >= 75:
                self.alerts.append({
                    'type': 'RSI_OVERBOUGHT',
                    'severity': 'MODERATE',
                    'symbol': symbol,
                    'message': f"{symbol} RSI overbought at {rsi:.0f}",
                })
            elif rsi <= 25:
                self.alerts.append({
                    'type': 'RSI_OVERSOLD',
                    'severity': 'MODERATE',
                    'symbol': symbol,
                    'message': f"{symbol} RSI oversold at {rsi:.0f}",
                })
            
            # ===== MOMENTUM DIVERGENCE (1D vs 1W moving opposite) =====
            if chg_1d * chg_1w < 0 and abs(chg_1d) >= 3 and abs(chg_1w) >= 5:
                direction = "reversal up" if chg_1d > 0 else "reversal down"
                self.alerts.append({
                    'type': 'MOMENTUM_SHIFT',
                    'severity': 'MODERATE',
                    'symbol': symbol,
                    'message': f"{symbol} {direction}: 1D={chg_1d:+.1f}% vs 1W={chg_1w:+.1f}%",
                })
            
            # ===== EXTREME 1W MOVERS (10%+ weekly) =====
            if abs(chg_1w) >= 10:
                self.alerts.append({
                    'type': 'WEEKLY_EXTREME',
                    'severity': 'MODERATE',
                    'symbol': symbol,
                    'message': f"{symbol} {chg_1w:+.1f}% this week",
                })
        
        # Sort by severity
        severity_order = {'HIGH': 0, 'MODERATE': 1, 'LOW': 2}
        self.alerts.sort(key=lambda x: severity_order.get(x['severity'], 2))
        
        return self.alerts[:20]

