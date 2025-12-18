"""
Order Flow Analysis Engine
Calculates CVD, volume delta, flow imbalance, and price impact metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FlowData:
    """Flow analysis data for a single period."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    delta: float          # Volume delta (positive=buying, negative=selling)
    cvd: float            # Cumulative Volume Delta
    flow_in: float        # Buying volume estimate
    flow_out: float       # Selling volume estimate
    imbalance: float      # Flow imbalance ratio (-1 to 1)
    price_impact: float   # Price change per unit volume


@dataclass 
class FlowSignal:
    """Trade signal generated from flow analysis."""
    date: date
    signal_type: str      # BULLISH_DIV, BEARISH_DIV, ACCUMULATION, DISTRIBUTION, BREAKOUT
    strength: int         # Signal strength 0-100
    price: float          # Price at signal
    cvd: float            # CVD at signal
    description: str      # Human readable description


class FlowAnalyzer:
    """
    Analyzes order flow using price and volume data.
    
    Since NGX doesn't provide bid/ask data, we estimate delta using:
    - Candle direction method: Close > Open = buying, Close < Open = selling
    - Volume weighted by candle position within range
    """
    
    def __init__(self):
        self.flow_data: List[FlowData] = []
        self.signals: List[FlowSignal] = []
    
    def calculate_delta(self, open_p: float, high: float, low: float, 
                        close: float, volume: int) -> Tuple[float, float, float]:
        """
        Calculate volume delta for a single candle.
        
        Uses the candle body position method:
        - Buying pressure = volume * (close - low) / (high - low)
        - Selling pressure = volume * (high - close) / (high - low)
        
        Returns:
            Tuple of (delta, flow_in, flow_out)
        """
        if volume == 0:
            return 0.0, 0.0, 0.0
        
        range_size = high - low
        if range_size == 0:
            # Doji candle - neutral
            return 0.0, volume / 2, volume / 2
        
        # Calculate buying/selling pressure based on close position
        close_position = (close - low) / range_size  # 0 to 1
        
        flow_in = volume * close_position       # Buying volume estimate
        flow_out = volume * (1 - close_position)  # Selling volume estimate
        delta = flow_in - flow_out
        
        return delta, flow_in, flow_out
    
    def calculate_imbalance(self, flow_in: float, flow_out: float) -> float:
        """
        Calculate flow imbalance ratio.
        
        Returns:
            Imbalance from -1 (all selling) to +1 (all buying)
        """
        total = flow_in + flow_out
        if total == 0:
            return 0.0
        return (flow_in - flow_out) / total
    
    def calculate_price_impact(self, price_change: float, volume: int) -> float:
        """
        Calculate price impact - how much price moves per unit volume.
        Higher values indicate lower liquidity / stronger moves.
        """
        if volume == 0:
            return 0.0
        return abs(price_change) / volume * 1000000  # Normalize
    
    def analyze(self, price_history: List[Dict]) -> List[FlowData]:
        """
        Analyze price history and calculate all flow metrics.
        
        Args:
            price_history: List of dicts with date, open, high, low, close, volume
            
        Returns:
            List of FlowData objects with calculated metrics
        """
        self.flow_data = []
        cvd = 0.0
        prev_close = None
        
        for candle in price_history:
            try:
                open_p = float(candle.get('open', 0))
                high = float(candle.get('high', 0))
                low = float(candle.get('low', 0))
                close = float(candle.get('close', 0))
                volume = int(candle.get('volume', 0))
                
                # Parse date
                raw_date = candle.get('date')
                if isinstance(raw_date, str):
                    candle_date = datetime.strptime(raw_date, '%Y-%m-%d').date()
                elif isinstance(raw_date, (date, datetime)):
                    candle_date = raw_date if isinstance(raw_date, date) else raw_date.date()
                else:
                    candle_date = date.today()
                
                # Calculate delta and flow
                delta, flow_in, flow_out = self.calculate_delta(open_p, high, low, close, volume)
                cvd += delta
                
                # Calculate imbalance
                imbalance = self.calculate_imbalance(flow_in, flow_out)
                
                # Calculate price impact
                price_change = close - prev_close if prev_close else 0
                price_impact = self.calculate_price_impact(price_change, volume)
                
                self.flow_data.append(FlowData(
                    date=candle_date,
                    open=open_p,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    delta=delta,
                    cvd=cvd,
                    flow_in=flow_in,
                    flow_out=flow_out,
                    imbalance=imbalance,
                    price_impact=price_impact
                ))
                
                prev_close = close
                
            except Exception as e:
                logger.warning(f"Error processing candle: {e}")
                continue
        
        return self.flow_data
    
    def detect_divergence(self, lookback: int = 10) -> List[FlowSignal]:
        """
        Detect price/CVD divergences.
        
        Bullish divergence: Price making lower lows, CVD making higher lows
        Bearish divergence: Price making higher highs, CVD making lower highs
        """
        if len(self.flow_data) < lookback * 2:
            return []
        
        signals = []
        
        for i in range(lookback, len(self.flow_data)):
            # Get current and lookback data
            current = self.flow_data[i]
            lookback_data = self.flow_data[i-lookback:i]
            
            # Find lows and highs in lookback period
            price_low = min(d.low for d in lookback_data)
            price_high = max(d.high for d in lookback_data)
            cvd_low = min(d.cvd for d in lookback_data)
            cvd_high = max(d.cvd for d in lookback_data)
            
            # Bullish divergence: price new low, CVD higher low
            if current.low <= price_low and current.cvd > cvd_low:
                strength = min(100, int(abs(current.cvd - cvd_low) / abs(cvd_high - cvd_low) * 100)) if cvd_high != cvd_low else 50
                signals.append(FlowSignal(
                    date=current.date,
                    signal_type="BULLISH_DIVERGENCE",
                    strength=strength,
                    price=current.close,
                    cvd=current.cvd,
                    description=f"Price at low {current.close:.2f} while CVD rising"
                ))
            
            # Bearish divergence: price new high, CVD lower high
            elif current.high >= price_high and current.cvd < cvd_high:
                strength = min(100, int(abs(cvd_high - current.cvd) / abs(cvd_high - cvd_low) * 100)) if cvd_high != cvd_low else 50
                signals.append(FlowSignal(
                    date=current.date,
                    signal_type="BEARISH_DIVERGENCE", 
                    strength=strength,
                    price=current.close,
                    cvd=current.cvd,
                    description=f"Price at high {current.close:.2f} while CVD falling"
                ))
        
        return signals
    
    def detect_accumulation(self, lookback: int = 5) -> List[FlowSignal]:
        """
        Detect accumulation/distribution patterns.
        
        Accumulation: High volume, flat price, positive CVD trend
        Distribution: High volume, flat price, negative CVD trend
        """
        if len(self.flow_data) < lookback:
            return []
        
        signals = []
        avg_volume = sum(d.volume for d in self.flow_data) / len(self.flow_data)
        
        for i in range(lookback, len(self.flow_data)):
            window = self.flow_data[i-lookback:i+1]
            
            # Check high volume
            window_volume = sum(d.volume for d in window) / len(window)
            if window_volume < avg_volume * 1.5:
                continue
            
            # Check flat price (less than 2% range)
            price_range = max(d.high for d in window) - min(d.low for d in window)
            avg_price = sum(d.close for d in window) / len(window)
            if price_range / avg_price > 0.02:
                continue
            
            # Check CVD trend
            cvd_change = window[-1].cvd - window[0].cvd
            current = window[-1]
            
            if cvd_change > 0:
                signals.append(FlowSignal(
                    date=current.date,
                    signal_type="ACCUMULATION",
                    strength=min(100, int(abs(cvd_change) / avg_volume * 100)),
                    price=current.close,
                    cvd=current.cvd,
                    description=f"High volume {window_volume:,.0f} with flat price, positive CVD"
                ))
            else:
                signals.append(FlowSignal(
                    date=current.date,
                    signal_type="DISTRIBUTION",
                    strength=min(100, int(abs(cvd_change) / avg_volume * 100)),
                    price=current.close,
                    cvd=current.cvd,
                    description=f"High volume {window_volume:,.0f} with flat price, negative CVD"
                ))
        
        return signals
    
    def generate_all_signals(self) -> List[FlowSignal]:
        """Generate all types of signals."""
        self.signals = []
        self.signals.extend(self.detect_divergence())
        self.signals.extend(self.detect_accumulation())
        
        # Sort by date
        self.signals.sort(key=lambda s: s.date)
        
        return self.signals
    
    def get_current_metrics(self) -> Dict:
        """Get current flow metrics summary."""
        if not self.flow_data:
            return {}
        
        latest = self.flow_data[-1]
        recent = self.flow_data[-5:] if len(self.flow_data) >= 5 else self.flow_data
        
        return {
            "cvd": latest.cvd,
            "cvd_trend": "RISING" if len(recent) > 1 and recent[-1].cvd > recent[0].cvd else "FALLING",
            "delta": latest.delta,
            "imbalance": latest.imbalance,
            "imbalance_pct": f"{latest.imbalance * 100:+.1f}%",
            "price_impact": latest.price_impact,
            "total_flow_in": sum(d.flow_in for d in recent),
            "total_flow_out": sum(d.flow_out for d in recent),
            "net_flow": sum(d.delta for d in recent),
        }


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = [
        {"date": "2024-01-01", "open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000000},
        {"date": "2024-01-02", "open": 104, "high": 108, "low": 103, "close": 107, "volume": 1200000},
        {"date": "2024-01-03", "open": 107, "high": 110, "low": 105, "close": 106, "volume": 900000},
        {"date": "2024-01-04", "open": 106, "high": 107, "low": 102, "close": 103, "volume": 1500000},
        {"date": "2024-01-05", "open": 103, "high": 106, "low": 101, "close": 105, "volume": 1100000},
    ]
    
    analyzer = FlowAnalyzer()
    flow_data = analyzer.analyze(sample_data)
    
    print("Flow Analysis Results:")
    print("-" * 60)
    for fd in flow_data:
        print(f"{fd.date}: Price={fd.close:.2f} Delta={fd.delta:+,.0f} CVD={fd.cvd:+,.0f} Imbalance={fd.imbalance:+.2f}")
    
    print("\nCurrent Metrics:")
    metrics = analyzer.get_current_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")
