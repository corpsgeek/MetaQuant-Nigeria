"""
Advanced Flow Analysis Module for MetaQuant Nigeria.
Provides institutional-grade order flow metrics and volume profile analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class FlowAnalysis:
    """Advanced flow analysis calculations."""
    
    # NGX Trading Hours (West Africa Time)
    MARKET_OPEN = 10  # 10:00 AM
    MARKET_CLOSE = 14  # 2:30 PM (using 14 for simplicity)
    
    def __init__(self, ohlcv_data: List[Dict]):
        """
        Initialize with OHLCV data.
        
        Args:
            ohlcv_data: List of dicts with datetime, open, high, low, close, volume
        """
        self.data = sorted(ohlcv_data, key=lambda x: x['datetime'])
        self._calculate_deltas()
    
    def _calculate_deltas(self):
        """Calculate delta for each bar."""
        for bar in self.data:
            o = bar.get('open', 0) or 0
            c = bar.get('close', 0) or 0
            v = bar.get('volume', 0) or 0
            
            # Delta: positive if close > open (buying), negative otherwise
            if c > o:
                bar['delta'] = v
                bar['trade_type'] = 'BUY'
            elif c < o:
                bar['delta'] = -v
                bar['trade_type'] = 'SELL'
            else:
                bar['delta'] = 0
                bar['trade_type'] = 'FLAT'
    
    # =========================================================================
    # CUMULATIVE DELTA METRICS
    # =========================================================================
    
    def cumulative_delta(self) -> List[Dict]:
        """
        Calculate running cumulative delta.
        
        Returns:
            List of dicts with datetime and cumulative_delta
        """
        result = []
        cum_delta = 0
        
        for bar in self.data:
            cum_delta += bar.get('delta', 0)
            result.append({
                'datetime': bar['datetime'],
                'cumulative_delta': cum_delta,
                'delta': bar.get('delta', 0)
            })
        
        return result
    
    def session_delta(self) -> Dict[str, float]:
        """
        Calculate cumulative delta by trading session (date).
        
        Returns:
            Dict mapping date string to session delta
        """
        sessions = defaultdict(float)
        
        for bar in self.data:
            dt = bar['datetime']
            if isinstance(dt, datetime):
                date_str = dt.strftime('%Y-%m-%d')
            else:
                date_str = str(dt)[:10]
            
            sessions[date_str] += bar.get('delta', 0)
        
        return dict(sessions)
    
    def delta_divergence(self, lookback: int = 10) -> List[Dict]:
        """
        Detect delta divergence (price moving opposite to delta).
        
        Price up + Delta down = Distribution (bearish)
        Price down + Delta up = Accumulation (bullish)
        
        Args:
            lookback: Number of bars to compare
            
        Returns:
            List of divergence signals
        """
        if len(self.data) < lookback:
            return []
        
        divergences = []
        
        for i in range(lookback, len(self.data)):
            # Calculate price change and delta sum over lookback
            start_price = self.data[i - lookback].get('close', 0) or 0
            end_price = self.data[i].get('close', 0) or 0
            price_change = end_price - start_price
            
            delta_sum = sum(
                self.data[j].get('delta', 0) 
                for j in range(i - lookback, i + 1)
            )
            
            # Detect divergence
            if price_change > 0 and delta_sum < 0:
                divergences.append({
                    'datetime': self.data[i]['datetime'],
                    'type': 'DISTRIBUTION',
                    'signal': 'BEARISH',
                    'price_change': price_change,
                    'delta_sum': delta_sum,
                    'severity': abs(delta_sum) / (abs(price_change) + 1)
                })
            elif price_change < 0 and delta_sum > 0:
                divergences.append({
                    'datetime': self.data[i]['datetime'],
                    'type': 'ACCUMULATION',
                    'signal': 'BULLISH',
                    'price_change': price_change,
                    'delta_sum': delta_sum,
                    'severity': abs(delta_sum) / (abs(price_change) + 1)
                })
        
        return divergences
    
    def delta_momentum(self, period: int = 5) -> List[Dict]:
        """
        Calculate rate of change in delta (delta momentum).
        
        Args:
            period: Lookback period for momentum calculation
            
        Returns:
            List of delta momentum values
        """
        if len(self.data) < period:
            return []
        
        result = []
        
        for i in range(period, len(self.data)):
            current_delta = sum(
                self.data[j].get('delta', 0) 
                for j in range(i - period // 2, i + 1)
            )
            previous_delta = sum(
                self.data[j].get('delta', 0) 
                for j in range(i - period, i - period // 2)
            )
            
            momentum = current_delta - previous_delta
            
            result.append({
                'datetime': self.data[i]['datetime'],
                'delta_momentum': momentum,
                'trend': 'INCREASING' if momentum > 0 else 'DECREASING'
            })
        
        return result
    
    def delta_zscore(self, lookback: int = 20) -> List[Dict]:
        """
        Calculate Z-score of delta (standard deviations from mean).
        
        Args:
            lookback: Period for mean/std calculation
            
        Returns:
            List of Z-scores
        """
        if len(self.data) < lookback:
            return []
        
        result = []
        
        for i in range(lookback, len(self.data)):
            window = [self.data[j].get('delta', 0) for j in range(i - lookback, i)]
            
            mean_delta = statistics.mean(window) if window else 0
            try:
                std_delta = statistics.stdev(window) if len(window) > 1 else 1
            except:
                std_delta = 1
            
            current_delta = self.data[i].get('delta', 0)
            zscore = (current_delta - mean_delta) / std_delta if std_delta > 0 else 0
            
            result.append({
                'datetime': self.data[i]['datetime'],
                'delta': current_delta,
                'zscore': zscore,
                'signal': 'EXTREME_BUY' if zscore > 2 else 'EXTREME_SELL' if zscore < -2 else 'NORMAL'
            })
        
        return result
    
    def delta_exhaustion(self, lookback: int = 10, threshold: float = 0.3) -> List[Dict]:
        """
        Detect delta exhaustion (diminishing delta despite price continuation).
        
        Args:
            lookback: Period to analyze
            threshold: Ratio threshold for exhaustion
            
        Returns:
            List of exhaustion signals
        """
        if len(self.data) < lookback:
            return []
        
        exhaustion_signals = []
        
        for i in range(lookback, len(self.data)):
            # Price trend
            prices = [self.data[j].get('close', 0) or 0 for j in range(i - lookback, i + 1)]
            price_trend = prices[-1] - prices[0]
            
            # Delta trend (looking for diminishing)
            deltas = [abs(self.data[j].get('delta', 0)) for j in range(i - lookback, i + 1)]
            
            if len(deltas) > 1:
                first_half = sum(deltas[:len(deltas)//2])
                second_half = sum(deltas[len(deltas)//2:])
                
                # Exhaustion: price moving but delta diminishing
                if first_half > 0:
                    ratio = second_half / first_half
                    
                    if ratio < threshold and abs(price_trend) > 0:
                        exhaustion_signals.append({
                            'datetime': self.data[i]['datetime'],
                            'type': 'BUYING_EXHAUSTION' if price_trend > 0 else 'SELLING_EXHAUSTION',
                            'ratio': ratio,
                            'price_trend': price_trend
                        })
        
        return exhaustion_signals
    
    # =========================================================================
    # VOLUME PROFILE
    # =========================================================================
    
    def volume_profile(self, num_levels: int = 20) -> Dict[str, Any]:
        """
        Calculate volume-at-price profile.
        
        Args:
            num_levels: Number of price levels for histogram
            
        Returns:
            Dict with volume profile data
        """
        if not self.data:
            return {}
        
        # Get price range
        prices = [bar.get('close', 0) or 0 for bar in self.data if bar.get('close')]
        if not prices:
            return {}
        
        min_price = min(prices)
        max_price = max(prices)
        
        if min_price == max_price:
            return {}
        
        # Create price levels
        level_size = (max_price - min_price) / num_levels
        levels = defaultdict(lambda: {'volume': 0, 'buy_volume': 0, 'sell_volume': 0, 'count': 0})
        
        for bar in self.data:
            price = bar.get('close', 0) or 0
            volume = bar.get('volume', 0) or 0
            delta = bar.get('delta', 0)
            
            # Find level
            level_idx = int((price - min_price) / level_size)
            level_idx = min(level_idx, num_levels - 1)
            level_price = min_price + (level_idx + 0.5) * level_size
            
            levels[level_price]['volume'] += volume
            levels[level_price]['count'] += 1
            
            if delta > 0:
                levels[level_price]['buy_volume'] += volume
            else:
                levels[level_price]['sell_volume'] += abs(delta) if delta < 0 else 0
        
        # Convert to list and sort
        profile = []
        for price, data in sorted(levels.items()):
            delta_at_price = data['buy_volume'] - data['sell_volume']
            profile.append({
                'price': price,
                'volume': data['volume'],
                'buy_volume': data['buy_volume'],
                'sell_volume': data['sell_volume'],
                'delta': delta_at_price,
                'count': data['count']
            })
        
        # Calculate POC, VAH, VAL
        total_volume = sum(p['volume'] for p in profile)
        
        # POC: highest volume level
        poc_level = max(profile, key=lambda x: x['volume']) if profile else None
        
        # Value Area (70% of volume)
        if total_volume > 0 and profile:
            sorted_by_vol = sorted(profile, key=lambda x: x['volume'], reverse=True)
            cumulative = 0
            value_area = []
            
            for level in sorted_by_vol:
                cumulative += level['volume']
                value_area.append(level['price'])
                if cumulative >= total_volume * 0.7:
                    break
            
            vah = max(value_area) if value_area else max_price
            val = min(value_area) if value_area else min_price
        else:
            vah = max_price
            val = min_price
        
        return {
            'profile': profile,
            'poc': poc_level['price'] if poc_level else None,
            'poc_volume': poc_level['volume'] if poc_level else 0,
            'vah': vah,
            'val': val,
            'total_volume': total_volume,
            'price_range': (min_price, max_price)
        }
    
    def volume_nodes(self, profile: Dict = None, threshold_pct: float = 0.1) -> Dict[str, List]:
        """
        Identify high and low volume nodes.
        
        Args:
            profile: Volume profile dict (calculated if not provided)
            threshold_pct: Percentage threshold for node classification
            
        Returns:
            Dict with high_volume_nodes and low_volume_nodes
        """
        if profile is None:
            profile = self.volume_profile()
        
        if not profile or 'profile' not in profile:
            return {'high_volume_nodes': [], 'low_volume_nodes': []}
        
        levels = profile['profile']
        if not levels:
            return {'high_volume_nodes': [], 'low_volume_nodes': []}
        
        avg_volume = statistics.mean([l['volume'] for l in levels])
        
        high_threshold = avg_volume * (1 + threshold_pct)
        low_threshold = avg_volume * (1 - threshold_pct)
        
        hvn = [l for l in levels if l['volume'] > high_threshold]
        lvn = [l for l in levels if l['volume'] < low_threshold]
        
        return {
            'high_volume_nodes': sorted(hvn, key=lambda x: x['volume'], reverse=True),
            'low_volume_nodes': sorted(lvn, key=lambda x: x['volume'])
        }
    
    # =========================================================================
    # STATISTICAL METRICS
    # =========================================================================
    
    def vwap_analysis(self) -> Dict:
        """
        Calculate Volume Weighted Average Price (VWAP) and standard deviation bands.
        
        Returns:
            Dict with VWAP and band levels
        """
        if not self.data:
            return {}
        
        # Calculate VWAP: sum(price * volume) / sum(volume)
        total_pv = 0  # price * volume
        total_volume = 0
        
        prices = []
        
        for bar in self.data:
            typical_price = ((bar.get('high', 0) or 0) + 
                           (bar.get('low', 0) or 0) + 
                           (bar.get('close', 0) or 0)) / 3
            volume = bar.get('volume', 0) or 0
            
            total_pv += typical_price * volume
            total_volume += volume
            prices.append(typical_price)
        
        vwap = total_pv / total_volume if total_volume > 0 else 0
        
        # Calculate standard deviation
        if len(prices) > 1:
            std_dev = statistics.stdev(prices)
        else:
            std_dev = 0
        
        # Current price position
        current_price = self.data[-1].get('close', 0) if self.data else 0
        deviation = (current_price - vwap) / std_dev if std_dev > 0 else 0
        
        # Determine position
        if deviation > 2:
            position = 'EXTREME_ABOVE'
        elif deviation > 1:
            position = 'ABOVE'
        elif deviation < -2:
            position = 'EXTREME_BELOW'
        elif deviation < -1:
            position = 'BELOW'
        else:
            position = 'FAIR_VALUE'
        
        return {
            'vwap': vwap,
            'std_dev': std_dev,
            'upper_1': vwap + std_dev,
            'lower_1': vwap - std_dev,
            'upper_2': vwap + 2 * std_dev,
            'lower_2': vwap - 2 * std_dev,
            'current_price': current_price,
            'deviation': deviation,
            'position': position
        }
    
    def rvol_analysis(self, current_bar: Dict = None, lookback: int = 20) -> Dict:
        """
        Calculate Relative Volume (RVOL) statistics.
        
        Args:
            current_bar: Optional specific bar (uses last bar if not provided)
            lookback: Period for average calculation
            
        Returns:
            Dict with RVOL metrics
        """
        if len(self.data) < lookback:
            return {'rvol': 1, 'percentile': 50}
        
        volumes = [bar.get('volume', 0) or 0 for bar in self.data[-lookback:]]
        avg_volume = statistics.mean(volumes) if volumes else 1
        
        current_volume = (current_bar or self.data[-1]).get('volume', 0) or 0
        rvol = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Calculate percentile
        sorted_vols = sorted(volumes)
        percentile = (sorted_vols.index(min(sorted_vols, key=lambda x: abs(x - current_volume))) / len(sorted_vols)) * 100 if sorted_vols else 50
        
        return {
            'rvol': rvol,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'percentile': percentile,
            'classification': self._classify_rvol(rvol)
        }
    
    def _classify_rvol(self, rvol: float) -> str:
        """Classify RVOL into categories."""
        if rvol >= 5:
            return 'EXTREME'
        elif rvol >= 3:
            return 'VERY_HIGH'
        elif rvol >= 2:
            return 'HIGH'
        elif rvol >= 1.5:
            return 'ABOVE_AVERAGE'
        elif rvol >= 0.75:
            return 'NORMAL'
        else:
            return 'LOW'
    
    def block_trade_analysis(self, thresholds: Tuple[float, float, float] = (3, 5, 10)) -> Dict:
        """
        Categorize block trades by size.
        
        Args:
            thresholds: Tuple of RVOL thresholds for blocks (default: 3x, 5x, 10x)
            
        Returns:
            Dict with block trade statistics
        """
        if not self.data:
            return {}
        
        volumes = [bar.get('volume', 0) or 0 for bar in self.data]
        avg_vol = statistics.mean(volumes) if volumes else 0
        
        blocks = {
            f'>{thresholds[0]}x': [],
            f'>{thresholds[1]}x': [],
            f'>{thresholds[2]}x': []
        }
        
        for bar in self.data:
            vol = bar.get('volume', 0) or 0
            if avg_vol > 0:
                rvol = vol / avg_vol
                
                if rvol >= thresholds[2]:
                    blocks[f'>{thresholds[2]}x'].append(bar)
                elif rvol >= thresholds[1]:
                    blocks[f'>{thresholds[1]}x'].append(bar)
                elif rvol >= thresholds[0]:
                    blocks[f'>{thresholds[0]}x'].append(bar)
        
        return {
            'avg_volume': avg_vol,
            'blocks': blocks,
            'total_blocks': sum(len(v) for v in blocks.values())
        }
    
    def absorption_detection(self, volume_threshold: float = 2, price_threshold: float = 0.005) -> List[Dict]:
        """
        Detect absorption (high volume with minimal price movement).
        
        Args:
            volume_threshold: RVOL threshold
            price_threshold: Max price change percentage
            
        Returns:
            List of absorption events
        """
        if len(self.data) < 2:
            return []
        
        volumes = [bar.get('volume', 0) or 0 for bar in self.data]
        avg_vol = statistics.mean(volumes) if volumes else 0
        
        absorptions = []
        
        for bar in self.data:
            vol = bar.get('volume', 0) or 0
            o = bar.get('open', 0) or 0
            c = bar.get('close', 0) or 0
            
            if avg_vol > 0 and o > 0:
                rvol = vol / avg_vol
                price_change_pct = abs(c - o) / o
                
                if rvol >= volume_threshold and price_change_pct <= price_threshold:
                    absorptions.append({
                        'datetime': bar['datetime'],
                        'rvol': rvol,
                        'price_change_pct': price_change_pct,
                        'volume': vol,
                        'type': 'BUY_ABSORPTION' if c >= o else 'SELL_ABSORPTION'
                    })
        
        return absorptions
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    
    def summary_stats(self) -> Dict:
        """Calculate comprehensive summary statistics."""
        if not self.data:
            return {}
        
        volumes = [bar.get('volume', 0) or 0 for bar in self.data]
        deltas = [bar.get('delta', 0) for bar in self.data]
        
        buy_bars = len([d for d in deltas if d > 0])
        sell_bars = len([d for d in deltas if d < 0])
        
        cum_delta = sum(deltas)
        
        return {
            'total_bars': len(self.data),
            'total_volume': sum(volumes),
            'avg_volume': statistics.mean(volumes) if volumes else 0,
            'cumulative_delta': cum_delta,
            'buy_bars': buy_bars,
            'sell_bars': sell_bars,
            'buy_pressure_pct': (buy_bars / len(self.data) * 100) if self.data else 0,
            'avg_delta': statistics.mean(deltas) if deltas else 0,
            'delta_std': statistics.stdev(deltas) if len(deltas) > 1 else 0,
            'sentiment': 'BULLISH' if cum_delta > 0 else 'BEARISH' if cum_delta < 0 else 'NEUTRAL'
        }
    
    # =========================================================================
    # PHASE 2: ADVANCED STATISTICAL ENGINE
    # =========================================================================
    
    def rvol_percentile(self, lookback: int = 100) -> Dict:
        """
        Calculate RVOL percentile ranking across history.
        
        Args:
            lookback: Historical lookback period
            
        Returns:
            Dict with RVOL and percentile metrics
        """
        if len(self.data) < 2:
            return {'rvol': 1, 'percentile': 50, 'rank': 'NORMAL'}
        
        volumes = [bar.get('volume', 0) or 0 for bar in self.data[-lookback:]]
        current_vol = volumes[-1] if volumes else 0
        avg_vol = statistics.mean(volumes[:-1]) if len(volumes) > 1 else 1
        
        rvol = current_vol / avg_vol if avg_vol > 0 else 1
        
        # Calculate actual percentile
        sorted_vols = sorted(volumes)
        rank = sum(1 for v in sorted_vols if v <= current_vol)
        percentile = (rank / len(sorted_vols)) * 100 if sorted_vols else 50
        
        # Classify
        if percentile >= 95:
            rank_text = 'EXTREME'
        elif percentile >= 80:
            rank_text = 'VERY_HIGH'
        elif percentile >= 60:
            rank_text = 'ABOVE_AVG'
        elif percentile >= 40:
            rank_text = 'NORMAL'
        elif percentile >= 20:
            rank_text = 'BELOW_AVG'
        else:
            rank_text = 'LOW'
        
        return {
            'rvol': rvol,
            'percentile': percentile,
            'rank': rank_text,
            'current_volume': current_vol,
            'avg_volume': avg_vol
        }
    
    def trade_frequency(self, window_minutes: int = 60) -> Dict:
        """
        Analyze trade frequency patterns.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            Dict with trade frequency metrics
        """
        if len(self.data) < 2:
            return {}
        
        # Group by hour
        hourly_counts = defaultdict(int)
        hourly_volumes = defaultdict(float)
        
        for bar in self.data:
            dt = bar['datetime']
            if isinstance(dt, datetime):
                hour = dt.hour
            else:
                hour = 12  # Default
            
            hourly_counts[hour] += 1
            hourly_volumes[hour] += bar.get('volume', 0) or 0
        
        # Find peak hours
        if hourly_counts:
            peak_hour = max(hourly_counts.keys(), key=lambda h: hourly_counts[h])
            peak_volume_hour = max(hourly_volumes.keys(), key=lambda h: hourly_volumes[h])
        else:
            peak_hour = 12
            peak_volume_hour = 12
        
        # Trades per bar average
        total_bars = len(self.data)
        
        return {
            'total_bars': total_bars,
            'hourly_distribution': dict(hourly_counts),
            'hourly_volumes': dict(hourly_volumes),
            'peak_hour': peak_hour,
            'peak_volume_hour': peak_volume_hour,
            'avg_bars_per_hour': total_bars / len(hourly_counts) if hourly_counts else 0
        }
    
    def volume_acceleration(self, lookback: int = 10) -> Dict:
        """
        Detect volume acceleration/deceleration.
        
        Args:
            lookback: Period for comparison
            
        Returns:
            Dict with acceleration metrics
        """
        if len(self.data) < lookback * 2:
            return {'acceleration': 0, 'trend': 'STABLE'}
        
        recent_vols = [bar.get('volume', 0) or 0 for bar in self.data[-lookback:]]
        prior_vols = [bar.get('volume', 0) or 0 for bar in self.data[-lookback*2:-lookback]]
        
        recent_avg = statistics.mean(recent_vols) if recent_vols else 0
        prior_avg = statistics.mean(prior_vols) if prior_vols else 0
        
        if prior_avg > 0:
            acceleration = ((recent_avg - prior_avg) / prior_avg) * 100
        else:
            acceleration = 0
        
        if acceleration > 50:
            trend = 'STRONG_ACCELERATION'
        elif acceleration > 20:
            trend = 'ACCELERATION'
        elif acceleration > -20:
            trend = 'STABLE'
        elif acceleration > -50:
            trend = 'DECELERATION'
        else:
            trend = 'STRONG_DECELERATION'
        
        return {
            'acceleration': acceleration,
            'trend': trend,
            'recent_avg': recent_avg,
            'prior_avg': prior_avg
        }
    
    def price_efficiency(self) -> Dict:
        """
        Calculate price efficiency ratio (price movement per volume unit).
        
        Returns:
            Dict with efficiency metrics
        """
        if len(self.data) < 2:
            return {}
        
        total_price_movement = 0
        total_volume = 0
        
        for bar in self.data:
            h = bar.get('high', 0) or 0
            l = bar.get('low', 0) or 0
            v = bar.get('volume', 0) or 0
            
            total_price_movement += abs(h - l)
            total_volume += v
        
        efficiency = (total_price_movement / total_volume * 1000000) if total_volume > 0 else 0
        
        # Net efficiency (directional)
        start_price = self.data[0].get('close', 0) or 0
        end_price = self.data[-1].get('close', 0) or 0
        net_movement = end_price - start_price
        
        net_efficiency = (net_movement / total_volume * 1000000) if total_volume > 0 else 0
        
        return {
            'efficiency': efficiency,
            'net_efficiency': net_efficiency,
            'total_price_movement': total_price_movement,
            'total_volume': total_volume,
            'interpretation': 'HIGH_EFFICIENCY' if efficiency > 1 else 'NORMAL' if efficiency > 0.1 else 'LOW_EFFICIENCY'
        }
    
    def trade_size_distribution(self) -> Dict:
        """
        Analyze distribution of trade sizes (small/medium/large).
        
        Returns:
            Dict with trade size distribution
        """
        if not self.data:
            return {}
        
        volumes = [bar.get('volume', 0) or 0 for bar in self.data]
        if not volumes:
            return {}
        
        avg_vol = statistics.mean(volumes)
        
        small = [v for v in volumes if v < avg_vol * 0.5]
        medium = [v for v in volumes if avg_vol * 0.5 <= v < avg_vol * 2]
        large = [v for v in volumes if v >= avg_vol * 2]
        
        total = len(volumes)
        
        return {
            'small_pct': len(small) / total * 100 if total else 0,
            'medium_pct': len(medium) / total * 100 if total else 0,
            'large_pct': len(large) / total * 100 if total else 0,
            'small_count': len(small),
            'medium_count': len(medium),
            'large_count': len(large),
            'avg_volume': avg_vol,
            'dominant': 'RETAIL' if len(small) > len(large) else 'INSTITUTIONAL' if len(large) > len(small) else 'MIXED'
        }
    
    # =========================================================================
    # PATTERN RECOGNITION
    # =========================================================================
    
    def detect_climax(self, volume_threshold: float = 3, price_threshold: float = 0.02) -> List[Dict]:
        """
        Detect buying/selling climax patterns.
        
        Args:
            volume_threshold: RVOL threshold for climax
            price_threshold: Min price change for climax
            
        Returns:
            List of climax events
        """
        if len(self.data) < 10:
            return []
        
        volumes = [bar.get('volume', 0) or 0 for bar in self.data]
        avg_vol = statistics.mean(volumes) if volumes else 0
        
        climaxes = []
        
        for i, bar in enumerate(self.data):
            vol = bar.get('volume', 0) or 0
            o = bar.get('open', 0) or 0
            c = bar.get('close', 0) or 0
            
            if avg_vol > 0 and o > 0:
                rvol = vol / avg_vol
                price_change = (c - o) / o
                
                if rvol >= volume_threshold and abs(price_change) >= price_threshold:
                    climax_type = 'BUYING_CLIMAX' if price_change > 0 else 'SELLING_CLIMAX'
                    climaxes.append({
                        'datetime': bar['datetime'],
                        'type': climax_type,
                        'rvol': rvol,
                        'price_change_pct': price_change * 100,
                        'volume': vol
                    })
        
        return climaxes
    
    def detect_stopping_volume(self, volume_threshold: float = 2, price_threshold: float = 0.003) -> List[Dict]:
        """
        Detect stopping volume (high volume with minimal price change).
        
        Args:
            volume_threshold: Min RVOL for stopping volume
            price_threshold: Max price change for stopping volume
            
        Returns:
            List of stopping volume events
        """
        if len(self.data) < 5:
            return []
        
        volumes = [bar.get('volume', 0) or 0 for bar in self.data]
        avg_vol = statistics.mean(volumes) if volumes else 0
        
        stopping = []
        
        for i in range(1, len(self.data)):
            bar = self.data[i]
            prev_bar = self.data[i-1]
            
            vol = bar.get('volume', 0) or 0
            o = bar.get('open', 0) or 0
            c = bar.get('close', 0) or 0
            prev_c = prev_bar.get('close', 0) or 0
            
            if avg_vol > 0 and o > 0:
                rvol = vol / avg_vol
                price_change = abs((c - o) / o) if o > 0 else 0
                
                # Stopping volume: high vol, minimal price move, after a trend
                if rvol >= volume_threshold and price_change <= price_threshold:
                    trend = 'DOWNTREND_STOP' if c > prev_c else 'UPTREND_STOP'
                    stopping.append({
                        'datetime': bar['datetime'],
                        'type': trend,
                        'rvol': rvol,
                        'price_change_pct': price_change * 100,
                        'volume': vol
                    })
        
        return stopping
    
    def detect_volume_void(self, threshold: float = 0.3) -> List[Dict]:
        """
        Detect volume voids (low liquidity zones).
        
        Args:
            threshold: RVOL threshold for void
            
        Returns:
            List of void zones
        """
        profile = self.volume_profile(num_levels=20)
        if not profile or 'profile' not in profile:
            return []
        
        levels = profile['profile']
        avg_vol = statistics.mean([l['volume'] for l in levels]) if levels else 0
        
        voids = []
        for level in levels:
            if level['volume'] < avg_vol * threshold:
                voids.append({
                    'price': level['price'],
                    'volume': level['volume'],
                    'volume_ratio': level['volume'] / avg_vol if avg_vol > 0 else 0,
                    'type': 'VOLUME_VOID'
                })
        
        return voids
    
    # =========================================================================
    # ALERT GENERATION
    # =========================================================================
    
    def generate_alerts(self) -> List[Dict]:
        """
        Generate all active alerts based on current data.
        
        Returns:
            List of alert dictionaries with context
        """
        alerts = []
        
        # Get key price levels for context
        current_price = self.data[-1].get('close', 0) if self.data else 0
        profile = self.volume_profile(num_levels=15)
        poc = profile.get('poc', 0) if profile else 0
        vah = profile.get('vah', 0) if profile else 0
        val = profile.get('val', 0) if profile else 0
        
        # Z-score alerts
        zscore_data = self.delta_zscore()
        if zscore_data:
            latest = zscore_data[-1]
            if latest['signal'] != 'NORMAL':
                if latest['signal'] == 'EXTREME_BUY':
                    context = "Unusually high buying pressure. Volume entering at above-normal rates - potential institutional activity."
                    action = "Watch for continuation or exhaustion."
                else:
                    context = "Unusually high selling pressure. Heavy distribution occurring - potential panic or forced selling."
                    action = "Watch for capitulation bottom or further weakness."
                
                # Price position relative to key levels
                price_info = f"Price: ₦{current_price:,.2f}"
                if poc:
                    price_info += f" | POC: ₦{poc:,.2f}"
                if current_price > vah:
                    price_info += " (Above Value Area)"
                elif current_price < val:
                    price_info += " (Below Value Area)"
                
                alerts.append({
                    'type': 'DELTA_ZSCORE',
                    'severity': 'HIGH',
                    'message': f"Extreme delta Z-score: {latest['zscore']:.2f}σ",
                    'context': context,
                    'action': action,
                    'price_info': price_info,
                    'signal': latest['signal']
                })
        
        # Divergence alerts
        divergences = self.delta_divergence()
        if divergences:
            latest = divergences[-1]
            price_chg = latest.get('price_change', 0)
            if latest['type'] == 'DISTRIBUTION':
                context = f"Price up ₦{abs(price_chg):,.2f} but delta falling. Smart money may be selling into strength."
                action = f"Be cautious of longs. Watch support at VAL ₦{val:,.2f}."
            else:
                context = f"Price down ₦{abs(price_chg):,.2f} but delta rising. Smart money may be buying weakness."
                action = f"Watch for reversal above VAL ₦{val:,.2f}. Potential buying opportunity."
            
            alerts.append({
                'type': 'DIVERGENCE',
                'severity': 'MEDIUM',
                'message': f"{latest['type']} detected - {latest['signal']}",
                'context': context,
                'action': action,
                'price_info': f"Current: ₦{current_price:,.2f} | VAH: ₦{vah:,.2f} | VAL: ₦{val:,.2f}",
                'signal': latest['signal']
            })
        
        # Volume acceleration
        accel = self.volume_acceleration()
        if accel.get('trend') in ['STRONG_ACCELERATION', 'STRONG_DECELERATION']:
            if 'ACCELERATION' in accel['trend']:
                context = f"Volume surging +{accel['acceleration']:.0f}% vs prior period. Increased participation."
                action = "Trend likely to continue. Look for entries in direction of move."
            else:
                context = f"Volume dropping {accel['acceleration']:.0f}% vs prior period. Participation waning."
                action = "Wait for volume confirmation before new positions."
            
            alerts.append({
                'type': 'VOLUME_ACCELERATION',
                'severity': 'MEDIUM',
                'message': f"Volume {accel['trend']}: {accel['acceleration']:.1f}%",
                'context': context,
                'action': action,
                'price_info': f"Current: ₦{current_price:,.2f} | POC: ₦{poc:,.2f}",
                'signal': 'BULLISH' if 'ACCELERATION' in accel['trend'] else 'BEARISH'
            })
        
        # RVOL percentile
        rvol = self.rvol_percentile()
        if rvol.get('percentile', 50) >= 90:
            context = f"Volume at {rvol['percentile']:.0f}th percentile ({rvol['current_volume']:,.0f} vs avg {rvol['avg_volume']:,.0f})."
            action = "High volume = high conviction. Watch price direction for signal."
            
            alerts.append({
                'type': 'RVOL_EXTREME',
                'severity': 'HIGH',
                'message': f"RVOL at {rvol['percentile']:.0f}th percentile ({rvol['rvol']:.1f}x)",
                'context': context,
                'action': action,
                'price_info': f"Price: ₦{current_price:,.2f} | Key levels: ₦{val:,.2f} - ₦{vah:,.2f}",
                'signal': 'ATTENTION'
            })
        
        # Exhaustion
        exhaustion = self.delta_exhaustion()
        if exhaustion:
            latest = exhaustion[-1]
            trend_price = latest.get('price_trend', 0)
            if 'BUYING' in latest['type']:
                context = f"Price up ₦{abs(trend_price):,.2f} but buying pressure fading. Potential top forming."
                action = f"Tighten stops. Watch for break below POC ₦{poc:,.2f}."
            else:
                context = f"Price down ₦{abs(trend_price):,.2f} but selling pressure fading. Potential bottom forming."
                action = f"Watch for reversal above VAL ₦{val:,.2f}."
            
            alerts.append({
                'type': 'EXHAUSTION',
                'severity': 'HIGH',
                'message': f"{latest['type']} - delta diminishing",
                'context': context,
                'action': action,
                'price_info': f"Current: ₦{current_price:,.2f} | POC: ₦{poc:,.2f} | VAL: ₦{val:,.2f}",
                'signal': 'REVERSAL_WARNING'
            })
        
        # Climax detection
        climaxes = self.detect_climax()
        if climaxes:
            latest = climaxes[-1]
            climax_vol = latest.get('volume', 0)
            climax_chg = latest.get('price_change_pct', 0)
            if 'BUYING' in latest['type']:
                context = f"Extreme volume ({latest['rvol']:.1f}x, {climax_vol:,.0f} shares) with +{climax_chg:.1f}% move. Euphoric buying."
                action = f"Wait for pullback to POC ₦{poc:,.2f} before buying."
            else:
                context = f"Extreme volume ({latest['rvol']:.1f}x, {climax_vol:,.0f} shares) with {climax_chg:.1f}% drop. Panic selling."
                action = f"Watch for reversal near VAL ₦{val:,.2f}."
            
            alerts.append({
                'type': 'CLIMAX',
                'severity': 'HIGH',
                'message': f"{latest['type']} at {latest['rvol']:.1f}x RVOL",
                'context': context,
                'action': action,
                'price_info': f"Current: ₦{current_price:,.2f} | VAH: ₦{vah:,.2f} | VAL: ₦{val:,.2f}",
                'signal': 'REVERSAL_WARNING'
            })
        
        # Stopping volume
        stopping = self.detect_stopping_volume()
        if stopping:
            latest = stopping[-1]
            stop_vol = latest.get('volume', 0)
            if 'DOWNTREND' in latest['type']:
                context = f"High volume ({stop_vol:,.0f}) bar with minimal price drop. Buyers absorbing selling."
                action = f"Look for breakout above POC ₦{poc:,.2f}."
            else:
                context = f"High volume ({stop_vol:,.0f}) bar with minimal price rise. Sellers absorbing buying."
                action = f"Watch for break below POC ₦{poc:,.2f}."
            
            alerts.append({
                'type': 'STOPPING_VOLUME',
                'severity': 'MEDIUM',
                'message': f"{latest['type']} detected",
                'context': context,
                'action': action,
                'price_info': f"Current: ₦{current_price:,.2f} | POC: ₦{poc:,.2f}",
                'signal': 'TREND_CHANGE'
            })
        
        # Absorption
        absorptions = self.absorption_detection()
        if len(absorptions) >= 3:
            total_abs_vol = sum(a.get('volume', 0) for a in absorptions)
            context = f"{len(absorptions)} absorption events ({total_abs_vol:,.0f} total volume). Large orders filled without moving price."
            action = f"Smart money accumulating. Watch for breakout above VAH ₦{vah:,.2f}."
            
            alerts.append({
                'type': 'ABSORPTION',
                'severity': 'MEDIUM',
                'message': f"{len(absorptions)} absorption events detected",
                'context': context,
                'action': action,
                'price_info': f"Current: ₦{current_price:,.2f} | Value Area: ₦{val:,.2f} - ₦{vah:,.2f}",
                'signal': 'ACCUMULATION'
            })
        
        return alerts
    
    def iceberg_detection(self) -> List[Dict]:
        """
        Detect potential iceberg orders (large hidden orders).
        Iceberg characteristics: consistent volume at same price despite multiple fills.
        
        Returns:
            List of suspected iceberg events
        """
        if len(self.data) < 10:
            return []
        
        icebergs = []
        price_levels = defaultdict(list)
        
        # Group trades by price level
        for bar in self.data:
            price = round(bar.get('close', 0) or 0, 2)
            price_levels[price].append(bar)
        
        # Calculate average volume
        all_volumes = [b.get('volume', 0) or 0 for b in self.data if b.get('volume')]
        avg_vol = sum(all_volumes) / len(all_volumes) if all_volumes else 1
        
        # Look for price levels with repeated high-volume fills
        for price, bars in price_levels.items():
            if len(bars) >= 3:  # At least 3 fills at same price
                total_vol = sum(b.get('volume', 0) or 0 for b in bars)
                avg_fill = total_vol / len(bars)
                
                # If average fill is high and consistent, likely iceberg
                if avg_fill >= avg_vol * 1.5:
                    volumes = [b.get('volume', 0) or 0 for b in bars]
                    if volumes:
                        vol_std = statistics.stdev(volumes) if len(volumes) > 1 else 0
                        vol_mean = statistics.mean(volumes)
                        cv = vol_std / vol_mean if vol_mean > 0 else 0
                        
                        # Low coefficient of variation = consistent fills = iceberg
                        if cv < 0.5:
                            icebergs.append({
                                'price': price,
                                'fills': len(bars),
                                'total_volume': total_vol,
                                'avg_fill_size': avg_fill,
                                'side': 'BUY' if sum(b.get('delta', 0) for b in bars) > 0 else 'SELL'
                            })
        
        return icebergs
    
    def detect_imbalance(self) -> Dict:
        """
        Detect buy/sell imbalances in the order flow.
        
        Returns:
            Dict with imbalance analysis
        """
        if len(self.data) < 5:
            return {}
        
        recent_data = self.data[-50:]  # Last 50 bars
        
        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        stacked_buys = 0
        stacked_sells = 0
        current_stack = 0
        current_stack_type = None
        max_buy_stack = 0
        max_sell_stack = 0
        
        for bar in recent_data:
            delta = bar.get('delta', 0)
            volume = bar.get('volume', 0) or 0
            
            if delta > 0:
                buy_volume += volume
                buy_count += 1
                
                if current_stack_type == 'BUY':
                    current_stack += 1
                else:
                    if current_stack_type == 'SELL' and current_stack >= 3:
                        stacked_sells += 1
                        max_sell_stack = max(max_sell_stack, current_stack)
                    current_stack = 1
                    current_stack_type = 'BUY'
            elif delta < 0:
                sell_volume += volume
                sell_count += 1
                
                if current_stack_type == 'SELL':
                    current_stack += 1
                else:
                    if current_stack_type == 'BUY' and current_stack >= 3:
                        stacked_buys += 1
                        max_buy_stack = max(max_buy_stack, current_stack)
                    current_stack = 1
                    current_stack_type = 'SELL'
        
        # Check final stack
        if current_stack >= 3:
            if current_stack_type == 'BUY':
                stacked_buys += 1
                max_buy_stack = max(max_buy_stack, current_stack)
            else:
                stacked_sells += 1
                max_sell_stack = max(max_sell_stack, current_stack)
        
        # Calculate diagonal (price moving with delta)
        diagonal_count = 0
        for i in range(1, len(recent_data)):
            prev = recent_data[i-1]
            curr = recent_data[i]
            
            prev_close = prev.get('close', 0) or 0
            curr_close = curr.get('close', 0) or 0
            curr_delta = curr.get('delta', 0)
            
            # Price and delta moving together
            if (curr_close > prev_close and curr_delta > 0) or \
               (curr_close < prev_close and curr_delta < 0):
                diagonal_count += 1
        
        # Exhaustion detection
        exhaustion = 'None'
        if len(recent_data) >= 10:
            last_10 = recent_data[-10:]
            volumes = [b.get('volume', 0) or 0 for b in last_10]
            deltas = [b.get('delta', 0) for b in last_10]
            
            if volumes:
                # High volume but delta weakening = exhaustion
                avg_vol_last = sum(volumes) / len(volumes)
                sum_delta_last5 = sum(deltas[-5:])
                sum_delta_prev5 = sum(deltas[:5])
                
                all_vols = [b.get('volume', 0) or 0 for b in self.data if b.get('volume')]
                overall_avg = sum(all_vols) / len(all_vols) if all_vols else 1
                
                if avg_vol_last > overall_avg * 1.5:
                    if sum_delta_prev5 > 0 and sum_delta_last5 < sum_delta_prev5 * 0.5:
                        exhaustion = 'BUY_EXHAUST'
                    elif sum_delta_prev5 < 0 and sum_delta_last5 > sum_delta_prev5 * 0.5:
                        exhaustion = 'SELL_EXHAUST'
        
        total_vol = buy_volume + sell_volume
        
        return {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_imbalance_count': stacked_buys,
            'sell_imbalance_count': stacked_sells,
            'max_buy_stack': max_buy_stack,
            'max_sell_stack': max_sell_stack,
            'diagonal_count': diagonal_count,
            'exhaustion': exhaustion,
            'buy_ratio': buy_volume / total_vol * 100 if total_vol else 50,
            'dominant_side': 'BUY' if buy_volume > sell_volume else 'SELL'
        }
    
    def session_history(self, num_days: int = 10) -> List[Dict]:
        """
        Get historical session data for the last N trading days.
        
        Args:
            num_days: Number of trading days to analyze
            
        Returns:
            List of session history records
        """
        if not self.data:
            return []
        
        # Group data by date
        days_data = defaultdict(list)
        for bar in self.data:
            dt = bar['datetime']
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                except:
                    continue
            days_data[dt.date()].append(bar)
        
        # Get last N days
        sorted_dates = sorted(days_data.keys(), reverse=True)[:num_days]
        
        history = []
        for date in sorted_dates:
            bars = days_data[date]
            
            # Session breakdown for this day
            sessions = {
                'open': {'start': 10, 'end': 10.5, 'delta': 0, 'vol': 0},
                'core': {'start': 10.5, 'end': 13, 'delta': 0, 'vol': 0},
                'close': {'start': 13, 'end': 14.5, 'delta': 0, 'vol': 0}
            }
            
            for bar in bars:
                dt = bar['datetime']
                if isinstance(dt, str):
                    try:
                        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                    except:
                        continue
                
                hour = dt.hour + dt.minute / 60
                delta = bar.get('delta', 0)
                vol = bar.get('volume', 0) or 0
                
                for session_name, session_info in sessions.items():
                    if session_info['start'] <= hour < session_info['end']:
                        session_info['delta'] += delta
                        session_info['vol'] += vol
                        break
            
            # Calculate totals
            total_delta = sum(s['delta'] for s in sessions.values())
            total_vol = sum(s['vol'] for s in sessions.values())
            
            # Classify pattern
            if sessions['open']['delta'] > 0 and sessions['core']['delta'] > 0:
                pattern = 'Morning Rally'
            elif sessions['open']['delta'] < 0 and total_delta > 0:
                pattern = 'Reversal'
            elif sessions['core']['delta'] < 0 and abs(sessions['core']['delta']) > abs(sessions['open']['delta']):
                pattern = 'Distribution'
            elif sessions['close']['delta'] > sessions['open']['delta']:
                pattern = 'Late Strength'
            elif sessions['close']['delta'] < 0 and total_delta < 0:
                pattern = 'Sell-Off'
            else:
                pattern = 'Mixed'
            
            history.append({
                'date': date,
                'day_name': date.strftime('%a'),
                'open_delta': sessions['open']['delta'],
                'core_delta': sessions['core']['delta'],
                'close_delta': sessions['close']['delta'],
                'total_delta': total_delta,
                'total_volume': total_vol,
                'result': 'WIN' if total_delta > 0 else 'LOSS',
                'pattern': pattern
            })
        
        return history
    
    # =========================================================================
    # TRADE SIGNAL GENERATOR (PHASE 5)
    # =========================================================================
    
    def generate_trade_signals(self) -> List[Dict]:
        """
        Generate automated trade signals based on flow analysis.
        
        Signal Types:
        - Delta Divergence (Accumulation/Distribution)
        - Volume Profile Breakouts (POC, VAH, VAL)
        - Session-Based Triggers (OR breakout, Core momentum)
        
        Returns:
            List of trade signals with entry, target, stop, and confidence
        """
        if not self.data or len(self.data) < 10:
            return []
        
        signals = []
        
        # Get current market context
        current_price = self.data[-1].get('close', 0)
        profile = self.volume_profile(num_levels=15)
        poc = profile.get('poc', current_price) if profile else current_price
        vah = profile.get('vah', current_price * 1.02) if profile else current_price * 1.02
        val = profile.get('val', current_price * 0.98) if profile else current_price * 0.98
        
        # Check each signal type
        divergence_signal = self._detect_divergence_signal(current_price, poc, vah, val)
        if divergence_signal:
            signals.append(divergence_signal)
        
        volume_signal = self._detect_volume_breakout(current_price, poc, vah, val, profile)
        if volume_signal:
            signals.append(volume_signal)
        
        session_signal = self._detect_session_trigger(current_price, poc, vah, val)
        if session_signal:
            signals.append(session_signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return signals
    
    def _detect_divergence_signal(self, price: float, poc: float, vah: float, val: float) -> Optional[Dict]:
        """Detect delta divergence signals (accumulation/distribution)."""
        try:
            divergences = self.delta_divergence(lookback=10)
            if not divergences:
                return None
            
            latest = divergences[-1]
            div_type = latest.get('type', '')
            
            if not div_type:
                return None
            
            # Get supporting data
            zscore_data = self.delta_zscore()
            zscore = zscore_data[-1].get('zscore', 0) if zscore_data else 0
            
            momentum_data = self.delta_momentum()
            momentum = momentum_data[-1].get('delta_momentum', 0) if momentum_data else 0
            
            rvol_info = self.rvol_analysis()
            rvol = rvol_info.get('rvol', 1) if rvol_info else 1
            
            # Calculate confidence
            confidence = self._calculate_signal_confidence(
                divergence_strength=abs(zscore),
                price_vs_profile=(price, poc, vah, val),
                rvol=rvol,
                momentum=momentum
            )
            
            if div_type == 'ACCUMULATION':
                # BUY signal - price down, delta up
                entry = price
                stop = val * 0.995  # Just below VAL
                target = vah  # Target VAH
                risk = entry - stop
                reward = target - entry
                
                return {
                    'signal_type': 'BUY',
                    'pattern': 'ACCUMULATION',
                    'description': 'Delta rising while price falling - Smart money buying',
                    'entry': entry,
                    'target': target,
                    'stop': stop,
                    'risk_reward': reward / risk if risk > 0 else 0,
                    'confidence': confidence,
                    'components': {
                        'divergence': f'ACCUMULATION (Z: {zscore:+.1f}σ)',
                        'volume': f'RVOL {rvol:.1f}x',
                        'momentum': f'Delta Mom: {momentum:+,.0f}'
                    },
                    'context': f'Price at ₦{price:,.2f} with accumulation. Target VAH ₦{vah:,.2f}'
                }
            
            elif div_type == 'DISTRIBUTION':
                # SELL signal - price up, delta down
                entry = price
                stop = vah * 1.005  # Just above VAH
                target = val  # Target VAL
                risk = stop - entry
                reward = entry - target
                
                return {
                    'signal_type': 'SELL',
                    'pattern': 'DISTRIBUTION',
                    'description': 'Delta falling while price rising - Smart money selling',
                    'entry': entry,
                    'target': target,
                    'stop': stop,
                    'risk_reward': reward / risk if risk > 0 else 0,
                    'confidence': confidence,
                    'components': {
                        'divergence': f'DISTRIBUTION (Z: {zscore:+.1f}σ)',
                        'volume': f'RVOL {rvol:.1f}x',
                        'momentum': f'Delta Mom: {momentum:+,.0f}'
                    },
                    'context': f'Price at ₦{price:,.2f} with distribution. Target VAL ₦{val:,.2f}'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting divergence signal: {e}")
            return None
    
    def _detect_volume_breakout(self, price: float, poc: float, vah: float, val: float, profile: Dict) -> Optional[Dict]:
        """Detect volume profile breakout signals."""
        try:
            if not profile:
                return None
            
            # Get recent bars for breakout detection
            recent_bars = self.data[-5:] if len(self.data) >= 5 else self.data
            
            # Check for VAH breakout (bullish)
            if price > vah:
                prev_close = recent_bars[-2].get('close', 0) if len(recent_bars) >= 2 else 0
                if prev_close <= vah:  # Just broke out
                    rvol_info = self.rvol_analysis()
                    rvol = rvol_info.get('rvol', 1) if rvol_info else 1
                    
                    if rvol >= 1.5:  # Volume confirmation
                        range_size = vah - val
                        entry = price
                        target = vah + range_size  # 1R extension
                        stop = poc  # Stop at POC
                        risk = entry - stop
                        reward = target - entry
                        
                        confidence = min(85, 50 + (rvol * 10) + ((price - vah) / vah * 100))
                        
                        return {
                            'signal_type': 'BUY',
                            'pattern': 'VAH_BREAKOUT',
                            'description': 'Price broke above Value Area High with volume',
                            'entry': entry,
                            'target': target,
                            'stop': stop,
                            'risk_reward': reward / risk if risk > 0 else 0,
                            'confidence': confidence,
                            'components': {
                                'divergence': 'N/A',
                                'volume': f'VAH Break (RVOL {rvol:.1f}x)',
                                'profile': f'Above ₦{vah:,.2f}'
                            },
                            'context': f'Bullish breakout above VAH. Target +1R at ₦{target:,.2f}'
                        }
            
            # Check for VAL breakdown (bearish)
            if price < val:
                prev_close = recent_bars[-2].get('close', 0) if len(recent_bars) >= 2 else 0
                if prev_close >= val:  # Just broke down
                    rvol_info = self.rvol_analysis()
                    rvol = rvol_info.get('rvol', 1) if rvol_info else 1
                    
                    if rvol >= 1.5:
                        range_size = vah - val
                        entry = price
                        target = val - range_size  # 1R extension
                        stop = poc
                        risk = stop - entry
                        reward = entry - target
                        
                        confidence = min(85, 50 + (rvol * 10) + ((val - price) / val * 100))
                        
                        return {
                            'signal_type': 'SELL',
                            'pattern': 'VAL_BREAKDOWN',
                            'description': 'Price broke below Value Area Low with volume',
                            'entry': entry,
                            'target': target,
                            'stop': stop,
                            'risk_reward': reward / risk if risk > 0 else 0,
                            'confidence': confidence,
                            'components': {
                                'divergence': 'N/A',
                                'volume': f'VAL Break (RVOL {rvol:.1f}x)',
                                'profile': f'Below ₦{val:,.2f}'
                            },
                            'context': f'Bearish breakdown below VAL. Target -1R at ₦{target:,.2f}'
                        }
            
            # Check for POC reclaim
            if len(recent_bars) >= 3:
                prices = [b.get('close', 0) for b in recent_bars[-3:]]
                if prices[0] < poc < prices[-1]:  # Reclaimed POC from below
                    cum_delta = self.cumulative_delta()
                    if cum_delta and cum_delta[-1].get('cumulative_delta', 0) > 0:
                        entry = price
                        target = vah
                        stop = val
                        risk = entry - stop
                        reward = target - entry
                        
                        return {
                            'signal_type': 'BUY',
                            'pattern': 'POC_RECLAIM',
                            'description': 'Price reclaimed POC from below with positive delta',
                            'entry': entry,
                            'target': target,
                            'stop': stop,
                            'risk_reward': reward / risk if risk > 0 else 0,
                            'confidence': 65,
                            'components': {
                                'divergence': 'Delta Positive',
                                'volume': 'POC Reclaimed',
                                'profile': f'Above ₦{poc:,.2f}'
                            },
                            'context': f'Bullish POC reclaim. Target VAH ₦{vah:,.2f}'
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting volume breakout: {e}")
            return None
    
    def _detect_session_trigger(self, price: float, poc: float, vah: float, val: float) -> Optional[Dict]:
        """Detect session-based trade triggers."""
        try:
            # Get opening range data
            or_data = self.opening_range_analysis()
            if not or_data:
                return None
            
            or_high = or_data.get('or_high', 0)
            or_low = or_data.get('or_low', 0)
            breakout = or_data.get('breakout', 'NO_BREAKOUT')
            
            # Get session breakdown
            sessions = self.intraday_session_breakdown()
            core_session = sessions.get('core', {})
            core_delta = core_session.get('delta', 0)
            core_trend = core_session.get('trend', 'NEUTRAL')
            
            # Opening Range Breakout
            if breakout == 'BULLISH_BREAKOUT':
                or_range = or_high - or_low
                if or_range > 0:
                    entry = price
                    target = or_high + (or_range * 2)  # 2R target
                    stop = or_low
                    risk = entry - stop
                    reward = target - entry
                    
                    # Higher confidence if core session confirms
                    conf = 60
                    if core_delta > 0:
                        conf += 15
                    if core_trend == 'BULLISH':
                        conf += 10
                    
                    return {
                        'signal_type': 'BUY',
                        'pattern': 'OR_BREAKOUT',
                        'description': 'Opening Range breakout to upside',
                        'entry': entry,
                        'target': target,
                        'stop': stop,
                        'risk_reward': reward / risk if risk > 0 else 0,
                        'confidence': min(90, conf),
                        'components': {
                            'divergence': 'N/A',
                            'volume': 'OR Break Up',
                            'session': f'Core: {core_trend}'
                        },
                        'context': f'OR high ₦{or_high:,.2f} broken. Target +2R ₦{target:,.2f}'
                    }
            
            elif breakout == 'BEARISH_BREAKDOWN':
                or_range = or_high - or_low
                if or_range > 0:
                    entry = price
                    target = or_low - (or_range * 2)  # 2R target
                    stop = or_high
                    risk = stop - entry
                    reward = entry - target
                    
                    conf = 60
                    if core_delta < 0:
                        conf += 15
                    if core_trend == 'BEARISH':
                        conf += 10
                    
                    return {
                        'signal_type': 'SELL',
                        'pattern': 'OR_BREAKDOWN',
                        'description': 'Opening Range breakdown to downside',
                        'entry': entry,
                        'target': target,
                        'stop': stop,
                        'risk_reward': reward / risk if risk > 0 else 0,
                        'confidence': min(90, conf),
                        'components': {
                            'divergence': 'N/A',
                            'volume': 'OR Break Down',
                            'session': f'Core: {core_trend}'
                        },
                        'context': f'OR low ₦{or_low:,.2f} broken. Target -2R ₦{target:,.2f}'
                    }
            
            # Core session momentum signal
            if abs(core_delta) > 0:
                zscore_data = self.delta_zscore()
                zscore = zscore_data[-1].get('zscore', 0) if zscore_data else 0
                
                if zscore > 2 and core_delta > 0 and core_trend in ['BULLISH', 'ACCUMULATION']:
                    entry = price
                    target = vah
                    stop = poc
                    risk = entry - stop
                    reward = target - entry
                    
                    return {
                        'signal_type': 'BUY',
                        'pattern': 'CORE_MOMENTUM',
                        'description': 'Strong bullish momentum in core session',
                        'entry': entry,
                        'target': target,
                        'stop': stop,
                        'risk_reward': reward / risk if risk > 0 else 0,
                        'confidence': min(80, 55 + abs(zscore) * 10),
                        'components': {
                            'divergence': f'Z-Score +{zscore:.1f}σ',
                            'volume': f'Core Δ: {core_delta:+,.0f}',
                            'session': core_trend
                        },
                        'context': f'Core session bullish with extreme Z-score'
                    }
                
                elif zscore < -2 and core_delta < 0 and core_trend in ['BEARISH', 'DISTRIBUTION']:
                    entry = price
                    target = val
                    stop = poc
                    risk = stop - entry
                    reward = entry - target
                    
                    return {
                        'signal_type': 'SELL',
                        'pattern': 'CORE_MOMENTUM',
                        'description': 'Strong bearish momentum in core session',
                        'entry': entry,
                        'target': target,
                        'stop': stop,
                        'risk_reward': reward / risk if risk > 0 else 0,
                        'confidence': min(80, 55 + abs(zscore) * 10),
                        'components': {
                            'divergence': f'Z-Score {zscore:.1f}σ',
                            'volume': f'Core Δ: {core_delta:+,.0f}',
                            'session': core_trend
                        },
                        'context': f'Core session bearish with extreme Z-score'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting session trigger: {e}")
            return None
    
    def _calculate_signal_confidence(self, divergence_strength: float, price_vs_profile: tuple, 
                                     rvol: float, momentum: float) -> float:
        """
        Calculate signal confidence score (0-100).
        
        Scoring:
        - Divergence strength (0-30 points)
        - Volume profile position (0-25 points)
        - RVOL confirmation (0-15 points)
        - Momentum alignment (0-15 points)
        - Base score (15 points)
        """
        score = 15  # Base
        
        # Divergence strength (Z-score based)
        if divergence_strength >= 3:
            score += 30
        elif divergence_strength >= 2:
            score += 25
        elif divergence_strength >= 1.5:
            score += 20
        elif divergence_strength >= 1:
            score += 15
        else:
            score += divergence_strength * 10
        
        # Volume profile position
        price, poc, vah, val = price_vs_profile
        value_range = vah - val if vah > val else 1
        
        # Near VAL (buy) or VAH (sell) is better
        dist_to_val = abs(price - val) / value_range
        dist_to_vah = abs(price - vah) / value_range
        
        if min(dist_to_val, dist_to_vah) < 0.2:
            score += 25
        elif min(dist_to_val, dist_to_vah) < 0.5:
            score += 15
        else:
            score += 10
        
        # RVOL confirmation
        if rvol >= 3:
            score += 15
        elif rvol >= 2:
            score += 12
        elif rvol >= 1.5:
            score += 8
        else:
            score += 5
        
        # Momentum alignment
        if abs(momentum) > 1000:
            score += 15
        elif abs(momentum) > 500:
            score += 10
        else:
            score += 5
        
        return min(100, max(0, score))
    
    # =========================================================================
    # SESSION ANALYTICS (PHASE 4)
    # =========================================================================
    
    def intraday_session_breakdown(self) -> Dict:
        """
        Break down trading activity by intraday session.
        NGX sessions: Open (10:00-10:30), Core (10:30-13:00), Close (13:00-14:30)
        
        Returns:
            Dict with session-level metrics
        """
        sessions = {
            'open': {'start': 10, 'end': 10.5, 'data': [], 'label': 'Opening (10:00-10:30)'},
            'core': {'start': 10.5, 'end': 13, 'data': [], 'label': 'Core (10:30-13:00)'},
            'close': {'start': 13, 'end': 14.5, 'data': [], 'label': 'Closing (13:00-14:30)'}
        }
        
        for bar in self.data:
            dt = bar['datetime']
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                except:
                    continue
            
            hour = dt.hour + dt.minute / 60
            
            for session_name, session_info in sessions.items():
                if session_info['start'] <= hour < session_info['end']:
                    session_info['data'].append(bar)
                    break
        
        result = {}
        for session_name, session_info in sessions.items():
            bars = session_info['data']
            if not bars:
                result[session_name] = {
                    'label': session_info['label'],
                    'bars': 0,
                    'volume': 0,
                    'delta': 0,
                    'avg_rvol': 0,
                    'trend': 'NO_DATA'
                }
                continue
            
            total_vol = sum(b.get('volume', 0) or 0 for b in bars)
            total_delta = sum(b.get('delta', 0) for b in bars)
            avg_vol = statistics.mean([b.get('volume', 0) or 0 for b in bars]) if bars else 0
            
            # Price movement
            first_open = bars[0].get('open', 0) or 0
            last_close = bars[-1].get('close', 0) or 0
            price_change = ((last_close - first_open) / first_open * 100) if first_open else 0
            
            if total_delta > 0 and price_change > 0:
                trend = 'BULLISH'
            elif total_delta < 0 and price_change < 0:
                trend = 'BEARISH'
            elif total_delta > 0 and price_change < 0:
                trend = 'ACCUMULATION'
            elif total_delta < 0 and price_change > 0:
                trend = 'DISTRIBUTION'
            else:
                trend = 'NEUTRAL'
            
            result[session_name] = {
                'label': session_info['label'],
                'bars': len(bars),
                'volume': total_vol,
                'delta': total_delta,
                'avg_volume': avg_vol,
                'price_change': price_change,
                'trend': trend
            }
        
        return result
    
    def time_of_day_patterns(self) -> Dict:
        """
        Analyze trading patterns by time of day.
        
        Returns:
            Dict with hourly analysis
        """
        hourly_data = defaultdict(lambda: {'volume': [], 'delta': [], 'count': 0})
        
        for bar in self.data:
            dt = bar['datetime']
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                except:
                    continue
            
            hour = dt.hour
            hourly_data[hour]['volume'].append(bar.get('volume', 0) or 0)
            hourly_data[hour]['delta'].append(bar.get('delta', 0))
            hourly_data[hour]['count'] += 1
        
        result = {}
        for hour in sorted(hourly_data.keys()):
            data = hourly_data[hour]
            if data['count'] == 0:
                continue
            
            avg_vol = statistics.mean(data['volume']) if data['volume'] else 0
            total_delta = sum(data['delta'])
            
            # Determine bias
            buy_bars = sum(1 for d in data['delta'] if d > 0)
            sell_bars = sum(1 for d in data['delta'] if d < 0)
            
            if buy_bars > sell_bars * 1.5:
                bias = 'STRONG_BUY'
            elif buy_bars > sell_bars:
                bias = 'BUY'
            elif sell_bars > buy_bars * 1.5:
                bias = 'STRONG_SELL'
            elif sell_bars > buy_bars:
                bias = 'SELL'
            else:
                bias = 'NEUTRAL'
            
            result[hour] = {
                'hour_label': f"{hour:02d}:00",
                'avg_volume': avg_vol,
                'total_delta': total_delta,
                'bar_count': data['count'],
                'buy_bars': buy_bars,
                'sell_bars': sell_bars,
                'bias': bias
            }
        
        return result
    
    def opening_range_analysis(self, range_minutes: int = 30) -> Dict:
        """
        Analyze the opening range (first N minutes).
        
        Args:
            range_minutes: Minutes to consider as opening range
            
        Returns:
            Opening range metrics
        """
        if not self.data:
            return {}
        
        # Get today's data
        today_data = []
        latest_date = None
        
        for bar in reversed(self.data):
            dt = bar['datetime']
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                except:
                    continue
            
            if latest_date is None:
                latest_date = dt.date()
            
            if dt.date() == latest_date:
                today_data.append(bar)
            else:
                break
        
        today_data = list(reversed(today_data))
        
        if not today_data:
            return {}
        
        # Get opening range bars
        opening_bars = []
        first_dt = None
        
        for bar in today_data:
            dt = bar['datetime']
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                except:
                    continue
            
            if first_dt is None:
                first_dt = dt
            
            if (dt - first_dt).total_seconds() <= range_minutes * 60:
                opening_bars.append(bar)
        
        if not opening_bars:
            return {}
        
        # Calculate opening range
        or_high = max(b.get('high', 0) or 0 for b in opening_bars)
        or_low = min(b.get('low', 0) or b.get('close', 0) or 0 for b in opening_bars)
        or_volume = sum(b.get('volume', 0) or 0 for b in opening_bars)
        or_delta = sum(b.get('delta', 0) for b in opening_bars)
        
        # Current price
        current_price = today_data[-1].get('close', 0) if today_data else 0
        
        # Position relative to OR
        if current_price > or_high:
            position = 'ABOVE_OR'
            breakout = 'BULLISH_BREAKOUT'
        elif current_price < or_low:
            position = 'BELOW_OR'
            breakout = 'BEARISH_BREAKDOWN'
        else:
            position = 'INSIDE_OR'
            breakout = 'NO_BREAKOUT'
        
        return {
            'or_high': or_high,
            'or_low': or_low,
            'or_range': or_high - or_low,
            'or_volume': or_volume,
            'or_delta': or_delta,
            'current_price': current_price,
            'position': position,
            'breakout': breakout,
            'bars_in_or': len(opening_bars)
        }
    
    def session_comparison(self) -> Dict:
        """
        Compare today's session to recent sessions.
        
        Returns:
            Comparison metrics
        """
        session_data = self.session_delta()
        
        if len(session_data) < 2:
            return {}
        
        dates = sorted(session_data.keys())
        today = dates[-1]
        today_delta = session_data[today]
        
        # Historical stats
        historical_deltas = [session_data[d] for d in dates[:-1]]
        
        if not historical_deltas:
            return {}
        
        avg_delta = statistics.mean(historical_deltas)
        std_delta = statistics.stdev(historical_deltas) if len(historical_deltas) > 1 else 0
        
        # Z-score
        zscore = (today_delta - avg_delta) / std_delta if std_delta > 0 else 0
        
        # Percentile
        count_below = sum(1 for d in historical_deltas if d < today_delta)
        percentile = (count_below / len(historical_deltas)) * 100
        
        # Streak
        streak = 0
        streak_type = 'BUY' if today_delta > 0 else 'SELL'
        for d in reversed(dates):
            if (session_data[d] > 0) == (today_delta > 0):
                streak += 1
            else:
                break
        
        return {
            'today_delta': today_delta,
            'avg_delta': avg_delta,
            'zscore': zscore,
            'percentile': percentile,
            'streak': streak,
            'streak_type': streak_type,
            'sessions_analyzed': len(dates),
            'today_vs_avg': 'ABOVE' if today_delta > avg_delta else 'BELOW'
        }
    
    def get_all_statistics(self) -> Dict:
        """
        Get comprehensive statistics for display.
        
        Returns:
            Dict with all statistical metrics
        """
        return {
            'summary': self.summary_stats(),
            'rvol': self.rvol_percentile(),
            'trade_frequency': self.trade_frequency(),
            'volume_acceleration': self.volume_acceleration(),
            'price_efficiency': self.price_efficiency(),
            'trade_size_distribution': self.trade_size_distribution(),
            'intraday_sessions': self.intraday_session_breakdown(),
            'time_patterns': self.time_of_day_patterns(),
            'opening_range': self.opening_range_analysis(),
            'session_comparison': self.session_comparison(),
            'alerts': self.generate_alerts()
        }

