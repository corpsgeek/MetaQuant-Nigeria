# Flow Analyzer for Daemon

import logging
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

from config import Config
from data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class FlowAnalyzer:
    """Order flow analysis for daemon."""
    
    def __init__(self, config: Config):
        self.config = config
        self.fetcher = DataFetcher(config)
        
    async def analyze(self, symbol: str) -> Dict:
        """Analyze order flow for a symbol."""
        df = await self.fetcher.get_ohlcv(symbol)
        if df is None or df.empty:
            return {'error': f'No data for {symbol}'}
        
        # Calculate flow metrics
        delta = self._calculate_delta(df)
        vwap = self._calculate_vwap(df)
        volume_profile = self._analyze_volume(df)
        
        current_price = float(df['close'].iloc[-1])
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'delta': delta,
            'vwap': vwap,
            'vwap_position': 'above' if current_price > vwap else 'below',
            'volume_profile': volume_profile,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_delta(self, df: pd.DataFrame) -> Dict:
        """Calculate cumulative delta."""
        # Simplified: up candles = buying, down candles = selling
        df = df.copy()
        df['direction'] = np.where(df['close'] >= df['open'], 1, -1)
        df['delta'] = df['direction'] * df['volume']
        
        cum_delta = df['delta'].sum()
        recent_delta = df.tail(10)['delta'].sum()
        
        # Normalize
        avg_volume = df['volume'].mean()
        normalized = recent_delta / (avg_volume * 10) if avg_volume > 0 else 0
        
        return {
            'cumulative': float(cum_delta),
            'recent': float(recent_delta),
            'normalized': float(np.clip(normalized, -1, 1)),
            'signal': 'bullish' if normalized > 0.3 else 'bearish' if normalized < -0.3 else 'neutral'
        }
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate VWAP."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return float(vwap)
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns."""
        recent = df.tail(5)['volume'].mean()
        historical = df['volume'].mean()
        
        ratio = recent / historical if historical > 0 else 1
        
        return {
            'recent_avg': float(recent),
            'historical_avg': float(historical),
            'ratio': float(ratio),
            'signal': 'high' if ratio > 1.5 else 'low' if ratio < 0.7 else 'normal'
        }


async def generate_flow_alert(config: Config, symbol: str) -> str:
    """Generate formatted flow alert for Telegram."""
    analyzer = FlowAnalyzer(config)
    result = await analyzer.analyze(symbol)
    
    if 'error' in result:
        return f"❌ {result['error']}"
    
    delta = result['delta']
    volume = result['volume_profile']
    
    # Emoji based on signal
    delta_emoji = '🟢' if delta['signal'] == 'bullish' else '🔴' if delta['signal'] == 'bearish' else '⚪'
    vol_emoji = '📈' if volume['signal'] == 'high' else '📉' if volume['signal'] == 'low' else '📊'
    
    return f"""
📊 <b>FLOW: {symbol}</b>
━━━━━━━━━━━━━━━━━━━━
Price: ₦{result['current_price']:,.2f}
VWAP: ₦{result['vwap']:,.2f} ({result['vwap_position']})

<b>Delta Analysis</b>
{delta_emoji} Signal: {delta['signal'].upper()}
Normalized: {delta['normalized']:.2f}

<b>Volume</b>
{vol_emoji} Recent/Historical: {volume['ratio']:.2f}x
    """.strip()


async def scan_all_flow(config: Config) -> List[str]:
    """Scan all watchlist for flow alerts."""
    analyzer = FlowAnalyzer(config)
    alerts = []
    
    for symbol in config.default_watchlist:
        try:
            result = await analyzer.analyze(symbol)
            if 'error' not in result:
                delta = result['delta']
                volume = result['volume_profile']
                
                # Alert on strong signals only
                if abs(delta['normalized']) > config.flow_delta_threshold / 2:
                    signal = delta['signal'].upper()
                    alerts.append(f"📊 <b>{symbol}</b>: Strong {signal} flow ({delta['normalized']:.2f})")
                
                if volume['ratio'] > 2.0:
                    alerts.append(f"📈 <b>{symbol}</b>: Volume spike ({volume['ratio']:.1f}x avg)")
                    
        except Exception as e:
            logger.error(f"Flow scan error for {symbol}: {e}")
    
    return alerts
