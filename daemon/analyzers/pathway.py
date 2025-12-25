# Pathway Analyzer for Daemon
# Headless version of PathwaySynthesizer

import logging
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent for src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config
from data.fetcher import DataFetcher
from data.db import get_db_connection

logger = logging.getLogger(__name__)


class PathwayAnalyzer:
    """Headless pathway synthesis for daemon."""
    
    HORIZONS = {
        '2d': 2,
        '3d': 3,
        '1w': 5,
        '1m': 22
    }
    
    WEIGHTS = {
        'ml': 0.25,
        'flow': 0.20,
        'sector': 0.15,
        'fundamentals': 0.15,
        'technicals': 0.15,
        'disclosures': 0.10
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.fetcher = DataFetcher(config)
        
    async def synthesize(self, symbol: str) -> Dict:
        """Generate pathway synthesis for a symbol."""
        # Fetch latest data
        df = await self.fetcher.get_ohlcv(symbol)
        if df is None or df.empty:
            return {'error': f'No data for {symbol}'}
        
        current_price = float(df['close'].iloc[-1])
        
        # Gather signals
        signals = await self._gather_signals(symbol, df)
        
        # Generate predictions
        predictions = {}
        for horizon_name, days in self.HORIZONS.items():
            predictions[horizon_name] = self._predict_horizon(
                symbol, current_price, signals, horizon_name, days
            )
        
        # Bid/offer probability
        bid_offer = self._calculate_bid_offer(signals)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predictions': predictions,
            'bid_offer': bid_offer,
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _gather_signals(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Gather all signals for a symbol."""
        signals = {}
        
        # Technical signals
        signals['technicals'] = self._get_technical_signals(df)
        
        # Flow signals
        signals['flow'] = self._get_flow_signals(df)
        
        # ML signals (simplified for daemon)
        signals['ml'] = {'direction': 0, 'confidence': 0.5}
        
        # Sector signals
        signals['sector'] = {'momentum': 0}
        
        # Fundamentals
        signals['fundamentals'] = {'pe_signal': 0}
        
        # Disclosures
        signals['disclosures'] = {'impact': 0}
        
        return signals
    
    def _get_technical_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate technical signals."""
        if len(df) < 20:
            return {'rsi': 0, 'trend': 0}
        
        closes = df['close'].values
        
        # RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100
        
        # RSI signal
        if rsi < 30:
            rsi_signal = 0.8
        elif rsi > 70:
            rsi_signal = -0.8
        else:
            rsi_signal = 0
        
        # Trend
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:])
        trend = np.clip((sma_10 - sma_20) / sma_20 * 20, -1, 1)
        
        return {
            'rsi': float(rsi),
            'rsi_signal': float(rsi_signal),
            'trend': float(trend)
        }
    
    def _get_flow_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate flow signals."""
        if len(df) < 5:
            return {'delta': 0, 'vwap_position': 0}
        
        # Simple delta approximation
        recent = df.tail(5)
        up_vol = recent[recent['close'] >= recent['open']]['volume'].sum()
        down_vol = recent[recent['close'] < recent['open']]['volume'].sum()
        total = up_vol + down_vol
        
        delta = (up_vol - down_vol) / total if total > 0 else 0
        
        # VWAP position
        vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
        current = df['close'].iloc[-1]
        vwap_pos = (current - vwap) / vwap if vwap > 0 else 0
        
        return {
            'delta': float(delta),
            'vwap_position': float(np.clip(vwap_pos * 10, -1, 1))
        }
    
    def _predict_horizon(self, symbol: str, price: float, signals: Dict, 
                         name: str, days: int) -> Dict:
        """Generate prediction for a horizon."""
        # Calculate weighted signal
        tech = signals.get('technicals', {})
        flow = signals.get('flow', {})
        
        signal_score = (
            tech.get('trend', 0) * self.WEIGHTS['technicals'] +
            flow.get('delta', 0) * self.WEIGHTS['flow']
        )
        
        # Volatility estimate (5% default)
        volatility = 0.05
        
        # Expected return
        base_return = signal_score * volatility * (days / 5)
        base_return = np.clip(base_return, -0.30, 0.30)
        
        expected_price = price * (1 + base_return)
        
        # Scenarios
        bull_return = base_return + volatility * 0.8
        bear_return = base_return - volatility * 0.8
        
        # Probabilities
        if signal_score > 0.2:
            bull_prob, base_prob, bear_prob = 0.45, 0.35, 0.20
        elif signal_score < -0.2:
            bull_prob, base_prob, bear_prob = 0.20, 0.35, 0.45
        else:
            bull_prob, base_prob, bear_prob = 0.30, 0.40, 0.30
        
        return {
            'expected_price': round(expected_price, 2),
            'expected_return': round(base_return * 100, 2),
            'bull': {
                'price': round(price * (1 + bull_return), 2),
                'probability': round(bull_prob * 100, 1)
            },
            'base': {
                'price': round(expected_price, 2),
                'probability': round(base_prob * 100, 1)
            },
            'bear': {
                'price': round(price * (1 + bear_return), 2),
                'probability': round(bear_prob * 100, 1)
            }
        }
    
    def _calculate_bid_offer(self, signals: Dict) -> Dict:
        """Calculate bid/offer probability."""
        flow = signals.get('flow', {})
        delta = flow.get('delta', 0)
        
        if delta > 0.3:
            return {'full_bid': 60, 'mixed': 30, 'full_offer': 10}
        elif delta < -0.3:
            return {'full_bid': 10, 'mixed': 30, 'full_offer': 60}
        else:
            return {'full_bid': 30, 'mixed': 40, 'full_offer': 30}


async def generate_pathway_alert(config: Config, symbol: str) -> str:
    """Generate formatted pathway alert for Telegram."""
    analyzer = PathwayAnalyzer(config)
    result = await analyzer.synthesize(symbol)
    
    if 'error' in result:
        return f"❌ {result['error']}"
    
    price = result['current_price']
    preds = result['predictions']
    
    lines = [
        f"🔮 <b>PATHWAY: {symbol}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Current: ₦{price:,.2f}",
        ""
    ]
    
    for horizon, data in preds.items():
        ret = data['expected_return']
        bull_prob = data['bull']['probability']
        sign = '+' if ret >= 0 else ''
        lines.append(f"<b>{horizon.upper()}</b>: {sign}{ret}% ({bull_prob}% Bull)")
    
    bidoffer = result['bid_offer']
    lines.extend([
        "",
        f"📊 Bid: {bidoffer['full_bid']}% | Mix: {bidoffer['mixed']}% | Offer: {bidoffer['full_offer']}%"
    ])
    
    return "\n".join(lines)


async def generate_watchlist_pathways(config: Config) -> str:
    """Generate pathway summary for all watchlist symbols."""
    analyzer = PathwayAnalyzer(config)
    
    lines = [
        "🔮 <b>MIDDAY PATHWAY SYNTHESIS</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]
    
    for symbol in config.default_watchlist[:10]:  # Top 10
        try:
            result = await analyzer.synthesize(symbol)
            if 'error' not in result:
                ret_2d = result['predictions']['2d']['expected_return']
                bull_prob = result['predictions']['2d']['bull']['probability']
                sign = '+' if ret_2d >= 0 else ''
                emoji = '🟢' if ret_2d > 0 else '🔴' if ret_2d < 0 else '⚪'
                lines.append(f"{emoji} <b>{symbol}</b>: {sign}{ret_2d}% (2D) - {bull_prob}% Bull")
        except Exception as e:
            logger.error(f"Pathway error for {symbol}: {e}")
    
    return "\n".join(lines)


async def generate_close_signals(config: Config) -> str:
    """Generate close signals for watchlist."""
    analyzer = PathwayAnalyzer(config)
    
    lines = [
        "⏰ <b>PRE-CLOSE SIGNALS</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]
    
    bullish = []
    bearish = []
    
    for symbol in config.default_watchlist[:10]:
        try:
            result = await analyzer.synthesize(symbol)
            if 'error' not in result:
                bidoffer = result['bid_offer']
                if bidoffer['full_bid'] > 50:
                    bullish.append(symbol)
                elif bidoffer['full_offer'] > 50:
                    bearish.append(symbol)
        except:
            pass
    
    if bullish:
        lines.append(f"🟢 <b>Likely Full Bid:</b> {', '.join(bullish)}")
    if bearish:
        lines.append(f"🔴 <b>Likely Full Offer:</b> {', '.join(bearish)}")
    
    if not bullish and not bearish:
        lines.append("⚪ No strong close signals detected")
    
    return "\n".join(lines)
