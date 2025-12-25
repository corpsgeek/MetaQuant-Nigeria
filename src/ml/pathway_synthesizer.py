"""
Pandora Black Box - Price Pathway Synthesizer.

Synthesizes ALL available data sources to generate multi-horizon
price pathway predictions with probability distributions.

Data Sources:
- ML Engine (XGBoost, LSTM, Ensemble)
- Flow Tape (Delta, VWAP, Volume Profile, Blocks)
- Market Intelligence (Sector Rotation, Smart Money)
- Fundamentals (P/E, EPS, Dividends)
- Corporate Disclosures (AI Impact Scores)
- Technical Analysis (RSI, MACD, Patterns)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PathwayScenario:
    """Price scenario with probability."""
    price: float
    return_pct: float
    probability: float


@dataclass
class HorizonPrediction:
    """Prediction for a single time horizon."""
    horizon_name: str
    horizon_days: int
    base_price: float
    expected_price: float
    expected_return: float
    bull_scenario: PathwayScenario
    base_scenario: PathwayScenario
    bear_scenario: PathwayScenario
    confidence: float


@dataclass
class BidOfferProbability:
    """Session close probability distribution."""
    full_bid_prob: float
    mixed_prob: float
    full_offer_prob: float


class PathwaySynthesizer:
    """
    Pandora Black Box - Synthesizes ALL data for price pathways.
    
    Combines signals from:
    - ML predictions
    - Order flow analysis
    - Sector rotation
    - Fundamentals
    - Disclosures
    - Technical indicators
    """
    
    # Weighting for different signal sources
    WEIGHTS = {
        'ml_ensemble': 0.25,
        'flow_delta': 0.20,
        'sector_momentum': 0.15,
        'fundamentals': 0.15,
        'technicals': 0.15,
        'disclosures': 0.10,
    }
    
    # Time horizons in days
    HORIZONS = {
        '2D': 2,
        '3D': 3,
        '1W': 7,
        '1M': 30,
    }
    
    def __init__(self, db):
        """Initialize with database connection."""
        self.db = db
        self._ml_engine = None
        self._cache = {}
        self._last_update = {}
    
    @property
    def ml_engine(self):
        """Lazy load ML engine."""
        if self._ml_engine is None:
            try:
                from src.ml import MLEngine
                self._ml_engine = MLEngine(db=self.db)  # Pass db as keyword arg
            except Exception as e:
                logger.warning(f"ML Engine not available: {e}")
        return self._ml_engine
    
    def synthesize(self, symbol: str) -> Dict:
        """
        Generate comprehensive price pathway synthesis.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with predictions, probabilities, and signal breakdown
        """
        logger.info(f"Synthesizing pathway for {symbol}...")
        
        # Get current price
        current_price = self._get_current_price(symbol)
        logger.info(f"Current price for {symbol}: {current_price}")
        
        if not current_price or current_price <= 0:
            logger.warning(f"Invalid current price for {symbol}: {current_price}")
            return {'error': f'Could not get valid current price for {symbol}'}
        
        # Gather all signals
        signals = self._gather_all_signals(symbol)
        logger.info(f"Signals for {symbol}: ML={signals.get('ml', {}).get('direction')}, Flow={signals.get('flow', {}).get('delta_direction')}")
        
        # Generate predictions for each horizon
        predictions = {}
        for horizon_name, horizon_days in self.HORIZONS.items():
            predictions[horizon_name] = self._generate_horizon_prediction(
                symbol, current_price, signals, horizon_name, horizon_days
            )
        
        # Calculate bid/offer probability
        bid_offer = self._calculate_bid_offer_probability(symbol, signals)
        
        # Generate AI narrative
        narrative = self._generate_narrative(symbol, signals, predictions)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'bid_offer': bid_offer,
            'signals': signals,
            'narrative': narrative,
            'confidence': self._calculate_overall_confidence(signals)
        }
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for symbol."""
        try:
            result = self.db.conn.execute("""
                SELECT close FROM intraday_ohlcv
                WHERE symbol = ?
                ORDER BY datetime DESC LIMIT 1
            """, [symbol]).fetchone()
            
            if result:
                return float(result[0])
            
            # Try daily prices
            result = self.db.conn.execute("""
                SELECT close FROM daily_prices
                WHERE stock_id = (SELECT id FROM stocks WHERE symbol = ?)
                ORDER BY date DESC LIMIT 1
            """, [symbol]).fetchone()
            
            return float(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def _gather_all_signals(self, symbol: str) -> Dict:
        """Gather signals from all data sources."""
        signals = {
            'ml': self._get_ml_signals(symbol),
            'flow': self._get_flow_signals(symbol),
            'sector': self._get_sector_signals(symbol),
            'fundamental': self._get_fundamental_signals(symbol),
            'disclosure': self._get_disclosure_signals(symbol),
            'technical': self._get_technical_signals(symbol),
        }
        return signals
    
    def _get_ml_signals(self, symbol: str) -> Dict:
        """Get signals from ML Engine (XGBoost, LSTM, Ensemble)."""
        signals = {
            'direction': 0,  # -1 to 1
            'confidence': 0.5,
            'source': 'ml_ensemble'
        }
        
        try:
            if self.ml_engine and self.ml_engine.predictor:
                # Get OHLCV data for prediction
                import pandas as pd
                result = self.db.conn.execute("""
                    SELECT datetime, open, high, low, close, volume
                    FROM intraday_ohlcv
                    WHERE symbol = ?
                    ORDER BY datetime DESC
                    LIMIT 100
                """, [symbol]).fetchall()
                
                if result and len(result) >= 20:
                    df = pd.DataFrame(result, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                    df = df.sort_values('datetime').reset_index(drop=True)
                    
                    prediction = self.ml_engine.predictor.predict(df, symbol)
                    if prediction and prediction.get('success'):
                        # direction_code is -1, 0, or 1
                        signals['direction'] = prediction.get('direction_code', 0)
                        signals['confidence'] = prediction.get('confidence', 50) / 100
                        signals['prediction'] = prediction
        except Exception as e:
            logger.warning(f"ML signal error for {symbol}: {e}")
        
        return signals
    
    def _get_flow_signals(self, symbol: str) -> Dict:
        """Get order flow signals (delta, VWAP, volume profile)."""
        signals = {
            'delta_direction': 0,  # -1 to 1
            'vwap_position': 0,    # price vs VWAP
            'volume_strength': 0.5,
            'block_imbalance': 0,
            'source': 'flow_delta'
        }
        
        try:
            # Get recent intraday data
            result = self.db.conn.execute("""
                SELECT open, high, low, close, volume
                FROM intraday_ohlcv
                WHERE symbol = ?
                ORDER BY datetime DESC
                LIMIT 50
            """, [symbol]).fetchall()
            
            if result:
                closes = [r[3] for r in result]
                volumes = [r[4] for r in result]
                
                # Calculate delta direction from price momentum
                if len(closes) >= 10:
                    recent_avg = np.mean(closes[:5])
                    older_avg = np.mean(closes[5:10])
                    signals['delta_direction'] = np.clip(
                        (recent_avg - older_avg) / older_avg * 10, -1, 1
                    ) if older_avg > 0 else 0
                
                # Volume strength
                if len(volumes) >= 20:
                    recent_vol = np.mean(volumes[:5])
                    avg_vol = np.mean(volumes[:20])
                    signals['volume_strength'] = min(recent_vol / avg_vol, 2) / 2 if avg_vol > 0 else 0.5
                
                # VWAP position estimate
                if len(closes) >= 5:
                    vwap_est = np.average(closes[:20], weights=volumes[:20] if len(volumes) >= 20 else None)
                    current = closes[0]
                    signals['vwap_position'] = np.clip((current - vwap_est) / vwap_est * 20, -1, 1)
                    
        except Exception as e:
            logger.warning(f"Flow signal error for {symbol}: {e}")
        
        return signals
    
    def _get_sector_signals(self, symbol: str) -> Dict:
        """Get sector rotation and relative strength signals."""
        signals = {
            'sector_momentum': 0,
            'relative_strength': 0,
            'rotation_phase': 'neutral',
            'source': 'sector_momentum'
        }
        
        try:
            # Get stock's sector
            result = self.db.conn.execute("""
                SELECT sector FROM stocks WHERE symbol = ?
            """, [symbol]).fetchone()
            
            if result and result[0]:
                sector = result[0]
                
                # Get sector stocks and their recent price changes from intraday data
                sector_stocks = self.db.conn.execute("""
                    SELECT s.symbol
                    FROM stocks s
                    WHERE s.sector = ?
                    LIMIT 20
                """, [sector]).fetchall()
                
                if sector_stocks:
                    changes = []
                    for (sym,) in sector_stocks[:10]:  # Limit to 10 for performance
                        price_data = self.db.conn.execute("""
                            SELECT close FROM intraday_ohlcv
                            WHERE symbol = ?
                            ORDER BY datetime DESC LIMIT 2
                        """, [sym]).fetchall()
                        
                        if len(price_data) >= 2:
                            current = price_data[0][0]
                            prev = price_data[1][0]
                            if prev > 0:
                                change = (current - prev) / prev * 100
                                changes.append(change)
                    
                    if changes:
                        signals['sector_momentum'] = float(np.clip(np.mean(changes) / 5, -1, 1))
                
        except Exception as e:
            logger.warning(f"Sector signal error for {symbol}: {e}")
        
        return signals
    
    def _get_fundamental_signals(self, symbol: str) -> Dict:
        """Get fundamental valuation signals."""
        signals = {
            'pe_signal': 0,      # Undervalued = positive
            'eps_momentum': 0,
            'dividend_signal': 0,
            'value_score': 0,
            'source': 'fundamentals'
        }
        
        try:
            # Get fundamental data using symbol directly
            result = self.db.conn.execute("""
                SELECT fs.pe_ratio, fs.eps, fs.dividend_yield, fs.market_cap,
                       s.sector
                FROM fundamental_snapshots fs
                JOIN stocks s ON fs.symbol = s.symbol
                WHERE s.symbol = ?
                ORDER BY fs.date DESC LIMIT 1
            """, [symbol]).fetchone()
            
            if result:
                pe, eps, div_yield, mcap, sector = result
                
                # P/E signal (lower is better for value)
                if pe and pe > 0:
                    # Get sector average P/E
                    sector_pe = self.db.conn.execute("""
                        SELECT AVG(fs.pe_ratio)
                        FROM fundamental_snapshots fs
                        JOIN stocks s ON fs.symbol = s.symbol
                        WHERE s.sector = ? AND fs.pe_ratio > 0
                        AND fs.date = (SELECT MAX(date) FROM fundamental_snapshots)
                    """, [sector]).fetchone()
                    
                    if sector_pe and sector_pe[0]:
                        # Positive if undervalued vs sector
                        signals['pe_signal'] = float(np.clip((sector_pe[0] - pe) / sector_pe[0], -1, 1))
                
                # EPS signal
                if eps:
                    signals['eps_momentum'] = 0.5 if eps > 0 else -0.5
                
                # Dividend signal
                if div_yield and div_yield > 0:
                    signals['dividend_signal'] = min(div_yield / 10, 1)
                
                # Combined value score
                signals['value_score'] = float(
                    signals['pe_signal'] * 0.5 +
                    signals['eps_momentum'] * 0.3 +
                    signals['dividend_signal'] * 0.2
                )
                
        except Exception as e:
            logger.warning(f"Fundamental signal error for {symbol}: {e}")
        
        return signals
    
    def _get_disclosure_signals(self, symbol: str) -> Dict:
        """Get corporate disclosure impact signals."""
        signals = {
            'recent_impact': 0,
            'disclosure_count': 0,
            'avg_impact_score': 0,
            'source': 'disclosures'
        }
        
        try:
            # Get recent disclosures for this symbol (DuckDB date syntax)
            result = self.db.conn.execute("""
                SELECT impact_score, created_at
                FROM corporate_disclosures
                WHERE company_symbol = ?
                AND created_at >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY created_at DESC
            """, [symbol]).fetchall()
            
            if result:
                signals['disclosure_count'] = len(result)
                impact_scores = [r[0] for r in result if r[0] is not None]
                if impact_scores:
                    signals['avg_impact_score'] = float(np.mean(impact_scores))
                    # Normalize to -1 to 1 (impact scores are typically -2 to +2)
                    signals['recent_impact'] = float(np.clip(signals['avg_impact_score'] / 2, -1, 1))
                    
        except Exception as e:
            logger.warning(f"Disclosure signal error for {symbol}: {e}")
        
        return signals
    
    def _get_technical_signals(self, symbol: str) -> Dict:
        """Get technical indicator signals (RSI, MACD, patterns)."""
        signals = {
            'rsi_signal': 0,
            'macd_signal': 0,
            'trend_signal': 0,
            'pattern_signal': 0,
            'source': 'technicals'
        }
        
        try:
            # Get recent price data for calculations
            result = self.db.conn.execute("""
                SELECT close FROM intraday_ohlcv
                WHERE symbol = ?
                ORDER BY datetime DESC
                LIMIT 100
            """, [symbol]).fetchall()
            
            if result and len(result) >= 14:
                closes = np.array([r[0] for r in result])[::-1]  # Oldest to newest
                
                # RSI calculation
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = float(np.mean(gains[-14:]))
                avg_loss = float(np.mean(losses[-14:]))
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
                
                # RSI signal: oversold = bullish, overbought = bearish
                if rsi < 30:
                    signals['rsi_signal'] = 0.8
                elif rsi < 40:
                    signals['rsi_signal'] = 0.4
                elif rsi > 70:
                    signals['rsi_signal'] = -0.8
                elif rsi > 60:
                    signals['rsi_signal'] = -0.4
                else:
                    signals['rsi_signal'] = 0
                
                # Simple trend signal (SMA crossover)
                if len(closes) >= 20:
                    sma_10 = float(np.mean(closes[-10:]))
                    sma_20 = float(np.mean(closes[-20:]))
                    signals['trend_signal'] = float(np.clip((sma_10 - sma_20) / sma_20 * 20, -1, 1))
                
                # MACD signal - simplified to avoid scalar issues
                if len(closes) >= 26:
                    ema_12 = self._ema(closes, 12)
                    ema_26 = self._ema(closes, 26)
                    macd_val = ema_12 - ema_26
                    # Simple signal based on MACD vs zero
                    signals['macd_signal'] = float(np.clip(macd_val / (abs(ema_26) * 0.02 + 0.001), -1, 1))
                    
        except Exception as e:
            logger.warning(f"Technical signal error for {symbol}: {e}")
        
        return signals
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average - returns latest value."""
        if len(data) < period:
            return float(np.mean(data))
        
        multiplier = 2 / (period + 1)
        ema = float(data[0])
        for price in data[1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema
    
    def _generate_horizon_prediction(
        self, 
        symbol: str, 
        current_price: float,
        signals: Dict,
        horizon_name: str,
        horizon_days: int
    ) -> Dict:
        """Generate price prediction for a specific time horizon."""
        
        # Calculate weighted signal score (-1 to 1)
        signal_score = self._calculate_weighted_signal(signals)
        
        # Get historical volatility
        volatility = self._get_volatility(symbol, horizon_days)
        
        # Helper to safely convert to float
        def safe_float(val, default=0.0):
            try:
                v = float(val)
                if np.isnan(v) or np.isinf(v):
                    return default
                return v
            except:
                return default
        
        # Ensure all values are valid
        signal_score = safe_float(signal_score, 0.0)
        volatility = safe_float(volatility, 0.05)
        current_price = safe_float(current_price, 100.0)
        
        # Base expected return (signal * volatility scaling)
        base_return = signal_score * volatility * (horizon_days / 5)  # Scale by days
        
        # Clamp reasonable bounds
        base_return = safe_float(np.clip(base_return, -0.30, 0.30), 0.0)
        
        expected_price = current_price * (1 + base_return)
        
        # Generate scenarios with probabilities
        # Bull scenario: more optimistic
        bull_return = base_return + volatility * 0.8
        bull_price = current_price * (1 + bull_return)
        
        # Bear scenario: more pessimistic
        bear_return = base_return - volatility * 0.8
        bear_price = current_price * (1 + bear_return)
        
        # Probabilities based on signal strength
        if signal_score > 0.2:
            bull_prob = 0.35 + signal_score * 0.2
            bear_prob = max(0.05, 0.15 - signal_score * 0.1)
        elif signal_score < -0.2:
            bull_prob = max(0.05, 0.15 - abs(signal_score) * 0.1)
            bear_prob = 0.35 + abs(signal_score) * 0.2
        else:
            bull_prob = 0.25
            bear_prob = 0.25
        
        base_prob = max(0.1, 1 - bull_prob - bear_prob)
        
        # Ensure probabilities sum to 1 and are valid
        bull_prob = safe_float(bull_prob, 0.33)
        base_prob = safe_float(base_prob, 0.34)
        bear_prob = safe_float(bear_prob, 0.33)
        
        total = bull_prob + base_prob + bear_prob
        if total <= 0:
            total = 1
        bull_prob /= total
        base_prob /= total
        bear_prob /= total
        
        return {
            'horizon_name': horizon_name,
            'horizon_days': horizon_days,
            'current_price': safe_float(current_price, 100),
            'expected_price': safe_float(expected_price, current_price),
            'expected_return': safe_float(base_return * 100, 0),
            'bull': {
                'price': safe_float(bull_price, current_price),
                'return_pct': safe_float(bull_return * 100, 0),
                'probability': safe_float(bull_prob * 100, 33)
            },
            'base': {
                'price': safe_float(expected_price, current_price),
                'return_pct': safe_float(base_return * 100, 0),
                'probability': safe_float(base_prob * 100, 34)
            },
            'bear': {
                'price': safe_float(bear_price, current_price),
                'return_pct': safe_float(bear_return * 100, 0),
                'probability': safe_float(bear_prob * 100, 33)
            },
            'confidence': safe_float(self._calculate_overall_confidence(signals) * 100, 50)
        }
    
    def _calculate_weighted_signal(self, signals: Dict) -> float:
        """Calculate weighted average of all signals."""
        weighted_sum = 0
        total_weight = 0
        
        # ML signal
        ml = signals.get('ml', {})
        ml_direction = ml.get('direction', 0)
        ml_conf = ml.get('confidence', 0.5)
        weighted_sum += ml_direction * ml_conf * self.WEIGHTS['ml_ensemble']
        total_weight += self.WEIGHTS['ml_ensemble']
        
        # Flow signal
        flow = signals.get('flow', {})
        flow_signal = flow.get('delta_direction', 0) * 0.5 + flow.get('vwap_position', 0) * 0.3 + flow.get('volume_strength', 0.5) * 0.2
        weighted_sum += flow_signal * self.WEIGHTS['flow_delta']
        total_weight += self.WEIGHTS['flow_delta']
        
        # Sector signal
        sector = signals.get('sector', {})
        sector_signal = sector.get('sector_momentum', 0)
        weighted_sum += sector_signal * self.WEIGHTS['sector_momentum']
        total_weight += self.WEIGHTS['sector_momentum']
        
        # Fundamental signal
        fund = signals.get('fundamental', {})
        fund_signal = fund.get('value_score', 0)
        weighted_sum += fund_signal * self.WEIGHTS['fundamentals']
        total_weight += self.WEIGHTS['fundamentals']
        
        # Technical signal
        tech = signals.get('technical', {})
        tech_signal = (
            tech.get('rsi_signal', 0) * 0.3 +
            tech.get('macd_signal', 0) * 0.3 +
            tech.get('trend_signal', 0) * 0.4
        )
        weighted_sum += tech_signal * self.WEIGHTS['technicals']
        total_weight += self.WEIGHTS['technicals']
        
        # Disclosure signal
        disc = signals.get('disclosure', {})
        disc_signal = disc.get('recent_impact', 0)
        weighted_sum += disc_signal * self.WEIGHTS['disclosures']
        total_weight += self.WEIGHTS['disclosures']
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _get_volatility(self, symbol: str, horizon_days: int) -> float:
        """Calculate historical volatility for the symbol."""
        try:
            result = self.db.conn.execute("""
                SELECT close FROM intraday_ohlcv
                WHERE symbol = ?
                ORDER BY datetime DESC
                LIMIT 100
            """, [symbol]).fetchall()
            
            if result and len(result) >= 20:
                closes = np.array([float(r[0]) for r in result if r[0] and r[0] > 0])
                
                if len(closes) >= 10:
                    # Safe returns calculation - avoid division by zero
                    returns = []
                    for i in range(1, len(closes)):
                        if closes[i-1] > 0:
                            ret = (closes[i] - closes[i-1]) / closes[i-1]
                            if not np.isnan(ret) and not np.isinf(ret):
                                returns.append(ret)
                    
                    if returns:
                        daily_vol = float(np.std(returns))
                        if np.isnan(daily_vol) or daily_vol <= 0:
                            return 0.05
                        
                        # Scale to horizon (rough approximation)
                        horizon_vol = daily_vol * np.sqrt(horizon_days)
                        return min(float(horizon_vol), 0.30)  # Cap at 30%
                
        except Exception as e:
            logger.warning(f"Volatility error for {symbol}: {e}")
        
        return 0.05  # Default 5% volatility
    
    def _calculate_bid_offer_probability(self, symbol: str, signals: Dict) -> Dict:
        """Calculate probability of full bid vs full offer at session close."""
        
        # Base probability on signals
        weighted_signal = self._calculate_weighted_signal(signals)
        
        # Flow signals weight more for intraday
        flow = signals.get('flow', {})
        intraday_bias = flow.get('delta_direction', 0) * 0.3 + flow.get('vwap_position', 0) * 0.3
        
        # Combined bias
        bias = weighted_signal * 0.5 + intraday_bias * 0.5
        
        # Convert to probabilities
        if bias > 0.3:
            full_bid = 0.55 + bias * 0.3
            full_offer = 0.10 - bias * 0.05
        elif bias < -0.3:
            full_bid = 0.10 + bias * 0.05
            full_offer = 0.55 - bias * 0.3
        else:
            full_bid = 0.35 + bias * 0.3
            full_offer = 0.35 - bias * 0.3
        
        mixed = 1 - full_bid - full_offer
        
        # Normalize
        total = full_bid + mixed + full_offer
        
        return {
            'full_bid': round((full_bid / total) * 100, 1),
            'mixed': round((mixed / total) * 100, 1),
            'full_offer': round((full_offer / total) * 100, 1)
        }
    
    def _calculate_overall_confidence(self, signals: Dict) -> float:
        """Calculate overall confidence in the synthesis."""
        confidence_factors = []
        
        # ML confidence
        ml = signals.get('ml', {})
        confidence_factors.append(ml.get('confidence', 0.5))
        
        # Data availability confidence
        flow = signals.get('flow', {})
        if flow.get('volume_strength', 0) > 0:
            confidence_factors.append(0.7)
        
        fund = signals.get('fundamental', {})
        if fund.get('pe_signal', 0) != 0:
            confidence_factors.append(0.8)
        
        tech = signals.get('technical', {})
        if tech.get('rsi_signal', 0) != 0:
            confidence_factors.append(0.75)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _generate_narrative(self, symbol: str, signals: Dict, predictions: Dict) -> str:
        """Generate AI narrative for the synthesis."""
        weighted_signal = self._calculate_weighted_signal(signals)
        
        direction = "bullish" if weighted_signal > 0.1 else "bearish" if weighted_signal < -0.1 else "neutral"
        
        # Build narrative
        parts = [f"{symbol} shows a {direction} outlook based on synthesis of all signals."]
        
        # Key drivers
        drivers = []
        
        ml = signals.get('ml', {})
        if ml.get('confidence', 0) > 0.6:
            drivers.append(f"ML models show {'positive' if ml.get('direction', 0) > 0 else 'negative'} momentum")
        
        flow = signals.get('flow', {})
        if abs(flow.get('delta_direction', 0)) > 0.3:
            drivers.append(f"Order flow is {'accumulating' if flow.get('delta_direction', 0) > 0 else 'distributing'}")
        
        fund = signals.get('fundamental', {})
        if fund.get('pe_signal', 0) > 0.3:
            drivers.append("Valuation is attractive vs sector")
        elif fund.get('pe_signal', 0) < -0.3:
            drivers.append("Valuation appears stretched vs sector")
        
        if drivers:
            parts.append("Key drivers: " + "; ".join(drivers) + ".")
        
        # Short-term outlook
        pred_2d = predictions.get('2D', {})
        if pred_2d:
            exp_ret = pred_2d.get('expected_return', 0)
            parts.append(f"2-day expected move: {'+' if exp_ret >= 0 else ''}{exp_ret}%.")
        
        return " ".join(parts)
