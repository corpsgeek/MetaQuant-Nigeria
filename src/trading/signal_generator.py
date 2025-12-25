"""
Signal Generator - Generate daily trading signals based on optimized strategies.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    score: float  # -1 to +1
    current_price: float
    stop_loss: float
    take_profit: float
    attribution: Dict
    strategy: Dict
    rank: int = 0
    
    @property
    def signal_strength(self) -> str:
        """Human-readable signal strength."""
        abs_score = abs(self.score)
        if abs_score >= 0.7:
            return "STRONG"
        elif abs_score >= 0.4:
            return "MODERATE"
        else:
            return "WEAK"


class SignalGenerator:
    """
    Generates daily trading signals based on optimized strategies.
    
    Process:
    1. Load optimized strategies for each stock
    2. Calculate current signal score using multi-factor model
    3. Apply strategy thresholds to determine BUY/SELL/HOLD
    4. Rank signals by score
    5. Log to database
    """
    
    def __init__(self, trading_tables, db, ml_engine=None, 
                 score_calculator=None):
        """
        Initialize the signal generator.
        
        Args:
            trading_tables: TradingTables for strategies and signal logging
            db: Database manager
            ml_engine: ML engine for predictions
            score_calculator: Optional callable(symbol, price_data) -> score_dict
        """
        self.tables = trading_tables
        self.db = db
        self.ml_engine = ml_engine
        self.score_calculator = score_calculator
        
        # Cache strategies
        self._strategy_cache = {}
        self._last_refresh = None
    
    def generate_signals(self, price_data: Dict[str, pd.DataFrame],
                        current_prices: Dict[str, float] = None,
                        symbols: List[str] = None) -> List[TradingSignal]:
        """
        Generate trading signals for all stocks.
        
        Args:
            price_data: Dict of symbol -> DataFrame with OHLCV
            current_prices: Optional dict of symbol -> current price
            symbols: Optional list of symbols to process (default: all with strategies)
        
        Returns:
            List of TradingSignal sorted by score (strongest first)
        """
        # Refresh strategies cache
        self._refresh_strategies()
        
        # Determine symbols to process - use price_data keys if no strategy cache
        if symbols is None:
            if self._strategy_cache:
                symbols = list(self._strategy_cache.keys())
            else:
                symbols = list(price_data.keys())
        
        signals = []
        
        for symbol in symbols:
            if symbol not in price_data or price_data[symbol].empty:
                continue
            
            try:
                signal = self._generate_signal(
                    symbol, 
                    price_data[symbol],
                    current_prices.get(symbol) if current_prices else None
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Signal generation failed for {symbol}: {e}")
        
        # Sort: BUYs by highest score first, then SELLs by lowest score, then HOLDs
        def signal_sort_key(s):
            if s.signal == 'BUY':
                return (0, -s.score)  # BUYs first, highest score first
            elif s.signal == 'SELL':
                return (1, s.score)   # SELLs second, lowest score first
            else:
                return (2, abs(s.score))  # HOLDs last
        
        signals.sort(key=signal_sort_key)
        
        # Assign ranks
        for i, signal in enumerate(signals):
            signal.rank = i + 1
        
        # Log signals to database
        self._log_signals(signals)
        
        logger.info(f"Generated {len(signals)} signals: "
                   f"{sum(1 for s in signals if s.signal == 'BUY')} BUY, "
                   f"{sum(1 for s in signals if s.signal == 'SELL')} SELL, "
                   f"{sum(1 for s in signals if s.signal == 'HOLD')} HOLD")
        
        return signals
    
    def get_buy_signals(self, signals: List[TradingSignal], 
                       max_count: int = 15) -> List[TradingSignal]:
        """Get top BUY signals."""
        buys = [s for s in signals if s.signal == 'BUY']
        return buys[:max_count]
    
    def get_sell_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Get all SELL signals."""
        return [s for s in signals if s.signal == 'SELL']
    
    def _generate_signal(self, symbol: str, df: pd.DataFrame,
                        current_price: float = None) -> Optional[TradingSignal]:
        """Generate signal for a single stock."""
        if df.empty or len(df) < 20:
            return None
        
        # Get current price
        if current_price is None:
            current_price = float(df['close'].iloc[-1])
        
        # Get strategy for this stock
        strategy = self._get_strategy(symbol)
        
        # Calculate signal score
        if self.score_calculator:
            score_result = self.score_calculator(symbol, df, current_price)
        else:
            score_result = self._compute_score(symbol, df, current_price)
        
        composite_score = score_result.get('composite_score', 0)
        attribution = score_result.get('component_scores', {})
        
        # Apply strategy thresholds
        buy_threshold = strategy.get('buy_threshold', 0.3)
        sell_threshold = strategy.get('sell_threshold', -0.3)
        
        if composite_score >= buy_threshold:
            signal = 'BUY'
        elif composite_score <= sell_threshold:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Calculate stop loss and take profit prices
        stop_loss_pct = strategy.get('optimal_stop_loss', 0.05)
        take_profit_pct = strategy.get('optimal_take_profit', 0.15)
        
        return TradingSignal(
            symbol=symbol,
            signal=signal,
            score=composite_score,
            current_price=current_price,
            stop_loss=current_price * (1 - stop_loss_pct),
            take_profit=current_price * (1 + take_profit_pct),
            attribution=attribution,
            strategy=strategy
        )
    
    def _compute_score(self, symbol: str, df: pd.DataFrame, 
                      current_price: float) -> Dict:
        """
        Compute signal score using multi-factor model.
        
        Similar to BacktestEngine._compute_full_score but for live use.
        """
        scores = {
            'momentum': 0.0,
            'ml': 0.0,
            'pca_alpha': 0.0,
            'fundamental': 0.0,
            'factor_align': 0.0,
            'anomaly': 0.0,
            'trend': 0.0
        }
        
        try:
            # Reset index if DatetimeIndex to avoid rmul errors
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index(drop=True)
            
            close = pd.to_numeric(df['close'], errors='coerce').astype(float).values
            
            # ===== MOMENTUM (25%) =====
            raw_momentum = 0.0
            if len(close) >= 20:
                mom_5 = (close[-1] - close[-5]) / close[-5] if close[-5] > 0 else 0
                mom_20 = (close[-1] - close[-20]) / close[-20] if close[-20] > 0 else 0
                raw_momentum = (float(mom_5) + float(mom_20)) / 2
                scores['momentum'] = max(-1, min(1, raw_momentum * 5))
            
            # ===== TREND (5%) =====
            if len(close) >= 50:
                ma_20 = float(np.mean(close[-20:]))
                ma_50 = float(np.mean(close[-50:]))
                scores['trend'] = 1.0 if ma_20 > ma_50 else -1.0
            
            # ===== ML PREDICTION (20%) =====
            if self.ml_engine and hasattr(self.ml_engine, 'predict'):
                try:
                    ml_result = self.ml_engine.predict(symbol)
                    if ml_result and ml_result.get('success'):
                        direction = ml_result.get('direction', 'FLAT')
                        raw_conf = ml_result.get('confidence', 50)
                        confidence = raw_conf / 100 if raw_conf > 1 else raw_conf
                        
                        if direction == 'UP':
                            scores['ml'] = confidence
                        elif direction == 'DOWN':
                            scores['ml'] = -confidence
                        else:
                            exp_ret = ml_result.get('expected_return', 0)
                            scores['ml'] = max(-0.3, min(0.3, exp_ret / 10))
                except:
                    scores['ml'] = scores['momentum'] * 0.5
            else:
                scores['ml'] = scores['momentum'] * 0.5
            
            # ===== PCA ALPHA (20%) =====
            # Factor-adjusted signal removes systematic risk
            if self.ml_engine and hasattr(self.ml_engine, 'get_pca_score'):
                raw_signal = scores['momentum'] * 0.5 + scores['ml'] * 0.5
                scores['pca_alpha'] = self.ml_engine.get_pca_score(symbol, raw_signal)
            else:
                scores['pca_alpha'] = scores['momentum']
            
            # ===== FACTOR ALIGNMENT (10%) =====
            # How well aligned with current regime
            if self.ml_engine and hasattr(self.ml_engine, 'get_factor_alignment'):
                scores['factor_align'] = self.ml_engine.get_factor_alignment(symbol)
            
            # ===== FUNDAMENTALS (15%) =====
            if self.db:
                fund_score = self._score_fundamentals(symbol)
                scores['fundamental'] = fund_score
            
            if scores['fundamental'] == 0 and len(close) >= 20:
                # Calculate volatility using numpy for array
                returns = np.diff(close[-21:]) / close[-21:-1]
                volatility = float(np.std(returns))
                scores['fundamental'] = max(-0.2, min(0.2, 0.1 - volatility * 5))
            
            # ===== COMPOSITE SCORE =====
            # New weights with PCA components
            weights = {
                'momentum': 0.25,
                'ml': 0.20,
                'pca_alpha': 0.20,
                'fundamental': 0.15,
                'factor_align': 0.10,
                'trend': 0.05,
                'anomaly': 0.05
            }
            
            composite = sum(scores[k] * weights[k] for k in scores)
            composite = max(-1, min(1, composite))
            
            return {
                'composite_score': composite,
                'component_scores': scores
            }
            
        except Exception as e:
            logger.debug(f"Score calculation failed for {symbol}: {e}")
            return {'composite_score': 0, 'component_scores': scores}
    
    def _score_fundamentals(self, symbol: str) -> float:
        """Score based on fundamental metrics."""
        try:
            stock = self.db.get_stock(symbol)
            if not stock:
                return 0.0
            
            fund = self.db.get_fundamentals(stock['id'])
            if not fund:
                return 0.0
            
            score = 0.0
            
            # P/E ratio
            pe = fund.get('pe_ratio')
            if pe and pe > 0:
                if pe < 10:
                    score += 0.4
                elif pe < 20:
                    score += 0.2
                elif pe < 30:
                    score += 0.1
            
            # Dividend yield
            div_yield = fund.get('dividend_yield')
            if div_yield and div_yield > 0:
                if div_yield > 4:
                    score += 0.3
                elif div_yield > 1:
                    score += 0.2
            
            # ROE
            roe = fund.get('roe')
            if roe and roe > 0:
                if roe > 15:
                    score += 0.3
                elif roe > 8:
                    score += 0.2
            
            return max(-1, min(1, score))
            
        except:
            return 0.0
    
    def _get_strategy(self, symbol: str) -> Dict:
        """Get strategy for a stock, with caching."""
        if symbol not in self._strategy_cache:
            strategy = self.tables.get_stock_strategy(symbol)
            if strategy:
                self._strategy_cache[symbol] = strategy
            else:
                self._strategy_cache[symbol] = {
                    'optimal_stop_loss': 0.05,
                    'optimal_take_profit': 0.15,
                    'buy_threshold': 0.3,
                    'sell_threshold': -0.3
                }
        return self._strategy_cache[symbol]
    
    def _refresh_strategies(self, force: bool = False):
        """Refresh strategy cache if stale."""
        now = datetime.now()
        if force or self._last_refresh is None or \
           (now - self._last_refresh).seconds > 300:  # 5 min cache
            strategies = self.tables.get_active_strategies()
            self._strategy_cache = {s['symbol']: s for s in strategies}
            self._last_refresh = now
    
    def _log_signals(self, signals: List[TradingSignal]):
        """Log signals to database."""
        for signal in signals:
            try:
                self.tables.log_signal(
                    symbol=signal.symbol,
                    signal=signal.signal,
                    score=signal.score,
                    current_price=signal.current_price,
                    attribution=signal.attribution
                )
            except Exception as e:
                logger.debug(f"Failed to log signal for {signal.symbol}: {e}")
    
    def get_latest_signals(self, limit: int = 50) -> List[Dict]:
        """Get recently logged signals from database."""
        return self.tables.get_latest_signals(limit)
