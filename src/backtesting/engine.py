"""
Backtesting Engine for MetaQuant Nigeria.
Tests trading strategies on historical data using unified signal scoring.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from .signal_scorer import SignalScorer
from .metrics import calculate_metrics, calculate_position_size

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    entry_score: float
    entry_signal: str
    entry_attribution: Dict[str, float] = None  # Component scores at entry


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float
    holding_days: int
    entry_signal: str
    exit_signal: str
    entry_score: float
    exit_score: float
    entry_attribution: Dict[str, float] = None  # Signal component scores at entry
    exit_attribution: Dict[str, float] = None   # Signal component scores at exit


class BacktestEngine:
    """
    Backtesting engine with multi-signal scoring.
    
    Features:
    - Uses SignalScorer for unified signals
    - Per-stock stop loss/take profit (volatility-adjusted)
    - Supports multiple positions
    - Tracks equity curve
    - Calculates comprehensive metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 10_000_000,  # 10M NGN
        max_positions: int = 10,
        position_size_pct: float = 0.10,  # 10% per position
        stop_loss_pct: float = 0.05,  # Default 5% stop loss
        take_profit_pct: float = 0.15,  # Default 15% take profit
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
        signal_weights: Optional[Dict[str, float]] = None,
        stock_params: Optional[Dict[str, Dict[str, float]]] = None,  # Per-stock SL/TP
        db=None,  # Database for fundamentals and signal history
        ml_engine=None,  # ML engine for predictions
        use_full_signals: bool = True  # Use all data sources
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital in NGN
            max_positions: Maximum concurrent positions
            position_size_pct: Position size as % of capital
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
            buy_threshold: Score threshold for buying
            sell_threshold: Score threshold for selling
            signal_weights: Custom signal weights
            stock_params: Per-stock parameters {symbol: {'stop_loss': 0.05, 'take_profit': 0.15}}
            db: Database manager for fundamentals
            ml_engine: ML engine for predictions
            use_full_signals: Whether to use all data sources
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.default_stop_loss = stop_loss_pct
        self.default_take_profit = take_profit_pct
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # Per-stock parameters
        self.stock_params = stock_params or {}
        
        # Data sources
        self.db = db
        self.ml_engine = ml_engine
        self.use_full_signals = use_full_signals
        
        # Signal scorer
        self.scorer = SignalScorer(weights=signal_weights)
        
        # State
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # Cache for fundamentals
        self._fundamental_cache: Dict[str, Dict] = {}
        
        # Results
        self.results: Optional[Dict] = None
    
    def run(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        signal_data: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            symbols: List of stock symbols to trade
            price_data: Dict of symbol -> DataFrame with OHLCV
            signal_data: Dict with all historical signals by date/symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict with backtest results
        """
        logger.info(f"Starting backtest for {len(symbols)} symbols...")
        
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Get date range from data
        all_dates = set()
        for sym, df in price_data.items():
            if sym in symbols and not df.empty:
                if 'date' in df.columns:
                    # Convert to string format YYYY-MM-DD
                    dates_str = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').tolist()
                    all_dates.update(dates_str)
                elif isinstance(df.index, pd.DatetimeIndex):
                    all_dates.update(df.index.strftime('%Y-%m-%d').tolist())
        
        dates = sorted(all_dates)
        
        # Log original date range
        if dates:
            logger.info(f"Data available: {dates[0]} to {dates[-1]} ({len(dates)} total days)")
        
        # CRITICAL: Filter out future dates (synthetic data)
        today_str = datetime.now().strftime('%Y-%m-%d')
        dates = [d for d in dates if d <= today_str]
        
        if not dates:
            return {'success': False, 'error': 'No historical data (only future synthetic dates found)'}
        
        logger.info(f"Valid historical data: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        
        # Apply date filters
        if start_date:
            start_str = str(start_date)[:10]  # Ensure YYYY-MM-DD format
            dates = [d for d in dates if d >= start_str]
        if end_date:
            end_str = str(end_date)[:10]  # Ensure YYYY-MM-DD format
            dates = [d for d in dates if d <= end_str]
        
        if not dates:
            return {'success': False, 'error': 'No dates in range after filtering'}
        
        logger.info(f"Backtesting from {dates[0]} to {dates[-1]} ({len(dates)} days)")
        
        # Main backtest loop
        prev_equity = self.initial_capital
        
        for date in dates:
            # Get prices for this date
            date_prices = {}
            for sym in symbols:
                df = price_data.get(sym)
                if df is None or df.empty:
                    continue
                
                # Find price on this date
                if 'date' in df.columns:
                    row = df[df['date'].astype(str) == date]
                else:
                    try:
                        row = df.loc[date:date]
                    except:
                        continue
                
                if not row.empty:
                    date_prices[sym] = float(row['close'].iloc[0])
            
            if not date_prices:
                continue
            
            # Update positions and check stops
            self._check_stops_and_targets(date, date_prices)
            
            # Score all symbols
            scores = {}
            for sym in symbols:
                if sym not in date_prices:
                    continue
                
                # Use full multi-source signals if enabled
                if self.use_full_signals:
                    score_result = self._compute_full_score(
                        sym, date, price_data.get(sym), date_prices[sym]
                    )
                elif signal_data:
                    # Use pre-computed historical signals
                    score_result = self.scorer.score_for_backtest(
                        date=date,
                        symbol=sym,
                        historical_data=signal_data
                    )
                else:
                    # Fallback: Simple price-based scoring using momentum
                    score_result = self._price_based_score(sym, date, price_data.get(sym), date_prices[sym])
                
                scores[sym] = score_result
            
            # Execute trades based on scores
            self._execute_signals(date, scores, date_prices)
            
            # Calculate equity
            equity = self.capital
            for sym, pos in self.positions.items():
                if sym in date_prices:
                    equity += pos.quantity * date_prices[sym]
            
            # Record equity
            self.equity_curve.append({
                'date': date,
                'equity': equity,
                'positions': len(self.positions)
            })
            
            # Daily return
            if prev_equity > 0:
                daily_return = (equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)
            prev_equity = equity
        
        # Close remaining positions
        if dates:
            final_prices = {}
            for sym in self.positions:
                df = price_data.get(sym)
                if df is not None and not df.empty:
                    final_prices[sym] = float(df['close'].iloc[-1])
            
            for sym in list(self.positions.keys()):
                if sym in final_prices:
                    self._close_position(sym, dates[-1], final_prices[sym], 'END_OF_BACKTEST', 0)
        
        # Calculate metrics
        metrics = calculate_metrics(
            [t.__dict__ for t in self.trades],
            self.initial_capital
        )
        
        # OVERRIDE metrics with correct return calculated from actual capital
        # After closing all positions, self.capital = final cash value
        final_equity = self.capital  # All positions closed, this is the correct final value
        actual_return = final_equity - self.initial_capital
        actual_return_pct = (actual_return / self.initial_capital) * 100
        
        logger.info(f"DEBUG: initial_capital={self.initial_capital:,.0f}, final_equity={final_equity:,.0f}, return={actual_return_pct:.2f}%")
        
        metrics['total_return'] = round(actual_return, 2)
        metrics['total_return_pct'] = round(actual_return_pct, 2)
        metrics['final_equity'] = round(final_equity, 2)
        
        self.results = {
            'success': True,
            'symbols': symbols,
            'start_date': dates[0] if dates else None,
            'end_date': dates[-1] if dates else None,
            'total_days': len(dates),
            'metrics': metrics,
            'trades': [t.__dict__ for t in self.trades],
            'equity_curve': self.equity_curve,
            'settings': {
                'initial_capital': self.initial_capital,
                'max_positions': self.max_positions,
                'position_size_pct': self.position_size_pct,
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'weights': self.scorer.weights
            }
        }
        
        logger.info(f"Backtest complete. Return: {metrics['total_return_pct']:.2f}%, "
                    f"Sharpe: {metrics['sharpe_ratio']:.2f}, Win Rate: {metrics['win_rate']:.1f}%")
        
        return self.results
    
    def _check_stops_and_targets(self, date: str, prices: Dict[str, float]):
        """Check stop losses and take profits using per-stock parameters."""
        for sym in list(self.positions.keys()):
            if sym not in prices:
                continue
            
            pos = self.positions[sym]
            price = prices[sym]
            
            # Get stock-specific or default parameters
            params = self.stock_params.get(sym, {})
            stop_loss_pct = params.get('stop_loss', self.default_stop_loss)
            take_profit_pct = params.get('take_profit', self.default_take_profit)
            
            # Stop loss
            if price <= pos.entry_price * (1 - stop_loss_pct):
                self._close_position(sym, date, price, 'STOP_LOSS', -1)
                continue
            
            # Take profit
            if price >= pos.entry_price * (1 + take_profit_pct):
                self._close_position(sym, date, price, 'TAKE_PROFIT', 1)
    
    def _execute_signals(self, date: str, scores: Dict[str, Dict], prices: Dict[str, float]):
        """Execute buy/sell signals."""
        
        # Check existing positions for sell signals
        for sym in list(self.positions.keys()):
            if sym not in scores:
                continue
            
            score_result = scores[sym]
            if score_result['composite_score'] <= self.sell_threshold:
                if sym in prices:
                    self._close_position(
                        sym, date, prices[sym], 
                        score_result['signal'], 
                        score_result['composite_score'],
                        score_result.get('component_scores', {})
                    )
        
        # Check for buy signals
        if len(self.positions) < self.max_positions:
            # Rank by score
            buy_candidates = [
                (sym, s) for sym, s in scores.items()
                if s['composite_score'] >= self.buy_threshold and sym not in self.positions
            ]
            buy_candidates.sort(key=lambda x: x[1]['composite_score'], reverse=True)
            
            for sym, score_result in buy_candidates:
                if len(self.positions) >= self.max_positions:
                    break
                if sym not in prices:
                    continue
                
                self._open_position(
                    sym, date, prices[sym],
                    score_result['signal'],
                    score_result['composite_score'],
                    score_result.get('component_scores', {})
                )
    
    def _open_position(self, symbol: str, date: str, price: float, signal: str, score: float, attribution: Dict[str, float] = None):
        """Open a new position."""
        position_value = self.capital * self.position_size_pct
        quantity = int(position_value / price)
        
        if quantity <= 0:
            return
        
        cost = quantity * price
        if cost > self.capital:
            return
        
        self.capital -= cost
        
        self.positions[symbol] = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            entry_score=score,
            entry_signal=signal,
            entry_attribution=attribution or {}
        )
        
        logger.debug(f"OPEN: {symbol} @ {price:.2f} x {quantity} (score: {score:.3f})")
    
    def _close_position(self, symbol: str, date: str, price: float, signal: str, score: float, exit_attribution: Dict[str, float] = None):
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        proceeds = pos.quantity * price
        pnl = proceeds - (pos.quantity * pos.entry_price)
        return_pct = (price - pos.entry_price) / pos.entry_price * 100
        
        # SANITY CHECK: Cap returns at ±50% per trade to filter bad data
        MAX_RETURN_PCT = 50.0
        if abs(return_pct) > MAX_RETURN_PCT:
            # Clamp the return to realistic levels
            capped_return_pct = MAX_RETURN_PCT if return_pct > 0 else -MAX_RETURN_PCT
            capped_price = pos.entry_price * (1 + capped_return_pct / 100)
            proceeds = pos.quantity * capped_price
            pnl = proceeds - (pos.quantity * pos.entry_price)
            logger.debug(f"CAPPED: {symbol} {return_pct:.1f}% -> {capped_return_pct:.1f}%")
            return_pct = capped_return_pct
        
        # Calculate holding days
        try:
            entry_dt = datetime.strptime(pos.entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(date, '%Y-%m-%d')
            holding_days = (exit_dt - entry_dt).days
        except:
            holding_days = 1
        
        trade = Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=date,
            exit_price=price,
            quantity=pos.quantity,
            pnl=pnl,
            return_pct=return_pct,
            holding_days=holding_days,
            entry_signal=pos.entry_signal,
            exit_signal=signal,
            entry_score=pos.entry_score,
            exit_score=score,
            entry_attribution=pos.entry_attribution or {},
            exit_attribution=exit_attribution or {}
        )
        
        self.trades.append(trade)
        self.capital += proceeds
        del self.positions[symbol]
        
        # Debug: Log trades with huge returns (now capped)
        if abs(return_pct) >= MAX_RETURN_PCT:
            logger.warning(f"CAPPED TRADE: {symbol} entry={pos.entry_price:.2f} exit={price:.2f} capped to {return_pct:.1f}%")
        
        logger.debug(f"CLOSE: {symbol} @ {price:.2f}, PnL: {pnl:.2f} ({return_pct:.1f}%)")
    
    def _compute_full_score(
        self, 
        symbol: str, 
        date: str, 
        df: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """
        Compute full multi-source score using all available data.
        
        Weights:
        - Momentum: 35%
        - ML Prediction: 25% 
        - Fundamentals: 20%
        - Anomaly: 10%
        - Trend: 10%
        """
        scores = {
            'momentum': 0.0,
            'ml': 0.0,
            'fundamental': 0.0,
            'anomaly': 0.0,
            'trend': 0.0
        }
        
        if df is None or df.empty or len(df) < 25:
            return {'composite_score': 0, 'signal': 'HOLD', 'component_scores': scores}
        
        try:
            # Convert to float
            close = pd.to_numeric(df['close'], errors='coerce').astype(float)
            
            # ===== MOMENTUM (35%) =====
            if len(close) >= 20:
                mom_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if close.iloc[-5] > 0 else 0
                mom_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if close.iloc[-20] > 0 else 0
                
                # Normalize to -1 to 1
                momentum_score = (float(mom_5) + float(mom_20)) / 2
                momentum_score = max(-1, min(1, momentum_score * 5))  # Scale up
                scores['momentum'] = momentum_score
            
            # ===== TREND (10%) =====
            if len(close) >= 50:
                ma_20 = close.tail(20).mean()
                ma_50 = close.tail(50).mean()
                scores['trend'] = 1.0 if ma_20 > ma_50 else -1.0
            
            # ===== ML PREDICTION (25%) =====
            ml_attempted = False
            if self.ml_engine and hasattr(self.ml_engine, 'predict'):
                try:
                    ml_result = self.ml_engine.predict(symbol)
                    ml_attempted = True
                    if ml_result and ml_result.get('success'):
                        direction = ml_result.get('direction', 'FLAT')
                        # Normalize confidence from 0-100 to 0-1
                        raw_conf = ml_result.get('confidence', 50)
                        confidence = raw_conf / 100 if raw_conf > 1 else raw_conf
                        
                        if direction == 'UP':
                            scores['ml'] = confidence  # +0.0 to +1.0
                        elif direction == 'DOWN':
                            scores['ml'] = -confidence  # -1.0 to 0.0
                        else:  # FLAT
                            # Use expected return to determine slight bias
                            exp_ret = ml_result.get('expected_return', 0)
                            scores['ml'] = max(-0.3, min(0.3, exp_ret / 10))
                except Exception as e:
                    logger.debug(f"ML prediction error for {symbol}: {e}")
            
            # Fallback: Use momentum as ML proxy if ML prediction failed
            if not ml_attempted or scores['ml'] == 0:
                # Use momentum as a simple ML proxy
                scores['ml'] = scores['momentum'] * 0.5  # Damped version of momentum
            
            # ===== FUNDAMENTALS (20%) =====
            if self.db:
                fund_score = self._score_fundamentals(symbol)
                scores['fundamental'] = fund_score
            
            # Fallback: Give slight positive bias if no fundamental data
            if scores['fundamental'] == 0:
                # Use price stability as proxy - lower volatility = better fundamentals
                if len(close) >= 20:
                    volatility = close.pct_change().tail(20).std()
                    # Low volatility stocks get small positive, high volatility get negative
                    scores['fundamental'] = max(-0.2, min(0.2, 0.1 - volatility * 5))
            
            # ===== ANOMALY (10%) =====
            if self.ml_engine and hasattr(self.ml_engine, 'detect_anomalies'):
                try:
                    anomalies = self.ml_engine.detect_anomalies(symbol)
                    if anomalies:
                        # Recent unusual activity could indicate opportunity
                        scores['anomaly'] = 0.5  # Neutral positive
                except:
                    pass
            
            # ===== COMPOSITE SCORE =====
            weights = {
                'momentum': 0.35,
                'ml': 0.25,
                'fundamental': 0.20,
                'anomaly': 0.10,
                'trend': 0.10
            }
            
            composite = sum(scores[k] * weights[k] for k in scores)
            composite = max(-1, min(1, composite))
            
            # Determine signal
            if composite > 0.15:
                signal = 'BUY'
            elif composite < -0.15:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'composite_score': composite,
                'signal': signal,
                'component_scores': scores
            }
            
        except Exception as e:
            logger.debug(f"Full score error for {symbol}: {e}")
            return {'composite_score': 0, 'signal': 'HOLD', 'component_scores': scores}
    
    def _score_fundamentals(self, symbol: str) -> float:
        """Score based on fundamental metrics."""
        if symbol in self._fundamental_cache:
            fund = self._fundamental_cache[symbol]
        else:
            try:
                stock = self.db.get_stock(symbol)
                if stock:
                    fund = self.db.get_fundamentals(stock['id'])
                    self._fundamental_cache[symbol] = fund
                else:
                    return 0.0
            except Exception as e:
                logger.debug(f"Fundamental fetch failed for {symbol}: {e}")
                return 0.0
        
        if not fund:
            return 0.0
        
        score = 0.0
        
        # P/E ratio (lower is better, but not negative) - LOOSENED
        pe = fund.get('pe_ratio')
        if pe and pe > 0:
            if pe < 10:
                score += 0.4
            elif pe < 20:
                score += 0.2
            elif pe < 30:
                score += 0.1
            else:
                score -= 0.1
        
        # Dividend yield (higher is better) - LOOSENED
        div_yield = fund.get('dividend_yield')
        if div_yield and div_yield > 0:
            if div_yield > 4:
                score += 0.3
            elif div_yield > 1:
                score += 0.2
            else:
                score += 0.1
        
        # ROE (higher is better) - LOOSENED
        roe = fund.get('roe')
        if roe and roe > 0:
            if roe > 15:
                score += 0.3
            elif roe > 8:
                score += 0.2
            elif roe > 3:
                score += 0.1
        
        # Market cap - give small boost for any stock with market cap data
        mkt_cap = fund.get('market_cap')
        if mkt_cap and mkt_cap > 0:
            score += 0.05
        
        return max(-1, min(1, score))
    
    def _price_based_score(self, symbol: str, date: str, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Generate trading signal from price data when no external signals available.
        Uses momentum (5, 20 day) and mean reversion.
        """
        if df is None or df.empty or len(df) < 25:
            return {'composite_score': 0, 'signal': 'HOLD', 'component_scores': {}}
        
        try:
            # Convert Decimal to float if needed
            close_col = df['close'].apply(lambda x: float(x) if hasattr(x, '__float__') else x)
            
            # Get recent prices
            recent_close = close_col.tail(25).values
            if len(recent_close) < 20:
                return {'composite_score': 0, 'signal': 'HOLD', 'component_scores': {}}
            
            # Momentum signals
            mom_5 = (recent_close[-1] - recent_close[-5]) / recent_close[-5] if recent_close[-5] > 0 else 0
            mom_20 = (recent_close[-1] - recent_close[-20]) / recent_close[-20] if recent_close[-20] > 0 else 0
            
            # Mean reversion: price vs 20-day MA
            ma_20 = np.mean(recent_close[-20:])
            deviation = (recent_close[-1] - ma_20) / ma_20 if ma_20 > 0 else 0
            
            # Trend strength
            trend = 1 if mom_20 > 0 else -1
            
            # Composite score - more conservative
            # Only strong momentum triggers signals
            momentum_score = (mom_5 + mom_20) / 2  # Average, not scaled up
            
            # Mean reversion bonus
            if trend > 0 and deviation < -0.10:
                momentum_score += 0.15  # Oversold in uptrend
            elif trend < 0 and deviation > 0.10:
                momentum_score -= 0.15  # Overbought in downtrend
            
            # Clamp to [-1, 1]
            score = max(-1, min(1, momentum_score))
            
            # Loosened thresholds - trade more stocks
            # Buy: positive momentum with some strength
            if (score > 0.03 and mom_5 > 0.01) or (mom_5 > 0 and mom_20 > 0.02):
                signal = 'BUY'
            # Sell: negative momentum
            elif score < -0.03 and (mom_5 < -0.01 or mom_20 < -0.02):
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'composite_score': score,
                'signal': signal,
                'component_scores': {
                    'momentum_5d': mom_5,
                    'momentum_20d': mom_20,
                    'deviation': deviation
                }
            }
        except Exception as e:
            logger.debug(f"Price scoring error for {symbol}: {e}")
            return {'composite_score': 0, 'signal': 'HOLD', 'component_scores': {}}
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.__dict__ for t in self.trades])
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve)

