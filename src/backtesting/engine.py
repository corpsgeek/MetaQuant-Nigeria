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
        stock_params: Optional[Dict[str, Dict[str, float]]] = None  # Per-stock SL/TP
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
        
        # Signal scorer
        self.scorer = SignalScorer(weights=signal_weights)
        
        # State
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
        
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
        
        # Get date range
        all_dates = set()
        for sym, df in price_data.items():
            if sym in symbols and not df.empty:
                if 'date' in df.columns:
                    all_dates.update(df['date'].astype(str).tolist())
                elif isinstance(df.index, pd.DatetimeIndex):
                    all_dates.update(df.index.strftime('%Y-%m-%d').tolist())
        
        dates = sorted(all_dates)
        
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        
        if not dates:
            return {'success': False, 'error': 'No dates in range'}
        
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
                
                # If we have signal data, use the full scorer
                if signal_data:
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
                        score_result['composite_score']
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
                    score_result['composite_score']
                )
    
    def _open_position(self, symbol: str, date: str, price: float, signal: str, score: float):
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
            entry_signal=signal
        )
        
        logger.debug(f"OPEN: {symbol} @ {price:.2f} x {quantity} (score: {score:.3f})")
    
    def _close_position(self, symbol: str, date: str, price: float, signal: str, score: float):
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        proceeds = pos.quantity * price
        pnl = proceeds - (pos.quantity * pos.entry_price)
        return_pct = (price - pos.entry_price) / pos.entry_price * 100
        
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
            exit_score=score
        )
        
        self.trades.append(trade)
        self.capital += proceeds
        del self.positions[symbol]
        
        logger.debug(f"CLOSE: {symbol} @ {price:.2f}, PnL: {pnl:.2f} ({return_pct:.1f}%)")
    
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
            
            # Composite score
            # Trend following: positive momentum = buy
            momentum_score = (mom_5 * 2 + mom_20) * 3  # Scale up
            
            # Mean reversion: if trending up but oversold, stronger buy
            if trend > 0 and deviation < -0.05:
                momentum_score += 0.3  # Oversold in uptrend = buy
            elif trend < 0 and deviation > 0.05:
                momentum_score -= 0.3  # Overbought in downtrend = sell
            
            # Clamp to [-1, 1] and then scale to threshold range
            score = max(-1, min(1, momentum_score))
            
            # Determine signal
            if score > 0.1:
                signal = 'BUY'
            elif score < -0.1:
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

