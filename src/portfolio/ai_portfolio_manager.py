"""
AI Portfolio Manager for MetaQuant Nigeria.
Autonomous trading agent with target return and risk management.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from .risk_manager import RiskManager, RiskLimits

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"


@dataclass
class TradeRecommendation:
    """AI-generated trade recommendation."""
    symbol: str
    action: TradeAction
    shares: int
    price: float
    value: float
    conviction: float  # 0-1
    stop_loss: float
    take_profit: float
    reasoning: str
    signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    """Configuration for AI Portfolio Manager."""
    capital: float = 10_000_000  # ₦10M
    target_return_pct: float = 0.25  # 25% annual
    max_drawdown_pct: float = 0.10  # 10%
    risk_per_trade_pct: float = 0.02  # 2%
    max_positions: int = 15
    rebalance_threshold: float = 0.10  # 10% drift


class AIPortfolioManager:
    """
    Autonomous portfolio manager using ML and AI signals.
    
    Features:
    - Target return optimization
    - Risk-adjusted position sizing
    - Multi-signal integration (ML, Flow, Fundamentals, Intel)
    - Automatic rebalancing
    - Drawdown protection
    """
    
    def __init__(
        self,
        config: PortfolioConfig,
        db=None,
        ml_engine=None
    ):
        """
        Initialize AI Portfolio Manager.
        
        Args:
            config: Portfolio configuration
            db: Database manager for data access
            ml_engine: ML engine for predictions
        """
        self.config = config
        self.db = db
        self.ml_engine = ml_engine
        
        # Risk manager
        self.risk_manager = RiskManager(
            capital=config.capital,
            limits=RiskLimits(
                max_drawdown_pct=config.max_drawdown_pct,
                risk_per_trade_pct=config.risk_per_trade_pct
            )
        )
        
        # Portfolio state
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.cash = config.capital
        self.equity = config.capital
        self.equity_history: List[Dict] = []
        self.trades: List[Dict] = []
        self.recommendations: List[TradeRecommendation] = []
        
        # Performance tracking
        self.start_date = datetime.now()
        self.total_return = 0.0
        self.annualized_return = 0.0
        
        # Load persisted state if available
        self._load_state()
        
        logger.info(f"AI Portfolio Manager initialized: ₦{config.capital:,.0f}, "
                    f"Target: {config.target_return_pct:.0%}")
    
    def _save_state(self):
        """Save portfolio state to database."""
        if not self.db:
            return
        
        try:
            # Save positions
            self.db.conn.execute("DELETE FROM ai_portfolio_positions")
            for symbol, pos in self.positions.items():
                self.db.conn.execute("""
                    INSERT INTO ai_portfolio_positions (symbol, shares, entry_price, entry_date, stop_loss, take_profit, sector)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [symbol, pos['shares'], pos['entry_price'], pos['entry_date'], 
                      pos.get('stop_loss'), pos.get('take_profit'), pos.get('sector', 'Unknown')])
            
            # Save state
            import json
            self.db.conn.execute("DELETE FROM ai_portfolio_state")
            self.db.conn.execute("""
                INSERT INTO ai_portfolio_state (id, cash, equity, config, start_date)
                VALUES (1, ?, ?, ?, ?)
            """, [self.cash, self.equity, json.dumps({
                'capital': self.config.capital,
                'target_return_pct': self.config.target_return_pct
            }), self.start_date])
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        """Load portfolio state from database."""
        if not self.db:
            return
        
        try:
            # Load positions
            positions = self.db.conn.execute("SELECT * FROM ai_portfolio_positions").fetchall()
            if positions:
                cols = [desc[0] for desc in self.db.conn.description]
                for row in positions:
                    pos = dict(zip(cols, row))
                    self.positions[pos['symbol']] = {
                        'shares': pos['shares'],
                        'entry_price': pos['entry_price'],
                        'entry_date': pos['entry_date'] if isinstance(pos['entry_date'], datetime) else datetime.now(),
                        'stop_loss': pos['stop_loss'],
                        'take_profit': pos['take_profit'],
                        'sector': pos.get('sector', 'Unknown')
                    }
            
            # Load state
            state = self.db.conn.execute("SELECT * FROM ai_portfolio_state WHERE id = 1").fetchone()
            if state:
                cols = [desc[0] for desc in self.db.conn.description]
                s = dict(zip(cols, state))
                self.cash = s['cash']
                self.equity = s['equity']
                if s.get('start_date'):
                    self.start_date = s['start_date'] if isinstance(s['start_date'], datetime) else datetime.now()
            
            # Load trades
            trades = self.db.conn.execute("SELECT * FROM ai_portfolio_trades ORDER BY trade_date DESC LIMIT 100").fetchall()
            if trades:
                cols = [desc[0] for desc in self.db.conn.description]
                for row in trades:
                    t = dict(zip(cols, row))
                    self.trades.append({
                        'date': t['trade_date'] if isinstance(t['trade_date'], datetime) else datetime.now(),
                        'symbol': t['symbol'],
                        'action': t['action'],
                        'shares': t['shares'],
                        'price': t['price'],
                        'entry_price': t.get('entry_price', t['price']),
                        'exit_price': t['price'],
                        'value': t['value'],
                        'pnl': t.get('pnl', 0),
                        'reasoning': t.get('reasoning', '')
                    })
                self.trades = list(reversed(self.trades))  # Chronological
            
            if self.positions:
                logger.info(f"Loaded {len(self.positions)} positions, {len(self.trades)} trades from database")
            
        except Exception as e:
            logger.debug(f"No previous state to load: {e}")
    
    def _save_trade(self, trade: Dict):
        """Save a single trade to database."""
        if not self.db:
            return
        try:
            self.db.conn.execute("""
                INSERT INTO ai_portfolio_trades (trade_date, symbol, action, shares, price, entry_price, value, pnl, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [trade['date'], trade['symbol'], trade['action'], trade['shares'], 
                  trade['price'], trade.get('entry_price', trade['price']), 
                  trade['value'], trade.get('pnl', 0), trade.get('reasoning', '')])
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
    
    def analyze_opportunities(
        self,
        price_data: Dict[str, pd.DataFrame],
        signal_data: Optional[Dict] = None
    ) -> List[TradeRecommendation]:
        """
        Analyze all opportunities and generate recommendations.
        
        Args:
            price_data: Dict of symbol -> DataFrame with OHLCV
            signal_data: Optional external signals
            
        Returns:
            List of trade recommendations
        """
        recommendations = []
        
        for symbol, df in price_data.items():
            if df.empty or len(df) < 25:
                continue
            
            try:
                rec = self._analyze_stock(symbol, df, signal_data)
                if rec and rec.action != TradeAction.HOLD:
                    recommendations.append(rec)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
        
        # Sort by conviction
        recommendations.sort(key=lambda r: r.conviction, reverse=True)
        
        self.recommendations = recommendations
        return recommendations
    
    def _analyze_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        signal_data: Optional[Dict]
    ) -> Optional[TradeRecommendation]:
        """Analyze a single stock for trade opportunity."""
        # Convert to float
        close = pd.to_numeric(df['close'], errors='coerce').astype(float)
        current_price = float(close.iloc[-1])
        
        if current_price <= 0:
            return None
        
        # Calculate signals
        signals = {}
        
        # Momentum
        if len(close) >= 20:
            mom_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            mom_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            signals['momentum_5d'] = float(mom_5)
            signals['momentum_20d'] = float(mom_20)
        
        # Volatility
        returns = close.pct_change().dropna()
        volatility = float(returns.std()) if len(returns) > 10 else 0.02
        signals['volatility'] = volatility
        
        # Trend
        if len(close) >= 50:
            ma_20 = close.tail(20).mean()
            ma_50 = close.tail(50).mean()
            signals['trend'] = 1 if ma_20 > ma_50 else -1
        else:
            signals['trend'] = 0
        
        # ML score if available
        if self.ml_engine and hasattr(self.ml_engine, 'predict'):
            try:
                ml_result = self.ml_engine.predict(symbol)
                signals['ml_score'] = ml_result.get('score', 0)
            except:
                signals['ml_score'] = 0
        
        # Composite score
        score = self._calculate_composite_score(signals)
        signals['composite'] = score
        
        # Determine action
        if symbol in self.positions:
            # Existing position - check for sell
            pos = self.positions[symbol]
            pos_return = (current_price - pos['entry_price']) / pos['entry_price']
            
            if score < -0.2 or pos_return < -0.05:  # Sell signal or stop hit
                return TradeRecommendation(
                    symbol=symbol,
                    action=TradeAction.SELL,
                    shares=pos['shares'],
                    price=current_price,
                    value=pos['shares'] * current_price,
                    conviction=abs(score),
                    stop_loss=0,
                    take_profit=0,
                    reasoning=f"Sell signal: score={score:.2f}, return={pos_return:.1%}",
                    signals=signals
                )
        else:
            # New position - check for buy
            if score > 0.15 and len(self.positions) < self.config.max_positions:
                # Position sizing
                stop_loss_pct = max(0.03, min(0.10, volatility * 2))
                take_profit_pct = stop_loss_pct * 2.5  # 2.5:1 reward/risk
                
                shares = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    entry_price=current_price,
                    stop_loss_pct=stop_loss_pct,
                    conviction=abs(score),
                    volatility=volatility
                )
                
                if shares > 0:
                    value = shares * current_price
                    return TradeRecommendation(
                        symbol=symbol,
                        action=TradeAction.BUY,
                        shares=shares,
                        price=current_price,
                        value=value,
                        conviction=abs(score),
                        stop_loss=current_price * (1 - stop_loss_pct),
                        take_profit=current_price * (1 + take_profit_pct),
                        reasoning=f"Buy signal: score={score:.2f}, momentum={signals.get('momentum_20d', 0):.1%}",
                        signals=signals
                    )
        
        return None
    
    def _calculate_composite_score(self, signals: Dict) -> float:
        """Calculate composite score from all signals."""
        score = 0.0
        
        # Momentum (40%)
        mom_5 = signals.get('momentum_5d', 0)
        mom_20 = signals.get('momentum_20d', 0)
        if mom_5 > 0.02 and mom_20 > 0.03:
            score += 0.4 * min(1, (mom_5 + mom_20) * 2)
        elif mom_5 < -0.02 and mom_20 < -0.03:
            score -= 0.4 * min(1, abs(mom_5 + mom_20) * 2)
        
        # Trend (20%)
        trend = signals.get('trend', 0)
        score += 0.2 * trend
        
        # ML Score (30%)
        ml = signals.get('ml_score', 0)
        score += 0.3 * ml
        
        # Volatility adjustment (10%)
        vol = signals.get('volatility', 0.02)
        if vol > 0.04:  # High vol = reduce score
            score *= 0.8
        
        return max(-1, min(1, score))
    
    def execute_recommendation(self, rec: TradeRecommendation) -> bool:
        """
        Execute a trade recommendation.
        
        Args:
            rec: Trade recommendation to execute
            
        Returns:
            True if executed successfully
        """
        if rec.action == TradeAction.BUY:
            cost = rec.shares * rec.price
            if cost > self.cash:
                logger.warning(f"Insufficient cash for {rec.symbol}")
                return False
            
            self.cash -= cost
            self.positions[rec.symbol] = {
                'shares': rec.shares,
                'entry_price': rec.price,
                'entry_date': datetime.now(),
                'stop_loss': rec.stop_loss,
                'take_profit': rec.take_profit,
                'sector': 'Unknown'  # Would get from stock data
            }
            
            self.risk_manager.add_position(rec.symbol, 'Unknown', cost)
            
            trade = {
                'date': datetime.now(),
                'symbol': rec.symbol,
                'action': 'BUY',
                'shares': rec.shares,
                'price': rec.price,
                'entry_price': rec.price,
                'value': cost,
                'reasoning': rec.reasoning
            }
            self.trades.append(trade)
            self._save_trade(trade)
            self._save_state()
            
            logger.info(f"BOUGHT {rec.shares} {rec.symbol} @ ₦{rec.price:,.2f}")
            return True
            
        elif rec.action == TradeAction.SELL:
            if rec.symbol not in self.positions:
                return False
            
            pos = self.positions[rec.symbol]
            proceeds = pos['shares'] * rec.price
            pnl = proceeds - (pos['shares'] * pos['entry_price'])
            
            self.cash += proceeds
            self.risk_manager.remove_position(rec.symbol)
            del self.positions[rec.symbol]
            
            trade = {
                'date': datetime.now(),
                'symbol': rec.symbol,
                'action': 'SELL',
                'shares': pos['shares'],
                'price': rec.price,
                'entry_price': pos['entry_price'],
                'exit_price': rec.price,
                'value': proceeds,
                'pnl': pnl,
                'reasoning': rec.reasoning
            }
            self.trades.append(trade)
            self._save_trade(trade)
            self._save_state()
            
            logger.info(f"SOLD {pos['shares']} {rec.symbol} @ ₦{rec.price:,.2f}, PnL: ₦{pnl:,.0f}")
            return True
        
        return False
    
    def update_portfolio(self, price_data: Dict[str, pd.DataFrame]):
        """
        Update portfolio valuations and check risk limits.
        
        Args:
            price_data: Current prices
        """
        # Calculate current equity
        positions_value = 0
        for symbol, pos in self.positions.items():
            if symbol in price_data and not price_data[symbol].empty:
                current_price = float(price_data[symbol]['close'].iloc[-1])
                pos['current_price'] = current_price
                pos['current_value'] = pos['shares'] * current_price
                pos['pnl'] = (current_price - pos['entry_price']) * pos['shares']
                pos['return_pct'] = (current_price - pos['entry_price']) / pos['entry_price']
                positions_value += pos['current_value']
        
        self.equity = self.cash + positions_value
        
        # Update risk manager
        risk_status = self.risk_manager.update_equity(self.equity)
        
        # Record equity history
        self.equity_history.append({
            'date': datetime.now(),
            'equity': self.equity,
            'cash': self.cash,
            'positions_value': positions_value,
            'positions': len(self.positions),
            'drawdown': risk_status['drawdown_pct']
        })
        
        # Calculate returns
        self.total_return = (self.equity - self.config.capital) / self.config.capital
        
        # Annualize
        days_elapsed = (datetime.now() - self.start_date).days
        if days_elapsed > 30:
            self.annualized_return = (1 + self.total_return) ** (365 / days_elapsed) - 1
        else:
            self.annualized_return = self.total_return
    
    def get_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            'capital': self.config.capital,
            'equity': self.equity,
            'cash': self.cash,
            'positions_value': self.equity - self.cash,
            'total_return_pct': round(self.total_return * 100, 2),
            'annualized_return_pct': round(self.annualized_return * 100, 2),
            'target_return_pct': self.config.target_return_pct * 100,
            'on_target': self.annualized_return >= self.config.target_return_pct,
            'positions': len(self.positions),
            'max_positions': self.config.max_positions,
            'trades': len(self.trades),
            'risk_status': self.risk_manager.get_risk_status(),
            'recommendations': len(self.recommendations)
        }
    
    def get_holdings(self) -> List[Dict]:
        """Get current holdings with P&L."""
        holdings = []
        for symbol, pos in self.positions.items():
            holdings.append({
                'symbol': symbol,
                'shares': pos['shares'],
                'entry_price': pos['entry_price'],
                'current_price': pos.get('current_price', pos['entry_price']),
                'cost': pos['shares'] * pos['entry_price'],
                'value': pos.get('current_value', pos['shares'] * pos['entry_price']),
                'pnl': pos.get('pnl', 0),
                'return_pct': pos.get('return_pct', 0) * 100,
                'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                'stop_loss': pos.get('stop_loss', 0),
                'take_profit': pos.get('take_profit', 0)
            })
        
        holdings.sort(key=lambda h: h['pnl'], reverse=True)
        return holdings
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_history:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_history)
