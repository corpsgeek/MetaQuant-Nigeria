"""
Trade Executor - Execute paper trades based on signals.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from .signal_generator import TradingSignal
from .portfolio_book import PortfolioBookManager

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Executes paper trades based on generated signals.
    
    Features:
    - Opens positions on BUY signals
    - Closes positions on SELL signals
    - Manages stop loss and take profit
    - Enforces position limits (max 15)
    - Enforces minimum holding period (3 days)
    - Sends notifications on trade events
    """
    
    def __init__(self, portfolio_manager: PortfolioBookManager,
                 trading_tables, notifier=None, max_positions: int = 15,
                 min_hold_days: int = 3):
        """
        Initialize the trade executor.
        
        Args:
            portfolio_manager: PortfolioBookManager instance
            trading_tables: TradingTables for signal/trade logging
            notifier: Optional notification handler
            max_positions: Maximum concurrent positions (default 15)
            min_hold_days: Minimum days to hold before selling (default 3)
        """
        self.portfolio = portfolio_manager
        self.tables = trading_tables
        self.notifier = notifier
        self.max_positions = max_positions
        self.min_hold_days = min_hold_days
    
    def execute_signals(self, signals: List[TradingSignal],
                       current_prices: Dict[str, float],
                       book_id: int = None) -> Dict[str, Any]:
        """
        Execute trades based on signals.
        
        Args:
            signals: List of TradingSignal objects
            current_prices: Dict of symbol -> current price
            book_id: Portfolio book ID (uses active if not specified)
        
        Returns:
            Dict with opened, closed, and skipped trade info
        """
        book_id = book_id or self.portfolio.active_book_id
        
        opened = []
        closed = []
        skipped = []
        
        # First, process SELL signals for existing positions
        for signal in signals:
            if signal.signal == 'SELL':
                result = self._process_sell_signal(signal, current_prices, book_id)
                if result:
                    closed.append(result)
        
        # Then, process BUY signals
        buy_signals = [s for s in signals if s.signal == 'BUY']
        buy_signals.sort(key=lambda s: s.score, reverse=True)  # Best first
        
        for signal in buy_signals:
            # Check position limit
            current_positions = self.portfolio.get_position_count(book_id)
            if current_positions >= self.max_positions:
                skipped.append({
                    'symbol': signal.symbol,
                    'reason': 'MAX_POSITIONS'
                })
                break
            
            result = self._process_buy_signal(signal, current_prices, book_id)
            if result:
                opened.append(result)
            else:
                skipped.append({
                    'symbol': signal.symbol,
                    'reason': 'EXECUTION_FAILED'
                })
        
        # Check for stop loss / take profit triggers
        triggered = self.portfolio.check_stop_loss_take_profit(
            current_prices, book_id
        )
        for t in triggered:
            closed.append(t)
            self._notify_trade_close(t)
        
        # Log summary
        logger.info(f"Execution complete: {len(opened)} opened, "
                   f"{len(closed)} closed, {len(skipped)} skipped")
        
        return {
            'opened': opened,
            'closed': closed,
            'skipped': skipped,
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_buy_signal(self, signal: TradingSignal,
                           current_prices: Dict[str, float],
                           book_id: int) -> Optional[Dict]:
        """Process a BUY signal."""
        symbol = signal.symbol
        
        # Check if already have position
        if self.portfolio.has_position(symbol, book_id):
            return None
        
        # Get current price
        price = current_prices.get(symbol, signal.current_price)
        
        # Get strategy parameters
        strategy = signal.strategy or {}
        stop_loss_pct = strategy.get('optimal_stop_loss', 0.05)
        take_profit_pct = strategy.get('optimal_take_profit', 0.15)
        
        # Open position
        trade_id = self.portfolio.open_position(
            symbol=symbol,
            current_price=price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            entry_score=signal.score,
            entry_attribution=signal.attribution,
            book_id=book_id
        )
        
        if trade_id:
            result = {
                'trade_id': trade_id,
                'symbol': symbol,
                'action': 'BUY',
                'price': price,
                'score': signal.score,
                'stop_loss': price * (1 - stop_loss_pct),
                'take_profit': price * (1 + take_profit_pct)
            }
            
            # Mark signal as acted upon
            # (This would require the signal_id from log)
            
            # Send notification
            self._notify_trade_open(result)
            
            return result
        
        return None
    
    def _process_sell_signal(self, signal: TradingSignal,
                            current_prices: Dict[str, float],
                            book_id: int) -> Optional[Dict]:
        """Process a SELL signal."""
        symbol = signal.symbol
        
        # Check if have position to sell
        positions = self.portfolio.get_open_positions(book_id)
        position = next((p for p in positions if p.symbol == symbol), None)
        
        if not position:
            return None
        
        # Check minimum holding period
        if position.days_held < self.min_hold_days:
            logger.debug(f"Skipping SELL for {symbol}: only {position.days_held} days, "
                        f"need {self.min_hold_days}")
            return None
        
        # Get current price
        price = current_prices.get(symbol, signal.current_price)
        
        # Close position
        success = self.portfolio.close_position(
            trade_id=position.trade_id,
            current_price=price,
            exit_reason='SIGNAL',
            exit_score=signal.score,
            exit_attribution=signal.attribution
        )
        
        if success:
            pnl = (price - position.entry_price) * position.quantity
            pnl_pct = (price - position.entry_price) / position.entry_price * 100
            
            result = {
                'trade_id': position.trade_id,
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'holding_days': position.days_held,
                'exit_reason': 'SIGNAL'
            }
            
            self._notify_trade_close(result)
            
            return result
        
        return None
    
    def close_all_positions(self, current_prices: Dict[str, float],
                           book_id: int = None, reason: str = "MANUAL") -> List[Dict]:
        """
        Close all open positions.
        
        Args:
            current_prices: Dict of symbol -> current price
            book_id: Portfolio book ID
            reason: Exit reason (MANUAL, EMERGENCY, etc.)
        
        Returns:
            List of closed trade info
        """
        book_id = book_id or self.portfolio.active_book_id
        positions = self.portfolio.get_open_positions(book_id)
        
        closed = []
        for position in positions:
            price = current_prices.get(position.symbol, position.current_price)
            
            success = self.portfolio.close_position(
                trade_id=position.trade_id,
                current_price=price,
                exit_reason=reason
            )
            
            if success:
                pnl = (price - position.entry_price) * position.quantity
                closed.append({
                    'trade_id': position.trade_id,
                    'symbol': position.symbol,
                    'pnl': pnl,
                    'exit_reason': reason
                })
        
        logger.info(f"Closed {len(closed)} positions ({reason})")
        return closed
    
    def run_daily_execution(self, signals: List[TradingSignal],
                           current_prices: Dict[str, float],
                           book_id: int = None) -> Dict[str, Any]:
        """
        Run the full daily execution cycle.
        
        1. Check stop loss / take profit on existing positions
        2. Execute SELL signals (respect min hold period)
        3. Execute BUY signals (respect max positions)
        4. Generate summary notifications
        """
        book_id = book_id or self.portfolio.active_book_id
        
        # Execute signals
        result = self.execute_signals(signals, current_prices, book_id)
        
        # Generate daily summary
        summary = self.portfolio.get_portfolio_summary(book_id)
        summary['execution'] = result
        
        # Send daily summary notification
        self._notify_daily_summary(summary)
        
        return summary
    
    # ==================== Notifications ====================
    
    def _notify_trade_open(self, trade: Dict):
        """Send notification for opened trade."""
        if not self.notifier:
            return
        
        try:
            message = (
                f"🟢 OPENED: {trade['symbol']}\n"
                f"Price: ₦{trade['price']:,.2f}\n"
                f"Score: {trade['score']:.2f}\n"
                f"SL: ₦{trade['stop_loss']:,.2f} | TP: ₦{trade['take_profit']:,.2f}"
            )
            self.notifier.send('trade_open', message, trade)
        except Exception as e:
            logger.debug(f"Notification failed: {e}")
    
    def _notify_trade_close(self, trade: Dict):
        """Send notification for closed trade."""
        if not self.notifier:
            return
        
        try:
            pnl = trade.get('pnl', 0)
            pnl_pct = trade.get('pnl_pct', 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            
            message = (
                f"{emoji} CLOSED: {trade['symbol']}\n"
                f"P&L: ₦{pnl:,.0f} ({pnl_pct:+.2f}%)\n"
                f"Reason: {trade.get('exit_reason', 'SIGNAL')}"
            )
            self.notifier.send('trade_close', message, trade)
        except Exception as e:
            logger.debug(f"Notification failed: {e}")
    
    def _notify_daily_summary(self, summary: Dict):
        """Send daily summary notification."""
        if not self.notifier:
            return
        
        try:
            execution = summary.get('execution', {})
            opened = len(execution.get('opened', []))
            closed = len(execution.get('closed', []))
            
            message = (
                f"📊 DAILY SUMMARY\n"
                f"Portfolio: {summary.get('book_name', 'Default')}\n"
                f"Value: ₦{summary.get('total_value', 0):,.0f}\n"
                f"Return: {summary.get('total_return_pct', 0):+.2f}%\n"
                f"Open: {summary.get('open_positions', 0)} positions\n"
                f"Today: {opened} opened, {closed} closed"
            )
            self.notifier.send('daily_summary', message, summary)
        except Exception as e:
            logger.debug(f"Notification failed: {e}")
