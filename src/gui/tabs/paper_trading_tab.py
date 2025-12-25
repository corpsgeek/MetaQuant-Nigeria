"""
Paper Trading Tab - GUI for paper trading system.

Displays:
- Portfolio book selector
- Active positions with live P&L
- Today's signals (BUY/SELL recommendations)
- Trade history
- Performance dashboard
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import theme
from ..theme import COLORS

logger = logging.getLogger(__name__)


class PaperTradingTab:
    """Paper trading tab for the MetaQuant GUI."""
    
    def __init__(self, parent, db, ml_engine=None, price_provider=None):
        """
        Initialize the paper trading tab.
        
        Args:
            parent: Parent tkinter widget
            db: Database manager
            ml_engine: ML engine for predictions
            price_provider: Callable that returns current prices dict
        """
        self.parent = parent
        self.db = db
        self.ml_engine = ml_engine
        self.price_provider = price_provider
        
        # Trading components (initialized lazily)
        self._trading_tables = None
        self._portfolio_manager = None
        self._signal_generator = None
        self._trade_executor = None
        self._strategy_optimizer = None
        
        # State
        self._current_book_id = None
        self._current_signals = []
        self._price_data = {}
        
        # Build UI
        self._build_ui()
        
        # Initialize trading system
        self._init_trading_system()
    
    def _init_trading_system(self):
        """Initialize trading components."""
        try:
            from ...trading import (
                TradingTables, PortfolioBookManager,
                SignalGenerator, TradeExecutor, StrategyOptimizer
            )
            from ...backtesting import BacktestEngine
            
            # Initialize tables
            self._trading_tables = TradingTables(self.db)
            
            # Initialize portfolio manager
            self._portfolio_manager = PortfolioBookManager(
                self._trading_tables,
                price_provider=self._get_current_price
            )
            
            # Initialize signal generator
            self._signal_generator = SignalGenerator(
                self._trading_tables, self.db, self.ml_engine
            )
            
            # Initialize trade executor
            self._trade_executor = TradeExecutor(
                self._portfolio_manager, self._trading_tables
            )
            
            # Initialize strategy optimizer
            self._strategy_optimizer = StrategyOptimizer(
                BacktestEngine, self._trading_tables, self.db, self.ml_engine
            )
            
            # Load books
            self._load_portfolio_books()
            
            # Update UI
            self._update_all()
            
            logger.info("Paper trading system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {e}")
            self.status_label.config(text=f"Error: {e}", foreground=COLORS['loss'])
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        if self.price_provider:
            try:
                prices = self.price_provider()
                return prices.get(symbol)
            except:
                pass
        return None
    
    def _build_ui(self):
        """Build the UI layout."""
        # Main container - this is what gets added to notebook
        self.frame = ttk.Frame(self.parent)
        
        # Top bar - Portfolio selector and actions
        self._create_top_bar(self.frame)
        
        # Main content - split into left and right panels
        content_frame = ttk.Frame(self.frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left panel - Positions and signals
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self._create_positions_panel(left_frame)
        self._create_signals_panel(left_frame)
        
        # Right panel - Performance and history
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self._create_performance_panel(right_frame)
        self._create_history_panel(right_frame)
        
        # Status bar
        self.status_label = ttk.Label(self.frame, text="Ready", foreground=COLORS['text_primary'])
        self.status_label.pack(fill=tk.X, pady=5)
    
    def _create_top_bar(self, parent):
        """Create top bar with portfolio selector and actions."""
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Portfolio selector
        ttk.Label(top_frame, text="Portfolio Book:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.book_combo = ttk.Combobox(top_frame, width=25, state='readonly')
        self.book_combo.pack(side=tk.LEFT, padx=5)
        self.book_combo.bind('<<ComboboxSelected>>', self._on_book_change)
        
        # Action buttons
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="📊 Generate Signals", 
                  command=self._generate_signals).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="▶️ Execute Trades",
                  command=self._execute_trades).pack(side=tk.LEFT, padx=2)
        
        # Load Data button - loads price data for signal generation
        ttk.Button(btn_frame, text="📥 Load Data",
                  command=self._load_price_data_and_update).pack(side=tk.LEFT, padx=2)
        
        # Optimize button - generates per-stock optimal parameters
        ttk.Button(btn_frame, text="🔧 Optimize",
                  command=self._quick_optimize_strategies).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="➕ New Book",
                  command=self._create_new_book).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="🔄 Refresh",
                  command=self._update_all).pack(side=tk.LEFT, padx=2)
    
    def _create_positions_panel(self, parent):
        """Create open positions panel."""
        pos_frame = ttk.LabelFrame(parent, text="📈 Open Positions (0)")
        pos_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.positions_label = pos_frame  # For updating title
        
        columns = ('Symbol', 'Entry', 'Current', 'P&L', 'P&L%', 'Days', 'SL', 'TP')
        self.positions_tree = ttk.Treeview(pos_frame, columns=columns, 
                                           show='headings', height=6)
        
        col_widths = {'Symbol': 70, 'Entry': 70, 'Current': 70, 'P&L': 80, 
                     'P&L%': 60, 'Days': 40, 'SL': 70, 'TP': 70}
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=col_widths.get(col, 60))
        
        scrollbar = ttk.Scrollbar(pos_frame, orient=tk.VERTICAL, 
                                  command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Position colors
        self.positions_tree.tag_configure('profit', foreground=COLORS['gain'])
        self.positions_tree.tag_configure('loss', foreground=COLORS['loss'])
        
        # Context menu for positions
        self._create_position_context_menu()
    
    def _create_position_context_menu(self):
        """Create right-click context menu for positions."""
        self.pos_menu = tk.Menu(self.parent, tearoff=0)
        self.pos_menu.add_command(label="Close Position", command=self._close_selected_position)
        
        self.positions_tree.bind("<Button-3>", self._show_position_menu)
    
    def _show_position_menu(self, event):
        """Show position context menu."""
        item = self.positions_tree.identify_row(event.y)
        if item:
            self.positions_tree.selection_set(item)
            self.pos_menu.post(event.x_root, event.y_root)
    
    def _create_signals_panel(self, parent):
        """Create signals panel."""
        sig_frame = ttk.LabelFrame(parent, text="📡 Today's Signals")
        sig_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Rank', 'Symbol', 'Signal', 'Score', 'Price', 'SL', 'TP')
        self.signals_tree = ttk.Treeview(sig_frame, columns=columns, 
                                         show='headings', height=8)
        
        col_widths = {'Rank': 40, 'Symbol': 70, 'Signal': 50, 'Score': 50, 
                     'Price': 70, 'SL': 70, 'TP': 70}
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=col_widths.get(col, 60))
        
        scrollbar = ttk.Scrollbar(sig_frame, orient=tk.VERTICAL, 
                                  command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=scrollbar.set)
        
        self.signals_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Signal colors
        self.signals_tree.tag_configure('buy', foreground=COLORS['gain'])
        self.signals_tree.tag_configure('sell', foreground=COLORS['loss'])
        self.signals_tree.tag_configure('hold', foreground=COLORS['warning'])
    
    def _create_performance_panel(self, parent):
        """Create performance metrics panel."""
        perf_frame = ttk.LabelFrame(parent, text="📊 Performance")
        perf_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Metrics grid
        self.perf_metrics = {}
        metrics = [
            ('total_value', 'Portfolio Value'),
            ('total_return_pct', 'Total Return'),
            ('unrealized_pnl', 'Unrealized P&L'),
            ('realized_pnl', 'Realized P&L'),
            ('win_rate', 'Win Rate'),
            ('total_trades', 'Total Trades')
        ]
        
        for i, (key, label) in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            frame = ttk.Frame(perf_frame)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky='w')
            
            ttk.Label(frame, text=label, font=('Helvetica', 9)).pack(anchor='w')
            value_label = ttk.Label(frame, text='-', font=('Helvetica', 12, 'bold'))
            value_label.pack(anchor='w')
            self.perf_metrics[key] = value_label
    
    def _create_history_panel(self, parent):
        """Create trade history panel."""
        hist_frame = ttk.LabelFrame(parent, text="📋 Trade History")
        hist_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Date', 'Symbol', 'P&L', 'Return', 'Days', 'Reason')
        self.history_tree = ttk.Treeview(hist_frame, columns=columns, 
                                         show='headings', height=8)
        
        col_widths = {'Date': 75, 'Symbol': 70, 'P&L': 80, 'Return': 60, 
                     'Days': 40, 'Reason': 80}
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=col_widths.get(col, 60))
        
        scrollbar = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, 
                                  command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # History colors
        self.history_tree.tag_configure('profit', foreground=COLORS['gain'])
        self.history_tree.tag_configure('loss', foreground=COLORS['loss'])
    
    # ==================== Data Loading ====================
    
    def _load_portfolio_books(self):
        """Load portfolio books into combo box."""
        if not self._portfolio_manager:
            return
        
        books = self._portfolio_manager.get_books()
        self.book_combo['values'] = [f"{b['name']} (₦{b['current_capital']:,.0f})" 
                                      for b in books]
        
        if books:
            self.book_combo.current(0)
            self._current_book_id = books[0]['id']
            self._portfolio_manager.set_active_book(self._current_book_id)
    
    def _on_book_change(self, event):
        """Handle portfolio book selection change."""
        if not self._portfolio_manager:
            return
        
        idx = self.book_combo.current()
        books = self._portfolio_manager.get_books()
        
        if idx >= 0 and idx < len(books):
            self._current_book_id = books[idx]['id']
            self._portfolio_manager.set_active_book(self._current_book_id)
            self._update_all()
    
    def _update_all(self):
        """Update all UI components."""
        self._update_positions()
        self._update_signals()
        self._update_performance()
        self._update_history()
    
    def _update_positions(self):
        """Update positions display."""
        if not self._portfolio_manager:
            return
        
        # Clear existing
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        positions = self._portfolio_manager.get_open_positions(self._current_book_id)
        
        for pos in positions:
            tag = 'profit' if pos.unrealized_pnl >= 0 else 'loss'
            self.positions_tree.insert('', 'end', values=(
                pos.symbol,
                f"₦{pos.entry_price:,.2f}",
                f"₦{pos.current_price:,.2f}",
                f"₦{pos.unrealized_pnl:,.0f}",
                f"{pos.unrealized_pnl_pct:+.1f}%",
                pos.days_held,
                f"₦{pos.stop_loss:,.2f}",
                f"₦{pos.take_profit:,.2f}"
            ), tags=(tag,))
        
        # Update title
        self.positions_label.config(text=f"📈 Open Positions ({len(positions)})")
    
    def _update_signals(self):
        """Update signals display."""
        # Clear existing
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)
        
        for sig in self._current_signals[:30]:  # Top 30
            tag = sig.signal.lower()
            self.signals_tree.insert('', 'end', values=(
                sig.rank,
                sig.symbol,
                sig.signal,
                f"{sig.score:+.2f}",
                f"₦{sig.current_price:,.2f}",
                f"₦{sig.stop_loss:,.2f}",
                f"₦{sig.take_profit:,.2f}"
            ), tags=(tag,))
    
    def _update_performance(self):
        """Update performance metrics display."""
        if not self._portfolio_manager:
            return
        
        summary = self._portfolio_manager.get_portfolio_summary(self._current_book_id)
        
        # Format and display metrics
        self.perf_metrics['total_value'].config(
            text=f"₦{summary.get('total_value', 0):,.0f}"
        )
        
        ret = summary.get('total_return_pct', 0)
        self.perf_metrics['total_return_pct'].config(
            text=f"{ret:+.2f}%",
            foreground=COLORS['gain'] if ret >= 0 else COLORS['loss']
        )
        
        unrealized = summary.get('unrealized_pnl', 0)
        self.perf_metrics['unrealized_pnl'].config(
            text=f"₦{unrealized:,.0f}",
            foreground=COLORS['gain'] if unrealized >= 0 else COLORS['loss']
        )
        
        realized = summary.get('realized_pnl', 0)
        self.perf_metrics['realized_pnl'].config(
            text=f"₦{realized:,.0f}",
            foreground=COLORS['gain'] if realized >= 0 else COLORS['loss']
        )
        
        self.perf_metrics['win_rate'].config(
            text=f"{summary.get('win_rate', 0):.1f}%"
        )
        
        self.perf_metrics['total_trades'].config(
            text=str(summary.get('total_trades', 0))
        )
    
    def _update_history(self):
        """Update trade history display."""
        if not self._portfolio_manager:
            return
        
        # Clear existing
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        history = self._portfolio_manager.get_trade_history(self._current_book_id, limit=50)
        
        for trade in history:
            pnl = trade.get('pnl', 0)
            tag = 'profit' if pnl >= 0 else 'loss'
            
            self.history_tree.insert('', 'end', values=(
                trade.get('exit_date', '')[:10],
                trade.get('symbol', ''),
                f"₦{pnl:,.0f}",
                f"{trade.get('return_pct', 0):+.1f}%",
                trade.get('holding_days', 0),
                trade.get('exit_reason', '-')
            ), tags=(tag,))
    
    # ==================== Actions ====================
    
    def _generate_signals(self):
        """Generate trading signals."""
        if not self._signal_generator:
            messagebox.showerror("Error", "Signal generator not initialized")
            return
        
        self.status_label.config(text="Loading price data...", foreground=COLORS['warning'])
        self.frame.update()
        
        try:
            # Load price data from database if not already loaded
            if not self._price_data:
                self._load_price_data()
            
            if not self._price_data:
                messagebox.showwarning("Warning", "No price data available in database")
                return
            
            self.status_label.config(text=f"Generating signals for {len(self._price_data)} stocks...")
            self.frame.update()
            
            # Generate signals
            current_prices = {}
            for s, df in self._price_data.items():
                if not df.empty:
                    current_prices[s] = float(df['close'].iloc[-1])
            
            signals = self._signal_generator.generate_signals(
                self._price_data,
                current_prices=current_prices
            )
            
            self._current_signals = signals
            self._update_signals()
            
            buy_count = sum(1 for s in signals if s.signal == 'BUY')
            sell_count = sum(1 for s in signals if s.signal == 'SELL')
            
            self.status_label.config(
                text=f"Generated {len(signals)} signals: {buy_count} BUY, {sell_count} SELL",
                foreground=COLORS['gain']
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            self.status_label.config(text=f"Error: {e}", foreground=COLORS['loss'])
    
    def _load_price_data(self):
        """Load price data from database."""
        import pandas as pd
        
        try:
            # Get all stocks with price data
            stocks = self.db.conn.execute("""
                SELECT DISTINCT s.symbol, s.id 
                FROM stocks s 
                JOIN daily_prices dp ON s.id = dp.stock_id
            """).fetchall()
            
            self._price_data = {}
            
            for symbol, stock_id in stocks:
                # Get daily prices
                prices = self.db.conn.execute("""
                    SELECT date, open, high, low, close, volume
                    FROM daily_prices
                    WHERE stock_id = ?
                    ORDER BY date
                """, [stock_id]).fetchall()
                
                if prices:
                    df = pd.DataFrame(prices, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    self._price_data[symbol] = df
            
            logger.info(f"Loaded price data for {len(self._price_data)} stocks")
            
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
    
    def _load_price_data_and_update(self):
        """Load price data button handler."""
        self.status_label.config(text="Loading price data from database...", foreground=COLORS['warning'])
        self.frame.update()
        
        self._load_price_data()
        
        if self._price_data:
            self.status_label.config(
                text=f"Loaded data for {len(self._price_data)} stocks. Ready to generate signals.",
                foreground=COLORS['gain']
            )
        else:
            self.status_label.config(
                text="No price data found. Run backtest first to populate data.",
                foreground=COLORS['loss']
            )
    
    def _quick_optimize_strategies(self):
        """
        Run backtest-based optimization via subprocess.
        Completely isolates memory from GUI process to prevent segfaults.
        """
        if not self._price_data:
            messagebox.showwarning("Warning", "Load price data first")
            return
        
        if hasattr(self, '_opt_process') and self._opt_process and self._opt_process.poll() is None:
            messagebox.showwarning("Warning", "Optimization already in progress")
            return
        
        stock_count = len(self._price_data)
        
        if not messagebox.askyesno("Run Backtest Optimization?", 
            f"This will run actual backtests on {stock_count} stocks.\n\n"
            "Uses SEPARATE PROCESS to prevent crashes:\n"
            "• Batches of 5 stocks\n"
            "• 2-second delays between batches\n"
            "• Isolated memory space\n\n"
            f"Estimated time: ~{stock_count // 5 * 30}s ({stock_count // 5} batches)\n\n"
            "Continue?"):
            return
        
        import subprocess
        import os
        
        # Get the path to the optimization script
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        script_path = os.path.join(project_root, 'scripts', 'optimize_strategies.py')
        
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Optimization script not found: {script_path}")
            return
        
        self.status_label.config(
            text="Starting optimization subprocess...",
            foreground=COLORS['warning']
        )
        self.frame.update()
        
        # Run optimization script as subprocess
        try:
            self._opt_process = subprocess.Popen(
                ['python', script_path, '--batch-size', '5', '--delay', '2.0'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=project_root
            )
            
            # Start monitoring the process
            self._poll_subprocess_optimization()
            
        except Exception as e:
            logger.error(f"Failed to start optimization: {e}")
            self.status_label.config(text=f"Error: {e}", foreground=COLORS['loss'])
    
    def _poll_subprocess_optimization(self):
        """Poll subprocess optimization progress."""
        if not hasattr(self, '_opt_process') or self._opt_process is None:
            return
        
        # Check if process is still running
        if self._opt_process.poll() is None:
            # Read any available output
            try:
                line = self._opt_process.stdout.readline()
                if line:
                    # Parse progress from log line
                    if 'Processing batch' in line:
                        # Extract batch info
                        self.status_label.config(
                            text=line.strip().split(' - ')[-1][:60],
                            foreground=COLORS['warning']
                        )
                    elif 'Optimized:' in line:
                        self.status_label.config(
                            text=line.strip().split(' - ')[-1],
                            foreground=COLORS['gain']
                        )
            except:
                pass
            
            # Poll again in 500ms
            self.frame.after(500, self._poll_subprocess_optimization)
        else:
            # Process finished
            exit_code = self._opt_process.returncode
            
            if exit_code == 0:
                self.status_label.config(
                    text="Optimization complete! Click Generate Signals to update.",
                    foreground=COLORS['gain']
                )
                # Refresh strategy cache
                if self._signal_generator:
                    self._signal_generator._refresh_strategies(force=True)
            else:
                self.status_label.config(
                    text=f"Optimization failed (exit code {exit_code})",
                    foreground=COLORS['loss']
                )
            
            self._opt_process = None
    
    def _execute_trades(self):
        """Execute trades based on current signals."""
        if not self._trade_executor or not self._current_signals:
            messagebox.showwarning("Warning", "Generate signals first")
            return
        
        if not messagebox.askyesno("Confirm", 
            "Execute trades based on current signals?\n\n"
            f"BUY signals: {sum(1 for s in self._current_signals if s.signal == 'BUY')}\n"
            f"SELL signals: {sum(1 for s in self._current_signals if s.signal == 'SELL')}"):
            return
        
        self.status_label.config(text="Executing trades...", foreground=COLORS['warning'])
        self.parent.update()
        
        try:
            # Get current prices
            current_prices = {}
            for s, df in self._price_data.items():
                if not df.empty:
                    current_prices[s] = float(df['close'].iloc[-1])
            
            # Execute
            result = self._trade_executor.execute_signals(
                self._current_signals,
                current_prices,
                self._current_book_id
            )
            
            # Update UI
            self._update_all()
            
            opened = len(result.get('opened', []))
            closed = len(result.get('closed', []))
            
            self.status_label.config(
                text=f"Executed: {opened} opened, {closed} closed",
                foreground=COLORS['gain']
            )
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            self.status_label.config(text=f"Error: {e}", foreground=COLORS['loss'])
    
    def _optimize_strategies(self):
        """Optimize trading strategies in background thread."""
        if not self._strategy_optimizer:
            messagebox.showerror("Error", "Strategy optimizer not initialized")
            return
        
        if not self._price_data:
            messagebox.showwarning("Warning", "No price data available. Run backtest first to load data.")
            return
        
        if not messagebox.askyesno("Confirm", 
            "Run strategy optimization?\n\n"
            "This will run backtests across multiple time horizons\n"
            "to find optimal parameters for each stock.\n\n"
            "This may take several minutes."):
            return
        
        self.status_label.config(text="Optimizing strategies (running in background)...", foreground=COLORS['warning'])
        
        import threading
        
        def run_optimization():
            try:
                symbols = list(self._price_data.keys())
                total = len(symbols)
                optimized = 0
                
                for i, symbol in enumerate(symbols):
                    if symbol not in self._price_data or self._price_data[symbol].empty:
                        continue
                    
                    try:
                        strategy = self._strategy_optimizer.optimize_stock(symbol, self._price_data)
                        if strategy:
                            self._trading_tables.save_stock_strategy(symbol, strategy)
                            optimized += 1
                    except Exception as e:
                        logger.debug(f"Failed to optimize {symbol}: {e}")
                    
                    # Update status every 10 stocks
                    if (i + 1) % 10 == 0:
                        self.frame.after(0, lambda c=i+1, t=total: 
                            self.status_label.config(text=f"Optimizing {c}/{t} stocks..."))
                
                # Done
                self.frame.after(0, lambda: self.status_label.config(
                    text=f"Optimized {optimized} strategies", 
                    foreground=COLORS['gain']
                ))
                
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                self.frame.after(0, lambda: self.status_label.config(
                    text=f"Error: {e}", 
                    foreground=COLORS['loss']
                ))
        
        # Run in background thread
        thread = threading.Thread(target=run_optimization, daemon=True)
        thread.start()
    
    def _create_new_book(self):
        """Create a new portfolio book."""
        if not self._portfolio_manager:
            return
        
        # Simple dialog
        dialog = tk.Toplevel(self.parent)
        dialog.title("New Portfolio Book")
        dialog.geometry("300x150")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Book Name:").pack(pady=5)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack(pady=5)
        
        ttk.Label(dialog, text="Initial Capital (₦):").pack(pady=5)
        capital_entry = ttk.Entry(dialog, width=30)
        capital_entry.insert(0, "10000000")
        capital_entry.pack(pady=5)
        
        def create():
            name = name_entry.get().strip()
            try:
                capital = float(capital_entry.get())
            except:
                capital = 10_000_000
            
            if name:
                self._portfolio_manager.create_book(name, capital)
                self._load_portfolio_books()
                dialog.destroy()
        
        ttk.Button(dialog, text="Create", command=create).pack(pady=10)
    
    def _close_selected_position(self):
        """Close the selected position."""
        if not self._portfolio_manager:
            return
        
        selection = self.positions_tree.selection()
        if not selection:
            return
        
        item = self.positions_tree.item(selection[0])
        symbol = item['values'][0]
        
        if not messagebox.askyesno("Confirm", f"Close position in {symbol}?"):
            return
        
        positions = self._portfolio_manager.get_open_positions(self._current_book_id)
        position = next((p for p in positions if p.symbol == symbol), None)
        
        if position:
            self._portfolio_manager.close_position(
                position.trade_id, position.current_price, "MANUAL"
            )
            self._update_all()
    
    def set_price_data(self, price_data: Dict):
        """Set price data for signal generation."""
        self._price_data = price_data
