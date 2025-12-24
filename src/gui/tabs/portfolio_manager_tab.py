"""
AI Portfolio Manager Tab for MetaQuant Nigeria.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading

import pandas as pd

from ..theme import COLORS, get_font
from ...database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Try to import portfolio modules
try:
    from ...portfolio import AIPortfolioManager, PortfolioConfig
    PORTFOLIO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Portfolio manager not available: {e}")
    PORTFOLIO_AVAILABLE = False


class PortfolioManagerTab:
    """AI Portfolio Manager Tab with autonomous trading."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager, ml_engine=None):
        self.parent = parent
        self.db = db
        self.ml_engine = ml_engine
        
        # State
        self.manager: Optional[AIPortfolioManager] = None
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.is_running = False
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        if not PORTFOLIO_AVAILABLE:
            self._create_unavailable_ui()
            return
        
        self._create_ui()
        self.frame.after(1000, self._load_data)
    
    def _create_unavailable_ui(self):
        """Show unavailable message."""
        container = ttk.Frame(self.frame)
        container.pack(expand=True)
        ttk.Label(container, text="🤖 AI Portfolio Manager Loading...",
                  font=get_font('heading'), foreground=COLORS['warning']).pack(pady=20)
    
    def _create_ui(self):
        """Create the portfolio manager UI."""
        # Header
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(header, text="🤖 AI Portfolio Manager",
                  font=get_font('title'), foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        # Main content - paned window
        main_pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ========== LEFT: Configuration & Control ==========
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)
        
        # Config
        config_frame = ttk.LabelFrame(left_frame, text="⚙️ Configuration")
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        config_inner = ttk.Frame(config_frame)
        config_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Capital
        row1 = ttk.Frame(config_inner)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Capital (₦):", width=15).pack(side=tk.LEFT)
        self.capital_var = tk.StringVar(value="10,000,000")
        ttk.Entry(row1, textvariable=self.capital_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Target Return
        row2 = ttk.Frame(config_inner)
        row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="Target Return %:", width=15).pack(side=tk.LEFT)
        self.target_var = tk.StringVar(value="25")
        ttk.Entry(row2, textvariable=self.target_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="per year", foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        
        # Max Drawdown
        row3 = ttk.Frame(config_inner)
        row3.pack(fill=tk.X, pady=3)
        ttk.Label(row3, text="Max Drawdown %:", width=15).pack(side=tk.LEFT)
        self.drawdown_var = tk.StringVar(value="10")
        ttk.Entry(row3, textvariable=self.drawdown_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Risk per Trade
        row4 = ttk.Frame(config_inner)
        row4.pack(fill=tk.X, pady=3)
        ttk.Label(row4, text="Risk/Trade %:", width=15).pack(side=tk.LEFT)
        self.risk_var = tk.StringVar(value="2")
        ttk.Entry(row4, textvariable=self.risk_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        btn_frame = ttk.Frame(config_inner)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="▶️ Start Manager", command=self._start_manager)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = ttk.Button(btn_frame, text="🔍 Analyze", command=self._analyze, state='disabled')
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(btn_frame, text="Not started", foreground=COLORS['text_muted'])
        self.status_label.pack(side=tk.LEFT, padx=15)
        
        # Performance Summary
        perf_frame = ttk.LabelFrame(left_frame, text="📊 Performance")
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        perf_inner = ttk.Frame(perf_frame)
        perf_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Metrics
        self.equity_label = ttk.Label(perf_inner, text="Equity: --", font=get_font('heading'))
        self.equity_label.pack(anchor='w')
        
        self.return_label = ttk.Label(perf_inner, text="Return: --", font=get_font('normal'))
        self.return_label.pack(anchor='w')
        
        self.target_progress = ttk.Progressbar(perf_inner, length=200, mode='determinate', maximum=100)
        self.target_progress.pack(fill=tk.X, pady=5)
        self.target_info = ttk.Label(perf_inner, text="Progress to target: --", foreground=COLORS['text_muted'])
        self.target_info.pack(anchor='w')
        
        self.drawdown_label = ttk.Label(perf_inner, text="Drawdown: --", foreground=COLORS['text_muted'])
        self.drawdown_label.pack(anchor='w')
        
        # Recommendations
        rec_frame = ttk.LabelFrame(left_frame, text="💡 Recommendations")
        rec_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Action', 'Symbol', 'Shares', 'Value', 'Conviction')
        self.rec_tree = ttk.Treeview(rec_frame, columns=columns, show='headings', height=8)
        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=70)
        
        rec_scroll = ttk.Scrollbar(rec_frame, orient=tk.VERTICAL, command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=rec_scroll.set)
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rec_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Execute button
        self.exec_btn = ttk.Button(rec_frame, text="⚡ Execute Selected", command=self._execute_selected, state='disabled')
        self.exec_btn.pack(pady=5)
        
        # ========== RIGHT: Holdings & Trades ==========
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)
        
        # Holdings
        holdings_frame = ttk.LabelFrame(right_frame, text="📋 Current Holdings")
        holdings_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Symbol', 'Shares', 'Entry ₦', 'Current ₦', 'Value', 'P&L', 'Return %')
        self.holdings_tree = ttk.Treeview(holdings_frame, columns=columns, show='headings', height=10)
        col_widths = {'Symbol': 60, 'Shares': 50, 'Entry ₦': 70, 'Current ₦': 70, 'Value': 80, 'P&L': 75, 'Return %': 60}
        for col in columns:
            self.holdings_tree.heading(col, text=col)
            self.holdings_tree.column(col, width=col_widths.get(col, 65))
        
        holdings_scroll = ttk.Scrollbar(holdings_frame, orient=tk.VERTICAL, command=self.holdings_tree.yview)
        self.holdings_tree.configure(yscrollcommand=holdings_scroll.set)
        self.holdings_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        holdings_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure tags
        self.holdings_tree.tag_configure('profit', foreground=COLORS['gain'])
        self.holdings_tree.tag_configure('loss', foreground=COLORS['loss'])
        
        # Recent trades
        trades_frame = ttk.LabelFrame(right_frame, text="📜 Recent Trades")
        trades_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Date', 'Action', 'Symbol', 'Shares', 'Entry ₦', 'Exit ₦', 'P&L')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=8)
        col_widths = {'Date': 75, 'Action': 45, 'Symbol': 60, 'Shares': 50, 'Entry ₦': 70, 'Exit ₦': 70, 'P&L': 75}
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=col_widths.get(col, 65))
        
        trades_scroll = ttk.Scrollbar(trades_frame, orient=tk.VERTICAL, command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=trades_scroll.set)
        self.trades_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trades_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.trades_tree.tag_configure('BUY', foreground=COLORS['gain'])
        self.trades_tree.tag_configure('SELL', foreground=COLORS['loss'])
    
    def _load_data(self):
        """Load price data."""
        try:
            stocks = self.db.get_all_stocks()
            for stock in stocks[:50]:
                symbol = stock.get('symbol')
                stock_id = stock.get('id')
                if not stock_id:
                    continue
                
                history = self.db.get_price_history(stock_id, days=100)
                if history:
                    df = pd.DataFrame(history)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                    self.price_data[symbol] = df
            
            self.status_label.config(text=f"Loaded {len(self.price_data)} stocks")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _start_manager(self):
        """Initialize the AI Portfolio Manager."""
        try:
            capital = float(self.capital_var.get().replace(',', ''))
            target = float(self.target_var.get()) / 100
            max_dd = float(self.drawdown_var.get()) / 100
            risk = float(self.risk_var.get()) / 100
            
            config = PortfolioConfig(
                capital=capital,
                target_return_pct=target,
                max_drawdown_pct=max_dd,
                risk_per_trade_pct=risk
            )
            
            self.manager = AIPortfolioManager(config, self.db, self.ml_engine)
            self.is_running = True
            
            self.start_btn.config(text="⏹️ Stop", command=self._stop_manager)
            self.analyze_btn.state(['!disabled'])
            self.exec_btn.state(['!disabled'])
            
            self._update_display()
            self.status_label.config(text="✅ Running", foreground=COLORS['gain'])
            
        except Exception as e:
            logger.error(f"Error starting manager: {e}")
            messagebox.showerror("Error", str(e))
    
    def _stop_manager(self):
        """Stop the portfolio manager."""
        self.is_running = False
        self.start_btn.config(text="▶️ Start Manager", command=self._start_manager)
        self.analyze_btn.state(['disabled'])
        self.status_label.config(text="Stopped", foreground=COLORS['warning'])
    
    def _analyze(self):
        """Run analysis for opportunities."""
        if not self.manager or not self.price_data:
            return
        
        self.status_label.config(text="Analyzing...", foreground=COLORS['warning'])
        
        def analyze():
            try:
                recs = self.manager.analyze_opportunities(self.price_data)
                self.manager.update_portfolio(self.price_data)
                self.frame.after(0, self._display_recommendations)
                self.frame.after(0, self._update_display)
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                self.frame.after(0, lambda: self.status_label.config(text=f"Error: {e}"))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def _display_recommendations(self):
        """Display trade recommendations."""
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        
        if not self.manager:
            return
        
        for rec in self.manager.recommendations[:20]:
            self.rec_tree.insert('', 'end', values=(
                rec.action.value,
                rec.symbol,
                rec.shares,
                f"₦{rec.value:,.0f}",
                f"{rec.conviction:.0%}"
            ))
        
        self.status_label.config(text=f"Found {len(self.manager.recommendations)} opportunities", 
                                  foreground=COLORS['gain'])
    
    def _update_display(self):
        """Update performance and holdings display."""
        if not self.manager:
            return
        
        status = self.manager.get_status()
        
        # Performance
        self.equity_label.config(text=f"Equity: ₦{status['equity']:,.0f}")
        ret = status['total_return_pct']
        self.return_label.config(
            text=f"Return: {ret:+.1f}%",
            foreground=COLORS['gain'] if ret >= 0 else COLORS['loss']
        )
        
        # Target progress
        target = status['target_return_pct']
        progress = min(100, (ret / target) * 100) if target > 0 else 0
        self.target_progress['value'] = progress
        self.target_info.config(text=f"Progress to {target:.0f}% target: {progress:.0f}%")
        
        # Drawdown
        dd = status['risk_status']['drawdown_pct']
        self.drawdown_label.config(text=f"Drawdown: -{dd:.1f}%")
        
        # Holdings
        for item in self.holdings_tree.get_children():
            self.holdings_tree.delete(item)
        
        for h in self.manager.get_holdings():
            tag = 'profit' if h['pnl'] > 0 else 'loss'
            self.holdings_tree.insert('', 'end', values=(
                h['symbol'],
                h['shares'],
                f"₦{h['entry_price']:,.2f}",
                f"₦{h['current_price']:,.2f}",
                f"₦{h['value']:,.0f}",
                f"₦{h['pnl']:,.0f}",
                f"{h['return_pct']:+.1f}%"
            ), tags=(tag,))
        
        # Trades
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
        
        for t in self.manager.trades[-20:]:
            tag = t['action']
            entry_p = t.get('entry_price', t.get('price', 0))
            exit_p = t.get('exit_price', t.get('price', 0))
            self.trades_tree.insert('', 0, values=(
                t['date'].strftime('%Y-%m-%d'),
                t['action'],
                t['symbol'],
                t.get('shares', 0),
                f"₦{entry_p:,.2f}",
                f"₦{exit_p:,.2f}",
                f"₦{t.get('pnl', 0):,.0f}" if 'pnl' in t else '-'
            ), tags=(tag,))
    
    def _execute_selected(self):
        """Execute selected recommendation."""
        if not self.manager:
            return
        
        selected = self.rec_tree.selection()
        if not selected:
            messagebox.showinfo("Info", "Select a recommendation to execute")
            return
        
        for item in selected:
            values = self.rec_tree.item(item, 'values')
            symbol = values[1]
            
            # Find recommendation
            for rec in self.manager.recommendations:
                if rec.symbol == symbol:
                    success = self.manager.execute_recommendation(rec)
                    if success:
                        self.rec_tree.delete(item)
                    break
        
        self._update_display()
    
    def refresh(self):
        """Refresh data."""
        self._load_data()
        if self.manager and self.is_running:
            self._analyze()
