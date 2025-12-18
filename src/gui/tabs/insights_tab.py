"""
AI Insights Tab for MetaQuant Nigeria.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional
import threading
import logging

try:
    import ttkbootstrap as ttk_bs
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.ai.insight_engine import InsightEngine
from src.gui.theme import COLORS, get_font


logger = logging.getLogger(__name__)


class InsightsTab:
    """AI insights tab with stock analysis."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.insight_engine = InsightEngine()
        self.frame = ttk.Frame(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI."""
        # Header
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="🤖 AI-Powered Insights", font=get_font('subheading')).pack(side=tk.LEFT)
        
        status = self.insight_engine.get_status()
        status_text = f"Using: {status.get('primary', 'none').title()}"
        ttk.Label(header, text=status_text, foreground=COLORS['text_muted']).pack(side=tk.RIGHT)
        
        # Main content
        main = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Input
        self._create_input_panel(main)
        
        # Right panel - Output
        self._create_output_panel(main)
    
    def _create_input_panel(self, parent):
        """Create input panel."""
        left = ttk.Frame(parent, width=300)
        parent.add(left, weight=1)
        
        # Stock analysis
        stock_frame = ttk.LabelFrame(left, text="Stock Analysis", padding=10)
        stock_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(stock_frame, text="Enter stock symbol:").pack(anchor=tk.W)
        
        input_row = ttk.Frame(stock_frame)
        input_row.pack(fill=tk.X, pady=5)
        
        self.symbol_var = tk.StringVar()
        ttk.Entry(input_row, textvariable=self.symbol_var, width=15).pack(side=tk.LEFT)
        
        if TTKBOOTSTRAP_AVAILABLE:
            btn = ttk_bs.Button(input_row, text="Analyze", bootstyle="success", command=self._analyze_stock)
        else:
            btn = ttk.Button(input_row, text="Analyze", command=self._analyze_stock)
        btn.pack(side=tk.LEFT, padx=10)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(left, text="Quick Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(0, 15))
        
        buttons = [
            ("📊 Portfolio Health", self._analyze_portfolio),
            ("📈 Market Outlook", self._market_outlook),
            ("🔍 Compare Stocks", self._compare_stocks),
        ]
        
        for text, cmd in buttons:
            ttk.Button(actions_frame, text=text, command=cmd).pack(fill=tk.X, pady=2)
        
        # Compare input
        compare_frame = ttk.LabelFrame(left, text="Compare Stocks", padding=10)
        compare_frame.pack(fill=tk.X)
        
        ttk.Label(compare_frame, text="Enter symbols (comma-separated):").pack(anchor=tk.W)
        
        self.compare_var = tk.StringVar()
        ttk.Entry(compare_frame, textvariable=self.compare_var, width=25).pack(fill=tk.X, pady=5)
    
    def _create_output_panel(self, parent):
        """Create output panel."""
        right = ttk.Frame(parent)
        parent.add(right, weight=2)
        
        # Output label
        ttk.Label(right, text="AI Analysis", font=get_font('subheading')).pack(anchor=tk.W, pady=(0, 10))
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            right,
            wrap=tk.WORD,
            bg=COLORS['bg_medium'],
            fg=COLORS['text_primary'],
            font=get_font('body'),
            padx=15,
            pady=15
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self._set_output("Enter a stock symbol and click 'Analyze' to get AI-powered insights.\n\n"
                        "You can also:\n"
                        "• Analyze your portfolio health\n"
                        "• Get market outlook\n"
                        "• Compare multiple stocks")
    
    def _set_output(self, text: str):
        """Set output text."""
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
    
    def _show_loading(self):
        """Show loading indicator."""
        self._set_output("🔄 Analyzing... Please wait.\n\nThis may take a few moments.")
    
    def _analyze_stock(self):
        """Analyze a single stock."""
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            self._set_output("Please enter a stock symbol.")
            return
        
        stock = self.db.get_stock(symbol)
        if not stock:
            self._set_output(f"Stock '{symbol}' not found in database.\n\n"
                           "Make sure to refresh data first.")
            return
        
        self._show_loading()
        
        def run():
            result = self.insight_engine.analyze_stock(stock)
            self.frame.after(0, lambda: self._set_output(result))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _analyze_portfolio(self):
        """Analyze portfolio health."""
        # Get first portfolio
        portfolios = self.db.get_portfolios()
        if not portfolios:
            self._set_output("No portfolios found. Create a portfolio first.")
            return
        
        portfolio_id = portfolios[0]['id']
        positions = self.db.get_portfolio_positions(portfolio_id)
        
        if not positions:
            self._set_output("Portfolio has no positions.")
            return
        
        self._show_loading()
        
        # Build summary
        total_value = sum(float(p.get('market_value') or 0) for p in positions)
        total_cost = sum(float(p.get('quantity', 0)) * float(p.get('avg_cost', 0)) for p in positions)
        pnl = total_value - total_cost
        ret = (pnl / total_cost * 100) if total_cost > 0 else 0
        
        summary = {
            'total_value': total_value,
            'total_cost': total_cost,
            'unrealized_pnl': pnl,
            'return_percent': ret,
            'position_count': len(positions),
            'sector_allocation': {},
            'top_performers': sorted(positions, key=lambda p: float(p.get('return_percent') or 0), reverse=True)[:3],
            'worst_performers': sorted(positions, key=lambda p: float(p.get('return_percent') or 0))[:3],
        }
        
        def run():
            result = self.insight_engine.analyze_portfolio(summary)
            self.frame.after(0, lambda: self._set_output(result))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _market_outlook(self):
        """Get market outlook."""
        self._show_loading()
        
        stocks = self.db.get_all_stocks()
        
        market_data = {
            'total_market_cap': sum(float(s.get('market_cap') or 0) for s in stocks),
            'asi': 0,
            'asi_change': 0,
            'total_volume': sum(int(s.get('volume') or 0) for s in stocks),
            'advancers': sum(1 for s in stocks if (s.get('change_percent') or 0) > 0),
            'decliners': sum(1 for s in stocks if (s.get('change_percent') or 0) < 0),
            'unchanged': sum(1 for s in stocks if (s.get('change_percent') or 0) == 0),
        }
        
        def run():
            result = self.insight_engine.get_market_outlook(market_data)
            self.frame.after(0, lambda: self._set_output(result))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _compare_stocks(self):
        """Compare multiple stocks."""
        symbols = [s.strip().upper() for s in self.compare_var.get().split(',') if s.strip()]
        
        if len(symbols) < 2:
            self._set_output("Enter at least 2 stock symbols separated by commas.")
            return
        
        stocks = []
        for sym in symbols:
            stock = self.db.get_stock(sym)
            if stock:
                stocks.append(stock)
        
        if len(stocks) < 2:
            self._set_output("Could not find enough stocks. Check the symbols.")
            return
        
        self._show_loading()
        
        def run():
            result = self.insight_engine.compare_stocks(stocks)
            self.frame.after(0, lambda: self._set_output(result))
        
        threading.Thread(target=run, daemon=True).start()
    
    def refresh(self):
        """Refresh tab."""
        pass
