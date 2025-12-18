"""
Flow Analysis Tab for MetaQuant Nigeria.
Provides order flow analysis with CVD charts, metrics, and trade signals.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Dict
import logging
from datetime import datetime

try:
    import ttkbootstrap as ttk_bs
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.analysis.flow_analyzer import FlowAnalyzer, FlowData, FlowSignal
from src.gui.theme import COLORS, get_font, format_currency, format_percent


logger = logging.getLogger(__name__)


class FlowTab:
    """Order Flow Analysis tab."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.frame = ttk.Frame(parent)
        self.analyzer = FlowAnalyzer()
        self.current_stock = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        # Main container with left panel and chart area
        self.main_pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls and Metrics
        left_panel = ttk.Frame(self.main_pane, width=320)
        self.main_pane.add(left_panel, weight=0)
        
        # Right panel - Charts and Signals
        right_panel = ttk.Frame(self.main_pane)
        self.main_pane.add(right_panel, weight=1)
        
        self._create_controls(left_panel)
        self._create_metrics(left_panel)
        self._create_signals_table(left_panel)
        self._create_charts(right_panel)
    
    def _create_controls(self, parent):
        """Create control panel."""
        frame = ttk.LabelFrame(parent, text="Stock Selection", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Stock selector
        ttk.Label(frame, text="Symbol:").pack(anchor=tk.W)
        
        self.stock_var = tk.StringVar()
        self.stock_combo = ttk.Combobox(
            frame,
            textvariable=self.stock_var,
            state="readonly",
            width=25
        )
        self.stock_combo.pack(fill=tk.X, pady=5)
        self.stock_combo.bind('<<ComboboxSelected>>', lambda e: self._load_stock_data())
        
        # Load stocks
        stocks = self.db.get_all_stocks()
        self.stock_combo['values'] = [f"{s['symbol']} - {s['name'][:30]}" for s in stocks]
        self._stocks_map = {f"{s['symbol']} - {s['name'][:30]}": s for s in stocks}
        
        # Timeframe selector
        ttk.Label(frame, text="Period:").pack(anchor=tk.W, pady=(10, 0))
        
        self.period_var = tk.StringVar(value="30 Days")
        period_combo = ttk.Combobox(
            frame,
            textvariable=self.period_var,
            state="readonly",
            values=["7 Days", "14 Days", "30 Days", "60 Days", "90 Days", "180 Days"],
            width=25
        )
        period_combo.pack(fill=tk.X, pady=5)
        period_combo.bind('<<ComboboxSelected>>', lambda e: self._load_stock_data())
        
        # Analyze button
        if TTKBOOTSTRAP_AVAILABLE:
            analyze_btn = ttk_bs.Button(
                frame,
                text="🔍 Analyze Flow",
                bootstyle="primary",
                command=self._load_stock_data
            )
        else:
            analyze_btn = ttk.Button(frame, text="🔍 Analyze Flow", command=self._load_stock_data)
        analyze_btn.pack(fill=tk.X, pady=10)
    
    def _create_metrics(self, parent):
        """Create metrics dashboard."""
        frame = ttk.LabelFrame(parent, text="Flow Metrics", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create metric labels
        self.metrics = {}
        
        metrics_config = [
            ("cvd", "CVD", "📊"),
            ("cvd_trend", "Trend", "📈"),
            ("imbalance_pct", "Imbalance", "⚖️"),
            ("net_flow", "Net Flow", "💰"),
        ]
        
        for key, label, icon in metrics_config:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=3)
            
            ttk.Label(row, text=f"{icon} {label}:", width=12).pack(side=tk.LEFT)
            
            value_label = ttk.Label(
                row, 
                text="-", 
                font=get_font('body'),
                foreground=COLORS['text_primary']
            )
            value_label.pack(side=tk.RIGHT)
            self.metrics[key] = value_label
    
    def _create_signals_table(self, parent):
        """Create signals table."""
        frame = ttk.LabelFrame(parent, text="Trade Signals", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Signals treeview
        columns = ('date', 'signal', 'strength')
        self.signals_tree = ttk.Treeview(
            frame,
            columns=columns,
            show='headings',
            height=10
        )
        
        self.signals_tree.heading('date', text='Date')
        self.signals_tree.heading('signal', text='Signal')
        self.signals_tree.heading('strength', text='Str.')
        
        self.signals_tree.column('date', width=80)
        self.signals_tree.column('signal', width=120)
        self.signals_tree.column('strength', width=40)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=scrollbar.set)
        
        self.signals_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags for signal colors
        self.signals_tree.tag_configure('bullish', foreground=COLORS['gain'])
        self.signals_tree.tag_configure('bearish', foreground=COLORS['loss'])
    
    def _create_charts(self, parent):
        """Create charts area."""
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(
                parent,
                text="Matplotlib not installed.\npip install matplotlib",
                font=get_font('body')
            ).pack(expand=True)
            return
        
        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor=COLORS['bg_dark'])
        
        # Price + CVD chart
        self.ax_price = self.fig.add_subplot(3, 1, 1)
        self.ax_cvd = self.ax_price.twinx()
        
        # Volume Delta chart
        self.ax_delta = self.fig.add_subplot(3, 1, 2)
        
        # Imbalance chart
        self.ax_imbalance = self.fig.add_subplot(3, 1, 3)
        
        # Style charts
        for ax in [self.ax_price, self.ax_delta, self.ax_imbalance]:
            ax.set_facecolor(COLORS['bg_card'])
            ax.tick_params(colors=COLORS['text_secondary'])
            ax.spines['bottom'].set_color(COLORS['border'])
            ax.spines['left'].set_color(COLORS['border'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        self.ax_cvd.tick_params(colors=COLORS['primary'])
        self.ax_cvd.spines['right'].set_color(COLORS['primary'])
        
        self.fig.tight_layout(pad=2)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _load_stock_data(self):
        """Load and analyze stock data."""
        selection = self.stock_var.get()
        if not selection or selection not in self._stocks_map:
            return
        
        stock = self._stocks_map[selection]
        self.current_stock = stock
        
        # Parse period
        period_str = self.period_var.get()
        days = int(period_str.split()[0])
        
        # Get price history from database
        price_history = self.db.get_price_history(stock['id'], days=days)
        
        if not price_history:
            # No history - generate sample data for demo
            self._show_demo_data(stock, days)
            return
        
        # Convert to format expected by analyzer
        data = [
            {
                'date': str(p.get('date', '')),
                'open': p.get('open', 0),
                'high': p.get('high', 0),
                'low': p.get('low', 0),
                'close': p.get('close', 0),
                'volume': p.get('volume', 0),
            }
            for p in reversed(price_history)  # Oldest first
        ]
        
        self._analyze_and_display(data)
    
    def _show_demo_data(self, stock, days):
        """Show demo data when no price history available."""
        import random
        from datetime import timedelta
        
        # Generate realistic sample data
        base_price = float(stock.get('last_price', 100) or 100)
        data = []
        current_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            daily_change = random.uniform(-0.03, 0.03)
            open_p = base_price
            close = base_price * (1 + daily_change)
            high = max(open_p, close) * (1 + random.uniform(0, 0.01))
            low = min(open_p, close) * (1 - random.uniform(0, 0.01))
            volume = random.randint(100000, 2000000)
            
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'open': open_p,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
            })
            
            base_price = close
            current_date += timedelta(days=1)
        
        self._analyze_and_display(data)
    
    def _analyze_and_display(self, data: List[Dict]):
        """Analyze data and update display."""
        # Run analysis
        flow_data = self.analyzer.analyze(data)
        signals = self.analyzer.generate_all_signals()
        metrics = self.analyzer.get_current_metrics()
        
        # Update metrics display
        self._update_metrics(metrics)
        
        # Update signals table
        self._update_signals(signals)
        
        # Update charts
        self._update_charts(flow_data, signals)
    
    def _update_metrics(self, metrics: Dict):
        """Update metrics display."""
        if 'cvd' in metrics:
            cvd = metrics['cvd']
            self.metrics['cvd'].config(
                text=f"{cvd:+,.0f}",
                foreground=COLORS['gain'] if cvd > 0 else COLORS['loss']
            )
        
        if 'cvd_trend' in metrics:
            trend = metrics['cvd_trend']
            self.metrics['cvd_trend'].config(
                text=f"{'↑' if trend == 'RISING' else '↓'} {trend}",
                foreground=COLORS['gain'] if trend == 'RISING' else COLORS['loss']
            )
        
        if 'imbalance_pct' in metrics:
            self.metrics['imbalance_pct'].config(text=metrics['imbalance_pct'])
        
        if 'net_flow' in metrics:
            net = metrics['net_flow']
            self.metrics['net_flow'].config(
                text=f"{net:+,.0f}",
                foreground=COLORS['gain'] if net > 0 else COLORS['loss']
            )
    
    def _update_signals(self, signals: List[FlowSignal]):
        """Update signals table."""
        # Clear existing
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)
        
        # Add signals (most recent first)
        for signal in reversed(signals[-20:]):
            tag = 'bullish' if 'BULLISH' in signal.signal_type or signal.signal_type == 'ACCUMULATION' else 'bearish'
            
            # Abbreviate signal type
            abbrev = {
                'BULLISH_DIVERGENCE': 'Bull Div',
                'BEARISH_DIVERGENCE': 'Bear Div',
                'ACCUMULATION': 'Accum',
                'DISTRIBUTION': 'Distrib',
            }.get(signal.signal_type, signal.signal_type)
            
            self.signals_tree.insert('', tk.END, values=(
                signal.date.strftime('%m/%d'),
                abbrev,
                f"{signal.strength}%"
            ), tags=(tag,))
    
    def _update_charts(self, flow_data: List[FlowData], signals: List[FlowSignal]):
        """Update charts with flow data."""
        if not MATPLOTLIB_AVAILABLE or not flow_data:
            return
        
        dates = [fd.date for fd in flow_data]
        prices = [fd.close for fd in flow_data]
        cvd = [fd.cvd for fd in flow_data]
        delta = [fd.delta for fd in flow_data]
        imbalance = [fd.imbalance for fd in flow_data]
        
        # Clear axes
        self.ax_price.clear()
        self.ax_cvd.clear()
        self.ax_delta.clear()
        self.ax_imbalance.clear()
        
        # Price + CVD chart
        self.ax_price.plot(dates, prices, color=COLORS['text_primary'], linewidth=1.5, label='Price')
        self.ax_price.set_ylabel('Price (₦)', color=COLORS['text_primary'])
        self.ax_price.set_title(f'{self.current_stock["symbol"]} - Price & CVD', color=COLORS['text_primary'], fontsize=10)
        
        self.ax_cvd.plot(dates, cvd, color=COLORS['primary'], linewidth=1.5, linestyle='--', label='CVD')
        self.ax_cvd.set_ylabel('CVD', color=COLORS['primary'])
        self.ax_cvd.fill_between(dates, 0, cvd, alpha=0.2, color=COLORS['primary'])
        
        # Mark signals on price chart
        for sig in signals:
            if sig.date in dates:
                idx = dates.index(sig.date)
                color = COLORS['gain'] if 'BULLISH' in sig.signal_type or sig.signal_type == 'ACCUMULATION' else COLORS['loss']
                self.ax_price.axvline(x=sig.date, color=color, alpha=0.3, linestyle=':')
        
        # Volume Delta chart
        colors = [COLORS['gain'] if d > 0 else COLORS['loss'] for d in delta]
        self.ax_delta.bar(dates, delta, color=colors, alpha=0.7)
        self.ax_delta.axhline(y=0, color=COLORS['text_muted'], linestyle='-', linewidth=0.5)
        self.ax_delta.set_ylabel('Delta', color=COLORS['text_primary'])
        self.ax_delta.set_title('Volume Delta', color=COLORS['text_primary'], fontsize=10)
        
        # Imbalance chart
        self.ax_imbalance.fill_between(dates, 0, imbalance, 
                                        where=[i >= 0 for i in imbalance],
                                        color=COLORS['gain'], alpha=0.5)
        self.ax_imbalance.fill_between(dates, 0, imbalance,
                                        where=[i < 0 for i in imbalance],
                                        color=COLORS['loss'], alpha=0.5)
        self.ax_imbalance.axhline(y=0, color=COLORS['text_muted'], linestyle='-', linewidth=0.5)
        self.ax_imbalance.set_ylabel('Imbalance', color=COLORS['text_primary'])
        self.ax_imbalance.set_title('Flow Imbalance', color=COLORS['text_primary'], fontsize=10)
        self.ax_imbalance.set_ylim(-1, 1)
        
        # Rotate x-axis labels
        for ax in [self.ax_price, self.ax_delta, self.ax_imbalance]:
            ax.tick_params(axis='x', rotation=45)
            ax.set_facecolor(COLORS['bg_card'])
            ax.tick_params(colors=COLORS['text_secondary'])
        
        self.fig.tight_layout(pad=2)
        self.canvas.draw()
    
    def refresh(self):
        """Refresh data."""
        if self.current_stock:
            self._load_stock_data()
