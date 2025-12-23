"""
Flow Tape Tab - Advanced Intraday Trade Flow Visualization.
Institutional-grade flow analysis with volume profile, delta metrics, and pattern detection.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import math

from src.gui.theme import COLORS, get_font
from src.database.db_manager import DatabaseManager
from src.collectors.intraday_collector import IntradayCollector
from src.analysis.flow_analysis import FlowAnalysis

logger = logging.getLogger(__name__)


class FlowTapeTab:
    """Advanced Flow Tape visualization tab."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.collector = IntradayCollector(db)
        
        self.current_symbol = None
        self.current_interval = '15m'
        self.all_symbols = []
        self.flow_analysis = None  # Will hold FlowAnalysis instance
        self.auto_refresh_id = None  # Timer ID for auto-refresh
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        parent.add(self.frame, text="📊 Flow Tape")
        
        self._setup_ui()
        self._load_symbols()
        self._start_auto_refresh()
        
        # Background sync disabled - causes too many API calls and timeouts
        # self.collector.start_background_sync(interval_seconds=180)
    
    def _setup_ui(self):
        """Setup the UI components with sub-tabs."""
        # Header section (shared)
        self._create_header()
        
        # Delta metrics bar (shared)
        self._create_delta_metrics()
        
        # Sub-tabs notebook
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Trade Tape & Volume Profile
        self.tape_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.tape_tab, text="📋 Tape & Profile")
        self._create_tape_tab_content()
        
        # Tab 2: Alerts & Statistics
        self.alerts_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.alerts_tab, text="⚠️ Alerts")
        self._create_alerts_tab_content()
        
        # Tab 3: Charts & Visualizations
        self.charts_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.charts_tab, text="📈 Charts")
        self._create_charts_tab_content()
        
        # Tab 4: Session Analytics
        self.sessions_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.sessions_tab, text="📅 Sessions")
        self._create_sessions_tab_content()
    
    def _create_header(self):
        """Create header with controls."""
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=15, pady=10)
        
        # Title
        ttk.Label(
            header,
            text="📊 Flow Tape",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        # Subtitle
        self.subtitle_label = ttk.Label(
            header,
            text="Advanced Flow Analysis",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Controls frame
        controls = ttk.Frame(header)
        controls.pack(side=tk.RIGHT)
        
        # Symbol selector
        ttk.Label(controls, text="Symbol:", font=get_font('small')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.symbol_var = tk.StringVar()
        self.symbol_combo = ttk.Combobox(
            controls,
            textvariable=self.symbol_var,
            width=12,
            state='readonly'
        )
        self.symbol_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.symbol_combo.bind('<<ComboboxSelected>>', self._on_symbol_change)
        
        # Interval toggle
        ttk.Label(controls, text="Interval:", font=get_font('small')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.interval_var = tk.StringVar(value='15m')
        
        ttk.Radiobutton(
            controls, text="15m", variable=self.interval_var, value='15m',
            command=self._on_interval_change
        ).pack(side=tk.LEFT)
        
        ttk.Radiobutton(
            controls, text="1h", variable=self.interval_var, value='1h',
            command=self._on_interval_change
        ).pack(side=tk.LEFT, padx=(5, 15))
        
        # Refresh button
        ttk.Button(
            controls,
            text="↻ Refresh",
            command=self.refresh
        ).pack(side=tk.LEFT)
        
        # Data source status indicator
        self.data_source_label = ttk.Label(
            controls,
            text="",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.data_source_label.pack(side=tk.LEFT, padx=(15, 0))
        
        # Cumulative delta display
        self.cum_delta_frame = ttk.Frame(header)
        self.cum_delta_frame.pack(side=tk.RIGHT, padx=(0, 20))
        
        ttk.Label(
            self.cum_delta_frame,
            text="Cumulative Δ:",
            font=get_font('small')
        ).pack(side=tk.LEFT)
        
        self.cum_delta_label = ttk.Label(
            self.cum_delta_frame,
            text="--",
            font=get_font('body'),
            foreground=COLORS['text_secondary']
        )
        self.cum_delta_label.pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_delta_metrics(self):
        """Create advanced delta metrics bar."""
        metrics_frame = ttk.LabelFrame(self.frame, text="Delta Metrics")
        metrics_frame.pack(fill=tk.X, padx=15, pady=5)
        
        inner = ttk.Frame(metrics_frame)
        inner.pack(fill=tk.X, padx=10, pady=8)
        
        self.delta_labels = {}
        
        metrics = [
            ('session_delta', 'Session Δ'),
            ('delta_momentum', 'Δ Momentum'),
            ('delta_zscore', 'Δ Z-Score'),
            ('divergence', 'Divergence'),
            ('buy_absorption', 'Absorption'),
            ('sentiment', 'Sentiment')
        ]
        
        for key, label in metrics:
            frame = ttk.Frame(inner)
            frame.pack(side=tk.LEFT, padx=(0, 25))
            
            ttk.Label(
                frame,
                text=f"{label}:",
                font=get_font('small'),
                foreground=COLORS['text_muted']
            ).pack(side=tk.LEFT)
            
            self.delta_labels[key] = ttk.Label(
                frame,
                text="--",
                font=get_font('small')
            )
            self.delta_labels[key].pack(side=tk.LEFT, padx=(5, 0))
    
    # =========================================================================
    # SUB-TAB 1: TAPE & PROFILE (MEGA ENHANCED)
    # =========================================================================
    
    def _create_tape_tab_content(self):
        """Create mega-enhanced content for the Tape & Profile sub-tab."""
        # Top row: Real-time indicators
        indicator_row = ttk.Frame(self.tape_tab)
        indicator_row.pack(fill=tk.X, padx=10, pady=(5, 0))
        self._create_realtime_indicators(indicator_row)
        
        # Main content area
        main_frame = ttk.Frame(self.tape_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left column: Trade tape + Block tracker
        left_col = ttk.Frame(main_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Trade tape (main)
        tape_frame = ttk.LabelFrame(left_col, text="📋 Trade Tape (Enhanced)")
        tape_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self._create_enhanced_trade_tape(tape_frame)
        
        # Block trade tracker
        block_frame = ttk.LabelFrame(left_col, text="🐋 Block Trades (>3x RVOL)")
        block_frame.pack(fill=tk.X, pady=(5, 0))
        self._create_block_tracker(block_frame)
        
        # Middle column: Footprint + Imbalance
        mid_col = ttk.Frame(main_frame)
        mid_col.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        # Footprint chart
        footprint_frame = ttk.LabelFrame(mid_col, text="📊 Flow Summary")
        footprint_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self._create_footprint_chart(footprint_frame)
        
        # Imbalance detector
        imbalance_frame = ttk.LabelFrame(mid_col, text="⚖️ Imbalance Detector")
        imbalance_frame.pack(fill=tk.X, pady=(5, 0))
        self._create_imbalance_detector(imbalance_frame)
        
        # Right column: Enhanced Volume Profile
        right_col = ttk.Frame(main_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Volume Profile with delta overlay
        profile_frame = ttk.LabelFrame(right_col, text="📊 Volume Profile (Delta)")
        profile_frame.pack(fill=tk.BOTH, expand=True)
        self._create_enhanced_volume_profile(profile_frame)
        
        # Stats bar at bottom
        self._create_enhanced_stats_bar()
    
    def _create_realtime_indicators(self, parent):
        """Create real-time flow indicators bar."""
        # Flow velocity
        vel_frame = ttk.Frame(parent)
        vel_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(vel_frame, text="⚡ Flow Velocity:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.velocity_label = ttk.Label(vel_frame, text="--", font=get_font('body'))
        self.velocity_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Aggression meter
        agg_frame = ttk.Frame(parent)
        agg_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(agg_frame, text="🎯 Aggression:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.aggression_label = ttk.Label(agg_frame, text="--", font=get_font('body'))
        self.aggression_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Sweep counter
        sweep_frame = ttk.Frame(parent)
        sweep_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(sweep_frame, text="🌊 Sweeps:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.sweep_label = ttk.Label(sweep_frame, text="0", font=get_font('body'))
        self.sweep_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Iceberg alert
        ice_frame = ttk.Frame(parent)
        ice_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(ice_frame, text="🧊 Icebergs:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.iceberg_label = ttk.Label(ice_frame, text="0", font=get_font('body'))
        self.iceberg_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Absorption signal
        abs_frame = ttk.Frame(parent)
        abs_frame.pack(side=tk.LEFT)
        
        ttk.Label(abs_frame, text="🛡️ Absorption:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.absorption_label = ttk.Label(abs_frame, text="None", font=get_font('body'))
        self.absorption_label.pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_enhanced_trade_tape(self, parent):
        """Create the enhanced main trade tape table."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Enhanced columns
        columns = ('time', 'price', 'volume', 'delta', 'side', 'rvol', 'cumulative', 'signal')
        
        self.tape_tree = ttk.Treeview(
            container,
            columns=columns,
            show='headings',
            height=15
        )
        
        # Configure columns
        headings = {
            'time': ('Time', 90),
            'price': ('Price', 80),
            'volume': ('Volume', 80),
            'delta': ('Delta', 80),
            'side': ('Side', 60),
            'rvol': ('RVOL', 60),
            'cumulative': ('Cum Δ', 90),
            'signal': ('Signal', 80)
        }
        
        for col, (heading, width) in headings.items():
            self.tape_tree.heading(col, text=heading)
            self.tape_tree.column(col, width=width, anchor='center' if col in ['side', 'signal'] else 'e')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.tape_tree.yview)
        self.tape_tree.configure(yscrollcommand=scrollbar.set)
        
        self.tape_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enhanced tags for coloring
        self.tape_tree.tag_configure('buy', foreground=COLORS['gain'])
        self.tape_tree.tag_configure('sell', foreground=COLORS['loss'])
        self.tape_tree.tag_configure('block', background='#2a2a4a')
        self.tape_tree.tag_configure('extreme_buy', foreground='#00ff00', background='#1a3a1a')
        self.tape_tree.tag_configure('extreme_sell', foreground='#ff4444', background='#3a1a1a')
        self.tape_tree.tag_configure('sweep', foreground='#ffff00', background='#3a3a1a')
        self.tape_tree.tag_configure('iceberg', foreground='#00ffff', background='#1a3a3a')
    
    def _create_block_tracker(self, parent):
        """Create block trade tracker panel."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=5, pady=5)
        
        # Block trades list (compact)
        columns = ('time', 'price', 'size', 'delta', 'type')
        
        self.block_tree = ttk.Treeview(
            container,
            columns=columns,
            show='headings',
            height=4
        )
        
        for col, heading, width in [
            ('time', 'Time', 80),
            ('price', 'Price', 70),
            ('size', 'Size', 80),
            ('delta', 'Δ', 70),
            ('type', 'Type', 70)
        ]:
            self.block_tree.heading(col, text=heading)
            self.block_tree.column(col, width=width, anchor='center')
        
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.block_tree.yview)
        self.block_tree.configure(yscrollcommand=scrollbar.set)
        
        self.block_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.block_tree.tag_configure('buy_block', foreground='#00ff00')
        self.block_tree.tag_configure('sell_block', foreground='#ff4444')
    
    def _create_footprint_chart(self, parent):
        """Create footprint-style volume chart."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.footprint_canvas = tk.Canvas(
            container,
            width=200,
            height=250,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.footprint_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Legend
        legend_frame = ttk.Frame(container)
        legend_frame.pack(fill=tk.X, pady=(5, 0))
        
        for text, color in [('🟢 Bid', '#2a8a2a'), ('🔴 Ask', '#8a2a2a'), ('📍 POC', '#8888ff')]:
            ttk.Label(legend_frame, text=text, font=get_font('small'),
                     foreground=color).pack(side=tk.LEFT, padx=(0, 10))
    
    def _create_imbalance_detector(self, parent):
        """Create imbalance detection panel."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=5, pady=5)
        
        self.imbalance_labels = {}
        
        # Imbalance indicators
        for key, label in [
            ('buy_imbalance', '🟢 Buy Stack'),
            ('sell_imbalance', '🔴 Sell Stack'),
            ('diagonal', '↗️ Diagonal'),
            ('exhaustion', '⚠️ Exhaustion')
        ]:
            frame = ttk.Frame(container)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.imbalance_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.imbalance_labels[key].pack(side=tk.RIGHT)
    
    def _create_enhanced_volume_profile(self, parent):
        """Create enhanced volume profile with delta overlay."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Current price display
        price_frame = ttk.Frame(container)
        price_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(price_frame, text="Current:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        
        self.current_price_label = ttk.Label(
            price_frame, text="--", font=get_font('body'),
            foreground=COLORS['primary']
        )
        self.current_price_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Profile canvas (wider for delta overlay)
        self.profile_canvas = tk.Canvas(
            container,
            width=220,
            height=350,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.profile_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Value area stats
        va_frame = ttk.LabelFrame(container, text="Value Area")
        va_frame.pack(fill=tk.X, pady=(5, 0))
        
        va_inner = ttk.Frame(va_frame)
        va_inner.pack(fill=tk.X, padx=5, pady=3)
        
        self.profile_labels = {}
        for key, label in [('poc', 'POC'), ('vah', 'VAH'), ('val', 'VAL')]:
            frame = ttk.Frame(va_inner)
            frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted'], width=5).pack(side=tk.LEFT)
            
            self.profile_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.profile_labels[key].pack(side=tk.LEFT)
        
        # Volume nodes
        nodes_frame = ttk.LabelFrame(container, text="Volume Nodes")
        nodes_frame.pack(fill=tk.X, pady=(5, 0))
        
        nodes_inner = ttk.Frame(nodes_frame)
        nodes_inner.pack(fill=tk.X, padx=5, pady=3)
        
        for key, label in [('hvn', 'HVN (High Vol)'), ('lvn', 'LVN (Low Vol)')]:
            frame = ttk.Frame(nodes_inner)
            frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.profile_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.profile_labels[key].pack(side=tk.RIGHT)
        
        # Delta profile stats
        delta_frame = ttk.LabelFrame(container, text="Delta Profile")
        delta_frame.pack(fill=tk.X, pady=(5, 0))
        
        delta_inner = ttk.Frame(delta_frame)
        delta_inner.pack(fill=tk.X, padx=5, pady=3)
        
        for key, label in [('delta_poc', 'Δ @ POC'), ('delta_balance', 'Balance')]:
            frame = ttk.Frame(delta_inner)
            frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.profile_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.profile_labels[key].pack(side=tk.RIGHT)
    
    def _create_enhanced_stats_bar(self):
        """Create enhanced stats footer for tape tab."""
        stats_frame = ttk.Frame(self.tape_tab)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_labels = {}
        
        stats = [
            ('bars', 'Bars'),
            ('avg_volume', 'Avg Vol'),
            ('buy_pressure', 'Buy %'),
            ('sell_pressure', 'Sell %'),
            ('block_trades', 'Blocks'),
            ('sweeps', 'Sweeps'),
            ('rvol_percentile', 'RVOL %ile'),
            ('date_range', 'Range')
        ]
        
        for key, label in stats:
            frame = ttk.Frame(stats_frame)
            frame.pack(side=tk.LEFT, padx=(0, 15))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.stats_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.stats_labels[key].pack(side=tk.LEFT, padx=(3, 0))
    
    # =========================================================================
    # SUB-TAB 2: ALERTS
    # =========================================================================
    
    def _create_alerts_tab_content(self):
        """Create content for the Alerts sub-tab."""
        main_frame = ttk.Frame(self.alerts_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Alerts section (scrollable)
        alerts_frame = ttk.LabelFrame(main_frame, text="⚠️ Active Alerts")
        alerts_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create canvas with scrollbar
        canvas_frame = ttk.Frame(alerts_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.alerts_canvas = tk.Canvas(
            canvas_frame,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        
        alerts_scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient=tk.VERTICAL,
            command=self.alerts_canvas.yview
        )
        
        self.alerts_container = ttk.Frame(self.alerts_canvas)
        
        self.alerts_canvas.configure(yscrollcommand=alerts_scrollbar.set)
        
        alerts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.alerts_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create window in canvas
        self.alerts_window = self.alerts_canvas.create_window(
            (0, 0),
            window=self.alerts_container,
            anchor='nw'
        )
        
        # Bind resize
        self.alerts_container.bind('<Configure>', self._on_alerts_configure)
        self.alerts_canvas.bind('<Configure>', self._on_alerts_canvas_configure)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(main_frame, text="📊 Statistics Summary")
        stats_frame.pack(fill=tk.X)
        
        stats_inner = ttk.Frame(stats_frame)
        stats_inner.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_summary_labels = {}
        
        stats_items = [
            ('rvol', 'RVOL'),
            ('efficiency', 'Price Efficiency'),
            ('trade_dist', 'Trade Distribution'),
            ('total_vol', 'Total Volume')
        ]
        
        for key, label in stats_items:
            frame = ttk.Frame(stats_inner)
            frame.pack(side=tk.LEFT, padx=(0, 30))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.stats_summary_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.stats_summary_labels[key].pack(side=tk.LEFT, padx=(5, 0))
    
    def _on_alerts_configure(self, event):
        """Update scroll region when alerts container changes."""
        self.alerts_canvas.configure(scrollregion=self.alerts_canvas.bbox('all'))
    
    def _on_alerts_canvas_configure(self, event):
        """Update inner frame width when canvas resizes."""
        self.alerts_canvas.itemconfig(self.alerts_window, width=event.width)
    
    # =========================================================================
    # SUB-TAB 3: CHARTS (MEGA ENHANCED)
    # =========================================================================
    
    def _create_charts_tab_content(self):
        """Create simplified content for the Charts sub-tab."""
        # Main container - no scroll, just pack
        main_frame = ttk.Frame(self.charts_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Row 1: Key Metrics Bar
        metrics_frame = ttk.LabelFrame(main_frame, text="📊 Key Metrics")
        metrics_frame.pack(fill=tk.X, pady=(0, 5))
        self._create_chart_metrics(metrics_frame)
        
        # Row 2: Price Action + Cumulative Delta (side by side) - use grid for equal sizing
        row2 = ttk.Frame(main_frame)
        row2.pack(fill=tk.BOTH, expand=True, pady=5)
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)
        row2.rowconfigure(0, weight=1)
        
        # Price Action chart
        price_frame = ttk.LabelFrame(row2, text="📈 Price Action")
        price_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 3))
        self._create_price_chart(price_frame)
        
        # Cumulative Delta chart  
        delta_frame = ttk.LabelFrame(row2, text="📉 Cumulative Delta")
        delta_frame.grid(row=0, column=1, sticky='nsew', padx=(3, 0))
        self._create_delta_chart(delta_frame)
        
        # Row 3: Delta Momentum Oscillator (full width)
        momentum_frame = ttk.LabelFrame(main_frame, text="📊 Delta Momentum Oscillator")
        momentum_frame.pack(fill=tk.X, pady=5)
        self._create_momentum_chart(momentum_frame)
        
        # Row 4: RVOL + Session Heatmap (side by side) - use grid
        row4 = ttk.Frame(main_frame)
        row4.pack(fill=tk.X, pady=(5, 0))
        row4.columnconfigure(0, weight=1)
        row4.columnconfigure(1, weight=1)
        
        rvol_frame = ttk.LabelFrame(row4, text="📊 RVOL Distribution")
        rvol_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 3))
        self._create_rvol_chart(rvol_frame)
        
        heatmap_frame = ttk.LabelFrame(row4, text="🌡️ Session Heatmap")
        heatmap_frame.grid(row=0, column=1, sticky='nsew', padx=(3, 0))
        self._create_session_heatmap(heatmap_frame)
    
    def _create_chart_metrics(self, parent):
        """Create key metrics bar for charts."""
        self.chart_indicators = {}
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=10, pady=5)
        
        metrics = [
            ('current_price', '💰 Price'),
            ('cum_delta', '📉 Cum Delta'),
            ('delta_momentum', '📊 Momentum'),
            ('rvol', '📈 RVOL'),
            ('signal', '🎯 Signal')
        ]
        
        for key, label in metrics:
            frame = ttk.Frame(container)
            frame.pack(side=tk.LEFT, padx=(0, 25))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.chart_indicators[key] = ttk.Label(
                frame, text="--", font=get_font('body'))
            self.chart_indicators[key].pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_chart_indicators(self, parent):
        """Create chart-specific indicators row."""
        self.chart_indicators = {}
        
        indicators = [
            ('vwap', '📍 VWAP'),
            ('upper_band', '🔺 Upper Band'),
            ('lower_band', '🔻 Lower Band'),
            ('delta_momentum', '📊 Momentum'),
            ('signal', '🎯 Signal')
        ]
        
        for key, label in indicators:
            frame = ttk.Frame(parent)
            frame.pack(side=tk.LEFT, padx=(0, 20))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.chart_indicators[key] = ttk.Label(
                frame, text="--", font=get_font('body'))
            self.chart_indicators[key].pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_vwap_chart(self, parent):
        """Create VWAP chart with deviation bands."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.vwap_canvas = tk.Canvas(
            container,
            height=200,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.vwap_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Legend
        legend_frame = ttk.Frame(container)
        legend_frame.pack(fill=tk.X, pady=(5, 0))
        
        for text, color in [
            ('— Price', '#ffffff'),
            ('— VWAP', '#ffaa00'),
            ('░ +1σ/-1σ', '#444466'),
            ('░ +2σ/-2σ', '#333355')
        ]:
            ttk.Label(legend_frame, text=text, font=get_font('small'),
                     foreground=color).pack(side=tk.LEFT, padx=(0, 15))
    
    def _create_momentum_chart(self, parent):
        """Create delta momentum oscillator."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=5, pady=5)
        
        self.momentum_canvas = tk.Canvas(
            container,
            height=80,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.momentum_canvas.pack(fill=tk.X)
        
        # Stats
        stats_frame = ttk.Frame(container)
        stats_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.momentum_labels = {}
        for key, label in [('current', 'Current'), ('signal_line', 'Signal'), ('histogram', 'Hist')]:
            frame = ttk.Frame(stats_frame)
            frame.pack(side=tk.LEFT, padx=(0, 15))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.momentum_labels[key] = ttk.Label(frame, text="--", font=get_font('small'))
            self.momentum_labels[key].pack(side=tk.LEFT, padx=(3, 0))
    
    def _create_rvol_chart(self, parent):
        """Create RVOL histogram."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=5, pady=5)
        
        self.rvol_canvas = tk.Canvas(
            container,
            height=80,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.rvol_canvas.pack(fill=tk.X)
        
        # Stats
        stats_frame = ttk.Frame(container)
        stats_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.rvol_labels = {}
        for key, label in [('current', 'Current RVOL'), ('avg', 'Avg'), ('max', 'Max')]:
            frame = ttk.Frame(stats_frame)
            frame.pack(side=tk.LEFT, padx=(0, 15))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.rvol_labels[key] = ttk.Label(frame, text="--", font=get_font('small'))
            self.rvol_labels[key].pack(side=tk.LEFT, padx=(3, 0))
    
    def _create_session_heatmap(self, parent):
        """Create session heatmap."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=5, pady=5)
        
        self.heatmap_canvas = tk.Canvas(
            container,
            height=60,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.heatmap_canvas.pack(fill=tk.X)
        
        # Legend
        legend_frame = ttk.Frame(container)
        legend_frame.pack(fill=tk.X, pady=(5, 0))
        
        for text, color in [
            ('🟢 Strong Buy', '#00aa00'),
            ('🟡 Neutral', '#aaaa00'),
            ('🔴 Strong Sell', '#aa0000')
        ]:
            ttk.Label(legend_frame, text=text, font=get_font('small'),
                     foreground=color).pack(side=tk.LEFT, padx=(0, 15))
    
    # =========================================================================
    # SUB-TAB 4: SESSIONS
    # =========================================================================
    
    def _create_sessions_tab_content(self):
        """Create content for the Sessions sub-tab."""
        main_frame = ttk.Frame(self.sessions_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Intraday sessions row
        sessions_frame = ttk.LabelFrame(main_frame, text="📅 Intraday Sessions (NGX)")
        sessions_frame.pack(fill=tk.X, pady=(0, 10))
        
        sessions_inner = ttk.Frame(sessions_frame)
        sessions_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.session_labels = {}
        
        # Session cards
        for key, info in [
            ('open', ('🔔 Opening', '10:00-10:30')),
            ('core', ('⚡ Core', '10:30-13:00')),
            ('close', ('🔔 Closing', '13:00-14:30'))
        ]:
            card = ttk.Frame(sessions_inner)
            card.pack(side=tk.LEFT, padx=(0, 30))
            
            ttk.Label(card, text=info[0], font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='w')
            ttk.Label(card, text=info[1], font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='w')
            
            self.session_labels[key] = ttk.Label(
                card, text="--", font=get_font('subheading'))
            self.session_labels[key].pack(anchor='w', pady=(5, 0))
        
        # Opening Range section
        or_frame = ttk.LabelFrame(main_frame, text="📐 Opening Range Analysis")
        or_frame.pack(fill=tk.X, pady=(0, 10))
        
        or_inner = ttk.Frame(or_frame)
        or_inner.pack(fill=tk.X, padx=15, pady=10)
        
        for key, label in [
            ('or_high', 'OR High'),
            ('or_low', 'OR Low'),
            ('or_range', 'Range'),
            ('or_breakout', 'Status')
        ]:
            frame = ttk.Frame(or_inner)
            frame.pack(side=tk.LEFT, padx=(0, 25))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.session_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.session_labels[key].pack(side=tk.LEFT, padx=(5, 0))
        
        # Session comparison
        compare_frame = ttk.LabelFrame(main_frame, text="📊 Session Comparison")
        compare_frame.pack(fill=tk.X)
        
        compare_inner = ttk.Frame(compare_frame)
        compare_inner.pack(fill=tk.X, padx=15, pady=10)
        
        for key, label in [
            ('streak', 'Streak'),
            ('today_vs_avg', 'vs Average'),
            ('zscore', 'Z-Score'),
            ('percentile', 'Percentile')
        ]:
            frame = ttk.Frame(compare_inner)
            frame.pack(side=tk.LEFT, padx=(0, 25))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.session_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.session_labels[key].pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_delta_chart(self, parent):
        """Create delta bar chart visualization."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.delta_chart_canvas = tk.Canvas(
            container,
            width=200,
            height=180,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.delta_chart_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Delta stats
        stats_frame = ttk.Frame(container)
        stats_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.delta_chart_labels = {}
        for key, label in [('total', 'Total Δ'), ('trend', 'Trend')]:
            frame = ttk.Frame(stats_frame)
            frame.pack(side=tk.LEFT, padx=(0, 15))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.delta_chart_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.delta_chart_labels[key].pack(side=tk.LEFT, padx=(3, 0))
    
    def _create_price_chart(self, parent):
        """Create price mini-chart visualization."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.price_chart_canvas = tk.Canvas(
            container,
            width=200,
            height=180,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.price_chart_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Price stats
        stats_frame = ttk.Frame(container)
        stats_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.price_chart_labels = {}
        for key, label in [('high', 'High'), ('low', 'Low'), ('change', 'Chg')]:
            frame = ttk.Frame(stats_frame)
            frame.pack(side=tk.LEFT, padx=(0, 10))
            
            ttk.Label(frame, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            self.price_chart_labels[key] = ttk.Label(
                frame, text="--", font=get_font('small'))
            self.price_chart_labels[key].pack(side=tk.LEFT, padx=(3, 0))
    
    def _load_symbols(self):
        """Load available symbols from database."""
        try:
            result = self.db.conn.execute("""
                SELECT DISTINCT symbol FROM intraday_ohlcv 
                ORDER BY symbol
            """).fetchall()
            
            self.all_symbols = [r[0] for r in result]
            self.symbol_combo['values'] = self.all_symbols
            
            if self.all_symbols:
                default = 'DANGCEM' if 'DANGCEM' in self.all_symbols else self.all_symbols[0]
                self.symbol_var.set(default)
                self.current_symbol = default
                self._load_data()
                
        except Exception as e:
            logger.error(f"Error loading symbols: {e}")
    
    def _on_symbol_change(self, event=None):
        """Handle symbol selection change."""
        self.current_symbol = self.symbol_var.get()
        self._load_data()
    
    def _on_interval_change(self):
        """Handle interval toggle."""
        self.current_interval = self.interval_var.get()
        self._load_data()
    
    def _start_auto_refresh(self):
        """Start auto-refresh timer for live data during market hours."""
        self._do_auto_refresh()
    
    def _do_auto_refresh(self):
        """Perform auto-refresh and schedule next one."""
        try:
            # Check if market is open (NGX: 10:00 - 14:30 WAT, Mon-Fri)
            now = datetime.now()
            if now.weekday() < 5:  # Monday to Friday
                hour = now.hour
                minute = now.minute
                time_decimal = hour + minute / 60
                
                # Market hours: 10:00 - 14:30
                if 10 <= time_decimal <= 14.5:
                    # Refresh data
                    if self.current_symbol:
                        self._load_data()
                        logger.info(f"Auto-refresh: {self.current_symbol} at {now.strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Error in auto-refresh: {e}")
        
        # Schedule next refresh in 60 seconds
        self.auto_refresh_id = self.frame.after(60000, self._do_auto_refresh)
    
    def _stop_auto_refresh(self):
        """Stop auto-refresh timer."""
        if self.auto_refresh_id:
            self.frame.after_cancel(self.auto_refresh_id)
            self.auto_refresh_id = None
    
    def _load_data(self):
        """Load and display data for current symbol/interval."""
        if not self.current_symbol:
            return
        
        try:
            # Clear existing data
            for item in self.tape_tree.get_children():
                self.tape_tree.delete(item)
            
            # Set loading indicator
            self.data_source_label.config(text="⏳ Loading...", foreground=COLORS['text_muted'])
            self.frame.update_idletasks()
            
            # Track data source
            is_live = False
            
            # ALWAYS fetch fresh data from TradingView first
            live_data = self.collector.fetch_history(
                self.current_symbol,
                self.current_interval,
                n_bars=200  # Get recent bars
            )
            
            if live_data:
                # Store the fresh data in database
                stored = self.collector.store_ohlcv(live_data)
                logger.info(f"Fetched and stored {stored} bars for {self.current_symbol}")
                
                # Use the live data directly
                data = live_data
                is_live = True
            else:
                # Fallback to database if live fetch fails
                logger.warning(f"Live fetch failed, using cached data for {self.current_symbol}")
                data = self.collector.get_ohlcv(
                    symbol=self.current_symbol,
                    interval=self.current_interval,
                    limit=500
                )
            
            if not data:
                self.subtitle_label.config(text=f"{self.current_symbol} • No data available")
                self.data_source_label.config(text="❌ No Data", foreground=COLORS['loss'])
                return
            
            # Update data source indicator
            if is_live:
                self.data_source_label.config(text="🟢 LIVE", foreground=COLORS['gain'])
            else:
                # Show how old the cached data is
                latest_time = max(d['datetime'] for d in data)
                if hasattr(latest_time, 'strftime'):
                    age_str = latest_time.strftime('%b %d %H:%M')
                else:
                    age_str = str(latest_time)[:16]
                self.data_source_label.config(
                    text=f"🟡 CACHED ({age_str})", 
                    foreground=COLORS['warning']
                )
            
            # Create FlowAnalysis instance
            self.flow_analysis = FlowAnalysis(data)
            
            # Update subtitle with latest timestamp
            latest_time = max(d['datetime'] for d in data) if data else None
            time_str = latest_time.strftime('%H:%M:%S') if hasattr(latest_time, 'strftime') else str(latest_time)[:19]
            self.subtitle_label.config(
                text=f"{self.current_symbol} • {len(data)} bars • {self.current_interval} • Last: {time_str}"
            )
            
            # Sort by datetime descending for display
            display_data = sorted(data, key=lambda x: x['datetime'], reverse=True)
            
            # Get analysis results
            cum_delta_data = self.flow_analysis.cumulative_delta()
            cum_delta_map = {str(d['datetime']): d['cumulative_delta'] for d in cum_delta_data}
            
            zscore_data = self.flow_analysis.delta_zscore()
            zscore_map = {str(d['datetime']): d for d in zscore_data}
            
            # Calculate average volume for RVOL
            volumes = [d['volume'] for d in data if d['volume']]
            avg_vol = sum(volumes) / len(volumes) if volumes else 1
            
            # Track stats
            buy_count = 0
            sell_count = 0
            block_count = 0
            sweep_count = 0
            block_trades = []
            
            # Populate tape with enhanced columns
            for row in display_data:
                dt = row['datetime']
                dt_str = str(dt)
                time_str = dt.strftime('%Y-%m-%d %H:%M') if isinstance(dt, datetime) else dt_str[:16]
                
                price = row.get('close', 0) or 0
                volume = row.get('volume', 0) or 0
                open_price = row.get('open', 0) or 0
                
                # Calculate delta
                if price > open_price:
                    delta = volume
                    side = '🟢 BUY'
                    tag = 'buy'
                    buy_count += 1
                elif price < open_price:
                    delta = -volume
                    side = '🔴 SELL'
                    tag = 'sell'
                    sell_count += 1
                else:
                    delta = 0
                    side = '⚪ FLAT'
                    tag = ''
                
                # Get cumulative delta
                cum_delta = cum_delta_map.get(dt_str, 0)
                
                # Check for extreme Z-score and signal detection
                zscore_info = zscore_map.get(dt_str, {})
                signal = ''
                
                if zscore_info.get('signal') == 'EXTREME_BUY':
                    tag = 'extreme_buy'
                    signal = '⚡ EXT_B'
                elif zscore_info.get('signal') == 'EXTREME_SELL':
                    tag = 'extreme_sell'
                    signal = '⚡ EXT_S'
                
                # RVOL and block detection
                rvol = volume / avg_vol if avg_vol > 0 else 1
                rvol_str = f"{rvol:.1f}x"
                
                if rvol >= 3:
                    block_count += 1
                    if not tag.startswith('extreme'):
                        tag = 'block'
                    if not signal:
                        signal = '🐋 BLOCK'
                    
                    # Track block trade
                    block_trades.append({
                        'time': time_str,
                        'price': price,
                        'volume': volume,
                        'delta': delta,
                        'type': 'BUY' if delta > 0 else 'SELL'
                    })
                
                # Sweep detection (high RVOL + price movement)
                high = row.get('high', 0) or 0
                low = row.get('low', 0) or 0
                price_range = (high - low) / price * 100 if price else 0
                
                if rvol >= 2 and price_range >= 0.5:
                    sweep_count += 1
                    if not signal:
                        tag = 'sweep'
                        signal = '🌊 SWEEP'
                
                # Format values
                delta_str = f"{delta:+,.0f}" if delta != 0 else "0"
                cum_str = f"{cum_delta:+,.0f}"
                
                # Insert row with new columns
                tags = (tag,) if tag else ()
                self.tape_tree.insert('', 'end', values=(
                    time_str,
                    f"₦{price:,.2f}",
                    f"{volume:,.0f}",
                    delta_str,
                    side,
                    rvol_str,
                    cum_str,
                    signal
                ), tags=tags)
            
            # Update block tracker
            self._update_block_tracker(block_trades)
            
            # Update all metrics
            self._update_delta_metrics()
            self._update_realtime_indicators(buy_count, sell_count, block_count, sweep_count, data)
            self._update_alerts()
            self._update_alerts_stats()
            self._update_chart_metrics(data, buy_count, sweep_count)
            self._update_price_chart(data)
            self._update_delta_chart(data)
            self._update_momentum_chart(data)
            self._update_rvol_chart(data)
            self._update_session_heatmap(data)
            self._update_footprint_chart(data)
            self._update_imbalance_detector(data)
            self._update_volume_profile()
            self._update_cumulative_delta()
            self._update_session_analytics()
            self._update_stats(len(data), avg_vol, buy_count, block_count, sweep_count, data)
            
        except Exception as e:
            logger.error(f"Error loading flow tape data: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_delta_metrics(self):
        """Update the delta metrics panel."""
        if not self.flow_analysis:
            return
        
        try:
            # Session delta
            session_deltas = self.flow_analysis.session_delta()
            if session_deltas:
                latest_date = max(session_deltas.keys())
                session_delta = session_deltas[latest_date]
                color = COLORS['gain'] if session_delta >= 0 else COLORS['loss']
                self.delta_labels['session_delta'].config(
                    text=f"{session_delta:+,.0f}",
                    foreground=color
                )
            
            # Delta momentum
            momentum = self.flow_analysis.delta_momentum()
            if momentum:
                latest_momentum = momentum[-1]
                mom_val = latest_momentum['delta_momentum']
                color = COLORS['gain'] if mom_val >= 0 else COLORS['loss']
                trend = "↑" if mom_val > 0 else "↓" if mom_val < 0 else "→"
                self.delta_labels['delta_momentum'].config(
                    text=f"{trend} {abs(mom_val):,.0f}",
                    foreground=color
                )
            
            # Z-score
            zscore_data = self.flow_analysis.delta_zscore()
            if zscore_data:
                latest_z = zscore_data[-1]['zscore']
                signal = zscore_data[-1]['signal']
                if signal == 'EXTREME_BUY':
                    color, text = COLORS['gain'], f"+{latest_z:.1f}σ ⚡"
                elif signal == 'EXTREME_SELL':
                    color, text = COLORS['loss'], f"{latest_z:.1f}σ ⚡"
                else:
                    color, text = COLORS['text_secondary'], f"{latest_z:+.1f}σ"
                self.delta_labels['delta_zscore'].config(text=text, foreground=color)
            
            # Divergence
            divergences = self.flow_analysis.delta_divergence()
            if divergences:
                latest = divergences[-1]
                div_type = latest['type']
                color = COLORS['gain'] if div_type == 'ACCUMULATION' else COLORS['loss']
                self.delta_labels['divergence'].config(
                    text=div_type[:4],
                    foreground=color
                )
            else:
                self.delta_labels['divergence'].config(
                    text="None",
                    foreground=COLORS['text_muted']
                )
            
            # Absorption
            absorptions = self.flow_analysis.absorption_detection()
            if absorptions:
                self.delta_labels['buy_absorption'].config(
                    text=f"{len(absorptions)} detected",
                    foreground=COLORS['warning']
                )
            else:
                self.delta_labels['buy_absorption'].config(
                    text="None",
                    foreground=COLORS['text_muted']
                )
            
            # Sentiment
            stats = self.flow_analysis.summary_stats()
            sentiment = stats.get('sentiment', 'NEUTRAL')
            if sentiment == 'BULLISH':
                color = COLORS['gain']
            elif sentiment == 'BEARISH':
                color = COLORS['loss']
            else:
                color = COLORS['warning']
            self.delta_labels['sentiment'].config(text=sentiment, foreground=color)
            
        except Exception as e:
            logger.error(f"Error updating delta metrics: {e}")
    
    def _update_block_tracker(self, block_trades):
        """Update the block trade tracker panel."""
        # Clear existing
        for item in self.block_tree.get_children():
            self.block_tree.delete(item)
        
        # Add recent blocks (limit to 10)
        for trade in block_trades[:10]:
            tag = 'buy_block' if trade['type'] == 'BUY' else 'sell_block'
            self.block_tree.insert('', 'end', values=(
                trade['time'][-8:],  # Just time portion
                f"₦{trade['price']:,.2f}",
                f"{trade['volume']:,.0f}",
                f"{trade['delta']:+,.0f}",
                trade['type']
            ), tags=(tag,))
    
    def _update_realtime_indicators(self, buy_count, sell_count, block_count, sweep_count, data):
        """Update real-time flow indicators."""
        try:
            total_trades = buy_count + sell_count
            
            # Flow velocity (trades per minute based on data)
            if data and len(data) >= 2:
                first_dt = data[0]['datetime']
                last_dt = data[-1]['datetime']
                if hasattr(first_dt, 'timestamp'):
                    time_span = (last_dt - first_dt).total_seconds() / 60
                    velocity = len(data) / max(time_span, 1)
                    
                    if velocity > 5:
                        vel_text = f"🔥 {velocity:.1f}/min"
                        vel_color = COLORS['loss']
                    elif velocity > 2:
                        vel_text = f"⚡ {velocity:.1f}/min"
                        vel_color = COLORS['warning']
                    else:
                        vel_text = f"🐢 {velocity:.1f}/min"
                        vel_color = COLORS['text_muted']
                    
                    self.velocity_label.config(text=vel_text, foreground=vel_color)
            
            # Aggression meter
            if total_trades > 0:
                buy_ratio = buy_count / total_trades * 100
                if buy_ratio > 65:
                    agg_text = f"🟢 {buy_ratio:.0f}% BUY"
                    agg_color = COLORS['gain']
                elif buy_ratio < 35:
                    agg_text = f"🔴 {100-buy_ratio:.0f}% SELL"
                    agg_color = COLORS['loss']
                else:
                    agg_text = f"⚪ {buy_ratio:.0f}% BUY"
                    agg_color = COLORS['warning']
                
                self.aggression_label.config(text=agg_text, foreground=agg_color)
            
            # Sweep counter
            sweep_color = COLORS['warning'] if sweep_count > 0 else COLORS['text_muted']
            self.sweep_label.config(text=str(sweep_count), foreground=sweep_color)
            
            # Iceberg detection (simplified)
            if self.flow_analysis:
                icebergs = self.flow_analysis.iceberg_detection()
                ice_count = len(icebergs) if icebergs else 0
                ice_color = COLORS['primary'] if ice_count > 0 else COLORS['text_muted']
                self.iceberg_label.config(text=str(ice_count), foreground=ice_color)
            
            # Absorption signal
            if self.flow_analysis:
                absorptions = self.flow_analysis.absorption_detection()
                if absorptions:
                    self.absorption_label.config(text=f"{len(absorptions)} found", foreground=COLORS['warning'])
                else:
                    self.absorption_label.config(text="None", foreground=COLORS['text_muted'])
                    
        except Exception as e:
            logger.error(f"Error updating realtime indicators: {e}")
    
    def _update_footprint_chart(self, data):
        """Update the flow summary chart (buy/sell pressure visualization)."""
        if not data or not self.flow_analysis:
            return
        
        try:
            self.footprint_canvas.delete('all')
            
            canvas_width = 200
            canvas_height = 250
            padding = 15
            
            # Get imbalance data
            imbalance = self.flow_analysis.detect_imbalance()
            
            if not imbalance:
                self.footprint_canvas.create_text(
                    canvas_width / 2, canvas_height / 2,
                    text="No data", fill=COLORS['text_muted']
                )
                return
            
            buy_vol = imbalance.get('buy_volume', 0)
            sell_vol = imbalance.get('sell_volume', 0)
            total_vol = buy_vol + sell_vol or 1
            buy_pct = buy_vol / total_vol
            
            # Draw pressure bar (horizontal)
            bar_y = 30
            bar_height = 25
            bar_left = padding
            bar_right = canvas_width - padding
            bar_width = bar_right - bar_left
            
            # Buy side (green)
            buy_width = bar_width * buy_pct
            self.footprint_canvas.create_rectangle(
                bar_left, bar_y, bar_left + buy_width, bar_y + bar_height,
                fill='#2a8a2a', outline=''
            )
            
            # Sell side (red)
            self.footprint_canvas.create_rectangle(
                bar_left + buy_width, bar_y, bar_right, bar_y + bar_height,
                fill='#8a2a2a', outline=''
            )
            
            # Labels
            self.footprint_canvas.create_text(
                bar_left + 5, bar_y + bar_height/2,
                text=f"BUY {buy_pct*100:.0f}%", anchor='w',
                fill='white', font=('Arial', 9, 'bold')
            )
            
            self.footprint_canvas.create_text(
                bar_right - 5, bar_y + bar_height/2,
                text=f"{(1-buy_pct)*100:.0f}% SELL", anchor='e',
                fill='white', font=('Arial', 9, 'bold')
            )
            
            # Metrics section
            y_start = 80
            line_height = 22
            
            metrics = [
                ('📈 Buy Volume', f"{buy_vol:,.0f}", COLORS['gain']),
                ('📉 Sell Volume', f"{sell_vol:,.0f}", COLORS['loss']),
                ('🐋 Block Stacks', f"{imbalance.get('buy_imbalance_count', 0)} buy, {imbalance.get('sell_imbalance_count', 0)} sell", COLORS['text_secondary']),
                ('↗️ Diagonal Moves', f"{imbalance.get('diagonal_count', 0)}", COLORS['primary']),
                ('⚠️ Exhaustion', imbalance.get('exhaustion', 'None'), COLORS['warning'] if imbalance.get('exhaustion') != 'None' else COLORS['text_muted']),
                ('🎯 Dominant Side', imbalance.get('dominant_side', 'NEUTRAL'), COLORS['gain'] if imbalance.get('dominant_side') == 'BUY' else COLORS['loss'])
            ]
            
            for i, (label, value, color) in enumerate(metrics):
                y = y_start + i * line_height
                
                self.footprint_canvas.create_text(
                    padding, y, text=label, anchor='w',
                    fill=COLORS['text_muted'], font=('Arial', 9)
                )
                
                self.footprint_canvas.create_text(
                    canvas_width - padding, y, text=value, anchor='e',
                    fill=color, font=('Arial', 9, 'bold')
                )
            
            # Draw separator
            self.footprint_canvas.create_line(
                padding, y_start - 10, canvas_width - padding, y_start - 10,
                fill=COLORS['text_muted'], dash=(2, 2)
            )
                    
        except Exception as e:
            logger.error(f"Error updating footprint chart: {e}")
    
    def _update_imbalance_detector(self, data):
        """Update the imbalance detection panel."""
        if not data or not self.flow_analysis:
            return
        
        try:
            # Calculate imbalances
            imbalance = self.flow_analysis.detect_imbalance()
            
            if imbalance:
                # Buy imbalance
                buy_imb = imbalance.get('buy_imbalance_count', 0)
                color = COLORS['gain'] if buy_imb > 0 else COLORS['text_muted']
                self.imbalance_labels['buy_imbalance'].config(
                    text=f"{buy_imb} stacks", foreground=color
                )
                
                # Sell imbalance
                sell_imb = imbalance.get('sell_imbalance_count', 0)
                color = COLORS['loss'] if sell_imb > 0 else COLORS['text_muted']
                self.imbalance_labels['sell_imbalance'].config(
                    text=f"{sell_imb} stacks", foreground=color
                )
                
                # Diagonal
                diagonal = imbalance.get('diagonal_count', 0)
                self.imbalance_labels['diagonal'].config(text=str(diagonal))
                
                # Exhaustion
                exhaustion = imbalance.get('exhaustion', 'None')
                color = COLORS['warning'] if exhaustion != 'None' else COLORS['text_muted']
                self.imbalance_labels['exhaustion'].config(text=exhaustion, foreground=color)
            else:
                for key in self.imbalance_labels:
                    self.imbalance_labels[key].config(text="--", foreground=COLORS['text_muted'])
                    
        except Exception as e:
            logger.error(f"Error updating imbalance detector: {e}")
    
    def _update_alerts(self):
        """Update the alerts panel."""
        if not self.flow_analysis:
            return
        
        try:
            # Clear existing alerts
            for widget in self.alerts_container.winfo_children():
                widget.destroy()
            
            # Generate alerts
            alerts = self.flow_analysis.generate_alerts()
            
            if not alerts:
                ttk.Label(
                    self.alerts_container,
                    text="✓ No active alerts",
                    font=get_font('small'),
                    foreground=COLORS['gain']
                ).pack(anchor='w')
                return
            
            # Display all alerts
            for alert in alerts:
                alert_frame = ttk.Frame(self.alerts_container)
                alert_frame.pack(fill=tk.X, pady=3)
                
                # Severity indicator
                if alert['severity'] == 'HIGH':
                    icon = '🔴'
                    color = COLORS['loss']
                elif alert['severity'] == 'MEDIUM':
                    icon = '🟡'
                    color = COLORS['warning']
                else:
                    icon = '🔵'
                    color = COLORS['primary']
                
                # Top row - alert type and message
                top_row = ttk.Frame(alert_frame)
                top_row.pack(fill=tk.X)
                
                ttk.Label(
                    top_row,
                    text=icon,
                    font=('Arial', 10)
                ).pack(side=tk.LEFT)
                
                ttk.Label(
                    top_row,
                    text=f" [{alert['type']}]",
                    font=get_font('small'),
                    foreground=color
                ).pack(side=tk.LEFT)
                
                ttk.Label(
                    top_row,
                    text=f" {alert['message']}",
                    font=get_font('small'),
                    foreground=COLORS['text_secondary']
                ).pack(side=tk.LEFT)
                
                # Context row - explanation
                if alert.get('context'):
                    context_row = ttk.Frame(alert_frame)
                    context_row.pack(fill=tk.X, padx=(20, 0))
                    
                    ttk.Label(
                        context_row,
                        text=f"📋 {alert['context']}",
                        font=get_font('small'),
                        foreground=COLORS['text_muted'],
                        wraplength=600
                    ).pack(anchor='w')
                
                # Action row - what to do
                if alert.get('action'):
                    action_row = ttk.Frame(alert_frame)
                    action_row.pack(fill=tk.X, padx=(20, 0))
                    
                    ttk.Label(
                        action_row,
                        text=f"➡️ {alert['action']}",
                        font=get_font('small'),
                        foreground=COLORS['primary']
                    ).pack(anchor='w')
                
                # Price info row - key levels
                if alert.get('price_info'):
                    price_row = ttk.Frame(alert_frame)
                    price_row.pack(fill=tk.X, padx=(20, 0))
                    
                    ttk.Label(
                        price_row,
                        text=f"💰 {alert['price_info']}",
                        font=get_font('small'),
                        foreground=COLORS['gain']
                    ).pack(anchor='w')
                
        except Exception as e:
            logger.error(f"Error updating alerts: {e}")
    
    def _update_alerts_stats(self):
        """Update the statistics summary in the alerts tab."""
        if not self.flow_analysis:
            return
        
        try:
            # RVOL
            rvol_info = self.flow_analysis.rvol_percentile()
            if rvol_info:
                rank = rvol_info.get('rank', 'NORMAL')
                pct = rvol_info.get('percentile', 50)
                color = COLORS['gain'] if rank in ['EXTREME', 'HIGH'] else COLORS['text_secondary']
                self.stats_summary_labels['rvol'].config(
                    text=f"{pct:.0f}% ({rank})",
                    foreground=color
                )
            
            # Price efficiency
            efficiency = self.flow_analysis.price_efficiency()
            if efficiency:
                eff_val = efficiency.get('efficiency', 0)
                self.stats_summary_labels['efficiency'].config(
                    text=f"{eff_val:.4f} $/vol"
                )
            
            # Trade distribution
            dist = self.flow_analysis.trade_size_distribution()
            if dist:
                dominance = dist.get('dominance', 'MIXED')
                self.stats_summary_labels['trade_dist'].config(text=dominance)
            
            # Total volume
            summary = self.flow_analysis.summary_stats()
            if summary:
                total_vol = summary.get('total_volume', 0)
                self.stats_summary_labels['total_vol'].config(
                    text=f"{total_vol:,.0f}"
                )
                
        except Exception as e:
            logger.error(f"Error updating alerts stats: {e}")
    
    def _update_chart_metrics(self, data, buy_count, sweep_count):
        """Update the key metrics bar in charts tab."""
        if not data or not self.flow_analysis:
            return
        
        try:
            # Current price
            current_price = data[-1].get('close', 0) if data else 0
            self.chart_indicators['current_price'].config(
                text=f"₦{current_price:,.2f}"
            )
            
            # Cumulative delta
            cum_delta = self.flow_analysis.summary_stats().get('cumulative_delta', 0)
            color = COLORS['gain'] if cum_delta >= 0 else COLORS['loss']
            self.chart_indicators['cum_delta'].config(
                text=f"{cum_delta:+,.0f}",
                foreground=color
            )
            
            # Delta momentum
            momentum_data = self.flow_analysis.delta_momentum()
            if momentum_data:
                current_mom = momentum_data[-1].get('delta_momentum', 0)
                color = COLORS['gain'] if current_mom >= 0 else COLORS['loss']
                trend = "↑" if current_mom > 0 else "↓" if current_mom < 0 else "→"
                self.chart_indicators['delta_momentum'].config(
                    text=f"{trend} {abs(current_mom):,.0f}",
                    foreground=color
                )
            
            # RVOL
            rvol_info = self.flow_analysis.rvol_analysis()
            rvol = rvol_info.get('rvol', 1)
            color = COLORS['loss'] if rvol >= 3 else COLORS['warning'] if rvol >= 2 else COLORS['text_secondary']
            self.chart_indicators['rvol'].config(
                text=f"{rvol:.1f}x",
                foreground=color
            )
            
            # Signal
            total_trades = buy_count + (len(data) - buy_count)
            buy_ratio = buy_count / total_trades * 100 if total_trades else 50
            
            if buy_ratio > 65:
                signal = "🟢 BULLISH"
                color = COLORS['gain']
            elif buy_ratio < 35:
                signal = "🔴 BEARISH"
                color = COLORS['loss']
            else:
                signal = "⚪ NEUTRAL"
                color = COLORS['warning']
            
            self.chart_indicators['signal'].config(text=signal, foreground=color)
            
        except Exception as e:
            logger.error(f"Error updating chart metrics: {e}")
    
    def _update_momentum_chart(self, data):
        """Update delta momentum oscillator."""
        if not data or not self.flow_analysis:
            return
        
        try:
            self.momentum_canvas.delete('all')
            self.momentum_canvas.update_idletasks()
            
            # Get momentum data
            momentum_data = self.flow_analysis.delta_momentum()
            
            if not momentum_data or len(momentum_data) < 5:
                return
            
            # Take last 50 points
            chart_data = momentum_data[-50:]
            
            canvas_width = max(self.momentum_canvas.winfo_width(), 300)
            canvas_height = max(self.momentum_canvas.winfo_height(), 70)
            padding = 10
            
            # Calculate signal line (simple moving average of momentum)
            mom_values = [d.get('delta_momentum', 0) for d in chart_data]
            signal_period = 5
            signal_line = []
            for i in range(len(mom_values)):
                if i < signal_period - 1:
                    signal_line.append(sum(mom_values[:i+1]) / (i+1))
                else:
                    signal_line.append(sum(mom_values[i-signal_period+1:i+1]) / signal_period)
            
            # Find range
            all_vals = mom_values + signal_line
            max_val = max(abs(v) for v in all_vals) or 1
            
            bar_width = (canvas_width - 2 * padding) / len(chart_data)
            zero_y = canvas_height / 2
            
            # Draw zero line
            self.momentum_canvas.create_line(
                padding, zero_y, canvas_width - padding, zero_y,
                fill='#444444', dash=(2, 2)
            )
            
            # Draw histogram bars
            for i, (mom, sig) in enumerate(zip(mom_values, signal_line)):
                x = padding + i * bar_width
                
                # Histogram (momentum - signal)
                hist = mom - sig
                hist_height = (hist / max_val) * (canvas_height / 2 - padding)
                
                color = '#2a8a2a' if hist > 0 else '#8a2a2a'
                
                self.momentum_canvas.create_rectangle(
                    x + 1, zero_y, x + bar_width - 1, zero_y - hist_height,
                    fill=color, outline=''
                )
            
            # Draw signal line
            signal_points = []
            for i, sig in enumerate(signal_line):
                x = padding + i * bar_width + bar_width / 2
                y = zero_y - (sig / max_val) * (canvas_height / 2 - padding)
                signal_points.extend([x, y])
            
            if len(signal_points) >= 4:
                self.momentum_canvas.create_line(signal_points, fill='#ffaa00', width=1, smooth=True)
            
            # Update labels
            current_mom = mom_values[-1] if mom_values else 0
            current_sig = signal_line[-1] if signal_line else 0
            current_hist = current_mom - current_sig
            
            self.momentum_labels['current'].config(
                text=f"{current_mom:+,.0f}",
                foreground=COLORS['gain'] if current_mom > 0 else COLORS['loss']
            )
            self.momentum_labels['signal_line'].config(text=f"{current_sig:+,.0f}")
            self.momentum_labels['histogram'].config(
                text=f"{current_hist:+,.0f}",
                foreground=COLORS['gain'] if current_hist > 0 else COLORS['loss']
            )
            
            # Update chart indicator
            self.chart_indicators['delta_momentum'].config(
                text=f"{current_mom:+,.0f}",
                foreground=COLORS['gain'] if current_mom > 0 else COLORS['loss']
            )
            
        except Exception as e:
            logger.error(f"Error updating momentum chart: {e}")
    
    def _update_rvol_chart(self, data):
        """Update RVOL histogram chart."""
        if not data:
            return
        
        try:
            self.rvol_canvas.delete('all')
            self.rvol_canvas.update_idletasks()
            
            # Calculate RVOL for each bar
            volumes = [d.get('volume', 0) or 0 for d in data if d.get('volume')]
            
            if not volumes:
                return
            
            avg_vol = sum(volumes) / len(volumes) if volumes else 1
            
            chart_data = sorted(data, key=lambda x: x['datetime'])[-50:]
            
            canvas_width = max(self.rvol_canvas.winfo_width(), 300)
            canvas_height = max(self.rvol_canvas.winfo_height(), 70)
            padding = 10
            
            bar_width = (canvas_width - 2 * padding) / len(chart_data)
            
            # Calculate RVOLs
            rvols = []
            for bar in chart_data:
                vol = bar.get('volume', 0) or 0
                rvol = vol / avg_vol if avg_vol > 0 else 1
                rvols.append(rvol)
            
            max_rvol = max(rvols) if rvols else 1
            
            # Draw 1x reference line
            ref_y = canvas_height - padding - ((1 / max_rvol) * (canvas_height - 2 * padding))
            self.rvol_canvas.create_line(
                padding, ref_y, canvas_width - padding, ref_y,
                fill='#666666', dash=(2, 2)
            )
            self.rvol_canvas.create_text(
                padding - 5, ref_y, text="1x", anchor='e',
                fill='#666666', font=('Arial', 7)
            )
            
            # Draw bars
            for i, (bar, rvol) in enumerate(zip(chart_data, rvols)):
                x = padding + i * bar_width
                
                bar_height = (rvol / max_rvol) * (canvas_height - 2 * padding)
                
                # Color based on RVOL level
                if rvol >= 3:
                    color = '#aa0000'  # Extreme
                elif rvol >= 2:
                    color = '#aaaa00'  # High
                elif rvol >= 1:
                    color = '#2a8a2a'  # Normal
                else:
                    color = '#444444'  # Low
                
                self.rvol_canvas.create_rectangle(
                    x + 1, canvas_height - padding, x + bar_width - 1, canvas_height - padding - bar_height,
                    fill=color, outline=''
                )
            
            # Update labels
            current_rvol = rvols[-1] if rvols else 0
            avg_rvol = sum(rvols) / len(rvols) if rvols else 0
            
            self.rvol_labels['current'].config(
                text=f"{current_rvol:.1f}x",
                foreground=COLORS['loss'] if current_rvol >= 3 else COLORS['warning'] if current_rvol >= 2 else COLORS['text_secondary']
            )
            self.rvol_labels['avg'].config(text=f"{avg_rvol:.1f}x")
            self.rvol_labels['max'].config(text=f"{max_rvol:.1f}x")
            
        except Exception as e:
            logger.error(f"Error updating RVOL chart: {e}")
    
    def _update_session_heatmap(self, data):
        """Update session heatmap visualization."""
        if not data or not self.flow_analysis:
            return
        
        try:
            self.heatmap_canvas.delete('all')
            self.heatmap_canvas.update_idletasks()
            
            # Get session breakdown
            sessions = self.flow_analysis.intraday_session_breakdown()
            
            canvas_width = max(self.heatmap_canvas.winfo_width(), 300)
            canvas_height = 60
            padding = 10
            
            session_width = (canvas_width - 2 * padding) / 3
            
            session_order = ['open', 'core', 'close']
            session_names = {'open': 'OPEN', 'core': 'CORE', 'close': 'CLOSE'}
            
            for i, key in enumerate(session_order):
                session = sessions.get(key, {})
                delta = session.get('delta', 0)
                trend = session.get('trend', 'NEUTRAL')
                
                x = padding + i * session_width
                
                # Color based on trend
                if trend == 'BULLISH':
                    color = '#00aa00'
                    intensity = min(abs(delta) / 10000, 1)
                    r = int(0 * (1 - intensity) + 0 * intensity)
                    g = int(100 + 70 * intensity)
                    b = int(0 * (1 - intensity) + 0 * intensity)
                    color = f'#{r:02x}{g:02x}{b:02x}'
                elif trend == 'BEARISH':
                    color = '#aa0000'
                    intensity = min(abs(delta) / 10000, 1)
                    r = int(100 + 70 * intensity)
                    g = int(0 * (1 - intensity) + 0 * intensity)
                    b = int(0 * (1 - intensity) + 0 * intensity)
                    color = f'#{r:02x}{g:02x}{b:02x}'
                else:
                    color = '#aaaa00'
                
                # Draw session block
                self.heatmap_canvas.create_rectangle(
                    x + 2, padding, x + session_width - 2, canvas_height - padding,
                    fill=color, outline='#333333'
                )
                
                # Session name
                self.heatmap_canvas.create_text(
                    x + session_width / 2, canvas_height / 2 - 10,
                    text=session_names[key], fill='white',
                    font=('Arial', 10, 'bold')
                )
                
                # Delta value
                self.heatmap_canvas.create_text(
                    x + session_width / 2, canvas_height / 2 + 8,
                    text=f"{delta:+,.0f}", fill='white',
                    font=('Arial', 9)
                )
            
        except Exception as e:
            logger.error(f"Error updating session heatmap: {e}")
    
    def _update_delta_chart(self, data):
        """Update the cumulative delta line chart."""
        if not data or not self.flow_analysis:
            return
        
        try:
            self.delta_chart_canvas.delete('all')
            self.delta_chart_canvas.update_idletasks()
            
            # Get cumulative delta data (already in chronological order)
            cum_delta_data = self.flow_analysis.cumulative_delta()
            
            if not cum_delta_data or len(cum_delta_data) < 2:
                return
            
            # Take last 50 points for readability
            chart_data = cum_delta_data[-50:] if len(cum_delta_data) > 50 else cum_delta_data
            
            # Canvas dimensions
            canvas_width = max(self.delta_chart_canvas.winfo_width(), 200)
            canvas_height = max(self.delta_chart_canvas.winfo_height(), 150)
            padding = 15
            
            # Get delta range
            deltas = [d['cumulative_delta'] for d in chart_data]
            min_delta = min(deltas)
            max_delta = max(deltas)
            delta_range = max_delta - min_delta or 1
            
            chart_height = canvas_height - 2 * padding
            bar_width = (canvas_width - 2 * padding) / len(chart_data)
            
            # Draw zero line if delta crosses zero
            if min_delta < 0 < max_delta:
                zero_y = padding + (1 - (0 - min_delta) / delta_range) * chart_height
                self.delta_chart_canvas.create_line(
                    padding, zero_y, canvas_width - padding, zero_y,
                    fill='#555555', dash=(2, 2)
                )
            
            # Build points for line chart
            points = []
            for i, d in enumerate(chart_data):
                x = padding + i * bar_width + bar_width / 2
                y = padding + (1 - (d['cumulative_delta'] - min_delta) / delta_range) * chart_height
                points.extend([x, y])
            
            # Fill area under curve
            if len(points) >= 4:
                # Determine fill color based on final delta
                final_delta = chart_data[-1]['cumulative_delta']
                if final_delta >= 0:
                    fill_color = '#1a3a1a'  # Green tint
                    line_color = '#00ff00'
                else:
                    fill_color = '#3a1a1a'  # Red tint
                    line_color = '#ff4444'
                
                # Create filled polygon
                fill_points = [padding, canvas_height - padding] + points + [canvas_width - padding, canvas_height - padding]
                self.delta_chart_canvas.create_polygon(
                    fill_points,
                    fill=fill_color, outline=''
                )
                
                # Draw the line
                self.delta_chart_canvas.create_line(
                    *points,
                    fill=line_color, width=2, smooth=True
                )
            
            # Update stats
            total_delta = chart_data[-1]['cumulative_delta']
            start_delta = chart_data[0]['cumulative_delta']
            delta_change = total_delta - start_delta
            
            color = COLORS['gain'] if total_delta >= 0 else COLORS['loss']
            self.delta_chart_labels['total'].config(
                text=f"{total_delta:+,.0f}",
                foreground=color
            )
            
            # Trend based on delta change
            if delta_change > 0:
                trend = "↑ Accumulation"
                trend_color = COLORS['gain']
            elif delta_change < 0:
                trend = "↓ Distribution"
                trend_color = COLORS['loss']
            else:
                trend = "→ Neutral"
                trend_color = COLORS['warning']
            
            self.delta_chart_labels['trend'].config(text=trend, foreground=trend_color)
            
        except Exception as e:
            logger.error(f"Error updating delta chart: {e}")
    
    def _update_price_chart(self, data):
        """Update the price line chart with key levels."""
        if not data or not self.flow_analysis:
            return
        
        try:
            self.price_chart_canvas.delete('all')
            
            # Ensure data is sorted chronologically
            chart_data = sorted(data, key=lambda x: x['datetime'])
            
            # Take last 50 bars
            chart_data = chart_data[-50:] if len(chart_data) > 50 else chart_data
            
            if not chart_data:
                return
            
            # Canvas dimensions
            canvas_width = 200
            canvas_height = 180
            padding = 15
            
            # Get price range from closes (simpler, more reliable)
            closes = [bar.get('close', 0) or 0 for bar in chart_data if bar.get('close')]
            
            if not closes:
                return
            
            min_price = min(closes) * 0.998  # Small buffer
            max_price = max(closes) * 1.002
            price_range = max_price - min_price or 1
            
            chart_height = canvas_height - 2 * padding
            bar_width = (canvas_width - 2 * padding) / len(chart_data)
            
            # Get volume profile levels
            profile = self.flow_analysis.volume_profile(num_levels=15)
            poc = profile.get('poc', 0) if profile else 0
            vah = profile.get('vah', 0) if profile else 0
            val = profile.get('val', 0) if profile else 0
            
            # Draw VAH/VAL zone
            if val and vah and min_price < val < max_price and min_price < vah < max_price:
                val_y = padding + (1 - (val - min_price) / price_range) * chart_height
                vah_y = padding + (1 - (vah - min_price) / price_range) * chart_height
                self.price_chart_canvas.create_rectangle(
                    padding, vah_y, canvas_width - padding, val_y,
                    fill='#2a2a4a', outline=''
                )
            
            # Draw POC line
            if poc and min_price < poc < max_price:
                poc_y = padding + (1 - (poc - min_price) / price_range) * chart_height
                self.price_chart_canvas.create_line(
                    padding, poc_y, canvas_width - padding, poc_y,
                    fill='#8888ff', width=2
                )
            
            # Build price line
            points = []
            for i, bar in enumerate(chart_data):
                c = bar.get('close', 0) or 0
                if c:
                    x = padding + i * bar_width + bar_width / 2
                    y = padding + (1 - (c - min_price) / price_range) * chart_height
                    points.extend([x, y])
            
            # Draw price line
            if len(points) >= 4:
                first_close = chart_data[0].get('close', 0) or 0
                last_close = chart_data[-1].get('close', 0) or 0
                
                line_color = '#00ff00' if last_close >= first_close else '#ff4444'
                
                # Fill area
                fill_points = [padding, canvas_height - padding] + points + [canvas_width - padding, canvas_height - padding]
                fill_color = '#1a3a1a' if last_close >= first_close else '#3a1a1a'
                self.price_chart_canvas.create_polygon(
                    fill_points,
                    fill=fill_color, outline=''
                )
                
                # Draw line
                self.price_chart_canvas.create_line(
                    *points,
                    fill=line_color, width=2, smooth=True
                )
            
            # Update stats
            period_high = max(closes)
            period_low = min(closes)
            first_close = chart_data[0].get('close', 0) or 0
            last_close = chart_data[-1].get('close', 0) or 0
            change = ((last_close - first_close) / first_close * 100) if first_close else 0
            
            self.price_chart_labels['high'].config(text=f"₦{period_high:,.2f}")
            self.price_chart_labels['low'].config(text=f"₦{period_low:,.2f}")
            
            change_color = COLORS['gain'] if change >= 0 else COLORS['loss']
            self.price_chart_labels['change'].config(
                text=f"{change:+.1f}%",
                foreground=change_color
            )
            
            change_color = COLORS['gain'] if change >= 0 else COLORS['loss']
            self.price_chart_labels['change'].config(
                text=f"{change:+.1f}%",
                foreground=change_color
            )
            
        except Exception as e:
            logger.error(f"Error updating price chart: {e}")
    
    def _update_volume_profile(self):
        """Update the volume profile visualization."""
        if not self.flow_analysis:
            return
        
        try:
            # Clear canvas
            self.profile_canvas.delete('all')
            
            # Get volume profile
            profile = self.flow_analysis.volume_profile(num_levels=15)
            
            if not profile or 'profile' not in profile:
                return
            
            levels = profile['profile']
            if not levels:
                return
            
            # Get current price (latest close)
            current_price = self.flow_analysis.data[-1].get('close', 0) if self.flow_analysis.data else 0
            
            # Update current price label
            if current_price:
                self.current_price_label.config(text=f"₦{current_price:,.2f}")
            
            # Canvas dimensions
            canvas_width = 180
            canvas_height = 350
            padding = 10
            
            # Price range
            price_range = profile.get('price_range', (0, 1))
            min_price, max_price = price_range
            
            # Find max volume for scaling
            max_vol = max(l['volume'] for l in levels)
            if max_vol == 0:
                return
            
            # Draw bars
            bar_height = (canvas_height - 2 * padding) / len(levels)
            
            poc = profile.get('poc', 0)
            vah = profile.get('vah', 0)
            val = profile.get('val', 0)
            
            # Current price Y position
            current_price_y = None
            
            for i, level in enumerate(reversed(levels)):  # Reversed so higher prices at top
                y = padding + i * bar_height
                bar_width = (level['volume'] / max_vol) * (canvas_width - padding - 40)
                
                # Determine color based on delta
                if level['delta'] > 0:
                    color = '#2a5a2a'  # Green for buying
                else:
                    color = '#5a2a2a'  # Red for selling
                
                # Highlight POC
                is_poc = abs(level['price'] - poc) < (max_price - min_price) / len(levels)
                if is_poc:
                    color = '#4a4a8a'  # Blue for POC
                
                # Draw bar
                self.profile_canvas.create_rectangle(
                    40, y, 40 + bar_width, y + bar_height - 2,
                    fill=color, outline=''
                )
                
                # Price label
                self.profile_canvas.create_text(
                    35, y + bar_height / 2,
                    text=f"{level['price']:.0f}",
                    anchor='e',
                    fill=COLORS['text_muted'],
                    font=('Arial', 7)
                )
                
                # Check if current price is in this level
                level_min = level['price'] - (max_price - min_price) / len(levels) / 2
                level_max = level['price'] + (max_price - min_price) / len(levels) / 2
                if level_min <= current_price <= level_max:
                    current_price_y = y + bar_height / 2
            
            # Draw current price marker (arrow)
            if current_price_y:
                # Draw arrow pointing to the current price level
                self.profile_canvas.create_polygon(
                    5, current_price_y,
                    15, current_price_y - 5,
                    15, current_price_y + 5,
                    fill='#00ffff', outline=''
                )
                self.profile_canvas.create_text(
                    18, current_price_y,
                    text="➤",
                    anchor='w',
                    fill='#00ffff',
                    font=('Arial', 10)
                )
            
            # Update profile labels
            self.profile_labels['poc'].config(text=f"₦{poc:,.2f}")
            self.profile_labels['vah'].config(text=f"₦{vah:,.2f}")
            self.profile_labels['val'].config(text=f"₦{val:,.2f}")
            
            # Calculate and update HVN/LVN (High/Low Volume Nodes)
            if levels:
                sorted_by_vol = sorted(levels, key=lambda x: x['volume'], reverse=True)
                
                # HVN = highest volume level (that's not POC)
                hvn_level = None
                for level in sorted_by_vol:
                    if abs(level['price'] - poc) > (max_price - min_price) / len(levels) * 2:
                        hvn_level = level
                        break
                
                if hvn_level and 'hvn' in self.profile_labels:
                    self.profile_labels['hvn'].config(text=f"₦{hvn_level['price']:,.2f}")
                
                # LVN = lowest volume level
                sorted_by_vol_asc = sorted(levels, key=lambda x: x['volume'])
                lvn_level = sorted_by_vol_asc[0] if sorted_by_vol_asc else None
                
                if lvn_level and 'lvn' in self.profile_labels:
                    self.profile_labels['lvn'].config(text=f"₦{lvn_level['price']:,.2f}")
            
            # Delta at POC
            if levels and 'delta_poc' in self.profile_labels:
                poc_level = min(levels, key=lambda x: abs(x['price'] - poc))
                poc_delta = poc_level.get('delta', 0)
                color = COLORS['gain'] if poc_delta >= 0 else COLORS['loss']
                self.profile_labels['delta_poc'].config(
                    text=f"{poc_delta:+,.0f}",
                    foreground=color
                )
            
            # Delta balance (total buy vs sell delta)
            if levels and 'delta_balance' in self.profile_labels:
                total_buying = sum(l['delta'] for l in levels if l['delta'] > 0)
                total_selling = abs(sum(l['delta'] for l in levels if l['delta'] < 0))
                
                if total_buying > total_selling:
                    balance = f"🟢 {(total_buying/max(total_selling,1)):.1f}x Buy"
                    color = COLORS['gain']
                else:
                    balance = f"🔴 {(total_selling/max(total_buying,1)):.1f}x Sell"
                    color = COLORS['loss']
                
                self.profile_labels['delta_balance'].config(text=balance, foreground=color)
            
        except Exception as e:
            logger.error(f"Error updating volume profile: {e}")
    
    def _update_cumulative_delta(self):
        """Update cumulative delta display."""
        if not self.flow_analysis:
            return
        
        stats = self.flow_analysis.summary_stats()
        cum_delta = stats.get('cumulative_delta', 0)
        
        if cum_delta >= 0:
            self.cum_delta_label.config(
                text=f"+{cum_delta:,.0f}",
                foreground=COLORS['gain']
            )
        else:
            self.cum_delta_label.config(
                text=f"{cum_delta:,.0f}",
                foreground=COLORS['loss']
            )
    
    def _update_stats(self, bar_count, avg_vol, buy_count, block_count, sweep_count, data):
        """Update enhanced stats bar."""
        self.stats_labels['bars'].config(text=f"{bar_count:,}")
        self.stats_labels['avg_volume'].config(text=f"{avg_vol:,.0f}")
        
        buy_pct = (buy_count / bar_count * 100) if bar_count else 0
        sell_pct = 100 - buy_pct
        
        self.stats_labels['buy_pressure'].config(
            text=f"{buy_pct:.1f}%",
            foreground=COLORS['gain'] if buy_pct > 50 else COLORS['loss']
        )
        
        if 'sell_pressure' in self.stats_labels:
            self.stats_labels['sell_pressure'].config(
                text=f"{sell_pct:.1f}%",
                foreground=COLORS['loss'] if sell_pct > 50 else COLORS['gain']
            )
        
        self.stats_labels['block_trades'].config(text=str(block_count))
        
        if 'sweeps' in self.stats_labels:
            self.stats_labels['sweeps'].config(text=str(sweep_count))
        
        # RVOL percentile
        if self.flow_analysis:
            rvol_info = self.flow_analysis.rvol_analysis()
            self.stats_labels['rvol_percentile'].config(
                text=f"{rvol_info.get('percentile', 50):.0f}%"
            )
        
        # Date range
        if data:
            earliest = min(d['datetime'] for d in data)
            latest = max(d['datetime'] for d in data)
            if isinstance(earliest, datetime) and isinstance(latest, datetime):
                self.stats_labels['date_range'].config(
                    text=f"{earliest.strftime('%b %d')} - {latest.strftime('%b %d')}"
                )
    
    def _update_session_analytics(self):
        """Update the session analytics panel."""
        if not self.flow_analysis:
            return
        
        try:
            # Intraday session breakdown
            sessions = self.flow_analysis.intraday_session_breakdown()
            
            for key in ['open', 'core', 'close']:
                session = sessions.get(key, {})
                delta = session.get('delta', 0)
                trend = session.get('trend', 'NO_DATA')
                
                if trend == 'NO_DATA':
                    text = '--'
                    color = COLORS['text_muted']
                else:
                    # Show delta with icon
                    if delta > 0:
                        icon = '↑'
                        color = COLORS['gain']
                    elif delta < 0:
                        icon = '↓'
                        color = COLORS['loss']
                    else:
                        icon = '→'
                        color = COLORS['warning']
                    
                    text = f"{icon} {abs(delta):,.0f}"
                
                self.session_labels[key].config(text=text, foreground=color)
            
            # Opening range status
            or_data = self.flow_analysis.opening_range_analysis()
            if or_data:
                breakout = or_data.get('breakout', 'NO_BREAKOUT')
                if breakout == 'BULLISH_BREAKOUT':
                    text = "↑ Above OR"
                    color = COLORS['gain']
                elif breakout == 'BEARISH_BREAKDOWN':
                    text = "↓ Below OR"
                    color = COLORS['loss']
                else:
                    text = "⬌ Inside OR"
                    color = COLORS['warning']
                
                self.session_labels['or_breakout'].config(text=text, foreground=color)
            
            # Streak info
            comparison = self.flow_analysis.session_comparison()
            if comparison:
                streak = comparison.get('streak', 0)
                streak_type = comparison.get('streak_type', 'BUY')
                
                if streak_type == 'BUY':
                    text = f"🟢 {streak}d Buy"
                    color = COLORS['gain']
                else:
                    text = f"🔴 {streak}d Sell"
                    color = COLORS['loss']
                
                self.session_labels['streak'].config(text=text, foreground=color)
            
        except Exception as e:
            logger.error(f"Error updating session analytics: {e}")
    
    def refresh(self):
        """Refresh the current display."""
        self._load_data()
