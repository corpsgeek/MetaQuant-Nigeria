"""
Flow Tape Tab - Advanced Intraday Trade Flow Visualization.
Institutional-grade flow analysis with volume profile, delta metrics, and pattern detection.
"""

import tkinter as tk
from tkinter import ttk
import logging
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import math

from src.gui.theme import COLORS, get_font
from src.database.db_manager import DatabaseManager
from src.collectors.intraday_collector import IntradayCollector
from src.analysis.flow_analysis import FlowAnalysis

# Try to import AI Insight Engine
try:
    from src.ai.insight_engine import InsightEngine
    INSIGHT_ENGINE_AVAILABLE = True
except ImportError:
    INSIGHT_ENGINE_AVAILABLE = False

# Try to import Pathway Synthesizer
try:
    from src.ml.pathway_synthesizer import PathwaySynthesizer
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False

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
        self.insight_engine = None  # AI Insight Engine for Groq
        
        # Initialize AI Insight Engine if available
        if INSIGHT_ENGINE_AVAILABLE:
            try:
                groq_api_key = os.environ.get('GROQ_API_KEY')
                if groq_api_key:
                    self.insight_engine = InsightEngine(groq_api_key=groq_api_key)
                    logger.info("AI Insight Engine initialized with Groq")
                else:
                    logger.warning("GROQ_API_KEY not found in environment")
            except Exception as e:
                logger.warning(f"Failed to initialize AI Insight Engine: {e}")
        
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
        
        # Tab 5: Trade Signals
        self.signals_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.signals_tab, text="🎯 Signals")
        self._create_signals_tab_content()
        
        # Tab 6: AI Synthesis
        self.synthesis_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.synthesis_tab, text="🤖 AI Synthesis")
        self._create_synthesis_tab_content()
        
        # Tab 7: Pathway Predictions (Pandora Black Box)
        self.pathway_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.pathway_tab, text="🔮 Pathway")
        self._create_pathway_tab_content()
        
        # Note: Fundamentals moved to standalone tab
    
    
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
        """Create super enhanced Alerts sub-tab with comprehensive dashboard."""
        main_frame = ttk.Frame(self.alerts_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # ========== ROW 1: ALERT SUMMARY BAR ==========
        summary_frame = ttk.LabelFrame(main_frame, text="🚨 Alert Summary")
        summary_frame.pack(fill=tk.X, pady=(0, 8))
        
        summary_inner = ttk.Frame(summary_frame)
        summary_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Configure grid weights
        for i in range(5):
            summary_inner.columnconfigure(i, weight=1)
        
        self.alert_counts = {}
        severity_items = [
            ('critical', '🔴 CRITICAL', COLORS['loss']),
            ('high', '🟠 HIGH', '#FF8C00'),
            ('medium', '🟡 MEDIUM', COLORS['warning']),
            ('low', '🟢 LOW', COLORS['gain']),
            ('total', '📊 TOTAL', COLORS['primary'])
        ]
        
        for i, (key, label, color) in enumerate(severity_items):
            card = ttk.Frame(summary_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=5, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('small'),
                     foreground=color).pack(anchor='center', pady=(5, 0))
            
            count_label = ttk.Label(card, text="0", font=get_font('heading'),
                                   foreground=color)
            count_label.pack(anchor='center', pady=(0, 5))
            
            self.alert_counts[key] = count_label
        
        # ========== ROW 2: ACTIVE ALERTS (Scrollable) ==========
        alerts_frame = ttk.LabelFrame(main_frame, text="⚠️ Active Alerts")
        alerts_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
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
        
        # ========== ROW 3: ALERT STATISTICS ==========
        stats_frame = ttk.LabelFrame(main_frame, text="📊 Alert Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 8))
        
        stats_inner = ttk.Frame(stats_frame)
        stats_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Configure equal columns
        for i in range(4):
            stats_inner.columnconfigure(i, weight=1)
        
        self.alert_stats_labels = {}
        stats_sections = [
            ('frequency', '🔥 Most Frequent', [
                ('freq_1', '#1'),
                ('freq_2', '#2'),
                ('freq_3', '#3')
            ]),
            ('signals', '📈 Signal Mix', [
                ('bullish_pct', 'Bullish'),
                ('bearish_pct', 'Bearish'),
                ('neutral_pct', 'Neutral')
            ]),
            ('rvol_stats', '📊 RVOL Stats', [
                ('current_rvol', 'Current'),
                ('avg_rvol', 'Average'),
                ('peak_rvol', 'Peak')
            ]),
            ('metrics', '⏱️ Metrics', [
                ('total_today', 'Today'),
                ('active_now', 'Active'),
                ('last_update', 'Updated')
            ])
        ]
        
        for col, (section_key, section_title, items) in enumerate(stats_sections):
            section = ttk.Frame(stats_inner)
            section.grid(row=0, column=col, padx=10, sticky='nsew')
            
            ttk.Label(section, text=section_title, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='w')
            
            ttk.Separator(section, orient='horizontal').pack(fill=tk.X, pady=3)
            
            for key, label in items:
                row = ttk.Frame(section)
                row.pack(fill=tk.X, pady=1)
                
                ttk.Label(row, text=f"{label}:", font=get_font('tiny'),
                         foreground=COLORS['text_muted']).pack(side=tk.LEFT)
                
                value_label = ttk.Label(row, text="--", font=get_font('small'))
                value_label.pack(side=tk.RIGHT)
                self.alert_stats_labels[key] = value_label
        
        # ========== ROW 4: ALERT HISTORY TABLE ==========
        history_frame = ttk.LabelFrame(main_frame, text="📜 Recent Alert History")
        history_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Create treeview for history
        columns = ('time', 'type', 'message', 'severity', 'signal')
        self.alert_history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=4)
        
        col_config = [
            ('time', 'Time', 60, 'center'),
            ('type', 'Type', 100, 'center'),
            ('message', 'Message', 280, 'w'),
            ('severity', 'Severity', 70, 'center'),
            ('signal', 'Signal', 70, 'center')
        ]
        
        for col, heading, width, anchor in col_config:
            self.alert_history_tree.heading(col, text=heading)
            self.alert_history_tree.column(col, width=width, anchor=anchor)
        
        # Scrollbar
        history_scroll = ttk.Scrollbar(history_frame, orient='vertical', command=self.alert_history_tree.yview)
        self.alert_history_tree.configure(yscrollcommand=history_scroll.set)
        
        self.alert_history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        
        # Tags for coloring
        self.alert_history_tree.tag_configure('critical', foreground=COLORS['loss'])
        self.alert_history_tree.tag_configure('high', foreground='#FF8C00')
        self.alert_history_tree.tag_configure('medium', foreground=COLORS['warning'])
        self.alert_history_tree.tag_configure('low', foreground=COLORS['gain'])
        
        # Initialize alert history storage
        self.alert_history = []
    
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
        """Create super enhanced Sessions sub-tab with comprehensive analytics."""
        main_frame = ttk.Frame(self.sessions_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # ========== ROW 1: TODAY'S SESSION SUMMARY ==========
        summary_frame = ttk.LabelFrame(main_frame, text="📊 Today's Session Summary")
        summary_frame.pack(fill=tk.X, pady=(0, 8))
        
        summary_inner = ttk.Frame(summary_frame)
        summary_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Configure grid weights for equal distribution
        for i in range(5):
            summary_inner.columnconfigure(i, weight=1)
        
        self.session_cards = {}
        session_items = [
            ('open', '🔔 OPENING', '10:00-10:30'),
            ('core', '⚡ CORE', '10:30-13:00'),
            ('close', '🔔 CLOSING', '13:00-14:30'),
            ('current', '📍 CURRENT', 'Position'),
            ('status', '🎯 STATUS', 'Bias')
        ]
        
        for i, (key, title, subtitle) in enumerate(session_items):
            card = ttk.Frame(summary_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=5, pady=3, sticky='nsew')
            
            # Title
            ttk.Label(card, text=title, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(5, 0))
            
            # Subtitle
            ttk.Label(card, text=subtitle, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            # Delta value
            delta_label = ttk.Label(card, text="Δ: --", font=get_font('subheading'))
            delta_label.pack(anchor='center', pady=2)
            
            # Volume/Extra info
            extra_label = ttk.Label(card, text="--", font=get_font('tiny'),
                                   foreground=COLORS['text_muted'])
            extra_label.pack(anchor='center', pady=(0, 5))
            
            self.session_cards[key] = {
                'delta': delta_label,
                'extra': extra_label,
                'frame': card
            }
        
        # ========== ROW 2: LEFT=OPENING RANGE, RIGHT=SESSION DELTA CHART ==========
        row2 = ttk.Frame(main_frame)
        row2.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=2)
        
        # Opening Range Analysis
        or_frame = ttk.LabelFrame(row2, text="📐 Opening Range Analysis")
        or_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        or_inner = ttk.Frame(or_frame)
        or_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        self.or_labels = {}
        or_items = [
            ('or_high', 'OR High', COLORS['gain']),
            ('or_low', 'OR Low', COLORS['loss']),
            ('or_range', 'Range', COLORS['text_primary']),
            ('or_status', 'Status', COLORS['primary']),
            ('or_extension', 'Extension', COLORS['warning'])
        ]
        
        for key, label, color in or_items:
            row = ttk.Frame(or_inner)
            row.pack(fill=tk.X, pady=2)
            
            ttk.Label(row, text=f"{label}:", font=get_font('small'),
                     foreground=COLORS['text_muted'], width=12).pack(side=tk.LEFT)
            
            value_label = ttk.Label(row, text="--", font=get_font('body'),
                                   foreground=color)
            value_label.pack(side=tk.LEFT)
            self.or_labels[key] = value_label
        
        # Session Delta Chart
        chart_frame = ttk.LabelFrame(row2, text="📊 Session Delta Chart")
        chart_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        self.session_chart_canvas = tk.Canvas(
            chart_frame,
            bg=COLORS['bg_medium'],
            highlightthickness=0,
            height=120
        )
        self.session_chart_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ========== ROW 3: HISTORICAL SESSION PATTERNS ==========
        history_frame = ttk.LabelFrame(main_frame, text="📅 Historical Session Patterns")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        # Create treeview for history
        columns = ('date', 'day', 'open_d', 'core_d', 'close_d', 'total', 'result', 'pattern')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=6)
        
        col_config = [
            ('date', 'Date', 80, 'center'),
            ('day', 'Day', 45, 'center'),
            ('open_d', 'Open Δ', 70, 'e'),
            ('core_d', 'Core Δ', 70, 'e'),
            ('close_d', 'Close Δ', 70, 'e'),
            ('total', 'Total Δ', 80, 'e'),
            ('result', 'Result', 55, 'center'),
            ('pattern', 'Pattern', 100, 'center')
        ]
        
        for col, heading, width, anchor in col_config:
            self.history_tree.heading(col, text=heading)
            self.history_tree.column(col, width=width, anchor=anchor)
        
        # Scrollbar
        history_scroll = ttk.Scrollbar(history_frame, orient='vertical', command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        
        # Tags for coloring
        self.history_tree.tag_configure('win', foreground=COLORS['gain'])
        self.history_tree.tag_configure('loss', foreground=COLORS['loss'])
        
        # ========== ROW 4: SESSION ANALYTICS ==========
        analytics_frame = ttk.LabelFrame(main_frame, text="📈 Session Analytics")
        analytics_frame.pack(fill=tk.X, pady=(0, 5))
        
        analytics_inner = ttk.Frame(analytics_frame)
        analytics_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Configure equal columns
        for i in range(4):
            analytics_inner.columnconfigure(i, weight=1)
        
        self.analytics_labels = {}
        analytics_sections = [
            ('win_rates', '🔥 Win Rates', [
                ('open_wr', 'Opening'),
                ('core_wr', 'Core'),
                ('close_wr', 'Closing')
            ]),
            ('avg_delta', '📊 Avg Delta', [
                ('open_avg', 'Opening'),
                ('core_avg', 'Core'),
                ('close_avg', 'Closing')
            ]),
            ('best_time', '⏱️ Best Time', [
                ('best_session', 'Session'),
                ('best_pct', 'Win %'),
                ('total_days', 'Days')
            ]),
            ('patterns', '🎯 Patterns', [
                ('rally_pct', 'Morning Rally'),
                ('reversal_pct', 'Reversal'),
                ('dist_pct', 'Distribution')
            ])
        ]
        
        for col, (section_key, section_title, items) in enumerate(analytics_sections):
            section = ttk.Frame(analytics_inner)
            section.grid(row=0, column=col, padx=10, sticky='nsew')
            
            ttk.Label(section, text=section_title, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='w')
            
            ttk.Separator(section, orient='horizontal').pack(fill=tk.X, pady=3)
            
            for key, label in items:
                row = ttk.Frame(section)
                row.pack(fill=tk.X, pady=1)
                
                ttk.Label(row, text=f"{label}:", font=get_font('tiny'),
                         foreground=COLORS['text_muted']).pack(side=tk.LEFT)
                
                value_label = ttk.Label(row, text="--", font=get_font('small'))
                value_label.pack(side=tk.RIGHT)
                self.analytics_labels[key] = value_label
    
    # =========================================================================
    # SIGNALS TAB
    # =========================================================================
    
    def _create_signals_tab_content(self):
        """Create Trade Signals sub-tab with comprehensive signal dashboard."""
        main_frame = ttk.Frame(self.signals_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # ========== ROW 1: CURRENT SIGNAL STATUS ==========
        signal_frame = ttk.LabelFrame(main_frame, text="🎯 Current Signal Status")
        signal_frame.pack(fill=tk.X, pady=(0, 8))
        
        signal_inner = ttk.Frame(signal_frame)
        signal_inner.pack(fill=tk.X, padx=15, pady=12)
        
        # No signal placeholder (will be replaced dynamically)
        self.signal_status_frame = ttk.Frame(signal_inner)
        self.signal_status_frame.pack(fill=tk.X)
        
        self.no_signal_label = ttk.Label(
            self.signal_status_frame,
            text="⏳ Analyzing... waiting for signals",
            font=get_font('body'),
            foreground=COLORS['text_muted']
        )
        self.no_signal_label.pack(anchor='center', pady=10)
        
        # ========== ROW 2: SIGNAL COMPONENTS ==========
        components_frame = ttk.LabelFrame(main_frame, text="📋 Signal Components")
        components_frame.pack(fill=tk.X, pady=(0, 8))
        
        components_inner = ttk.Frame(components_frame)
        components_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Configure equal columns
        for i in range(4):
            components_inner.columnconfigure(i, weight=1)
        
        self.signal_components = {}
        component_items = [
            ('divergence', '⚡ Delta Divergence', 'Pattern'),
            ('volume', '📊 Volume Profile', 'Level'),
            ('session', '⏱️ Session Trigger', 'Status'),
            ('confluence', '🎯 Confluence', 'Score')
        ]
        
        for i, (key, title, subtitle) in enumerate(component_items):
            card = ttk.Frame(components_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=5, pady=3, sticky='nsew')
            
            ttk.Label(card, text=title, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(5, 0))
            
            ttk.Label(card, text=subtitle, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            value_label = ttk.Label(card, text="--", font=get_font('subheading'))
            value_label.pack(anchor='center', pady=(3, 5))
            
            self.signal_components[key] = value_label
        
        # ========== ROW 3: RECENT SIGNALS TABLE ==========
        history_frame = ttk.LabelFrame(main_frame, text="📜 Recent Signals")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        # Create treeview
        columns = ('time', 'ticker', 'signal', 'pattern', 'entry', 'target', 'stop', 'rr', 'conf', 'status')
        self.signals_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=6)
        
        col_config = [
            ('time', 'Time', 50, 'center'),
            ('ticker', 'Ticker', 70, 'center'),
            ('signal', 'Signal', 50, 'center'),
            ('pattern', 'Pattern', 90, 'center'),
            ('entry', 'Entry', 70, 'e'),
            ('target', 'Target', 70, 'e'),
            ('stop', 'Stop', 65, 'e'),
            ('rr', 'R:R', 45, 'center'),
            ('conf', 'Conf', 40, 'center'),
            ('status', 'Status', 60, 'center')
        ]
        
        for col, heading, width, anchor in col_config:
            self.signals_tree.heading(col, text=heading)
            self.signals_tree.column(col, width=width, anchor=anchor)
        
        # Scrollbar
        signals_scroll = ttk.Scrollbar(history_frame, orient='vertical', command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=signals_scroll.set)
        
        self.signals_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        signals_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        
        # Tags for coloring
        self.signals_tree.tag_configure('buy', foreground=COLORS['gain'])
        self.signals_tree.tag_configure('sell', foreground=COLORS['loss'])
        self.signals_tree.tag_configure('hit', foreground=COLORS['gain'])
        self.signals_tree.tag_configure('stopped', foreground=COLORS['loss'])
        
        # ========== ROW 4: SIGNAL STATISTICS ==========
        stats_frame = ttk.LabelFrame(main_frame, text="📈 Signal Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        
        stats_inner = ttk.Frame(stats_frame)
        stats_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Configure equal columns
        for i in range(4):
            stats_inner.columnconfigure(i, weight=1)
        
        self.signal_stats = {}
        stats_items = [
            ('signals_today', '📊 Today', 'Total Signals'),
            ('buy_sell', '📈 Buy/Sell', 'Ratio'),
            ('avg_conf', '🎯 Avg Conf', 'Confidence'),
            ('best_pattern', '🏆 Best Pattern', 'Type')
        ]
        
        for i, (key, title, subtitle) in enumerate(stats_items):
            card = ttk.Frame(stats_inner)
            card.grid(row=0, column=i, padx=10, sticky='nsew')
            
            ttk.Label(card, text=title, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='w')
            
            ttk.Label(card, text=subtitle, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='w')
            
            value_label = ttk.Label(card, text="--", font=get_font('subheading'))
            value_label.pack(anchor='w', pady=(3, 0))
            
            self.signal_stats[key] = value_label
        
        # Initialize signal history storage
        self.signal_history = []
    
    # =========================================================================
    # AI SYNTHESIS TAB
    # =========================================================================
    
    def _create_synthesis_tab_content(self):
        """Create SUPER SUPER SUPER enhanced AI Synthesis sub-tab with comprehensive dashboard."""
        # Main scrollable frame - FULL WIDTH
        self.synth_canvas = tk.Canvas(self.synthesis_tab, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.synthesis_tab, orient="vertical", command=self.synth_canvas.yview)
        self.synthesis_scrollable = ttk.Frame(self.synth_canvas)
        
        self.synthesis_scrollable.bind(
            "<Configure>",
            lambda e: self.synth_canvas.configure(scrollregion=self.synth_canvas.bbox("all"))
        )
        
        # Bind canvas width to update scrollable frame width
        def _on_canvas_configure(event):
            self.synth_canvas.itemconfig(self.synth_canvas_window, width=event.width)
        self.synth_canvas.bind("<Configure>", _on_canvas_configure)
        
        self.synth_canvas_window = self.synth_canvas.create_window((0, 0), window=self.synthesis_scrollable, anchor="nw")
        self.synth_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.synth_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            self.synth_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.synth_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main = self.synthesis_scrollable
        
        # ========== HEADER WITH SYMBOL & REFRESH ==========
        header = ttk.Frame(main)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="🤖 AI Flow Synthesis", font=get_font('subheading'),
                  foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        self.synth_symbol_label = ttk.Label(header, text="-- | --", font=get_font('body'),
                                            foreground=COLORS['text_secondary'])
        self.synth_symbol_label.pack(side=tk.LEFT, padx=20)
        
        refresh_btn = ttk.Button(header, text="↻ Refresh", command=self._refresh_synthesis)
        refresh_btn.pack(side=tk.RIGHT)
        
        # ========== ROW 1: SYNTHESIS OVERVIEW ==========
        overview_frame = ttk.LabelFrame(main, text="🧠 Synthesis Overview")
        overview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        overview_cards = ttk.Frame(overview_frame)
        overview_cards.pack(fill=tk.X, padx=10, pady=10)
        
        self.synthesis_overview = {}
        
        overview_items = [
            ('score', '🎯 Score', '/100'),
            ('bias', '📊 Bias', ''),
            ('confidence', '💪 Conf', '%'),
            ('action', '⚡ Action', '')
        ]
        
        for key, title, suffix in overview_items:
            card = ttk.Frame(overview_cards, relief='ridge', borderwidth=1)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
            
            ttk.Label(card, text=title, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='w', padx=10, pady=(5, 0))
            
            value_frame = ttk.Frame(card)
            value_frame.pack(anchor='w', padx=10)
            
            value_label = ttk.Label(value_frame, text="--", font=get_font('heading'),
                                   foreground=COLORS['primary'])
            value_label.pack(side=tk.LEFT)
            
            if suffix:
                ttk.Label(value_frame, text=suffix, font=get_font('body'),
                         foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            driver_label = ttk.Label(card, text="--", font=get_font('small'),
                                    foreground=COLORS['text_secondary'], wraplength=150)
            driver_label.pack(anchor='w', padx=10, pady=(0, 5))
            
            self.synthesis_overview[key] = {'value': value_label, 'drivers': driver_label}
        
        # ========== ROW 2: PRICE & VWAP DASHBOARD ==========
        vwap_frame = ttk.LabelFrame(main, text="💰 Price & VWAP Dashboard")
        vwap_frame.pack(fill=tk.X, padx=10, pady=5)
        
        vwap_row = ttk.Frame(vwap_frame)
        vwap_row.pack(fill=tk.X, padx=10, pady=5)
        
        self.synth_vwap = {}
        
        # Price Card
        p_card = ttk.Frame(vwap_row, relief='ridge', borderwidth=1, padding=5)
        p_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(p_card, text="📈 Current", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_vwap['price'] = ttk.Label(p_card, text="₦--", font=get_font('subheading'))
        self.synth_vwap['price'].pack()
        self.synth_vwap['change'] = ttk.Label(p_card, text="--", font=get_font('small'))
        self.synth_vwap['change'].pack()
        
        # VWAP Card
        v_card = ttk.Frame(vwap_row, relief='ridge', borderwidth=1, padding=5)
        v_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(v_card, text="📊 VWAP", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_vwap['vwap'] = ttk.Label(v_card, text="₦--", font=get_font('subheading'))
        self.synth_vwap['vwap'].pack()
        self.synth_vwap['vwap_diff'] = ttk.Label(v_card, text="--", font=get_font('small'))
        self.synth_vwap['vwap_diff'].pack()
        
        # Upper Band Card
        ub_card = ttk.Frame(vwap_row, relief='ridge', borderwidth=1, padding=5)
        ub_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(ub_card, text="⬆️ +1σ Band", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_vwap['upper1'] = ttk.Label(ub_card, text="₦--", font=get_font('body'))
        self.synth_vwap['upper1'].pack()
        self.synth_vwap['upper2'] = ttk.Label(ub_card, text="+2σ: ₦--", font=get_font('small'), foreground=COLORS['text_muted'])
        self.synth_vwap['upper2'].pack()
        
        # Lower Band Card
        lb_card = ttk.Frame(vwap_row, relief='ridge', borderwidth=1, padding=5)
        lb_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(lb_card, text="⬇️ -1σ Band", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_vwap['lower1'] = ttk.Label(lb_card, text="₦--", font=get_font('body'))
        self.synth_vwap['lower1'].pack()
        self.synth_vwap['lower2'] = ttk.Label(lb_card, text="-2σ: ₦--", font=get_font('small'), foreground=COLORS['text_muted'])
        self.synth_vwap['lower2'].pack()
        
        # VWAP Position Indicator
        vwap_pos_frame = ttk.Frame(vwap_frame)
        vwap_pos_frame.pack(fill=tk.X, padx=10, pady=5)
        self.synth_vwap['position_bar'] = tk.Canvas(vwap_pos_frame, height=20, bg=COLORS['bg_medium'], highlightthickness=0)
        self.synth_vwap['position_bar'].pack(fill=tk.X)
        
        # ========== ROW 3: DELTA ANALYSIS ==========
        delta_frame = ttk.LabelFrame(main, text="📉 Delta Analysis")
        delta_frame.pack(fill=tk.X, padx=10, pady=5)
        
        delta_row = ttk.Frame(delta_frame)
        delta_row.pack(fill=tk.X, padx=10, pady=5)
        
        self.synth_delta = {}
        
        delta_items = [
            ('cum_delta', '∑ Cumulative', 'Total Delta'),
            ('trend', '📈 5-Bar Trend', 'Direction'),
            ('momentum', '⚡ Momentum', 'Speed'),
            ('zscore', '📊 Z-Score', 'Standard Dev'),
            ('divergence', '🔀 Divergence', 'Price vs Delta')
        ]
        
        for key, title, subtitle in delta_items:
            card = ttk.Frame(delta_row, relief='ridge', borderwidth=1, padding=5)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
            ttk.Label(card, text=title, font=get_font('small'), foreground=COLORS['text_muted']).pack()
            self.synth_delta[key] = ttk.Label(card, text="--", font=get_font('subheading'))
            self.synth_delta[key].pack()
            ttk.Label(card, text=subtitle, font=get_font('small'), foreground=COLORS['text_muted']).pack()
        
        # ========== ROW 4: VOLUME PROFILE ==========
        profile_frame = ttk.LabelFrame(main, text="📊 Volume Profile")
        profile_frame.pack(fill=tk.X, padx=10, pady=5)
        
        profile_row = ttk.Frame(profile_frame)
        profile_row.pack(fill=tk.X, padx=10, pady=5)
        
        self.synth_profile = {}
        
        # POC Card
        poc_card = ttk.Frame(profile_row, relief='ridge', borderwidth=1, padding=5)
        poc_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(poc_card, text="🎯 POC", font=get_font('small'), foreground=COLORS['warning']).pack()
        self.synth_profile['poc'] = ttk.Label(poc_card, text="₦--", font=get_font('subheading'))
        self.synth_profile['poc'].pack()
        self.synth_profile['poc_vol'] = ttk.Label(poc_card, text="Vol: --", font=get_font('small'), foreground=COLORS['text_muted'])
        self.synth_profile['poc_vol'].pack()
        
        # VAH Card
        vah_card = ttk.Frame(profile_row, relief='ridge', borderwidth=1, padding=5)
        vah_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(vah_card, text="⬆️ VAH", font=get_font('small'), foreground=COLORS['loss']).pack()
        self.synth_profile['vah'] = ttk.Label(vah_card, text="₦--", font=get_font('subheading'))
        self.synth_profile['vah'].pack()
        
        # VAL Card
        val_card = ttk.Frame(profile_row, relief='ridge', borderwidth=1, padding=5)
        val_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(val_card, text="⬇️ VAL", font=get_font('small'), foreground=COLORS['gain']).pack()
        self.synth_profile['val'] = ttk.Label(val_card, text="₦--", font=get_font('subheading'))
        self.synth_profile['val'].pack()
        
        # Price Position
        pos_card = ttk.Frame(profile_row, relief='ridge', borderwidth=1, padding=5)
        pos_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(pos_card, text="📍 Position", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_profile['position'] = ttk.Label(pos_card, text="--", font=get_font('body'))
        self.synth_profile['position'].pack()
        
        # ========== ROW 5: FLOW PRESSURE GAUGES ==========
        flow_frame = ttk.LabelFrame(main, text="💹 Flow Pressure")
        flow_frame.pack(fill=tk.X, padx=10, pady=5)
        
        flow_row = ttk.Frame(flow_frame)
        flow_row.pack(fill=tk.X, padx=10, pady=5)
        
        self.synth_flow = {}
        
        # Block Trades
        block_card = ttk.Frame(flow_row, relief='ridge', borderwidth=1, padding=5)
        block_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(block_card, text="🏛 Block Trades", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_flow['blocks'] = ttk.Label(block_card, text="--", font=get_font('subheading'))
        self.synth_flow['blocks'].pack()
        self.synth_flow['block_bias'] = ttk.Label(block_card, text="--", font=get_font('small'))
        self.synth_flow['block_bias'].pack()
        
        # RVOL
        rvol_card = ttk.Frame(flow_row, relief='ridge', borderwidth=1, padding=5)
        rvol_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(rvol_card, text="📊 RVOL", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_flow['rvol'] = ttk.Label(rvol_card, text="--x", font=get_font('subheading'))
        self.synth_flow['rvol'].pack()
        self.synth_flow['rvol_status'] = ttk.Label(rvol_card, text="--", font=get_font('small'))
        self.synth_flow['rvol_status'].pack()
        
        # Session Bias
        session_card = ttk.Frame(flow_row, relief='ridge', borderwidth=1, padding=5)
        session_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(session_card, text="📅 Session", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_flow['session_delta'] = ttk.Label(session_card, text="--", font=get_font('subheading'))
        self.synth_flow['session_delta'].pack()
        self.synth_flow['session_bias'] = ttk.Label(session_card, text="--", font=get_font('small'))
        self.synth_flow['session_bias'].pack()
        
        # Flow Gauge
        gauge_card = ttk.Frame(flow_row, relief='ridge', borderwidth=1, padding=5)
        gauge_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        ttk.Label(gauge_card, text="⚖️ Flow Balance", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.synth_flow['gauge'] = tk.Canvas(gauge_card, width=120, height=25, bg=COLORS['bg_medium'], highlightthickness=0)
        self.synth_flow['gauge'].pack(pady=3)
        self.synth_flow['gauge_label'] = ttk.Label(gauge_card, text="--", font=get_font('small'))
        self.synth_flow['gauge_label'].pack()
        
        # ========== ROW 6: COMPONENT SCORES ==========
        components_frame = ttk.LabelFrame(main, text="📈 Component Scores")
        components_frame.pack(fill=tk.X, padx=10, pady=5)
        
        components_cards = ttk.Frame(components_frame)
        components_cards.pack(fill=tk.X, padx=10, pady=10)
        
        self.synthesis_components = {}
        
        component_items = [
            ('tape', '📋 Tape'),
            ('alerts', '⚠️ Alerts'),
            ('charts', '📈 Charts'),
            ('sessions', '📅 Sessions'),
            ('signals', '🎯 Signals')
        ]
        
        for key, title in component_items:
            card = ttk.Frame(components_cards, relief='ridge', borderwidth=1, padding=5)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)
            
            ttk.Label(card, text=title, font=get_font('small'), foreground=COLORS['text_muted']).pack()
            
            score_frame = ttk.Frame(card)
            score_frame.pack()
            score_label = ttk.Label(score_frame, text="--", font=get_font('heading'), foreground=COLORS['primary'])
            score_label.pack(side=tk.LEFT)
            ttk.Label(score_frame, text="/10", font=get_font('small'), foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            bar_canvas = tk.Canvas(card, width=100, height=8, bg=COLORS['bg_medium'], highlightthickness=0)
            bar_canvas.pack(pady=3)
            
            driver_label = ttk.Label(card, text="--", font=get_font('small'), foreground=COLORS['text_secondary'], wraplength=100)
            driver_label.pack()
            
            self.synthesis_components[key] = {'score': score_label, 'bar': bar_canvas, 'drivers': driver_label}
        
        # ========== ROW 7: AI NARRATIVE REPORT ==========
        narrative_frame = ttk.LabelFrame(main, text="📝 AI Narrative Report")
        narrative_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        text_container = ttk.Frame(narrative_frame)
        text_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.narrative_text = tk.Text(
            text_container,
            wrap=tk.WORD,
            font=get_font('body'),
            bg=COLORS['bg_medium'],
            fg=COLORS['text_primary'],
            padx=10,
            pady=10,
            height=12,
            state='disabled'
        )
        self.narrative_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        narrative_scroll = ttk.Scrollbar(text_container, command=self.narrative_text.yview)
        narrative_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.narrative_text.config(yscrollcommand=narrative_scroll.set)
        
        # ========== ROW 8: KEY INSIGHTS ==========
        insights_frame = ttk.LabelFrame(main, text="💡 Key Insights")
        insights_frame.pack(fill=tk.X, padx=10, pady=5)
        
        insights_row = ttk.Frame(insights_frame)
        insights_row.pack(fill=tk.X, padx=10, pady=5)
        
        self.synth_insights = []
        for i in range(6):
            card = ttk.Frame(insights_row, padding=3)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            icon_lbl = ttk.Label(card, text="⚪", font=('', 12))
            icon_lbl.pack()
            text_lbl = ttk.Label(card, text="--", font=get_font('small'), foreground=COLORS['text_primary'])
            text_lbl.pack()
            
            self.synth_insights.append({'icon': icon_lbl, 'text': text_lbl})
        
        self.insights_container = insights_row  # Keep for backward compatibility
        
        # ========== ROW 9: SIGNAL SUMMARY ==========
        signal_summary_frame = ttk.LabelFrame(main, text="🎯 Signal Summary")
        signal_summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        signal_content = ttk.Frame(signal_summary_frame)
        signal_content.pack(fill=tk.X, padx=10, pady=10)
        
        signal_left = ttk.Frame(signal_content)
        signal_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.synthesis_signal = {}
        
        signal_row = ttk.Frame(signal_left)
        signal_row.pack(fill=tk.X)
        
        self.synthesis_signal['icon'] = ttk.Label(signal_row, text="⚪", font=get_font('heading'))
        self.synthesis_signal['icon'].pack(side=tk.LEFT)
        
        self.synthesis_signal['type'] = ttk.Label(signal_row, text="NO SIGNAL", font=get_font('heading'), foreground=COLORS['text_muted'])
        self.synthesis_signal['type'].pack(side=tk.LEFT, padx=(5, 15))
        
        self.synthesis_signal['pattern'] = ttk.Label(signal_row, text="--", font=get_font('body'), foreground=COLORS['text_secondary'])
        self.synthesis_signal['pattern'].pack(side=tk.LEFT)
        
        levels_row = ttk.Frame(signal_left)
        levels_row.pack(fill=tk.X, pady=(5, 0))
        
        self.synthesis_signal['levels'] = ttk.Label(levels_row, 
            text="Entry: -- | Target: -- | Stop: -- | R:R: --",
            font=get_font('body'), foreground=COLORS['text_secondary'])
        self.synthesis_signal['levels'].pack(side=tk.LEFT)
        
        signal_right = ttk.Frame(signal_content)
        signal_right.pack(side=tk.RIGHT)
        
        ttk.Label(signal_right, text="Confluence", font=get_font('small'), foreground=COLORS['text_muted']).pack(anchor='e')
        
        confluence_row = ttk.Frame(signal_right)
        confluence_row.pack(anchor='e')
        
        self.synthesis_signal['confluence_bar'] = tk.Canvas(confluence_row, width=150, height=15, bg=COLORS['bg_medium'], highlightthickness=0)
        self.synthesis_signal['confluence_bar'].pack(side=tk.LEFT, padx=(0, 5))
        
        self.synthesis_signal['confluence_pct'] = ttk.Label(confluence_row, text="0%", font=get_font('body'), foreground=COLORS['text_muted'])
        self.synthesis_signal['confluence_pct'].pack(side=tk.LEFT)
        
        # ========== ROW 10: ALERTS SUMMARY ==========
        alerts_frame = ttk.LabelFrame(main, text="🚨 Active Alerts")
        alerts_frame.pack(fill=tk.X, padx=10, pady=5)
        
        alerts_row = ttk.Frame(alerts_frame)
        alerts_row.pack(fill=tk.X, padx=10, pady=5)
        
        self.synth_alerts = []
        for i in range(4):
            alert_card = ttk.Frame(alerts_row, padding=3)
            alert_card.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            icon_lbl = ttk.Label(alert_card, text="⚪", font=('', 12))
            icon_lbl.pack(side=tk.LEFT)
            text_lbl = ttk.Label(alert_card, text="--", font=get_font('small'), foreground=COLORS['text_muted'])
            text_lbl.pack(side=tk.LEFT, padx=5)
            
            self.synth_alerts.append({'icon': icon_lbl, 'text': text_lbl})
        
        # ========== ROW 11: LIVE STATUS BAR ==========
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.synthesis_status = {}
        
        self.synthesis_status['live'] = ttk.Label(status_frame, text="📡 Waiting...", font=get_font('small'), foreground=COLORS['text_muted'])
        self.synthesis_status['live'].pack(side=tk.LEFT)
        
        self.synthesis_status['ai'] = ttk.Label(status_frame, text="Powered by Groq AI", font=get_font('small'), foreground=COLORS['text_muted'])
        self.synthesis_status['ai'].pack(side=tk.LEFT, padx=20)
        
        self.synthesis_status['update'] = ttk.Label(status_frame, text="Last Update: --", font=get_font('small'), foreground=COLORS['text_muted'])
        self.synthesis_status['update'].pack(side=tk.RIGHT)
    
    # =========================================================================
    # PATHWAY PREDICTIONS TAB (PANDORA BLACK BOX)
    # =========================================================================
    
    def _create_pathway_tab_content(self):
        """Create Pandora Black Box - Price Pathway Predictions sub-tab."""
        # Initialize pathway synthesizer
        self.pathway_synth = None
        if PATHWAY_AVAILABLE:
            try:
                self.pathway_synth = PathwaySynthesizer(self.db)
                logger.info("PathwaySynthesizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PathwaySynthesizer: {e}")
        
        # Main scrollable frame
        canvas = tk.Canvas(self.pathway_tab, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.pathway_tab, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_win = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        main = scrollable
        
        # ========== HEADER ==========
        header = ttk.Frame(main)
        header.pack(fill=tk.X, padx=15, pady=10)
        
        ttk.Label(header, text="🔮 Pandora Black Box - Price Pathway Predictions",
                  font=get_font('subheading'), foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        self.pathway_symbol_label = ttk.Label(header, text="Select a symbol", 
                                              font=get_font('body'), foreground=COLORS['text_secondary'])
        self.pathway_symbol_label.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(header, text="🔄 Refresh Pathway", 
                   command=self._refresh_pathway).pack(side=tk.RIGHT)
        
        # ========== ROW 1: HORIZON PREDICTION CARDS ==========
        horizons_frame = ttk.LabelFrame(main, text="🎯 Multi-Horizon Price Pathways")
        horizons_frame.pack(fill=tk.X, padx=15, pady=8)
        
        horizons_row = ttk.Frame(horizons_frame)
        horizons_row.pack(fill=tk.X, padx=10, pady=10)
        
        for i in range(4):
            horizons_row.columnconfigure(i, weight=1)
        
        self.pathway_cards = {}
        horizon_items = [
            ('2D', '2 Days'),
            ('3D', '3 Days'),
            ('1W', '1 Week'),
            ('1M', '1 Month')
        ]
        
        for i, (key, label) in enumerate(horizon_items):
            card = ttk.Frame(horizons_row, relief='ridge', borderwidth=2)
            card.grid(row=0, column=i, padx=5, pady=5, sticky='nsew')
            
            # Header
            ttk.Label(card, text=f"📊 {label}", font=get_font('body'),
                      foreground=COLORS['primary']).pack(anchor='center', pady=(8, 5))
            
            # Expected price
            price_label = ttk.Label(card, text="₦--", font=get_font('heading'))
            price_label.pack(anchor='center')
            
            # Return %
            return_label = ttk.Label(card, text="+0.0%", font=get_font('body'))
            return_label.pack(anchor='center')
            
            ttk.Separator(card, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
            
            # Scenarios frame
            scenarios_frame = ttk.Frame(card)
            scenarios_frame.pack(fill=tk.X, padx=10)
            
            # Bull
            bull_row = ttk.Frame(scenarios_frame)
            bull_row.pack(fill=tk.X, pady=2)
            ttk.Label(bull_row, text="🐂", font=get_font('small')).pack(side=tk.LEFT)
            bull_price = ttk.Label(bull_row, text="₦--", font=get_font('small'), foreground=COLORS['gain'])
            bull_price.pack(side=tk.LEFT, padx=5)
            bull_prob = ttk.Label(bull_row, text="(--%)  ", font=get_font('tiny'), foreground=COLORS['text_muted'])
            bull_prob.pack(side=tk.RIGHT)
            
            # Base
            base_row = ttk.Frame(scenarios_frame)
            base_row.pack(fill=tk.X, pady=2)
            ttk.Label(base_row, text="➡️", font=get_font('small')).pack(side=tk.LEFT)
            base_price = ttk.Label(base_row, text="₦--", font=get_font('small'))
            base_price.pack(side=tk.LEFT, padx=5)
            base_prob = ttk.Label(base_row, text="(--%)  ", font=get_font('tiny'), foreground=COLORS['text_muted'])
            base_prob.pack(side=tk.RIGHT)
            
            # Bear
            bear_row = ttk.Frame(scenarios_frame)
            bear_row.pack(fill=tk.X, pady=2)
            ttk.Label(bear_row, text="🐻", font=get_font('small')).pack(side=tk.LEFT)
            bear_price = ttk.Label(bear_row, text="₦--", font=get_font('small'), foreground=COLORS['loss'])
            bear_price.pack(side=tk.LEFT, padx=5)
            bear_prob = ttk.Label(bear_row, text="(--%)  ", font=get_font('tiny'), foreground=COLORS['text_muted'])
            bear_prob.pack(side=tk.RIGHT)
            
            # Confidence
            conf_label = ttk.Label(card, text="Confidence: --%", font=get_font('tiny'), 
                                   foreground=COLORS['text_muted'])
            conf_label.pack(anchor='center', pady=(5, 8))
            
            self.pathway_cards[key] = {
                'price': price_label,
                'return': return_label,
                'bull_price': bull_price,
                'bull_prob': bull_prob,
                'base_price': base_price,
                'base_prob': base_prob,
                'bear_price': bear_price,
                'bear_prob': bear_prob,
                'confidence': conf_label
            }
        
        # ========== ROW 2: BID/OFFER PROBABILITY ==========
        bidoffer_frame = ttk.LabelFrame(main, text="📈 Session Close Probability")
        bidoffer_frame.pack(fill=tk.X, padx=15, pady=8)
        
        bo_row = ttk.Frame(bidoffer_frame)
        bo_row.pack(fill=tk.X, padx=20, pady=15)
        
        self.bidoffer_labels = {}
        
        # Full Bid
        bid_frame = ttk.Frame(bo_row)
        bid_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Label(bid_frame, text="📗 Full Bid", font=get_font('body'), 
                  foreground=COLORS['gain']).pack()
        self.bidoffer_labels['bid'] = ttk.Label(bid_frame, text="--%", 
                                                 font=get_font('heading'), foreground=COLORS['gain'])
        self.bidoffer_labels['bid'].pack()
        
        # Canvas for gauge
        self.bidoffer_canvas = tk.Canvas(bo_row, width=400, height=40, 
                                          bg=COLORS['bg_dark'], highlightthickness=0)
        self.bidoffer_canvas.pack(side=tk.LEFT, expand=True, padx=20)
        
        # Mixed
        mixed_frame = ttk.Frame(bo_row)
        mixed_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Label(mixed_frame, text="⚖️ Mixed", font=get_font('body')).pack()
        self.bidoffer_labels['mixed'] = ttk.Label(mixed_frame, text="--%", font=get_font('heading'))
        self.bidoffer_labels['mixed'].pack()
        
        # Full Offer
        offer_frame = ttk.Frame(bo_row)
        offer_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Label(offer_frame, text="📕 Full Offer", font=get_font('body'),
                  foreground=COLORS['loss']).pack()
        self.bidoffer_labels['offer'] = ttk.Label(offer_frame, text="--%", 
                                                   font=get_font('heading'), foreground=COLORS['loss'])
        self.bidoffer_labels['offer'].pack()
        
        # ========== ROW 3: SIGNAL BREAKDOWN ==========
        signals_frame = ttk.LabelFrame(main, text="🔍 Signal Contribution")
        signals_frame.pack(fill=tk.X, padx=15, pady=8)
        
        signals_row = ttk.Frame(signals_frame)
        signals_row.pack(fill=tk.X, padx=10, pady=10)
        
        for i in range(6):
            signals_row.columnconfigure(i, weight=1)
        
        self.signal_cards = {}
        signal_items = [
            ('ml', '🤖 ML Ensemble', 'ml_ensemble'),
            ('flow', '📊 Flow Delta', 'flow_delta'),
            ('sector', '🔄 Sector', 'sector_momentum'),
            ('fund', '💰 Fundamentals', 'fundamentals'),
            ('tech', '📈 Technicals', 'technicals'),
            ('disc', '📋 Disclosures', 'disclosures')
        ]
        
        for i, (key, label, weight_key) in enumerate(signal_items):
            card = ttk.Frame(signals_row, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=3, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('small'),
                      foreground=COLORS['text_muted']).pack(anchor='center', pady=(5, 2))
            
            # Signal direction
            signal_label = ttk.Label(card, text="--", font=get_font('body'))
            signal_label.pack(anchor='center')
            
            # Contribution bar (simple canvas)
            bar_canvas = tk.Canvas(card, width=80, height=12, 
                                   bg=COLORS['bg_secondary'], highlightthickness=0)
            bar_canvas.pack(pady=(2, 5))
            
            self.signal_cards[key] = {
                'signal': signal_label,
                'bar': bar_canvas
            }
        
        # ========== ROW 4: AI NARRATIVE ==========
        narrative_frame = ttk.LabelFrame(main, text="💬 AI Narrative")
        narrative_frame.pack(fill=tk.X, padx=15, pady=8)
        
        self.pathway_narrative = tk.Text(narrative_frame, height=4, wrap=tk.WORD,
                                          bg=COLORS['bg_secondary'], fg=COLORS['text_primary'],
                                          font=get_font('body'), relief='flat')
        self.pathway_narrative.pack(fill=tk.X, padx=10, pady=10)
        self.pathway_narrative.insert('1.0', "Select a symbol and click Refresh to generate pathway predictions...")
        self.pathway_narrative.config(state='disabled')
        
        # ========== STATUS BAR ==========
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.pathway_status = ttk.Label(status_frame, text="⏳ Waiting for symbol selection...",
                                         font=get_font('small'), foreground=COLORS['text_muted'])
        self.pathway_status.pack(side=tk.LEFT)
        
        self.pathway_timestamp = ttk.Label(status_frame, text="",
                                            font=get_font('small'), foreground=COLORS['text_muted'])
        self.pathway_timestamp.pack(side=tk.RIGHT)
    
    def _refresh_pathway(self):
        """Refresh pathway predictions for current symbol."""
        if not self.current_symbol:
            self.pathway_status.configure(text="⚠️ Select a symbol first")
            return
        
        if not self.pathway_synth:
            self.pathway_status.configure(text="⚠️ Pathway synthesizer not available")
            return
        
        self.pathway_status.configure(text=f"🔄 Generating pathway for {self.current_symbol}...")
        self.pathway_symbol_label.configure(text=self.current_symbol)
        
        import threading
        def generate():
            try:
                result = self.pathway_synth.synthesize(self.current_symbol)
                self.frame.after(0, lambda: self._display_pathway(result))
            except Exception as e:
                logger.error(f"Pathway generation failed: {e}")
                self.frame.after(0, lambda: self.pathway_status.configure(
                    text=f"❌ Error: {str(e)[:50]}"))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def _display_pathway(self, result: Dict):
        """Display pathway prediction results."""
        if 'error' in result:
            self.pathway_status.configure(text=f"⚠️ {result['error']}")
            return
        
        # Update horizon cards
        predictions = result.get('predictions', {})
        for horizon, data in predictions.items():
            if horizon in self.pathway_cards:
                card = self.pathway_cards[horizon]
                
                # Price and return
                exp_price = data.get('expected_price', 0)
                exp_ret = data.get('expected_return', 0)
                card['price'].configure(text=f"₦{exp_price:,.2f}")
                
                ret_color = COLORS['gain'] if exp_ret >= 0 else COLORS['loss']
                card['return'].configure(text=f"{'+' if exp_ret >= 0 else ''}{exp_ret}%", foreground=ret_color)
                
                # Bull scenario
                bull = data.get('bull', {})
                card['bull_price'].configure(text=f"₦{bull.get('price', 0):,.2f}")
                card['bull_prob'].configure(text=f"({bull.get('probability', 0)}%)")
                
                # Base scenario
                base = data.get('base', {})
                card['base_price'].configure(text=f"₦{base.get('price', 0):,.2f}")
                card['base_prob'].configure(text=f"({base.get('probability', 0)}%)")
                
                # Bear scenario
                bear = data.get('bear', {})
                card['bear_price'].configure(text=f"₦{bear.get('price', 0):,.2f}")
                card['bear_prob'].configure(text=f"({bear.get('probability', 0)}%)")
                
                # Confidence
                conf = data.get('confidence', 50)
                card['confidence'].configure(text=f"Confidence: {conf}%")
        
        # Update bid/offer gauge
        bidoffer = result.get('bid_offer', {})
        bid_pct = bidoffer.get('full_bid', 33)
        mixed_pct = bidoffer.get('mixed', 34)
        offer_pct = bidoffer.get('full_offer', 33)
        
        self.bidoffer_labels['bid'].configure(text=f"{bid_pct}%")
        self.bidoffer_labels['mixed'].configure(text=f"{mixed_pct}%")
        self.bidoffer_labels['offer'].configure(text=f"{offer_pct}%")
        
        # Draw gauge
        self._draw_bidoffer_gauge(bid_pct, mixed_pct, offer_pct)
        
        # Update signal cards
        signals = result.get('signals', {})
        self._update_signal_cards(signals)
        
        # Update narrative
        narrative = result.get('narrative', '')
        self.pathway_narrative.config(state='normal')
        self.pathway_narrative.delete('1.0', tk.END)
        self.pathway_narrative.insert('1.0', narrative)
        self.pathway_narrative.config(state='disabled')
        
        # Update status
        self.pathway_status.configure(text=f"✅ Pathway generated for {self.current_symbol}")
        self.pathway_timestamp.configure(text=f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    def _draw_bidoffer_gauge(self, bid_pct: float, mixed_pct: float, offer_pct: float):
        """Draw the bid/offer probability gauge."""
        canvas = self.bidoffer_canvas
        canvas.delete('all')
        
        w = 400
        h = 30
        y = 5
        
        # Normalize
        total = bid_pct + mixed_pct + offer_pct
        if total > 0:
            bid_w = (bid_pct / total) * w
            mixed_w = (mixed_pct / total) * w
            offer_w = (offer_pct / total) * w
        else:
            bid_w = mixed_w = offer_w = w / 3
        
        # Draw bars
        x = 0
        canvas.create_rectangle(x, y, x + bid_w, y + h, fill='#27ae60', outline='')
        x += bid_w
        canvas.create_rectangle(x, y, x + mixed_w, y + h, fill='#7f8c8d', outline='')
        x += mixed_w
        canvas.create_rectangle(x, y, x + offer_w, y + h, fill='#e74c3c', outline='')
    
    def _update_signal_cards(self, signals: Dict):
        """Update signal contribution cards."""
        mapping = {
            'ml': 'ml',
            'flow': 'flow',
            'sector': 'sector',
            'fund': 'fundamental',
            'tech': 'technical',
            'disc': 'disclosure'
        }
        
        for card_key, signal_key in mapping.items():
            if card_key in self.signal_cards:
                signal_data = signals.get(signal_key, {})
                card = self.signal_cards[card_key]
                
                # Get primary signal value
                if card_key == 'ml':
                    val = signal_data.get('direction', 0)
                elif card_key == 'flow':
                    val = signal_data.get('delta_direction', 0)
                elif card_key == 'sector':
                    val = signal_data.get('sector_momentum', 0)
                elif card_key == 'fund':
                    val = signal_data.get('value_score', 0)
                elif card_key == 'tech':
                    val = (signal_data.get('rsi_signal', 0) + signal_data.get('trend_signal', 0)) / 2
                else:
                    val = signal_data.get('recent_impact', 0)
                
                # Display
                if val > 0.2:
                    display = "📈 Bullish"
                    color = COLORS['gain']
                elif val < -0.2:
                    display = "📉 Bearish"
                    color = COLORS['loss']
                else:
                    display = "➡️ Neutral"
                    color = COLORS['text_muted']
                
                card['signal'].configure(text=display, foreground=color)
                
                # Draw bar
                bar = card['bar']
                bar.delete('all')
                bar_val = (val + 1) / 2 * 80  # Scale -1 to 1 → 0 to 80
                if val > 0:
                    bar.create_rectangle(40, 2, 40 + bar_val/2, 10, fill=COLORS['gain'], outline='')
                else:
                    bar.create_rectangle(40 + bar_val/2, 2, 40, 10, fill=COLORS['loss'], outline='')
    
    # =========================================================================
    # FUNDAMENTALS TAB
    # =========================================================================

    
    def _create_fundamentals_tab_content(self):
        """Create Fundamentals sub-tab with key financial metrics."""
        main_frame = ttk.Frame(self.fundamentals_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # ========== ROW 1: KEY METRICS HEADER ==========
        header_frame = ttk.LabelFrame(main_frame, text="📊 Key Metrics")
        header_frame.pack(fill=tk.X, pady=(0, 8))
        
        header_inner = ttk.Frame(header_frame)
        header_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Configure equal columns
        for i in range(5):
            header_inner.columnconfigure(i, weight=1)
        
        self.fundamental_cards = {}
        key_metrics = [
            ('market_cap', '💰 Market Cap', '₦0'),
            ('pe_ratio', '📈 P/E Ratio', '--'),
            ('pb_ratio', '📊 P/B Ratio', '--'),
            ('eps', '💵 EPS', '₦0'),
            ('dividend', '🎁 Dividend', '--%')
        ]
        
        for i, (key, label, default) in enumerate(key_metrics):
            card = ttk.Frame(header_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=4, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(6, 0))
            
            value_label = ttk.Label(card, text=default, font=get_font('heading'))
            value_label.pack(anchor='center', pady=(3, 6))
            
            self.fundamental_cards[key] = value_label
        
        # ========== ROW 2: VALUATION & FINANCIAL HEALTH ==========
        mid_frame = ttk.Frame(main_frame)
        mid_frame.pack(fill=tk.X, pady=(0, 8))
        mid_frame.columnconfigure(0, weight=1)
        mid_frame.columnconfigure(1, weight=1)
        
        # Left: Valuation Ratios
        valuation_frame = ttk.LabelFrame(mid_frame, text="📐 Valuation Ratios")
        valuation_frame.grid(row=0, column=0, padx=(0, 5), sticky='nsew')
        
        val_inner = ttk.Frame(valuation_frame)
        val_inner.pack(fill=tk.X, padx=10, pady=8)
        
        self.valuation_labels = {}
        val_items = [
            ('pe_ratio', 'P/E Ratio (TTM)'),
            ('pb_ratio', 'Price to Book'),
            ('ps_ratio', 'Price to Sales'),
            ('book_value', 'Book Value/Share'),
            ('shares_out', 'Shares Outstanding')
        ]
        
        for key, label in val_items:
            row = ttk.Frame(val_inner)
            row.pack(fill=tk.X, pady=2)
            
            ttk.Label(row, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            value = ttk.Label(row, text="--", font=get_font('body'))
            value.pack(side=tk.RIGHT)
            self.valuation_labels[key] = value
        
        # Right: Financial Health
        health_frame = ttk.LabelFrame(mid_frame, text="🏥 Financial Health")
        health_frame.grid(row=0, column=1, padx=(5, 0), sticky='nsew')
        
        health_inner = ttk.Frame(health_frame)
        health_inner.pack(fill=tk.X, padx=10, pady=8)
        
        self.health_labels = {}
        health_items = [
            ('roe', 'Return on Equity'),
            ('debt_equity', 'Debt to Equity'),
            ('current_ratio', 'Current Ratio'),
            ('revenue', 'Revenue'),
            ('net_income', 'Net Income')
        ]
        
        for key, label in health_items:
            row = ttk.Frame(health_inner)
            row.pack(fill=tk.X, pady=2)
            
            ttk.Label(row, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            
            value = ttk.Label(row, text="--", font=get_font('body'))
            value.pack(side=tk.RIGHT)
            self.health_labels[key] = value
        
        # ========== ROW 3: PRICE PERFORMANCE ==========
        perf_frame = ttk.LabelFrame(main_frame, text="📈 Price Performance")
        perf_frame.pack(fill=tk.X, pady=(0, 8))
        
        perf_inner = ttk.Frame(perf_frame)
        perf_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Configure equal columns
        for i in range(6):
            perf_inner.columnconfigure(i, weight=1)
        
        self.perf_labels = {}
        perf_items = [
            ('week', '1 Week'),
            ('month', '1 Month'),
            ('quarter', '3 Months'),
            ('half_year', '6 Months'),
            ('ytd', 'YTD'),
            ('year', '1 Year')
        ]
        
        for i, (key, label) in enumerate(perf_items):
            card = ttk.Frame(perf_inner)
            card.grid(row=0, column=i, padx=5, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            value = ttk.Label(card, text="--%", font=get_font('subheading'))
            value.pack(anchor='center')
            self.perf_labels[key] = value
        
        # ========== ROW 4: VALUATION GAUGE ==========
        gauge_frame = ttk.LabelFrame(main_frame, text="🎯 Valuation Assessment")
        gauge_frame.pack(fill=tk.X, pady=(0, 5))
        
        gauge_inner = ttk.Frame(gauge_frame)
        gauge_inner.pack(fill=tk.X, padx=15, pady=10)
        
        # Valuation status
        self.valuation_status = ttk.Label(
            gauge_inner,
            text="⏳ Loading fundamental data...",
            font=get_font('body'),
            foreground=COLORS['text_muted']
        )
        self.valuation_status.pack(anchor='center')
        
        # Valuation details
        self.valuation_details = ttk.Label(
            gauge_inner,
            text="",
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        )
        self.valuation_details.pack(anchor='center', pady=(5, 0))
    
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
            # Alert stats now updated inside _update_alerts
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
            self._update_sessions_tab()  # Super enhanced sessions tab
            self._update_signals_tab()  # Trade signal generator
            self._update_synthesis_tab()  # AI Synthesis dashboard
            # Note: Fundamentals now handled by standalone FundamentalsTab
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
        """Update the super enhanced alerts panel."""
        if not self.flow_analysis:
            return
        
        try:
            from datetime import datetime
            
            # Clear existing alerts
            for widget in self.alerts_container.winfo_children():
                widget.destroy()
            
            # Generate alerts
            alerts = self.flow_analysis.generate_alerts()
            
            # ========== UPDATE SUMMARY BAR ==========
            severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
            for alert in alerts:
                sev = alert.get('severity', 'LOW').lower()
                if sev == 'high':
                    # Check if it's actually critical (extreme signals)
                    if 'EXTREME' in alert.get('signal', '') or 'ZSCORE' in alert.get('type', ''):
                        severity_counts['critical'] += 1
                    else:
                        severity_counts['high'] += 1
                elif sev == 'medium':
                    severity_counts['medium'] += 1
                else:
                    severity_counts['low'] += 1
            
            total = sum(severity_counts.values())
            
            self.alert_counts['critical'].config(text=str(severity_counts['critical']))
            self.alert_counts['high'].config(text=str(severity_counts['high']))
            self.alert_counts['medium'].config(text=str(severity_counts['medium']))
            self.alert_counts['low'].config(text=str(severity_counts['low']))
            self.alert_counts['total'].config(text=str(total))
            
            # ========== DISPLAY ACTIVE ALERTS ==========
            if not alerts:
                no_alert_frame = ttk.Frame(self.alerts_container)
                no_alert_frame.pack(fill=tk.X, pady=20)
                
                ttk.Label(
                    no_alert_frame,
                    text="✓ No active alerts - Market conditions normal",
                    font=get_font('body'),
                    foreground=COLORS['gain']
                ).pack(anchor='center')
                return
            
            # Sort by severity (critical > high > medium > low)
            severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            sorted_alerts = sorted(alerts, key=lambda x: severity_order.get(x.get('severity', 'LOW'), 2))
            
            for alert in sorted_alerts:
                self._create_alert_card(alert)
            
            # ========== UPDATE ALERT HISTORY ==========
            now = datetime.now()
            for alert in alerts:
                history_entry = {
                    'time': now.strftime('%H:%M'),
                    'type': alert.get('type', 'UNKNOWN'),
                    'message': alert.get('message', '')[:50],
                    'severity': alert.get('severity', 'LOW'),
                    'signal': alert.get('signal', 'NEUTRAL')
                }
                
                # Add to history if not already there (avoid duplicates)
                if not any(h['type'] == history_entry['type'] and 
                          h['message'] == history_entry['message'] for h in self.alert_history[-10:]):
                    self.alert_history.append(history_entry)
            
            # Update history table
            for item in self.alert_history_tree.get_children():
                self.alert_history_tree.delete(item)
            
            for entry in reversed(self.alert_history[-10:]):  # Last 10 entries
                sev = entry['severity'].lower()
                tag = 'critical' if 'EXTREME' in str(entry.get('signal', '')) else sev
                
                sev_display = {'HIGH': '🟠 HIGH', 'MEDIUM': '🟡 MED', 'LOW': '🟢 LOW'}
                
                self.alert_history_tree.insert('', 'end', values=(
                    entry['time'],
                    entry['type'],
                    entry['message'],
                    sev_display.get(entry['severity'], entry['severity']),
                    entry['signal']
                ), tags=(tag,))
            
            # ========== UPDATE ALERT STATISTICS ==========
            self._update_alert_statistics(alerts)
            
        except Exception as e:
            logger.error(f"Error updating alerts: {e}")
    
    def _create_alert_card(self, alert):
        """Create a styled alert card."""
        # Determine severity colors
        severity = alert.get('severity', 'LOW')
        if severity == 'HIGH':
            if 'EXTREME' in alert.get('signal', '') or 'ZSCORE' in alert.get('type', ''):
                icon = '🔴'
                border_color = COLORS['loss']
                severity_text = 'CRITICAL'
            else:
                icon = '🟠'
                border_color = '#FF8C00'
                severity_text = 'HIGH'
        elif severity == 'MEDIUM':
            icon = '🟡'
            border_color = COLORS['warning']
            severity_text = 'MEDIUM'
        else:
            icon = '🟢'
            border_color = COLORS['gain']
            severity_text = 'LOW'
        
        # Alert card container
        card = ttk.Frame(self.alerts_container, relief='groove', borderwidth=1)
        card.pack(fill=tk.X, pady=4, padx=2)
        
        # Header row
        header = ttk.Frame(card)
        header.pack(fill=tk.X, padx=8, pady=(6, 3))
        
        ttk.Label(
            header,
            text=f"{icon} {severity_text} • {alert.get('type', 'UNKNOWN')}",
            font=get_font('body'),
            foreground=border_color
        ).pack(side=tk.LEFT)
        
        # Timestamp
        from datetime import datetime
        ttk.Label(
            header,
            text=datetime.now().strftime('%H:%M:%S'),
            font=get_font('tiny'),
            foreground=COLORS['text_muted']
        ).pack(side=tk.RIGHT)
        
        # Message
        ttk.Label(
            card,
            text=alert.get('message', ''),
            font=get_font('body'),
            foreground=COLORS['text_primary'],
            wraplength=550
        ).pack(anchor='w', padx=8, pady=2)
        
        # Context
        if alert.get('context'):
            ttk.Label(
                card,
                text=f"📌 {alert['context']}",
                font=get_font('small'),
                foreground=COLORS['text_muted'],
                wraplength=550
            ).pack(anchor='w', padx=8, pady=1)
        
        # Action
        if alert.get('action'):
            ttk.Label(
                card,
                text=f"💡 {alert['action']}",
                font=get_font('small'),
                foreground=COLORS['primary'],
                wraplength=550
            ).pack(anchor='w', padx=8, pady=1)
        
        # Price info
        if alert.get('price_info'):
            ttk.Label(
                card,
                text=f"📊 {alert['price_info']}",
                font=get_font('small'),
                foreground=COLORS['gain']
            ).pack(anchor='w', padx=8, pady=(1, 6))
    
    def _update_alert_statistics(self, alerts):
        """Update alert statistics section."""
        try:
            # Frequency analysis
            type_counts = {}
            signal_counts = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
            
            for alert in alerts:
                t = alert.get('type', 'UNKNOWN')
                type_counts[t] = type_counts.get(t, 0) + 1
                
                signal = alert.get('signal', 'NEUTRAL')
                if 'BUY' in signal or 'BULLISH' in signal or signal == 'ACCUMULATION':
                    signal_counts['BULLISH'] += 1
                elif 'SELL' in signal or 'BEARISH' in signal or signal == 'DISTRIBUTION':
                    signal_counts['BEARISH'] += 1
                else:
                    signal_counts['NEUTRAL'] += 1
            
            # Top 3 most frequent
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (key, label) in enumerate([('freq_1', '#1'), ('freq_2', '#2'), ('freq_3', '#3')]):
                if i < len(sorted_types):
                    self.alert_stats_labels[key].config(text=sorted_types[i][0])
                else:
                    self.alert_stats_labels[key].config(text="--")
            
            # Signal mix
            total_signals = sum(signal_counts.values()) or 1
            self.alert_stats_labels['bullish_pct'].config(
                text=f"{signal_counts['BULLISH']/total_signals*100:.0f}%",
                foreground=COLORS['gain']
            )
            self.alert_stats_labels['bearish_pct'].config(
                text=f"{signal_counts['BEARISH']/total_signals*100:.0f}%",
                foreground=COLORS['loss']
            )
            self.alert_stats_labels['neutral_pct'].config(
                text=f"{signal_counts['NEUTRAL']/total_signals*100:.0f}%",
                foreground=COLORS['warning']
            )
            
            # RVOL stats
            if self.flow_analysis:
                rvol_info = self.flow_analysis.rvol_percentile()
                if rvol_info:
                    self.alert_stats_labels['current_rvol'].config(
                        text=f"{rvol_info.get('rvol', 1):.1f}x"
                    )
                    self.alert_stats_labels['avg_rvol'].config(
                        text=f"{rvol_info.get('avg_volume', 0):,.0f}"
                    )
                    self.alert_stats_labels['peak_rvol'].config(
                        text=f"{rvol_info.get('percentile', 50):.0f}%ile"
                    )
            
            # Metrics
            from datetime import datetime
            self.alert_stats_labels['total_today'].config(text=str(len(self.alert_history)))
            self.alert_stats_labels['active_now'].config(text=str(len(alerts)))
            self.alert_stats_labels['last_update'].config(text=datetime.now().strftime('%H:%M'))
            
        except Exception as e:
            logger.error(f"Error updating alert statistics: {e}")
    
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
    
    def _update_sessions_tab(self):
        """Update the super enhanced Sessions tab with live data."""
        if not self.flow_analysis:
            return
        
        try:
            # ========== UPDATE SESSION SUMMARY CARDS ==========
            sessions = self.flow_analysis.intraday_session_breakdown()
            
            for key in ['open', 'core', 'close']:
                if key in sessions and key in self.session_cards:
                    session = sessions[key]
                    delta = session.get('delta', 0)
                    volume = session.get('volume', 0)
                    trend = session.get('trend', 'NEUTRAL')
                    
                    # Delta label
                    color = COLORS['gain'] if delta >= 0 else COLORS['loss']
                    self.session_cards[key]['delta'].config(
                        text=f"Δ: {delta:+,.0f}",
                        foreground=color
                    )
                    
                    # Volume label
                    vol_str = f"{volume/1000:.1f}K" if volume >= 1000 else f"{volume:.0f}"
                    self.session_cards[key]['extra'].config(text=f"Vol: {vol_str}")
            
            # Current position card
            from datetime import datetime
            now = datetime.now()
            hour = now.hour + now.minute / 60
            
            if hour < 10:
                position = "Pre-Market"
                progress = 0
            elif hour < 10.5:
                position = "In Opening"
                progress = (hour - 10) / 0.5 * 100
            elif hour < 13:
                position = "In Core"
                progress = (hour - 10.5) / 2.5 * 100
            elif hour < 14.5:
                position = "In Closing"
                progress = (hour - 13) / 1.5 * 100
            else:
                position = "After Hours"
                progress = 100
            
            self.session_cards['current']['delta'].config(text=position)
            self.session_cards['current']['extra'].config(text=f"{progress:.0f}% done")
            
            # Status/Bias card
            total_delta = sum(s.get('delta', 0) for s in sessions.values())
            if total_delta > 0:
                bias = "🟢 BULLISH"
                color = COLORS['gain']
            elif total_delta < 0:
                bias = "🔴 BEARISH"
                color = COLORS['loss']
            else:
                bias = "⚪ NEUTRAL"
                color = COLORS['warning']
            
            self.session_cards['status']['delta'].config(text=bias, foreground=color)
            self.session_cards['status']['extra'].config(text=f"Tot: {total_delta:+,.0f}")
            
            # ========== UPDATE OPENING RANGE ==========
            or_data = self.flow_analysis.opening_range_analysis()
            if or_data:
                or_high = or_data.get('or_high', 0)
                or_low = or_data.get('or_low', 0)
                or_range = or_data.get('or_range', 0)
                breakout = or_data.get('breakout', 'NO_BREAKOUT')
                current_price = or_data.get('current_price', 0)
                
                self.or_labels['or_high'].config(text=f"₦{or_high:,.2f}")
                self.or_labels['or_low'].config(text=f"₦{or_low:,.2f}")
                
                range_pct = (or_range / or_low * 100) if or_low else 0
                self.or_labels['or_range'].config(text=f"₦{or_range:,.2f} ({range_pct:.1f}%)")
                
                # Status with color
                if breakout == 'BULLISH_BREAKOUT':
                    status_text = "🟢 ABOVE OR HIGH"
                    color = COLORS['gain']
                elif breakout == 'BEARISH_BREAKDOWN':
                    status_text = "🔴 BELOW OR LOW"
                    color = COLORS['loss']
                else:
                    status_text = "⚪ INSIDE OR"
                    color = COLORS['warning']
                
                self.or_labels['or_status'].config(text=status_text, foreground=color)
                
                # Extension
                if or_range > 0:
                    if current_price > or_high:
                        extension = current_price - or_high
                        ext_r = extension / or_range
                        self.or_labels['or_extension'].config(
                            text=f"+₦{extension:,.2f} (+{ext_r:.1f}R)",
                            foreground=COLORS['gain']
                        )
                    elif current_price < or_low:
                        extension = or_low - current_price
                        ext_r = extension / or_range
                        self.or_labels['or_extension'].config(
                            text=f"-₦{extension:,.2f} (-{ext_r:.1f}R)",
                            foreground=COLORS['loss']
                        )
                    else:
                        self.or_labels['or_extension'].config(
                            text="Inside Range",
                            foreground=COLORS['text_muted']
                        )
            
            # ========== UPDATE SESSION DELTA CHART ==========
            self._draw_session_delta_chart(sessions)
            
            # ========== UPDATE HISTORICAL PATTERNS TABLE ==========
            history = self.flow_analysis.session_history(num_days=10)
            
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            for record in history:
                date_str = record['date'].strftime('%b %d')
                day = record['day_name']
                open_d = f"{record['open_delta']:+,.0f}"
                core_d = f"{record['core_delta']:+,.0f}"
                close_d = f"{record['close_delta']:+,.0f}"
                total = f"{record['total_delta']:+,.0f}"
                result = "🟢" if record['result'] == 'WIN' else "🔴"
                pattern = record['pattern']
                
                tag = 'win' if record['result'] == 'WIN' else 'loss'
                
                self.history_tree.insert('', 'end', values=(
                    date_str, day, open_d, core_d, close_d, total, result, pattern
                ), tags=(tag,))
            
            # ========== UPDATE SESSION ANALYTICS ==========
            if history:
                # Win rates by session
                open_wins = sum(1 for r in history if r['open_delta'] > 0)
                core_wins = sum(1 for r in history if r['core_delta'] > 0)
                close_wins = sum(1 for r in history if r['close_delta'] > 0)
                n = len(history)
                
                self.analytics_labels['open_wr'].config(text=f"{open_wins/n*100:.0f}%")
                self.analytics_labels['core_wr'].config(text=f"{core_wins/n*100:.0f}%")
                self.analytics_labels['close_wr'].config(text=f"{close_wins/n*100:.0f}%")
                
                # Average delta
                avg_open = sum(r['open_delta'] for r in history) / n
                avg_core = sum(r['core_delta'] for r in history) / n
                avg_close = sum(r['close_delta'] for r in history) / n
                
                self.analytics_labels['open_avg'].config(text=f"{avg_open:+,.0f}")
                self.analytics_labels['core_avg'].config(text=f"{avg_core:+,.0f}")
                self.analytics_labels['close_avg'].config(text=f"{avg_close:+,.0f}")
                
                # Best time
                win_rates = {'Opening': open_wins/n, 'Core': core_wins/n, 'Closing': close_wins/n}
                best = max(win_rates, key=win_rates.get)
                
                self.analytics_labels['best_session'].config(text=best)
                self.analytics_labels['best_pct'].config(text=f"{win_rates[best]*100:.0f}%")
                self.analytics_labels['total_days'].config(text=str(n))
                
                # Pattern counts
                patterns = [r['pattern'] for r in history]
                rally_pct = patterns.count('Morning Rally') / n * 100
                reversal_pct = patterns.count('Reversal') / n * 100
                dist_pct = patterns.count('Distribution') / n * 100
                
                self.analytics_labels['rally_pct'].config(text=f"{rally_pct:.0f}%")
                self.analytics_labels['reversal_pct'].config(text=f"{reversal_pct:.0f}%")
                self.analytics_labels['dist_pct'].config(text=f"{dist_pct:.0f}%")
                
        except Exception as e:
            logger.error(f"Error updating sessions tab: {e}")
    
    def _update_signals_tab(self):
        """Update the Trade Signals tab with generated signals."""
        if not self.flow_analysis:
            return
        
        try:
            from datetime import datetime
            
            # Generate signals
            signals = self.flow_analysis.generate_trade_signals()
            
            # ========== UPDATE CURRENT SIGNAL STATUS ==========
            for widget in self.signal_status_frame.winfo_children():
                widget.destroy()
            
            if not signals:
                ttk.Label(
                    self.signal_status_frame,
                    text="⏳ No active signals - Waiting for trade setup",
                    font=get_font('body'),
                    foreground=COLORS['text_muted']
                ).pack(anchor='center', pady=10)
            else:
                # Show top signal
                top_signal = signals[0]
                signal_type = top_signal.get('signal_type', 'NONE')
                
                if signal_type == 'BUY':
                    icon = "🟢"
                    color = COLORS['gain']
                    label_text = "BUY SIGNAL"
                else:
                    icon = "🔴"
                    color = COLORS['loss']
                    label_text = "SELL SIGNAL"
                
                # Signal header row
                header_row = ttk.Frame(self.signal_status_frame)
                header_row.pack(fill=tk.X)
                
                ttk.Label(
                    header_row,
                    text=f"{icon} {label_text} - {top_signal.get('pattern', '')}",
                    font=get_font('heading'),
                    foreground=color
                ).pack(side=tk.LEFT)
                
                conf = top_signal.get('confidence', 0)
                conf_color = COLORS['gain'] if conf >= 70 else COLORS['warning'] if conf >= 50 else COLORS['loss']
                ttk.Label(
                    header_row,
                    text=f"Confidence: {conf:.0f}%",
                    font=get_font('body'),
                    foreground=conf_color
                ).pack(side=tk.RIGHT)
                
                # Price levels row
                levels_row = ttk.Frame(self.signal_status_frame)
                levels_row.pack(fill=tk.X, pady=(5, 0))
                
                entry = top_signal.get('entry', 0)
                target = top_signal.get('target', 0)
                stop = top_signal.get('stop', 0)
                rr = top_signal.get('risk_reward', 0)
                
                ttk.Label(
                    levels_row,
                    text=f"Entry: ₦{entry:,.2f}  |  Target: ₦{target:,.2f}  |  Stop: ₦{stop:,.2f}  |  R:R: {rr:.1f}:1",
                    font=get_font('body'),
                    foreground=COLORS['text_secondary']
                ).pack(side=tk.LEFT)
                
                # Context row
                if top_signal.get('context'):
                    ctx_row = ttk.Frame(self.signal_status_frame)
                    ctx_row.pack(fill=tk.X, pady=(3, 0))
                    
                    ttk.Label(
                        ctx_row,
                        text=f"💡 {top_signal['context']}",
                        font=get_font('small'),
                        foreground=COLORS['primary']
                    ).pack(side=tk.LEFT)
            
            # ========== UPDATE SIGNAL COMPONENTS ==========
            if signals:
                top = signals[0]
                components = top.get('components', {})
                
                self.signal_components['divergence'].config(
                    text=components.get('divergence', 'N/A'),
                    foreground=COLORS['gain'] if 'ACCUM' in str(components.get('divergence', '')) else COLORS['loss'] if 'DIST' in str(components.get('divergence', '')) else COLORS['text_secondary']
                )
                self.signal_components['volume'].config(
                    text=components.get('volume', components.get('profile', 'N/A'))
                )
                self.signal_components['session'].config(
                    text=components.get('session', 'N/A')
                )
                
                # Confluence score
                conf = top.get('confidence', 0)
                if conf >= 70:
                    conf_text = f"{conf:.0f}% HIGH"
                    conf_color = COLORS['gain']
                elif conf >= 50:
                    conf_text = f"{conf:.0f}% MED"
                    conf_color = COLORS['warning']
                else:
                    conf_text = f"{conf:.0f}% LOW"
                    conf_color = COLORS['loss']
                
                self.signal_components['confluence'].config(text=conf_text, foreground=conf_color)
            else:
                for key in ['divergence', 'volume', 'session', 'confluence']:
                    self.signal_components[key].config(text="--", foreground=COLORS['text_muted'])
            
            # ========== UPDATE SIGNAL HISTORY ==========
            now = datetime.now()
            
            for signal in signals:
                entry = {
                    'time': now.strftime('%H:%M'),
                    'ticker': self.current_symbol or 'N/A',
                    'signal_type': signal.get('signal_type', 'NONE'),
                    'pattern': signal.get('pattern', ''),
                    'entry': signal.get('entry', 0),
                    'target': signal.get('target', 0),
                    'stop': signal.get('stop', 0),
                    'risk_reward': signal.get('risk_reward', 0),
                    'confidence': signal.get('confidence', 0),
                    'status': '⏳ Active'
                }
                
                # Check for duplicates
                if not any(h['pattern'] == entry['pattern'] and 
                          h.get('ticker') == entry['ticker'] and
                          abs(h['entry'] - entry['entry']) < 0.01 
                          for h in self.signal_history[-10:]):
                    self.signal_history.append(entry)
            
            # Update tree
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
            for entry in reversed(self.signal_history[-15:]):
                signal_icon = "🟢" if entry['signal_type'] == 'BUY' else "🔴"
                tag = 'buy' if entry['signal_type'] == 'BUY' else 'sell'
                
                self.signals_tree.insert('', 'end', values=(
                    entry['time'],
                    entry.get('ticker', 'N/A'),
                    signal_icon,
                    entry['pattern'],
                    f"₦{entry['entry']:,.2f}",
                    f"₦{entry['target']:,.2f}",
                    f"₦{entry['stop']:,.2f}",
                    f"{entry['risk_reward']:.1f}:1",
                    f"{entry['confidence']:.0f}%",
                    entry['status']
                ), tags=(tag,))
            
            # ========== UPDATE SIGNAL STATISTICS ==========
            total = len(self.signal_history)
            buy_count = sum(1 for s in self.signal_history if s['signal_type'] == 'BUY')
            sell_count = total - buy_count
            
            self.signal_stats['signals_today'].config(text=str(total))
            self.signal_stats['buy_sell'].config(
                text=f"{buy_count} / {sell_count}",
                foreground=COLORS['gain'] if buy_count > sell_count else COLORS['loss'] if sell_count > buy_count else COLORS['warning']
            )
            
            if self.signal_history:
                avg_conf = sum(s['confidence'] for s in self.signal_history) / len(self.signal_history)
                self.signal_stats['avg_conf'].config(text=f"{avg_conf:.0f}%")
                
                # Best pattern
                pattern_counts = {}
                for s in self.signal_history:
                    p = s['pattern']
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
                
                if pattern_counts:
                    best = max(pattern_counts, key=pattern_counts.get)
                    self.signal_stats['best_pattern'].config(text=best)
            
        except Exception as e:
            logger.error(f"Error updating signals tab: {e}")
    
    def _update_synthesis_tab(self):
        """Update the AI Synthesis tab with comprehensive analysis from all data sources."""
        if not self.flow_analysis:
            return
        
        try:
            from datetime import datetime
            
            # ========== GENERATE SYNTHESIS ==========
            synthesis = self._generate_ai_synthesis()
            
            # ========== UPDATE SYNTHESIS OVERVIEW ==========
            # Score
            score = synthesis.get('synthesis_score', 0)
            score_color = COLORS['gain'] if score >= 70 else COLORS['warning'] if score >= 40 else COLORS['loss']
            self.synthesis_overview['score']['value'].config(
                text=f"{score:.0f}/100",
                foreground=score_color
            )
            self.synthesis_overview['score']['drivers'].config(
                text=f"Drivers: {synthesis.get('score_drivers', '--')}"
            )
            
            # Bias
            bias = synthesis.get('bias', 'NEUTRAL')
            if bias == 'BULLISH':
                bias_icon = "🟢"
                bias_color = COLORS['gain']
            elif bias == 'BEARISH':
                bias_icon = "🔴"
                bias_color = COLORS['loss']
            else:
                bias_icon = "⚪"
                bias_color = COLORS['warning']
            
            self.synthesis_overview['bias']['value'].config(
                text=f"{bias_icon} {bias}",
                foreground=bias_color
            )
            self.synthesis_overview['bias']['drivers'].config(
                text=f"Drivers: {synthesis.get('bias_drivers', '--')}"
            )
            
            # Confidence
            confidence = synthesis.get('confidence', 0)
            conf_level = "HIGH" if confidence >= 70 else "MEDIUM" if confidence >= 50 else "LOW"
            conf_color = COLORS['gain'] if confidence >= 70 else COLORS['warning'] if confidence >= 50 else COLORS['loss']
            self.synthesis_overview['confidence']['value'].config(
                text=f"{conf_level} ({confidence:.0f}%)",
                foreground=conf_color
            )
            self.synthesis_overview['confidence']['drivers'].config(
                text=f"Drivers: {synthesis.get('confidence_drivers', '--')}"
            )
            
            # Action
            action = synthesis.get('action', 'HOLD')
            if action == 'BUY':
                action_icon = "🟢"
                action_color = COLORS['gain']
            elif action == 'SELL':
                action_icon = "🔴"
                action_color = COLORS['loss']
            else:
                action_icon = "⚪"
                action_color = COLORS['warning']
            
            self.synthesis_overview['action']['value'].config(
                text=f"{action_icon} {action}",
                foreground=action_color
            )
            self.synthesis_overview['action']['drivers'].config(
                text=f"Drivers: {synthesis.get('action_drivers', '--')}"
            )
            
            # ========== UPDATE SYMBOL HEADER ==========
            if hasattr(self, 'synth_symbol_label'):
                self.synth_symbol_label.config(text=f"{self.current_symbol or '--'} | {synthesis.get('bias', '--')}")
            
            # ========== UPDATE PRICE & VWAP DASHBOARD ==========
            if hasattr(self, 'synth_vwap'):
                try:
                    vwap_data = self.flow_analysis.vwap_analysis()
                    if vwap_data:
                        price = vwap_data.get('current_price', 0)
                        vwap = vwap_data.get('vwap', 0)
                        
                        self.synth_vwap['price'].config(text=f"₦{price:,.2f}")
                        change = ((price - vwap) / vwap * 100) if vwap > 0 else 0
                        chg_color = COLORS['gain'] if change >= 0 else COLORS['loss']
                        self.synth_vwap['change'].config(text=f"{change:+.2f}% vs VWAP", foreground=chg_color)
                        
                        self.synth_vwap['vwap'].config(text=f"₦{vwap:,.2f}")
                        self.synth_vwap['vwap_diff'].config(text=f"{change:+.2f}%", foreground=chg_color)
                        
                        self.synth_vwap['upper1'].config(text=f"₦{vwap_data.get('upper_band_1', 0):,.2f}")
                        self.synth_vwap['upper2'].config(text=f"+2σ: ₦{vwap_data.get('upper_band_2', 0):,.2f}")
                        
                        self.synth_vwap['lower1'].config(text=f"₦{vwap_data.get('lower_band_1', 0):,.2f}")
                        self.synth_vwap['lower2'].config(text=f"-2σ: ₦{vwap_data.get('lower_band_2', 0):,.2f}")
                        
                        # Draw VWAP position bar
                        bar = self.synth_vwap['position_bar']
                        bar.delete('all')
                        w = bar.winfo_width() or 400
                        bar.create_rectangle(0, 0, w, 20, fill=COLORS['bg_medium'], outline='')
                        
                        upper2 = vwap_data.get('upper_band_2', vwap * 1.04)
                        lower2 = vwap_data.get('lower_band_2', vwap * 0.96)
                        spread = upper2 - lower2 if upper2 != lower2 else 1
                        price_pos = (price - lower2) / spread
                        price_x = max(10, min(w - 10, int(price_pos * w)))
                        
                        bar.create_text(w/2, 10, text=f"VWAP: ₦{vwap:,.2f}", fill=COLORS['text_muted'], font=('', 8))
                        bar.create_oval(price_x - 5, 5, price_x + 5, 15, fill=COLORS['primary'], outline='')
                except Exception as e:
                    logger.debug(f"VWAP update failed: {e}")
            
            # ========== UPDATE DELTA ANALYSIS ==========
            if hasattr(self, 'synth_delta'):
                try:
                    cum_delta = self.flow_analysis.cumulative_delta()
                    if cum_delta and len(cum_delta) > 0:
                        last_delta = cum_delta[-1].get('cumulative_delta', 0)
                        delta_color = COLORS['gain'] if last_delta >= 0 else COLORS['loss']
                        self.synth_delta['cum_delta'].config(text=f"{last_delta:+,.0f}", foreground=delta_color)
                        
                        if len(cum_delta) >= 5:
                            delta_5 = last_delta - cum_delta[-5].get('cumulative_delta', 0)
                            trend_color = COLORS['gain'] if delta_5 >= 0 else COLORS['loss']
                            self.synth_delta['trend'].config(text=f"{delta_5:+,.0f}", foreground=trend_color)
                except:
                    pass
                
                try:
                    momentum = self.flow_analysis.delta_momentum()
                    if momentum and len(momentum) > 0:
                        mom = momentum[-1].get('delta_momentum', 0)
                        mom_color = COLORS['gain'] if mom >= 0 else COLORS['loss']
                        self.synth_delta['momentum'].config(text=f"{mom:+,.0f}", foreground=mom_color)
                except:
                    pass
                
                try:
                    zscore = self.flow_analysis.delta_zscore()
                    if zscore and len(zscore) > 0:
                        z = zscore[-1].get('zscore', 0)
                        z_color = COLORS['gain'] if z >= 0 else COLORS['loss']
                        self.synth_delta['zscore'].config(text=f"{z:+.2f}σ", foreground=z_color)
                except:
                    pass
                
                try:
                    divergence = self.flow_analysis.delta_divergence()
                    if divergence and len(divergence) > 0:
                        div_type = divergence[-1].get('type', 'NONE')
                        div_color = COLORS['warning'] if div_type != 'NONE' else COLORS['text_muted']
                        self.synth_delta['divergence'].config(text=div_type, foreground=div_color)
                except:
                    pass
            
            # ========== UPDATE VOLUME PROFILE ==========
            if hasattr(self, 'synth_profile'):
                try:
                    profile = self.flow_analysis.volume_profile()
                    if profile:
                        self.synth_profile['poc'].config(text=f"₦{profile.get('poc_price', 0):,.2f}")
                        self.synth_profile['poc_vol'].config(text=f"Vol: {profile.get('poc_volume', 0):,.0f}")
                        self.synth_profile['vah'].config(text=f"₦{profile.get('vah', 0):,.2f}")
                        self.synth_profile['val'].config(text=f"₦{profile.get('val', 0):,.2f}")
                        
                        vwap_data = self.flow_analysis.vwap_analysis()
                        price = vwap_data.get('current_price', 0) if vwap_data else 0
                        if price > profile.get('vah', 0):
                            pos_text = "ABOVE VALUE"
                            pos_color = COLORS['loss']
                        elif price < profile.get('val', 0):
                            pos_text = "BELOW VALUE"
                            pos_color = COLORS['gain']
                        else:
                            pos_text = "IN VALUE AREA"
                            pos_color = COLORS['warning']
                        self.synth_profile['position'].config(text=pos_text, foreground=pos_color)
                except:
                    pass
            
            # ========== UPDATE FLOW PRESSURE ==========
            if hasattr(self, 'synth_flow'):
                try:
                    blocks = self.flow_analysis.block_trade_analysis()
                    if blocks:
                        total = blocks.get('total_blocks', 0)
                        buy = blocks.get('buy_blocks', 0)
                        sell = blocks.get('sell_blocks', 0)
                        self.synth_flow['blocks'].config(text=f"{total}")
                        if buy > sell:
                            bias_text = f"BUY {buy}:{sell}"
                            bias_color = COLORS['gain']
                        elif sell > buy:
                            bias_text = f"SELL {sell}:{buy}"
                            bias_color = COLORS['loss']
                        else:
                            bias_text = "BALANCED"
                            bias_color = COLORS['warning']
                        self.synth_flow['block_bias'].config(text=bias_text, foreground=bias_color)
                except:
                    pass
                
                try:
                    rvol = self.flow_analysis.rvol_analysis()
                    if rvol:
                        r = rvol.get('rvol', 1)
                        self.synth_flow['rvol'].config(text=f"{r:.2f}x")
                        if r >= 2:
                            status = "HIGH"
                            r_color = COLORS['gain']
                        elif r >= 1.5:
                            status = "ELEVATED"
                            r_color = COLORS['warning']
                        else:
                            status = "NORMAL"
                            r_color = COLORS['text_muted']
                        self.synth_flow['rvol_status'].config(text=status, foreground=r_color)
                except:
                    pass
                
                try:
                    sessions = self.flow_analysis.intraday_session_breakdown()
                    if sessions:
                        total_delta = sum(s.get('delta', 0) for s in sessions.values())
                        delta_color = COLORS['gain'] if total_delta >= 0 else COLORS['loss']
                        self.synth_flow['session_delta'].config(text=f"{total_delta:+,.0f}", foreground=delta_color)
                        bias = "ACCUM" if total_delta > 0 else "DIST" if total_delta < 0 else "NEUTRAL"
                        self.synth_flow['session_bias'].config(text=bias, foreground=delta_color)
                except:
                    pass
                
                # Flow gauge
                try:
                    gauge = self.synth_flow['gauge']
                    gauge.delete('all')
                    
                    cum_delta = self.flow_analysis.cumulative_delta()
                    if cum_delta and len(cum_delta) > 0:
                        delta = cum_delta[-1].get('cumulative_delta', 0)
                        max_delta = max(abs(d.get('cumulative_delta', 0)) for d in cum_delta) or 1
                        pct = (delta / max_delta + 1) / 2  # 0 to 1
                        
                        gauge.create_rectangle(0, 5, 60, 20, fill=COLORS['loss'], outline='')
                        gauge.create_rectangle(60, 5, 120, 20, fill=COLORS['gain'], outline='')
                        gauge.create_line(60, 0, 60, 25, fill=COLORS['text_muted'], width=2)
                        
                        pos_x = int(pct * 120)
                        gauge.create_oval(pos_x - 4, 8, pos_x + 4, 17, fill='white', outline='')
                        
                        bias = "BUYERS" if delta > 0 else "SELLERS" if delta < 0 else "BALANCED"
                        bias_color = COLORS['gain'] if delta > 0 else COLORS['loss'] if delta < 0 else COLORS['warning']
                        self.synth_flow['gauge_label'].config(text=bias, foreground=bias_color)
                except:
                    pass
            
            # ========== UPDATE ENHANCED INSIGHTS ==========
            if hasattr(self, 'synth_insights'):
                insights_data = synthesis.get('key_insights', [])
                for i, card in enumerate(self.synth_insights):
                    if i < len(insights_data):
                        insight = insights_data[i]
                        card['icon'].config(text=insight.get('icon', '💡'))
                        card['text'].config(text=insight.get('title', '--'))
                    else:
                        card['icon'].config(text="⚪")
                        card['text'].config(text="--")
            
            # ========== UPDATE ALERTS ==========
            if hasattr(self, 'synth_alerts'):
                try:
                    alerts = self.flow_analysis.generate_alerts()[:4]
                    for i, card in enumerate(self.synth_alerts):
                        if i < len(alerts):
                            alert = alerts[i]
                            severity = alert.get('severity', 'LOW')
                            if severity == 'HIGH':
                                icon = "🔴"
                            elif severity == 'MEDIUM':
                                icon = "🟡"
                            else:
                                icon = "⚪"
                            card['icon'].config(text=icon)
                            card['text'].config(text=alert.get('message', '--')[:40], foreground=COLORS['text_primary'])
                        else:
                            card['icon'].config(text="⚪")
                            card['text'].config(text="--", foreground=COLORS['text_muted'])
                except:
                    pass
            
            # ========== UPDATE COMPONENT SCORES ==========
            component_scores = synthesis.get('component_scores', {})
            
            for key, data in component_scores.items():
                if key in self.synthesis_components:
                    score = data.get('score', 0)
                    drivers = data.get('drivers', '--')
                    
                    # Update score
                    score_color = COLORS['gain'] if score >= 7 else COLORS['warning'] if score >= 4 else COLORS['loss']
                    self.synthesis_components[key]['score'].config(
                        text=f"{score:.1f}",
                        foreground=score_color
                    )
                    
                    # Update bar
                    bar = self.synthesis_components[key]['bar']
                    bar.delete('all')
                    bar_width = min(120, int(score * 12))
                    bar.create_rectangle(0, 0, bar_width, 8, fill=score_color, outline='')
                    
                    # Update drivers
                    self.synthesis_components[key]['drivers'].config(
                        text=f"• {drivers}"
                    )
            
            # ========== UPDATE AI NARRATIVE ==========
            narrative = synthesis.get('narrative', 'No analysis available.')
            self.narrative_text.config(state='normal')
            self.narrative_text.delete('1.0', tk.END)
            self.narrative_text.insert('1.0', narrative)
            self.narrative_text.config(state='disabled')
            
            # ========== UPDATE KEY INSIGHTS (Legacy - now handled above with synth_insights) ==========
            # Only run this if synth_insights is not available (backward compatibility)
            if not hasattr(self, 'synth_insights'):
                for widget in self.insights_container.winfo_children():
                    widget.destroy()
                
                insights = synthesis.get('key_insights', [])
                if not insights:
                    ttk.Label(self.insights_container, text="No key insights at this time.",
                             font=get_font('body'), foreground=COLORS['text_muted']).pack(anchor='w')
                else:
                    insights_row = ttk.Frame(self.insights_container)
                    insights_row.pack(fill=tk.X)
                    
                    for insight in insights[:4]:  # Max 4 insights
                        card = ttk.Frame(insights_row, relief='ridge', borderwidth=1)
                        card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
                        
                        icon = insight.get('icon', '💡')
                        title = insight.get('title', 'Insight')
                        detail = insight.get('detail', '')
                        color = insight.get('color', COLORS['primary'])
                        
                        ttk.Label(card, text=f"{icon} {title}",
                                 font=get_font('body'), foreground=color).pack(anchor='w', padx=10, pady=(5, 0))
                        ttk.Label(card, text=detail,
                                 font=get_font('small'), foreground=COLORS['text_secondary']).pack(anchor='w', padx=10, pady=(0, 5))
            
            # ========== UPDATE SIGNAL SUMMARY ==========
            signal_summary = synthesis.get('signal_summary', {})
            
            signal_type = signal_summary.get('type', 'NONE')
            if signal_type == 'BUY':
                self.synthesis_signal['icon'].config(text="🟢")
                self.synthesis_signal['type'].config(text="BUY", foreground=COLORS['gain'])
            elif signal_type == 'SELL':
                self.synthesis_signal['icon'].config(text="🔴")
                self.synthesis_signal['type'].config(text="SELL", foreground=COLORS['loss'])
            else:
                self.synthesis_signal['icon'].config(text="⚪")
                self.synthesis_signal['type'].config(text="NO SIGNAL", foreground=COLORS['text_muted'])
            
            self.synthesis_signal['pattern'].config(
                text=signal_summary.get('pattern', '--')
            )
            
            entry = signal_summary.get('entry', 0)
            target = signal_summary.get('target', 0)
            stop = signal_summary.get('stop', 0)
            rr = signal_summary.get('risk_reward', 0)
            
            self.synthesis_signal['levels'].config(
                text=f"Entry: ₦{entry:,.2f} | Target: ₦{target:,.2f} | Stop: ₦{stop:,.2f} | R:R: {rr:.1f}:1"
            )
            
            # Confluence bar
            confluence = signal_summary.get('confluence', 0)
            bar = self.synthesis_signal['confluence_bar']
            bar.delete('all')
            
            conf_color = COLORS['gain'] if confluence >= 70 else COLORS['warning'] if confluence >= 50 else COLORS['loss']
            bar_width = int(150 * confluence / 100)
            bar.create_rectangle(0, 0, bar_width, 15, fill=conf_color, outline='')
            
            self.synthesis_signal['confluence_pct'].config(
                text=f"{confluence:.0f}%",
                foreground=conf_color
            )
            
            # ========== UPDATE STATUS BAR ==========
            now = datetime.now()
            self.synthesis_status['live'].config(
                text="📡 LIVE" if self.auto_refresh_id else "📡 Manual",
                foreground=COLORS['gain'] if self.auto_refresh_id else COLORS['text_muted']
            )
            self.synthesis_status['update'].config(
                text=f"Last Update: {now.strftime('%H:%M:%S')}"
            )
            
        except Exception as e:
            logger.error(f"Error updating synthesis tab: {e}")
            import traceback
            traceback.print_exc()
    
    def _refresh_synthesis(self):
        """Manual refresh button handler for AI Synthesis tab."""
        if hasattr(self, 'synthesis_status') and 'live' in self.synthesis_status:
            self.synthesis_status['live'].config(text="📡 Refreshing...", foreground=COLORS['warning'])
        self._update_synthesis_tab()
    
    def _generate_ai_synthesis(self) -> dict:
        """Generate comprehensive AI synthesis from all data sources."""
        from datetime import datetime
        
        synthesis = {
            'synthesis_score': 0,
            'bias': 'NEUTRAL',
            'confidence': 0,
            'action': 'HOLD',
            'score_drivers': '',
            'bias_drivers': '',
            'confidence_drivers': '',
            'action_drivers': '',
            'component_scores': {},
            'narrative': '',
            'key_insights': [],
            'signal_summary': {}
        }
        
        try:
            # ========== 1. TAPE SCORE ==========
            tape_score, tape_drivers = self._score_tape_data()
            synthesis['component_scores']['tape'] = {
                'score': tape_score,
                'drivers': tape_drivers
            }
            
            # ========== 2. ALERTS SCORE ==========
            alert_score, alert_drivers = self._score_alert_data()
            synthesis['component_scores']['alerts'] = {
                'score': alert_score,
                'drivers': alert_drivers
            }
            
            # ========== 3. CHARTS SCORE ==========
            chart_score, chart_drivers = self._score_chart_data()
            synthesis['component_scores']['charts'] = {
                'score': chart_score,
                'drivers': chart_drivers
            }
            
            # ========== 4. SESSIONS SCORE ==========
            session_score, session_drivers = self._score_session_data()
            synthesis['component_scores']['sessions'] = {
                'score': session_score,
                'drivers': session_drivers
            }
            
            # ========== 5. SIGNALS SCORE ==========
            signal_score, signal_drivers = self._score_signal_data()
            synthesis['component_scores']['signals'] = {
                'score': signal_score,
                'drivers': signal_drivers
            }
            
            # ========== COMPUTE WEIGHTED SYNTHESIS SCORE ==========
            weights = {
                'tape': 0.25,
                'charts': 0.20,
                'signals': 0.25,
                'sessions': 0.15,
                'alerts': 0.15
            }
            
            total_score = (
                tape_score * weights['tape'] +
                chart_score * weights['charts'] +
                signal_score * weights['signals'] +
                session_score * weights['sessions'] +
                alert_score * weights['alerts']
            ) * 10  # Scale to 0-100
            
            synthesis['synthesis_score'] = min(100, max(0, total_score))
            
            # Identify top drivers for score
            sorted_scores = sorted(
                [(k, v['score']) for k, v in synthesis['component_scores'].items()],
                key=lambda x: x[1], reverse=True
            )
            top_drivers = [f"{k.title()}: {s:.1f}" for k, s in sorted_scores[:2]]
            synthesis['score_drivers'] = ", ".join(top_drivers)
            
            # ========== DETERMINE BIAS ==========
            bullish_signals = 0
            bearish_signals = 0
            bias_drivers = []
            
            # Delta direction
            cum_delta = self.flow_analysis.cumulative_delta()
            if cum_delta:
                latest_delta = cum_delta[-1].get('cumulative_delta', 0)
                if latest_delta > 0:
                    bullish_signals += 2
                    bias_drivers.append("Δ+")
                elif latest_delta < 0:
                    bearish_signals += 2
                    bias_drivers.append("Δ-")
            
            # Session bias
            sessions = self.flow_analysis.intraday_session_breakdown()
            total_delta = sum(s.get('delta', 0) for s in sessions.values())
            if total_delta > 0:
                bullish_signals += 1
                bias_drivers.append("Session+")
            elif total_delta < 0:
                bearish_signals += 1
                bias_drivers.append("Session-")
            
            # VWAP position
            vwap = self.flow_analysis.vwap_analysis()
            if vwap:
                current_price = vwap.get('current_price', 0)
                vwap_val = vwap.get('vwap', 0)
                if current_price > vwap_val:
                    bullish_signals += 1
                    bias_drivers.append("AboveVWAP")
                elif current_price < vwap_val:
                    bearish_signals += 1
                    bias_drivers.append("BelowVWAP")
            
            if bullish_signals > bearish_signals:
                synthesis['bias'] = 'BULLISH'
            elif bearish_signals > bullish_signals:
                synthesis['bias'] = 'BEARISH'
            else:
                synthesis['bias'] = 'NEUTRAL'
            
            synthesis['bias_drivers'] = ", ".join(bias_drivers[:3])
            
            # ========== DETERMINE CONFIDENCE ==========
            conf_factors = []
            confidence = 50  # Base
            
            # Volume confirmation
            rvol = self.flow_analysis.rvol_analysis()
            if rvol and rvol.get('rvol', 1) > 1.5:
                confidence += 15
                conf_factors.append("HighVol")
            elif rvol and rvol.get('rvol', 1) < 0.7:
                confidence -= 10
                conf_factors.append("LowVol")
            
            # Signal alignment
            signals = self.flow_analysis.generate_trade_signals()
            if signals:
                sig_conf = signals[0].get('confidence', 0)
                if sig_conf >= 70:
                    confidence += 20
                    conf_factors.append("StrongSig")
                elif sig_conf >= 50:
                    confidence += 10
                    conf_factors.append("ModSig")
            
            # Component alignment
            avg_score = (tape_score + chart_score + signal_score + session_score + alert_score) / 5
            if avg_score >= 7:
                confidence += 15
                conf_factors.append("Aligned")
            elif avg_score <= 4:
                confidence -= 10
                conf_factors.append("Mixed")
            
            synthesis['confidence'] = min(100, max(0, confidence))
            synthesis['confidence_drivers'] = ", ".join(conf_factors[:3])
            
            # ========== DETERMINE ACTION ==========
            action_drivers = []
            
            if synthesis['bias'] == 'BULLISH' and synthesis['confidence'] >= 60 and synthesis['synthesis_score'] >= 60:
                synthesis['action'] = 'BUY'
                action_drivers = ["Bullish bias", f"Conf {synthesis['confidence']:.0f}%", f"Score {synthesis['synthesis_score']:.0f}"]
            elif synthesis['bias'] == 'BEARISH' and synthesis['confidence'] >= 60 and synthesis['synthesis_score'] >= 60:
                synthesis['action'] = 'SELL'
                action_drivers = ["Bearish bias", f"Conf {synthesis['confidence']:.0f}%", f"Score {synthesis['synthesis_score']:.0f}"]
            else:
                synthesis['action'] = 'HOLD'
                if synthesis['confidence'] < 60:
                    action_drivers.append("Low conf")
                if synthesis['bias'] == 'NEUTRAL':
                    action_drivers.append("Neutral bias")
                if synthesis['synthesis_score'] < 60:
                    action_drivers.append("Weak score")
            
            synthesis['action_drivers'] = ", ".join(action_drivers[:3])
            
            # ========== BUILD NARRATIVE ==========
            synthesis['narrative'] = self._build_narrative(synthesis)
            
            # ========== KEY INSIGHTS ==========
            synthesis['key_insights'] = self._generate_key_insights(synthesis)
            
            # ========== SIGNAL SUMMARY ==========
            # Validate and select best valid signal
            valid_signal = None
            if signals:
                for sig in signals:
                    sig_type = sig.get('signal_type', 'NONE')
                    entry = sig.get('entry', 0)
                    target = sig.get('target', 0)
                    stop = sig.get('stop', 0)
                    
                    # Validate signal logic
                    is_valid = False
                    if sig_type == 'BUY':
                        # BUY: target > entry > stop
                        is_valid = target > entry > stop and entry > 0
                    elif sig_type == 'SELL':
                        # SELL: stop > entry > target
                        is_valid = stop > entry > target and entry > 0
                    
                    if is_valid:
                        valid_signal = sig
                        break
                    else:
                        logger.debug(f"Invalid signal rejected: {sig_type} entry={entry}, target={target}, stop={stop}")
            
            if valid_signal:
                synthesis['signal_summary'] = {
                    'type': valid_signal.get('signal_type', 'NONE'),
                    'pattern': valid_signal.get('pattern', '--'),
                    'entry': valid_signal.get('entry', 0),
                    'target': valid_signal.get('target', 0),
                    'stop': valid_signal.get('stop', 0),
                    'risk_reward': valid_signal.get('risk_reward', 0),
                    'confluence': valid_signal.get('confidence', 0)
                }
            else:
                synthesis['signal_summary'] = {
                    'type': 'NONE',
                    'pattern': 'No valid signal',
                    'entry': 0,
                    'target': 0,
                    'stop': 0,
                    'risk_reward': 0,
                    'confluence': 0
                }
            
        except Exception as e:
            logger.error(f"Error generating AI synthesis: {e}")
            import traceback
            traceback.print_exc()
        
        return synthesis
    
    def _score_tape_data(self) -> tuple:
        """Score tape and profile data (0-10 scale)."""
        score = 5.0
        drivers = []
        
        try:
            # Cumulative delta trend
            cum_delta = self.flow_analysis.cumulative_delta()
            if cum_delta and len(cum_delta) > 1:
                recent_delta = cum_delta[-1].get('cumulative_delta', 0)
                prev_delta = cum_delta[-5].get('cumulative_delta', 0) if len(cum_delta) > 5 else 0
                delta_trend = recent_delta - prev_delta
                
                if delta_trend > 0:
                    score += 1.5
                    drivers.append("Δ rising")
                elif delta_trend < 0:
                    score -= 1.0
                    drivers.append("Δ falling")
            
            # Volume profile position
            profile = self.flow_analysis.volume_profile()
            if profile:
                poc = profile.get('poc_price', 0)
                current = profile.get('current_price', 0)
                if current > poc:
                    score += 1.0
                    drivers.append("Above POC")
                elif current < poc:
                    score -= 0.5
                    drivers.append("Below POC")
            
            # Block trade activity
            blocks = self.flow_analysis.block_trade_analysis()
            if blocks and blocks.get('total_blocks', 0) > 2:
                score += 1.5
                drivers.append(f"{blocks['total_blocks']} blocks")
            
            # RVOL
            rvol = self.flow_analysis.rvol_analysis()
            if rvol and rvol.get('rvol', 1) > 2:
                score += 1.0
                drivers.append(f"RVOL {rvol['rvol']:.1f}x")
            
        except Exception as e:
            logger.error(f"Error scoring tape data: {e}")
        
        return min(10, max(0, score)), ", ".join(drivers[:3]) if drivers else "Standard flow"
    
    def _score_alert_data(self) -> tuple:
        """Score alert data (0-10 scale)."""
        score = 5.0
        drivers = []
        
        try:
            alerts = self.flow_analysis.generate_alerts()
            
            if not alerts:
                return 5.0, "No active alerts"
            
            # Count by severity
            high_count = sum(1 for a in alerts if a.get('severity', '') == 'HIGH')
            med_count = sum(1 for a in alerts if a.get('severity', '') == 'MEDIUM')
            
            # More alerts = more action = potentially higher score
            if high_count > 0:
                score += min(2, high_count * 0.75)
                drivers.append(f"{high_count} HIGH")
            
            if med_count > 0:
                score += min(1, med_count * 0.25)
                drivers.append(f"{med_count} MED")
            
            # Check for bullish vs bearish signals
            bullish = sum(1 for a in alerts if 'bullish' in a.get('signal', '').lower() or 'buy' in a.get('signal', '').lower())
            bearish = sum(1 for a in alerts if 'bearish' in a.get('signal', '').lower() or 'sell' in a.get('signal', '').lower())
            
            if bullish > bearish:
                score += 1
                drivers.append("Net bullish")
            elif bearish > bullish:
                score += 0.5  # Still actionable
                drivers.append("Net bearish")
            
        except Exception as e:
            logger.error(f"Error scoring alert data: {e}")
        
        return min(10, max(0, score)), ", ".join(drivers[:3]) if drivers else "Normal conditions"
    
    def _score_chart_data(self) -> tuple:
        """Score chart/technical data (0-10 scale)."""
        score = 5.0
        drivers = []
        
        try:
            # VWAP analysis
            vwap = self.flow_analysis.vwap_analysis()
            if vwap:
                current = vwap.get('current_price', 0)
                vwap_val = vwap.get('vwap', 0)
                upper_1 = vwap.get('upper_band_1', 0)
                lower_1 = vwap.get('lower_band_1', 0)
                
                if current > upper_1:
                    score += 1.5
                    drivers.append("Above VWAP+1σ")
                elif current > vwap_val:
                    score += 1.0
                    drivers.append("Above VWAP")
                elif current < lower_1:
                    score -= 1.0
                    drivers.append("Below VWAP-1σ")
                elif current < vwap_val:
                    score -= 0.5
                    drivers.append("Below VWAP")
            
            # Delta momentum
            momentum = self.flow_analysis.delta_momentum()
            if momentum:
                recent_mom = momentum[-1].get('momentum', 0) if momentum else 0
                if recent_mom > 0:
                    score += 1.0
                    drivers.append("Δ momentum+")
                elif recent_mom < 0:
                    score -= 0.5
                    drivers.append("Δ momentum-")
            
            # RVOL
            rvol = self.flow_analysis.rvol_analysis()
            if rvol:
                rvol_val = rvol.get('rvol', 1)
                if rvol_val >= 2:
                    score += 1.5
                    drivers.append(f"RVOL {rvol_val:.1f}x")
                elif rvol_val >= 1.5:
                    score += 0.75
                    drivers.append(f"RVOL {rvol_val:.1f}x")
            
        except Exception as e:
            logger.error(f"Error scoring chart data: {e}")
        
        return min(10, max(0, score)), ", ".join(drivers[:3]) if drivers else "Standard technicals"
    
    def _score_session_data(self) -> tuple:
        """Score session data (0-10 scale)."""
        score = 5.0
        drivers = []
        
        try:
            sessions = self.flow_analysis.intraday_session_breakdown()
            
            # Total session delta
            total_delta = sum(s.get('delta', 0) for s in sessions.values())
            if total_delta > 0:
                score += 1.5
                drivers.append(f"Tot Δ+{total_delta:,.0f}")
            elif total_delta < 0:
                score -= 0.5
                drivers.append(f"Tot Δ{total_delta:,.0f}")
            
            # Opening range analysis
            or_data = self.flow_analysis.opening_range_analysis()
            if or_data:
                breakout = or_data.get('breakout', 'NO_BREAKOUT')
                if breakout == 'BULLISH_BREAKOUT':
                    score += 2.0
                    drivers.append("OR Breakout↑")
                elif breakout == 'BEARISH_BREAKDOWN':
                    score += 1.0  # Still actionable
                    drivers.append("OR Breakdown↓")
                else:
                    drivers.append("Inside OR")
            
            # Session trend consistency
            consistent = True
            if sessions:
                deltas = [s.get('delta', 0) for s in sessions.values() if s.get('delta', 0) != 0]
                if deltas and len(deltas) > 1:
                    all_positive = all(d > 0 for d in deltas)
                    all_negative = all(d < 0 for d in deltas)
                    if all_positive or all_negative:
                        score += 1.0
                        drivers.append("Consistent")
                        consistent = True
                    else:
                        drivers.append("Mixed")
            
        except Exception as e:
            logger.error(f"Error scoring session data: {e}")
        
        return min(10, max(0, score)), ", ".join(drivers[:3]) if drivers else "Normal session"
    
    def _score_signal_data(self) -> tuple:
        """Score trade signals data (0-10 scale)."""
        score = 5.0
        drivers = []
        
        try:
            signals = self.flow_analysis.generate_trade_signals()
            
            if not signals:
                return 4.0, "No active signals"
            
            top_signal = signals[0]
            confidence = top_signal.get('confidence', 0)
            rr = top_signal.get('risk_reward', 0)
            signal_type = top_signal.get('signal_type', 'NONE')
            pattern = top_signal.get('pattern', '')
            
            # Confidence score contribution
            if confidence >= 75:
                score += 2.5
                drivers.append(f"Conf {confidence:.0f}%")
            elif confidence >= 60:
                score += 1.5
                drivers.append(f"Conf {confidence:.0f}%")
            elif confidence >= 45:
                score += 0.5
                drivers.append(f"Conf {confidence:.0f}%")
            
            # Risk/reward contribution
            if rr >= 3:
                score += 1.5
                drivers.append(f"R:R {rr:.1f}")
            elif rr >= 2:
                score += 1.0
                drivers.append(f"R:R {rr:.1f}")
            elif rr >= 1.5:
                score += 0.5
            
            # Signal clarity
            if signal_type in ['BUY', 'SELL']:
                score += 0.5
                drivers.append(signal_type)
            
            # Pattern bonus
            if pattern:
                drivers.append(pattern[:15])
            
        except Exception as e:
            logger.error(f"Error scoring signal data: {e}")
        
        return min(10, max(0, score)), ", ".join(drivers[:3]) if drivers else "Waiting for setup"
    
    def _build_narrative(self, synthesis: dict) -> str:
        """Build AI narrative report text - uses Groq if available, falls back to rule-based."""
        symbol = self.current_symbol or "N/A"
        score = synthesis.get('synthesis_score', 0)
        bias = synthesis.get('bias', 'NEUTRAL')
        action = synthesis.get('action', 'HOLD')
        confidence = synthesis.get('confidence', 0)
        
        # Try to use Groq AI for narrative
        if self.insight_engine:
            logger.info(f"Attempting Groq AI narrative generation for {symbol}")
            try:
                # Build comprehensive context for AI
                context_data = {
                    'symbol': symbol,
                    'synthesis_score': score,
                    'bias': bias,
                    'action': action,
                    'confidence': confidence,
                    'score_drivers': synthesis.get('score_drivers', ''),
                    'bias_drivers': synthesis.get('bias_drivers', ''),
                    'action_drivers': synthesis.get('action_drivers', ''),
                    'component_scores': synthesis.get('component_scores', {}),
                }
                
                # ========== PRICE & VWAP DATA ==========
                try:
                    vwap = self.flow_analysis.vwap_analysis()
                    if vwap:
                        context_data['current_price'] = vwap.get('current_price', 0)
                        context_data['vwap'] = vwap.get('vwap', 0)
                        context_data['upper_band_1'] = vwap.get('upper_band_1', 0)
                        context_data['lower_band_1'] = vwap.get('lower_band_1', 0)
                        context_data['upper_band_2'] = vwap.get('upper_band_2', 0)
                        context_data['lower_band_2'] = vwap.get('lower_band_2', 0)
                        vwap_diff = ((context_data['current_price'] - context_data['vwap']) / context_data['vwap'] * 100) if context_data['vwap'] else 0
                        context_data['vwap_diff_pct'] = vwap_diff
                except:
                    context_data.update({'current_price': 0, 'vwap': 0, 'vwap_diff_pct': 0})
                
                # ========== CUMULATIVE DELTA ==========
                try:
                    cum_delta = self.flow_analysis.cumulative_delta()
                    if cum_delta and len(cum_delta) > 0:
                        context_data['cumulative_delta'] = cum_delta[-1].get('cumulative_delta', 0)
                        # Delta trend (last 5 bars)
                        if len(cum_delta) >= 5:
                            delta_5_bars_ago = cum_delta[-5].get('cumulative_delta', 0)
                            context_data['delta_trend_5bar'] = context_data['cumulative_delta'] - delta_5_bars_ago
                        else:
                            context_data['delta_trend_5bar'] = 0
                except:
                    context_data.update({'cumulative_delta': 0, 'delta_trend_5bar': 0})
                
                # ========== DELTA MOMENTUM & ZSCORE ==========
                try:
                    momentum = self.flow_analysis.delta_momentum()
                    if momentum:
                        context_data['delta_momentum'] = momentum[-1].get('delta_momentum', 0)
                except:
                    context_data['delta_momentum'] = 0
                
                try:
                    zscore = self.flow_analysis.delta_zscore()
                    if zscore:
                        context_data['delta_zscore'] = zscore[-1].get('zscore', 0)
                except:
                    context_data['delta_zscore'] = 0
                
                # ========== VOLUME PROFILE (POC, VAH, VAL) ==========
                try:
                    profile = self.flow_analysis.volume_profile()
                    if profile:
                        context_data['poc_price'] = profile.get('poc_price', 0)
                        context_data['vah'] = profile.get('vah', 0)
                        context_data['val'] = profile.get('val', 0)
                        context_data['poc_volume'] = profile.get('poc_volume', 0)
                        # Price position relative to value area
                        price = context_data.get('current_price', 0)
                        if price > context_data['vah']:
                            context_data['price_vs_profile'] = "ABOVE VALUE AREA (potential resistance)"
                        elif price < context_data['val']:
                            context_data['price_vs_profile'] = "BELOW VALUE AREA (potential support)"
                        else:
                            context_data['price_vs_profile'] = "INSIDE VALUE AREA (consolidation)"
                except:
                    context_data.update({'poc_price': 0, 'vah': 0, 'val': 0, 'price_vs_profile': 'N/A'})
                
                # ========== BLOCK TRADES ==========
                try:
                    blocks = self.flow_analysis.block_trade_analysis()
                    if blocks:
                        context_data['total_blocks'] = blocks.get('total_blocks', 0)
                        context_data['buy_blocks'] = blocks.get('buy_blocks', 0)
                        context_data['sell_blocks'] = blocks.get('sell_blocks', 0)
                        context_data['block_volume'] = blocks.get('total_block_volume', 0)
                        context_data['block_imbalance'] = "BUY HEAVY" if blocks.get('buy_blocks', 0) > blocks.get('sell_blocks', 0) else "SELL HEAVY" if blocks.get('sell_blocks', 0) > blocks.get('buy_blocks', 0) else "BALANCED"
                except:
                    context_data.update({'total_blocks': 0, 'buy_blocks': 0, 'sell_blocks': 0, 'block_imbalance': 'N/A'})
                
                # ========== RVOL (RELATIVE VOLUME) ==========
                try:
                    rvol = self.flow_analysis.rvol_analysis()
                    if rvol:
                        context_data['rvol'] = rvol.get('rvol', 1)
                        context_data['current_volume'] = rvol.get('current_volume', 0)
                        context_data['avg_volume'] = rvol.get('average_volume', 0)
                        if context_data['rvol'] >= 2:
                            context_data['rvol_interpretation'] = "UNUSUALLY HIGH (institutional activity likely)"
                        elif context_data['rvol'] >= 1.5:
                            context_data['rvol_interpretation'] = "ELEVATED (increased interest)"
                        elif context_data['rvol'] >= 0.8:
                            context_data['rvol_interpretation'] = "NORMAL"
                        else:
                            context_data['rvol_interpretation'] = "LOW (lack of conviction)"
                except:
                    context_data.update({'rvol': 1, 'rvol_interpretation': 'N/A'})
                
                # ========== SESSION BREAKDOWN ==========
                try:
                    sessions = self.flow_analysis.intraday_session_breakdown()
                    if sessions:
                        session_details = []
                        total_session_delta = 0
                        for name, data in sessions.items():
                            delta = data.get('delta', 0)
                            total_session_delta += delta
                            session_details.append(f"{name}: Δ{delta:+,.0f}, Vol:{data.get('volume', 0):,.0f}")
                        context_data['session_breakdown'] = "; ".join(session_details)
                        context_data['total_session_delta'] = total_session_delta
                        context_data['session_bias'] = "ACCUMULATION" if total_session_delta > 0 else "DISTRIBUTION" if total_session_delta < 0 else "NEUTRAL"
                except:
                    context_data.update({'session_breakdown': 'N/A', 'total_session_delta': 0, 'session_bias': 'N/A'})
                
                # ========== DELTA DIVERGENCE ==========
                try:
                    divergence = self.flow_analysis.delta_divergence()
                    if divergence:
                        latest_div = divergence[-1] if divergence else {}
                        context_data['divergence_type'] = latest_div.get('type', 'NONE')
                        context_data['divergence_strength'] = latest_div.get('strength', 0)
                except:
                    context_data.update({'divergence_type': 'NONE', 'divergence_strength': 0})
                
                # ========== ALERTS ==========
                try:
                    alerts = self.flow_analysis.generate_alerts()
                    if alerts:
                        high_alerts = [a for a in alerts if a.get('severity') == 'HIGH']
                        med_alerts = [a for a in alerts if a.get('severity') == 'MEDIUM']
                        context_data['high_alerts'] = len(high_alerts)
                        context_data['med_alerts'] = len(med_alerts)
                        context_data['alert_summary'] = "; ".join([a.get('message', '')[:50] for a in alerts[:3]])
                except:
                    context_data.update({'high_alerts': 0, 'med_alerts': 0, 'alert_summary': 'None'})
                
                # ========== SIGNAL DETAILS ==========
                signal = synthesis.get('signal_summary', {})
                context_data['signal_type'] = signal.get('type', 'NONE')
                context_data['signal_pattern'] = signal.get('pattern', '--')
                context_data['entry'] = signal.get('entry', 0)
                context_data['target'] = signal.get('target', 0)
                context_data['stop'] = signal.get('stop', 0)
                context_data['risk_reward'] = signal.get('risk_reward', 0)
                context_data['signal_confluence'] = signal.get('confluence', 0)
                
                # Build comprehensive prompt
                system_prompt = """You are an elite institutional order flow analyst for the Nigerian Stock Exchange (NGX).
Provide a detailed, professional trading analysis based on comprehensive order flow data.
Your analysis should be actionable, specific, and data-driven.
Use precise price levels and quantified observations.
Be direct and decisive in your recommendations.
Target audience: Professional traders and portfolio managers."""

                component_summary = "\n".join([
                    f"  • {k.upper()}: {v.get('score', 0):.1f}/10 — {v.get('drivers', '-')}"
                    for k, v in context_data['component_scores'].items()
                ])

                prompt = f"""Provide a comprehensive order flow analysis for {symbol} on the Nigerian Stock Exchange:

══════════════════════════════════════════════════════════════════
                    SYNTHESIS DASHBOARD
══════════════════════════════════════════════════════════════════

▸ OVERALL SCORE: {score:.0f}/100
▸ MARKET BIAS: {bias}
▸ CONFIDENCE: {confidence:.0f}%
▸ ACTION: {action}
▸ Score Drivers: {context_data.get('score_drivers', 'N/A')}
▸ Bias Drivers: {context_data.get('bias_drivers', 'N/A')}

──────────────────────────────────────────────────────────────────
                    COMPONENT BREAKDOWN
──────────────────────────────────────────────────────────────────
{component_summary}

──────────────────────────────────────────────────────────────────
                    PRICE & VWAP ANALYSIS
──────────────────────────────────────────────────────────────────
▸ Current Price: ₦{context_data.get('current_price', 0):,.2f}
▸ VWAP: ₦{context_data.get('vwap', 0):,.2f} ({context_data.get('vwap_diff_pct', 0):+.2f}% from VWAP)
▸ Upper Band +1σ: ₦{context_data.get('upper_band_1', 0):,.2f}
▸ Lower Band -1σ: ₦{context_data.get('lower_band_1', 0):,.2f}

──────────────────────────────────────────────────────────────────
                    VOLUME PROFILE
──────────────────────────────────────────────────────────────────
▸ POC (Point of Control): ₦{context_data.get('poc_price', 0):,.2f}
▸ Value Area High (VAH): ₦{context_data.get('vah', 0):,.2f}
▸ Value Area Low (VAL): ₦{context_data.get('val', 0):,.2f}
▸ Price Position: {context_data.get('price_vs_profile', 'N/A')}

──────────────────────────────────────────────────────────────────
                    DELTA ANALYSIS
──────────────────────────────────────────────────────────────────
▸ Cumulative Delta: {context_data.get('cumulative_delta', 0):+,.0f}
▸ Delta Trend (5 bars): {context_data.get('delta_trend_5bar', 0):+,.0f}
▸ Delta Momentum: {context_data.get('delta_momentum', 0):+,.0f}
▸ Delta Z-Score: {context_data.get('delta_zscore', 0):+.2f}σ
▸ Divergence: {context_data.get('divergence_type', 'NONE')} (Strength: {context_data.get('divergence_strength', 0):.0f}%)

──────────────────────────────────────────────────────────────────
                    INSTITUTIONAL FLOW
──────────────────────────────────────────────────────────────────
▸ Block Trades: {context_data.get('total_blocks', 0)} total ({context_data.get('buy_blocks', 0)} buys, {context_data.get('sell_blocks', 0)} sells)
▸ Block Imbalance: {context_data.get('block_imbalance', 'N/A')}
▸ Relative Volume (RVOL): {context_data.get('rvol', 1):.2f}x ({context_data.get('rvol_interpretation', 'N/A')})

──────────────────────────────────────────────────────────────────
                    SESSION ANALYSIS
──────────────────────────────────────────────────────────────────
▸ Session Breakdown: {context_data.get('session_breakdown', 'N/A')}
▸ Total Session Delta: {context_data.get('total_session_delta', 0):+,.0f}
▸ Session Bias: {context_data.get('session_bias', 'N/A')}

──────────────────────────────────────────────────────────────────
                    ALERTS
──────────────────────────────────────────────────────────────────
▸ HIGH Priority: {context_data.get('high_alerts', 0)} | MEDIUM Priority: {context_data.get('med_alerts', 0)}
▸ Summary: {context_data.get('alert_summary', 'None')}

──────────────────────────────────────────────────────────────────
                    TRADE SIGNAL
──────────────────────────────────────────────────────────────────
▸ Signal: {context_data.get('signal_type', 'NONE')} ({context_data.get('signal_pattern', '--')})
▸ Entry: ₦{context_data.get('entry', 0):,.2f}
▸ Target: ₦{context_data.get('target', 0):,.2f}
▸ Stop Loss: ₦{context_data.get('stop', 0):,.2f}
▸ Risk/Reward: {context_data.get('risk_reward', 0):.2f}
▸ Signal Confluence: {context_data.get('signal_confluence', 0):.0f}%

══════════════════════════════════════════════════════════════════

Based on all the above data, provide a DETAILED professional analysis including:

1. **MARKET STRUCTURE ASSESSMENT** (3-4 sentences)
   - Overall market condition and where price sits in the value area
   - Quality of the current price action

2. **ORDER FLOW INTERPRETATION**
   - What the delta readings tell us about buyer/seller aggression
   - Block trade implications for institutional positioning
   - Session flow patterns and their meaning

3. **KEY LEVELS TO WATCH**
   - Exact support levels (POC, VAL, VWAP bands)
   - Exact resistance levels (VAH, upper bands)
   - Where stop clusters likely reside

4. **TRADE RECOMMENDATION**
   - If signal present: detailed entry strategy with specific prices
   - Position sizing suggestion based on risk/reward
   - Time horizon expectation

5. **RISK FACTORS**
   - What could invalidate this thesis
   - Key levels that must hold
   - Volume/flow conditions to monitor

6. **CONFIDENCE ASSESSMENT**
   - How aligned are all the flow indicators?
   - Any conflicting signals?
   - Probability estimate for the trade"""

                ai_response = self.insight_engine.generate(prompt, system_prompt)
                logger.info(f"Groq AI response received: {len(ai_response) if ai_response else 0} chars")
                if ai_response:
                    logger.info(f"Response preview: {ai_response[:100]}...")
                if ai_response and "unavailable" not in ai_response.lower():
                    lines = []
                    lines.append(f"═══════════════════════════════════════════════════════")
                    lines.append(f"  🤖 AI SYNTHESIS REPORT: {symbol}")
                    lines.append(f"═══════════════════════════════════════════════════════")
                    lines.append(f"  Score: {score:.0f}/100 | Bias: {bias} | Action: {action}")
                    lines.append(f"═══════════════════════════════════════════════════════")
                    lines.append("")
                    lines.append(ai_response)
                    lines.append("")
                    lines.append(f"─────────────────────────────────────────────────────────")
                    lines.append(f"  Generated via Groq AI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    lines.append(f"─────────────────────────────────────────────────────────")
                    return "\n".join(lines)
                    
            except Exception as e:
                logger.error(f"Groq AI generation failed, falling back to rule-based: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info("No insight_engine available, using rule-based narrative")
        
        # Fallback to rule-based narrative
        lines = []
        
        lines.append(f"═══════════════════════════════════════════════════════")
        lines.append(f"  AI SYNTHESIS REPORT: {symbol}")
        lines.append(f"═══════════════════════════════════════════════════════")
        lines.append("")
        
        # Summary
        lines.append(f"▶ OVERALL ASSESSMENT: {action}")
        lines.append(f"  Synthesis Score: {score:.0f}/100 | Bias: {bias} | Confidence: {confidence:.0f}%")
        lines.append("")
        
        # Component breakdown
        lines.append("▶ COMPONENT ANALYSIS:")
        for key, data in synthesis.get('component_scores', {}).items():
            score_val = data.get('score', 0)
            drivers = data.get('drivers', '--')
            bar = "█" * int(score_val) + "░" * (10 - int(score_val))
            lines.append(f"  {key.title():12} [{bar}] {score_val:.1f}/10 - {drivers}")
        lines.append("")
        
        # Market context
        lines.append("▶ MARKET CONTEXT:")
        
        # Delta analysis
        try:
            cum_delta = self.flow_analysis.cumulative_delta()
            if cum_delta:
                latest_delta = cum_delta[-1].get('cumulative_delta', 0)
                lines.append(f"  • Cumulative Delta: {latest_delta:+,.0f} ({'Accumulation' if latest_delta > 0 else 'Distribution'})")
        except:
            pass
        
        # VWAP position
        try:
            vwap = self.flow_analysis.vwap_analysis()
            if vwap:
                current = vwap.get('current_price', 0)
                vwap_val = vwap.get('vwap', 0)
                diff_pct = ((current - vwap_val) / vwap_val * 100) if vwap_val else 0
                lines.append(f"  • VWAP Position: ₦{current:,.2f} vs ₦{vwap_val:,.2f} ({diff_pct:+.2f}%)")
        except:
            pass
        
        # Session summary
        try:
            sessions = self.flow_analysis.intraday_session_breakdown()
            total_delta = sum(s.get('delta', 0) for s in sessions.values())
            lines.append(f"  • Session Delta: {total_delta:+,.0f}")
        except:
            pass
        
        lines.append("")
        
        # Recommendation
        lines.append("▶ RECOMMENDATION:")
        if action == 'BUY':
            lines.append("  ✅ BULLISH OPPORTUNITY - Consider accumulating on dips")
            lines.append(f"     Drivers: {synthesis.get('action_drivers', '-')}")
        elif action == 'SELL':
            lines.append("  ⚠️ BEARISH SETUP - Consider reducing exposure")
            lines.append(f"     Drivers: {synthesis.get('action_drivers', '-')}")
        else:
            lines.append("  ⏸️ HOLD/WAIT - No clear edge currently")
            lines.append(f"     Reason: {synthesis.get('action_drivers', '-')}")
        
        lines.append("")
        lines.append(f"─────────────────────────────────────────────────────────")
        lines.append(f"  Generated (Rule-based): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  💡 Set GROQ_API_KEY for AI-powered insights")
        lines.append(f"─────────────────────────────────────────────────────────")
        
        return "\n".join(lines)
    
    def _generate_key_insights(self, synthesis: dict) -> list:
        """Generate key insight cards from synthesis data."""
        insights = []
        
        try:
            # Check delta divergence
            divergences = self.flow_analysis.delta_divergence()
            if divergences:
                latest = divergences[-1] if divergences else {}
                div_type = latest.get('divergence_type', '')
                if 'ACCUM' in div_type:
                    insights.append({
                        'icon': '🟢',
                        'title': 'ACCUMULATION',
                        'detail': 'Bullish delta divergence',
                        'color': COLORS['gain']
                    })
                elif 'DIST' in div_type:
                    insights.append({
                        'icon': '🔴',
                        'title': 'DISTRIBUTION',
                        'detail': 'Bearish delta divergence',
                        'color': COLORS['loss']
                    })
            
            # Check opening range
            or_data = self.flow_analysis.opening_range_analysis()
            if or_data:
                breakout = or_data.get('breakout', 'NO_BREAKOUT')
                if breakout == 'BULLISH_BREAKOUT':
                    insights.append({
                        'icon': '📈',
                        'title': 'OR BREAKOUT',
                        'detail': f"Above ₦{or_data.get('or_high', 0):,.2f}",
                        'color': COLORS['gain']
                    })
                elif breakout == 'BEARISH_BREAKDOWN':
                    insights.append({
                        'icon': '📉',
                        'title': 'OR BREAKDOWN',
                        'detail': f"Below ₦{or_data.get('or_low', 0):,.2f}",
                        'color': COLORS['loss']
                    })
            
            # Check RVOL spikes
            rvol = self.flow_analysis.rvol_analysis()
            if rvol and rvol.get('rvol', 1) >= 2:
                insights.append({
                    'icon': '📊',
                    'title': 'HIGH VOLUME',
                    'detail': f"RVOL {rvol['rvol']:.1f}x average",
                    'color': COLORS['warning']
                })
            
            # Check blocks
            blocks = self.flow_analysis.block_trade_analysis()
            if blocks and blocks.get('total_blocks', 0) >= 3:
                insights.append({
                    'icon': '🏛️',
                    'title': 'INSTITUTIONAL',
                    'detail': f"{blocks['total_blocks']} block trades",
                    'color': COLORS['primary']
                })
            
            # Add synthesis-based insight
            if synthesis.get('synthesis_score', 0) >= 75:
                insights.append({
                    'icon': '⭐',
                    'title': 'STRONG SETUP',
                    'detail': f"Score {synthesis['synthesis_score']:.0f}/100",
                    'color': COLORS['gain']
                })
            elif synthesis.get('synthesis_score', 0) <= 30:
                insights.append({
                    'icon': '⚠️',
                    'title': 'WEAK SETUP',
                    'detail': f"Score {synthesis['synthesis_score']:.0f}/100",
                    'color': COLORS['loss']
                })
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights[:4]  # Max 4 insights
    
    def _update_fundamentals_tab(self):
        """Update the Fundamentals tab with data from TradingView screener."""
        if not self.current_symbol:
            return
        
        try:
            from src.collectors.tradingview_collector import TradingViewCollector
            
            collector = TradingViewCollector()
            fund_data = collector.get_fundamental_data(self.current_symbol)
            
            if not fund_data:
                self.valuation_status.config(
                    text="❌ Fundamental data not available for this stock",
                    foreground=COLORS['loss']
                )
                return
            
            fundamentals = fund_data.get('fundamentals', {})
            performance = fund_data.get('performance', {})
            
            # ========== UPDATE KEY METRICS CARDS ==========
            # Market Cap
            mcap = fundamentals.get('market_cap', 0)
            if mcap and mcap > 0:
                if mcap >= 1e12:
                    mcap_text = f"₦{mcap/1e12:.1f}T"
                elif mcap >= 1e9:
                    mcap_text = f"₦{mcap/1e9:.1f}B"
                elif mcap >= 1e6:
                    mcap_text = f"₦{mcap/1e6:.1f}M"
                else:
                    mcap_text = f"₦{mcap:,.0f}"
                self.fundamental_cards['market_cap'].config(text=mcap_text)
            
            # P/E Ratio
            pe = fundamentals.get('pe_ratio')
            if pe is not None:
                pe_color = COLORS['gain'] if pe < 15 else COLORS['loss'] if pe > 25 else COLORS['warning']
                self.fundamental_cards['pe_ratio'].config(text=f"{pe:.1f}x", foreground=pe_color)
                self.valuation_labels['pe_ratio'].config(text=f"{pe:.2f}")
            
            # P/B Ratio
            pb = fundamentals.get('pb_ratio')
            if pb is not None:
                pb_color = COLORS['gain'] if pb < 1.5 else COLORS['loss'] if pb > 3 else COLORS['warning']
                self.fundamental_cards['pb_ratio'].config(text=f"{pb:.1f}x", foreground=pb_color)
                self.valuation_labels['pb_ratio'].config(text=f"{pb:.2f}")
            
            # EPS
            eps = fundamentals.get('eps')
            if eps is not None:
                eps_color = COLORS['gain'] if eps > 0 else COLORS['loss']
                self.fundamental_cards['eps'].config(text=f"₦{eps:.2f}", foreground=eps_color)
            
            # Dividend Yield
            div_yield = fundamentals.get('dividend_yield')
            if div_yield is not None:
                div_color = COLORS['gain'] if div_yield > 3 else COLORS['text_secondary']
                self.fundamental_cards['dividend'].config(text=f"{div_yield:.1f}%", foreground=div_color)
            else:
                self.fundamental_cards['dividend'].config(text="N/A", foreground=COLORS['text_muted'])
            
            # ========== UPDATE VALUATION RATIOS ==========
            ps = fundamentals.get('ps_ratio')
            if ps is not None:
                self.valuation_labels['ps_ratio'].config(text=f"{ps:.2f}")
            
            bv = fundamentals.get('book_value')
            if bv is not None:
                self.valuation_labels['book_value'].config(text=f"₦{bv:.2f}")
            
            shares = fundamentals.get('shares_outstanding')
            if shares is not None:
                if shares >= 1e9:
                    shares_text = f"{shares/1e9:.1f}B"
                elif shares >= 1e6:
                    shares_text = f"{shares/1e6:.1f}M"
                else:
                    shares_text = f"{shares:,.0f}"
                self.valuation_labels['shares_out'].config(text=shares_text)
            
            # ========== UPDATE FINANCIAL HEALTH ==========
            roe = fundamentals.get('roe')
            if roe is not None:
                roe_color = COLORS['gain'] if roe > 15 else COLORS['loss'] if roe < 0 else COLORS['warning']
                self.health_labels['roe'].config(text=f"{roe:.1f}%", foreground=roe_color)
            
            de = fundamentals.get('debt_to_equity')
            if de is not None:
                de_color = COLORS['gain'] if de < 1 else COLORS['loss'] if de > 2 else COLORS['warning']
                self.health_labels['debt_equity'].config(text=f"{de:.2f}", foreground=de_color)
            
            cr = fundamentals.get('current_ratio')
            if cr is not None:
                cr_color = COLORS['gain'] if cr > 1.5 else COLORS['loss'] if cr < 1 else COLORS['warning']
                self.health_labels['current_ratio'].config(text=f"{cr:.2f}", foreground=cr_color)
            
            revenue = fundamentals.get('revenue')
            if revenue is not None and revenue > 0:
                if revenue >= 1e12:
                    rev_text = f"₦{revenue/1e12:.1f}T"
                elif revenue >= 1e9:
                    rev_text = f"₦{revenue/1e9:.1f}B"
                elif revenue >= 1e6:
                    rev_text = f"₦{revenue/1e6:.1f}M"
                else:
                    rev_text = f"₦{revenue:,.0f}"
                self.health_labels['revenue'].config(text=rev_text)
            
            net_inc = fundamentals.get('net_income')
            if net_inc is not None:
                ni_color = COLORS['gain'] if net_inc > 0 else COLORS['loss']
                if abs(net_inc) >= 1e9:
                    ni_text = f"₦{net_inc/1e9:.1f}B"
                elif abs(net_inc) >= 1e6:
                    ni_text = f"₦{net_inc/1e6:.1f}M"
                else:
                    ni_text = f"₦{net_inc:,.0f}"
                self.health_labels['net_income'].config(text=ni_text, foreground=ni_color)
            
            # ========== UPDATE PERFORMANCE ==========
            for key in ['week', 'month', 'quarter', 'half_year', 'ytd', 'year']:
                perf_val = performance.get(key)
                if perf_val is not None:
                    color = COLORS['gain'] if perf_val > 0 else COLORS['loss']
                    self.perf_labels[key].config(
                        text=f"{perf_val:+.1f}%",
                        foreground=color
                    )
            
            # ========== VALUATION ASSESSMENT ==========
            self._calculate_valuation_assessment(fundamentals)
            
        except Exception as e:
            logger.error(f"Error updating fundamentals tab: {e}")
            self.valuation_status.config(
                text=f"⚠️ Error loading data: {str(e)[:50]}",
                foreground=COLORS['warning']
            )
    
    def _calculate_valuation_assessment(self, fundamentals):
        """Calculate overall valuation assessment."""
        try:
            scores = []
            details = []
            
            # P/E Score
            pe = fundamentals.get('pe_ratio')
            if pe is not None and pe > 0:
                if pe < 10:
                    scores.append(2)
                    details.append("P/E < 10 (Undervalued)")
                elif pe < 20:
                    scores.append(1)
                    details.append("P/E 10-20 (Fair)")
                elif pe < 30:
                    scores.append(0)
                    details.append("P/E 20-30 (Premium)")
                else:
                    scores.append(-1)
                    details.append("P/E > 30 (Expensive)")
            
            # P/B Score
            pb = fundamentals.get('pb_ratio')
            if pb is not None and pb > 0:
                if pb < 1:
                    scores.append(2)
                    details.append("P/B < 1 (Below Book)")
                elif pb < 2:
                    scores.append(1)
                    details.append("P/B 1-2 (Fair)")
                elif pb < 3:
                    scores.append(0)
                    details.append("P/B 2-3 (Premium)")
                else:
                    scores.append(-1)
                    details.append("P/B > 3 (Expensive)")
            
            # ROE Score
            roe = fundamentals.get('roe')
            if roe is not None:
                if roe > 20:
                    scores.append(2)
                    details.append("ROE > 20% (Excellent)")
                elif roe > 10:
                    scores.append(1)
                    details.append("ROE > 10% (Good)")
                elif roe > 0:
                    scores.append(0)
                    details.append("ROE > 0% (Moderate)")
                else:
                    scores.append(-1)
                    details.append("ROE < 0% (Unprofitable)")
            
            if scores:
                avg_score = sum(scores) / len(scores)
                
                if avg_score >= 1.5:
                    status = "🟢 UNDERVALUED"
                    color = COLORS['gain']
                elif avg_score >= 0.5:
                    status = "🟡 FAIRLY VALUED"
                    color = COLORS['warning']
                elif avg_score >= -0.5:
                    status = "🟠 PREMIUM VALUED"
                    color = '#FF8C00'
                else:
                    status = "🔴 OVERVALUED"
                    color = COLORS['loss']
                
                self.valuation_status.config(
                    text=f"{status} (Score: {avg_score:.1f})",
                    foreground=color
                )
                self.valuation_details.config(
                    text=" • ".join(details[:3])
                )
            else:
                self.valuation_status.config(
                    text="⚪ Insufficient data for assessment",
                    foreground=COLORS['text_muted']
                )
                
        except Exception as e:
            logger.error(f"Error calculating valuation: {e}")
    
    def _draw_session_delta_chart(self, sessions):
        """Draw session delta bar chart."""
        canvas = self.session_chart_canvas
        canvas.delete('all')
        
        canvas.update_idletasks()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width < 50 or height < 30:
            width, height = 400, 120
        
        # Data
        session_keys = ['open', 'core', 'close']
        deltas = [sessions.get(k, {}).get('delta', 0) for k in session_keys]
        labels = ['Opening', 'Core', 'Closing']
        
        if max(abs(d) for d in deltas) == 0:
            canvas.create_text(width/2, height/2, text="No session data", 
                             fill=COLORS['text_muted'], font=get_font('small'))
            return
        
        # Chart dimensions
        padding = 40
        chart_width = width - padding * 2
        chart_height = height - padding
        bar_width = chart_width / (len(deltas) * 2)
        
        max_delta = max(abs(d) for d in deltas) or 1
        
        # Zero line
        zero_y = padding + chart_height / 2
        canvas.create_line(padding, zero_y, width - padding, zero_y, 
                          fill=COLORS['text_muted'], dash=(2, 2))
        
        # Bars
        for i, (delta, label) in enumerate(zip(deltas, labels)):
            x = padding + (i * 2 + 0.5) * bar_width
            bar_h = (delta / max_delta) * (chart_height / 2 - 10)
            
            color = COLORS['gain'] if delta >= 0 else COLORS['loss']
            
            if delta >= 0:
                canvas.create_rectangle(
                    x, zero_y - bar_h,
                    x + bar_width, zero_y,
                    fill=color, outline=''
                )
            else:
                canvas.create_rectangle(
                    x, zero_y,
                    x + bar_width, zero_y - bar_h,
                    fill=color, outline=''
                )
            
            # Value label
            canvas.create_text(
                x + bar_width / 2, 
                zero_y - bar_h - 8 if delta >= 0 else zero_y - bar_h + 12,
                text=f"{delta:+,.0f}", fill=color, font=get_font('tiny')
            )
            
            # Session label
            canvas.create_text(
                x + bar_width / 2, height - 10,
                text=label, fill=COLORS['text_muted'], font=get_font('tiny')
            )
    
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
        """Legacy method - now handled by _update_sessions_tab."""
        pass  # All session analytics now in _update_sessions_tab
    
    def refresh(self):
        """Refresh the current display."""
        self._load_data()
