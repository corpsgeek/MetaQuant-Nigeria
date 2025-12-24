"""
Market Intelligence Tab for MetaQuant Nigeria.
Consolidates Live Market, Sector Rotation, and Flow Analysis.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict
import logging
import threading
import queue
from datetime import datetime

try:
    import ttkbootstrap as ttk_bs
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.collectors.tradingview_collector import TradingViewCollector
from src.analysis.sector_analysis import SectorAnalysis
from src.gui.theme import COLORS, get_font

import os

# Try to import AI Insight Engine
try:
    from src.ai.insight_engine import InsightEngine
    INSIGHT_ENGINE_AVAILABLE = True
except ImportError:
    INSIGHT_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketIntelligenceTab:
    """
    Market Intelligence mega-tab with sub-views:
    - Live Market overview (using rich LiveMarketTab)
    - Sector Rotation analysis
    - Flow Monitor
    """
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.collector = TradingViewCollector()
        self.sector_analysis = SectorAnalysis(db)
        self.all_stocks_data = []
        self._update_queue = queue.Queue()  # Thread-safe queue for UI updates
        self.insight_engine = None  # AI Insight Engine for Groq
        
        # Initialize AI Insight Engine if available
        if INSIGHT_ENGINE_AVAILABLE:
            try:
                groq_api_key = os.environ.get('GROQ_API_KEY')
                if groq_api_key:
                    self.insight_engine = InsightEngine(groq_api_key=groq_api_key)
                    logger.info("Market Intel: AI Insight Engine initialized with Groq")
                else:
                    logger.warning("Market Intel: GROQ_API_KEY not found")
            except Exception as e:
                logger.warning(f"Market Intel: Failed to initialize AI Insight Engine: {e}")
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        # Create internal notebook for sub-tabs
        self._create_sub_tabs()
        
        # Load data for sector rotation and flow tabs
        self._load_sector_data()
    
    def _create_sub_tabs(self):
        """Create sub-tabs within Market Intelligence."""
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Use the rich LiveMarketTab directly
        from src.gui.tabs.live_market_tab import LiveMarketTab
        self.live_market_tab = LiveMarketTab(self.sub_notebook, self.db)
        
        # Import Smart Money Detector
        from src.analysis.smart_money_detector import SmartMoneyDetector, AnomalyScanner
        self.smart_money_detector = SmartMoneyDetector(self.db)
        self.anomaly_scanner = AnomalyScanner()
        
        # Create frames for other sub-tabs
        self.sector_frame = ttk.Frame(self.sub_notebook)
        self.flow_frame = ttk.Frame(self.sub_notebook)
        self.smart_money_frame = ttk.Frame(self.sub_notebook)
        self.synthesis_frame = ttk.Frame(self.sub_notebook)
        
        # Add tabs - use LiveMarketTab's frame directly
        self.sub_notebook.add(self.live_market_tab.frame, text="📈 Live Market")
        self.sub_notebook.add(self.sector_frame, text="🔄 Sector Rotation")
        self.sub_notebook.add(self.flow_frame, text="💧 Flow Monitor")
        self.sub_notebook.add(self.smart_money_frame, text="🕵️ Smart Money")
        self.sub_notebook.add(self.synthesis_frame, text="🤖 AI Synthesis")
        
        # Build other sub-tab UIs
        self._build_sector_rotation_ui()
        self._build_flow_monitor_ui()
        self._build_smart_money_ui()
        self._build_ai_synthesis_ui()
    
    # ==================== SECTOR ROTATION ====================
    
    def _build_sector_rotation_ui(self):
        """Build super rich sector rotation analysis view."""
        # Header
        header = ttk.Frame(self.sector_frame)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            header,
            text="🔄 Sector Rotation Analysis",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        ttk.Label(
            header,
            text="Market-cap weighted sector indices with rotation tracking",
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        ).pack(side=tk.LEFT, padx=20)
        
        if TTKBOOTSTRAP_AVAILABLE:
            refresh_btn = ttk_bs.Button(
                header,
                text="↻ Analyze Sectors",
                bootstyle="info-outline",
                command=self._load_sector_data
            )
        else:
            refresh_btn = ttk.Button(header, text="↻ Analyze Sectors", command=self._load_sector_data)
        refresh_btn.pack(side=tk.RIGHT)
        
        # ===== ROW 1: Sector Momentum Matrix + Rotation Clock =====
        row1 = ttk.Frame(self.sector_frame)
        row1.pack(fill=tk.X, padx=10, pady=5)
        
        # LEFT: Sector Momentum Matrix (simpler table format)
        matrix_frame = ttk.LabelFrame(row1, text="📊 Sector Momentum", padding=5)
        matrix_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Use Treeview instead of canvas for clarity
        mm_columns = ('sector', 'chg_1d', 'chg_1w', 'chg_1m')
        self.momentum_tree = ttk.Treeview(matrix_frame, columns=mm_columns, show='headings', height=6)
        
        self.momentum_tree.heading('sector', text='Sector')
        self.momentum_tree.heading('chg_1d', text='1D')
        self.momentum_tree.heading('chg_1w', text='1W')
        self.momentum_tree.heading('chg_1m', text='1M')
        
        self.momentum_tree.column('sector', width=90, anchor='w')
        self.momentum_tree.column('chg_1d', width=55, anchor='e')
        self.momentum_tree.column('chg_1w', width=55, anchor='e')
        self.momentum_tree.column('chg_1m', width=55, anchor='e')
        
        self.momentum_tree.tag_configure('hot', foreground='#00ff00')
        self.momentum_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.momentum_tree.tag_configure('loss', foreground=COLORS['loss'])
        self.momentum_tree.tag_configure('cold', foreground='#ff4444')
        
        self.momentum_tree.pack(fill=tk.BOTH, expand=True)
        self.momentum_tree.bind('<<TreeviewSelect>>', self._on_momentum_select)
        
        # RIGHT: Rotation Clock + Phase Info
        clock_frame = ttk.LabelFrame(row1, text="🕐 Rotation Cycle", padding=10)
        clock_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Clock canvas
        self.clock_canvas = tk.Canvas(
            clock_frame,
            bg=COLORS['bg_dark'],
            width=140,
            height=100,
            highlightthickness=0
        )
        self.clock_canvas.pack(side=tk.LEFT, padx=5)
        
        # Phase info
        phase_info = ttk.Frame(clock_frame)
        phase_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.phase_labels = {}
        
        ttk.Label(phase_info, text="Phase:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(anchor='w')
        self.phase_labels['phase'] = ttk.Label(phase_info, text="--", 
                                                font=get_font('body'),
                                                foreground=COLORS['primary'])
        self.phase_labels['phase'].pack(anchor='w', pady=(0, 8))
        
        ttk.Label(phase_info, text="Leading:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(anchor='w')
        self.phase_labels['leading'] = ttk.Label(phase_info, text="--", 
                                                  font=get_font('small'),
                                                  foreground=COLORS['gain'])
        self.phase_labels['leading'].pack(anchor='w', pady=(0, 5))
        
        ttk.Label(phase_info, text="Lagging:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(anchor='w')
        self.phase_labels['lagging'] = ttk.Label(phase_info, text="--", 
                                                  font=get_font('small'),
                                                  foreground=COLORS['loss'])
        self.phase_labels['lagging'].pack(anchor='w')
        
        # ===== ROW 2: Correlation Heatmap + RS Leaderboard =====
        row2 = ttk.Frame(self.sector_frame)
        row2.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # LEFT: Sector Correlation (simplified to top pairs only)
        corr_frame = ttk.LabelFrame(row2, text="🔗 Correlation Pairs", padding=5)
        corr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        corr_columns = ('pair', 'correlation', 'direction')
        self.corr_tree = ttk.Treeview(corr_frame, columns=corr_columns, show='headings', height=6)
        
        self.corr_tree.heading('pair', text='Sector Pair')
        self.corr_tree.heading('correlation', text='Corr')
        self.corr_tree.heading('direction', text='Move')
        
        self.corr_tree.column('pair', width=140, anchor='w')
        self.corr_tree.column('correlation', width=50, anchor='e')
        self.corr_tree.column('direction', width=70, anchor='center')
        
        self.corr_tree.tag_configure('positive', foreground=COLORS['gain'])
        self.corr_tree.tag_configure('negative', foreground=COLORS['loss'])
        
        self.corr_tree.pack(fill=tk.BOTH, expand=True)
        
        # RIGHT: RS Leaderboard
        rs_frame = ttk.LabelFrame(row2, text="💪 RS Leaders (vs Sector)", padding=5)
        rs_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        rs_columns = ('symbol', 'sector', 'vs_sector')
        self.rs_tree = ttk.Treeview(rs_frame, columns=rs_columns, show='headings', height=6)
        
        self.rs_tree.heading('symbol', text='Symbol')
        self.rs_tree.heading('sector', text='Sector')
        self.rs_tree.heading('vs_sector', text='Outperform')
        
        self.rs_tree.column('symbol', width=65, anchor='w')
        self.rs_tree.column('sector', width=80, anchor='w')
        self.rs_tree.column('vs_sector', width=70, anchor='e')
        
        self.rs_tree.tag_configure('outperform', foreground=COLORS['gain'])
        self.rs_tree.tag_configure('strong', foreground='#00ff00')
        
        self.rs_tree.pack(fill=tk.BOTH, expand=True)
        
        # ===== ROW 3: Sector Rankings + Components =====
        row3 = ttk.Frame(self.sector_frame)
        row3.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # LEFT: Sector Rankings
        left_frame = ttk.LabelFrame(row3, text="📊 Sector Rankings", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        columns = ('sector', 'chg_1d', 'chg_1w', 'count')
        self.sector_tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=5)
        
        self.sector_tree.heading('sector', text='Sector')
        self.sector_tree.heading('chg_1d', text='1D')
        self.sector_tree.heading('chg_1w', text='1W')
        self.sector_tree.heading('count', text='#')
        
        self.sector_tree.column('sector', width=100, anchor='w')
        self.sector_tree.column('chg_1d', width=55, anchor='e')
        self.sector_tree.column('chg_1w', width=55, anchor='e')
        self.sector_tree.column('count', width=30, anchor='center')
        
        self.sector_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.sector_tree.tag_configure('loss', foreground=COLORS['loss'])
        
        self.sector_tree.pack(fill=tk.BOTH, expand=True)
        self.sector_tree.bind('<<TreeviewSelect>>', self._on_sector_select)
        
        # RIGHT: Sector Components
        right_frame = ttk.LabelFrame(row3, text="📋 Sector Components (click to load)", padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.selected_sector_label = ttk.Label(right_frame, text="Select a sector →", 
                                                font=get_font('small'),
                                                foreground=COLORS['text_muted'])
        self.selected_sector_label.pack(anchor='w')
        
        columns = ('symbol', 'price', 'chg_1d', 'chg_1w', 'trend')
        self.component_tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=5)
        
        self.component_tree.heading('symbol', text='Symbol')
        self.component_tree.heading('price', text='Price')
        self.component_tree.heading('chg_1d', text='1D')
        self.component_tree.heading('chg_1w', text='1W')
        self.component_tree.heading('trend', text='Trend')
        
        self.component_tree.column('symbol', width=60, anchor='w')
        self.component_tree.column('price', width=60, anchor='e')
        self.component_tree.column('chg_1d', width=50, anchor='e')
        self.component_tree.column('chg_1w', width=50, anchor='e')
        self.component_tree.column('trend', width=60, anchor='center')
        
        self.component_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.component_tree.tag_configure('loss', foreground=COLORS['loss'])
        
        self.component_tree.pack(fill=tk.BOTH, expand=True)
    
    # ==================== FLOW MONITOR ====================
    
    def _build_flow_monitor_ui(self):
        """Build super rich flow monitor view."""
        # Header
        header = ttk.Frame(self.flow_frame)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            header,
            text="💧 Flow Monitor",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        ttk.Label(
            header,
            text="Multi-timeframe relative strength analysis",
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        ).pack(side=tk.LEFT, padx=20)
        
        if TTKBOOTSTRAP_AVAILABLE:
            refresh_btn = ttk_bs.Button(
                header,
                text="↻ Analyze Flows",
                bootstyle="warning-outline",
                command=self._load_flow_data
            )
        else:
            refresh_btn = ttk.Button(header, text="↻ Analyze Flows", command=self._load_flow_data)
        refresh_btn.pack(side=tk.RIGHT)
        
        # ===== TOP ROW: Summary Cards =====
        summary_frame = ttk.Frame(self.flow_frame)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.flow_cards = {}
        flow_types = [
            ('strong_inflow', '🚀 Strong Inflow', '+5%+ (1W)', COLORS['gain']),
            ('moderate_inflow', '📈 Inflow', '+0-5% (1W)', COLORS['gain']),
            ('neutral', '➡️ Neutral', '±0% (1W)', COLORS['text_muted']),
            ('moderate_outflow', '📉 Outflow', '-0-5% (1W)', COLORS['loss']),
            ('strong_outflow', '💨 Strong Outflow', '-5%+ (1W)', COLORS['loss']),
        ]
        
        for key, title, desc, color in flow_types:
            card = ttk.LabelFrame(summary_frame, text=title, padding=8)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            
            count_label = ttk.Label(card, text="0", font=get_font('subheading'),
                                    foreground=color)
            count_label.pack()
            
            ttk.Label(card, text=desc, font=get_font('small'),
                     foreground=COLORS['text_secondary']).pack()
            
            self.flow_cards[key] = count_label
        
        # ===== PRESSURE GAUGE ROW: Money Flow Pressure + Duration Classification =====
        gauge_row = ttk.Frame(self.flow_frame)
        gauge_row.pack(fill=tk.X, padx=10, pady=5)
        
        # LEFT: Money Flow Pressure Gauge
        pressure_frame = ttk.LabelFrame(gauge_row, text="⚡ Money Flow Pressure", padding=8)
        pressure_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Pressure gauge canvas
        self.pressure_canvas = tk.Canvas(
            pressure_frame,
            bg=COLORS['bg_dark'],
            height=65,
            highlightthickness=0
        )
        self.pressure_canvas.pack(fill=tk.X, pady=(0, 3))
        
        # Pressure stats row 1 (centered)
        pressure_stats = ttk.Frame(pressure_frame)
        pressure_stats.pack(anchor='center')
        
        self.pressure_labels = {}
        for key, title in [('buying', '📈 Buying:'), ('selling', '📉 Selling:'), ('net', '⚡ Net:')]:
            frame = ttk.Frame(pressure_stats)
            frame.pack(side=tk.LEFT, padx=10)
            ttk.Label(frame, text=title, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            lbl = ttk.Label(frame, text="--", font=get_font('small'))
            lbl.pack(side=tk.LEFT, padx=3)
            self.pressure_labels[key] = lbl
        
        # Pressure stats row 2 (centered)
        pressure_stats2 = ttk.Frame(pressure_frame)
        pressure_stats2.pack(anchor='center', pady=(3, 0))
        
        for key, title in [('breadth', '📊 Breadth:'), ('avg_chg', 'Avg Chg:'), ('high_vol', '🔥 High Vol:')]:
            frame = ttk.Frame(pressure_stats2)
            frame.pack(side=tk.LEFT, padx=10)
            ttk.Label(frame, text=title, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            lbl = ttk.Label(frame, text="--", font=get_font('small'))
            lbl.pack(side=tk.LEFT, padx=3)
            self.pressure_labels[key] = lbl
        
        # Top movers row (centered)
        movers_row = ttk.Frame(pressure_frame)
        movers_row.pack(anchor='center', pady=(5, 0))
        
        ttk.Label(movers_row, text="🏆 Top:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.pressure_labels['top_gainer'] = ttk.Label(movers_row, text="--", font=get_font('small'),
                                                       foreground=COLORS['gain'])
        self.pressure_labels['top_gainer'].pack(side=tk.LEFT, padx=5)
        
        ttk.Label(movers_row, text="📉 Worst:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=(15, 0))
        self.pressure_labels['top_loser'] = ttk.Label(movers_row, text="--", font=get_font('small'),
                                                      foreground=COLORS['loss'])
        self.pressure_labels['top_loser'].pack(side=tk.LEFT, padx=5)
        
        # RIGHT: Duration-Based Classification
        duration_frame = ttk.LabelFrame(gauge_row, text="⏱️ Flow Duration Classification", padding=8)
        duration_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        dur_cols = ('type', 'count', 'description')
        self.duration_tree = ttk.Treeview(duration_frame, columns=dur_cols, show='headings', height=3)
        
        self.duration_tree.heading('type', text='Type')
        self.duration_tree.heading('count', text='Count')
        self.duration_tree.heading('description', text='Description')
        
        self.duration_tree.column('type', width=80, anchor='w')
        self.duration_tree.column('count', width=50, anchor='center')
        self.duration_tree.column('description', width=200, anchor='w')
        
        self.duration_tree.tag_configure('durable', foreground='#00ff00')
        self.duration_tree.tag_configure('shortterm', foreground='#f39c12')
        self.duration_tree.tag_configure('oneoff', foreground=COLORS['text_muted'])
        
        self.duration_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.duration_tree.bind('<<TreeviewSelect>>', self._on_duration_select)
        
        # Initialize duration data storage
        self._duration_data = {'durable': [], 'shortterm': [], 'oneoff': []}
        
        # Securities detail panel (shows when a duration type is selected)
        self.duration_detail_frame = ttk.LabelFrame(duration_frame, text="📋 Securities", padding=3)
        self.duration_detail_frame.pack(fill=tk.X, pady=(5, 0))
        
        detail_cols = ('symbol', 'chg_1d', 'chg_1w', 'chg_1m')
        self.duration_detail_tree = ttk.Treeview(self.duration_detail_frame, columns=detail_cols, 
                                                  show='headings', height=4)
        
        self.duration_detail_tree.heading('symbol', text='Symbol')
        self.duration_detail_tree.heading('chg_1d', text='1D%')
        self.duration_detail_tree.heading('chg_1w', text='1W%')
        self.duration_detail_tree.heading('chg_1m', text='1M%')
        
        self.duration_detail_tree.column('symbol', width=70, anchor='w')
        self.duration_detail_tree.column('chg_1d', width=55, anchor='e')
        self.duration_detail_tree.column('chg_1w', width=55, anchor='e')
        self.duration_detail_tree.column('chg_1m', width=55, anchor='e')
        
        self.duration_detail_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.duration_detail_tree.tag_configure('loss', foreground=COLORS['loss'])
        
        self.duration_detail_tree.pack(fill=tk.BOTH, expand=True)
        
        # ===== MIDDLE ROW: Sector Flows + Top Movers =====
        middle_frame = ttk.Frame(self.flow_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # LEFT: Sector Flow Aggregation
        sector_flow_frame = ttk.LabelFrame(middle_frame, text="📊 Sector Flow Summary", padding=10)
        sector_flow_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Sector flow heatmap canvas
        self.sector_flow_canvas = tk.Canvas(
            sector_flow_frame,
            bg=COLORS['bg_dark'],
            height=150,
            highlightthickness=0
        )
        self.sector_flow_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Sector flow table
        sector_cols = ('sector', 'avg_1w', 'inflows', 'outflows', 'net')
        self.sector_flow_tree = ttk.Treeview(sector_flow_frame, columns=sector_cols, 
                                              show='headings', height=5)
        
        self.sector_flow_tree.heading('sector', text='Sector')
        self.sector_flow_tree.heading('avg_1w', text='Avg 1W')
        self.sector_flow_tree.heading('inflows', text='In')
        self.sector_flow_tree.heading('outflows', text='Out')
        self.sector_flow_tree.heading('net', text='Net')
        
        self.sector_flow_tree.column('sector', width=100, anchor='w')
        self.sector_flow_tree.column('avg_1w', width=60, anchor='e')
        self.sector_flow_tree.column('inflows', width=40, anchor='center')
        self.sector_flow_tree.column('outflows', width=40, anchor='center')
        self.sector_flow_tree.column('net', width=50, anchor='center')
        
        self.sector_flow_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.sector_flow_tree.tag_configure('loss', foreground=COLORS['loss'])
        
        self.sector_flow_tree.pack(fill=tk.X)
        
        # RIGHT: Top Movers (Inflows & Outflows)
        movers_frame = ttk.Frame(middle_frame)
        movers_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Top Inflows
        inflow_frame = ttk.LabelFrame(movers_frame, text="🚀 Top Inflows (1W)", padding=5)
        inflow_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Header row
        header_row = ttk.Frame(inflow_frame)
        header_row.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(header_row, text="Symbol", width=12, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        header_perf = ttk.Frame(header_row)
        header_perf.pack(side=tk.RIGHT)
        ttk.Label(header_perf, text="1D", width=6, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=2)
        ttk.Label(header_perf, text="1W", width=6, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=2)
        ttk.Label(header_perf, text="1M", width=6, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=2)
        
        self.inflow_labels = []
        for i in range(5):
            row = ttk.Frame(inflow_frame)
            row.pack(fill=tk.X, pady=2)
            
            sym_lbl = ttk.Label(row, text="--", width=12, font=get_font('small'),
                               foreground=COLORS['gain'], cursor="hand2")
            sym_lbl.pack(side=tk.LEFT)
            
            perf_frame = ttk.Frame(row)
            perf_frame.pack(side=tk.RIGHT)
            
            d1_lbl = ttk.Label(perf_frame, text="--", width=6, font=get_font('small'))
            d1_lbl.pack(side=tk.LEFT, padx=2)
            
            w1_lbl = ttk.Label(perf_frame, text="--", width=6, font=get_font('small'),
                              foreground=COLORS['gain'])
            w1_lbl.pack(side=tk.LEFT, padx=2)
            
            m1_lbl = ttk.Label(perf_frame, text="--", width=6, font=get_font('small'))
            m1_lbl.pack(side=tk.LEFT, padx=2)
            
            self.inflow_labels.append((sym_lbl, d1_lbl, w1_lbl, m1_lbl))
        
        # Top Outflows
        outflow_frame = ttk.LabelFrame(movers_frame, text="💨 Top Outflows (1W)", padding=5)
        outflow_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Header row
        header_row2 = ttk.Frame(outflow_frame)
        header_row2.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(header_row2, text="Symbol", width=12, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        header_perf2 = ttk.Frame(header_row2)
        header_perf2.pack(side=tk.RIGHT)
        ttk.Label(header_perf2, text="1D", width=6, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=2)
        ttk.Label(header_perf2, text="1W", width=6, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=2)
        ttk.Label(header_perf2, text="1M", width=6, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=2)
        
        self.outflow_labels = []
        for i in range(5):
            row = ttk.Frame(outflow_frame)
            row.pack(fill=tk.X, pady=2)
            
            sym_lbl = ttk.Label(row, text="--", width=12, font=get_font('small'),
                               foreground=COLORS['loss'], cursor="hand2")
            sym_lbl.pack(side=tk.LEFT)
            
            perf_frame = ttk.Frame(row)
            perf_frame.pack(side=tk.RIGHT)
            
            d1_lbl = ttk.Label(perf_frame, text="--", width=6, font=get_font('small'))
            d1_lbl.pack(side=tk.LEFT, padx=2)
            
            w1_lbl = ttk.Label(perf_frame, text="--", width=6, font=get_font('small'),
                              foreground=COLORS['loss'])
            w1_lbl.pack(side=tk.LEFT, padx=2)
            
            m1_lbl = ttk.Label(perf_frame, text="--", width=6, font=get_font('small'))
            m1_lbl.pack(side=tk.LEFT, padx=2)
            
            self.outflow_labels.append((sym_lbl, d1_lbl, w1_lbl, m1_lbl))
        
        # ===== BOTTOM: Sector Detail Panel (collapsible - shows when sector is clicked) =====
        self.sector_detail_frame = ttk.LabelFrame(self.flow_frame, text="📊 Select a sector above to see securities", padding=5)
        self.sector_detail_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Sector detail info row
        self.sector_detail_info = ttk.Frame(self.sector_detail_frame)
        self.sector_detail_info.pack(fill=tk.X, pady=(0, 5))
        
        self.sector_detail_labels = {}
        for key, title in [('name', 'Sector:'), ('avg', 'Avg 1W:'), ('inflows', '📈 Inflows:'), ('outflows', '📉 Outflows:')]:
            frame = ttk.Frame(self.sector_detail_info)
            frame.pack(side=tk.LEFT, padx=8)
            ttk.Label(frame, text=title, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            lbl = ttk.Label(frame, text="--", font=get_font('small'))
            lbl.pack(side=tk.LEFT, padx=3)
            self.sector_detail_labels[key] = lbl
        
        # Container for tree (hidden by default)
        self.sector_detail_container = ttk.Frame(self.sector_detail_frame)
        # Don't pack yet - will show when sector is clicked
        
        # Sector securities tree with Flow column
        sec_cols = ('symbol', 'flow', 'chg_1d', 'chg_1w', 'chg_1m')
        self.sector_detail_tree = ttk.Treeview(self.sector_detail_container, columns=sec_cols, 
                                                show='headings', height=5)
        
        self.sector_detail_tree.heading('symbol', text='Symbol')
        self.sector_detail_tree.heading('flow', text='Flow')
        self.sector_detail_tree.heading('chg_1d', text='1D%')
        self.sector_detail_tree.heading('chg_1w', text='1W%')
        self.sector_detail_tree.heading('chg_1m', text='1M%')
        
        self.sector_detail_tree.column('symbol', width=80, anchor='w')
        self.sector_detail_tree.column('flow', width=70, anchor='center')
        self.sector_detail_tree.column('chg_1d', width=60, anchor='e')
        self.sector_detail_tree.column('chg_1w', width=60, anchor='e')
        self.sector_detail_tree.column('chg_1m', width=60, anchor='e')
        
        self.sector_detail_tree.tag_configure('inflow', foreground=COLORS['gain'])
        self.sector_detail_tree.tag_configure('outflow', foreground=COLORS['loss'])
        self.sector_detail_tree.tag_configure('neutral', foreground=COLORS['text_primary'])
        
        self.sector_detail_tree.pack(fill=tk.BOTH, expand=True)
        
        # Initialize sector data storage and bind click handler
        self._sector_stocks = {}
        self._sector_detail_visible = False
        self.sector_flow_tree.bind('<<TreeviewSelect>>', self._on_sector_select)
    
    # ==================== DATA LOADING ====================
    
    def _load_sector_data(self):
        """Load rich sector data from TradingView and database."""
        def fetch():
            try:
                # Get all stocks with performance data
                all_stocks = self.collector.get_all_stocks()
                stocks_list = all_stocks.to_dict('records') if not all_stocks.empty else []
                
                # Get sector mapping
                sector_map = {}
                try:
                    results = self.db.conn.execute(
                        "SELECT symbol, sector FROM stocks WHERE sector IS NOT NULL AND sector != ''"
                    ).fetchall()
                    sector_map = {row[0]: row[1] for row in results}
                except:
                    pass
                
                # Build sector data
                sector_data = {}
                for s in stocks_list:
                    symbol = s.get('symbol', '')
                    sector = sector_map.get(symbol, 'Other')
                    
                    if sector not in sector_data:
                        sector_data[sector] = {
                            'stocks': [],
                            'gainers': 0,
                            'losers': 0,
                            'total_mcap': 0,
                        }
                    
                    chg_1d = s.get('change', 0) or 0
                    chg_1w = s.get('Perf.W', 0) or 0
                    chg_1m = s.get('Perf.1M', 0) or 0
                    mcap = s.get('market_cap_basic', 0) or 0
                    
                    # Ensure numeric values (handle NaN)
                    if not isinstance(chg_1d, (int, float)) or chg_1d != chg_1d:
                        chg_1d = 0.0
                    if not isinstance(chg_1w, (int, float)) or chg_1w != chg_1w:
                        chg_1w = 0.0
                    if not isinstance(chg_1m, (int, float)) or chg_1m != chg_1m:
                        chg_1m = 0.0
                    if not isinstance(mcap, (int, float)) or mcap != mcap:
                        mcap = 0.0
                    
                    sector_data[sector]['stocks'].append({
                        'symbol': symbol,
                        'price': s.get('close', 0) or 0,
                        'chg_1d': chg_1d,
                        'chg_1w': chg_1w,
                        'chg_1m': chg_1m,
                        'mcap': mcap,
                    })
                    
                    sector_data[sector]['total_mcap'] += mcap
                    if chg_1d > 0:
                        sector_data[sector]['gainers'] += 1
                    elif chg_1d < 0:
                        sector_data[sector]['losers'] += 1
                
                # Calculate sector averages and find leaders
                sector_rankings = []
                for sector, data in sector_data.items():
                    stocks = data['stocks']
                    if not stocks:
                        continue
                    
                    avg_1d = sum(s['chg_1d'] for s in stocks) / len(stocks)
                    avg_1w = sum(s['chg_1w'] for s in stocks) / len(stocks)
                    avg_1m = sum(s['chg_1m'] for s in stocks) / len(stocks)
                    
                    # Calculate combined net effect (weighted average)
                    net_score = (avg_1d * 0.2) + (avg_1w * 0.4) + (avg_1m * 0.4)
                    
                    # Find leader (best performer in sector)
                    leader = max(stocks, key=lambda x: x['chg_1d']) if stocks else None
                    
                    # Calculate weights
                    for s in stocks:
                        s['weight'] = (s['mcap'] / data['total_mcap'] * 100) if data['total_mcap'] > 0 else 0
                    
                    sector_rankings.append({
                        'sector': sector,
                        'avg_1d': avg_1d,
                        'avg_1w': avg_1w,
                        'avg_1m': avg_1m,
                        'net_score': net_score,
                        'total_mcap': data['total_mcap'],
                        'count': len(stocks),
                        'gainers': data['gainers'],
                        'losers': data['losers'],
                        'leader': leader,
                        'stocks': stocks,
                    })
                
                # Sort by net score
                sector_rankings.sort(key=lambda x: x['net_score'], reverse=True)
                
                # Put result in queue (thread-safe)
                self._update_queue.put(('sector', sector_rankings))
            except Exception as e:
                logger.error(f"Error loading sector data: {e}")
        
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
        
        # Poll for results from main thread
        self._poll_sector_queue()
    
    def _poll_sector_queue(self):
        """Poll the update queue for sector data."""
        try:
            result = self._update_queue.get_nowait()
            if result[0] == 'sector':
                self._update_sector_ui(result[1])
            elif result[0] == 'flow':
                self._update_flow_ui(result[1])
            elif result[0] == 'smart_money':
                self._update_smart_money_ui(result[1], result[2])
        except queue.Empty:
            # No result yet, check again in 100ms
            self.frame.after(100, self._poll_sector_queue)
    
    def _update_sector_ui(self, sector_rankings: List[Dict]):
        """Update rich sector rotation UI."""
        if not sector_rankings:
            return
        
        # ===== Update Rotation Phase =====
        phase_data = self.sector_analysis.detect_rotation_phase(self.all_stocks_data)
        
        self.phase_labels['phase'].config(text=phase_data.get('description', 'Unknown')[:30])
        leading = ', '.join(phase_data.get('leading', ['--'])[:2])
        lagging = ', '.join(phase_data.get('lagging', ['--'])[:2])
        self.phase_labels['leading'].config(text=leading[:20])
        self.phase_labels['lagging'].config(text=lagging[:20])
        
        # ===== Draw Rotation Clock =====
        self._draw_rotation_clock(phase_data)
        
        # ===== Update Momentum Matrix (Treeview) =====
        self._update_momentum_matrix(sector_rankings)
        
        # ===== Update Correlation Pairs (Treeview) =====
        self._update_correlation_pairs(sector_rankings)
        
        # ===== Update Sector Rankings Table =====
        for item in self.sector_tree.get_children():
            self.sector_tree.delete(item)
        
        for sr in sector_rankings:
            avg_1w = sr.get('avg_1w', 0)
            tag = 'gain' if avg_1w > 0 else 'loss' if avg_1w < 0 else ''
            
            self.sector_tree.insert('', 'end', iid=sr['sector'], values=(
                sr['sector'][:14],
                f"{sr['avg_1d']:+.1f}%",
                f"{sr['avg_1w']:+.1f}%",
                sr['count']
            ), tags=(tag,))
        
        # ===== Update RS Leaderboard =====
        self._update_rs_leaderboard(sector_rankings)
        
        # Store for component loading
        self._sector_data = {sr['sector']: sr for sr in sector_rankings}
    
    def _update_momentum_matrix(self, sector_rankings: List[Dict]):
        """Update momentum matrix treeview."""
        for item in self.momentum_tree.get_children():
            self.momentum_tree.delete(item)
        
        for sr in sector_rankings[:8]:  # Top 8 sectors
            chg_1d = sr.get('avg_1d', 0)
            chg_1w = sr.get('avg_1w', 0)
            chg_1m = sr.get('avg_1m', 0)
            
            # Determine tag based on overall trend
            if chg_1w >= 3:
                tag = 'hot'
            elif chg_1w > 0:
                tag = 'gain'
            elif chg_1w <= -3:
                tag = 'cold'
            elif chg_1w < 0:
                tag = 'loss'
            else:
                tag = ''
            
            self.momentum_tree.insert('', 'end', iid=sr['sector'], values=(
                sr['sector'][:12],
                f"{chg_1d:+.1f}%",
                f"{chg_1w:+.1f}%",
                f"{chg_1m:+.1f}%"
            ), tags=(tag,))
    
    def _update_correlation_pairs(self, sector_rankings: List[Dict]):
        """Update correlation pairs treeview with top correlated sector pairs."""
        for item in self.corr_tree.get_children():
            self.corr_tree.delete(item)
        
        if not sector_rankings or len(sector_rankings) < 2:
            return
        
        # Calculate correlation based on sector performance patterns
        # Compare 1W performance similarity between sectors
        pairs = []
        sectors = sector_rankings[:6]
        
        for i, s1 in enumerate(sectors):
            for s2 in sectors[i+1:]:
                # Use performance differentials to estimate correlation
                avg1 = s1.get('avg_1w', 0)
                avg2 = s2.get('avg_1w', 0)
                
                # Simple correlation proxy: same direction = positive
                if avg1 * avg2 > 0:  # Same direction
                    diff = abs(avg1 - avg2)
                    corr = max(0.1, 1 - diff / 10)  # Similar magnitude = higher correlation
                elif avg1 * avg2 < 0:  # Opposite direction
                    corr = -max(0.1, min(1, abs(avg1 - avg2) / 10))
                else:  # One is zero
                    corr = 0
                
                pairs.append({
                    'pair': f"{s1['sector'][:6]}-{s2['sector'][:6]}",
                    'correlation': corr,
                    's1': s1['sector'],
                    's2': s2['sector']
                })
        
        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Show top 6 pairs
        for p in pairs[:6]:
            corr = p['correlation']
            if corr > 0.1:
                direction = "↑↑ Together"
                tag = 'positive'
            elif corr < -0.1:
                direction = "↑↓ Inverse"
                tag = 'negative'
            else:
                direction = "→ Neutral"
                tag = ''
            
            self.corr_tree.insert('', 'end', values=(
                p['pair'],
                f"{corr:.2f}",
                direction
            ), tags=(tag,))
    
    def _draw_momentum_matrix(self, momentum_matrix: List[Dict]):
        """Draw sector momentum matrix with 1D/1W/1M bars per sector."""
        self.momentum_canvas.delete("all")
        
        width = self.momentum_canvas.winfo_width()
        height = self.momentum_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width, height = 400, 160
        
        if not momentum_matrix:
            return
        
        sector_abbrev = {
            'Financial Services': 'Finance',
            'Industrial Goods': 'Indust',
            'Consumer Goods': 'Consumer',
            'Oil & Gas': 'O&G',
            'Agriculture': 'Agric',
            'Healthcare': 'Health',
            'Natural Resources': 'NatRes',
            'Construction': 'Constr',
            'Real Estate': 'RealEst',
            'Services': 'Service',
            'Conglomerates': 'Conglm',
            'Insurance': 'Insur',
        }
        
        cols = 4
        rows = min(3, (len(momentum_matrix) + cols - 1) // cols)
        
        cell_w = width // cols
        cell_h = height // rows
        bar_height = 6
        bar_spacing = 10
        
        for i, mm in enumerate(momentum_matrix[:12]):
            row = i // cols
            col = i % cols
            
            x1 = col * cell_w + 4
            y1 = row * cell_h + 4
            x2 = x1 + cell_w - 8
            y2 = y1 + cell_h - 8
            
            # Draw cell background
            self.momentum_canvas.create_rectangle(x1, y1, x2, y2, fill='#1e1e1e', outline='#333')
            
            # Sector name
            name = sector_abbrev.get(mm['sector'], mm['sector'][:7])
            self.momentum_canvas.create_text(
                (x1 + x2) / 2, y1 + 10,
                text=name, fill="white", font=('Arial', 7)
            )
            
            # Draw 3 bars for 1D, 1W, 1M
            bar_start_y = y1 + 22
            bar_max_width = (cell_w - 16) / 2
            
            for j, (period, value) in enumerate([('1D', mm['chg_1d']), ('1W', mm['chg_1w']), ('1M', mm['chg_1m'])]):
                bar_y = bar_start_y + j * bar_spacing
                
                # Period label
                self.momentum_canvas.create_text(
                    x1 + 12, bar_y + 3,
                    text=period, fill="#888", font=('Arial', 6)
                )
                
                # Bar
                bar_x_start = x1 + 24 + bar_max_width
                bar_width = min(bar_max_width, abs(value) / 10 * bar_max_width)
                
                if value > 0:
                    color = '#2ecc71' if value > 3 else '#27ae60'
                    self.momentum_canvas.create_rectangle(
                        bar_x_start, bar_y, 
                        bar_x_start + bar_width, bar_y + bar_height,
                        fill=color, outline=''
                    )
                elif value < 0:
                    color = '#e74c3c' if value < -3 else '#c0392b'
                    self.momentum_canvas.create_rectangle(
                        bar_x_start - bar_width, bar_y,
                        bar_x_start, bar_y + bar_height,
                        fill=color, outline=''
                    )
                
                # Value label
                self.momentum_canvas.create_text(
                    x2 - 12, bar_y + 3,
                    text=f"{value:+.1f}", fill="white", font=('Arial', 6)
                )
    
    def _draw_rotation_clock(self, phase_data: Dict):
        """Draw visual rotation clock dial."""
        self.clock_canvas.delete("all")
        
        width = self.clock_canvas.winfo_width()
        height = self.clock_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width, height = 160, 120
        
        cx, cy = width // 2, height // 2
        radius = min(width, height) // 2 - 10
        
        # Draw quadrants
        phases = ['EARLY', 'MID', 'LATE', 'CONTRACTION']
        phase_colors = {
            'EARLY': '#27ae60',
            'MID': '#3498db',
            'LATE': '#f39c12',
            'CONTRACTION': '#e74c3c'
        }
        phase_icons = {
            'EARLY': '💹',
            'MID': '📈',
            'LATE': '⚠️',
            'CONTRACTION': '🛡️'
        }
        
        import math
        
        current_phase = phase_data.get('phase', 'UNKNOWN')
        
        for i, phase in enumerate(phases):
            start_angle = i * 90
            
            # Draw arc segment
            is_current = (phase == current_phase)
            color = phase_colors[phase] if is_current else '#333'
            
            # Draw pie slice
            self.clock_canvas.create_arc(
                cx - radius, cy - radius, cx + radius, cy + radius,
                start=start_angle, extent=90,
                fill=color, outline='#555', width=1
            )
            
            # Phase label
            label_angle = math.radians(start_angle + 45)
            lx = cx + (radius * 0.6) * math.cos(label_angle)
            ly = cy - (radius * 0.6) * math.sin(label_angle)
            
            self.clock_canvas.create_text(
                lx, ly,
                text=phase_icons.get(phase, '?'),
                font=('Arial', 10)
            )
        
        # Center circle
        self.clock_canvas.create_oval(
            cx - 12, cy - 12, cx + 12, cy + 12,
            fill='#1e1e1e', outline='#555'
        )
        
        # Confidence in center
        conf = phase_data.get('confidence', 0)
        self.clock_canvas.create_text(
            cx, cy,
            text=f"{conf}%", fill="white", font=('Arial', 7, 'bold')
        )
    
    def _draw_correlation_heatmap(self, correlations: Dict[str, Dict[str, float]], sector_rankings: List[Dict]):
        """Draw sector correlation heatmap."""
        self.correlation_canvas.delete("all")
        
        width = self.correlation_canvas.winfo_width()
        height = self.correlation_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width, height = 200, 180
        
        sectors = [sr['sector'] for sr in sector_rankings[:6]]  # Limit to 6 for space
        
        if not sectors or not correlations:
            self.correlation_canvas.create_text(
                width // 2, height // 2,
                text="Loading...", fill="#888"
            )
            return
        
        sector_abbrev = {
            'Financial Services': 'Fin',
            'Industrial Goods': 'Ind',
            'Consumer Goods': 'Con',
            'Oil & Gas': 'O&G',
            'Agriculture': 'Agr',
            'Healthcare': 'Hth',
            'Natural Resources': 'Nat',
            'Construction': 'Cns',
            'Real Estate': 'RE',
            'Services': 'Svc',
            'Conglomerates': 'Cgl',
            'Insurance': 'Ins',
        }
        
        n = len(sectors)
        cell_size = min((width - 30) // n, (height - 30) // n)
        offset_x = 25
        offset_y = 20
        
        # Draw grid
        for i, s1 in enumerate(sectors):
            for j, s2 in enumerate(sectors):
                corr = correlations.get(s1, {}).get(s2, 0)
                
                x1 = offset_x + j * cell_size
                y1 = offset_y + i * cell_size
                
                # Color based on correlation
                if corr > 0.5:
                    color = '#27ae60'
                elif corr > 0:
                    color = '#2ecc71'
                elif corr < -0.5:
                    color = '#c0392b'
                elif corr < 0:
                    color = '#e74c3c'
                else:
                    color = '#555'
                
                self.correlation_canvas.create_rectangle(
                    x1, y1, x1 + cell_size - 1, y1 + cell_size - 1,
                    fill=color, outline='#222'
                )
        
        # Row labels (left side)
        for i, sector in enumerate(sectors):
            abbrev = sector_abbrev.get(sector, sector[:3])
            y = offset_y + i * cell_size + cell_size // 2
            self.correlation_canvas.create_text(
                12, y,
                text=abbrev, fill="#888", font=('Arial', 6)
            )
        
        # Column labels (top)
        for j, sector in enumerate(sectors):
            abbrev = sector_abbrev.get(sector, sector[:3])
            x = offset_x + j * cell_size + cell_size // 2
            self.correlation_canvas.create_text(
                x, 8,
                text=abbrev, fill="#888", font=('Arial', 6)
            )
    
    def _update_rs_leaderboard(self, sector_rankings: List[Dict]):
        """Update the RS Leaderboard with stocks outperforming their sector."""
        for item in self.rs_tree.get_children():
            self.rs_tree.delete(item)
        
        # Collect all stocks with their RS vs sector
        rs_leaders = []
        for sr in sector_rankings:
            sector_avg = sr.get('avg_1w', 0)
            for stock in sr.get('stocks', []):
                stock_1w = stock.get('chg_1w', 0)
                vs_sector = stock_1w - sector_avg
                rs_leaders.append({
                    'symbol': stock.get('symbol', '?'),
                    'sector': sr['sector'],
                    'rs_score': stock_1w,
                    'vs_sector': vs_sector
                })
        
        # Sort by vs_sector (outperformance)
        rs_leaders.sort(key=lambda x: -x['vs_sector'])
        
        # Show top 10 outperformers
        for rs in rs_leaders[:10]:
            vs = rs['vs_sector']
            if vs >= 5:
                tag = 'strong'
            elif vs > 0:
                tag = 'outperform'
            else:
                tag = 'underperform'
            
            self.rs_tree.insert('', 'end', values=(
                rs['symbol'],
                rs['sector'][:10],
                f"{vs:+.1f}%"
            ), tags=(tag,))
    
    def _on_sector_select(self, event):
        """Handle sector selection from sector rankings."""
        selection = self.sector_tree.selection()
        if not selection:
            return
        
        sector = selection[0]
        if sector and hasattr(self, '_sector_data') and sector in self._sector_data:
            self._update_components_ui(self._sector_data[sector])
    
    def _on_momentum_select(self, event):
        """Handle sector selection from momentum matrix."""
        selection = self.momentum_tree.selection()
        if not selection:
            return
        
        sector = selection[0]
        if sector and hasattr(self, '_sector_data') and sector in self._sector_data:
            self._update_components_ui(self._sector_data[sector])
    
    def _load_sector_components(self, sector: str):
        """Load components for selected sector (legacy method)."""
        if hasattr(self, '_sector_data') and sector in self._sector_data:
            self._update_components_ui(self._sector_data[sector])
    
    def _update_components_ui(self, sector_data: Dict):
        """Update sector components table with rich data."""
        sector_name = sector_data.get('sector', 'Unknown')
        self.selected_sector_label.config(
            text=f"📊 {sector_name} ({sector_data.get('count', 0)} stocks)",
            foreground=COLORS['primary']
        )
        
        for item in self.component_tree.get_children():
            self.component_tree.delete(item)
        
        stocks = sector_data.get('stocks', [])
        # Sort by weight descending
        stocks_sorted = sorted(stocks, key=lambda x: x.get('weight', 0), reverse=True)
        
        for s in stocks_sorted:
            chg_1d = s.get('chg_1d', 0)
            chg_1w = s.get('chg_1w', 0)
            
            # Determine tag
            if chg_1d >= 5:
                tag = 'strong'
            elif chg_1d > 0:
                tag = 'gain'
            elif chg_1d < 0:
                tag = 'loss'
            else:
                tag = ''
            
            # Momentum indicator
            if chg_1w > 5:
                momentum = '🚀 Strong'
            elif chg_1w > 0:
                momentum = '↑ Up'
            elif chg_1w < -5:
                momentum = '💨 Weak'
            elif chg_1w < 0:
                momentum = '↓ Down'
            else:
                momentum = '→ Flat'
            
            self.component_tree.insert('', 'end', values=(
                s.get('symbol', ''),
                f"₦{s.get('price', 0):,.2f}",
                f"{chg_1d:+.1f}%",
                f"{chg_1w:+.1f}%",
                momentum
            ), tags=(tag,))
    
    def _load_flow_data(self):
        """Load rich flow analysis data from TradingView."""
        def fetch():
            try:
                all_stocks = self.collector.get_all_stocks()
                stocks_list = all_stocks.to_dict('records') if not all_stocks.empty else []
                
                # Get sector mapping
                sector_map = {}
                try:
                    results = self.db.conn.execute(
                        "SELECT symbol, sector FROM stocks WHERE sector IS NOT NULL AND sector != ''"
                    ).fetchall()
                    sector_map = {row[0]: row[1] for row in results}
                except:
                    pass
                
                stocks = []
                for s in stocks_list:
                    symbol = s.get('symbol', '')
                    chg_1d = s.get('change', 0) or 0
                    chg_1w = s.get('Perf.W', 0) or 0
                    chg_1m = s.get('Perf.1M', 0) or 0
                    chg_3m = s.get('Perf.3M', 0) or 0
                    volume = s.get('volume', 0) or 0
                    avg_vol = s.get('average_volume_10d_calc', 0) or 0
                    
                    # Calculate volume ratio
                    vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0
                    
                    # Calculate momentum score (weighted average of timeframes)
                    momentum_score = (chg_1d * 0.1 + chg_1w * 0.3 + chg_1m * 0.4 + chg_3m * 0.2)
                    
                    # Determine momentum label
                    if momentum_score > 10:
                        momentum = '🚀 Surging'
                    elif momentum_score > 3:
                        momentum = '📈 Strong'
                    elif momentum_score > 0:
                        momentum = '↗️ Up'
                    elif momentum_score > -3:
                        momentum = '↘️ Down'
                    elif momentum_score > -10:
                        momentum = '📉 Weak'
                    else:
                        momentum = '💨 Falling'
                    
                    stocks.append({
                        'symbol': symbol,
                        'sector': sector_map.get(symbol, 'Other'),
                        'price': s.get('close', 0) or 0,
                        'chg_1d': chg_1d,
                        'chg_1w': chg_1w,
                        'chg_1m': chg_1m,
                        'chg_3m': chg_3m,
                        'momentum': momentum,
                        'momentum_score': momentum_score,
                        'vol_ratio': vol_ratio,
                    })
                
                # Sort by 1W performance
                stocks.sort(key=lambda x: x['chg_1w'], reverse=True)
                
                # Put result in queue (thread-safe)
                self._update_queue.put(('flow', stocks))
            except Exception as e:
                logger.error(f"Error loading flow data: {e}")
        
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
        
        # Poll for results
        self._poll_sector_queue()
    
    def _update_flow_ui(self, stocks: List[Dict]):
        """Update rich flow monitor UI."""
        # ===== Update 5-tier flow classification cards =====
        strong_inflow = sum(1 for s in stocks if s['chg_1w'] >= 5)
        moderate_inflow = sum(1 for s in stocks if 0 < s['chg_1w'] < 5)
        neutral = sum(1 for s in stocks if s['chg_1w'] == 0)
        moderate_outflow = sum(1 for s in stocks if -5 < s['chg_1w'] < 0)
        strong_outflow = sum(1 for s in stocks if s['chg_1w'] <= -5)
        
        self.flow_cards['strong_inflow'].config(text=str(strong_inflow))
        self.flow_cards['moderate_inflow'].config(text=str(moderate_inflow))
        self.flow_cards['neutral'].config(text=str(neutral))
        self.flow_cards['moderate_outflow'].config(text=str(moderate_outflow))
        self.flow_cards['strong_outflow'].config(text=str(strong_outflow))
        
        # ===== Update Money Flow Pressure Gauge =====
        buying_count = strong_inflow + moderate_inflow
        selling_count = strong_outflow + moderate_outflow
        total = len(stocks) or 1
        
        buying_pct = (buying_count / total) * 100
        selling_pct = (selling_count / total) * 100
        net_flow = buying_pct - selling_pct
        
        self.pressure_labels['buying'].config(
            text=f"{buying_count} ({buying_pct:.0f}%)",
            foreground=COLORS['gain']
        )
        self.pressure_labels['selling'].config(
            text=f"{selling_count} ({selling_pct:.0f}%)",
            foreground=COLORS['loss']
        )
        self.pressure_labels['net'].config(
            text=f"{net_flow:+.1f}%",
            foreground=COLORS['gain'] if net_flow > 0 else (COLORS['loss'] if net_flow < 0 else COLORS['text_primary'])
        )
        
        # Additional metrics
        breadth = ((buying_count - selling_count) / total) * 100
        avg_chg_1d = sum(s['chg_1d'] for s in stocks) / total if stocks else 0
        high_vol_count = sum(1 for s in stocks if s.get('vol_ratio', 1) >= 2.0)
        
        self.pressure_labels['breadth'].config(
            text=f"{breadth:+.0f}%",
            foreground=COLORS['gain'] if breadth > 0 else COLORS['loss']
        )
        self.pressure_labels['avg_chg'].config(
            text=f"{avg_chg_1d:+.2f}%",
            foreground=COLORS['gain'] if avg_chg_1d > 0 else COLORS['loss']
        )
        self.pressure_labels['high_vol'].config(
            text=str(high_vol_count),
            foreground='#f39c12' if high_vol_count > 5 else COLORS['text_primary']
        )
        
        # Top gainer and loser
        if stocks:
            top_gainer = max(stocks, key=lambda x: x['chg_1w'])
            top_loser = min(stocks, key=lambda x: x['chg_1w'])
            self.pressure_labels['top_gainer'].config(
                text=f"{top_gainer['symbol']} {top_gainer['chg_1w']:+.1f}%"
            )
            self.pressure_labels['top_loser'].config(
                text=f"{top_loser['symbol']} {top_loser['chg_1w']:+.1f}%"
            )
        
        # Draw pressure gauge
        self._draw_pressure_gauge(buying_pct, selling_pct)
        
        # ===== Update Duration Classification =====
        # Durable: positive 1D, 1W, and 1M (sustained trend)
        # Short-term: positive 1D and 1W, but negative 1M (recent momentum)
        # One-off: positive 1D only, negative 1W (single day spike)
        
        durable = []
        shortterm = []
        oneoff = []
        
        for s in stocks:
            if s['chg_1d'] > 0 and s['chg_1w'] > 0 and s['chg_1m'] > 0:
                durable.append(s)
            elif s['chg_1d'] > 0 and s['chg_1w'] > 0:
                shortterm.append(s)
            elif s['chg_1d'] > 2 and s['chg_1w'] < 0:
                oneoff.append(s)
        
        # Store for click handler
        self._duration_data = {
            'durable': sorted(durable, key=lambda x: x['chg_1m'], reverse=True),
            'shortterm': sorted(shortterm, key=lambda x: x['chg_1w'], reverse=True),
            'oneoff': sorted(oneoff, key=lambda x: x['chg_1d'], reverse=True),
        }
        
        for item in self.duration_tree.get_children():
            self.duration_tree.delete(item)
        
        self.duration_tree.insert('', 'end', iid='durable', values=(
            '🏆 Durable',
            len(durable),
            'Positive 1D + 1W + 1M (sustained uptrend)'
        ), tags=('durable',))
        
        self.duration_tree.insert('', 'end', iid='shortterm', values=(
            '⚡ Short-term',
            len(shortterm),
            'Positive 1D + 1W only (recent momentum)'
        ), tags=('shortterm',))
        
        self.duration_tree.insert('', 'end', iid='oneoff', values=(
            '💥 One-off',
            len(oneoff),
            'Spike today, negative week (potential reversal)'
        ), tags=('oneoff',))
        
        # ===== Update Top Inflows Panel =====
        top_inflows = sorted(stocks, key=lambda x: x['chg_1w'], reverse=True)[:5]
        for i, stock in enumerate(top_inflows):
            if i < len(self.inflow_labels):
                sym_lbl, d1_lbl, w1_lbl, m1_lbl = self.inflow_labels[i]
                sym_lbl.config(text=stock['symbol'])
                d1_lbl.config(text=f"{stock['chg_1d']:+.1f}%",
                             foreground=COLORS['gain'] if stock['chg_1d'] > 0 else COLORS['loss'])
                w1_lbl.config(text=f"{stock['chg_1w']:+.1f}%")
                m1_lbl.config(text=f"{stock['chg_1m']:+.1f}%",
                             foreground=COLORS['gain'] if stock['chg_1m'] > 0 else COLORS['loss'])
        
        # ===== Update Top Outflows Panel =====
        top_outflows = sorted(stocks, key=lambda x: x['chg_1w'])[:5]
        for i, stock in enumerate(top_outflows):
            if i < len(self.outflow_labels):
                sym_lbl, d1_lbl, w1_lbl, m1_lbl = self.outflow_labels[i]
                sym_lbl.config(text=stock['symbol'])
                d1_lbl.config(text=f"{stock['chg_1d']:+.1f}%",
                             foreground=COLORS['gain'] if stock['chg_1d'] > 0 else COLORS['loss'])
                w1_lbl.config(text=f"{stock['chg_1w']:+.1f}%")
                m1_lbl.config(text=f"{stock['chg_1m']:+.1f}%",
                             foreground=COLORS['gain'] if stock['chg_1m'] > 0 else COLORS['loss'])
        
        # ===== Update Sector Flow Summary =====
        sector_flows = {}
        for s in stocks:
            sector = s['sector']
            if sector not in sector_flows:
                sector_flows[sector] = {'changes': [], 'inflows': 0, 'outflows': 0}
            sector_flows[sector]['changes'].append(s['chg_1w'])
            if s['chg_1w'] > 0:
                sector_flows[sector]['inflows'] += 1
            elif s['chg_1w'] < 0:
                sector_flows[sector]['outflows'] += 1
        
        # Update sector flow tree
        for item in self.sector_flow_tree.get_children():
            self.sector_flow_tree.delete(item)
        
        sector_rankings = []
        for sector, data in sector_flows.items():
            avg_1w = sum(data['changes']) / len(data['changes']) if data['changes'] else 0
            net = data['inflows'] - data['outflows']
            sector_rankings.append({
                'sector': sector,
                'avg_1w': avg_1w,
                'inflows': data['inflows'],
                'outflows': data['outflows'],
                'net': net
            })
        
        sector_rankings.sort(key=lambda x: x['avg_1w'], reverse=True)
        
        for sr in sector_rankings[:8]:
            tag = 'gain' if sr['avg_1w'] > 0 else 'loss' if sr['avg_1w'] < 0 else ''
            net_str = f"+{sr['net']}" if sr['net'] > 0 else str(sr['net'])
            
            self.sector_flow_tree.insert('', 'end', values=(
                sr['sector'][:15],
                f"{sr['avg_1w']:+.1f}%",
                sr['inflows'],
                sr['outflows'],
                net_str
            ), tags=(tag,))
        
        # ===== Draw Sector Flow Heatmap =====
        self._draw_sector_flow_heatmap(sector_rankings)
        
        # ===== Store stocks by sector for click handler =====
        self._sector_stocks = {}
        for s in stocks:
            sector = s['sector']
            if sector not in self._sector_stocks:
                self._sector_stocks[sector] = []
            self._sector_stocks[sector].append(s)
        
        # Sort each sector by 1W performance
        for sector in self._sector_stocks:
            self._sector_stocks[sector].sort(key=lambda x: x['chg_1w'], reverse=True)
        
        # Re-insert with iid for selection
        for item in self.sector_flow_tree.get_children():
            self.sector_flow_tree.delete(item)
        
        for sr in sector_rankings[:8]:
            tag = 'gain' if sr['avg_1w'] > 0 else 'loss' if sr['avg_1w'] < 0 else ''
            net_str = f"+{sr['net']}" if sr['net'] > 0 else str(sr['net'])
            
            self.sector_flow_tree.insert('', 'end', iid=sr['sector'], values=(
                sr['sector'][:15],
                f"{sr['avg_1w']:+.1f}%",
                sr['inflows'],
                sr['outflows'],
                net_str
            ), tags=(tag,))
    
    def _on_sector_select(self, event):
        """Handle sector selection to show securities in that sector."""
        selection = self.sector_flow_tree.selection()
        if not selection:
            return
        
        # Get the sector from iid
        sector = selection[0]
        
        # Get stocks for this sector
        stocks = self._sector_stocks.get(sector, [])
        
        if not stocks:
            return
        
        # Show the detail container if hidden
        if not self._sector_detail_visible:
            self.sector_detail_container.pack(fill=tk.BOTH, expand=True)
            self._sector_detail_visible = True
        
        # Calculate stats
        avg_1w = sum(s['chg_1w'] for s in stocks) / len(stocks) if stocks else 0
        inflows = sum(1 for s in stocks if s['chg_1w'] > 0)
        outflows = sum(1 for s in stocks if s['chg_1w'] < 0)
        
        # Update header labels
        self.sector_detail_labels['name'].config(text=sector, foreground=COLORS['primary'])
        self.sector_detail_labels['avg'].config(
            text=f"{avg_1w:+.1f}%",
            foreground=COLORS['gain'] if avg_1w > 0 else COLORS['loss']
        )
        self.sector_detail_labels['inflows'].config(text=str(inflows), foreground=COLORS['gain'])
        self.sector_detail_labels['outflows'].config(text=str(outflows), foreground=COLORS['loss'])
        
        # Update frame title
        self.sector_detail_frame.config(text=f"📊 {sector} Securities ({len(stocks)} stocks)")
        
        # Clear and populate detail tree
        for item in self.sector_detail_tree.get_children():
            self.sector_detail_tree.delete(item)
        
        for stock in stocks[:12]:  # Show top 12
            chg_1w = stock.get('chg_1w', 0)
            
            # Determine flow label and tag
            if chg_1w >= 5:
                flow = '🚀 Strong In'
                tag = 'inflow'
            elif chg_1w > 0:
                flow = '📈 Inflow'
                tag = 'inflow'
            elif chg_1w <= -5:
                flow = '💨 Strong Out'
                tag = 'outflow'
            elif chg_1w < 0:
                flow = '📉 Outflow'
                tag = 'outflow'
            else:
                flow = '➡️ Neutral'
                tag = 'neutral'
            
            self.sector_detail_tree.insert('', 'end', values=(
                stock['symbol'],
                flow,
                f"{stock['chg_1d']:+.1f}%",
                f"{chg_1w:+.1f}%",
                f"{stock['chg_1m']:+.1f}%"
            ), tags=(tag,))
    
    def _on_duration_select(self, event):
        """Handle duration classification selection to show securities."""
        selection = self.duration_tree.selection()
        if not selection:
            return
        
        # Get the category from iid
        category = selection[0]  # 'durable', 'shortterm', or 'oneoff'
        
        # Get stocks for this category
        stocks = self._duration_data.get(category, [])
        
        # Clear existing items
        for item in self.duration_detail_tree.get_children():
            self.duration_detail_tree.delete(item)
        
        # Update frame title
        category_names = {
            'durable': '🏆 Durable Trends',
            'shortterm': '⚡ Short-term Momentum',
            'oneoff': '💥 One-off Spikes'
        }
        self.duration_detail_frame.config(text=f"📋 {category_names.get(category, 'Securities')}")
        
        # Populate with stocks
        for stock in stocks[:10]:  # Show top 10
            chg_1d = stock.get('chg_1d', 0)
            tag = 'gain' if chg_1d > 0 else 'loss'
            
            self.duration_detail_tree.insert('', 'end', values=(
                stock['symbol'],
                f"{chg_1d:+.1f}%",
                f"{stock.get('chg_1w', 0):+.1f}%",
                f"{stock.get('chg_1m', 0):+.1f}%"
            ), tags=(tag,))
    
    def _draw_pressure_gauge(self, buying_pct: float, selling_pct: float):
        """Draw the money flow pressure gauge."""
        self.pressure_canvas.delete("all")
        
        w = self.pressure_canvas.winfo_width()
        h = self.pressure_canvas.winfo_height()
        if w <= 1:
            w, h = 300, 60
        
        bar_height = 25
        bar_y = h // 2 + 8  # Shift down to make room for text above
        margin = 10
        bar_width = w - 2 * margin
        
        # Background bar
        self.pressure_canvas.create_rectangle(
            margin, bar_y - bar_height//2,
            margin + bar_width, bar_y + bar_height//2,
            fill='#333', outline='#555'
        )
        
        # Calculate split point (neutral is 50%)
        neutral_x = margin + bar_width // 2
        
        # Buying pressure (green bar from center to right)
        if buying_pct > 0:
            buying_width = int((buying_pct / 100) * (bar_width / 2))
            self.pressure_canvas.create_rectangle(
                neutral_x, bar_y - bar_height//2,
                neutral_x + buying_width, bar_y + bar_height//2,
                fill='#27ae60', outline=''
            )
        
        # Selling pressure (red bar from center to left)
        if selling_pct > 0:
            selling_width = int((selling_pct / 100) * (bar_width / 2))
            self.pressure_canvas.create_rectangle(
                neutral_x - selling_width, bar_y - bar_height//2,
                neutral_x, bar_y + bar_height//2,
                fill='#e74c3c', outline=''
            )
        
        # Center line
        self.pressure_canvas.create_line(
            neutral_x, bar_y - bar_height//2 - 5,
            neutral_x, bar_y + bar_height//2 + 5,
            fill='white', width=2
        )
        
        # Labels
        self.pressure_canvas.create_text(
            margin + 5, bar_y,
            text="SELL", fill="#e74c3c", font=('Arial', 9, 'bold'), anchor='w'
        )
        self.pressure_canvas.create_text(
            margin + bar_width - 5, bar_y,
            text="BUY", fill="#27ae60", font=('Arial', 9, 'bold'), anchor='e'
        )
        
        # Net indicator arrow
        net_flow = buying_pct - selling_pct
        if abs(net_flow) > 5:
            arrow = "→" if net_flow > 0 else "←"
            color = "#27ae60" if net_flow > 0 else "#e74c3c"
            self.pressure_canvas.create_text(
                neutral_x, bar_y - bar_height//2 - 10,
                text=f"{arrow} Net: {net_flow:+.0f}%", fill=color, font=('Arial', 8, 'bold')
            )
    
    def _draw_sector_flow_heatmap(self, sector_rankings: List[Dict]):
        """Draw sector flow heatmap visualization."""
        self.sector_flow_canvas.delete("all")
        
        width = self.sector_flow_canvas.winfo_width()
        height = self.sector_flow_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width, height = 350, 160
        
        if not sector_rankings:
            return
        
        # Abbreviations for sector names
        sector_abbrev = {
            'Financial Services': 'Finance',
            'Industrial Goods': 'Industrial',
            'Consumer Goods': 'Consumer',
            'Oil & Gas': 'Oil&Gas',
            'Agriculture': 'Agric',
            'Healthcare': 'Health',
            'Natural Resources': 'NatRes',
            'Construction': 'Constr',
            'Real Estate': 'RealEst',
            'Services': 'Service',
            'Conglomerates': 'Conglom',
            'Insurance': 'Insur',
        }
        
        # Grid layout
        cols = 3
        rows = min(3, (len(sector_rankings) + cols - 1) // cols)
        
        cell_w = width // cols
        cell_h = height // rows
        
        for i, sr in enumerate(sector_rankings[:9]):
            row = i // cols
            col = i % cols
            
            x1 = col * cell_w + 2
            y1 = row * cell_h + 2
            x2 = x1 + cell_w - 4
            y2 = y1 + cell_h - 4
            
            # Color based on avg change
            avg = sr['avg_1w']
            if avg > 3:
                color = '#27ae60'  # Strong green
            elif avg > 0:
                color = '#2ecc71'  # Green
            elif avg < -3:
                color = '#c0392b'  # Strong red
            elif avg < 0:
                color = '#e74c3c'  # Red
            else:
                color = COLORS['text_muted']
            
            # Draw cell
            self.sector_flow_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='#333')
            
            # Sector name - use abbreviation
            full_name = sr['sector']
            name = sector_abbrev.get(full_name, full_name[:8])
            self.sector_flow_canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2 - 8,
                text=name, fill="white", font=('Arial', 8)
            )
            
            # Change value
            self.sector_flow_canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 10,
                text=f"{avg:+.1f}%", fill="white", font=('Arial', 11, 'bold')
            )
    
    # ==================== SMART MONEY ====================
    
    def _build_smart_money_ui(self):
        """Build the Pandora Box Smart Money detector UI."""
        # Header
        header = ttk.Frame(self.smart_money_frame)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            header,
            text="🕵️ Smart Money Detector",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        ttk.Label(
            header,
            text="Unusual volume, accumulation/distribution, and anomaly detection",
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        ).pack(side=tk.LEFT, padx=20)
        
        if TTKBOOTSTRAP_AVAILABLE:
            refresh_btn = ttk_bs.Button(
                header,
                text="↻ Scan Market",
                bootstyle="danger-outline",
                command=self._load_smart_money_data
            )
        else:
            refresh_btn = ttk.Button(header, text="↻ Scan Market", command=self._load_smart_money_data)
        refresh_btn.pack(side=tk.RIGHT)
        
        # ===== TOP ROW: Market Regime + Alerts =====
        top_row = ttk.Frame(self.smart_money_frame)
        top_row.pack(fill=tk.X, padx=10, pady=5)
        
        # Market Regime Indicator
        regime_frame = ttk.LabelFrame(top_row, text="📊 Market Regime", padding=10)
        regime_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Health Score Gauge
        self.regime_canvas = tk.Canvas(
            regime_frame,
            bg=COLORS['bg_dark'],
            height=80,
            width=200,
            highlightthickness=0
        )
        self.regime_canvas.pack(fill=tk.X)
        
        # Regime labels
        self.regime_labels = {}
        
        lbl_frame = ttk.Frame(regime_frame)
        lbl_frame.pack(fill=tk.X, pady=5)
        
        for key, title in [('health', 'Health:'), ('regime', 'Regime:'), ('trend', 'Trend:'), ('risk', 'Signal:')]:
            row = ttk.Frame(lbl_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=title, width=8, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="--", font=get_font('small'),
                           foreground=COLORS['primary'])
            lbl.pack(side=tk.LEFT)
            self.regime_labels[key] = lbl
        
        # Alerts Panel
        alerts_frame = ttk.LabelFrame(top_row, text="🚨 Anomaly Alerts", padding=5)
        alerts_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Alert list
        self.alerts_list = tk.Listbox(
            alerts_frame,
            height=6,
            bg=COLORS['bg_dark'],
            fg=COLORS['text_primary'],
            selectbackground=COLORS['primary'],
            font=('Consolas', 9)
        )
        self.alerts_list.pack(fill=tk.BOTH, expand=True)
        
        # ===== MIDDLE ROW: Unusual Volume + Accumulation/Distribution =====
        middle_row = ttk.Frame(self.smart_money_frame)
        middle_row.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Unusual Volume Scanner
        vol_frame = ttk.LabelFrame(middle_row, text="🔥 Unusual Volume (2x+ avg)", padding=5)
        vol_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        vol_cols = ('symbol', 'price', 'chg', 'vol_ratio', 'signal')
        self.unusual_vol_tree = ttk.Treeview(vol_frame, columns=vol_cols, show='headings', height=8)
        
        self.unusual_vol_tree.heading('symbol', text='Symbol')
        self.unusual_vol_tree.heading('price', text='Price')
        self.unusual_vol_tree.heading('chg', text='Chg%')
        self.unusual_vol_tree.heading('vol_ratio', text='Vol')
        self.unusual_vol_tree.heading('signal', text='Signal')
        
        self.unusual_vol_tree.column('symbol', width=70, anchor='w')
        self.unusual_vol_tree.column('price', width=70, anchor='e')
        self.unusual_vol_tree.column('chg', width=50, anchor='e')
        self.unusual_vol_tree.column('vol_ratio', width=45, anchor='e')
        self.unusual_vol_tree.column('signal', width=50, anchor='center')
        
        self.unusual_vol_tree.tag_configure('buy', foreground='#00ff00')
        self.unusual_vol_tree.tag_configure('sell', foreground='#ff4444')
        self.unusual_vol_tree.tag_configure('extreme', foreground='#ffff00')
        
        self.unusual_vol_tree.pack(fill=tk.BOTH, expand=True)
        
        # Accumulation/Distribution Panel
        ad_frame = ttk.Frame(middle_row)
        ad_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Accumulation (top)
        accum_frame = ttk.LabelFrame(ad_frame, text="📈 Accumulation (Smart Buying)", padding=3)
        accum_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 3))
        
        accum_cols = ('symbol', 'price', 'chg_1d', 'chg_1w', 'vol', 'score')
        self.accum_tree = ttk.Treeview(accum_frame, columns=accum_cols, show='headings', height=5)
        
        self.accum_tree.heading('symbol', text='Symbol')
        self.accum_tree.heading('price', text='Price')
        self.accum_tree.heading('chg_1d', text='1D%')
        self.accum_tree.heading('chg_1w', text='1W%')
        self.accum_tree.heading('vol', text='Vol')
        self.accum_tree.heading('score', text='Score')
        
        self.accum_tree.column('symbol', width=65, anchor='w')
        self.accum_tree.column('price', width=65, anchor='e')
        self.accum_tree.column('chg_1d', width=50, anchor='e')
        self.accum_tree.column('chg_1w', width=50, anchor='e')
        self.accum_tree.column('vol', width=45, anchor='e')
        self.accum_tree.column('score', width=45, anchor='center')
        
        self.accum_tree.tag_configure('strong', foreground='#00ff00')
        self.accum_tree.tag_configure('moderate', foreground=COLORS['gain'])
        self.accum_tree.tag_configure('weak', foreground=COLORS['text_primary'])
        
        self.accum_tree.pack(fill=tk.BOTH, expand=True)
        
        # Distribution (bottom)
        distrib_frame = ttk.LabelFrame(ad_frame, text="📉 Distribution (Smart Selling)", padding=3)
        distrib_frame.pack(fill=tk.BOTH, expand=True, pady=(3, 0))
        
        distrib_cols = ('symbol', 'price', 'chg_1d', 'chg_1w', 'vol', 'score')
        self.distrib_tree = ttk.Treeview(distrib_frame, columns=distrib_cols, show='headings', height=5)
        
        self.distrib_tree.heading('symbol', text='Symbol')
        self.distrib_tree.heading('price', text='Price')
        self.distrib_tree.heading('chg_1d', text='1D%')
        self.distrib_tree.heading('chg_1w', text='1W%')
        self.distrib_tree.heading('vol', text='Vol')
        self.distrib_tree.heading('score', text='Score')
        
        self.distrib_tree.column('symbol', width=65, anchor='w')
        self.distrib_tree.column('price', width=65, anchor='e')
        self.distrib_tree.column('chg_1d', width=50, anchor='e')
        self.distrib_tree.column('chg_1w', width=50, anchor='e')
        self.distrib_tree.column('vol', width=45, anchor='e')
        self.distrib_tree.column('score', width=45, anchor='center')
        
        self.distrib_tree.tag_configure('strong', foreground='#ff4444')
        self.distrib_tree.tag_configure('moderate', foreground=COLORS['loss'])
        self.distrib_tree.tag_configure('weak', foreground=COLORS['text_primary'])
        
        self.distrib_tree.pack(fill=tk.BOTH, expand=True)
        
        # ===== BOTTOM: Breakouts =====
        bottom_row = ttk.Frame(self.smart_money_frame)
        bottom_row.pack(fill=tk.X, padx=10, pady=5)
        
        # Bullish Breakouts
        bull_frame = ttk.LabelFrame(bottom_row, text="🚀 Bullish Breakouts", padding=5)
        bull_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.breakout_up_labels = []
        for i in range(3):
            lbl = ttk.Label(bull_frame, text="--", font=get_font('small'), foreground=COLORS['gain'])
            lbl.pack(anchor='w', pady=1)
            self.breakout_up_labels.append(lbl)
        
        # Bearish Breakdowns
        bear_frame = ttk.LabelFrame(bottom_row, text="💨 Bearish Breakdowns", padding=5)
        bear_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.breakout_down_labels = []
        for i in range(3):
            lbl = ttk.Label(bear_frame, text="--", font=get_font('small'), foreground=COLORS['loss'])
            lbl.pack(anchor='w', pady=1)
            self.breakout_down_labels.append(lbl)
        
        # ===== BOTTOM ROW 2: Stealth Accumulation + Block Trades =====
        bottom_row2 = ttk.Frame(self.smart_money_frame)
        bottom_row2.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Stealth Accumulation
        stealth_frame = ttk.LabelFrame(bottom_row2, text="🥷 Stealth Accumulation (Quiet Buying)", padding=3)
        stealth_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        stealth_cols = ('symbol', 'price', 'chg_1w', 'volatility', 'score', 'signal')
        self.stealth_tree = ttk.Treeview(stealth_frame, columns=stealth_cols, show='headings', height=4)
        
        self.stealth_tree.heading('symbol', text='Symbol')
        self.stealth_tree.heading('price', text='Price')
        self.stealth_tree.heading('chg_1w', text='1W%')
        self.stealth_tree.heading('volatility', text='Volatility')
        self.stealth_tree.heading('score', text='Score')
        self.stealth_tree.heading('signal', text='Signal')
        
        self.stealth_tree.column('symbol', width=60, anchor='w')
        self.stealth_tree.column('price', width=60, anchor='e')
        self.stealth_tree.column('chg_1w', width=45, anchor='e')
        self.stealth_tree.column('volatility', width=50, anchor='e')
        self.stealth_tree.column('score', width=40, anchor='center')
        self.stealth_tree.column('signal', width=75, anchor='center')
        
        self.stealth_tree.tag_configure('accumulating', foreground='#00ff00')
        self.stealth_tree.tag_configure('watching', foreground=COLORS['text_primary'])
        
        self.stealth_tree.pack(fill=tk.BOTH, expand=True)
        
        # Block Trades
        block_frame = ttk.LabelFrame(bottom_row2, text="🏛️ Block Trades (Institutional)", padding=3)
        block_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        block_cols = ('symbol', 'price', 'chg', 'vol', 'direction', 'size')
        self.block_tree = ttk.Treeview(block_frame, columns=block_cols, show='headings', height=4)
        
        self.block_tree.heading('symbol', text='Symbol')
        self.block_tree.heading('price', text='Price')
        self.block_tree.heading('chg', text='Chg%')
        self.block_tree.heading('vol', text='Vol')
        self.block_tree.heading('direction', text='Dir')
        self.block_tree.heading('size', text='Size')
        
        self.block_tree.column('symbol', width=60, anchor='w')
        self.block_tree.column('price', width=60, anchor='e')
        self.block_tree.column('chg', width=45, anchor='e')
        self.block_tree.column('vol', width=40, anchor='e')
        self.block_tree.column('direction', width=40, anchor='center')
        self.block_tree.column('size', width=55, anchor='center')
        
        self.block_tree.tag_configure('buy', foreground=COLORS['gain'])
        self.block_tree.tag_configure('sell', foreground=COLORS['loss'])
        self.block_tree.tag_configure('large', foreground='#ffff00')
        
        self.block_tree.pack(fill=tk.BOTH, expand=True)
    
    # ==================== AI SYNTHESIS ====================
    
    def _build_ai_synthesis_ui(self):
        """Build the AI Synthesis sub-tab with comprehensive market intelligence."""
        # Main scrollable frame
        canvas = tk.Canvas(self.synthesis_frame, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.synthesis_frame, orient="vertical", command=canvas.yview)
        self.synthesis_scrollable = ttk.Frame(canvas)
        
        self.synthesis_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.synthesis_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main = self.synthesis_scrollable
        
        # Header
        header = ttk.Frame(main)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            header,
            text="🤖 AI Market Intelligence Synthesis",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        ttk.Label(
            header,
            text="Comprehensive AI-powered market analysis",
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        ).pack(side=tk.LEFT, padx=20)
        
        if TTKBOOTSTRAP_AVAILABLE:
            refresh_btn = ttk_bs.Button(
                header,
                text="↻ Refresh Analysis",
                bootstyle="primary-outline",
                command=self._update_ai_synthesis
            )
        else:
            refresh_btn = ttk.Button(header, text="↻ Refresh Analysis", command=self._update_ai_synthesis)
        refresh_btn.pack(side=tk.RIGHT)
        
        # ========== ROW 1: MARKET OVERVIEW CARDS ==========
        overview_frame = ttk.LabelFrame(main, text="📊 Market Overview")
        overview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        overview_cards = ttk.Frame(overview_frame)
        overview_cards.pack(fill=tk.X, padx=5, pady=5)
        
        self.synthesis_overview = {}
        
        # Card 1: Market Health
        card1 = ttk.LabelFrame(overview_cards, text="🏥 Health", padding=8)
        card1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.synthesis_overview['health_value'] = ttk.Label(card1, text="--", font=get_font('subheading'),
                                                             foreground=COLORS['primary'])
        self.synthesis_overview['health_value'].pack()
        self.synthesis_overview['health_bar'] = tk.Canvas(card1, width=100, height=8, 
                                                           bg=COLORS['bg_medium'], highlightthickness=0)
        self.synthesis_overview['health_bar'].pack(pady=2)
        self.synthesis_overview['health_driver'] = ttk.Label(card1, text="--", font=get_font('small'),
                                                              foreground=COLORS['text_muted'])
        self.synthesis_overview['health_driver'].pack()
        
        # Card 2: Market Regime
        card2 = ttk.LabelFrame(overview_cards, text="📈 Regime", padding=8)
        card2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.synthesis_overview['regime_value'] = ttk.Label(card2, text="--", font=get_font('subheading'),
                                                             foreground=COLORS['primary'])
        self.synthesis_overview['regime_value'].pack()
        self.synthesis_overview['regime_icon'] = ttk.Label(card2, text="⚪", font=('', 16))
        self.synthesis_overview['regime_icon'].pack()
        self.synthesis_overview['regime_driver'] = ttk.Label(card2, text="--", font=get_font('small'),
                                                              foreground=COLORS['text_muted'])
        self.synthesis_overview['regime_driver'].pack()
        
        # Card 3: Breadth
        card3 = ttk.LabelFrame(overview_cards, text="📊 Breadth", padding=8)
        card3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.synthesis_overview['breadth_value'] = ttk.Label(card3, text="--", font=get_font('subheading'),
                                                              foreground=COLORS['primary'])
        self.synthesis_overview['breadth_value'].pack()
        self.synthesis_overview['breadth_detail'] = ttk.Label(card3, text="↑-- ↓--", font=get_font('body'))
        self.synthesis_overview['breadth_detail'].pack()
        self.synthesis_overview['breadth_driver'] = ttk.Label(card3, text="--", font=get_font('small'),
                                                               foreground=COLORS['text_muted'])
        self.synthesis_overview['breadth_driver'].pack()
        
        # Card 4: Trend
        card4 = ttk.LabelFrame(overview_cards, text="🧭 Trend", padding=8)
        card4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.synthesis_overview['trend_value'] = ttk.Label(card4, text="--", font=get_font('subheading'),
                                                            foreground=COLORS['primary'])
        self.synthesis_overview['trend_value'].pack()
        self.synthesis_overview['trend_icon'] = ttk.Label(card4, text="➡️", font=('', 16))
        self.synthesis_overview['trend_icon'].pack()
        self.synthesis_overview['trend_driver'] = ttk.Label(card4, text="--", font=get_font('small'),
                                                             foreground=COLORS['text_muted'])
        self.synthesis_overview['trend_driver'].pack()
        
        # ========== ROW 2: SECTOR + STOCK INTELLIGENCE ==========
        intel_frame = ttk.Frame(main)
        intel_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # LEFT: Sector Intelligence
        sector_intel = ttk.LabelFrame(intel_frame, text="🔄 Sector Intelligence", padding=5)
        sector_intel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.synthesis_sector = {}
        
        # Leading Sectors
        lead_row = ttk.Frame(sector_intel)
        lead_row.pack(fill=tk.X, pady=2)
        ttk.Label(lead_row, text="📈 Leading:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.synthesis_sector['leading'] = ttk.Label(lead_row, text="--", font=get_font('small'),
                                                      foreground=COLORS['gain'])
        self.synthesis_sector['leading'].pack(side=tk.LEFT, padx=5)
        
        # Lagging Sectors
        lag_row = ttk.Frame(sector_intel)
        lag_row.pack(fill=tk.X, pady=2)
        ttk.Label(lag_row, text="📉 Lagging:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.synthesis_sector['lagging'] = ttk.Label(lag_row, text="--", font=get_font('small'),
                                                      foreground=COLORS['loss'])
        self.synthesis_sector['lagging'].pack(side=tk.LEFT, padx=5)
        
        # Rotation Phase
        phase_row = ttk.Frame(sector_intel)
        phase_row.pack(fill=tk.X, pady=2)
        ttk.Label(phase_row, text="🔄 Phase:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.synthesis_sector['phase'] = ttk.Label(phase_row, text="--", font=get_font('small'),
                                                    foreground=COLORS['primary'])
        self.synthesis_sector['phase'].pack(side=tk.LEFT, padx=5)
        
        # Sector Flow Pressure
        sflow_row = ttk.Frame(sector_intel)
        sflow_row.pack(fill=tk.X, pady=2)
        ttk.Label(sflow_row, text="💧 Flow:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.synthesis_sector['flow'] = ttk.Label(sflow_row, text="--", font=get_font('small'))
        self.synthesis_sector['flow'].pack(side=tk.LEFT, padx=5)
        
        # RIGHT: Stock Intelligence
        stock_intel = ttk.LabelFrame(intel_frame, text="📊 Stock Intelligence", padding=5)
        stock_intel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.synthesis_stock = {}
        
        # Top Gainer
        gain_row = ttk.Frame(stock_intel)
        gain_row.pack(fill=tk.X, pady=2)
        ttk.Label(gain_row, text="🏆 Top Gainer:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.synthesis_stock['top_gainer'] = ttk.Label(gain_row, text="--", font=get_font('small'),
                                                        foreground=COLORS['gain'])
        self.synthesis_stock['top_gainer'].pack(side=tk.LEFT, padx=5)
        
        # Top Loser
        loss_row = ttk.Frame(stock_intel)
        loss_row.pack(fill=tk.X, pady=2)
        ttk.Label(loss_row, text="📉 Top Loser:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.synthesis_stock['top_loser'] = ttk.Label(loss_row, text="--", font=get_font('small'),
                                                       foreground=COLORS['loss'])
        self.synthesis_stock['top_loser'].pack(side=tk.LEFT, padx=5)
        
        # Most Active
        active_row = ttk.Frame(stock_intel)
        active_row.pack(fill=tk.X, pady=2)
        ttk.Label(active_row, text="🔥 Most Active:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.synthesis_stock['most_active'] = ttk.Label(active_row, text="--", font=get_font('small'),
                                                         foreground=COLORS['warning'])
        self.synthesis_stock['most_active'].pack(side=tk.LEFT, padx=5)
        
        # Stock Flow
        stflow_row = ttk.Frame(stock_intel)
        stflow_row.pack(fill=tk.X, pady=2)
        ttk.Label(stflow_row, text="💹 Flow Leaders:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.synthesis_stock['flow_leaders'] = ttk.Label(stflow_row, text="--", font=get_font('small'))
        self.synthesis_stock['flow_leaders'].pack(side=tk.LEFT, padx=5)
        
        # ========== ROW 3: FLOW PRESSURE GAUGES ==========
        flow_frame = ttk.LabelFrame(main, text="⚡ Money Flow Pressure")
        flow_frame.pack(fill=tk.X, padx=10, pady=5)
        
        flow_row = ttk.Frame(flow_frame)
        flow_row.pack(fill=tk.X, padx=5, pady=5)
        
        self.synthesis_flow = {}
        
        # Market-wide Flow Gauge
        market_flow = ttk.Frame(flow_row)
        market_flow.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(market_flow, text="📊 Market", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack()
        self.synthesis_flow['market_gauge'] = tk.Canvas(market_flow, width=200, height=20,
                                                         bg=COLORS['bg_medium'], highlightthickness=0)
        self.synthesis_flow['market_gauge'].pack(pady=2)
        self.synthesis_flow['market_label'] = ttk.Label(market_flow, text="Buy: --% | Sell: --%",
                                                         font=get_font('small'))
        self.synthesis_flow['market_label'].pack()
        
        # Sector Flow Gauge
        sector_flow = ttk.Frame(flow_row)
        sector_flow.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(sector_flow, text="🔄 Sector", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack()
        self.synthesis_flow['sector_gauge'] = tk.Canvas(sector_flow, width=200, height=20,
                                                         bg=COLORS['bg_medium'], highlightthickness=0)
        self.synthesis_flow['sector_gauge'].pack(pady=2)
        self.synthesis_flow['sector_label'] = ttk.Label(sector_flow, text="In: -- | Out: --",
                                                         font=get_font('small'))
        self.synthesis_flow['sector_label'].pack()
        
        # Stock Flow Gauge
        stock_flow = ttk.Frame(flow_row)
        stock_flow.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(stock_flow, text="📈 Stock", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack()
        self.synthesis_flow['stock_gauge'] = tk.Canvas(stock_flow, width=200, height=20,
                                                        bg=COLORS['bg_medium'], highlightthickness=0)
        self.synthesis_flow['stock_gauge'].pack(pady=2)
        self.synthesis_flow['stock_label'] = ttk.Label(stock_flow, text="Inflow: -- | Outflow: --",
                                                        font=get_font('small'))
        self.synthesis_flow['stock_label'].pack()
        
        # ========== ROW 4: AI NARRATIVE REPORT ==========
        narrative_frame = ttk.LabelFrame(main, text="🧠 AI Market Narrative")
        narrative_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Text container with scrollbar
        text_container = ttk.Frame(narrative_frame)
        text_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        text_scrollbar = ttk.Scrollbar(text_container)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.synthesis_narrative = tk.Text(
            text_container,
            wrap=tk.WORD,
            font=get_font('body'),
            bg=COLORS['bg_medium'],
            fg=COLORS['text_primary'],
            padx=10,
            pady=10,
            height=15,
            state=tk.DISABLED
        )
        self.synthesis_narrative.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.synthesis_narrative.config(yscrollcommand=text_scrollbar.set)
        text_scrollbar.config(command=self.synthesis_narrative.yview)
        
        # ========== ROW 5: KEY INSIGHTS (Market + Stock Level) ==========
        insights_frame = ttk.LabelFrame(main, text="💡 Key Insights")
        insights_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Market-Level Insights
        market_insights_row = ttk.Frame(insights_frame)
        market_insights_row.pack(fill=tk.X, padx=5, pady=3)
        
        ttk.Label(market_insights_row, text="📊 Market:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        
        self.synthesis_insights_market = []
        for i in range(4):
            insight_card = ttk.Frame(market_insights_row, padding=3)
            insight_card.pack(side=tk.LEFT, padx=5)
            
            icon_lbl = ttk.Label(insight_card, text="⚪", font=('', 12))
            icon_lbl.pack(side=tk.LEFT)
            
            text_lbl = ttk.Label(insight_card, text="--", font=get_font('small'))
            text_lbl.pack(side=tk.LEFT, padx=3)
            
            self.synthesis_insights_market.append({'icon': icon_lbl, 'text': text_lbl})
        
        # Stock-Level Insights
        stock_insights_row = ttk.Frame(insights_frame)
        stock_insights_row.pack(fill=tk.X, padx=5, pady=3)
        
        ttk.Label(stock_insights_row, text="📈 Stocks:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        
        self.synthesis_insights_stock = []
        for i in range(4):
            insight_card = ttk.Frame(stock_insights_row, padding=3)
            insight_card.pack(side=tk.LEFT, padx=5)
            
            icon_lbl = ttk.Label(insight_card, text="⚪", font=('', 12))
            icon_lbl.pack(side=tk.LEFT)
            
            text_lbl = ttk.Label(insight_card, text="--", font=get_font('small'))
            text_lbl.pack(side=tk.LEFT, padx=3)
            
            self.synthesis_insights_stock.append({'icon': icon_lbl, 'text': text_lbl})
        
        # ========== ROW 6: ANOMALY ALERTS ==========
        alerts_frame = ttk.LabelFrame(main, text="🚨 Smart Money Alerts")
        alerts_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.synthesis_alerts_container = ttk.Frame(alerts_frame)
        self.synthesis_alerts_container.pack(fill=tk.X, padx=5, pady=3)
        
        self.synthesis_alerts = []
        for i in range(3):
            alert_row = ttk.Frame(self.synthesis_alerts_container)
            alert_row.pack(fill=tk.X, pady=1)
            
            icon_lbl = ttk.Label(alert_row, text="⚠️", font=('', 10))
            icon_lbl.pack(side=tk.LEFT)
            
            text_lbl = ttk.Label(alert_row, text="--", font=get_font('small'),
                                foreground=COLORS['text_muted'])
            text_lbl.pack(side=tk.LEFT, padx=5)
            
            self.synthesis_alerts.append({'icon': icon_lbl, 'text': text_lbl})
        
        # ========== STATUS BAR ==========
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.synthesis_status = {}
        
        self.synthesis_status['live'] = ttk.Label(status_frame, text="📡 LIVE",
                                                   font=get_font('small'), foreground=COLORS['gain'])
        self.synthesis_status['live'].pack(side=tk.LEFT)
        
        self.synthesis_status['update'] = ttk.Label(status_frame, text="Last Update: --",
                                                     font=get_font('small'), foreground=COLORS['text_muted'])
        self.synthesis_status['update'].pack(side=tk.LEFT, padx=20)
        
        self.synthesis_status['ai'] = ttk.Label(status_frame, text="Powered by Groq AI",
                                                 font=get_font('small'), foreground=COLORS['text_muted'])
        self.synthesis_status['ai'].pack(side=tk.RIGHT)
    
    def _load_smart_money_data(self):
        """Load smart money analysis data."""
        def fetch():
            try:
                all_stocks = self.collector.get_all_stocks()
                stocks_list = all_stocks.to_dict('records') if not all_stocks.empty else []
                
                # Run smart money analysis
                analysis = self.smart_money_detector.analyze_stocks(stocks_list)
                alerts = self.anomaly_scanner.scan(stocks_list)
                
                # Put result in queue (thread-safe)
                self._update_queue.put(('smart_money', analysis, alerts))
            except Exception as e:
                logger.error(f"Error loading smart money data: {e}")
        
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
        
        # Poll for results
        self._poll_sector_queue()
    
    def _update_smart_money_ui(self, analysis: Dict, alerts: List[Dict]):
        """Update Smart Money UI with analysis results."""
        # ===== Market Regime =====
        regime = analysis.get('market_regime', {})
        
        health = regime.get('health_score', 50)
        self.regime_labels['health'].config(
            text=f"{health:.0f}/100",
            foreground=COLORS['gain'] if health >= 60 else (COLORS['loss'] if health < 40 else COLORS['text_primary'])
        )
        
        regime_text = regime.get('regime', 'NEUTRAL')
        self.regime_labels['regime'].config(
            text=regime_text,
            foreground=COLORS['gain'] if 'BULLISH' in regime_text else (COLORS['loss'] if 'BEARISH' in regime_text else COLORS['text_primary'])
        )
        
        self.regime_labels['trend'].config(text=regime.get('trend_strength', '--'))
        
        risk = regime.get('risk_signal', 'NEUTRAL')
        self.regime_labels['risk'].config(
            text=risk,
            foreground=COLORS['gain'] if risk == 'RISK_ON' else (COLORS['loss'] if risk == 'RISK_OFF' else COLORS['text_primary'])
        )
        
        # Draw health gauge
        self._draw_health_gauge(health)
        
        # ===== Alerts =====
        self.alerts_list.delete(0, tk.END)
        for alert in alerts[:8]:
            severity_icon = '🔴' if alert.get('severity') == 'HIGH' else '🟡'
            self.alerts_list.insert(tk.END, f"{severity_icon} {alert.get('message', '')}")
        
        # ===== Unusual Volume =====
        for item in self.unusual_vol_tree.get_children():
            self.unusual_vol_tree.delete(item)
        
        for stock in analysis.get('unusual_volume', [])[:12]:
            signal = stock.get('signal', 'WATCH')
            tag = 'buy' if signal == 'BUY' else ('sell' if signal == 'SELL' else '')
            if stock.get('category') == 'extreme':
                tag = 'extreme'
            
            self.unusual_vol_tree.insert('', 'end', values=(
                stock['symbol'],
                f"₦{stock['price']:,.2f}",
                f"{stock['change']:+.1f}%",
                f"{stock['vol_ratio']:.1f}x",
                signal
            ), tags=(tag,))
        
        # ===== Accumulation =====
        for item in self.accum_tree.get_children():
            self.accum_tree.delete(item)
        
        for stock in analysis.get('accumulation', [])[:8]:
            strength = stock.get('signal_strength', 'WEAK')
            tag = 'strong' if strength == 'STRONG' else ('moderate' if strength == 'MODERATE' else 'weak')
            
            self.accum_tree.insert('', 'end', values=(
                stock['symbol'],
                f"₦{stock['price']:,.2f}",
                f"{stock['change']:+.1f}%",
                f"{stock.get('chg_1w', 0):+.1f}%",
                f"{stock['vol_ratio']:.1f}x",
                stock['score']
            ), tags=(tag,))
        
        # ===== Distribution =====
        for item in self.distrib_tree.get_children():
            self.distrib_tree.delete(item)
        
        for stock in analysis.get('distribution', [])[:8]:
            strength = stock.get('signal_strength', 'WEAK')
            tag = 'strong' if strength == 'STRONG' else ('moderate' if strength == 'MODERATE' else 'weak')
            
            self.distrib_tree.insert('', 'end', values=(
                stock['symbol'],
                f"₦{stock['price']:,.2f}",
                f"{stock['change']:+.1f}%",
                f"{stock.get('chg_1w', 0):+.1f}%",
                f"{stock['vol_ratio']:.1f}x",
                stock['score']
            ), tags=(tag,))
        
        # ===== Breakouts =====
        for i, stock in enumerate(analysis.get('breakouts_up', [])[:3]):
            if i < len(self.breakout_up_labels):
                self.breakout_up_labels[i].config(
                    text=f"{stock['symbol']} {stock['change']:+.1f}% (RSI:{stock['rsi']:.0f})"
                )
        
        for i, stock in enumerate(analysis.get('breakouts_down', [])[:3]):
            if i < len(self.breakout_down_labels):
                self.breakout_down_labels[i].config(
                    text=f"{stock['symbol']} {stock['change']:+.1f}% (RSI:{stock['rsi']:.0f})"
                )
        
        # ===== Stealth Accumulation =====
        for item in self.stealth_tree.get_children():
            self.stealth_tree.delete(item)
        
        for stock in analysis.get('stealth_accumulation', [])[:6]:
            signal = stock.get('signal', 'WATCHING')
            tag = 'accumulating' if signal == 'ACCUMULATING' else 'watching'
            
            self.stealth_tree.insert('', 'end', values=(
                stock['symbol'],
                f"₦{stock['price']:,.2f}",
                f"{stock.get('chg_1w', 0):+.1f}%",
                f"{stock.get('volatility', 0):.1f}%",
                stock['score'],
                signal
            ), tags=(tag,))
        
        # ===== Block Trades =====
        for item in self.block_tree.get_children():
            self.block_tree.delete(item)
        
        for stock in analysis.get('block_trades', [])[:6]:
            direction = stock.get('direction', 'BUY')
            size = stock.get('size', 'MEDIUM')
            tag = 'large' if size == 'LARGE' else ('buy' if direction == 'BUY' else 'sell')
            
            self.block_tree.insert('', 'end', values=(
                stock['symbol'],
                f"₦{stock['price']:,.2f}",
                f"{stock['change']:+.1f}%",
                f"{stock['vol_ratio']:.1f}x",
                direction,
                size
            ), tags=(tag,))
    
    def _draw_health_gauge(self, health: float):
        """Draw the market health gauge."""
        self.regime_canvas.delete("all")
        
        w = self.regime_canvas.winfo_width()
        h = self.regime_canvas.winfo_height()
        if w <= 1:
            w, h = 200, 80
        
        # Background bar
        bar_y = h // 2
        bar_h = 20
        self.regime_canvas.create_rectangle(10, bar_y - bar_h//2, w - 10, bar_y + bar_h//2,
                                            fill='#333', outline='#555')
        
        # Health bar
        bar_width = int((w - 20) * (health / 100))
        
        if health >= 65:
            color = '#27ae60'
        elif health >= 50:
            color = '#f39c12'
        elif health >= 35:
            color = '#e67e22'
        else:
            color = '#e74c3c'
        
        self.regime_canvas.create_rectangle(10, bar_y - bar_h//2, 10 + bar_width, bar_y + bar_h//2,
                                            fill=color, outline='')
        
        # Text
        self.regime_canvas.create_text(w // 2, bar_y,
                                       text=f"{health:.0f}%", fill="white",
                                       font=('Arial', 12, 'bold'))
    
    # ==================== AI SYNTHESIS DATA & UPDATE ====================
    
    def _generate_market_synthesis(self) -> Dict:
        """Generate comprehensive market synthesis data from all sub-tabs."""
        synthesis = {
            # Overview
            'market_health': 50,
            'market_regime': 'NEUTRAL',
            'overall_bias': 'NEUTRAL',
            'trend': 'SIDEWAYS',
            'confidence': 50,
            
            # Breadth
            'total_stocks': 0,
            'advancers': 0,
            'decliners': 0,
            'unchanged': 0,
            'adv_dec_ratio': 1.0,
            
            # Sector
            'leading_sectors': [],
            'lagging_sectors': [],
            'rotation_phase': 'Unknown',
            'sector_rankings': [],
            'sector_inflows': 0,
            'sector_outflows': 0,
            
            # Stock Intelligence
            'top_gainers': [],
            'top_losers': [],
            'most_active': [],
            'flow_leaders': [],
            'flow_laggards': [],
            
            # Flow
            'buying_pressure': 50,
            'selling_pressure': 50,
            'net_flow': 0,
            'inflow_count': 0,
            'outflow_count': 0,
            
            # Smart Money
            'anomaly_count': 0,
            'anomalies': [],
            
            # Insights
            'market_insights': [],
            'stock_insights': [],
        }
        
        try:
            # ========== AGGREGATE FROM LIVE MARKET TAB ==========
            stocks = []
            
            # Try to get from live_market_tab
            if hasattr(self, 'live_market_tab') and hasattr(self.live_market_tab, 'all_stocks_data'):
                stocks = self.live_market_tab.all_stocks_data or []
                logger.info(f"AI Synthesis: Got {len(stocks)} stocks from live_market_tab.all_stocks_data")
            
            # Fallback: try to fetch directly from collector
            if not stocks and hasattr(self, 'collector'):
                try:
                    all_stocks = self.collector.get_all_stocks()
                    stocks = all_stocks.to_dict('records') if not all_stocks.empty else []
                    logger.info(f"AI Synthesis: Got {len(stocks)} stocks from collector fallback")
                except Exception as e:
                    logger.warning(f"AI Synthesis: Collector fallback failed - {e}")
            
            if stocks:
                synthesis['total_stocks'] = len(stocks)
                
                advancers = [s for s in stocks if (s.get('change', 0) or 0) > 0]
                decliners = [s for s in stocks if (s.get('change', 0) or 0) < 0]
                unchanged = [s for s in stocks if (s.get('change', 0) or 0) == 0]
                
                synthesis['advancers'] = len(advancers)
                synthesis['decliners'] = len(decliners)
                synthesis['unchanged'] = len(unchanged)
                
                if synthesis['decliners'] > 0:
                    synthesis['adv_dec_ratio'] = synthesis['advancers'] / synthesis['decliners']
                elif synthesis['advancers'] > 0:
                    synthesis['adv_dec_ratio'] = synthesis['advancers']  # High ratio
                
                # Top gainers/losers
                sorted_by_change = sorted(stocks, key=lambda x: x.get('change', 0) or 0, reverse=True)
                synthesis['top_gainers'] = sorted_by_change[:5]
                synthesis['top_losers'] = sorted_by_change[-5:][::-1]
                
                # Most active
                sorted_by_vol = sorted(stocks, key=lambda x: x.get('volume', 0) or 0, reverse=True)
                synthesis['most_active'] = sorted_by_vol[:5]
                
                # Calculate health from breadth
                if synthesis['total_stocks'] > 0:
                    breadth_pct = synthesis['advancers'] / synthesis['total_stocks'] * 100
                    synthesis['market_health'] = min(100, max(0, breadth_pct * 1.2))
                
                # Determine overall bias
                if synthesis['adv_dec_ratio'] > 1.5:
                    synthesis['overall_bias'] = 'BULLISH'
                    synthesis['market_regime'] = 'RISK-ON'
                    synthesis['trend'] = 'UPTREND'
                elif synthesis['adv_dec_ratio'] > 1.0:
                    synthesis['overall_bias'] = 'SLIGHTLY BULLISH'
                    synthesis['trend'] = 'MILD UPTREND'
                elif synthesis['adv_dec_ratio'] < 0.67:
                    synthesis['overall_bias'] = 'BEARISH'
                    synthesis['market_regime'] = 'RISK-OFF'
                    synthesis['trend'] = 'DOWNTREND'
                elif synthesis['adv_dec_ratio'] < 1.0:
                    synthesis['overall_bias'] = 'SLIGHTLY BEARISH'
                    synthesis['trend'] = 'MILD DOWNTREND'
                
                # Store stocks for other uses
                self.all_stocks_data = stocks
            
            # ========== AGGREGATE FROM SECTOR ROTATION TAB ==========
            if hasattr(self, 'sector_analysis') and stocks:
                try:
                    phase_data = self.sector_analysis.detect_rotation_phase(stocks)
                    synthesis['rotation_phase'] = phase_data.get('description', 'Unknown')[:40]
                    synthesis['leading_sectors'] = phase_data.get('leading', [])[:3]
                    synthesis['lagging_sectors'] = phase_data.get('lagging', [])[:3]
                except Exception as e:
                    logger.warning(f"AI Synthesis: Sector rotation analysis failed - {e}")
            
            # Fallback: Get sector data from sector_tree if phase data is empty
            if not synthesis['leading_sectors'] and hasattr(self, 'sector_tree'):
                try:
                    children = self.sector_tree.get_children()
                    if children:
                        # Get first 2 sectors as leading
                        leading = []
                        lagging = []
                        for i, item in enumerate(children):
                            values = self.sector_tree.item(item, 'values')
                            if values:
                                sector = values[0]
                                if i < 2:
                                    leading.append(sector)
                                elif i >= len(children) - 2:
                                    lagging.append(sector)
                        synthesis['leading_sectors'] = leading
                        synthesis['lagging_sectors'] = lagging
                except:
                    pass
            
            # ========== AGGREGATE FROM FLOW MONITOR TAB ==========
            # Calculate flow from stocks data directly (more reliable)
            if stocks:
                inflow_stocks = [s for s in stocks if (s.get('Perf.W', 0) or 0) >= 5]  # Strong inflow
                mod_inflow = [s for s in stocks if 0 < (s.get('Perf.W', 0) or 0) < 5]  # Moderate inflow
                outflow_stocks = [s for s in stocks if (s.get('Perf.W', 0) or 0) <= -5]  # Strong outflow
                mod_outflow = [s for s in stocks if -5 < (s.get('Perf.W', 0) or 0) < 0]  # Moderate outflow
                
                synthesis['inflow_count'] = len(inflow_stocks) + len(mod_inflow)
                synthesis['outflow_count'] = len(outflow_stocks) + len(mod_outflow)
                synthesis['sector_inflows'] = len(inflow_stocks) + len(mod_inflow)
                synthesis['sector_outflows'] = len(outflow_stocks) + len(mod_outflow)
                
                total_flow = synthesis['inflow_count'] + synthesis['outflow_count']
                if total_flow > 0:
                    synthesis['buying_pressure'] = synthesis['inflow_count'] / total_flow * 100
                    synthesis['selling_pressure'] = synthesis['outflow_count'] / total_flow * 100
                    synthesis['net_flow'] = synthesis['buying_pressure'] - synthesis['selling_pressure']
                
                # Flow leaders (top weekly performers)
                sorted_by_week = sorted(stocks, key=lambda x: x.get('Perf.W', 0) or 0, reverse=True)
                synthesis['flow_leaders'] = [s.get('symbol', '?') for s in sorted_by_week[:3]]
                synthesis['flow_laggards'] = [s.get('symbol', '?') for s in sorted_by_week[-3:][::-1]]
            
            # ========== AGGREGATE FROM SMART MONEY TAB ==========
            if hasattr(self, 'regime_labels'):
                try:
                    health_text = self.regime_labels.get('health', ttk.Label()).cget('text')
                    if health_text and health_text != '--':
                        # Parse health percentage
                        h = float(health_text.replace('%', '').strip())
                        synthesis['market_health'] = h
                    
                    regime_text = self.regime_labels.get('regime', ttk.Label()).cget('text')
                    if regime_text and regime_text != '--':
                        synthesis['market_regime'] = regime_text
                    
                    trend_text = self.regime_labels.get('trend', ttk.Label()).cget('text')
                    if trend_text and trend_text != '--':
                        synthesis['trend'] = trend_text
                except:
                    pass
            
            # Get anomaly alerts
            if hasattr(self, 'alerts_tree'):
                try:
                    anomalies = []
                    for item in self.alerts_tree.get_children()[:5]:
                        values = self.alerts_tree.item(item, 'values')
                        if values:
                            anomalies.append({
                                'time': values[0] if len(values) > 0 else '',
                                'type': values[1] if len(values) > 1 else '',
                                'symbol': values[2] if len(values) > 2 else '',
                                'message': values[3] if len(values) > 3 else '',
                            })
                    synthesis['anomalies'] = anomalies
                    synthesis['anomaly_count'] = len(anomalies)
                except:
                    pass
            
            # ========== GENERATE INSIGHTS ==========
            market_insights = []
            stock_insights = []
            
            # Market-level insights
            if synthesis['adv_dec_ratio'] > 2.0:
                market_insights.append(('🟢', 'Strong Breadth'))
            elif synthesis['adv_dec_ratio'] < 0.5:
                market_insights.append(('🔴', 'Weak Breadth'))
            
            if synthesis['buying_pressure'] > 65:
                market_insights.append(('💹', 'Heavy Buying'))
            elif synthesis['selling_pressure'] > 65:
                market_insights.append(('📉', 'Heavy Selling'))
            
            if synthesis['market_health'] >= 70:
                market_insights.append(('🏥', 'Healthy Market'))
            elif synthesis['market_health'] < 40:
                market_insights.append(('⚠️', 'Weak Market'))
            
            if synthesis['anomaly_count'] > 3:
                market_insights.append(('🚨', f'{synthesis["anomaly_count"]} Anomalies'))
            
            synthesis['market_insights'] = market_insights[:4]
            
            # Stock-level insights
            if synthesis['top_gainers']:
                top = synthesis['top_gainers'][0]
                stock_insights.append(('🏆', f"{top.get('symbol', '?')} +{top.get('change', 0):.1f}%"))
            
            if synthesis['top_losers']:
                bottom = synthesis['top_losers'][0]
                stock_insights.append(('📉', f"{bottom.get('symbol', '?')} {bottom.get('change', 0):.1f}%"))
            
            if synthesis['flow_leaders']:
                stock_insights.append(('💹', f"Inflow: {', '.join(synthesis['flow_leaders'][:2])}"))
            
            if synthesis['most_active']:
                active = synthesis['most_active'][0]
                stock_insights.append(('🔥', f"Active: {active.get('symbol', '?')}"))
            
            synthesis['stock_insights'] = stock_insights[:4]
            
            # Calculate confidence
            confidence_factors = []
            if synthesis['adv_dec_ratio'] > 1.2 or synthesis['adv_dec_ratio'] < 0.8:
                confidence_factors.append(20)  # Strong breadth direction
            if abs(synthesis['net_flow']) > 20:
                confidence_factors.append(20)  # Strong flow direction
            if synthesis['anomaly_count'] > 0:
                confidence_factors.append(15)  # Smart money signals
            confidence_factors.append(30)  # Base confidence
            synthesis['confidence'] = min(100, sum(confidence_factors))
            
        except Exception as e:
            logger.error(f"Error generating market synthesis: {e}")
            import traceback
            traceback.print_exc()
        
        return synthesis
    
    def _draw_flow_gauge(self, canvas: tk.Canvas, buy_pct: float, sell_pct: float):
        """Draw a flow pressure gauge on a canvas."""
        canvas.delete("all")
        w = canvas.winfo_width() or 200
        h = canvas.winfo_height() or 20
        
        # Background
        canvas.create_rectangle(0, 0, w, h, fill=COLORS['bg_medium'], outline='')
        
        # Green (buy) section
        buy_width = int(w * buy_pct / 100)
        canvas.create_rectangle(0, 0, buy_width, h, fill=COLORS['gain'], outline='')
        
        # Red (sell) section from right
        sell_width = int(w * sell_pct / 100)
        canvas.create_rectangle(w - sell_width, 0, w, h, fill=COLORS['loss'], outline='')
        
        # Center line
        canvas.create_line(w // 2, 0, w // 2, h, fill='white', width=2)
    
    def _update_ai_synthesis(self):
        """Update the AI Synthesis tab with aggregated data."""
        try:
            synthesis = self._generate_market_synthesis()
            
            # ========== UPDATE OVERVIEW CARDS ==========
            # Health
            health = synthesis.get('market_health', 50)
            self.synthesis_overview['health_value'].config(text=f"{health:.0f}%")
            health_color = COLORS['gain'] if health >= 60 else COLORS['warning'] if health >= 40 else COLORS['loss']
            self.synthesis_overview['health_value'].config(foreground=health_color)
            
            # Health bar
            hbar = self.synthesis_overview['health_bar']
            hbar.delete("all")
            bar_width = int(health / 100 * 100)
            hbar.create_rectangle(0, 0, 100, 8, fill=COLORS['bg_medium'], outline='')
            hbar.create_rectangle(0, 0, bar_width, 8, fill=health_color, outline='')
            
            self.synthesis_overview['health_driver'].config(
                text=f"A/D: {synthesis.get('adv_dec_ratio', 1):.2f}"
            )
            
            # Regime
            regime = synthesis.get('market_regime', 'NEUTRAL')
            self.synthesis_overview['regime_value'].config(text=regime)
            if 'RISK-ON' in regime or 'BULL' in regime:
                self.synthesis_overview['regime_icon'].config(text="🟢")
                self.synthesis_overview['regime_value'].config(foreground=COLORS['gain'])
            elif 'RISK-OFF' in regime or 'BEAR' in regime:
                self.synthesis_overview['regime_icon'].config(text="🔴")
                self.synthesis_overview['regime_value'].config(foreground=COLORS['loss'])
            else:
                self.synthesis_overview['regime_icon'].config(text="🟡")
                self.synthesis_overview['regime_value'].config(foreground=COLORS['warning'])
            
            self.synthesis_overview['regime_driver'].config(
                text=synthesis.get('overall_bias', 'NEUTRAL')
            )
            
            # Breadth
            adv = synthesis.get('advancers', 0)
            dec = synthesis.get('decliners', 0)
            ratio = synthesis.get('adv_dec_ratio', 1)
            self.synthesis_overview['breadth_value'].config(text=f"{ratio:.2f}")
            self.synthesis_overview['breadth_detail'].config(text=f"↑{adv} ↓{dec}")
            
            breadth_color = COLORS['gain'] if ratio > 1 else COLORS['loss'] if ratio < 1 else COLORS['text_primary']
            self.synthesis_overview['breadth_value'].config(foreground=breadth_color)
            
            self.synthesis_overview['breadth_driver'].config(
                text=f"Total: {synthesis.get('total_stocks', 0)}"
            )
            
            # Trend
            trend = synthesis.get('trend', 'SIDEWAYS')
            self.synthesis_overview['trend_value'].config(text=trend)
            if 'UP' in trend:
                self.synthesis_overview['trend_icon'].config(text="📈")
                self.synthesis_overview['trend_value'].config(foreground=COLORS['gain'])
            elif 'DOWN' in trend:
                self.synthesis_overview['trend_icon'].config(text="📉")
                self.synthesis_overview['trend_value'].config(foreground=COLORS['loss'])
            else:
                self.synthesis_overview['trend_icon'].config(text="➡️")
                self.synthesis_overview['trend_value'].config(foreground=COLORS['warning'])
            
            self.synthesis_overview['trend_driver'].config(
                text=f"Conf: {synthesis.get('confidence', 50):.0f}%"
            )
            
            # ========== UPDATE SECTOR INTELLIGENCE ==========
            leading = synthesis.get('leading_sectors', [])
            lagging = synthesis.get('lagging_sectors', [])
            
            self.synthesis_sector['leading'].config(text=', '.join(leading[:2]) if leading else '--')
            self.synthesis_sector['lagging'].config(text=', '.join(lagging[:2]) if lagging else '--')
            self.synthesis_sector['phase'].config(text=synthesis.get('rotation_phase', '--')[:25])
            
            sector_in = synthesis.get('sector_inflows', 0)
            sector_out = synthesis.get('sector_outflows', 0)
            flow_text = f"In: {sector_in} | Out: {sector_out}"
            self.synthesis_sector['flow'].config(text=flow_text)
            
            # ========== UPDATE STOCK INTELLIGENCE ==========
            top_gainers = synthesis.get('top_gainers', [])
            top_losers = synthesis.get('top_losers', [])
            most_active = synthesis.get('most_active', [])
            flow_leaders = synthesis.get('flow_leaders', [])
            
            if top_gainers:
                g = top_gainers[0]
                self.synthesis_stock['top_gainer'].config(
                    text=f"{g.get('symbol', '?')} +{g.get('change', 0):.1f}%"
                )
            
            if top_losers:
                l = top_losers[0]
                self.synthesis_stock['top_loser'].config(
                    text=f"{l.get('symbol', '?')} {l.get('change', 0):.1f}%"
                )
            
            if most_active:
                a = most_active[0]
                vol = a.get('volume', 0)
                vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K" if vol >= 1e3 else str(vol)
                self.synthesis_stock['most_active'].config(
                    text=f"{a.get('symbol', '?')} ({vol_str})"
                )
            
            if flow_leaders:
                self.synthesis_stock['flow_leaders'].config(
                    text=', '.join(flow_leaders[:3])
                )
            
            # ========== UPDATE FLOW GAUGES ==========
            buy_pct = synthesis.get('buying_pressure', 50)
            sell_pct = synthesis.get('selling_pressure', 50)
            
            self._draw_flow_gauge(self.synthesis_flow['market_gauge'], buy_pct, sell_pct)
            self.synthesis_flow['market_label'].config(text=f"Buy: {buy_pct:.0f}% | Sell: {sell_pct:.0f}%")
            
            in_count = synthesis.get('inflow_count', 0)
            out_count = synthesis.get('outflow_count', 0)
            total = in_count + out_count or 1
            self._draw_flow_gauge(self.synthesis_flow['sector_gauge'], in_count/total*100, out_count/total*100)
            self.synthesis_flow['sector_label'].config(text=f"In: {in_count} | Out: {out_count}")
            
            # Stock flow based on top movers
            stock_in = len([g for g in top_gainers if g.get('change', 0) > 0])
            stock_out = len([l for l in top_losers if l.get('change', 0) < 0])
            stock_total = stock_in + stock_out or 1
            self._draw_flow_gauge(self.synthesis_flow['stock_gauge'], stock_in/stock_total*100, stock_out/stock_total*100)
            self.synthesis_flow['stock_label'].config(text=f"Inflow: {stock_in} | Outflow: {stock_out}")
            
            # ========== UPDATE AI NARRATIVE ==========
            narrative = self._build_market_narrative(synthesis)
            self.synthesis_narrative.config(state=tk.NORMAL)
            self.synthesis_narrative.delete(1.0, tk.END)
            self.synthesis_narrative.insert(tk.END, narrative)
            self.synthesis_narrative.config(state=tk.DISABLED)
            
            # ========== UPDATE KEY INSIGHTS ==========
            market_insights = synthesis.get('market_insights', [])
            for i, card in enumerate(self.synthesis_insights_market):
                if i < len(market_insights):
                    icon, text = market_insights[i]
                    card['icon'].config(text=icon)
                    card['text'].config(text=text)
                else:
                    card['icon'].config(text="⚪")
                    card['text'].config(text="--")
            
            stock_insights = synthesis.get('stock_insights', [])
            for i, card in enumerate(self.synthesis_insights_stock):
                if i < len(stock_insights):
                    icon, text = stock_insights[i]
                    card['icon'].config(text=icon)
                    card['text'].config(text=text)
                else:
                    card['icon'].config(text="⚪")
                    card['text'].config(text="--")
            
            # ========== UPDATE ALERTS ==========
            anomalies = synthesis.get('anomalies', [])
            for i, alert in enumerate(self.synthesis_alerts):
                if i < len(anomalies):
                    a = anomalies[i]
                    alert_type = a.get('type', '')
                    if 'ACCUM' in alert_type.upper():
                        alert['icon'].config(text="🟢")
                    elif 'DIST' in alert_type.upper():
                        alert['icon'].config(text="🔴")
                    else:
                        alert['icon'].config(text="⚠️")
                    
                    alert['text'].config(text=f"{a.get('symbol', '?')}: {a.get('message', '--')[:50]}")
                    alert['text'].config(foreground=COLORS['warning'])
                else:
                    alert['icon'].config(text="⚪")
                    alert['text'].config(text="No alerts")
                    alert['text'].config(foreground=COLORS['text_muted'])
            
            # ========== UPDATE STATUS ==========
            self.synthesis_status['update'].config(
                text=f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            if self.insight_engine:
                self.synthesis_status['ai'].config(text="✨ Powered by Groq AI")
            else:
                self.synthesis_status['ai'].config(text="📊 Rule-based Analysis")
            
        except Exception as e:
            logger.error(f"Error updating AI synthesis: {e}")
            import traceback
            traceback.print_exc()
    
    def _build_market_narrative(self, synthesis: Dict) -> str:
        """Build comprehensive AI market narrative - uses Groq if available."""
        
        # Try to use Groq AI for super detailed narrative
        if self.insight_engine:
            try:
                logger.info("Generating Market Intel AI narrative via Groq...")
                
                # Build comprehensive context
                context = synthesis.copy()
                
                # Format stock data for AI
                top_gainers_str = "\n".join([
                    f"  • {s.get('symbol', '?')}: +{s.get('change', 0):.2f}%, Vol: {s.get('volume', 0):,.0f}"
                    for s in synthesis.get('top_gainers', [])[:5]
                ]) or "  None available"
                
                top_losers_str = "\n".join([
                    f"  • {s.get('symbol', '?')}: {s.get('change', 0):.2f}%, Vol: {s.get('volume', 0):,.0f}"
                    for s in synthesis.get('top_losers', [])[:5]
                ]) or "  None available"
                
                most_active_str = "\n".join([
                    f"  • {s.get('symbol', '?')}: {s.get('volume', 0):,.0f} vol, {s.get('change', 0):+.2f}%"
                    for s in synthesis.get('most_active', [])[:5]
                ]) or "  None available"
                
                anomalies_str = "\n".join([
                    f"  • [{a.get('type', '?')}] {a.get('symbol', '?')}: {a.get('message', '')[:60]}"
                    for a in synthesis.get('anomalies', [])[:5]
                ]) or "  No anomalies detected"
                
                system_prompt = """You are an elite institutional equity strategist and market intelligence analyst for the Nigerian Stock Exchange (NGX).
Your analysis is used by portfolio managers, hedge funds, and institutional investors.
Provide an extremely detailed, data-driven market intelligence report.
Your analysis should be comprehensive, actionable, and demonstrate deep market insight.
Use specific numbers, percentages, and stock symbols throughout.
Be decisive and provide clear recommendations.
The audience expects institutional-quality analysis with detailed drill-down."""

                prompt = f"""Generate a COMPREHENSIVE market intelligence synthesis report for the Nigerian Stock Exchange:

═══════════════════════════════════════════════════════════════════════════════
                        MARKET INTELLIGENCE DASHBOARD
═══════════════════════════════════════════════════════════════════════════════

▸ MARKET HEALTH: {synthesis.get('market_health', 50):.0f}%
▸ MARKET REGIME: {synthesis.get('market_regime', 'NEUTRAL')}
▸ OVERALL BIAS: {synthesis.get('overall_bias', 'NEUTRAL')}
▸ TREND: {synthesis.get('trend', 'SIDEWAYS')}
▸ CONFIDENCE: {synthesis.get('confidence', 50):.0f}%

───────────────────────────────────────────────────────────────────────────────
                        MARKET BREADTH ANALYSIS
───────────────────────────────────────────────────────────────────────────────
▸ Total Stocks: {synthesis.get('total_stocks', 0)}
▸ Advancers: {synthesis.get('advancers', 0)}
▸ Decliners: {synthesis.get('decliners', 0)}
▸ Unchanged: {synthesis.get('unchanged', 0)}
▸ Advance/Decline Ratio: {synthesis.get('adv_dec_ratio', 1):.2f}

───────────────────────────────────────────────────────────────────────────────
                        SECTOR ROTATION INTELLIGENCE
───────────────────────────────────────────────────────────────────────────────
▸ Rotation Phase: {synthesis.get('rotation_phase', 'Unknown')}
▸ Leading Sectors: {', '.join(synthesis.get('leading_sectors', ['None'])[:3])}
▸ Lagging Sectors: {', '.join(synthesis.get('lagging_sectors', ['None'])[:3])}

───────────────────────────────────────────────────────────────────────────────
                        MONEY FLOW ANALYSIS
───────────────────────────────────────────────────────────────────────────────
▸ Buying Pressure: {synthesis.get('buying_pressure', 50):.1f}%
▸ Selling Pressure: {synthesis.get('selling_pressure', 50):.1f}%
▸ Net Flow: {synthesis.get('net_flow', 0):+.1f}%
▸ Stocks with Inflows: {synthesis.get('inflow_count', 0)}
▸ Stocks with Outflows: {synthesis.get('outflow_count', 0)}
▸ Flow Leaders: {', '.join(synthesis.get('flow_leaders', ['None'])[:5])}
▸ Flow Laggards: {', '.join(synthesis.get('flow_laggards', ['None'])[:5])}

───────────────────────────────────────────────────────────────────────────────
                        STOCK-LEVEL INTELLIGENCE
───────────────────────────────────────────────────────────────────────────────

🏆 TOP GAINERS:
{top_gainers_str}

📉 TOP LOSERS:
{top_losers_str}

🔥 MOST ACTIVE (by Volume):
{most_active_str}

───────────────────────────────────────────────────────────────────────────────
                        SMART MONEY SIGNALS
───────────────────────────────────────────────────────────────────────────────
▸ Anomaly Count: {synthesis.get('anomaly_count', 0)}

🚨 ANOMALY ALERTS:
{anomalies_str}

═══════════════════════════════════════════════════════════════════════════════

Based on ALL the above data, provide an EXTREMELY DETAILED institutional-quality analysis including:

1. **MARKET STRUCTURE ASSESSMENT** (1 paragraph)
   - Overall market condition with specific metrics
   - Breadth quality analysis (A/D ratio interpretation)
   - Regime characterization

2. **SECTOR ROTATION ANALYSIS**
   - Current rotation phase interpretation
   - Which sectors are attracting capital and why
   - Sector allocation recommendations

3. **MONEY FLOW DEEP DIVE**
   - Institutional vs retail flow interpretation
   - Flow momentum and sustainability
   - Accumulation/distribution patterns

4. **STOCK-LEVEL DRILL DOWN**
   - Analysis of top gainers (what's driving the moves)
   - Analysis of top losers (red flags or opportunities)
   - Volume analysis of most active names
   - Which specific stocks warrant attention

5. **SMART MONEY INTERPRETATION**
   - What the anomalies are signaling
   - Unusual activity interpretation
   - Hidden accumulation/distribution patterns

6. **ACTIONABLE RECOMMENDATIONS**
   - Specific sector overweight/underweight calls
   - 3-5 specific stock ideas with rationale
   - Risk positioning guidance

7. **KEY RISKS TO MONITOR**
   - What could invalidate this thesis
   - Warning signals to watch
   - Key levels or metrics to track

8. **OUTLOOK & PROBABILITY ASSESSMENT**
   - Near-term (1-5 days) outlook
   - Confidence level and key drivers"""

                ai_response = self.insight_engine.generate(prompt, system_prompt)
                
                if ai_response and "unavailable" not in ai_response.lower():
                    logger.info("Groq AI narrative generated successfully")
                    lines = []
                    lines.append("═" * 60)
                    lines.append("  🤖 AI MARKET INTELLIGENCE SYNTHESIS")
                    lines.append("═" * 60)
                    lines.append(f"  Health: {synthesis.get('market_health', 50):.0f}% | Regime: {synthesis.get('market_regime', 'NEUTRAL')} | Bias: {synthesis.get('overall_bias', 'NEUTRAL')}")
                    lines.append("═" * 60)
                    lines.append("")
                    lines.append(ai_response)
                    lines.append("")
                    lines.append("─" * 60)
                    lines.append(f"  Generated via Groq AI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    lines.append("─" * 60)
                    return "\n".join(lines)
                    
            except Exception as e:
                logger.error(f"Groq AI narrative failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to rule-based narrative
        lines = []
        lines.append("═" * 60)
        lines.append("  📊 MARKET INTELLIGENCE REPORT")
        lines.append("═" * 60)
        lines.append("")
        
        # Market Overview
        lines.append("▶ MARKET OVERVIEW:")
        lines.append(f"  • Health Score: {synthesis.get('market_health', 50):.0f}%")
        lines.append(f"  • Market Regime: {synthesis.get('market_regime', 'NEUTRAL')}")
        lines.append(f"  • Overall Bias: {synthesis.get('overall_bias', 'NEUTRAL')}")
        lines.append(f"  • Trend Direction: {synthesis.get('trend', 'SIDEWAYS')}")
        lines.append("")
        
        # Breadth
        lines.append("▶ MARKET BREADTH:")
        lines.append(f"  • Advance/Decline: {synthesis.get('advancers', 0)} / {synthesis.get('decliners', 0)}")
        lines.append(f"  • A/D Ratio: {synthesis.get('adv_dec_ratio', 1):.2f}")
        lines.append(f"  • Total Traded: {synthesis.get('total_stocks', 0)} securities")
        lines.append("")
        
        # Sector
        lines.append("▶ SECTOR ROTATION:")
        lines.append(f"  • Phase: {synthesis.get('rotation_phase', 'Unknown')}")
        lines.append(f"  • Leading: {', '.join(synthesis.get('leading_sectors', ['--'])[:2])}")
        lines.append(f"  • Lagging: {', '.join(synthesis.get('lagging_sectors', ['--'])[:2])}")
        lines.append("")
        
        # Flow
        lines.append("▶ MONEY FLOW:")
        lines.append(f"  • Buying Pressure: {synthesis.get('buying_pressure', 50):.0f}%")
        lines.append(f"  • Selling Pressure: {synthesis.get('selling_pressure', 50):.0f}%")
        lines.append(f"  • Net Flow: {synthesis.get('net_flow', 0):+.0f}%")
        lines.append("")
        
        # Top Movers
        lines.append("▶ TOP MOVERS:")
        for g in synthesis.get('top_gainers', [])[:3]:
            lines.append(f"  • {g.get('symbol', '?')}: +{g.get('change', 0):.2f}%")
        lines.append("")
        
        lines.append("─" * 60)
        lines.append(f"  Generated (Rule-based): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  💡 Set GROQ_API_KEY for AI-powered insights")
        lines.append("─" * 60)
        
        return "\n".join(lines)
    
    def refresh(self):
        """Refresh all data."""
        self.live_market_tab.refresh()
        self._load_sector_data()
        self._load_flow_data()
        self._load_smart_money_data()
        # Update AI synthesis after data is refreshed
        self.frame.after(2000, self._update_ai_synthesis)
