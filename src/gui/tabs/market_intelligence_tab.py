"""
Market Intelligence Tab for MetaQuant Nigeria.
Consolidates Live Market, Sector Rotation, and Flow Analysis.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict
import logging
import threading
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
        
        # Add tabs - use LiveMarketTab's frame directly
        self.sub_notebook.add(self.live_market_tab.frame, text="📈 Live Market")
        self.sub_notebook.add(self.sector_frame, text="🔄 Sector Rotation")
        self.sub_notebook.add(self.flow_frame, text="💧 Flow Monitor")
        self.sub_notebook.add(self.smart_money_frame, text="🕵️ Smart Money")
        
        # Build other sub-tab UIs
        self._build_sector_rotation_ui()
        self._build_flow_monitor_ui()
        self._build_smart_money_ui()
    
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
        
        # ===== TOP ROW: Sector Heatmap + Rotation Cycle =====
        top_row = ttk.Frame(self.sector_frame)
        top_row.pack(fill=tk.X, padx=10, pady=5)
        
        # LEFT: Sector Performance Heatmap
        heatmap_frame = ttk.LabelFrame(top_row, text="📊 Sector Performance Heatmap", padding=5)
        heatmap_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.sector_heatmap_canvas = tk.Canvas(
            heatmap_frame,
            bg=COLORS['bg_dark'],
            height=140,
            highlightthickness=0
        )
        self.sector_heatmap_canvas.pack(fill=tk.BOTH, expand=True)
        
        # RIGHT: Rotation Cycle Indicator
        cycle_frame = ttk.LabelFrame(top_row, text="🔄 Rotation Cycle", padding=10)
        cycle_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Cycle summary
        self.cycle_labels = {}
        
        cycle_row1 = ttk.Frame(cycle_frame)
        cycle_row1.pack(fill=tk.X, pady=3)
        ttk.Label(cycle_row1, text="Leading:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.cycle_labels['leading'] = ttk.Label(cycle_row1, text="--", font=get_font('small'),
                                                  foreground=COLORS['gain'])
        self.cycle_labels['leading'].pack(side=tk.LEFT, padx=5)
        
        cycle_row2 = ttk.Frame(cycle_frame)
        cycle_row2.pack(fill=tk.X, pady=3)
        ttk.Label(cycle_row2, text="Lagging:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.cycle_labels['lagging'] = ttk.Label(cycle_row2, text="--", font=get_font('small'),
                                                  foreground=COLORS['loss'])
        self.cycle_labels['lagging'].pack(side=tk.LEFT, padx=5)
        
        cycle_row3 = ttk.Frame(cycle_frame)
        cycle_row3.pack(fill=tk.X, pady=3)
        ttk.Label(cycle_row3, text="Spread:", font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        self.cycle_labels['spread'] = ttk.Label(cycle_row3, text="--", font=get_font('small'),
                                                 foreground=COLORS['primary'])
        self.cycle_labels['spread'].pack(side=tk.LEFT, padx=5)
        
        # Sector leaders list
        leaders_frame = ttk.LabelFrame(cycle_frame, text="🏆 Sector Leaders", padding=5)
        leaders_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Header for leaders
        leader_header = ttk.Frame(leaders_frame)
        leader_header.pack(fill=tk.X)
        ttk.Label(leader_header, text="Sector", width=12, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        ttk.Label(leader_header, text="Leader", width=10, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        ttk.Label(leader_header, text="Chg%", width=7, font=get_font('small'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        
        self.sector_leader_labels = []
        for i in range(5):
            row = ttk.Frame(leaders_frame)
            row.pack(fill=tk.X, pady=1)
            sector_lbl = ttk.Label(row, text="--", width=12, font=get_font('small'))
            sector_lbl.pack(side=tk.LEFT)
            leader_lbl = ttk.Label(row, text="--", width=10, font=get_font('small'),
                                   foreground=COLORS['gain'])
            leader_lbl.pack(side=tk.LEFT)
            chg_lbl = ttk.Label(row, text="--", width=7, font=get_font('small'))
            chg_lbl.pack(side=tk.LEFT)
            self.sector_leader_labels.append((sector_lbl, leader_lbl, chg_lbl))
        
        # ===== MIDDLE ROW: Sector Rankings + Components =====
        middle_row = ttk.Frame(self.sector_frame)
        middle_row.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # LEFT: Sector Rankings with multi-period columns
        left_frame = ttk.LabelFrame(middle_row, text="📊 Sector Rankings (Multi-Period)", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        columns = ('sector', 'chg_1d', 'chg_1w', 'chg_1m', 'net', 'count')
        self.sector_tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=8)
        
        self.sector_tree.heading('sector', text='Sector')
        self.sector_tree.heading('chg_1d', text='1D')
        self.sector_tree.heading('chg_1w', text='1W')
        self.sector_tree.heading('chg_1m', text='1M')
        self.sector_tree.heading('net', text='Net')
        self.sector_tree.heading('count', text='#')
        
        self.sector_tree.column('sector', width=110, anchor='w')
        self.sector_tree.column('chg_1d', width=50, anchor='e')
        self.sector_tree.column('chg_1w', width=50, anchor='e')
        self.sector_tree.column('chg_1m', width=50, anchor='e')
        self.sector_tree.column('net', width=55, anchor='e')
        self.sector_tree.column('count', width=30, anchor='center')
        
        self.sector_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.sector_tree.tag_configure('loss', foreground=COLORS['loss'])
        self.sector_tree.tag_configure('hot', foreground='#00ff00')
        self.sector_tree.tag_configure('cold', foreground='#ff4444')
        
        self.sector_tree.pack(fill=tk.BOTH, expand=True)
        self.sector_tree.bind('<<TreeviewSelect>>', self._on_sector_select)
        
        # RIGHT: Sector Components with RS
        right_frame = ttk.LabelFrame(middle_row, text="📋 Sector Components (click sector to load)", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Selected sector label
        self.selected_sector_label = ttk.Label(right_frame, text="Select a sector →", 
                                                font=get_font('small'),
                                                foreground=COLORS['text_muted'])
        self.selected_sector_label.pack(anchor='w')
        
        columns = ('symbol', 'price', 'chg_1d', 'chg_1w', 'weight', 'momentum')
        self.component_tree = ttk.Treeview(right_frame, columns=columns, show='headings', height=8)
        
        self.component_tree.heading('symbol', text='Symbol')
        self.component_tree.heading('price', text='Price')
        self.component_tree.heading('chg_1d', text='1D')
        self.component_tree.heading('chg_1w', text='1W')
        self.component_tree.heading('weight', text='Wt%')
        self.component_tree.heading('momentum', text='Trend')
        
        self.component_tree.column('symbol', width=70, anchor='w')
        self.component_tree.column('price', width=70, anchor='e')
        self.component_tree.column('chg_1d', width=50, anchor='e')
        self.component_tree.column('chg_1w', width=50, anchor='e')
        self.component_tree.column('weight', width=45, anchor='e')
        self.component_tree.column('momentum', width=70, anchor='center')
        
        self.component_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.component_tree.tag_configure('loss', foreground=COLORS['loss'])
        self.component_tree.tag_configure('strong', foreground='#00ff00')
        
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.component_tree.yview)
        self.component_tree.configure(yscrollcommand=scrollbar.set)
        
        self.component_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
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
                
                self.frame.after(0, lambda: self._update_sector_ui(sector_rankings))
            except Exception as e:
                logger.error(f"Error loading sector data: {e}")
        
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
    
    def _update_sector_ui(self, sector_rankings: List[Dict]):
        """Update rich sector rotation UI."""
        if not sector_rankings:
            return
        
        # ===== Update Rotation Cycle =====
        leading = sector_rankings[0]
        lagging = sector_rankings[-1]
        spread = leading['avg_1d'] - lagging['avg_1d']
        
        self.cycle_labels['leading'].config(text=f"{leading['sector']} ({leading['avg_1d']:+.1f}%)")
        self.cycle_labels['lagging'].config(text=f"{lagging['sector']} ({lagging['avg_1d']:+.1f}%)")
        self.cycle_labels['spread'].config(text=f"{spread:.2f}%")
        
        # ===== Update Sector Leaders =====
        for i, sr in enumerate(sector_rankings[:5]):
            if i < len(self.sector_leader_labels):
                sector_lbl, leader_lbl, chg_lbl = self.sector_leader_labels[i]
                sector_lbl.config(text=sr['sector'][:11])
                
                leader = sr.get('leader')
                if leader:
                    leader_lbl.config(text=leader['symbol'][:9])
                    chg = leader['chg_1d']
                    chg_lbl.config(text=f"{chg:+.1f}%",
                                  foreground=COLORS['gain'] if chg > 0 else COLORS['loss'])
        
        # ===== Draw Sector Heatmap =====
        self._draw_sector_heatmap(sector_rankings)
        
        # ===== Update Sector Rankings Table =====
        for item in self.sector_tree.get_children():
            self.sector_tree.delete(item)
        
        for sr in sector_rankings:
            net_score = sr.get('net_score', 0)
            
            # Determine tag based on net score
            if net_score >= 3:
                tag = 'hot'
            elif net_score > 0:
                tag = 'gain'
            elif net_score <= -3:
                tag = 'cold'
            elif net_score < 0:
                tag = 'loss'
            else:
                tag = ''
            
            # Store full sector name as iid for selection
            self.sector_tree.insert('', 'end', iid=sr['sector'], values=(
                sr['sector'][:15],
                f"{sr['avg_1d']:+.1f}%",
                f"{sr['avg_1w']:+.1f}%",
                f"{sr.get('avg_1m', 0):+.1f}%",
                f"{net_score:+.1f}",
                sr['count']
            ), tags=(tag,))
        
        # Store for component loading
        self._sector_data = {sr['sector']: sr for sr in sector_rankings}
    
    def _draw_sector_heatmap(self, sector_rankings: List[Dict]):
        """Draw sector performance heatmap."""
        self.sector_heatmap_canvas.delete("all")
        
        width = self.sector_heatmap_canvas.winfo_width()
        height = self.sector_heatmap_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width, height = 400, 140
        
        if not sector_rankings:
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
        rows = min(3, (len(sector_rankings) + cols - 1) // cols)
        
        cell_w = width // cols
        cell_h = height // rows
        
        for i, sr in enumerate(sector_rankings[:12]):
            row = i // cols
            col = i % cols
            
            x1 = col * cell_w + 2
            y1 = row * cell_h + 2
            x2 = x1 + cell_w - 4
            y2 = y1 + cell_h - 4
            
            avg = sr['avg_1d']
            if avg > 2:
                color = '#1a8f3c'
            elif avg > 0:
                color = '#2ecc71'
            elif avg < -2:
                color = '#922b21'
            elif avg < 0:
                color = '#e74c3c'
            else:
                color = '#555555'
            
            self.sector_heatmap_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='#333')
            
            name = sector_abbrev.get(sr['sector'], sr['sector'][:7])
            self.sector_heatmap_canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2 - 8,
                text=name, fill="white", font=('Arial', 8)
            )
            
            self.sector_heatmap_canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 10,
                text=f"{avg:+.1f}%", fill="white", font=('Arial', 10, 'bold')
            )
    
    def _on_sector_select(self, event):
        """Handle sector selection."""
        selection = self.sector_tree.selection()
        if not selection:
            return
        
        # Use the iid which stores the full sector name
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
                f"{s.get('weight', 0):.1f}%",
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
                
                self.frame.after(0, lambda: self._update_flow_ui(stocks))
            except Exception as e:
                logger.error(f"Error loading flow data: {e}")
        
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
    
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
    
    def _load_smart_money_data(self):
        """Load smart money analysis data."""
        def fetch():
            try:
                all_stocks = self.collector.get_all_stocks()
                stocks_list = all_stocks.to_dict('records') if not all_stocks.empty else []
                
                # Run smart money analysis
                analysis = self.smart_money_detector.analyze_stocks(stocks_list)
                alerts = self.anomaly_scanner.scan(stocks_list)
                
                self.frame.after(0, lambda: self._update_smart_money_ui(analysis, alerts))
            except Exception as e:
                logger.error(f"Error loading smart money data: {e}")
        
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
    
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
    
    def refresh(self):
        """Refresh all data."""
        self.live_market_tab.refresh()
        self._load_sector_data()
        self._load_flow_data()
        self._load_smart_money_data()
