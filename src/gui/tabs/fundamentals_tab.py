"""
Fundamentals Tab - Comprehensive fundamental analysis for NGX stocks.

Super Enhanced with:
- Overview (Key metrics, valuation assessment)
- Sector Analysis (Comparison, percentile rankings)
- Peer Comparison (Side-by-side table)
- Fair Value (Intrinsic value calculator)
- Dividends (Yield analysis)
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

from src.database.db_manager import DatabaseManager
from src.gui.theme import COLORS, get_font
from src.collectors.tradingview_collector import TradingViewCollector

logger = logging.getLogger(__name__)


class FundamentalsTab:
    """
    Standalone Fundamentals Tab with comprehensive analysis features.
    """
    
    # NGX Sector Classifications
    SECTORS = {
        'Banking': ['ZENITHBANK', 'GTCO', 'UBA', 'ACCESSCORP', 'FBNH', 'STANBIC', 'FCMB', 'FIDELITYBK', 'WEMABANK', 'STERLINGNG'],
        'Consumer Goods': ['NESTLE', 'DANGSUGAR', 'FLOURMILL', 'UNILEVER', 'CADBURY', 'VITAFOAM', 'NASCON', 'HONYFLOUR', 'MCNICHOLS'],
        'Industrial': ['DANGCEM', 'WAPCO', 'BUACEMENT', 'BETAGLAS', 'BERGER', 'CAP', 'MEYER', 'CUTIX'],
        'Oil & Gas': ['SEPLAT', 'OANDO', 'TOTALENERG', 'CONOIL', 'ETERNA', 'ARDOVA', 'JAPAULGOLD'],
        'Insurance': ['AIICO', 'AXAMANSARD', 'CUSTODIAN', 'MANSARD', 'NEM', 'CORNERST', 'LASACO', 'LINKASSURE'],
        'Agriculture': ['PRESCO', 'OKOMUOIL', 'LIVESTOCK', 'ELLAHLAKES'],
        'Healthcare': ['GLAXOSMITH', 'MAYBAKER', 'NEIMETH', 'FIDSON', 'PHARMDEKO'],
        'Telecom': ['MTNN', 'AIRTELAFRI'],
        'Conglomerate': ['TRANSCORP', 'UACN', 'JOHNHOLT', 'SCOA', 'PZ'],
    }
    
    def __init__(self, parent: ttk.Frame, db: DatabaseManager):
        """
        Initialize the Fundamentals tab.
        
        Args:
            parent: Parent frame
            db: Database manager instance
        """
        self.parent = parent
        self.db = db
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.current_symbol = None
        self.all_stocks_data = None
        self.collector = TradingViewCollector()
        
        self._create_header()
        self._create_sub_notebook()
        
        # Initial load
        self._load_all_stocks()
    
    def _create_header(self):
        """Create header with symbol selector."""
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=15, pady=10)
        
        # Title
        ttk.Label(
            header,
            text="💰 Fundamentals",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        # Symbol selector
        selector_frame = ttk.Frame(header)
        selector_frame.pack(side=tk.RIGHT)
        
        ttk.Label(
            selector_frame,
            text="Symbol:",
            font=get_font('small')
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.symbol_var = tk.StringVar(value="DANGCEM")
        self.symbol_combo = ttk.Combobox(
            selector_frame,
            textvariable=self.symbol_var,
            width=12,
            state='readonly'
        )
        self.symbol_combo.pack(side=tk.LEFT)
        self.symbol_combo.bind('<<ComboboxSelected>>', lambda e: self._on_symbol_change())
        
        # Refresh button
        ttk.Button(
            selector_frame,
            text="🔄",
            width=3,
            command=self._refresh_data
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # Status label
        self.status_label = ttk.Label(
            header,
            text="",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.status_label.pack(side=tk.RIGHT, padx=(0, 20))
    
    def _create_sub_notebook(self):
        """Create sub-notebook with 5 tabs."""
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Overview
        self.overview_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.overview_tab, text="📊 Overview")
        self._create_overview_tab()
        
        # Tab 2: Sector Analysis
        self.sector_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.sector_tab, text="📈 Sector")
        self._create_sector_tab()
        
        # Tab 3: Peer Comparison
        self.peers_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.peers_tab, text="👥 Peers")
        self._create_peers_tab()
        
        # Tab 4: Fair Value
        self.fairvalue_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.fairvalue_tab, text="💎 Fair Value")
        self._create_fairvalue_tab()
        
        # Tab 5: Dividends
        self.dividends_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.dividends_tab, text="💵 Dividends")
        self._create_dividends_tab()
        
        # Tab 6: P/E History Chart
        self.pe_history_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.pe_history_tab, text="📈 P/E")
        self._create_valuation_chart_tab('pe')
        
        # Tab 7: P/B History Chart
        self.pb_history_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.pb_history_tab, text="📊 P/B")
        self._create_valuation_chart_tab('pb')
        
        # Tab 8: P/S History Chart
        self.ps_history_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.ps_history_tab, text="📉 P/S")
        self._create_valuation_chart_tab('ps')
    
    # =========================================================================
    # OVERVIEW TAB
    # =========================================================================
    
    def _create_overview_tab(self):
        """Create Super Enhanced Overview sub-tab with comprehensive metrics."""
        # Use scrollable frame for all content
        canvas = tk.Canvas(self.overview_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.overview_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Store canvas reference for width binding
        self._overview_canvas = canvas
        self._overview_scrollable = scrollable_frame
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window and store its ID for width updates
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Bind canvas resize to update frame width
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = scrollable_frame
        
        # =====================================================================
        # ROW 1: Stock Header with Price
        # =====================================================================
        header_frame = ttk.LabelFrame(main_frame, text="📌 Stock Info")
        header_frame.pack(fill=tk.X, padx=10, pady=(5, 8))
        
        header_inner = ttk.Frame(header_frame)
        header_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.stock_name_label = ttk.Label(
            header_inner,
            text="Loading...",
            font=get_font('heading'),
            foreground=COLORS['primary']
        )
        self.stock_name_label.pack(side=tk.LEFT)
        
        # Price and change on right
        price_frame = ttk.Frame(header_inner)
        price_frame.pack(side=tk.RIGHT)
        
        self.stock_price_label = ttk.Label(
            price_frame,
            text="₦0.00",
            font=get_font('heading')
        )
        self.stock_price_label.pack(side=tk.RIGHT)
        
        self.stock_change_label = ttk.Label(
            price_frame,
            text="0.00%",
            font=get_font('body')
        )
        self.stock_change_label.pack(side=tk.RIGHT, padx=(0, 15))
        
        # =====================================================================
        # ROW 2: 52-Week Range Bar
        # =====================================================================
        range_frame = ttk.LabelFrame(main_frame, text="📊 52-Week Price Range")
        range_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        range_inner = ttk.Frame(range_frame)
        range_inner.pack(fill=tk.X, padx=15, pady=10)
        
        # 52W Low label
        self.range_low_label = ttk.Label(
            range_inner,
            text="52W Low\n₦--",
            font=get_font('tiny'),
            foreground=COLORS['loss'],
            justify='center'
        )
        self.range_low_label.pack(side=tk.LEFT)
        
        # Range bar canvas
        self.range_canvas = tk.Canvas(
            range_inner,
            height=30,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.range_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # 52W High label
        self.range_high_label = ttk.Label(
            range_inner,
            text="52W High\n₦--",
            font=get_font('tiny'),
            foreground=COLORS['gain'],
            justify='center'
        )
        self.range_high_label.pack(side=tk.RIGHT)
        
        # =====================================================================
        # ROW 3: Key Metrics Grid (3 rows × 4 cols = 12 cards)
        # =====================================================================
        metrics_frame = ttk.LabelFrame(main_frame, text="📈 Key Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        metrics_inner = ttk.Frame(metrics_frame)
        metrics_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(4):
            metrics_inner.columnconfigure(i, weight=1)
        
        self.metric_cards = {}
        
        # Row 1: Valuation Metrics
        valuation_metrics = [
            ('pe_ratio', '📊 P/E', '--'),
            ('pb_ratio', '📈 P/B', '--'),
            ('ps_ratio', '📉 P/S', '--'),
            ('ev_ebitda', '💹 EV/EBITDA', '--'),
        ]
        
        # Row 2: Profitability Metrics
        profit_metrics = [
            ('eps', '💵 EPS', '₦--'),
            ('roe', '📊 ROE', '--%'),
            ('net_margin', '📈 Net Margin', '--%'),
            ('dividend', '🎁 Div Yield', '--%'),
        ]
        
        # Row 3: Size & Liquidity
        size_metrics = [
            ('market_cap', '💰 Market Cap', '₦--'),
            ('high_52w', '📈 52W High', '₦--'),
            ('low_52w', '📉 52W Low', '₦--'),
            ('volume', '📊 Avg Volume', '--'),
        ]
        
        all_metrics = [
            (valuation_metrics, 0),
            (profit_metrics, 1),
            (size_metrics, 2)
        ]
        
        for metrics_list, row in all_metrics:
            for col, (key, label, default) in enumerate(metrics_list):
                card = ttk.Frame(metrics_inner, relief='groove', borderwidth=1)
                card.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
                
                ttk.Label(card, text=label, font=get_font('tiny'),
                         foreground=COLORS['primary']).pack(anchor='center', pady=(4, 0))
                
                value_label = ttk.Label(card, text=default, font=get_font('subheading'))
                value_label.pack(anchor='center', pady=(2, 4))
                
                self.metric_cards[key] = value_label
        
        # =====================================================================
        # ROW 4: Price Performance
        # =====================================================================
        perf_frame = ttk.LabelFrame(main_frame, text="📈 Price Performance")
        perf_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        perf_inner = ttk.Frame(perf_frame)
        perf_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(6):
            perf_inner.columnconfigure(i, weight=1)
        
        self.perf_labels = {}
        perf_items = [
            ('week', '1W'), ('month', '1M'), ('quarter', '3M'),
            ('half_year', '6M'), ('ytd', 'YTD'), ('year', '1Y')
        ]
        
        for i, (key, label) in enumerate(perf_items):
            card = ttk.Frame(perf_inner)
            card.grid(row=0, column=i, padx=5, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            value = ttk.Label(card, text="--%", font=get_font('subheading'))
            value.pack(anchor='center')
            self.perf_labels[key] = value
        
        # =====================================================================
        # ROW 5: Valuation Score Gauge
        # =====================================================================
        gauge_frame = ttk.LabelFrame(main_frame, text="🎯 Valuation Score")
        gauge_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        gauge_inner = ttk.Frame(gauge_frame)
        gauge_inner.pack(fill=tk.X, padx=15, pady=10)
        
        # Gauge canvas
        self.gauge_canvas = tk.Canvas(
            gauge_inner,
            height=50,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.gauge_canvas.pack(fill=tk.X, pady=(0, 5))
        
        self.valuation_status = ttk.Label(
            gauge_inner,
            text="⏳ Calculating...",
            font=get_font('body'),
            foreground=COLORS['text_muted']
        )
        self.valuation_status.pack(anchor='center')
        
        self.valuation_details = ttk.Label(
            gauge_inner,
            text="",
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        )
        self.valuation_details.pack(anchor='center', pady=(3, 0))
        
        # =====================================================================
        # ROW 6: Quick Insights Panel
        # =====================================================================
        insights_frame = ttk.LabelFrame(main_frame, text="💡 Quick Insights")
        insights_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        insights_inner = ttk.Frame(insights_frame)
        insights_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.insights_labels = []
        for i in range(4):
            insight = ttk.Label(
                insights_inner,
                text="",
                font=get_font('body'),
                foreground=COLORS['text_secondary']
            )
            insight.pack(anchor='w', pady=2)
            self.insights_labels.append(insight)
        
        # =====================================================================
        # ROW 7: Technical Summary
        # =====================================================================
        tech_frame = ttk.LabelFrame(main_frame, text="📊 Technical Summary")
        tech_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        tech_inner = ttk.Frame(tech_frame)
        tech_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(6):
            tech_inner.columnconfigure(i, weight=1)
        
        self.tech_labels = {}
        tech_items = [
            ('rsi', 'RSI'),
            ('macd', 'MACD'),
            ('sma20', 'SMA20'),
            ('sma50', 'SMA50'),
            ('sma200', 'SMA200'),
            ('rec', 'Signal')
        ]
        
        for i, (key, label) in enumerate(tech_items):
            card = ttk.Frame(tech_inner)
            card.grid(row=0, column=i, padx=5, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            value = ttk.Label(card, text="--", font=get_font('body'))
            value.pack(anchor='center')
            self.tech_labels[key] = value
        
        # =====================================================================
        # ROW 8: Sector Position
        # =====================================================================
        sector_frame = ttk.LabelFrame(main_frame, text="🏢 Sector Position")
        sector_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        sector_inner = ttk.Frame(sector_frame)
        sector_inner.pack(fill=tk.X, padx=15, pady=8)
        
        self.overview_sector_label = ttk.Label(
            sector_inner,
            text="Sector: --",
            font=get_font('body'),
            foreground=COLORS['primary']
        )
        self.overview_sector_label.pack(side=tk.LEFT)
        
        self.sector_rank_label = ttk.Label(
            sector_inner,
            text="Rank: --",
            font=get_font('body'),
            foreground=COLORS['text_secondary']
        )
        self.sector_rank_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.sector_pe_compare = ttk.Label(
            sector_inner,
            text="P/E vs Sector: --",
            font=get_font('body'),
            foreground=COLORS['text_secondary']
        )
        self.sector_pe_compare.pack(side=tk.RIGHT)
    
    # =========================================================================
    # SECTOR ANALYSIS TAB (SUPER ENHANCED)
    # =========================================================================
    
    def _create_sector_tab(self):
        """Create Super Enhanced Sector Analysis sub-tab."""
        # Scrollable frame
        canvas = tk.Canvas(self.sector_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.sector_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = scrollable_frame
        
        # =====================================================================
        # ROW 1: Sector Header
        # =====================================================================
        header_frame = ttk.LabelFrame(main_frame, text="📌 Sector Classification")
        header_frame.pack(fill=tk.X, padx=10, pady=(5, 8))
        
        header_inner = ttk.Frame(header_frame)
        header_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.sector_name_label = ttk.Label(
            header_inner,
            text="Sector: --",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        )
        self.sector_name_label.pack(side=tk.LEFT)
        
        self.sector_stocks_label = ttk.Label(
            header_inner,
            text="(0 stocks)",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.sector_stocks_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # =====================================================================
        # ROW 2: Sector Statistics Summary
        # =====================================================================
        stats_frame = ttk.LabelFrame(main_frame, text="📊 Sector Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        stats_inner = ttk.Frame(stats_frame)
        stats_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(6):
            stats_inner.columnconfigure(i, weight=1)
        
        self.sector_stats = {}
        stats_items = [
            ('total_mcap', '💰 Total MCap'),
            ('avg_pe', '📊 Avg P/E'),
            ('avg_div', '🎁 Avg Yield'),
            ('gainers', '📈 Gainers'),
            ('losers', '📉 Losers'),
            ('avg_change', '📊 Avg Change')
        ]
        
        for i, (key, label) in enumerate(stats_items):
            card = ttk.Frame(stats_inner)
            card.grid(row=0, column=i, padx=5, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            value = ttk.Label(card, text="--", font=get_font('body'))
            value.pack(anchor='center')
            self.sector_stats[key] = value
        
        # =====================================================================
        # ROW 3: Leaders & Laggards
        # =====================================================================
        leaders_frame = ttk.LabelFrame(main_frame, text="🏆 Sector Leaders & Laggards")
        leaders_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        leaders_inner = ttk.Frame(leaders_frame)
        leaders_inner.pack(fill=tk.X, padx=10, pady=8)
        
        leaders_inner.columnconfigure(0, weight=1)
        leaders_inner.columnconfigure(1, weight=1)
        
        # Top 3 Leaders (left)
        leaders_left = ttk.Frame(leaders_inner)
        leaders_left.grid(row=0, column=0, sticky='nsew', padx=5)
        
        ttk.Label(leaders_left, text="🟢 TOP PERFORMERS",
                 font=get_font('small'), foreground=COLORS['gain']).pack(anchor='w')
        
        self.leader_labels = []
        for i in range(3):
            lbl = ttk.Label(leaders_left, text=f"{i+1}. --",
                          font=get_font('body'), foreground=COLORS['gain'])
            lbl.pack(anchor='w', pady=1)
            self.leader_labels.append(lbl)
        
        # Bottom 3 Laggards (right)
        leaders_right = ttk.Frame(leaders_inner)
        leaders_right.grid(row=0, column=1, sticky='nsew', padx=5)
        
        ttk.Label(leaders_right, text="🔴 BOTTOM PERFORMERS",
                 font=get_font('small'), foreground=COLORS['loss']).pack(anchor='w')
        
        self.laggard_labels = []
        for i in range(3):
            lbl = ttk.Label(leaders_right, text=f"{i+1}. --",
                          font=get_font('body'), foreground=COLORS['loss'])
            lbl.pack(anchor='w', pady=1)
            self.laggard_labels.append(lbl)
        
        # =====================================================================
        # ROW 4: Stock vs Sector Comparison Bars
        # =====================================================================
        compare_frame = ttk.LabelFrame(main_frame, text="📊 Stock vs Sector Average")
        compare_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        compare_inner = ttk.Frame(compare_frame)
        compare_inner.pack(fill=tk.X, padx=15, pady=10)
        
        # Canvas for comparison bars
        self.sector_compare_canvas = tk.Canvas(
            compare_inner,
            height=120,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.sector_compare_canvas.pack(fill=tk.X)
        
        self.sector_compare = {}  # Will store data for comparisons
        
        # =====================================================================
        # ROW 5: Sector Percentile Gauge
        # =====================================================================
        rank_frame = ttk.LabelFrame(main_frame, text="🎯 Sector Ranking")
        rank_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        rank_inner = ttk.Frame(rank_frame)
        rank_inner.pack(fill=tk.X, padx=15, pady=10)
        
        # Gauge canvas
        self.sector_gauge_canvas = tk.Canvas(
            rank_inner,
            height=50,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.sector_gauge_canvas.pack(fill=tk.X, pady=(0, 5))
        
        self.percentile_label = ttk.Label(
            rank_inner,
            text="Outperforms 0% of sector peers",
            font=get_font('body')
        )
        self.percentile_label.pack(anchor='center')
        
        # =====================================================================
        # ROW 6: Enhanced Sector Stocks Table
        # =====================================================================
        table_frame = ttk.LabelFrame(main_frame, text="📋 Sector Stocks")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ('symbol', 'name', 'price', 'change', 'pe', 'div', 'mcap', 'signal')
        self.sector_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        
        col_config = [
            ('symbol', 'Symbol', 70, 'center'),
            ('name', 'Name', 120, 'w'),
            ('price', 'Price', 70, 'e'),
            ('change', 'Day %', 60, 'center'),
            ('pe', 'P/E', 55, 'center'),
            ('div', 'Div %', 55, 'center'),
            ('mcap', 'MCap', 80, 'e'),
            ('signal', 'Signal', 70, 'center')
        ]
        
        for col, heading, width, anchor in col_config:
            self.sector_tree.heading(col, text=heading)
            self.sector_tree.column(col, width=width, anchor=anchor)
        
        sector_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.sector_tree.yview)
        self.sector_tree.configure(yscrollcommand=sector_scroll.set)
        
        self.sector_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        sector_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        
        self.sector_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.sector_tree.tag_configure('loss', foreground=COLORS['loss'])
        self.sector_tree.tag_configure('current', background=COLORS['bg_medium'])
    # =========================================================================
    # PEER COMPARISON TAB (SUPER ENHANCED)
    # =========================================================================
    
    def _create_peers_tab(self):
        """Create Super Enhanced Peer Comparison sub-tab."""
        # Scrollable frame
        canvas = tk.Canvas(self.peers_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.peers_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = scrollable_frame
        
        # =====================================================================
        # ROW 1: Peer Metrics Summary
        # =====================================================================
        summary_frame = ttk.LabelFrame(main_frame, text="📊 Your Stock vs Peer Average")
        summary_frame.pack(fill=tk.X, padx=10, pady=(5, 8))
        
        summary_inner = ttk.Frame(summary_frame)
        summary_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(5):
            summary_inner.columnconfigure(i, weight=1)
        
        self.peer_summary = {}
        summary_items = [
            ('pe', 'P/E'),
            ('div', 'Div %'),
            ('ytd', 'YTD %'),
            ('day', 'Day %'),
            ('mcap', 'MCap')
        ]
        
        for i, (key, label) in enumerate(summary_items):
            card = ttk.Frame(summary_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=3, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center', pady=(4, 0))
            
            # Your Stock
            stock_lbl = ttk.Label(card, text="You: --", font=get_font('small'))
            stock_lbl.pack(anchor='center')
            
            # Peer Avg
            avg_lbl = ttk.Label(card, text="Avg: --", font=get_font('small'),
                               foreground=COLORS['text_muted'])
            avg_lbl.pack(anchor='center', pady=(0, 4))
            
            self.peer_summary[key] = {'stock': stock_lbl, 'avg': avg_lbl}
        
        # =====================================================================
        # ROW 2: Metric Rankings
        # =====================================================================
        rank_frame = ttk.LabelFrame(main_frame, text="🏆 Your Metric Rankings")
        rank_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        rank_inner = ttk.Frame(rank_frame)
        rank_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(6):
            rank_inner.columnconfigure(i, weight=1)
        
        self.peer_rankings = {}
        rank_items = [
            ('pe_rank', 'P/E Rank'),
            ('div_rank', 'Div Rank'),
            ('ytd_rank', 'YTD Rank'),
            ('mcap_rank', 'MCap Rank'),
            ('eps_rank', 'EPS Rank'),
            ('change_rank', 'Day Rank')
        ]
        
        for i, (key, label) in enumerate(rank_items):
            card = ttk.Frame(rank_inner)
            card.grid(row=0, column=i, padx=5, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            value = ttk.Label(card, text="#--", font=get_font('body'))
            value.pack(anchor='center')
            self.peer_rankings[key] = value
        
        # =====================================================================
        # ROW 3: Peer Insights
        # =====================================================================
        insight_frame = ttk.LabelFrame(main_frame, text="💡 Peer Insights")
        insight_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        insight_inner = ttk.Frame(insight_frame)
        insight_inner.pack(fill=tk.X, padx=15, pady=8)
        
        self.peer_insight_labels = []
        for i in range(4):
            lbl = ttk.Label(insight_inner, text="", font=get_font('body'),
                          foreground=COLORS['text_secondary'])
            lbl.pack(anchor='w', pady=1)
            self.peer_insight_labels.append(lbl)
        
        # =====================================================================
        # ROW 4: Enhanced Peer Table
        # =====================================================================
        table_frame = ttk.LabelFrame(main_frame, text="👥 Peer Comparison Table")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ('symbol', 'name', 'price', 'change', 'pe', 'pb', 'eps', 'div', 'mcap', 'ytd', 'signal')
        self.peers_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
        
        col_config = [
            ('symbol', 'Symbol', 65, 'center'),
            ('name', 'Name', 100, 'w'),
            ('price', 'Price', 65, 'e'),
            ('change', 'Day %', 55, 'center'),
            ('pe', 'P/E', 50, 'center'),
            ('pb', 'P/B', 50, 'center'),
            ('eps', 'EPS', 55, 'e'),
            ('div', 'Div %', 50, 'center'),
            ('mcap', 'MCap', 70, 'e'),
            ('ytd', 'YTD %', 55, 'center'),
            ('signal', 'Signal', 65, 'center')
        ]
        
        for col, heading, width, anchor in col_config:
            self.peers_tree.heading(col, text=heading)
            self.peers_tree.column(col, width=width, anchor=anchor)
        
        peers_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.peers_tree.yview)
        self.peers_tree.configure(yscrollcommand=peers_scroll.set)
        
        self.peers_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        peers_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        
        self.peers_tree.tag_configure('current', background=COLORS['bg_medium'])
        self.peers_tree.tag_configure('best', foreground=COLORS['gain'])
        self.peers_tree.tag_configure('worst', foreground=COLORS['loss'])
    # =========================================================================
    # FAIR VALUE TAB (SUPER ENHANCED)
    # =========================================================================
    
    def _create_fairvalue_tab(self):
        """Create Super Enhanced Fair Value Calculator sub-tab."""
        # Scrollable frame
        canvas = tk.Canvas(self.fairvalue_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.fairvalue_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = scrollable_frame
        
        # =====================================================================
        # ROW 1: Fair Value Summary with Visual Gauge
        # =====================================================================
        summary_frame = ttk.LabelFrame(main_frame, text="💎 Fair Value Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=(5, 8))
        
        summary_inner = ttk.Frame(summary_frame)
        summary_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(4):
            summary_inner.columnconfigure(i, weight=1)
        
        self.fv_cards = {}
        fv_items = [
            ('current_price', '📍 Current', '₦0'),
            ('fair_value', '💎 Fair Value', '₦0'),
            ('upside', '📊 Upside', '0%'),
            ('margin_safety', '🛡️ Margin', '0%')
        ]
        
        for i, (key, label, default) in enumerate(fv_items):
            card = ttk.Frame(summary_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=4, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(4, 0))
            
            value = ttk.Label(card, text=default, font=get_font('subheading'))
            value.pack(anchor='center', pady=(2, 4))
            self.fv_cards[key] = value
        
        # Fair Value Gauge
        self.fv_gauge_canvas = tk.Canvas(
            summary_frame,
            height=40,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.fv_gauge_canvas.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        # =====================================================================
        # ROW 2: Valuation Methods (6 methods)
        # =====================================================================
        methods_frame = ttk.LabelFrame(main_frame, text="📐 Valuation Methods")
        methods_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        methods_inner = ttk.Frame(methods_frame)
        methods_inner.pack(fill=tk.X, padx=10, pady=8)
        
        self.method_labels = {}
        methods_config = [
            ('graham', '📚 Graham Number', 'sqrt(22.5 × EPS × BV)'),
            ('pe_sector', '📊 P/E (Sector)', 'EPS × Sector Avg P/E'),
            ('pe_fair', '📈 P/E (Fair=15)', 'EPS × 15'),
            ('ddm', '💵 DDM', 'Div / (r - g)'),
            ('book', '📖 Book Value', 'Book Value per Share'),
            ('epv', '⚡ Earnings Power', 'EPS × 10')
        ]
        
        for key, title, formula in methods_config:
            row = ttk.Frame(methods_inner)
            row.pack(fill=tk.X, pady=3)
            
            ttk.Label(row, text=title, font=get_font('body'),
                     foreground=COLORS['primary']).pack(side=tk.LEFT)
            
            ttk.Label(row, text=f"({formula})", font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=(5, 0))
            
            value = ttk.Label(row, text="₦--", font=get_font('body'))
            value.pack(side=tk.RIGHT)
            self.method_labels[key] = value
        
        # =====================================================================
        # ROW 3: Valuation Range Chart
        # =====================================================================
        range_frame = ttk.LabelFrame(main_frame, text="📊 Valuation Range")
        range_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        self.fv_range_canvas = tk.Canvas(
            range_frame,
            height=60,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.fv_range_canvas.pack(fill=tk.X, padx=15, pady=10)
        
        # =====================================================================
        # ROW 4: Confidence & Data Quality
        # =====================================================================
        conf_frame = ttk.LabelFrame(main_frame, text="🎯 Confidence Score")
        conf_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        conf_inner = ttk.Frame(conf_frame)
        conf_inner.pack(fill=tk.X, padx=15, pady=8)
        
        self.fv_confidence_label = ttk.Label(
            conf_inner,
            text="Confidence: --",
            font=get_font('body')
        )
        self.fv_confidence_label.pack(side=tk.LEFT)
        
        self.fv_methods_used = ttk.Label(
            conf_inner,
            text="Methods used: 0/6",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.fv_methods_used.pack(side=tk.RIGHT)
        
        # =====================================================================
        # ROW 5: Investment Recommendation
        # =====================================================================
        rec_frame = ttk.LabelFrame(main_frame, text="💡 Investment Recommendation")
        rec_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        rec_inner = ttk.Frame(rec_frame)
        rec_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.recommendation_label = ttk.Label(
            rec_inner,
            text="⏳ Calculating...",
            font=get_font('heading'),
            foreground=COLORS['text_muted']
        )
        self.recommendation_label.pack(anchor='center')
        
        self.rec_details_label = ttk.Label(
            rec_inner,
            text="",
            font=get_font('body'),
            foreground=COLORS['text_secondary']
        )
        self.rec_details_label.pack(anchor='center', pady=(5, 0))
        
        # Key considerations
        self.rec_considerations = ttk.Label(
            rec_inner,
            text="",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.rec_considerations.pack(anchor='center', pady=(5, 0))
    # =========================================================================
    # DIVIDENDS TAB (SUPER ENHANCED)
    # =========================================================================
    
    def _create_dividends_tab(self):
        """Create Super Enhanced Dividends Analysis sub-tab."""
        # Scrollable frame
        canvas = tk.Canvas(self.dividends_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.dividends_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = scrollable_frame
        
        # =====================================================================
        # ROW 1: Dividend Summary (6 cards)
        # =====================================================================
        div_frame = ttk.LabelFrame(main_frame, text="💵 Dividend Summary")
        div_frame.pack(fill=tk.X, padx=10, pady=(5, 8))
        
        div_inner = ttk.Frame(div_frame)
        div_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(6):
            div_inner.columnconfigure(i, weight=1)
        
        self.div_cards = {}
        div_items = [
            ('yield', '📊 Yield'),
            ('annual_div', '💰 Annual/Share'),
            ('sector_yield', '📈 Sector Avg'),
            ('vs_sector', '⚡ vs Sector'),
            ('rank', '🏆 Rank'),
            ('status', '🎯 Status')
        ]
        
        for i, (key, label) in enumerate(div_items):
            card = ttk.Frame(div_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=3, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(4, 0))
            
            value = ttk.Label(card, text="--", font=get_font('body'))
            value.pack(anchor='center', pady=(2, 4))
            self.div_cards[key] = value
        
        # =====================================================================
        # ROW 2: Income Calculator
        # =====================================================================
        calc_frame = ttk.LabelFrame(main_frame, text="🧮 Income Calculator")
        calc_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        calc_inner = ttk.Frame(calc_frame)
        calc_inner.pack(fill=tk.X, padx=15, pady=10)
        
        # Investment input
        input_frame = ttk.Frame(calc_inner)
        input_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(input_frame, text="Investment (₦):", font=get_font('body')).pack(side=tk.LEFT)
        
        self.div_investment_var = tk.StringVar(value="1000000")
        self.div_investment_entry = ttk.Entry(input_frame, textvariable=self.div_investment_var, width=15)
        self.div_investment_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        calc_btn = ttk.Button(input_frame, text="Calculate", command=self._calculate_dividend_income)
        calc_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Results
        result_frame = ttk.Frame(calc_inner)
        result_frame.pack(fill=tk.X)
        
        for i in range(4):
            result_frame.columnconfigure(i, weight=1)
        
        self.div_calc_results = {}
        calc_items = [
            ('shares', '📈 Shares'),
            ('annual', '💵 Annual Income'),
            ('monthly', '📅 Monthly'),
            ('yield_cost', '📊 Yield on Cost')
        ]
        
        for i, (key, label) in enumerate(calc_items):
            card = ttk.Frame(result_frame)
            card.grid(row=0, column=i, padx=5, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            value = ttk.Label(card, text="--", font=get_font('body'))
            value.pack(anchor='center')
            self.div_calc_results[key] = value
        
        # =====================================================================
        # ROW 3: Dividend Quality Gauge
        # =====================================================================
        quality_frame = ttk.LabelFrame(main_frame, text="🎯 Dividend Quality")
        quality_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        self.div_gauge_canvas = tk.Canvas(
            quality_frame,
            height=45,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.div_gauge_canvas.pack(fill=tk.X, padx=15, pady=8)
        
        self.div_quality_label = ttk.Label(
            quality_frame,
            text="Quality: --",
            font=get_font('body')
        )
        self.div_quality_label.pack(anchor='center', pady=(0, 8))
        
        # =====================================================================
        # ROW 4: Top Dividend Payers & Insights
        # =====================================================================
        top_frame = ttk.LabelFrame(main_frame, text="🏆 Top Sector Dividend Payers")
        top_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        top_inner = ttk.Frame(top_frame)
        top_inner.pack(fill=tk.X, padx=10, pady=8)
        
        top_inner.columnconfigure(0, weight=1)
        top_inner.columnconfigure(1, weight=1)
        
        # Top 3 payers (left)
        top_left = ttk.Frame(top_inner)
        top_left.grid(row=0, column=0, sticky='nsew', padx=5)
        
        ttk.Label(top_left, text="🥇 TOP 3 PAYERS",
                 font=get_font('small'), foreground=COLORS['gain']).pack(anchor='w')
        
        self.top_div_labels = []
        for i in range(3):
            lbl = ttk.Label(top_left, text=f"{i+1}. --",
                          font=get_font('body'), foreground=COLORS['gain'])
            lbl.pack(anchor='w', pady=1)
            self.top_div_labels.append(lbl)
        
        # Insights (right)
        top_right = ttk.Frame(top_inner)
        top_right.grid(row=0, column=1, sticky='nsew', padx=5)
        
        ttk.Label(top_right, text="💡 INSIGHTS",
                 font=get_font('small'), foreground=COLORS['primary']).pack(anchor='w')
        
        self.div_insight_labels = []
        for i in range(3):
            lbl = ttk.Label(top_right, text="",
                          font=get_font('body'), foreground=COLORS['text_secondary'])
            lbl.pack(anchor='w', pady=1)
            self.div_insight_labels.append(lbl)
        
        # =====================================================================
        # ROW 5: Enhanced Sector Dividend Table
        # =====================================================================
        table_frame = ttk.LabelFrame(main_frame, text="📋 Sector Dividend Comparison")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ('symbol', 'name', 'price', 'div_yield', 'annual', 'pe', 'signal')
        self.div_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        col_config = [
            ('symbol', 'Symbol', 65, 'center'),
            ('name', 'Name', 100, 'w'),
            ('price', 'Price', 65, 'e'),
            ('div_yield', 'Yield', 55, 'center'),
            ('annual', 'Annual', 60, 'e'),
            ('pe', 'P/E', 50, 'center'),
            ('signal', 'Signal', 65, 'center')
        ]
        
        for col, heading, width, anchor in col_config:
            self.div_tree.heading(col, text=heading)
            self.div_tree.column(col, width=width, anchor=anchor)
        
        div_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.div_tree.yview)
        self.div_tree.configure(yscrollcommand=div_scroll.set)
        
        self.div_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        div_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        
        self.div_tree.tag_configure('current', background=COLORS['bg_medium'])
        self.div_tree.tag_configure('high_yield', foreground=COLORS['gain'])
    
    def _calculate_dividend_income(self):
        """Calculate dividend income from investment."""
        try:
            investment = float(self.div_investment_var.get().replace(',', ''))
            data = self._get_stock_data(self.current_symbol)
            
            if not data:
                return
            
            close = data.get('close', 0) or 0
            div_yield = data.get('dividend_yield_recent', 0) or 0
            
            if close <= 0:
                return
            
            # Calculations
            shares = investment / close
            annual_div_per_share = close * (div_yield / 100)
            annual_income = shares * annual_div_per_share
            monthly_income = annual_income / 12
            
            self.div_calc_results['shares'].config(text=f"{shares:,.0f}")
            self.div_calc_results['annual'].config(text=f"₦{annual_income:,.0f}")
            self.div_calc_results['monthly'].config(text=f"₦{monthly_income:,.0f}")
            self.div_calc_results['yield_cost'].config(text=f"{div_yield:.1f}%")
            
        except ValueError:
            pass
    
    # =========================================================================
    # VALUATION HISTORY CHARTS (P/E, P/B, P/S)
    # =========================================================================
    
    def _create_valuation_chart_tab(self, metric_type: str):
        """Create Super Enhanced valuation ratio tab for P/E, P/B, or P/S."""
        
        # Config for each metric type
        parent_map = {
            'pe': 'pe_history_tab',
            'pb': 'pb_history_tab',
            'ps': 'ps_history_tab'
        }
        
        config = {
            'pe': {
                'title': 'P/E Analysis',
                'metric_name': 'P/E',
                'field': 'price_earnings_ttm',
                'lower_better': True,
                'good_threshold': 15,
                'high_threshold': 25
            },
            'pb': {
                'title': 'P/B Analysis',
                'metric_name': 'P/B',
                'field': 'price_book_ratio',
                'lower_better': True,
                'good_threshold': 1.5,
                'high_threshold': 3
            },
            'ps': {
                'title': 'P/S Analysis',
                'metric_name': 'P/S',
                'field': 'price_sales_ratio',
                'lower_better': True,
                'good_threshold': 2,
                'high_threshold': 5
            }
        }
        
        cfg = config[metric_type]
        parent = getattr(self, parent_map[metric_type])
        
        # Scrollable frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = scrollable_frame
        
        # =====================================================================
        # ROW 1: Ratio Summary Cards
        # =====================================================================
        summary_frame = ttk.LabelFrame(main_frame, text=f"📊 {cfg['metric_name']} Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=(5, 8))
        
        summary_inner = ttk.Frame(summary_frame)
        summary_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(5):
            summary_inner.columnconfigure(i, weight=1)
        
        ratio_cards = {}
        card_items = [
            ('current', f'📍 Current'),
            ('sector_avg', '📈 Sector Avg'),
            ('vs_sector', '⚡ vs Sector'),
            ('rank', '🏆 Rank'),
            ('status', '🎯 Status')
        ]
        
        for i, (key, label) in enumerate(card_items):
            card = ttk.Frame(summary_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=3, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('tiny'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(4, 0))
            
            value = ttk.Label(card, text="--", font=get_font('body'))
            value.pack(anchor='center', pady=(2, 4))
            ratio_cards[key] = value
        
        # =====================================================================
        # ROW 2: Valuation Gauge
        # =====================================================================
        gauge_frame = ttk.LabelFrame(main_frame, text="🎯 Valuation Zone")
        gauge_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        gauge_canvas = tk.Canvas(
            gauge_frame,
            height=45,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        gauge_canvas.pack(fill=tk.X, padx=15, pady=8)
        
        zone_label = ttk.Label(
            gauge_frame,
            text="⏳ Loading...",
            font=get_font('body')
        )
        zone_label.pack(anchor='center', pady=(0, 8))
        
        # =====================================================================
        # ROW 3: Sector Ranking
        # =====================================================================
        rank_frame = ttk.LabelFrame(main_frame, text=f"🏆 Sector {cfg['metric_name']} Ranking")
        rank_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        rank_inner = ttk.Frame(rank_frame)
        rank_inner.pack(fill=tk.X, padx=10, pady=8)
        
        rank_inner.columnconfigure(0, weight=1)
        rank_inner.columnconfigure(1, weight=1)
        
        # Top 3 (left - lowest ratios)
        rank_left = ttk.Frame(rank_inner)
        rank_left.grid(row=0, column=0, sticky='nsew', padx=5)
        
        ttk.Label(rank_left, text=f"🟢 LOWEST {cfg['metric_name']} (Value)",
                 font=get_font('small'), foreground=COLORS['gain']).pack(anchor='w')
        
        top_labels = []
        for i in range(3):
            lbl = ttk.Label(rank_left, text=f"{i+1}. --",
                          font=get_font('body'), foreground=COLORS['gain'])
            lbl.pack(anchor='w', pady=1)
            top_labels.append(lbl)
        
        # Bottom 3 (right - highest ratios)
        rank_right = ttk.Frame(rank_inner)
        rank_right.grid(row=0, column=1, sticky='nsew', padx=5)
        
        ttk.Label(rank_right, text=f"🔴 HIGHEST {cfg['metric_name']} (Expensive)",
                 font=get_font('small'), foreground=COLORS['loss']).pack(anchor='w')
        
        bottom_labels = []
        for i in range(3):
            lbl = ttk.Label(rank_right, text=f"{i+1}. --",
                          font=get_font('body'), foreground=COLORS['loss'])
            lbl.pack(anchor='w', pady=1)
            bottom_labels.append(lbl)
        
        # =====================================================================
        # ROW 4: Insights
        # =====================================================================
        insight_frame = ttk.LabelFrame(main_frame, text=f"💡 {cfg['metric_name']} Insights")
        insight_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        insight_inner = ttk.Frame(insight_frame)
        insight_inner.pack(fill=tk.X, padx=15, pady=8)
        
        insight_labels = []
        for i in range(3):
            lbl = ttk.Label(insight_inner, text="",
                          font=get_font('body'), foreground=COLORS['text_secondary'])
            lbl.pack(anchor='w', pady=1)
            insight_labels.append(lbl)
        
        # =====================================================================
        # ROW 5: Sector Comparison Table
        # =====================================================================
        table_frame = ttk.LabelFrame(main_frame, text=f"📋 Sector {cfg['metric_name']} Comparison")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ('symbol', 'name', 'price', 'ratio', 'vs_sector', 'signal')
        ratio_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        col_config = [
            ('symbol', 'Symbol', 65, 'center'),
            ('name', 'Name', 100, 'w'),
            ('price', 'Price', 70, 'e'),
            ('ratio', cfg['metric_name'], 60, 'center'),
            ('vs_sector', 'vs Avg', 60, 'center'),
            ('signal', 'Signal', 65, 'center')
        ]
        
        for col, heading, width, anchor in col_config:
            ratio_tree.heading(col, text=heading)
            ratio_tree.column(col, width=width, anchor=anchor)
        
        ratio_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=ratio_tree.yview)
        ratio_tree.configure(yscrollcommand=ratio_scroll.set)
        
        ratio_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        ratio_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        
        ratio_tree.tag_configure('current', background=COLORS['bg_medium'])
        ratio_tree.tag_configure('best', foreground=COLORS['gain'])
        ratio_tree.tag_configure('worst', foreground=COLORS['loss'])
        
        # =====================================================================
        # ROW 6: Historical Chart with Std Dev Bands
        # =====================================================================
        chart_frame = ttk.LabelFrame(main_frame, text=f"📈 {cfg['metric_name']} History Chart")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        
        chart_canvas = tk.Canvas(
            chart_frame,
            height=180,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        chart_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Legend
        legend_frame = ttk.Frame(chart_frame)
        legend_frame.pack(fill=tk.X, pady=(0, 5))
        
        legends = [
            (f'{cfg["metric_name"]} Line', COLORS['primary']),
            ('Mean', '#FFFFFF'),
            ('+1σ / -1σ', '#FFD700'),
            ('+2σ / -2σ', '#FF6B6B')
        ]
        
        for text, color in legends:
            frame = ttk.Frame(legend_frame)
            frame.pack(side=tk.LEFT, padx=10)
            tk.Canvas(frame, width=15, height=10, bg=color, highlightthickness=0).pack(side=tk.LEFT)
            ttk.Label(frame, text=text, font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=(3, 0))
        
        # Chart stats
        stats_frame = ttk.Frame(chart_frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        stats_labels = {}
        for key, label in [('avg', 'Avg'), ('std', 'Std Dev'), ('days', 'Days')]:
            frame = ttk.Frame(stats_frame)
            frame.pack(side=tk.LEFT, padx=10)
            ttk.Label(frame, text=f"{label}:", font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            val = ttk.Label(frame, text="--", font=get_font('body'))
            val.pack(side=tk.LEFT, padx=(3, 0))
            stats_labels[key] = val
        
        # Backfill button
        btn_frame = ttk.Frame(chart_frame)
        btn_frame.pack(anchor='center', pady=(5, 10))
        
        backfill_status = ttk.Label(btn_frame, text="", font=get_font('tiny'),
                                    foreground=COLORS['text_muted'])
        
        ttk.Button(
            btn_frame,
            text="📥 Backfill History",
            command=lambda: self._backfill_valuation(metric_type, backfill_status)
        ).pack(side=tk.LEFT)
        
        backfill_status.pack(side=tk.LEFT, padx=(10, 0))
        
        # Store references
        if not hasattr(self, 'valuation_charts'):
            self.valuation_charts = {}
        
        self.valuation_charts[metric_type] = {
            'gauge_canvas': gauge_canvas,
            'chart_canvas': chart_canvas,
            'cards': ratio_cards,
            'zone_label': zone_label,
            'top_labels': top_labels,
            'bottom_labels': bottom_labels,
            'insight_labels': insight_labels,
            'tree': ratio_tree,
            'stats_labels': stats_labels,
            'field': cfg['field'],
            'metric_name': cfg['metric_name'],
            'lower_better': cfg['lower_better'],
            'good_threshold': cfg['good_threshold'],
            'high_threshold': cfg['high_threshold']
        }
    
    def _backfill_valuation(self, metric_type: str, status_label):
        """Backfill valuation history from stored price data."""
        if not self.current_symbol:
            return
        
        try:
            status_label.config(text="⏳ Backfilling...", foreground=COLORS['warning'])
            self.frame.update()
            
            # Get current fundamentals
            data = self._get_stock_data(self.current_symbol)
            if not data:
                status_label.config(text="❌ No fundamental data", foreground=COLORS['loss'])
                return
            
            current_fundamentals = {
                'eps': data.get('earnings_per_share_basic_ttm'),
                'book_value': data.get('book_value_per_share'),
                'shares_outstanding': data.get('total_shares_outstanding'),
                'revenue': data.get('total_revenue'),
                'dividend_yield': data.get('dividend_yield_recent')
            }
            
            # Try different intervals
            for interval in ['15m', '1h', '1d']:
                count = self.db.backfill_fundamental_history(
                    self.current_symbol,
                    current_fundamentals,
                    interval=interval,
                    limit=365
                )
                if count > 0:
                    status_label.config(
                        text=f"✅ Backfilled {count} days",
                        foreground=COLORS['gain']
                    )
                    # Refresh all charts
                    self._update_valuation_charts()
                    logger.info(f"Backfilled {count} valuation snapshots for {self.current_symbol}")
                    return
            
            status_label.config(text="⚠️ No price history found", foreground=COLORS['warning'])
            
        except Exception as e:
            logger.error(f"Backfill error: {e}")
            status_label.config(text=f"❌ {str(e)[:30]}", foreground=COLORS['loss'])
    
    def _update_valuation_charts(self):
        """Update all valuation ratio tabs (P/E, P/B, P/S)."""
        if not hasattr(self, 'valuation_charts'):
            return
        
        for metric_type in self.valuation_charts:
            self._update_valuation_ratio_tab(metric_type)
    
    def _update_valuation_ratio_tab(self, metric_type: str):
        """Update a single valuation ratio tab with sector comparison."""
        try:
            if metric_type not in self.valuation_charts:
                return
            
            cfg = self.valuation_charts[metric_type]
            field = cfg['field']
            metric_name = cfg['metric_name']
            lower_better = cfg['lower_better']
            good_threshold = cfg['good_threshold']
            high_threshold = cfg['high_threshold']
            
            # Get current stock data
            current_data = self._get_stock_data(self.current_symbol)
            if not current_data:
                return
            
            current_ratio = current_data.get(field)
            
            # Reset cards
            for key in cfg['cards']:
                cfg['cards'][key].config(text="--", foreground=COLORS['text_secondary'])
            
            # Get sector data
            sector = self._get_stock_sector(self.current_symbol)
            if not sector:
                cfg['zone_label'].config(text="Unknown sector", foreground=COLORS['text_muted'])
                return
            
            sector_stocks = self._get_sector_stocks(sector)
            ratio_data = []
            
            for sym in sector_stocks:
                s_data = self._get_stock_data(sym)
                if s_data:
                    ratio_val = s_data.get(field)
                    ratio_data.append({
                        'symbol': sym,
                        'name': s_data.get('name', sym),
                        'price': s_data.get('close', 0) or 0,
                        'ratio': ratio_val,
                        'rec': s_data.get('Recommend.All'),
                        'is_current': sym == self.current_symbol
                    })
            
            # Calculate sector average
            valid_ratios = [d['ratio'] for d in ratio_data if d['ratio'] is not None and d['ratio'] > 0]
            
            if not valid_ratios:
                cfg['zone_label'].config(text="No ratio data available", foreground=COLORS['text_muted'])
                return
            
            avg_ratio = sum(valid_ratios) / len(valid_ratios)
            
            # ========== Summary Cards ==========
            if current_ratio is not None and current_ratio > 0:
                cfg['cards']['current'].config(text=f"{current_ratio:.1f}x")
            
            cfg['cards']['sector_avg'].config(text=f"{avg_ratio:.1f}x")
            
            if current_ratio is not None and current_ratio > 0:
                diff = current_ratio - avg_ratio
                diff_pct = (diff / avg_ratio) * 100 if avg_ratio else 0
                
                # For P/E, P/B, P/S - lower is typically better
                if lower_better:
                    diff_color = COLORS['gain'] if diff < 0 else COLORS['loss']
                else:
                    diff_color = COLORS['gain'] if diff > 0 else COLORS['loss']
                
                cfg['cards']['vs_sector'].config(text=f"{diff:+.1f}", foreground=diff_color)
                
                # Rank
                sorted_ratios = sorted(valid_ratios)
                rank = 1
                for i, r in enumerate(sorted_ratios):
                    if r >= current_ratio:
                        rank = i + 1
                        break
                    rank = i + 2
                total = len(valid_ratios)
                rank_color = COLORS['gain'] if rank <= 3 else COLORS['text_secondary']
                cfg['cards']['rank'].config(text=f"#{rank}/{total}", foreground=rank_color)
                
                # Status
                if current_ratio < good_threshold:
                    status = "🟢 Low"
                    status_color = COLORS['gain']
                elif current_ratio < high_threshold:
                    status = "🟡 Fair"
                    status_color = COLORS['warning']
                else:
                    status = "🔴 High"
                    status_color = COLORS['loss']
                cfg['cards']['status'].config(text=status, foreground=status_color)
            
            # ========== Valuation Gauge ==========
            self._draw_valuation_gauge(cfg, current_ratio, avg_ratio, good_threshold, high_threshold)
            
            # ========== Sector Ranking ==========
            sorted_data = sorted([d for d in ratio_data if d['ratio'] is not None and d['ratio'] > 0], 
                                key=lambda x: x['ratio'])
            
            # Top 3 (lowest)
            for i, label in enumerate(cfg['top_labels']):
                if i < len(sorted_data):
                    d = sorted_data[i]
                    label.config(text=f"{i+1}. {d['symbol']} ({d['ratio']:.1f}x)")
                else:
                    label.config(text=f"{i+1}. --")
            
            # Bottom 3 (highest)
            for i, label in enumerate(cfg['bottom_labels']):
                idx = len(sorted_data) - 1 - i
                if idx >= 0:
                    d = sorted_data[idx]
                    label.config(text=f"{i+1}. {d['symbol']} ({d['ratio']:.1f}x)")
                else:
                    label.config(text=f"{i+1}. --")
            
            # ========== Insights ==========
            self._generate_valuation_insights(cfg, current_ratio, avg_ratio, ratio_data, metric_name)
            
            # ========== Table ==========
            self._populate_valuation_table(cfg, ratio_data, avg_ratio)
            
            # ========== Historical Chart ==========
            self._update_valuation_history_chart(metric_type)
            
        except Exception as e:
            logger.error(f"Error updating {metric_type} tab: {e}")
    
    def _draw_valuation_gauge(self, cfg, current: float, avg: float, good: float, high: float):
        """Draw valuation gauge."""
        canvas = cfg['gauge_canvas']
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 100:
            return
        
        gauge_y = h // 2
        gauge_height = 14
        
        # Gradient from green (low) to red (high)
        segments = 20
        segment_w = (w - 40) / segments
        
        for i in range(segments):
            x1 = 20 + i * segment_w
            x2 = x1 + segment_w
            
            pct = i / segments
            if pct < 0.33:
                r, g, b = int(80 + pct * 3 * 140), 200, 80
            elif pct < 0.66:
                r, g, b = int(220), int(200 - (pct - 0.33) * 3 * 100), 80
            else:
                r, g, b = 220, int(100 - (pct - 0.66) * 3 * 50), 80
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_rectangle(x1, gauge_y - gauge_height//2, x2, gauge_y + gauge_height//2,
                                   fill=color, outline='')
        
        # Calculate pointer position
        if current is not None and current > 0:
            # Map current value to 0-1 range based on thresholds
            max_val = high * 1.5
            pct = min(1.0, current / max_val)
            pointer_x = 20 + pct * (w - 40)
            
            canvas.create_polygon(
                pointer_x - 6, gauge_y - gauge_height//2 - 4,
                pointer_x + 6, gauge_y - gauge_height//2 - 4,
                pointer_x, gauge_y - gauge_height//2 + 3,
                fill='white', outline=''
            )
            
            canvas.create_text(pointer_x, gauge_y + gauge_height//2 + 8,
                             text=f"{current:.1f}x", fill='white',
                             font=('Arial', 8, 'bold'), anchor='n')
        
        canvas.create_text(20, gauge_y, text="Low", fill=COLORS['gain'], font=('Arial', 7), anchor='w')
        canvas.create_text(w - 20, gauge_y, text="High", fill=COLORS['loss'], font=('Arial', 7), anchor='e')
        
        # Zone label
        if current is not None and current > 0:
            if current < good:
                cfg['zone_label'].config(text=f"🟢 Undervalued ({cfg['metric_name']} < {good})", foreground=COLORS['gain'])
            elif current < high:
                cfg['zone_label'].config(text=f"🟡 Fairly Valued ({good} ≤ {cfg['metric_name']} < {high})", foreground=COLORS['warning'])
            else:
                cfg['zone_label'].config(text=f"🔴 Overvalued ({cfg['metric_name']} ≥ {high})", foreground=COLORS['loss'])
        else:
            cfg['zone_label'].config(text="No ratio data", foreground=COLORS['text_muted'])
    
    def _generate_valuation_insights(self, cfg, current: float, avg: float, data: List[Dict], metric_name: str):
        """Generate valuation insights."""
        insights = []
        
        current_stock = [d for d in data if d['is_current']]
        if not current_stock:
            return
        current_stock = current_stock[0]
        
        sorted_data = sorted([d for d in data if d['ratio'] is not None and d['ratio'] > 0],
                            key=lambda x: x['ratio'])
        
        if sorted_data and current_stock['symbol'] == sorted_data[0]['symbol']:
            insights.append(f"🏆 Lowest {metric_name} in sector - potential value pick!")
        
        if sorted_data and current_stock['symbol'] == sorted_data[-1]['symbol']:
            insights.append(f"⚠️ Highest {metric_name} in sector - premium valuation")
        
        if current is not None and avg > 0:
            if current < avg * 0.7:
                insights.append(f"📈 Trading at 30%+ discount to sector average")
            elif current < avg:
                insights.append(f"✅ Below sector average {metric_name}")
            elif current > avg * 1.3:
                insights.append(f"⚠️ Trading at 30%+ premium to sector average")
        
        # Fill labels
        for i, label in enumerate(cfg['insight_labels']):
            if i < len(insights):
                label.config(text=insights[i])
            else:
                label.config(text="")
    
    def _populate_valuation_table(self, cfg, data: List[Dict], avg: float):
        """Populate valuation comparison table."""
        tree = cfg['tree']
        
        for item in tree.get_children():
            tree.delete(item)
        
        # Sort by ratio (low to high)
        sorted_data = sorted([d for d in data if d['ratio'] is not None], key=lambda x: x['ratio'] or 999)
        
        for d in sorted_data:
            ratio = d['ratio']
            vs_avg = ratio - avg if ratio and avg else 0
            
            # Signal
            rec = d.get('rec')
            if rec is not None:
                signal = "🟢 BUY" if rec >= 0.5 else "🔴 SELL" if rec <= -0.5 else "🟡 HOLD"
            else:
                signal = "--"
            
            # Determine tag
            if d['is_current']:
                tag = 'current'
            elif sorted_data.index(d) < 3:
                tag = 'best'
            elif sorted_data.index(d) >= len(sorted_data) - 3:
                tag = 'worst'
            else:
                tag = ''
            
            tree.insert('', 'end', values=(
                d['symbol'],
                d['name'][:12] if d['name'] else '--',
                f"₦{d['price']:,.0f}",
                f"{ratio:.1f}x" if ratio else "--",
                f"{vs_avg:+.1f}" if ratio else "--",
                signal
            ), tags=(tag,))
    
    def _update_valuation_history_chart(self, metric_type: str):
        """Update historical valuation chart with std dev bands."""
        try:
            if metric_type not in self.valuation_charts:
                return
            
            cfg = self.valuation_charts[metric_type]
            field = cfg['field']
            metric_name = cfg['metric_name']
            chart_canvas = cfg['chart_canvas']
            stats_labels = cfg.get('stats_labels', {})
            
            # Get historical data
            history = self.db.get_fundamental_history(self.current_symbol, limit=365)
            
            if not history:
                self._draw_chart_empty(chart_canvas, metric_name)
                return
            
            # Map fields from old to new
            field_map = {
                'price_earnings_ttm': 'pe_ratio',
                'price_book_ratio': 'pb_ratio',
                'price_sales_ratio': 'ps_ratio'
            }
            history_field = field_map.get(field, field)
            
            # Filter valid data
            valid_data = [d for d in history if d.get(history_field) is not None]
            valid_data.sort(key=lambda x: x['date'])
            
            if len(valid_data) < 2:
                self._draw_chart_empty(chart_canvas, metric_name)
                return
            
            # Extract values
            values = [d[history_field] for d in valid_data]
            dates = [d['date'] for d in valid_data]
            
            # Calculate statistics
            avg_val = sum(values) / len(values)
            variance = sum((x - avg_val) ** 2 for x in values) / len(values)
            std_val = variance ** 0.5
            
            # Update stats labels
            if 'avg' in stats_labels:
                stats_labels['avg'].config(text=f"{avg_val:.1f}")
            if 'std' in stats_labels:
                stats_labels['std'].config(text=f"{std_val:.1f}")
            if 'days' in stats_labels:
                stats_labels['days'].config(text=str(len(valid_data)))
            
            # Draw chart
            self._draw_valuation_chart(chart_canvas, values, avg_val, std_val, dates, metric_name)
            
        except Exception as e:
            logger.error(f"Error updating {metric_type} history chart: {e}")
    
    def _draw_chart_empty(self, canvas, metric_name):
        """Draw empty chart with message."""
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        canvas.create_text(
            w // 2, h // 2,
            text=f"📊 No {metric_name} history data yet\n\nData is stored daily.\nClick Backfill to populate from price history.",
            fill=COLORS['text_muted'],
            font=('Arial', 11),
            anchor='center',
            justify='center'
        )
    
    def _draw_valuation_chart(self, canvas, values, avg_val, std_val, dates, metric_name):
        """Draw valuation line chart with std dev bands."""
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 50 or h < 50:
            return
        
        margin = 50
        chart_w = w - 2 * margin
        chart_h = h - 2 * margin
        
        # Determine Y axis range
        min_val = min(values) - std_val if values else 0
        max_val = max(values) + std_val if values else 20
        min_val = max(0, min_val)
        val_range = max_val - min_val if max_val > min_val else 1
        
        # Draw horizontal grid lines and labels
        for i in range(5):
            y = margin + (i * chart_h / 4)
            v = max_val - (i * val_range / 4)
            canvas.create_line(margin, y, w - margin, y, fill='#333333', dash=(2, 2))
            canvas.create_text(margin - 5, y, text=f"{v:.0f}", 
                             fill=COLORS['text_muted'], anchor='e', font=('Arial', 8))
        
        # Draw std dev bands background
        bands = [
            (avg_val + 2 * std_val, avg_val + std_val, '#3D0A0A'),
            (avg_val + std_val, avg_val - std_val, '#1A1A0A'),
            (avg_val - std_val, avg_val - 2 * std_val, '#0A3D0A'),
        ]
        
        for top_v, bottom_v, color in bands:
            top_y = margin + (max_val - top_v) / val_range * chart_h
            bottom_y = margin + (max_val - bottom_v) / val_range * chart_h
            top_y = max(margin, min(h - margin, top_y))
            bottom_y = max(margin, min(h - margin, bottom_y))
            canvas.create_rectangle(margin, top_y, w - margin, bottom_y, 
                                   fill=color, outline='')
        
        # Draw std dev lines
        lines_config = [
            (avg_val + 2 * std_val, '#FF6B6B', '+2σ'),
            (avg_val + std_val, '#FFD700', '+1σ'),
            (avg_val, '#FFFFFF', 'Mean'),
            (avg_val - std_val, '#FFD700', '-1σ'),
            (avg_val - 2 * std_val, '#FF6B6B', '-2σ'),
        ]
        
        for v, color, label in lines_config:
            y = margin + (max_val - v) / val_range * chart_h
            if margin <= y <= h - margin:
                canvas.create_line(margin, y, w - margin, y, fill=color, width=1, dash=(5, 3))
                canvas.create_text(w - margin + 5, y, text=label,
                                 fill=color, anchor='w', font=('Arial', 7))
        
        # Draw value line
        if len(values) > 1:
            points = []
            for i, v in enumerate(values):
                x = margin + (i / (len(values) - 1)) * chart_w
                y = margin + (max_val - v) / val_range * chart_h
                points.extend([x, y])
            
            if len(points) >= 4:
                canvas.create_line(points, fill=COLORS['primary'], width=2, smooth=True)
            
            # Draw current value dot
            last_x = points[-2]
            last_y = points[-1]
            canvas.create_oval(last_x - 4, last_y - 4, last_x + 4, last_y + 4,
                              fill=COLORS['primary'], outline='white', width=1)
        
        # Y axis label
        canvas.create_text(15, h // 2, text=metric_name, fill=COLORS['text_muted'],
                          font=('Arial', 10), angle=90)
        
        # X axis: show first and last date
        if dates:
            canvas.create_text(margin, h - 10, text=str(dates[0])[:10],
                              fill=COLORS['text_muted'], font=('Arial', 7), anchor='w')
            canvas.create_text(w - margin, h - 10, text=str(dates[-1])[:10],
                              fill=COLORS['text_muted'], font=('Arial', 7), anchor='e')
    
    # Keep the old method for compatibility but redirect to new
    def _create_pe_history_tab(self):
        """Legacy method - now handled by _create_valuation_chart_tab."""
        pass
    
    def _backfill_pe_history(self):
        """Legacy method - redirect to new generic backfill."""
        if hasattr(self, 'valuation_charts') and 'pe' in self.valuation_charts:
            # Get status label from chart config
            pass  # Now handled by _backfill_valuation
    
    # =========================================================================
    # DATA LOADING & UPDATES
    # =========================================================================
    
    def _load_all_stocks(self):
        """Load all stocks data from TradingView screener."""
        try:
            self.status_label.config(text="Loading...", foreground=COLORS['warning'])
            df = self.collector.get_all_stocks()
            
            if not df.empty:
                self.all_stocks_data = df
                
                # Populate symbol combo
                if 'symbol' in df.columns:
                    symbols = sorted(df['symbol'].dropna().unique().tolist())
                elif 'ticker' in df.columns:
                    symbols = sorted([t.replace('NSENG:', '') for t in df['ticker'].dropna().unique().tolist()])
                else:
                    symbols = ['DANGCEM']
                
                self.symbol_combo['values'] = symbols
                
                if 'DANGCEM' in symbols:
                    self.symbol_var.set('DANGCEM')
                elif symbols:
                    self.symbol_var.set(symbols[0])
                
                self.current_symbol = self.symbol_var.get()
                self._update_all_tabs()
                
                # Save today's fundamental snapshots for P/E history tracking
                try:
                    count = self.db.save_all_fundamental_snapshots(df.to_dict('records'))
                    logger.info(f"Saved {count} fundamental snapshots")
                except Exception as e:
                    logger.error(f"Error saving snapshots: {e}")
                
                self.status_label.config(
                    text=f"✓ {len(df)} stocks loaded",
                    foreground=COLORS['gain']
                )
            else:
                self.status_label.config(text="No data", foreground=COLORS['loss'])
                
        except Exception as e:
            logger.error(f"Error loading stocks: {e}")
            self.status_label.config(text=f"Error: {e}", foreground=COLORS['loss'])
    
    def _on_symbol_change(self):
        """Handle symbol selection change."""
        self.current_symbol = self.symbol_var.get()
        self._update_all_tabs()
    
    def _refresh_data(self):
        """Refresh all data."""
        self._load_all_stocks()
    
    def _update_all_tabs(self):
        """Update all sub-tabs with current symbol data."""
        if not self.current_symbol or self.all_stocks_data is None:
            return
        
        self._update_overview_tab()
        self._update_sector_tab()
        self._update_peers_tab()
        self._update_fairvalue_tab()
        self._update_dividends_tab()
        self._update_valuation_charts()
    
    def _get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get data for a specific stock from loaded data."""
        if self.all_stocks_data is None:
            return None
        
        df = self.all_stocks_data
        
        if 'symbol' in df.columns:
            matches = df[df['symbol'].str.upper() == symbol.upper()]
        elif 'ticker' in df.columns:
            matches = df[df['ticker'].str.contains(symbol, case=False, na=False)]
        else:
            return None
        
        if matches.empty:
            return None
        
        row = matches.iloc[0]
        return row.to_dict()
    
    def _get_stock_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a stock."""
        symbol = symbol.upper()
        for sector, stocks in self.SECTORS.items():
            if symbol in stocks:
                return sector
        return None
    
    def _get_sector_stocks(self, sector: str) -> List[str]:
        """Get all stocks in a sector."""
        return self.SECTORS.get(sector, [])
    
    def _update_overview_tab(self):
        """Update Super Enhanced Overview tab with all metrics."""
        try:
            data = self._get_stock_data(self.current_symbol)
            if not data:
                return
            
            # ========== Stock Info Header ==========
            name = data.get('name', self.current_symbol)
            self.stock_name_label.config(text=f"{self.current_symbol} - {name}")
            
            close = data.get('close', 0) or 0
            self.stock_price_label.config(text=f"₦{close:,.2f}")
            
            change = data.get('change', 0) or 0
            change_color = COLORS['gain'] if change >= 0 else COLORS['loss']
            self.stock_change_label.config(text=f"{change:+.2f}%", foreground=change_color)
            
            # ========== 52-Week Range Bar ==========
            high_52w = data.get('High.52W') or data.get('price_52_week_high')
            low_52w = data.get('Low.52W') or data.get('price_52_week_low')
            
            if high_52w and low_52w:
                self.range_low_label.config(text=f"52W Low\n₦{low_52w:,.2f}")
                self.range_high_label.config(text=f"52W High\n₦{high_52w:,.2f}")
                self._draw_52w_range_bar(close, low_52w, high_52w)
            
            # ========== Key Metrics Grid (12 cards) ==========
            self._update_metric_cards(data)
            
            # ========== Performance ==========
            perf_map = {
                'week': 'Perf.W', 'month': 'Perf.1M', 'quarter': 'Perf.3M',
                'half_year': 'Perf.6M', 'ytd': 'Perf.YTD', 'year': 'Perf.Y'
            }
            for key, col in perf_map.items():
                val = data.get(col)
                if val is not None:
                    color = COLORS['gain'] if val > 0 else COLORS['loss']
                    self.perf_labels[key].config(text=f"{val:+.1f}%", foreground=color)
            
            # ========== Valuation Score Gauge ==========
            self._update_valuation_gauge(data)
            
            # ========== Quick Insights ==========
            self._generate_quick_insights(data)
            
            # ========== Technical Summary ==========
            self._update_technical_summary(data)
            
            # ========== Sector Position ==========
            self._update_sector_position(data)
            
        except Exception as e:
            logger.error(f"Error updating overview: {e}")
    
    def _update_metric_cards(self, data: Dict):
        """Update all 12 metric cards."""
        # --- Row 1: Valuation ---
        pe = data.get('price_earnings_ttm')
        if pe is not None:
            pe_color = COLORS['gain'] if pe < 15 else COLORS['loss'] if pe > 25 else COLORS['warning']
            self.metric_cards['pe_ratio'].config(text=f"{pe:.1f}x", foreground=pe_color)
        
        pb = data.get('price_book_ratio')
        if pb is not None:
            pb_color = COLORS['gain'] if pb < 2 else COLORS['text_secondary']
            self.metric_cards['pb_ratio'].config(text=f"{pb:.2f}x", foreground=pb_color)
        else:
            self.metric_cards['pb_ratio'].config(text="--")
        
        # P/S not available directly, show N/A
        self.metric_cards['ps_ratio'].config(text="--")
        
        ev_ebitda = data.get('enterprise_value_ebitda')
        if ev_ebitda is not None:
            self.metric_cards['ev_ebitda'].config(text=f"{ev_ebitda:.1f}x")
        else:
            self.metric_cards['ev_ebitda'].config(text="--")
        
        # --- Row 2: Profitability ---
        eps = data.get('earnings_per_share_basic_ttm')
        if eps is not None:
            eps_color = COLORS['gain'] if eps > 0 else COLORS['loss']
            self.metric_cards['eps'].config(text=f"₦{eps:.2f}", foreground=eps_color)
        
        roe = data.get('return_on_equity')
        if roe is not None:
            roe_color = COLORS['gain'] if roe > 15 else COLORS['text_secondary']
            self.metric_cards['roe'].config(text=f"{roe:.1f}%", foreground=roe_color)
        else:
            self.metric_cards['roe'].config(text="--")
        
        margin = data.get('net_margin')
        if margin is not None:
            margin_color = COLORS['gain'] if margin > 10 else COLORS['text_secondary']
            self.metric_cards['net_margin'].config(text=f"{margin:.1f}%", foreground=margin_color)
        else:
            self.metric_cards['net_margin'].config(text="--")
        
        div = data.get('dividend_yield_recent')
        if div is not None:
            div_color = COLORS['gain'] if div > 5 else COLORS['text_secondary']
            self.metric_cards['dividend'].config(text=f"{div:.1f}%", foreground=div_color)
        else:
            self.metric_cards['dividend'].config(text="N/A")
        
        # --- Row 3: Size & Liquidity ---
        mcap = data.get('market_cap_basic', 0) or 0
        if mcap >= 1e12:
            mcap_text = f"₦{mcap/1e12:.1f}T"
        elif mcap >= 1e9:
            mcap_text = f"₦{mcap/1e9:.1f}B"
        elif mcap >= 1e6:
            mcap_text = f"₦{mcap/1e6:.1f}M"
        else:
            mcap_text = f"₦{mcap:,.0f}"
        self.metric_cards['market_cap'].config(text=mcap_text)
        
        high_52w = data.get('High.52W') or data.get('price_52_week_high')
        if high_52w:
            self.metric_cards['high_52w'].config(text=f"₦{high_52w:,.2f}")
        
        low_52w = data.get('Low.52W') or data.get('price_52_week_low')
        if low_52w:
            self.metric_cards['low_52w'].config(text=f"₦{low_52w:,.2f}")
        
        vol = data.get('average_volume_10d_calc', 0) or 0
        if vol >= 1e6:
            vol_text = f"{vol/1e6:.1f}M"
        elif vol >= 1e3:
            vol_text = f"{vol/1e3:.1f}K"
        else:
            vol_text = f"{vol:,.0f}"
        self.metric_cards['volume'].config(text=vol_text)
    
    def _draw_52w_range_bar(self, current: float, low: float, high: float):
        """Draw 52-week range bar with current price position."""
        canvas = self.range_canvas
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 50 or high <= low:
            return
        
        # Draw the range bar (gray background)
        bar_y = h // 2
        bar_height = 12
        
        canvas.create_rectangle(
            10, bar_y - bar_height//2,
            w - 10, bar_y + bar_height//2,
            fill='#333333', outline=''
        )
        
        # Calculate current position percentage
        pct = (current - low) / (high - low)
        pct = max(0, min(1, pct))
        
        # Gradient fill from low to current position
        current_x = 10 + pct * (w - 20)
        
        # Color based on position
        if pct < 0.3:
            fill_color = COLORS['loss']  # Near 52W low
        elif pct > 0.7:
            fill_color = COLORS['gain']  # Near 52W high
        else:
            fill_color = COLORS['warning']  # Middle
        
        canvas.create_rectangle(
            10, bar_y - bar_height//2,
            current_x, bar_y + bar_height//2,
            fill=fill_color, outline=''
        )
        
        # Draw current price marker (triangle)
        canvas.create_polygon(
            current_x - 5, bar_y - bar_height//2 - 3,
            current_x + 5, bar_y - bar_height//2 - 3,
            current_x, bar_y - bar_height//2 + 4,
            fill='white', outline=''
        )
        
        # Current price label
        canvas.create_text(
            current_x, bar_y + bar_height//2 + 8,
            text=f"₦{current:,.0f}",
            fill='white', font=('Arial', 8), anchor='n'
        )
    
    def _update_valuation_gauge(self, data: Dict):
        """Draw valuation score gauge."""
        canvas = self.gauge_canvas
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 100:
            return
        
        # Calculate valuation score (0-100)
        score = 50  # Default neutral
        details = []
        
        pe = data.get('price_earnings_ttm')
        if pe is not None and pe > 0:
            if pe < 8:
                score += 25
                details.append("P/E very low")
            elif pe < 15:
                score += 15
                details.append("P/E attractive")
            elif pe < 25:
                score += 0
                details.append("P/E fair")
            else:
                score -= 15
                details.append("P/E expensive")
        
        div = data.get('dividend_yield_recent')
        if div is not None and div > 0:
            if div > 8:
                score += 10
                details.append("High yield")
            elif div > 4:
                score += 5
                details.append("Good yield")
        
        # Clamp score
        score = max(0, min(100, score))
        
        # Draw gauge background
        gauge_y = h // 2
        gauge_height = 16
        
        # Gradient: Red -> Yellow -> Green
        segments = 20
        segment_w = (w - 40) / segments
        
        for i in range(segments):
            x1 = 20 + i * segment_w
            x2 = x1 + segment_w
            
            # Calculate color based on position
            pct = i / segments
            if pct < 0.33:
                r = int(220)
                g = int(80 + pct * 3 * 100)
                b = 80
            elif pct < 0.66:
                r = int(220 - (pct - 0.33) * 3 * 100)
                g = 200
                b = 80
            else:
                r = 80
                g = int(200 - (pct - 0.66) * 3 * 50)
                b = 80
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_rectangle(x1, gauge_y - gauge_height//2, x2, gauge_y + gauge_height//2,
                                   fill=color, outline='')
        
        # Draw pointer
        pointer_x = 20 + (score / 100) * (w - 40)
        canvas.create_polygon(
            pointer_x - 6, gauge_y - gauge_height//2 - 5,
            pointer_x + 6, gauge_y - gauge_height//2 - 5,
            pointer_x, gauge_y - gauge_height//2 + 3,
            fill='white', outline=''
        )
        
        # Draw score text
        canvas.create_text(
            pointer_x, gauge_y + gauge_height//2 + 8,
            text=f"{score}/100",
            fill='white', font=('Arial', 9, 'bold'), anchor='n'
        )
        
        # Labels
        canvas.create_text(20, gauge_y, text="Cheap", fill=COLORS['gain'], font=('Arial', 7), anchor='w')
        canvas.create_text(w - 20, gauge_y, text="Expensive", fill=COLORS['loss'], font=('Arial', 7), anchor='e')
        
        # Update status labels
        if score >= 65:
            status = "🟢 UNDERVALUED"
            color = COLORS['gain']
        elif score >= 40:
            status = "🟡 FAIRLY VALUED"
            color = COLORS['warning']
        else:
            status = "🔴 OVERVALUED"
            color = COLORS['loss']
        
        self.valuation_status.config(text=status, foreground=color)
        self.valuation_details.config(text=" • ".join(details) if details else "Limited data available")
    
    def _generate_quick_insights(self, data: Dict):
        """Generate quick insights based on stock data."""
        insights = []
        close = data.get('close', 0) or 0
        
        # 52W High/Low insights
        high_52w = data.get('High.52W') or data.get('price_52_week_high')
        low_52w = data.get('Low.52W') or data.get('price_52_week_low')
        
        if high_52w and low_52w and close > 0:
            pct_from_high = ((close - high_52w) / high_52w) * 100
            pct_from_low = ((close - low_52w) / low_52w) * 100
            
            if pct_from_high > -5:
                insights.append(f"📈 Trading near 52-week high ({pct_from_high:+.1f}%)")
            elif pct_from_high < -30:
                insights.append(f"📉 Trading {abs(pct_from_high):.0f}% below 52-week high")
            
            if pct_from_low < 10 and pct_from_low > 0:
                insights.append(f"⚠️ Near 52-week low, potential risk")
        
        # P/E insight
        pe = data.get('price_earnings_ttm')
        if pe is not None:
            if pe < 10:
                insights.append(f"💎 Low P/E of {pe:.1f}x may indicate value")
            elif pe > 30:
                insights.append(f"⚠️ High P/E of {pe:.1f}x suggests premium valuation")
        
        # Dividend insight
        div = data.get('dividend_yield_recent')
        if div is not None and div > 5:
            insights.append(f"💵 Attractive dividend yield of {div:.1f}%")
        
        # Volume insight
        vol = data.get('volume', 0) or 0
        avg_vol = data.get('average_volume_10d_calc', 0) or 0
        if avg_vol > 0 and vol > 0:
            vol_ratio = vol / avg_vol
            if vol_ratio > 2:
                insights.append(f"🔥 Volume {vol_ratio:.1f}x above average - unusual activity")
        
        # YTD performance insight
        ytd = data.get('Perf.YTD')
        if ytd is not None:
            if ytd > 30:
                insights.append(f"🚀 Up {ytd:.0f}% YTD - strong momentum")
            elif ytd < -30:
                insights.append(f"📉 Down {abs(ytd):.0f}% YTD - underperformer")
        
        # Fill insight labels
        for i, label in enumerate(self.insights_labels):
            if i < len(insights):
                label.config(text=insights[i])
            else:
                label.config(text="")
    
    def _update_valuation_assessment(self, data: Dict):
        """Legacy method - now handled by _update_valuation_gauge."""
        pass
    
    def _update_technical_summary(self, data: Dict):
        """Update technical summary with RSI, MACD, SMAs, and signal."""
        try:
            # RSI
            rsi = data.get('RSI')
            if rsi is not None:
                if rsi > 70:
                    color = COLORS['loss']
                    text = f"{rsi:.0f} OB"  # Overbought
                elif rsi < 30:
                    color = COLORS['gain']
                    text = f"{rsi:.0f} OS"  # Oversold
                else:
                    color = COLORS['text_secondary']
                    text = f"{rsi:.0f}"
                self.tech_labels['rsi'].config(text=text, foreground=color)
            
            # MACD
            macd = data.get('MACD.macd')
            macd_signal = data.get('MACD.signal')
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    color = COLORS['gain']
                    text = "Bullish"
                else:
                    color = COLORS['loss']
                    text = "Bearish"
                self.tech_labels['macd'].config(text=text, foreground=color)
            
            # SMAs - compare price to SMA
            close = data.get('close', 0) or 0
            
            for sma_key, sma_col in [('sma20', 'SMA20'), ('sma50', 'SMA50'), ('sma200', 'SMA200')]:
                sma = data.get(sma_col)
                if sma is not None and close > 0:
                    if close > sma:
                        color = COLORS['gain']
                        text = "Above"
                    else:
                        color = COLORS['loss']
                        text = "Below"
                    self.tech_labels[sma_key].config(text=text, foreground=color)
            
            # Overall Recommendation
            rec = data.get('Recommend.All')
            if rec is not None:
                if rec >= 0.5:
                    color = COLORS['gain']
                    text = "🟢 BUY"
                elif rec <= -0.5:
                    color = COLORS['loss']
                    text = "🔴 SELL"
                else:
                    color = COLORS['warning']
                    text = "🟡 HOLD"
                self.tech_labels['rec'].config(text=text, foreground=color)
                
        except Exception as e:
            logger.error(f"Error updating technical summary: {e}")
    
    def _update_sector_position(self, data: Dict):
        """Update sector position in Overview."""
        try:
            sector = self._get_stock_sector(self.current_symbol)
            
            if sector:
                self.overview_sector_label.config(text=f"Sector: {sector}")
                
                # Get peers in sector
                peers = self._get_sector_stocks(sector)
                
                # Rank by P/E in sector
                pe = data.get('price_earnings_ttm')
                if pe and peers:
                    # Get P/E for all peers
                    pe_data = []
                    for peer in peers:
                        peer_data = self._get_stock_data(peer)
                        if peer_data:
                            peer_pe = peer_data.get('price_earnings_ttm')
                            if peer_pe and peer_pe > 0:
                                pe_data.append((peer, peer_pe))
                    
                    if pe_data:
                        # Sort by P/E (lowest = best value = rank 1)
                        pe_data.sort(key=lambda x: x[1])
                        rank = 1
                        for i, (sym, p) in enumerate(pe_data):
                            if sym == self.current_symbol:
                                rank = i + 1
                                break
                        
                        total = len(pe_data)
                        self.sector_rank_label.config(text=f"P/E Rank: #{rank}/{total}")
                        
                        # Average sector P/E
                        avg_pe = sum(p for _, p in pe_data) / len(pe_data)
                        diff = ((pe - avg_pe) / avg_pe) * 100 if avg_pe > 0 else 0
                        
                        if diff < -10:
                            color = COLORS['gain']
                            text = f"P/E {diff:+.0f}% vs Sector"
                        elif diff > 10:
                            color = COLORS['loss']
                            text = f"P/E {diff:+.0f}% vs Sector"
                        else:
                            color = COLORS['text_secondary']
                            text = f"P/E ~Sector Avg"
                        
                        self.sector_pe_compare.config(text=text, foreground=color)
            else:
                self.overview_sector_label.config(text="Sector: --")
                
        except Exception as e:
            logger.error(f"Error updating sector position: {e}")
    
    def _update_sector_tab(self):
        """Update Super Enhanced Sector Analysis tab."""
        try:
            sector = self._get_stock_sector(self.current_symbol)
            
            if not sector:
                self.sector_name_label.config(text="Sector: Unknown")
                return
            
            sector_stocks = self._get_sector_stocks(sector)
            self.sector_name_label.config(text=f"Sector: {sector}")
            self.sector_stocks_label.config(text=f"({len(sector_stocks)} stocks)")
            
            # Get current stock data
            current_data = self._get_stock_data(self.current_symbol)
            
            # Collect sector data
            sector_data = []
            for sym in sector_stocks:
                stock_data = self._get_stock_data(sym)
                if stock_data:
                    sector_data.append({
                        'symbol': sym,
                        'name': stock_data.get('name', sym),
                        'price': stock_data.get('close', 0) or 0,
                        'change': stock_data.get('change', 0) or 0,
                        'pe': stock_data.get('price_earnings_ttm'),
                        'div': stock_data.get('dividend_yield_recent'),
                        'mcap': stock_data.get('market_cap_basic', 0) or 0,
                        'ytd': stock_data.get('Perf.YTD'),
                        'rec': stock_data.get('Recommend.All')
                    })
            
            # ========== Sector Statistics ==========
            self._update_sector_stats(sector_data)
            
            # ========== Leaders & Laggards ==========
            self._update_leaders_laggards(sector_data)
            
            # ========== Comparison Bars ==========
            self._draw_sector_comparison_bars(current_data, sector_data)
            
            # ========== Ranking Gauge ==========
            self._draw_sector_ranking_gauge(current_data, sector_data)
            
            # ========== Enhanced Sector Table ==========
            self._populate_sector_table(sector_data)
                    
        except Exception as e:
            logger.error(f"Error updating sector tab: {e}")
    
    def _update_sector_stats(self, sector_data: List[Dict]):
        """Update sector statistics summary."""
        if not sector_data:
            return
        
        # Total market cap
        total_mcap = sum(d['mcap'] for d in sector_data)
        if total_mcap >= 1e12:
            mcap_text = f"₦{total_mcap/1e12:.1f}T"
        elif total_mcap >= 1e9:
            mcap_text = f"₦{total_mcap/1e9:.0f}B"
        else:
            mcap_text = f"₦{total_mcap/1e6:.0f}M"
        self.sector_stats['total_mcap'].config(text=mcap_text)
        
        # Average P/E
        pe_values = [d['pe'] for d in sector_data if d['pe'] and d['pe'] > 0]
        if pe_values:
            avg_pe = sum(pe_values) / len(pe_values)
            self.sector_stats['avg_pe'].config(text=f"{avg_pe:.1f}x")
        
        # Average Dividend Yield
        div_values = [d['div'] for d in sector_data if d['div'] is not None]
        if div_values:
            avg_div = sum(div_values) / len(div_values)
            self.sector_stats['avg_div'].config(text=f"{avg_div:.1f}%")
        
        # Gainers/Losers
        gainers = sum(1 for d in sector_data if d['change'] > 0)
        losers = sum(1 for d in sector_data if d['change'] < 0)
        self.sector_stats['gainers'].config(text=str(gainers), foreground=COLORS['gain'])
        self.sector_stats['losers'].config(text=str(losers), foreground=COLORS['loss'])
        
        # Average change
        changes = [d['change'] for d in sector_data]
        if changes:
            avg_change = sum(changes) / len(changes)
            color = COLORS['gain'] if avg_change > 0 else COLORS['loss']
            self.sector_stats['avg_change'].config(text=f"{avg_change:+.1f}%", foreground=color)
    
    def _update_leaders_laggards(self, sector_data: List[Dict]):
        """Update leaders and laggards panel."""
        if not sector_data:
            return
        
        # Sort by daily change
        sorted_data = sorted(sector_data, key=lambda x: x['change'], reverse=True)
        
        # Top 3 leaders
        for i, label in enumerate(self.leader_labels):
            if i < len(sorted_data):
                d = sorted_data[i]
                label.config(text=f"{i+1}. {d['symbol']} ({d['change']:+.1f}%)")
            else:
                label.config(text=f"{i+1}. --")
        
        # Bottom 3 laggards
        for i, label in enumerate(self.laggard_labels):
            idx = len(sorted_data) - 1 - i
            if idx >= 0 and idx < len(sorted_data):
                d = sorted_data[idx]
                label.config(text=f"{i+1}. {d['symbol']} ({d['change']:+.1f}%)")
            else:
                label.config(text=f"{i+1}. --")
    
    def _draw_sector_comparison_bars(self, current_data: Dict, sector_data: List[Dict]):
        """Draw horizontal comparison bars."""
        canvas = self.sector_compare_canvas
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 100 or not current_data:
            return
        
        # Metrics to compare
        metrics = [
            ('P/E', 'pe', lambda d: d.get('price_earnings_ttm') or d.get('pe')),
            ('Div %', 'div', lambda d: d.get('dividend_yield_recent') or d.get('div')),
            ('YTD %', 'ytd', lambda d: d.get('Perf.YTD') or d.get('ytd'))
        ]
        
        bar_height = 25
        row_height = h // 3
        
        for idx, (label, key, getter) in enumerate(metrics):
            y = idx * row_height + row_height // 2
            
            # Get values
            stock_val = getter(current_data) or 0
            sector_vals = [getter(d) for d in sector_data if getter(d) is not None]
            sector_avg = sum(sector_vals) / len(sector_vals) if sector_vals else 0
            
            # Skip if no data
            if sector_avg == 0 and stock_val == 0:
                continue
            
            # Draw label
            canvas.create_text(10, y, text=label, fill=COLORS['text_muted'], 
                             font=('Arial', 9), anchor='w')
            
            # Calculate bar widths (normalize)
            max_val = max(abs(stock_val), abs(sector_avg), 1)
            bar_start = 70
            bar_width = (w - bar_start - 100)
            
            stock_width = (stock_val / max_val) * (bar_width / 2) if max_val else 0
            sector_width = (sector_avg / max_val) * (bar_width / 2) if max_val else 0
            
            # Draw stock bar (blue)
            canvas.create_rectangle(
                bar_start, y - bar_height//2,
                bar_start + abs(stock_width), y - 2,
                fill=COLORS['primary'], outline=''
            )
            canvas.create_text(bar_start + abs(stock_width) + 5, y - bar_height//4,
                             text=f"Stock: {stock_val:.1f}", fill=COLORS['primary'],
                             font=('Arial', 8), anchor='w')
            
            # Draw sector bar (gray)
            canvas.create_rectangle(
                bar_start, y + 2,
                bar_start + abs(sector_width), y + bar_height//2,
                fill='#555555', outline=''
            )
            canvas.create_text(bar_start + abs(sector_width) + 5, y + bar_height//4,
                             text=f"Sector: {sector_avg:.1f}", fill='#888888',
                             font=('Arial', 8), anchor='w')
    
    def _draw_sector_ranking_gauge(self, current_data: Dict, sector_data: List[Dict]):
        """Draw sector ranking gauge."""
        canvas = self.sector_gauge_canvas
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 100 or not current_data:
            return
        
        # Calculate percentile based on YTD performance
        stock_ytd = current_data.get('Perf.YTD')
        if stock_ytd is None:
            self.percentile_label.config(text="Insufficient data for ranking")
            return
        
        ytd_values = [d['ytd'] for d in sector_data if d['ytd'] is not None]
        if not ytd_values:
            return
        
        better_than = sum(1 for p in ytd_values if stock_ytd > p)
        percentile = (better_than / len(ytd_values)) * 100
        
        # Draw gauge background
        gauge_y = h // 2
        gauge_height = 16
        
        # Gradient segments
        segments = 20
        segment_w = (w - 40) / segments
        
        for i in range(segments):
            x1 = 20 + i * segment_w
            x2 = x1 + segment_w
            
            pct = i / segments
            if pct < 0.33:
                r, g, b = 220, int(80 + pct * 3 * 100), 80
            elif pct < 0.66:
                r, g, b = int(220 - (pct - 0.33) * 3 * 100), 200, 80
            else:
                r, g, b = 80, int(200 - (pct - 0.66) * 3 * 50), 80
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_rectangle(x1, gauge_y - gauge_height//2, x2, gauge_y + gauge_height//2,
                                   fill=color, outline='')
        
        # Draw pointer
        pointer_x = 20 + (percentile / 100) * (w - 40)
        canvas.create_polygon(
            pointer_x - 6, gauge_y - gauge_height//2 - 5,
            pointer_x + 6, gauge_y - gauge_height//2 - 5,
            pointer_x, gauge_y - gauge_height//2 + 3,
            fill='white', outline=''
        )
        
        canvas.create_text(pointer_x, gauge_y + gauge_height//2 + 8,
                          text=f"{percentile:.0f}%", fill='white',
                          font=('Arial', 9, 'bold'), anchor='n')
        
        canvas.create_text(20, gauge_y, text="0%", fill=COLORS['loss'], font=('Arial', 7), anchor='w')
        canvas.create_text(w - 20, gauge_y, text="100%", fill=COLORS['gain'], font=('Arial', 7), anchor='e')
        
        self.percentile_label.config(
            text=f"Outperforms {percentile:.0f}% of sector peers (YTD)",
            foreground=COLORS['gain'] if percentile > 50 else COLORS['loss']
        )
    
    def _populate_sector_table(self, sector_data: List[Dict]):
        """Populate the enhanced sector stocks table."""
        for item in self.sector_tree.get_children():
            self.sector_tree.delete(item)
        
        # Sort by market cap
        sorted_data = sorted(sector_data, key=lambda x: x['mcap'], reverse=True)
        
        for d in sorted_data:
            mcap = d['mcap']
            mcap_text = f"₦{mcap/1e9:.1f}B" if mcap >= 1e9 else f"₦{mcap/1e6:.0f}M"
            
            # Signal based on recommendation
            rec = d.get('rec')
            if rec is not None:
                signal = "🟢 BUY" if rec >= 0.5 else "🔴 SELL" if rec <= -0.5 else "🟡 HOLD"
            else:
                signal = "--"
            
            tag = 'current' if d['symbol'] == self.current_symbol else (
                'gain' if d['change'] > 0 else 'loss' if d['change'] < 0 else ''
            )
            
            self.sector_tree.insert('', 'end', values=(
                d['symbol'],
                d['name'][:15] if d['name'] else '--',
                f"₦{d['price']:,.2f}",
                f"{d['change']:+.1f}%",
                f"{d['pe']:.1f}" if d['pe'] else "--",
                f"{d['div']:.1f}%" if d['div'] else "--",
                mcap_text,
                signal
            ), tags=(tag,))
    
    def _update_peers_tab(self):
        """Update Super Enhanced Peer Comparison tab."""
        try:
            sector = self._get_stock_sector(self.current_symbol)
            if not sector:
                return
            
            sector_stocks = self._get_sector_stocks(sector)
            current_data = self._get_stock_data(self.current_symbol)
            
            # Collect peer data
            peer_data = []
            for sym in sector_stocks:
                stock_data = self._get_stock_data(sym)
                if stock_data:
                    peer_data.append({
                        'symbol': sym,
                        'name': stock_data.get('name', sym),
                        'price': stock_data.get('close', 0) or 0,
                        'change': stock_data.get('change', 0) or 0,
                        'pe': stock_data.get('price_earnings_ttm'),
                        'pb': stock_data.get('price_book_ratio'),
                        'eps': stock_data.get('earnings_per_share_basic_ttm'),
                        'div': stock_data.get('dividend_yield_recent'),
                        'mcap': stock_data.get('market_cap_basic', 0) or 0,
                        'ytd': stock_data.get('Perf.YTD'),
                        'rec': stock_data.get('Recommend.All'),
                        'is_current': sym == self.current_symbol
                    })
            
            # ========== Peer Metrics Summary ==========
            self._update_peer_summary(current_data, peer_data)
            
            # ========== Metric Rankings ==========
            self._update_peer_rankings(peer_data)
            
            # ========== Peer Insights ==========
            self._generate_peer_insights(current_data, peer_data)
            
            # ========== Enhanced Peer Table ==========
            self._populate_peer_table(peer_data)
                    
        except Exception as e:
            logger.error(f"Error updating peers tab: {e}")
    
    def _update_peer_summary(self, current_data: Dict, peer_data: List[Dict]):
        """Update peer metrics summary cards."""
        if not current_data or not peer_data:
            return
        
        # P/E
        stock_pe = current_data.get('price_earnings_ttm')
        pe_values = [d['pe'] for d in peer_data if d['pe'] and d['pe'] > 0]
        avg_pe = sum(pe_values) / len(pe_values) if pe_values else 0
        
        self.peer_summary['pe']['stock'].config(
            text=f"You: {stock_pe:.1f}" if stock_pe else "You: --",
            foreground=COLORS['gain'] if stock_pe and avg_pe and stock_pe < avg_pe else COLORS['text_secondary']
        )
        self.peer_summary['pe']['avg'].config(text=f"Avg: {avg_pe:.1f}" if avg_pe else "Avg: --")
        
        # Dividend
        stock_div = current_data.get('dividend_yield_recent')
        div_values = [d['div'] for d in peer_data if d['div'] is not None]
        avg_div = sum(div_values) / len(div_values) if div_values else 0
        
        self.peer_summary['div']['stock'].config(
            text=f"You: {stock_div:.1f}%" if stock_div else "You: --",
            foreground=COLORS['gain'] if stock_div and avg_div and stock_div > avg_div else COLORS['text_secondary']
        )
        self.peer_summary['div']['avg'].config(text=f"Avg: {avg_div:.1f}%" if avg_div else "Avg: --")
        
        # YTD
        stock_ytd = current_data.get('Perf.YTD')
        ytd_values = [d['ytd'] for d in peer_data if d['ytd'] is not None]
        avg_ytd = sum(ytd_values) / len(ytd_values) if ytd_values else 0
        
        self.peer_summary['ytd']['stock'].config(
            text=f"You: {stock_ytd:+.1f}%" if stock_ytd else "You: --",
            foreground=COLORS['gain'] if stock_ytd and stock_ytd > avg_ytd else COLORS['loss'] if stock_ytd else COLORS['text_secondary']
        )
        self.peer_summary['ytd']['avg'].config(text=f"Avg: {avg_ytd:+.1f}%" if ytd_values else "Avg: --")
        
        # Day Change
        stock_day = current_data.get('change', 0) or 0
        day_values = [d['change'] for d in peer_data]
        avg_day = sum(day_values) / len(day_values) if day_values else 0
        
        self.peer_summary['day']['stock'].config(
            text=f"You: {stock_day:+.1f}%",
            foreground=COLORS['gain'] if stock_day > avg_day else COLORS['loss']
        )
        self.peer_summary['day']['avg'].config(text=f"Avg: {avg_day:+.1f}%")
        
        # MCap
        stock_mcap = current_data.get('market_cap_basic', 0) or 0
        mcap_values = [d['mcap'] for d in peer_data]
        avg_mcap = sum(mcap_values) / len(mcap_values) if mcap_values else 0
        
        stock_mcap_text = f"₦{stock_mcap/1e9:.1f}B" if stock_mcap >= 1e9 else f"₦{stock_mcap/1e6:.0f}M"
        avg_mcap_text = f"₦{avg_mcap/1e9:.1f}B" if avg_mcap >= 1e9 else f"₦{avg_mcap/1e6:.0f}M"
        
        self.peer_summary['mcap']['stock'].config(text=f"You: {stock_mcap_text}")
        self.peer_summary['mcap']['avg'].config(text=f"Avg: {avg_mcap_text}")
    
    def _update_peer_rankings(self, peer_data: List[Dict]):
        """Update metric rankings."""
        if not peer_data:
            return
        
        current = [d for d in peer_data if d['is_current']]
        if not current:
            return
        current = current[0]
        total = len(peer_data)
        
        # P/E Rank (lower is better)
        pe_sorted = sorted([d for d in peer_data if d['pe'] and d['pe'] > 0], key=lambda x: x['pe'])
        pe_rank = next((i+1 for i, d in enumerate(pe_sorted) if d['symbol'] == current['symbol']), None)
        if pe_rank:
            color = COLORS['gain'] if pe_rank <= 3 else COLORS['loss'] if pe_rank > len(pe_sorted) - 3 else COLORS['text_secondary']
            self.peer_rankings['pe_rank'].config(text=f"#{pe_rank}/{len(pe_sorted)}", foreground=color)
        
        # Div Rank (higher is better)
        div_sorted = sorted([d for d in peer_data if d['div'] is not None], key=lambda x: x['div'], reverse=True)
        div_rank = next((i+1 for i, d in enumerate(div_sorted) if d['symbol'] == current['symbol']), None)
        if div_rank:
            color = COLORS['gain'] if div_rank <= 3 else COLORS['text_secondary']
            self.peer_rankings['div_rank'].config(text=f"#{div_rank}/{len(div_sorted)}", foreground=color)
        
        # YTD Rank (higher is better)
        ytd_sorted = sorted([d for d in peer_data if d['ytd'] is not None], key=lambda x: x['ytd'], reverse=True)
        ytd_rank = next((i+1 for i, d in enumerate(ytd_sorted) if d['symbol'] == current['symbol']), None)
        if ytd_rank:
            color = COLORS['gain'] if ytd_rank <= 3 else COLORS['text_secondary']
            self.peer_rankings['ytd_rank'].config(text=f"#{ytd_rank}/{len(ytd_sorted)}", foreground=color)
        
        # MCap Rank (larger is different, just show rank)
        mcap_sorted = sorted(peer_data, key=lambda x: x['mcap'], reverse=True)
        mcap_rank = next((i+1 for i, d in enumerate(mcap_sorted) if d['symbol'] == current['symbol']), None)
        if mcap_rank:
            self.peer_rankings['mcap_rank'].config(text=f"#{mcap_rank}/{total}")
        
        # EPS Rank (higher is better)
        eps_sorted = sorted([d for d in peer_data if d['eps'] is not None], key=lambda x: x['eps'] or 0, reverse=True)
        eps_rank = next((i+1 for i, d in enumerate(eps_sorted) if d['symbol'] == current['symbol']), None)
        if eps_rank:
            color = COLORS['gain'] if eps_rank <= 3 else COLORS['text_secondary']
            self.peer_rankings['eps_rank'].config(text=f"#{eps_rank}/{len(eps_sorted)}", foreground=color)
        
        # Day Change Rank
        change_sorted = sorted(peer_data, key=lambda x: x['change'], reverse=True)
        change_rank = next((i+1 for i, d in enumerate(change_sorted) if d['symbol'] == current['symbol']), None)
        if change_rank:
            color = COLORS['gain'] if change_rank <= 3 else COLORS['loss'] if change_rank > total - 3 else COLORS['text_secondary']
            self.peer_rankings['change_rank'].config(text=f"#{change_rank}/{total}", foreground=color)
    
    def _generate_peer_insights(self, current_data: Dict, peer_data: List[Dict]):
        """Generate peer insights."""
        insights = []
        
        if not current_data or not peer_data:
            return
        
        current = [d for d in peer_data if d['is_current']]
        if not current:
            return
        current = current[0]
        
        # Highest dividend?
        div_sorted = sorted([d for d in peer_data if d['div'] is not None], key=lambda x: x['div'], reverse=True)
        if div_sorted and current['symbol'] == div_sorted[0]['symbol']:
            insights.append(f"🏆 Highest dividend yield ({current['div']:.1f}%) among peers!")
        
        # Best YTD?
        ytd_sorted = sorted([d for d in peer_data if d['ytd'] is not None], key=lambda x: x['ytd'], reverse=True)
        if ytd_sorted and current['symbol'] == ytd_sorted[0]['symbol']:
            insights.append(f"🚀 Best YTD performance ({current['ytd']:+.1f}%) in sector!")
        
        # Lowest P/E?
        pe_sorted = sorted([d for d in peer_data if d['pe'] and d['pe'] > 0], key=lambda x: x['pe'])
        if pe_sorted and current['symbol'] == pe_sorted[0]['symbol']:
            insights.append(f"💎 Lowest P/E ({current['pe']:.1f}x) - potential value pick!")
        
        # Highest P/E warning
        if pe_sorted and current['symbol'] == pe_sorted[-1]['symbol']:
            insights.append(f"⚠️ Highest P/E ({current['pe']:.1f}x) - premium valuation")
        
        # Largest MCap
        mcap_sorted = sorted(peer_data, key=lambda x: x['mcap'], reverse=True)
        if mcap_sorted and current['symbol'] == mcap_sorted[0]['symbol']:
            insights.append("🏢 Largest company by market cap in sector")
        
        # Today's best performer?
        change_sorted = sorted(peer_data, key=lambda x: x['change'], reverse=True)
        if change_sorted and current['symbol'] == change_sorted[0]['symbol']:
            insights.append(f"📈 Today's best performer ({current['change']:+.1f}%) in sector!")
        
        # Fill insight labels
        for i, label in enumerate(self.peer_insight_labels):
            if i < len(insights):
                label.config(text=insights[i])
            else:
                label.config(text="")
    
    def _populate_peer_table(self, peer_data: List[Dict]):
        """Populate enhanced peer table."""
        for item in self.peers_tree.get_children():
            self.peers_tree.delete(item)
        
        # Sort by market cap
        sorted_data = sorted(peer_data, key=lambda x: x['mcap'], reverse=True)
        
        for d in sorted_data:
            mcap = d['mcap']
            mcap_text = f"₦{mcap/1e9:.1f}B" if mcap >= 1e9 else f"₦{mcap/1e6:.0f}M"
            
            # Signal
            rec = d.get('rec')
            if rec is not None:
                signal = "🟢 BUY" if rec >= 0.5 else "🔴 SELL" if rec <= -0.5 else "🟡 HOLD"
            else:
                signal = "--"
            
            tag = 'current' if d['is_current'] else ''
            
            self.peers_tree.insert('', 'end', values=(
                d['symbol'],
                d['name'][:12] if d['name'] else '--',
                f"₦{d['price']:,.2f}",
                f"{d['change']:+.1f}%",
                f"{d['pe']:.1f}" if d['pe'] else "--",
                f"{d['pb']:.1f}" if d['pb'] else "--",
                f"₦{d['eps']:.2f}" if d['eps'] else "--",
                f"{d['div']:.1f}%" if d['div'] else "--",
                mcap_text,
                f"{d['ytd']:+.1f}%" if d['ytd'] else "--",
                signal
            ), tags=(tag,))
    
    def _update_fairvalue_tab(self):
        """Update Super Enhanced Fair Value Calculator tab."""
        try:
            data = self._get_stock_data(self.current_symbol)
            if not data:
                return
            
            close = data.get('close', 0) or 0
            eps = data.get('earnings_per_share_basic_ttm')
            pb = data.get('price_book_ratio')
            div = data.get('dividend_yield_recent')
            
            self.fv_cards['current_price'].config(text=f"₦{close:,.0f}")
            
            # Reset all method labels first
            for key in self.method_labels:
                self.method_labels[key].config(text="₦--", foreground=COLORS['text_muted'])
            
            # Reset summary cards
            self.fv_cards['fair_value'].config(text="₦--")
            self.fv_cards['upside'].config(text="--%", foreground=COLORS['text_muted'])
            self.fv_cards['margin_safety'].config(text="--%", foreground=COLORS['text_muted'])
            
            # Collect fair values from all methods
            fair_values = {}
            
            # Get sector P/E
            sector = self._get_stock_sector(self.current_symbol)
            sector_pe = 15  # Default
            
            if sector:
                sector_stocks = self._get_sector_stocks(sector)
                pe_values = []
                for sym in sector_stocks:
                    s_data = self._get_stock_data(sym)
                    if s_data:
                        s_pe = s_data.get('price_earnings_ttm')
                        if s_pe and s_pe > 0:
                            pe_values.append(s_pe)
                if pe_values:
                    sector_pe = sum(pe_values) / len(pe_values)
            
            # Method 1: Graham Number (simplified)
            if eps is not None and eps > 0:
                # Simplified Graham = sqrt(22.5 * EPS * BV) ≈ EPS * 15 when BV = EPS
                graham_value = eps * 15
                fair_values['graham'] = graham_value
                self.method_labels['graham'].config(text=f"₦{graham_value:,.0f}", foreground=COLORS['text_secondary'])
            
            # Method 2: P/E Based (Sector)
            if eps is not None and eps > 0:
                pe_sector_value = eps * sector_pe
                fair_values['pe_sector'] = pe_sector_value
                self.method_labels['pe_sector'].config(text=f"₦{pe_sector_value:,.0f}", foreground=COLORS['text_secondary'])
            
            # Method 3: P/E (Fair = 15)
            if eps is not None and eps > 0:
                pe_fair_value = eps * 15
                fair_values['pe_fair'] = pe_fair_value
                self.method_labels['pe_fair'].config(text=f"₦{pe_fair_value:,.0f}", foreground=COLORS['text_secondary'])
            
            # Method 4: DDM (Dividend Discount Model)
            if div is not None and div > 0 and close > 0:
                # DDM: P = D / (r - g), where r=12%, g=3%
                annual_div = close * (div / 100)
                ddm_value = annual_div / (0.12 - 0.03)  # 12% required return, 3% growth
                fair_values['ddm'] = ddm_value
                self.method_labels['ddm'].config(text=f"₦{ddm_value:,.0f}", foreground=COLORS['text_secondary'])
            else:
                self.method_labels['ddm'].config(text="N/A", foreground=COLORS['text_muted'])
            
            # Method 5: Book Value
            if pb is not None and pb > 0 and close > 0:
                book_value = close / pb
                fair_values['book'] = book_value
                self.method_labels['book'].config(text=f"₦{book_value:,.0f}", foreground=COLORS['text_secondary'])
            else:
                self.method_labels['book'].config(text="N/A", foreground=COLORS['text_muted'])
            
            # Method 6: Earnings Power Value (EPS * 10)
            if eps is not None and eps > 0:
                epv_value = eps * 10
                fair_values['epv'] = epv_value
                self.method_labels['epv'].config(text=f"₦{epv_value:,.0f}", foreground=COLORS['text_secondary'])
            
            # Calculate average fair value
            if fair_values:
                avg_fv = sum(fair_values.values()) / len(fair_values)
                min_fv = min(fair_values.values())
                max_fv = max(fair_values.values())
                
                self.fv_cards['fair_value'].config(text=f"₦{avg_fv:,.0f}")
                
                # Upside/Downside
                if close > 0:
                    upside = ((avg_fv - close) / close) * 100
                    margin = ((avg_fv - close) / avg_fv) * 100 if avg_fv > 0 else 0
                    
                    upside_color = COLORS['gain'] if upside > 0 else COLORS['loss']
                    self.fv_cards['upside'].config(text=f"{upside:+.0f}%", foreground=upside_color)
                    
                    margin_color = COLORS['gain'] if margin > 20 else COLORS['warning'] if margin > 0 else COLORS['loss']
                    self.fv_cards['margin_safety'].config(text=f"{margin:+.0f}%", foreground=margin_color)
                    
                    # Draw gauges and charts
                    self._draw_fv_gauge(close, avg_fv, min_fv, max_fv)
                    self._draw_fv_range_chart(close, fair_values)
                    
                    # Confidence
                    methods_used = len(fair_values)
                    confidence = ['🔴 Low', '🔴 Low', '🟠 Medium', '🟡 Good', '🟢 High', '🟢 Very High', '✨ Excellent'][min(methods_used, 6)]
                    self.fv_confidence_label.config(text=f"Confidence: {confidence}")
                    self.fv_methods_used.config(text=f"Methods used: {methods_used}/6")
                    
                    # Recommendation
                    self._generate_fv_recommendation(upside, margin, methods_used)
                    
        except Exception as e:
            logger.error(f"Error updating fair value tab: {e}")
    
    def _draw_fv_gauge(self, current: float, fair: float, min_fv: float, max_fv: float):
        """Draw fair value gauge."""
        canvas = self.fv_gauge_canvas
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 100 or fair <= 0:
            return
        
        # Gauge showing current vs fair
        gauge_y = h // 2
        gauge_height = 14
        
        # Draw gradient bar
        canvas.create_rectangle(20, gauge_y - gauge_height//2, w - 20, gauge_y + gauge_height//2,
                               fill='#333333', outline='')
        
        # Calculate position (0.5 = fair value in center)
        # Left = overvalued, Right = undervalued
        ratio = current / fair if fair > 0 else 1
        # Clamp ratio to 0.5 - 1.5 range
        ratio = max(0.5, min(1.5, ratio))
        # Convert to 0-1 position (1.5 = 0%, 0.5 = 100%)
        pct = (1.5 - ratio) / 1.0
        
        current_x = 20 + pct * (w - 40)
        
        # Color based on valuation
        if ratio < 0.8:
            color = COLORS['gain']  # Undervalued
        elif ratio > 1.2:
            color = COLORS['loss']  # Overvalued
        else:
            color = COLORS['warning']  # Fair
        
        # Draw marker
        canvas.create_polygon(
            current_x - 6, gauge_y - gauge_height//2 - 4,
            current_x + 6, gauge_y - gauge_height//2 - 4,
            current_x, gauge_y - gauge_height//2 + 4,
            fill=color, outline=''
        )
        
        # Labels
        canvas.create_text(20, gauge_y, text="Overvalued", fill=COLORS['loss'], font=('Arial', 7), anchor='w')
        canvas.create_text(w//2, gauge_y, text="Fair", fill=COLORS['warning'], font=('Arial', 7), anchor='center')
        canvas.create_text(w - 20, gauge_y, text="Undervalued", fill=COLORS['gain'], font=('Arial', 7), anchor='e')
    
    def _draw_fv_range_chart(self, current: float, fair_values: Dict):
        """Draw valuation range chart."""
        canvas = self.fv_range_canvas
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 100 or not fair_values:
            return
        
        values = list(fair_values.values())
        min_fv = min(values)
        max_fv = max(values)
        avg_fv = sum(values) / len(values)
        
        # Extend range to include current price
        all_values = values + [current]
        chart_min = min(all_values) * 0.8
        chart_max = max(all_values) * 1.2
        
        if chart_max <= chart_min:
            return
        
        # Y positions
        bar_y = h // 2
        bar_height = 16
        
        def x_pos(val):
            return 60 + ((val - chart_min) / (chart_max - chart_min)) * (w - 120)
        
        # Draw range bar
        min_x = x_pos(min_fv)
        max_x = x_pos(max_fv)
        avg_x = x_pos(avg_fv)
        current_x = x_pos(current)
        
        # Range fill
        canvas.create_rectangle(min_x, bar_y - bar_height//2, max_x, bar_y + bar_height//2,
                               fill='#2a4a2a', outline='')
        
        # Average line
        canvas.create_line(avg_x, bar_y - bar_height//2 - 5, avg_x, bar_y + bar_height//2 + 5,
                          fill=COLORS['gain'], width=2)
        canvas.create_text(avg_x, bar_y - bar_height//2 - 10, text=f"Avg: ₦{avg_fv:,.0f}",
                          fill=COLORS['gain'], font=('Arial', 7), anchor='s')
        
        # Current price line
        canvas.create_line(current_x, bar_y - bar_height//2 - 5, current_x, bar_y + bar_height//2 + 5,
                          fill=COLORS['primary'], width=2)
        canvas.create_text(current_x, bar_y + bar_height//2 + 10, text=f"Now: ₦{current:,.0f}",
                          fill=COLORS['primary'], font=('Arial', 7), anchor='n')
        
        # Min/Max labels
        canvas.create_text(min_x, bar_y, text=f"₦{min_fv:,.0f}", fill=COLORS['text_muted'], font=('Arial', 7), anchor='e')
        canvas.create_text(max_x, bar_y, text=f"₦{max_fv:,.0f}", fill=COLORS['text_muted'], font=('Arial', 7), anchor='w')
    
    def _generate_fv_recommendation(self, upside: float, margin: float, methods: int):
        """Generate investment recommendation."""
        if upside > 30:
            rec = "🟢 STRONG BUY"
            rec_color = COLORS['gain']
            details = f"Significant upside of {upside:.0f}% with {margin:.0f}% margin of safety"
        elif upside > 15:
            rec = "🟢 BUY"
            rec_color = COLORS['gain']
            details = f"Good upside of {upside:.0f}% to fair value"
        elif upside > 5:
            rec = "🟡 ACCUMULATE"
            rec_color = COLORS['warning']
            details = f"Modest upside of {upside:.0f}% - consider adding on dips"
        elif upside > -5:
            rec = "🟡 HOLD"
            rec_color = COLORS['warning']
            details = "Trading near fair value"
        elif upside > -15:
            rec = "🟠 REDUCE"
            rec_color = '#FF8C00'
            details = f"Slightly overvalued by {abs(upside):.0f}%"
        else:
            rec = "🔴 SELL"
            rec_color = COLORS['loss']
            details = f"Significantly overvalued by {abs(upside):.0f}%"
        
        self.recommendation_label.config(text=rec, foreground=rec_color)
        self.rec_details_label.config(text=details)
        
        # Considerations
        considerations = []
        if methods < 4:
            considerations.append("Limited data (only " + str(methods) + " methods)")
        if margin < 0:
            considerations.append("Negative margin of safety")
        if margin > 30:
            considerations.append("Strong margin of safety")
        
        self.rec_considerations.config(text=" • ".join(considerations) if considerations else "")
    
    def _update_dividends_tab(self):
        """Update Super Enhanced Dividends tab."""
        try:
            data = self._get_stock_data(self.current_symbol)
            if not data:
                return
            
            close = data.get('close', 0) or 0
            div_yield = data.get('dividend_yield_recent')
            
            # Reset cards
            for key in self.div_cards:
                self.div_cards[key].config(text="--", foreground=COLORS['text_secondary'])
            
            # ========== Summary Cards ==========
            if div_yield is not None:
                div_color = COLORS['gain'] if div_yield > 3 else COLORS['text_secondary']
                self.div_cards['yield'].config(text=f"{div_yield:.1f}%", foreground=div_color)
                
                # Annual dividend per share
                annual_div = close * (div_yield / 100)
                self.div_cards['annual_div'].config(text=f"₦{annual_div:.2f}")
            else:
                self.div_cards['yield'].config(text="N/A", foreground=COLORS['text_muted'])
            
            # Collect sector data
            sector = self._get_stock_sector(self.current_symbol)
            if not sector:
                return
            
            sector_stocks = self._get_sector_stocks(sector)
            div_data = []
            
            for sym in sector_stocks:
                s_data = self._get_stock_data(sym)
                if s_data:
                    s_div = s_data.get('dividend_yield_recent')
                    div_data.append({
                        'symbol': sym,
                        'name': s_data.get('name', sym),
                        'price': s_data.get('close', 0) or 0,
                        'div': s_div,
                        'pe': s_data.get('price_earnings_ttm'),
                        'rec': s_data.get('Recommend.All'),
                        'is_current': sym == self.current_symbol
                    })
            
            # ========== Sector Average & Ranking ==========
            div_values = [d['div'] for d in div_data if d['div'] is not None and d['div'] > 0]
            
            if div_values:
                avg_div = sum(div_values) / len(div_values)
                self.div_cards['sector_yield'].config(text=f"{avg_div:.1f}%")
                
                if div_yield is not None:
                    diff = div_yield - avg_div
                    diff_color = COLORS['gain'] if diff > 0 else COLORS['loss']
                    self.div_cards['vs_sector'].config(text=f"{diff:+.1f}%", foreground=diff_color)
                    
                    # Rank
                    sorted_divs = sorted(div_values, reverse=True)
                    rank = 1
                    for i, d in enumerate(sorted_divs):
                        if d <= div_yield:
                            rank = i + 1
                            break
                        rank = i + 2
                    total = len(div_values)
                    rank_color = COLORS['gain'] if rank <= 3 else COLORS['text_secondary']
                    self.div_cards['rank'].config(text=f"#{rank}/{total}", foreground=rank_color)
                    
                    # Status
                    if div_yield > avg_div * 1.5:
                        status = "🟢 High"
                        status_color = COLORS['gain']
                    elif div_yield > avg_div:
                        status = "🟢 Above"
                        status_color = COLORS['gain']
                    elif div_yield > avg_div * 0.5:
                        status = "🟡 Below"
                        status_color = COLORS['warning']
                    elif div_yield > 0:
                        status = "🔴 Low"
                        status_color = COLORS['loss']
                    else:
                        status = "❌ None"
                        status_color = COLORS['text_muted']
                    
                    self.div_cards['status'].config(text=status, foreground=status_color)
            
            # ========== Dividend Quality Gauge ==========
            self._draw_dividend_gauge(div_yield, div_values)
            
            # ========== Top 3 Payers ==========
            self._update_top_dividend_payers(div_data)
            
            # ========== Dividend Insights ==========
            self._generate_dividend_insights(div_yield, avg_div if div_values else 0, div_data)
            
            # ========== Income Calculator Auto-Update ==========
            self._calculate_dividend_income()
            
            # ========== Enhanced Table ==========
            self._populate_dividend_table(div_data)
                        
        except Exception as e:
            logger.error(f"Error updating dividends tab: {e}")
    
    def _draw_dividend_gauge(self, div_yield: float, all_divs: List[float]):
        """Draw dividend quality gauge."""
        canvas = self.div_gauge_canvas
        canvas.delete('all')
        canvas.update_idletasks()
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 100:
            return
        
        # Calculate quality score (0-100)
        if div_yield is None or div_yield <= 0:
            score = 0
            quality = "No Dividend"
        elif div_yield >= 10:
            score = 100
            quality = "Exceptional"
        elif div_yield >= 7:
            score = 80
            quality = "High Yield"
        elif div_yield >= 5:
            score = 65
            quality = "Good Yield"
        elif div_yield >= 3:
            score = 50
            quality = "Moderate"
        elif div_yield >= 1:
            score = 30
            quality = "Low Yield"
        else:
            score = 15
            quality = "Very Low"
        
        # Draw gauge
        gauge_y = h // 2
        gauge_height = 14
        
        # Gradient segments
        segments = 20
        segment_w = (w - 40) / segments
        
        for i in range(segments):
            x1 = 20 + i * segment_w
            x2 = x1 + segment_w
            
            pct = i / segments
            if pct < 0.3:
                r, g, b = 220, int(80 + pct * 3 * 100), 80
            elif pct < 0.6:
                r, g, b = int(220 - (pct - 0.3) * 3 * 100), 200, 80
            else:
                r, g, b = 80, 200, int(80 + (pct - 0.6) * 2.5 * 100)
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_rectangle(x1, gauge_y - gauge_height//2, x2, gauge_y + gauge_height//2,
                                   fill=color, outline='')
        
        # Draw pointer
        pointer_x = 20 + (score / 100) * (w - 40)
        canvas.create_polygon(
            pointer_x - 6, gauge_y - gauge_height//2 - 4,
            pointer_x + 6, gauge_y - gauge_height//2 - 4,
            pointer_x, gauge_y - gauge_height//2 + 3,
            fill='white', outline=''
        )
        
        canvas.create_text(20, gauge_y, text="0%", fill=COLORS['loss'], font=('Arial', 7), anchor='w')
        canvas.create_text(w - 20, gauge_y, text="10%+", fill=COLORS['gain'], font=('Arial', 7), anchor='e')
        
        self.div_quality_label.config(text=f"Quality: {quality} ({score}/100)")
    
    def _update_top_dividend_payers(self, div_data: List[Dict]):
        """Update top 3 dividend payers."""
        # Sort by dividend yield
        sorted_data = sorted([d for d in div_data if d['div'] is not None], 
                           key=lambda x: x['div'] or 0, reverse=True)
        
        for i, label in enumerate(self.top_div_labels):
            if i < len(sorted_data):
                d = sorted_data[i]
                medal = ["🥇", "🥈", "🥉"][i]
                label.config(text=f"{medal} {d['symbol']} ({d['div']:.1f}%)")
            else:
                label.config(text=f"{i+1}. --")
    
    def _generate_dividend_insights(self, div_yield: float, avg_div: float, div_data: List[Dict]):
        """Generate dividend insights."""
        insights = []
        
        current = [d for d in div_data if d['is_current']]
        if not current:
            return
        current = current[0]
        
        # Check if top payer
        sorted_data = sorted([d for d in div_data if d['div'] is not None], 
                           key=lambda x: x['div'] or 0, reverse=True)
        
        if sorted_data and current['symbol'] == sorted_data[0]['symbol']:
            insights.append("🏆 Highest dividend yield in sector!")
        
        # Above/below average
        if div_yield is not None and avg_div > 0:
            if div_yield > avg_div * 1.5:
                insights.append(f"📈 50%+ above sector average yield")
            elif div_yield > avg_div:
                insights.append(f"✅ Above sector average ({avg_div:.1f}%)")
            elif div_yield < avg_div * 0.5:
                insights.append(f"⚠️ Well below sector average")
        
        # High yield warning
        if div_yield is not None and div_yield > 8:
            insights.append("⚠️ Very high yield - check sustainability")
        
        # Fill labels
        for i, label in enumerate(self.div_insight_labels):
            if i < len(insights):
                label.config(text=insights[i])
            else:
                label.config(text="")
    
    def _populate_dividend_table(self, div_data: List[Dict]):
        """Populate dividend comparison table."""
        for item in self.div_tree.get_children():
            self.div_tree.delete(item)
        
        # Sort by dividend yield
        sorted_data = sorted(div_data, key=lambda x: x['div'] or 0, reverse=True)
        
        for d in sorted_data:
            price = d['price']
            div = d['div']
            annual = price * (div / 100) if div and price else 0
            
            # Signal
            rec = d.get('rec')
            if rec is not None:
                signal = "🟢 BUY" if rec >= 0.5 else "🔴 SELL" if rec <= -0.5 else "🟡 HOLD"
            else:
                signal = "--"
            
            tag = 'current' if d['is_current'] else ('high_yield' if div and div > 5 else '')
            
            self.div_tree.insert('', 'end', values=(
                d['symbol'],
                d['name'][:12] if d['name'] else '--',
                f"₦{price:,.0f}",
                f"{div:.1f}%" if div else "--",
                f"₦{annual:.2f}" if annual else "--",
                f"{d['pe']:.1f}" if d['pe'] else "--",
                signal
            ), tags=(tag,))
    
    def refresh(self):
        """Refresh the tab."""
        self._refresh_data()
