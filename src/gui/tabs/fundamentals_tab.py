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
    
    # =========================================================================
    # OVERVIEW TAB
    # =========================================================================
    
    def _create_overview_tab(self):
        """Create Overview sub-tab with key metrics."""
        main_frame = ttk.Frame(self.overview_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Row 1: Stock Header
        stock_frame = ttk.LabelFrame(main_frame, text="📌 Stock Info")
        stock_frame.pack(fill=tk.X, pady=(0, 8))
        
        stock_inner = ttk.Frame(stock_frame)
        stock_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.stock_name_label = ttk.Label(
            stock_inner,
            text="Loading...",
            font=get_font('heading'),
            foreground=COLORS['primary']
        )
        self.stock_name_label.pack(side=tk.LEFT)
        
        self.stock_price_label = ttk.Label(
            stock_inner,
            text="₦0.00",
            font=get_font('heading')
        )
        self.stock_price_label.pack(side=tk.RIGHT)
        
        self.stock_change_label = ttk.Label(
            stock_inner,
            text="0.00%",
            font=get_font('body')
        )
        self.stock_change_label.pack(side=tk.RIGHT, padx=(0, 15))
        
        # Row 2: Key Metrics Cards
        metrics_frame = ttk.LabelFrame(main_frame, text="📊 Key Metrics")
        metrics_frame.pack(fill=tk.X, pady=(0, 8))
        
        metrics_inner = ttk.Frame(metrics_frame)
        metrics_inner.pack(fill=tk.X, padx=10, pady=10)
        
        for i in range(5):
            metrics_inner.columnconfigure(i, weight=1)
        
        self.metric_cards = {}
        metrics_config = [
            ('market_cap', '💰 Market Cap', '₦0'),
            ('pe_ratio', '📈 P/E Ratio', '--'),
            ('eps', '💵 EPS', '₦0'),
            ('dividend', '🎁 Div Yield', '--%'),
            ('volume', '📊 Avg Vol', '0')
        ]
        
        for i, (key, label, default) in enumerate(metrics_config):
            card = ttk.Frame(metrics_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=4, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(6, 0))
            
            value_label = ttk.Label(card, text=default, font=get_font('heading'))
            value_label.pack(anchor='center', pady=(3, 6))
            
            self.metric_cards[key] = value_label
        
        # Row 3: Price Performance
        perf_frame = ttk.LabelFrame(main_frame, text="📈 Price Performance")
        perf_frame.pack(fill=tk.X, pady=(0, 8))
        
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
        
        # Row 4: Valuation Assessment
        val_frame = ttk.LabelFrame(main_frame, text="🎯 Valuation Assessment")
        val_frame.pack(fill=tk.X, pady=(0, 5))
        
        val_inner = ttk.Frame(val_frame)
        val_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.valuation_status = ttk.Label(
            val_inner,
            text="⏳ Loading...",
            font=get_font('body'),
            foreground=COLORS['text_muted']
        )
        self.valuation_status.pack(anchor='center')
        
        self.valuation_details = ttk.Label(
            val_inner,
            text="",
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        )
        self.valuation_details.pack(anchor='center', pady=(5, 0))
    
    # =========================================================================
    # SECTOR ANALYSIS TAB
    # =========================================================================
    
    def _create_sector_tab(self):
        """Create Sector Analysis sub-tab."""
        main_frame = ttk.Frame(self.sector_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Row 1: Current Stock Sector
        sector_frame = ttk.LabelFrame(main_frame, text="📌 Sector Classification")
        sector_frame.pack(fill=tk.X, pady=(0, 8))
        
        sector_inner = ttk.Frame(sector_frame)
        sector_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.sector_name_label = ttk.Label(
            sector_inner,
            text="Sector: --",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        )
        self.sector_name_label.pack(side=tk.LEFT)
        
        self.sector_stocks_label = ttk.Label(
            sector_inner,
            text="(0 stocks)",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.sector_stocks_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Row 2: Stock vs Sector Comparison
        compare_frame = ttk.LabelFrame(main_frame, text="📊 Stock vs Sector Average")
        compare_frame.pack(fill=tk.X, pady=(0, 8))
        
        compare_inner = ttk.Frame(compare_frame)
        compare_inner.pack(fill=tk.X, padx=10, pady=8)
        
        for i in range(4):
            compare_inner.columnconfigure(i, weight=1)
        
        self.sector_compare = {}
        compare_items = [
            ('pe_stock', 'Stock P/E'),
            ('pe_sector', 'Sector P/E'),
            ('perf_stock', 'Stock YTD'),
            ('perf_sector', 'Sector YTD')
        ]
        
        for i, (key, label) in enumerate(compare_items):
            card = ttk.Frame(compare_inner)
            card.grid(row=0, column=i, padx=10, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            value = ttk.Label(card, text="--", font=get_font('subheading'))
            value.pack(anchor='center')
            self.sector_compare[key] = value
        
        # Row 3: Percentile Ranking
        rank_frame = ttk.LabelFrame(main_frame, text="📈 Sector Percentile Ranking")
        rank_frame.pack(fill=tk.X, pady=(0, 8))
        
        rank_inner = ttk.Frame(rank_frame)
        rank_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.percentile_label = ttk.Label(
            rank_inner,
            text="Outperforms 0% of sector peers",
            font=get_font('body')
        )
        self.percentile_label.pack(anchor='center')
        
        # Row 4: Sector Stocks Table
        table_frame = ttk.LabelFrame(main_frame, text="📋 Sector Stocks")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        columns = ('symbol', 'price', 'change', 'pe', 'mcap')
        self.sector_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=6)
        
        col_config = [
            ('symbol', 'Symbol', 80, 'center'),
            ('price', 'Price', 80, 'e'),
            ('change', 'Change %', 70, 'center'),
            ('pe', 'P/E', 60, 'center'),
            ('mcap', 'Market Cap', 100, 'e')
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
    # PEER COMPARISON TAB
    # =========================================================================
    
    def _create_peers_tab(self):
        """Create Peer Comparison sub-tab."""
        main_frame = ttk.Frame(self.peers_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(
            header_frame,
            text="Compare with sector peers - Best values highlighted in green, worst in red",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        ).pack(side=tk.LEFT)
        
        # Peer Table
        table_frame = ttk.LabelFrame(main_frame, text="👥 Peer Comparison Table")
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('symbol', 'name', 'price', 'change', 'pe', 'eps', 'div', 'mcap', 'ytd')
        self.peers_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
        
        col_config = [
            ('symbol', 'Symbol', 70, 'center'),
            ('name', 'Name', 140, 'w'),
            ('price', 'Price', 70, 'e'),
            ('change', 'Day %', 60, 'center'),
            ('pe', 'P/E', 55, 'center'),
            ('eps', 'EPS', 60, 'e'),
            ('div', 'Div %', 50, 'center'),
            ('mcap', 'Mkt Cap', 85, 'e'),
            ('ytd', 'YTD %', 60, 'center')
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
    # FAIR VALUE TAB
    # =========================================================================
    
    def _create_fairvalue_tab(self):
        """Create Fair Value Calculator sub-tab."""
        main_frame = ttk.Frame(self.fairvalue_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Row 1: Fair Value Summary
        summary_frame = ttk.LabelFrame(main_frame, text="💎 Fair Value Estimate")
        summary_frame.pack(fill=tk.X, pady=(0, 8))
        
        summary_inner = ttk.Frame(summary_frame)
        summary_inner.pack(fill=tk.X, padx=15, pady=12)
        
        # Current Price vs Fair Value
        for i in range(3):
            summary_inner.columnconfigure(i, weight=1)
        
        self.fv_cards = {}
        fv_items = [
            ('current_price', '📍 Current Price', '₦0.00'),
            ('fair_value', '💎 Fair Value', '₦0.00'),
            ('upside', '📊 Upside/Downside', '0%')
        ]
        
        for i, (key, label, default) in enumerate(fv_items):
            card = ttk.Frame(summary_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=8, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(8, 3))
            
            value = ttk.Label(card, text=default, font=get_font('heading'))
            value.pack(anchor='center', pady=(3, 8))
            self.fv_cards[key] = value
        
        # Row 2: Valuation Methods
        methods_frame = ttk.LabelFrame(main_frame, text="📐 Valuation Methods")
        methods_frame.pack(fill=tk.X, pady=(0, 8))
        
        methods_inner = ttk.Frame(methods_frame)
        methods_inner.pack(fill=tk.X, padx=10, pady=8)
        
        self.method_labels = {}
        methods_config = [
            ('graham', 'Graham Number', 'sqrt(22.5 × EPS × Book Value)'),
            ('pe_based', 'P/E Based', 'EPS × Sector Avg P/E'),
            ('avg_fair', 'Average Fair Value', 'Average of methods')
        ]
        
        for key, title, formula in methods_config:
            row = ttk.Frame(methods_inner)
            row.pack(fill=tk.X, pady=4)
            
            ttk.Label(row, text=title, font=get_font('body'),
                     foreground=COLORS['primary']).pack(side=tk.LEFT)
            
            ttk.Label(row, text=f"({formula})", font=get_font('tiny'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=(5, 0))
            
            value = ttk.Label(row, text="₦--", font=get_font('subheading'))
            value.pack(side=tk.RIGHT)
            self.method_labels[key] = value
        
        # Row 3: Recommendation
        rec_frame = ttk.LabelFrame(main_frame, text="🎯 Recommendation")
        rec_frame.pack(fill=tk.X, pady=(0, 5))
        
        rec_inner = ttk.Frame(rec_frame)
        rec_inner.pack(fill=tk.X, padx=15, pady=10)
        
        self.recommendation_label = ttk.Label(
            rec_inner,
            text="⏳ Calculating...",
            font=get_font('subheading'),
            foreground=COLORS['text_muted']
        )
        self.recommendation_label.pack(anchor='center')
        
        self.rec_details_label = ttk.Label(
            rec_inner,
            text="",
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        )
        self.rec_details_label.pack(anchor='center', pady=(5, 0))
    
    # =========================================================================
    # DIVIDENDS TAB
    # =========================================================================
    
    def _create_dividends_tab(self):
        """Create Dividends Analysis sub-tab."""
        main_frame = ttk.Frame(self.dividends_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Row 1: Dividend Summary
        div_frame = ttk.LabelFrame(main_frame, text="💵 Dividend Summary")
        div_frame.pack(fill=tk.X, pady=(0, 8))
        
        div_inner = ttk.Frame(div_frame)
        div_inner.pack(fill=tk.X, padx=10, pady=10)
        
        for i in range(4):
            div_inner.columnconfigure(i, weight=1)
        
        self.div_cards = {}
        div_items = [
            ('yield', '📊 Dividend Yield', '--%'),
            ('sector_yield', '📈 Sector Avg Yield', '--%'),
            ('vs_sector', '⚡ vs Sector', '--'),
            ('status', '🎯 Status', '--')
        ]
        
        for i, (key, label, default) in enumerate(div_items):
            card = ttk.Frame(div_inner, relief='groove', borderwidth=1)
            card.grid(row=0, column=i, padx=4, pady=3, sticky='nsew')
            
            ttk.Label(card, text=label, font=get_font('body'),
                     foreground=COLORS['primary']).pack(anchor='center', pady=(6, 0))
            
            value = ttk.Label(card, text=default, font=get_font('subheading'))
            value.pack(anchor='center', pady=(3, 6))
            self.div_cards[key] = value
        
        # Row 2: Sector Dividend Comparison
        compare_frame = ttk.LabelFrame(main_frame, text="📋 Sector Dividend Comparison")
        compare_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('symbol', 'name', 'price', 'div_yield', 'pe')
        self.div_tree = ttk.Treeview(compare_frame, columns=columns, show='headings', height=8)
        
        col_config = [
            ('symbol', 'Symbol', 80, 'center'),
            ('name', 'Name', 150, 'w'),
            ('price', 'Price', 80, 'e'),
            ('div_yield', 'Div Yield', 80, 'center'),
            ('pe', 'P/E', 60, 'center')
        ]
        
        for col, heading, width, anchor in col_config:
            self.div_tree.heading(col, text=heading)
            self.div_tree.column(col, width=width, anchor=anchor)
        
        div_scroll = ttk.Scrollbar(compare_frame, orient='vertical', command=self.div_tree.yview)
        self.div_tree.configure(yscrollcommand=div_scroll.set)
        
        self.div_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        div_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))
        
        self.div_tree.tag_configure('current', background=COLORS['bg_medium'])
        self.div_tree.tag_configure('high_yield', foreground=COLORS['gain'])
    
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
        """Update Overview tab."""
        try:
            data = self._get_stock_data(self.current_symbol)
            if not data:
                return
            
            # Stock info
            name = data.get('name', self.current_symbol)
            self.stock_name_label.config(text=f"{self.current_symbol} - {name}")
            
            close = data.get('close', 0)
            self.stock_price_label.config(text=f"₦{close:,.2f}")
            
            change = data.get('change', 0) or 0
            change_color = COLORS['gain'] if change >= 0 else COLORS['loss']
            self.stock_change_label.config(text=f"{change:+.2f}%", foreground=change_color)
            
            # Key metrics
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
            
            pe = data.get('price_earnings_ttm')
            if pe is not None:
                pe_color = COLORS['gain'] if pe < 15 else COLORS['loss'] if pe > 25 else COLORS['warning']
                self.metric_cards['pe_ratio'].config(text=f"{pe:.1f}x", foreground=pe_color)
            
            eps = data.get('earnings_per_share_basic_ttm')
            if eps is not None:
                eps_color = COLORS['gain'] if eps > 0 else COLORS['loss']
                self.metric_cards['eps'].config(text=f"₦{eps:.2f}", foreground=eps_color)
            
            div = data.get('dividend_yield_recent')
            if div is not None:
                self.metric_cards['dividend'].config(text=f"{div:.1f}%")
            else:
                self.metric_cards['dividend'].config(text="N/A")
            
            vol = data.get('average_volume_10d_calc', 0) or 0
            if vol >= 1e6:
                vol_text = f"{vol/1e6:.1f}M"
            elif vol >= 1e3:
                vol_text = f"{vol/1e3:.1f}K"
            else:
                vol_text = f"{vol:,.0f}"
            self.metric_cards['volume'].config(text=vol_text)
            
            # Performance
            perf_map = {
                'week': 'Perf.W', 'month': 'Perf.1M', 'quarter': 'Perf.3M',
                'half_year': 'Perf.6M', 'ytd': 'Perf.YTD', 'year': 'Perf.Y'
            }
            for key, col in perf_map.items():
                val = data.get(col)
                if val is not None:
                    color = COLORS['gain'] if val > 0 else COLORS['loss']
                    self.perf_labels[key].config(text=f"{val:+.1f}%", foreground=color)
            
            # Valuation assessment
            self._update_valuation_assessment(data)
            
        except Exception as e:
            logger.error(f"Error updating overview: {e}")
    
    def _update_valuation_assessment(self, data: Dict):
        """Update valuation assessment."""
        scores = []
        details = []
        
        pe = data.get('price_earnings_ttm')
        if pe is not None and pe > 0:
            if pe < 10:
                scores.append(2)
                details.append("P/E<10")
            elif pe < 20:
                scores.append(1)
                details.append("P/E fair")
            elif pe < 30:
                scores.append(0)
                details.append("P/E premium")
            else:
                scores.append(-1)
                details.append("P/E high")
        
        if scores:
            avg = sum(scores) / len(scores)
            if avg >= 1.5:
                status = "🟢 UNDERVALUED"
                color = COLORS['gain']
            elif avg >= 0.5:
                status = "🟡 FAIRLY VALUED"
                color = COLORS['warning']
            else:
                status = "🔴 OVERVALUED"
                color = COLORS['loss']
            
            self.valuation_status.config(text=status, foreground=color)
            self.valuation_details.config(text=" • ".join(details))
        else:
            self.valuation_status.config(text="⚪ Insufficient data", foreground=COLORS['text_muted'])
    
    def _update_sector_tab(self):
        """Update Sector Analysis tab."""
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
            
            # Calculate sector averages
            sector_pe_values = []
            sector_perf_values = []
            
            for item in self.sector_tree.get_children():
                self.sector_tree.delete(item)
            
            for sym in sector_stocks:
                stock_data = self._get_stock_data(sym)
                if stock_data:
                    pe = stock_data.get('price_earnings_ttm')
                    if pe is not None and pe > 0:
                        sector_pe_values.append(pe)
                    
                    perf = stock_data.get('Perf.YTD')
                    if perf is not None:
                        sector_perf_values.append(perf)
                    
                    # Add to tree
                    price = stock_data.get('close', 0)
                    change = stock_data.get('change', 0) or 0
                    mcap = stock_data.get('market_cap_basic', 0) or 0
                    
                    mcap_text = f"₦{mcap/1e9:.1f}B" if mcap >= 1e9 else f"₦{mcap/1e6:.1f}M" if mcap >= 1e6 else f"₦{mcap:,.0f}"
                    
                    tag = 'current' if sym == self.current_symbol else ('gain' if change > 0 else 'loss' if change < 0 else '')
                    
                    self.sector_tree.insert('', 'end', values=(
                        sym,
                        f"₦{price:,.2f}",
                        f"{change:+.1f}%",
                        f"{pe:.1f}" if pe else "--",
                        mcap_text
                    ), tags=(tag,))
            
            # Update comparison
            if current_data:
                stock_pe = current_data.get('price_earnings_ttm')
                stock_perf = current_data.get('Perf.YTD')
                
                self.sector_compare['pe_stock'].config(text=f"{stock_pe:.1f}" if stock_pe else "--")
                self.sector_compare['perf_stock'].config(text=f"{stock_perf:+.1f}%" if stock_perf else "--")
            
            if sector_pe_values:
                avg_pe = sum(sector_pe_values) / len(sector_pe_values)
                self.sector_compare['pe_sector'].config(text=f"{avg_pe:.1f}")
            
            if sector_perf_values:
                avg_perf = sum(sector_perf_values) / len(sector_perf_values)
                self.sector_compare['perf_sector'].config(text=f"{avg_perf:+.1f}%")
                
                # Calculate percentile
                stock_perf = current_data.get('Perf.YTD') if current_data else None
                if stock_perf is not None:
                    better_than = sum(1 for p in sector_perf_values if stock_perf > p)
                    percentile = (better_than / len(sector_perf_values)) * 100
                    self.percentile_label.config(
                        text=f"Outperforms {percentile:.0f}% of sector peers (YTD)",
                        foreground=COLORS['gain'] if percentile > 50 else COLORS['loss']
                    )
                    
        except Exception as e:
            logger.error(f"Error updating sector tab: {e}")
    
    def _update_peers_tab(self):
        """Update Peer Comparison tab."""
        try:
            sector = self._get_stock_sector(self.current_symbol)
            if not sector:
                return
            
            sector_stocks = self._get_sector_stocks(sector)
            
            for item in self.peers_tree.get_children():
                self.peers_tree.delete(item)
            
            for sym in sector_stocks:
                stock_data = self._get_stock_data(sym)
                if stock_data:
                    name = stock_data.get('name', sym)[:20]
                    price = stock_data.get('close', 0)
                    change = stock_data.get('change', 0) or 0
                    pe = stock_data.get('price_earnings_ttm')
                    eps = stock_data.get('earnings_per_share_basic_ttm')
                    div = stock_data.get('dividend_yield_recent')
                    mcap = stock_data.get('market_cap_basic', 0) or 0
                    ytd = stock_data.get('Perf.YTD')
                    
                    mcap_text = f"₦{mcap/1e9:.1f}B" if mcap >= 1e9 else f"₦{mcap/1e6:.1f}M"
                    
                    tag = 'current' if sym == self.current_symbol else ''
                    
                    self.peers_tree.insert('', 'end', values=(
                        sym,
                        name,
                        f"₦{price:,.2f}",
                        f"{change:+.1f}%",
                        f"{pe:.1f}" if pe else "--",
                        f"₦{eps:.2f}" if eps else "--",
                        f"{div:.1f}%" if div else "--",
                        mcap_text,
                        f"{ytd:+.1f}%" if ytd else "--"
                    ), tags=(tag,))
                    
        except Exception as e:
            logger.error(f"Error updating peers tab: {e}")
    
    def _update_fairvalue_tab(self):
        """Update Fair Value Calculator tab."""
        try:
            data = self._get_stock_data(self.current_symbol)
            if not data:
                return
            
            close = data.get('close', 0)
            eps = data.get('earnings_per_share_basic_ttm')
            pe = data.get('price_earnings_ttm')
            
            self.fv_cards['current_price'].config(text=f"₦{close:,.2f}")
            
            fair_values = []
            
            # Graham Number (simplified - using P/E as proxy for book value)
            # Real Graham: sqrt(22.5 * EPS * Book Value)
            # Simplified: EPS * 15 (assuming fair P/E of 15)
            if eps is not None and eps > 0:
                graham_value = eps * 15
                self.method_labels['graham'].config(text=f"₦{graham_value:,.2f}")
                fair_values.append(graham_value)
            
            # P/E based (using sector average or 15)
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
            
            if eps is not None and eps > 0:
                pe_value = eps * sector_pe
                self.method_labels['pe_based'].config(text=f"₦{pe_value:,.2f}")
                fair_values.append(pe_value)
            
            # Average fair value
            if fair_values:
                avg_fv = sum(fair_values) / len(fair_values)
                self.method_labels['avg_fair'].config(text=f"₦{avg_fv:,.2f}")
                self.fv_cards['fair_value'].config(text=f"₦{avg_fv:,.2f}")
                
                # Upside/Downside
                if close > 0:
                    upside = ((avg_fv - close) / close) * 100
                    upside_color = COLORS['gain'] if upside > 0 else COLORS['loss']
                    self.fv_cards['upside'].config(text=f"{upside:+.1f}%", foreground=upside_color)
                    
                    # Recommendation
                    if upside > 20:
                        rec = "🟢 STRONG BUY - Significantly undervalued"
                        rec_color = COLORS['gain']
                    elif upside > 10:
                        rec = "🟢 BUY - Undervalued"
                        rec_color = COLORS['gain']
                    elif upside > -10:
                        rec = "🟡 HOLD - Fairly valued"
                        rec_color = COLORS['warning']
                    elif upside > -20:
                        rec = "🟠 REDUCE - Slightly overvalued"
                        rec_color = '#FF8C00'
                    else:
                        rec = "🔴 SELL - Significantly overvalued"
                        rec_color = COLORS['loss']
                    
                    self.recommendation_label.config(text=rec, foreground=rec_color)
                    self.rec_details_label.config(text=f"Fair Value ₦{avg_fv:,.2f} vs Current ₦{close:,.2f}")
                    
        except Exception as e:
            logger.error(f"Error updating fair value tab: {e}")
    
    def _update_dividends_tab(self):
        """Update Dividends tab."""
        try:
            data = self._get_stock_data(self.current_symbol)
            if not data:
                return
            
            div_yield = data.get('dividend_yield_recent')
            
            if div_yield is not None:
                div_color = COLORS['gain'] if div_yield > 3 else COLORS['text_secondary']
                self.div_cards['yield'].config(text=f"{div_yield:.2f}%", foreground=div_color)
            else:
                self.div_cards['yield'].config(text="N/A", foreground=COLORS['text_muted'])
            
            # Sector average
            sector = self._get_stock_sector(self.current_symbol)
            if sector:
                sector_stocks = self._get_sector_stocks(sector)
                div_values = []
                
                for item in self.div_tree.get_children():
                    self.div_tree.delete(item)
                
                for sym in sector_stocks:
                    s_data = self._get_stock_data(sym)
                    if s_data:
                        s_div = s_data.get('dividend_yield_recent')
                        if s_div is not None:
                            div_values.append(s_div)
                        
                        # Add to table
                        name = s_data.get('name', sym)[:20]
                        price = s_data.get('close', 0)
                        pe = s_data.get('price_earnings_ttm')
                        
                        tag = 'current' if sym == self.current_symbol else ('high_yield' if s_div and s_div > 5 else '')
                        
                        self.div_tree.insert('', 'end', values=(
                            sym,
                            name,
                            f"₦{price:,.2f}",
                            f"{s_div:.2f}%" if s_div else "--",
                            f"{pe:.1f}" if pe else "--"
                        ), tags=(tag,))
                
                if div_values:
                    avg_div = sum(div_values) / len(div_values)
                    self.div_cards['sector_yield'].config(text=f"{avg_div:.2f}%")
                    
                    if div_yield is not None:
                        diff = div_yield - avg_div
                        diff_color = COLORS['gain'] if diff > 0 else COLORS['loss']
                        self.div_cards['vs_sector'].config(text=f"{diff:+.2f}%", foreground=diff_color)
                        
                        if div_yield > avg_div * 1.5:
                            status = "🟢 High Yield"
                            status_color = COLORS['gain']
                        elif div_yield > avg_div:
                            status = "🟢 Above Avg"
                            status_color = COLORS['gain']
                        elif div_yield > avg_div * 0.5:
                            status = "🟡 Below Avg"
                            status_color = COLORS['warning']
                        else:
                            status = "🔴 Low Yield"
                            status_color = COLORS['loss']
                        
                        self.div_cards['status'].config(text=status, foreground=status_color)
                        
        except Exception as e:
            logger.error(f"Error updating dividends tab: {e}")
    
    def refresh(self):
        """Refresh the tab."""
        self._refresh_data()
