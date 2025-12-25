"""
Stock Screener Tab 2.0 for MetaQuant Nigeria.
Dashboard-style UI with filter chips, quick presets, and action buttons.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Dict, Any
import logging

try:
    import ttkbootstrap as ttk_bs
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.screener.screening_engine import (
    ScreeningEngine, PEFilter, MarketCapFilter, 
    DividendFilter, SectorFilter, VolumeFilter
)
from src.gui.theme import COLORS, get_font, format_currency, format_percent, get_change_color

logger = logging.getLogger(__name__)


class ScreenerTab:
    """Stock screener tab with dashboard-style filters and results."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager, ml_engine=None):
        self.parent = parent
        self.db = db
        self.ml_engine = ml_engine
        self.engine = ScreeningEngine(db)
        
        # State
        self.sort_column = 'change'
        self.sort_ascending = False
        self.current_results = []
        self.active_filters = {}  # {filter_name: display_text}
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        self._setup_tags()
        
        # Load initial data
        self.refresh()
    
    def _setup_ui(self):
        """Setup the screener UI with dashboard layout."""
        # Main container
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # === HERO SECTION ===
        self._create_hero_section(main)
        
        # === QUICK PRESETS BAR ===
        self._create_preset_bar(main)
        
        # === ACTIVE FILTERS CHIPS ===
        self._create_filter_chips(main)
        
        # === MAIN CONTENT: Filters + Results ===
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left: Custom Filters (collapsible)
        self._create_filter_panel(content)
        
        # Right: Results Dashboard
        self._create_results_section(content)
    
    def _create_hero_section(self, parent):
        """Create hero section with market summary."""
        hero = ttk.Frame(parent)
        hero.pack(fill=tk.X, pady=(0, 15))
        
        # Left: Title and description
        left = ttk.Frame(hero)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(
            left, text="📊 Stock Screener",
            font=get_font('heading'),
            foreground=COLORS['text_primary']
        ).pack(anchor='w')
        
        ttk.Label(
            left, text="Find investment opportunities with advanced filters",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        ).pack(anchor='w')
        
        # Right: Market summary cards
        right = ttk.Frame(hero)
        right.pack(side=tk.RIGHT)
        
        # Market summary metrics
        self.hero_cards = {}
        metrics = [
            ('total', '📈 Total Stocks', '0'),
            ('gainers', '🟢 Gainers', '0'),
            ('losers', '🔴 Losers', '0'),
            ('signals', '🎯 Buy Signals', '0')
        ]
        
        for key, label, default in metrics:
            card = self._create_metric_card(right, label, default)
            card.pack(side=tk.LEFT, padx=5)
            self.hero_cards[key] = card
    
    def _create_metric_card(self, parent, label: str, value: str) -> ttk.Frame:
        """Create a small metric card."""
        card = ttk.Frame(parent, style='Card.TFrame')
        
        # Container with padding
        inner = ttk.Frame(card)
        inner.pack(padx=12, pady=8)
        
        lbl = ttk.Label(inner, text=label, font=get_font('small'), 
                       foreground=COLORS['text_muted'])
        lbl.pack()
        
        val = ttk.Label(inner, text=value, font=get_font('subheading'),
                       foreground=COLORS['text_primary'])
        val.pack()
        
        # Store reference to value label
        card.value_label = val
        return card
    
    def _create_preset_bar(self, parent):
        """Create quick preset buttons bar."""
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(bar, text="Quick Screens:", font=get_font('body'),
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT, padx=(0, 10))
        
        presets = [
            ("🚀 Top Gainers", self._apply_top_gainers, 'success'),
            ("📉 Top Losers", self._apply_top_losers, 'danger'),
            ("🔥 Most Active", self._apply_most_active, 'warning'),
            ("💎 Value Stocks", self._apply_value_stocks, 'info'),
            ("💰 High Dividend", self._apply_dividend_stocks, 'primary'),
            ("🎯 ML Buy Signals", self._apply_ml_signals, 'success'),
            ("📊 Momentum", self._apply_momentum, 'warning'),
        ]
        
        for text, command, style in presets:
            if TTKBOOTSTRAP_AVAILABLE:
                btn = ttk_bs.Button(bar, text=text, bootstyle=f"{style}-outline",
                                   command=command, padding=(10, 5))
            else:
                btn = ttk.Button(bar, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=3)
    
    def _create_filter_chips(self, parent):
        """Create active filter chips bar."""
        self.chips_frame = ttk.Frame(parent)
        self.chips_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Label
        self.chips_label = ttk.Label(
            self.chips_frame, text="Active Filters:", 
            font=get_font('small'), foreground=COLORS['text_muted']
        )
        
        # Chips container
        self.chips_container = ttk.Frame(self.chips_frame)
        
        # Clear all button
        if TTKBOOTSTRAP_AVAILABLE:
            self.clear_all_btn = ttk_bs.Button(
                self.chips_frame, text="✕ Clear All", bootstyle="secondary-link",
                command=self._clear_all_filters, padding=(5, 2)
            )
        else:
            self.clear_all_btn = ttk.Button(
                self.chips_frame, text="✕ Clear All", command=self._clear_all_filters
            )
        
        self._update_chips_display()
    
    def _update_chips_display(self):
        """Update the filter chips display."""
        # Clear existing chips
        for widget in self.chips_container.winfo_children():
            widget.destroy()
        
        if self.active_filters:
            self.chips_label.pack(side=tk.LEFT, padx=(0, 5))
            self.chips_container.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            for filter_key, filter_text in self.active_filters.items():
                chip = self._create_chip(filter_key, filter_text)
                chip.pack(side=tk.LEFT, padx=2)
            
            self.clear_all_btn.pack(side=tk.RIGHT)
        else:
            self.chips_label.pack_forget()
            self.chips_container.pack_forget()
            self.clear_all_btn.pack_forget()
    
    def _create_chip(self, key: str, text: str) -> ttk.Frame:
        """Create a single filter chip with remove button."""
        chip = ttk.Frame(self.chips_container)
        
        # Chip label
        lbl = ttk.Label(chip, text=text, font=get_font('small'),
                       foreground=COLORS['primary'], padding=(8, 3))
        lbl.pack(side=tk.LEFT)
        
        # Remove button
        remove_btn = ttk.Label(chip, text="✕", font=get_font('small'),
                              foreground=COLORS['text_muted'], cursor='hand2')
        remove_btn.pack(side=tk.LEFT, padx=(0, 5))
        remove_btn.bind('<Button-1>', lambda e, k=key: self._remove_filter(k))
        
        # Style the chip
        chip.configure(style='Chip.TFrame')
        return chip
    
    def _remove_filter(self, key: str):
        """Remove a specific filter."""
        if key in self.active_filters:
            del self.active_filters[key]
            self._apply_active_filters()
            self._update_chips_display()
    
    def _create_filter_panel(self, parent):
        """Create the expandable filter panel."""
        # Collapsible filter section
        filter_frame = ttk.LabelFrame(parent, text="🔧 Advanced Filters", padding=10)
        filter_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Filter grid
        filters_grid = ttk.Frame(filter_frame)
        filters_grid.pack(fill=tk.BOTH, expand=True)
        
        row = 0
        
        # === FUNDAMENTAL FILTERS ===
        ttk.Label(filters_grid, text="Fundamentals", font=get_font('body'),
                 foreground=COLORS['primary']).grid(row=row, column=0, columnspan=2, 
                                                     sticky='w', pady=(0, 5))
        row += 1
        
        # P/E Ratio
        ttk.Label(filters_grid, text="P/E Ratio:").grid(row=row, column=0, sticky='w', pady=3)
        pe_frame = ttk.Frame(filters_grid)
        pe_frame.grid(row=row, column=1, sticky='e', pady=3)
        self.pe_min_var = tk.StringVar()
        self.pe_max_var = tk.StringVar()
        ttk.Entry(pe_frame, textvariable=self.pe_min_var, width=6).pack(side=tk.LEFT)
        ttk.Label(pe_frame, text=" - ").pack(side=tk.LEFT)
        ttk.Entry(pe_frame, textvariable=self.pe_max_var, width=6).pack(side=tk.LEFT)
        row += 1
        
        # P/B Ratio
        ttk.Label(filters_grid, text="P/B Ratio:").grid(row=row, column=0, sticky='w', pady=3)
        pb_frame = ttk.Frame(filters_grid)
        pb_frame.grid(row=row, column=1, sticky='e', pady=3)
        self.pb_min_var = tk.StringVar()
        self.pb_max_var = tk.StringVar()
        ttk.Entry(pb_frame, textvariable=self.pb_min_var, width=6).pack(side=tk.LEFT)
        ttk.Label(pb_frame, text=" - ").pack(side=tk.LEFT)
        ttk.Entry(pb_frame, textvariable=self.pb_max_var, width=6).pack(side=tk.LEFT)
        row += 1
        
        # Dividend Yield
        ttk.Label(filters_grid, text="Dividend Yield ≥:").grid(row=row, column=0, sticky='w', pady=3)
        self.div_min_var = tk.StringVar()
        div_frame = ttk.Frame(filters_grid)
        div_frame.grid(row=row, column=1, sticky='e', pady=3)
        ttk.Entry(div_frame, textvariable=self.div_min_var, width=6).pack(side=tk.LEFT)
        ttk.Label(div_frame, text="%").pack(side=tk.LEFT)
        row += 1
        
        # ROE
        ttk.Label(filters_grid, text="ROE ≥:").grid(row=row, column=0, sticky='w', pady=3)
        self.roe_min_var = tk.StringVar()
        roe_frame = ttk.Frame(filters_grid)
        roe_frame.grid(row=row, column=1, sticky='e', pady=3)
        ttk.Entry(roe_frame, textvariable=self.roe_min_var, width=6).pack(side=tk.LEFT)
        ttk.Label(roe_frame, text="%").pack(side=tk.LEFT)
        row += 1
        
        # Separator
        ttk.Separator(filters_grid, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # === MARKET FILTERS ===
        ttk.Label(filters_grid, text="Market", font=get_font('body'),
                 foreground=COLORS['primary']).grid(row=row, column=0, columnspan=2, 
                                                     sticky='w', pady=(0, 5))
        row += 1
        
        # Sector
        ttk.Label(filters_grid, text="Sector:").grid(row=row, column=0, sticky='w', pady=3)
        self.sector_var = tk.StringVar(value="All")
        sectors = ["All"] + self.db.get_sectors()
        self.sector_combo = ttk.Combobox(filters_grid, textvariable=self.sector_var,
                                        values=sectors, state="readonly", width=14)
        self.sector_combo.grid(row=row, column=1, sticky='e', pady=3)
        row += 1
        
        # Market Cap
        ttk.Label(filters_grid, text="Market Cap:").grid(row=row, column=0, sticky='w', pady=3)
        self.cap_var = tk.StringVar(value="All")
        cap_options = ["All", "Small (<10B)", "Mid (10-100B)", "Large (>100B)"]
        cap_combo = ttk.Combobox(filters_grid, textvariable=self.cap_var,
                                values=cap_options, state="readonly", width=14)
        cap_combo.grid(row=row, column=1, sticky='e', pady=3)
        row += 1
        
        # Separator
        ttk.Separator(filters_grid, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # === TECHNICAL FILTERS ===
        ttk.Label(filters_grid, text="Technical", font=get_font('body'),
                 foreground=COLORS['primary']).grid(row=row, column=0, columnspan=2, 
                                                     sticky='w', pady=(0, 5))
        row += 1
        
        # Price Change %
        ttk.Label(filters_grid, text="Price Change %:").grid(row=row, column=0, sticky='w', pady=3)
        chg_frame = ttk.Frame(filters_grid)
        chg_frame.grid(row=row, column=1, sticky='e', pady=3)
        self.chg_min_var = tk.StringVar()
        self.chg_max_var = tk.StringVar()
        ttk.Entry(chg_frame, textvariable=self.chg_min_var, width=6).pack(side=tk.LEFT)
        ttk.Label(chg_frame, text=" - ").pack(side=tk.LEFT)
        ttk.Entry(chg_frame, textvariable=self.chg_max_var, width=6).pack(side=tk.LEFT)
        row += 1
        
        # Volume
        ttk.Label(filters_grid, text="Min Volume:").grid(row=row, column=0, sticky='w', pady=3)
        self.vol_min_var = tk.StringVar()
        ttk.Entry(filters_grid, textvariable=self.vol_min_var, width=14).grid(
            row=row, column=1, sticky='e', pady=3)
        row += 1
        
        # ML Signal
        ttk.Label(filters_grid, text="ML Signal:").grid(row=row, column=0, sticky='w', pady=3)
        self.ml_signal_var = tk.StringVar(value="All")
        ml_options = ["All", "BUY Only", "SELL Only", "HOLD Only"]
        ml_combo = ttk.Combobox(filters_grid, textvariable=self.ml_signal_var,
                               values=ml_options, state="readonly", width=14)
        ml_combo.grid(row=row, column=1, sticky='e', pady=3)
        row += 1
        
        # Buttons
        btn_frame = ttk.Frame(filter_frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        if TTKBOOTSTRAP_AVAILABLE:
            apply_btn = ttk_bs.Button(btn_frame, text="🔍 Apply Filters",
                                     bootstyle="success", command=self._apply_custom_filters)
            clear_btn = ttk_bs.Button(btn_frame, text="Clear",
                                     bootstyle="secondary-outline", command=self._clear_all_filters)
        else:
            apply_btn = ttk.Button(btn_frame, text="🔍 Apply Filters", 
                                  command=self._apply_custom_filters)
            clear_btn = ttk.Button(btn_frame, text="Clear", 
                                  command=self._clear_all_filters)
        
        apply_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        clear_btn.pack(side=tk.LEFT)
    
    def _create_results_section(self, parent):
        """Create the results dashboard section."""
        results_frame = ttk.Frame(parent)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results header
        header = ttk.Frame(results_frame)
        header.pack(fill=tk.X, pady=(0, 10))
        
        self.results_count_label = ttk.Label(
            header, text="📋 Results: 0 stocks",
            font=get_font('subheading'), foreground=COLORS['text_primary']
        )
        self.results_count_label.pack(side=tk.LEFT)
        
        # Export button
        if TTKBOOTSTRAP_AVAILABLE:
            export_btn = ttk_bs.Button(header, text="📥 Export", bootstyle="info-outline",
                                      command=self._export_results)
        else:
            export_btn = ttk.Button(header, text="📥 Export", command=self._export_results)
        export_btn.pack(side=tk.RIGHT)
        
        # Results table
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Enhanced columns with signal
        columns = ('symbol', 'name', 'price', 'change', 'pe', 'dividend', 'volume', 'signal', 'action')
        
        self.results_tree = ttk.Treeview(
            table_frame, columns=columns, show='headings', selectmode='browse'
        )
        
        columns_config = [
            ('symbol', 'Symbol', 80),
            ('name', 'Company', 180),
            ('price', 'Price ₦', 90),
            ('change', 'Change %', 85),
            ('pe', 'P/E', 60),
            ('dividend', 'Div %', 60),
            ('volume', 'Volume', 90),
            ('signal', 'Signal', 70),
            ('action', 'Action', 80)
        ]
        
        for col_id, col_text, width in columns_config:
            self.results_tree.heading(
                col_id, text=col_text,
                command=lambda c=col_id: self._sort_by_column(c)
            )
            self.results_tree.column(col_id, width=width, minwidth=width-10)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, 
                                command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=y_scroll.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind events
        self.results_tree.bind('<Double-1>', self._on_stock_double_click)
        self.results_tree.bind('<Button-1>', self._on_click)
    
    def _setup_tags(self):
        """Setup tags for row styling."""
        self.results_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.results_tree.tag_configure('loss', foreground=COLORS['loss'])
        self.results_tree.tag_configure('buy', foreground=COLORS['gain'])
        self.results_tree.tag_configure('sell', foreground=COLORS['loss'])
        self.results_tree.tag_configure('oddrow', background=COLORS.get('bg_medium', '#1a1a2e'))
        self.results_tree.tag_configure('evenrow', background=COLORS.get('bg_dark', '#0f0f23'))
    
    # ==================== PRESET SCREENS ====================
    
    def _apply_top_gainers(self):
        """Screen for top gainers."""
        self.active_filters = {'preset': '🚀 Top Gainers'}
        self.engine.clear_filters()
        self.engine.sort_by('s.change_percent', ascending=False)
        self.engine.limit(30)
        self._run_screen()
        self._update_chips_display()
    
    def _apply_top_losers(self):
        """Screen for top losers."""
        self.active_filters = {'preset': '📉 Top Losers'}
        self.engine.clear_filters()
        self.engine.sort_by('s.change_percent', ascending=True)
        self.engine.limit(30)
        self._run_screen()
        self._update_chips_display()
    
    def _apply_most_active(self):
        """Screen for most active by volume."""
        self.active_filters = {'preset': '🔥 Most Active'}
        self.engine.clear_filters()
        self.engine.add_filter(VolumeFilter(min_volume=100000))
        self.engine.sort_by('s.volume', ascending=False)
        self.engine.limit(30)
        self._run_screen()
        self._update_chips_display()
    
    def _apply_value_stocks(self):
        """Screen for value stocks (low P/E)."""
        self.active_filters = {'preset': '💎 Value (P/E<15)'}
        self.engine.clear_filters()
        self.engine.add_filter(PEFilter(max_pe=15))
        self.engine.sort_by('f.pe_ratio', ascending=True)
        self._run_screen()
        self._update_chips_display()
    
    def _apply_dividend_stocks(self):
        """Screen for high dividend stocks."""
        self.active_filters = {'preset': '💰 Dividend ≥3%'}
        self.engine.clear_filters()
        self.engine.add_filter(DividendFilter(min_yield=3))
        self.engine.sort_by('f.dividend_yield', ascending=False)
        self._run_screen()
        self._update_chips_display()
    
    def _apply_ml_signals(self):
        """Screen for ML buy signals."""
        self.active_filters = {'preset': '🎯 ML Buy Signals'}
        self.engine.clear_filters()
        self._run_screen()
        # Filter for ML signals - post-process
        filtered = [r for r in self.current_results if self._get_ml_signal(r.get('symbol')) == 'BUY']
        self.current_results = filtered
        self._populate_results(filtered)
        self._update_chips_display()
    
    def _apply_momentum(self):
        """Screen for momentum stocks."""
        self.active_filters = {'preset': '📊 Momentum (+2%+)'}
        self.engine.clear_filters()
        self.engine.sort_by('s.change_percent', ascending=False)
        self._run_screen()
        # Filter for positive momentum
        filtered = [r for r in self.current_results if (r.get('change_percent') or 0) >= 2]
        self.current_results = filtered
        self._populate_results(filtered)
        self._update_chips_display()
    
    def _get_ml_signal(self, symbol: str) -> str:
        """Get ML signal for a symbol."""
        if self.ml_engine:
            try:
                pred = self.ml_engine.predict(symbol)
                if pred and pred.get('success'):
                    return pred.get('direction', 'HOLD')
            except:
                pass
        return 'HOLD'
    
    # ==================== CUSTOM FILTERS ====================
    
    def _apply_custom_filters(self):
        """Apply all custom filters."""
        self.engine.clear_filters()
        self.active_filters = {}
        
        # P/E Filter
        try:
            pe_min = float(self.pe_min_var.get()) if self.pe_min_var.get() else None
            pe_max = float(self.pe_max_var.get()) if self.pe_max_var.get() else None
            if pe_min is not None or pe_max is not None:
                self.engine.add_filter(PEFilter(min_pe=pe_min, max_pe=pe_max))
                pe_text = f"P/E: {pe_min or 0}-{pe_max or '∞'}"
                self.active_filters['pe'] = pe_text
        except ValueError:
            pass
        
        # Dividend Filter
        try:
            div_min = float(self.div_min_var.get()) if self.div_min_var.get() else None
            if div_min is not None:
                self.engine.add_filter(DividendFilter(min_yield=div_min))
                self.active_filters['div'] = f"Div ≥{div_min}%"
        except ValueError:
            pass
        
        # Sector Filter
        sector = self.sector_var.get()
        if sector and sector != "All":
            self.engine.add_filter(SectorFilter(sector=sector))
            self.active_filters['sector'] = f"Sector: {sector}"
        
        # Market Cap Filter
        cap = self.cap_var.get()
        if cap == "Small (<10B)":
            self.engine.add_filter(MarketCapFilter.small_cap())
            self.active_filters['cap'] = "Small Cap"
        elif cap == "Mid (10-100B)":
            self.engine.add_filter(MarketCapFilter.mid_cap())
            self.active_filters['cap'] = "Mid Cap"
        elif cap == "Large (>100B)":
            self.engine.add_filter(MarketCapFilter.large_cap())
            self.active_filters['cap'] = "Large Cap"
        
        # Volume Filter
        try:
            vol_min = int(self.vol_min_var.get()) if self.vol_min_var.get() else None
            if vol_min:
                self.engine.add_filter(VolumeFilter(min_volume=vol_min))
                self.active_filters['vol'] = f"Vol ≥{vol_min:,}"
        except ValueError:
            pass
        
        self._run_screen()
        self._update_chips_display()
    
    def _apply_active_filters(self):
        """Re-apply current active filters."""
        self._run_screen()
    
    def _clear_all_filters(self):
        """Clear all filters and inputs."""
        # Clear input fields
        for var in [self.pe_min_var, self.pe_max_var, self.pb_min_var, self.pb_max_var,
                    self.div_min_var, self.roe_min_var, self.chg_min_var, self.chg_max_var,
                    self.vol_min_var]:
            var.set('')
        
        self.sector_var.set('All')
        self.cap_var.set('All')
        self.ml_signal_var.set('All')
        
        # Clear active filters
        self.active_filters = {}
        self.engine.clear_filters()
        
        self._run_screen()
        self._update_chips_display()
    
    # ==================== RESULTS ====================
    
    def _run_screen(self):
        """Run the screening and populate results."""
        try:
            results = self.engine.run()
            self.current_results = results
            self._populate_results(results)
            self._update_hero_stats(results)
        except Exception as e:
            logger.error(f"Screening failed: {e}")
    
    def _populate_results(self, results: List[Dict[str, Any]]):
        """Populate the results table."""
        # Clear existing
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Sort
        sort_key_map = {
            'symbol': lambda x: (x.get('symbol') or '').lower(),
            'name': lambda x: (x.get('name') or '').lower(),
            'price': lambda x: x.get('last_price') or 0,
            'change': lambda x: x.get('change_percent') or 0,
            'pe': lambda x: x.get('pe_ratio') or 999,
            'dividend': lambda x: x.get('dividend_yield') or 0,
            'volume': lambda x: x.get('volume') or 0,
            'signal': lambda x: self._get_ml_signal(x.get('symbol', '')),
        }
        
        key_func = sort_key_map.get(self.sort_column, lambda x: x.get('symbol', ''))
        sorted_results = sorted(results, key=key_func, reverse=not self.sort_ascending)
        
        # Populate
        for i, stock in enumerate(sorted_results):
            price = stock.get('last_price', 0) or 0
            change = stock.get('change_percent', 0) or 0
            pe = stock.get('pe_ratio')
            div = stock.get('dividend_yield')
            volume = stock.get('volume', 0) or 0
            symbol = stock.get('symbol', '')
            
            # Get ML signal
            signal = self._get_ml_signal(symbol)
            
            # Format change
            if change > 0:
                change_text = f"▲ +{change:.2f}%"
                tag = 'gain'
            elif change < 0:
                change_text = f"▼ {change:.2f}%"
                tag = 'loss'
            else:
                change_text = f"  {change:.2f}%"
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            
            # Signal color
            signal_tag = 'buy' if signal == 'BUY' else ('sell' if signal == 'SELL' else '')
            
            self.results_tree.insert('', tk.END, values=(
                symbol,
                stock.get('name', '')[:25],
                f"₦{price:,.2f}",
                change_text,
                f"{pe:.1f}" if pe else "N/A",
                f"{div:.1f}%" if div else "N/A",
                f"{volume:,}",
                signal,
                "📊 Analyze"
            ), tags=(tag,))
        
        # Update count
        self.results_count_label.config(text=f"📋 Results: {len(results)} stocks")
    
    def _update_hero_stats(self, results: List[Dict]):
        """Update hero section statistics."""
        total = len(results)
        gainers = sum(1 for r in results if (r.get('change_percent') or 0) > 0)
        losers = sum(1 for r in results if (r.get('change_percent') or 0) < 0)
        
        # Count buy signals
        buy_signals = 0
        for r in results:
            if self._get_ml_signal(r.get('symbol', '')) == 'BUY':
                buy_signals += 1
        
        self.hero_cards['total'].value_label.config(text=str(total))
        self.hero_cards['gainers'].value_label.config(text=str(gainers), foreground=COLORS['gain'])
        self.hero_cards['losers'].value_label.config(text=str(losers), foreground=COLORS['loss'])
        self.hero_cards['signals'].value_label.config(text=str(buy_signals), foreground=COLORS['primary'])
    
    def _sort_by_column(self, column: str):
        """Sort by column."""
        if self.sort_column == column:
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_column = column
            self.sort_ascending = True
        
        self._populate_results(self.current_results)
    
    def _on_stock_double_click(self, event):
        """Handle double-click on stock."""
        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            symbol = item['values'][0]
            logger.info(f"Selected stock: {symbol}")
            # TODO: Open stock detail modal
    
    def _on_click(self, event):
        """Handle click - check if on action column."""
        region = self.results_tree.identify_region(event.x, event.y)
        if region == 'cell':
            column = self.results_tree.identify_column(event.x)
            if column == '#9':  # Action column
                item = self.results_tree.identify_row(event.y)
                if item:
                    values = self.results_tree.item(item, 'values')
                    symbol = values[0]
                    self._analyze_stock(symbol)
    
    def _analyze_stock(self, symbol: str):
        """Open stock analysis."""
        logger.info(f"Analyzing: {symbol}")
        # TODO: Open analysis tab/modal
        messagebox.showinfo("Analysis", f"Opening analysis for {symbol}...")
    
    def _export_results(self):
        """Export results to CSV."""
        if not self.current_results:
            messagebox.showwarning("Export", "No results to export")
            return
        
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfilename="screener_results.csv"
        )
        
        if filename:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['symbol', 'name', 'last_price', 
                                                       'change_percent', 'pe_ratio', 
                                                       'dividend_yield', 'volume', 'sector'])
                writer.writeheader()
                writer.writerows(self.current_results)
            messagebox.showinfo("Export", f"Exported {len(self.current_results)} stocks to {filename}")
    
    def refresh(self):
        """Refresh the screener."""
        sectors = ["All"] + self.db.get_sectors()
        self.sector_combo['values'] = sectors
        self._run_screen()
