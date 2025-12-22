"""
Stock Screener Tab for MetaQuant Nigeria.
Provides a UI for filtering and screening stocks.
"""

import tkinter as tk
from tkinter import ttk
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
    """Stock screener tab with filters and results display."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.engine = ScreeningEngine(db)
        
        # Sorting state
        self.sort_column = 'symbol'
        self.sort_ascending = True
        self.current_results = []
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        self._setup_tags()
        
        # Load initial data
        self.refresh()
    
    def _setup_ui(self):
        """Setup the screener UI."""
        # Main container with panels
        self.paned = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Filters
        self._create_filter_panel()
        
        # Right panel - Results
        self._create_results_panel()
    
    def _create_filter_panel(self):
        """Create the filter panel on the left."""
        filter_frame = ttk.Frame(self.paned, width=300)
        self.paned.add(filter_frame, weight=0)
        
        # Filter header
        header = ttk.Label(
            filter_frame,
            text="Filters",
            font=get_font('subheading')
        )
        header.pack(fill=tk.X, pady=(0, 15))
        
        # Quick filters (presets)
        preset_frame = ttk.LabelFrame(filter_frame, text="Quick Filters", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 15))
        
        presets = [
            ("Top Gainers", self._apply_top_gainers),
            ("Top Losers", self._apply_top_losers),
            ("Most Active", self._apply_most_active),
            ("Value Stocks", self._apply_value_stocks),
            ("Dividend Payers", self._apply_dividend_stocks),
        ]
        
        for name, command in presets:
            btn = ttk.Button(preset_frame, text=name, command=command)
            btn.pack(fill=tk.X, pady=2)
        
        # Custom filters
        custom_frame = ttk.LabelFrame(filter_frame, text="Custom Filters", padding=10)
        custom_frame.pack(fill=tk.X, pady=(0, 15))
        
        # P/E Filter
        pe_frame = ttk.Frame(custom_frame)
        pe_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pe_frame, text="P/E Ratio:").pack(side=tk.LEFT)
        
        self.pe_min_var = tk.StringVar()
        self.pe_max_var = tk.StringVar()
        
        pe_inputs = ttk.Frame(pe_frame)
        pe_inputs.pack(side=tk.RIGHT)
        ttk.Entry(pe_inputs, textvariable=self.pe_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(pe_inputs, text="-").pack(side=tk.LEFT)
        ttk.Entry(pe_inputs, textvariable=self.pe_max_var, width=8).pack(side=tk.LEFT, padx=2)
        
        # Dividend Yield Filter
        div_frame = ttk.Frame(custom_frame)
        div_frame.pack(fill=tk.X, pady=5)
        ttk.Label(div_frame, text="Dividend Yield %:").pack(side=tk.LEFT)
        
        self.div_min_var = tk.StringVar()
        div_entry = ttk.Entry(div_frame, textvariable=self.div_min_var, width=8)
        div_entry.pack(side=tk.RIGHT)
        
        # Sector Filter
        sector_frame = ttk.Frame(custom_frame)
        sector_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sector_frame, text="Sector:").pack(side=tk.LEFT)
        
        self.sector_var = tk.StringVar(value="All")
        sectors = ["All"] + self.db.get_sectors()
        self.sector_combo = ttk.Combobox(
            sector_frame, 
            textvariable=self.sector_var,
            values=sectors,
            state="readonly",
            width=15
        )
        self.sector_combo.pack(side=tk.RIGHT)
        
        # Market Cap Filter
        cap_frame = ttk.Frame(custom_frame)
        cap_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cap_frame, text="Market Cap:").pack(side=tk.LEFT)
        
        self.cap_var = tk.StringVar(value="All")
        cap_options = ["All", "Small Cap (<10B)", "Mid Cap (10-100B)", "Large Cap (>100B)"]
        cap_combo = ttk.Combobox(
            cap_frame,
            textvariable=self.cap_var,
            values=cap_options,
            state="readonly",
            width=15
        )
        cap_combo.pack(side=tk.RIGHT)
        
        # Apply/Clear buttons
        btn_frame = ttk.Frame(custom_frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        if TTKBOOTSTRAP_AVAILABLE:
            apply_btn = ttk_bs.Button(
                btn_frame, 
                text="Apply Filters",
                bootstyle="success",
                command=self._apply_custom_filters
            )
            clear_btn = ttk_bs.Button(
                btn_frame,
                text="Clear All",
                bootstyle="secondary",
                command=self._clear_filters
            )
        else:
            apply_btn = ttk.Button(btn_frame, text="Apply Filters", command=self._apply_custom_filters)
            clear_btn = ttk.Button(btn_frame, text="Clear All", command=self._clear_filters)
        
        apply_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Active filters display
        self.active_filters_label = ttk.Label(
            filter_frame,
            text="No filters active",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.active_filters_label.pack(fill=tk.X, pady=10)
    
    def _create_results_panel(self):
        """Create the results panel on the right."""
        results_frame = ttk.Frame(self.paned)
        self.paned.add(results_frame, weight=1)
        
        # Results header with count
        header_frame = ttk.Frame(results_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame,
            text="Screening Results",
            font=get_font('subheading')
        ).pack(side=tk.LEFT)
        
        self.results_count_label = ttk.Label(
            header_frame,
            text="0 stocks",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.results_count_label.pack(side=tk.RIGHT)
        
        # Results table
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Columns
        columns = ('symbol', 'name', 'price', 'change', 'pe', 'dividend', 'volume', 'sector')
        
        self.results_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        # Column headings with sort functionality
        columns_config = [
            ('symbol', 'Symbol', 80),
            ('name', 'Company', 200),
            ('price', 'Price ₦', 100),
            ('change', 'Change %', 80),
            ('pe', 'P/E', 70),
            ('dividend', 'Div %', 70),
            ('volume', 'Volume', 100),
            ('sector', 'Sector', 120)
        ]
        
        for col_id, col_text, width in columns_config:
            self.results_tree.heading(
                col_id, 
                text=col_text,
                command=lambda c=col_id: self._sort_by_column(c)
            )
            self.results_tree.column(col_id, width=width, minwidth=width-20)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Grid layout for table
        self.results_tree.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Bind double-click
        self.results_tree.bind('<Double-1>', self._on_stock_double_click)
    
    def _setup_tags(self):
        """Setup tags for row coloring."""
        # Alternating row colors
        self.results_tree.tag_configure('oddrow', background=COLORS['bg_medium'])
        self.results_tree.tag_configure('evenrow', background=COLORS['bg_dark'])
        
        # Change colors - using foreground won't work on treeview cells
        # So we'll format the text with indicators instead
    
    def _sort_by_column(self, column: str):
        """Sort the table by clicking column header."""
        # Toggle sort direction if same column
        if self.sort_column == column:
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_column = column
            self.sort_ascending = True
        
        # Update column headers with sort indicator
        for col in ('symbol', 'name', 'price', 'change', 'pe', 'dividend', 'volume', 'sector'):
            text = self.results_tree.heading(col, 'text')
            # Remove existing indicators
            text = text.replace(' ▲', '').replace(' ▼', '')
            if col == column:
                indicator = ' ▲' if self.sort_ascending else ' ▼'
                text = text + indicator
            self.results_tree.heading(col, text=text)
        
        # Re-populate with sorted data
        self._populate_results(self.current_results)
    
    def _run_screen(self):
        """Run the screening and populate results."""
        try:
            results = self.engine.run()
            self.current_results = results
            self._populate_results(results)
                
        except Exception as e:
            logger.error(f"Screening failed: {e}")
    
    def _populate_results(self, results: List[Dict[str, Any]]):
        """Populate the results table with sorting."""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Sort results
        sort_key_map = {
            'symbol': lambda x: (x.get('symbol') or '').lower(),
            'name': lambda x: (x.get('name') or '').lower(),
            'price': lambda x: x.get('last_price') or 0,
            'change': lambda x: x.get('change_percent') or 0,
            'pe': lambda x: x.get('pe_ratio') or 999,
            'dividend': lambda x: x.get('dividend_yield') or 0,
            'volume': lambda x: x.get('volume') or 0,
            'sector': lambda x: (x.get('sector') or '').lower(),
        }
        
        key_func = sort_key_map.get(self.sort_column, lambda x: x.get('symbol', ''))
        sorted_results = sorted(results, key=key_func, reverse=not self.sort_ascending)
        
        # Populate results with alternating colors
        for i, stock in enumerate(sorted_results):
            price = stock.get('last_price', 0) or 0
            change = stock.get('change_percent', 0) or 0
            pe = stock.get('pe_ratio')
            div = stock.get('dividend_yield')
            volume = stock.get('volume', 0) or 0
            
            # Format change with color indicator
            if change > 0:
                change_text = f"▲ +{change:.2f}%"
            elif change < 0:
                change_text = f"▼ {change:.2f}%"
            else:
                change_text = f"  {change:.2f}%"
            
            # Tag for alternating row colors
            tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            
            self.results_tree.insert('', tk.END, values=(
                stock.get('symbol', ''),
                stock.get('name', '')[:30],
                f"₦{price:,.2f}",
                change_text,
                f"{pe:.1f}" if pe else "N/A",
                f"{div:.1f}%" if div else "N/A",
                f"{volume:,}",
                stock.get('sector', '')[:15]
            ), tags=(tag,))
        
        # Update count
        self.results_count_label.config(text=f"{len(results)} stocks")
        
        # Update active filters
        active = self.engine.get_active_filters()
        if active:
            self.active_filters_label.config(
                text=f"Active: {', '.join(active)}",
                foreground=COLORS['primary']
            )
        else:
            self.active_filters_label.config(
                text="No filters active",
                foreground=COLORS['text_muted']
            )
    
    # ==================== Preset Filters ====================
    
    def _apply_top_gainers(self):
        self.engine.clear_filters()
        self.engine.add_filter(PEFilter())  # No filter
        self.engine.sort_by('s.change_percent', ascending=False)
        self.engine.limit(20)
        self._run_screen()
    
    def _apply_top_losers(self):
        self.engine.clear_filters()
        self.engine.sort_by('s.change_percent', ascending=True)
        self.engine.limit(20)
        self._run_screen()
    
    def _apply_most_active(self):
        self.engine.clear_filters()
        self.engine.add_filter(VolumeFilter(min_volume=100000))
        self.engine.sort_by('s.volume', ascending=False)
        self.engine.limit(20)
        self._run_screen()
    
    def _apply_value_stocks(self):
        self.engine.clear_filters()
        self.engine.add_filter(PEFilter(max_pe=15))
        self.engine.sort_by('f.pe_ratio', ascending=True)
        self._run_screen()
    
    def _apply_dividend_stocks(self):
        self.engine.clear_filters()
        self.engine.add_filter(DividendFilter(min_yield=3))
        self.engine.sort_by('f.dividend_yield', ascending=False)
        self._run_screen()
    
    # ==================== Custom Filters ====================
    
    def _apply_custom_filters(self):
        """Apply custom filter values."""
        self.engine.clear_filters()
        
        # P/E Filter
        try:
            pe_min = float(self.pe_min_var.get()) if self.pe_min_var.get() else None
            pe_max = float(self.pe_max_var.get()) if self.pe_max_var.get() else None
            if pe_min is not None or pe_max is not None:
                self.engine.add_filter(PEFilter(min_pe=pe_min, max_pe=pe_max))
        except ValueError:
            pass
        
        # Dividend Filter
        try:
            div_min = float(self.div_min_var.get()) if self.div_min_var.get() else None
            if div_min is not None:
                self.engine.add_filter(DividendFilter(min_yield=div_min))
        except ValueError:
            pass
        
        # Sector Filter
        sector = self.sector_var.get()
        if sector and sector != "All":
            self.engine.add_filter(SectorFilter(sector=sector))
        
        # Market Cap Filter
        cap = self.cap_var.get()
        if cap == "Small Cap (<10B)":
            self.engine.add_filter(MarketCapFilter.small_cap())
        elif cap == "Mid Cap (10-100B)":
            self.engine.add_filter(MarketCapFilter.mid_cap())
        elif cap == "Large Cap (>100B)":
            self.engine.add_filter(MarketCapFilter.large_cap())
        
        self._run_screen()
    
    def _clear_filters(self):
        """Clear all filter inputs and results."""
        self.pe_min_var.set('')
        self.pe_max_var.set('')
        self.div_min_var.set('')
        self.sector_var.set('All')
        self.cap_var.set('All')
        
        self.engine.clear_filters()
        self._run_screen()
    
    def _on_stock_double_click(self, event):
        """Handle double-click on a stock."""
        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            symbol = item['values'][0]
            # Could open a detail dialog here
            logger.info(f"Selected stock: {symbol}")
    
    def refresh(self):
        """Refresh the screener data."""
        # Refresh sector list
        sectors = ["All"] + self.db.get_sectors()
        self.sector_combo['values'] = sectors
        
        # Run current screen
        self._run_screen()
