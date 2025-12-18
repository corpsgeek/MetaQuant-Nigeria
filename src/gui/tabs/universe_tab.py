"""
Universe Tab for MetaQuant Nigeria.
Displays the complete security universe in a worksheet view.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
import logging

try:
    import ttkbootstrap as ttk_bs
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.gui.theme import COLORS, get_font, format_currency, format_percent


logger = logging.getLogger(__name__)


class UniverseTab:
    """Security Universe worksheet view."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        self.refresh()
    
    def _setup_ui(self):
        """Setup UI."""
        # Header
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            header, 
            text="📋 Security Universe", 
            font=get_font('subheading')
        ).pack(side=tk.LEFT)
        
        self.count_label = ttk.Label(
            header,
            text="0 securities",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.count_label.pack(side=tk.RIGHT)
        
        # Sector filter
        filter_frame = ttk.Frame(self.frame)
        filter_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(filter_frame, text="Sector:").pack(side=tk.LEFT)
        
        self.sector_var = tk.StringVar(value="All Sectors")
        self.sector_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.sector_var,
            state="readonly",
            width=25
        )
        self.sector_combo.pack(side=tk.LEFT, padx=10)
        self.sector_combo.bind('<<ComboboxSelected>>', lambda e: self._filter_by_sector())
        
        # Search
        ttk.Label(filter_frame, text="Search:").pack(side=tk.LEFT, padx=(20, 0))
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=20)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', lambda e: self._search())
        
        # Export button
        if TTKBOOTSTRAP_AVAILABLE:
            export_btn = ttk_bs.Button(
                filter_frame,
                text="📥 Export CSV",
                bootstyle="secondary-outline",
                command=self._export_csv
            )
        else:
            export_btn = ttk.Button(filter_frame, text="📥 Export CSV", command=self._export_csv)
        export_btn.pack(side=tk.RIGHT)
        
        # Universe table
        table_frame = ttk.Frame(self.frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ('symbol', 'name', 'sector', 'subsector', 'price', 'change', 'volume')
        
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        # Headings
        self.tree.heading('symbol', text='Symbol', command=lambda: self._sort('symbol'))
        self.tree.heading('name', text='Company Name', command=lambda: self._sort('name'))
        self.tree.heading('sector', text='Sector', command=lambda: self._sort('sector'))
        self.tree.heading('subsector', text='Subsector', command=lambda: self._sort('subsector'))
        self.tree.heading('price', text='Price (₦)', command=lambda: self._sort('last_price'))
        self.tree.heading('change', text='Change %', command=lambda: self._sort('change_percent'))
        self.tree.heading('volume', text='Volume', command=lambda: self._sort('volume'))
        
        # Column widths
        self.tree.column('symbol', width=100, minwidth=80)
        self.tree.column('name', width=280, minwidth=200)
        self.tree.column('sector', width=150, minwidth=120)
        self.tree.column('subsector', width=120, minwidth=100)
        self.tree.column('price', width=100, minwidth=80)
        self.tree.column('change', width=80, minwidth=60)
        self.tree.column('volume', width=100, minwidth=80)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Grid
        self.tree.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Tags for coloring
        self.tree.tag_configure('gain', foreground=COLORS['gain'])
        self.tree.tag_configure('loss', foreground=COLORS['loss'])
        
        # Store data
        self._all_stocks = []
        self._sort_column = 'symbol'
        self._sort_reverse = False
    
    def _load_sectors(self):
        """Load sector list."""
        sectors = self.db.get_sectors()
        self.sector_combo['values'] = ["All Sectors"] + sectors
    
    def _load_stocks(self):
        """Load all stocks."""
        self._all_stocks = self.db.get_all_stocks()
        self._display_stocks(self._all_stocks)
    
    def _display_stocks(self, stocks):
        """Display stocks in the tree."""
        # Clear
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Populate
        for stock in stocks:
            change = stock.get('change_percent') or 0
            tag = 'gain' if change >= 0 else 'loss'
            
            self.tree.insert('', tk.END, values=(
                stock.get('symbol', ''),
                stock.get('name', ''),
                stock.get('sector', ''),
                stock.get('subsector', '') or '-',
                format_currency(stock.get('last_price') or 0),
                format_percent(change),
                f"{stock.get('volume', 0):,}"
            ), tags=(tag,))
        
        self.count_label.config(text=f"{len(stocks)} securities")
    
    def _filter_by_sector(self):
        """Filter by selected sector."""
        sector = self.sector_var.get()
        
        if sector == "All Sectors":
            filtered = self._all_stocks
        else:
            filtered = [s for s in self._all_stocks if s.get('sector') == sector]
        
        self._display_stocks(filtered)
    
    def _search(self):
        """Search stocks."""
        query = self.search_var.get().upper()
        
        if not query:
            self._filter_by_sector()
            return
        
        filtered = [
            s for s in self._all_stocks
            if query in s.get('symbol', '').upper() or query in s.get('name', '').upper()
        ]
        self._display_stocks(filtered)
    
    def _sort(self, column: str):
        """Sort by column."""
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False
        
        def get_value(stock):
            val = stock.get(column)
            if val is None:
                return '' if isinstance(val, str) else 0
            return val
        
        self._all_stocks.sort(key=get_value, reverse=self._sort_reverse)
        self._display_stocks(self._all_stocks)
    
    def _export_csv(self):
        """Export to CSV."""
        from tkinter import filedialog
        import csv
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfilename="ngx_universe.csv"
        )
        
        if filepath:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Symbol', 'Name', 'Sector', 'Subsector', 'Price', 'Change %', 'Volume'])
                
                for stock in self._all_stocks:
                    writer.writerow([
                        stock.get('symbol', ''),
                        stock.get('name', ''),
                        stock.get('sector', ''),
                        stock.get('subsector', ''),
                        stock.get('last_price', ''),
                        stock.get('change_percent', ''),
                        stock.get('volume', '')
                    ])
            
            logger.info(f"Exported to {filepath}")
    
    def refresh(self):
        """Refresh data."""
        self._load_sectors()
        self._load_stocks()
