"""Reusable stock table widget."""

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any

from src.gui.theme import COLORS, get_font, format_currency, format_percent


class StockTable(ttk.Frame):
    """Reusable stock table widget."""
    
    def __init__(self, parent, columns=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        if columns is None:
            columns = ('symbol', 'name', 'price', 'change')
        
        self.columns = columns
        self._setup_table()
    
    def _setup_table(self):
        """Setup the treeview table."""
        self.tree = ttk.Treeview(self, columns=self.columns, show='headings', selectmode='browse')
        
        for col in self.columns:
            self.tree.heading(col, text=col.replace('_', ' ').title())
            self.tree.column(col, width=100)
        
        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure tags
        self.tree.tag_configure('gain', foreground=COLORS['gain'])
        self.tree.tag_configure('loss', foreground=COLORS['loss'])
    
    def clear(self):
        """Clear all items."""
        for item in self.tree.get_children():
            self.tree.delete(item)
    
    def add_stock(self, values: tuple, tags: tuple = ()):
        """Add a single stock row."""
        self.tree.insert('', tk.END, values=values, tags=tags)
    
    def set_stocks(self, stocks: List[Dict[str, Any]]):
        """Set all stocks from list of dicts."""
        self.clear()
        for stock in stocks:
            change = stock.get('change_percent', 0) or 0
            tag = ('gain',) if change >= 0 else ('loss',)
            
            values = tuple(
                stock.get(col, '') for col in self.columns
            )
            self.add_stock(values, tag)
    
    def get_selected(self):
        """Get selected item values."""
        sel = self.tree.selection()
        if sel:
            return self.tree.item(sel[0])['values']
        return None
    
    def bind_double_click(self, callback):
        """Bind double-click handler."""
        self.tree.bind('<Double-1>', callback)
