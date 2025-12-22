"""
Sector Detail Dialog for MetaQuant Nigeria.
Shows all stocks in a sector with their performance.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, List
import logging

try:
    import ttkbootstrap as ttk_bs
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.gui.theme import COLORS, get_font

logger = logging.getLogger(__name__)


class SectorDetailDialog:
    """
    Popup dialog showing all stocks in a sector.
    """
    
    def __init__(self, parent, sector: str, stocks: List[Dict[str, Any]], on_stock_click=None):
        """
        Initialize the sector detail dialog.
        
        Args:
            parent: Parent window
            sector: Sector name
            stocks: List of stocks in this sector
            on_stock_click: Callback when a stock is clicked
        """
        self.parent = parent
        self.sector = sector
        self.stocks = stocks
        self.on_stock_click = on_stock_click
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"📊 {sector}")
        self.dialog.geometry("700x500")
        self.dialog.configure(bg=COLORS['bg_dark'])
        
        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Setup UI
        self._setup_ui()
    
    def _center_window(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_w = self.parent.winfo_width()
        parent_h = self.parent.winfo_height()
        
        dialog_w = 700
        dialog_h = 500
        
        x = parent_x + (parent_w - dialog_w) // 2
        y = parent_y + (parent_h - dialog_h) // 2
        
        self.dialog.geometry(f"{dialog_w}x{dialog_h}+{x}+{y}")
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame,
            text=f"📊 {self.sector}",
            font=get_font('heading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        # Summary
        gainers = sum(1 for s in self.stocks if (s.get('change', 0) or 0) > 0)
        losers = sum(1 for s in self.stocks if (s.get('change', 0) or 0) < 0)
        avg_change = sum(s.get('change', 0) or 0 for s in self.stocks) / len(self.stocks) if self.stocks else 0
        
        summary_text = f"{len(self.stocks)} stocks  |  📈 {gainers} gainers  |  📉 {losers} losers  |  Avg: {avg_change:+.2f}%"
        
        ttk.Label(
            header_frame,
            text=summary_text,
            font=get_font('body'),
            foreground=COLORS['text_secondary']
        ).pack(side=tk.LEFT, padx=20)
        
        # Close button
        close_btn = ttk.Button(
            header_frame,
            text="✕",
            width=3,
            command=self.dialog.destroy
        )
        close_btn.pack(side=tk.RIGHT)
        
        # Stock table
        self._create_stock_table(main_frame)
    
    def _create_stock_table(self, parent):
        """Create stock table with all sector stocks."""
        # Create Treeview
        columns = ('symbol', 'name', 'price', 'change', 'volume', 'perf_week', 'perf_month')
        
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        self.tree.heading('symbol', text='Symbol')
        self.tree.heading('name', text='Name')
        self.tree.heading('price', text='Price')
        self.tree.heading('change', text='Change')
        self.tree.heading('volume', text='Volume')
        self.tree.heading('perf_week', text='1W Perf')
        self.tree.heading('perf_month', text='1M Perf')
        
        # Column widths
        self.tree.column('symbol', width=80, anchor='w')
        self.tree.column('name', width=150, anchor='w')
        self.tree.column('price', width=100, anchor='e')
        self.tree.column('change', width=80, anchor='e')
        self.tree.column('volume', width=90, anchor='e')
        self.tree.column('perf_week', width=80, anchor='e')
        self.tree.column('perf_month', width=80, anchor='e')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Row tags for coloring
        self.tree.tag_configure('gain', foreground=COLORS['gain'])
        self.tree.tag_configure('loss', foreground=COLORS['loss'])
        self.tree.tag_configure('evenrow', background=COLORS['bg_medium'])
        
        # Bind double-click
        self.tree.bind('<Double-1>', self._on_stock_double_click)
        
        # Populate table
        self._populate_table()
    
    def _populate_table(self):
        """Populate the table with stocks."""
        # Sort by change descending
        sorted_stocks = sorted(self.stocks, key=lambda x: x.get('change', 0) or 0, reverse=True)
        
        for i, stock in enumerate(sorted_stocks):
            symbol = stock.get('symbol', '')
            name = stock.get('name', symbol)[:20]  # Truncate name
            close = stock.get('close', 0) or 0
            change = stock.get('change', 0) or 0
            volume = stock.get('volume', 0) or 0
            perf_week = stock.get('Perf.W', 0) or 0
            perf_month = stock.get('Perf.1M', 0) or 0
            
            # Format volume
            if volume >= 1_000_000:
                vol_str = f"{volume/1_000_000:.1f}M"
            elif volume >= 1_000:
                vol_str = f"{volume/1_000:.1f}K"
            else:
                vol_str = str(int(volume))
            
            # Format changes
            def format_change(val):
                if val > 0:
                    return f"+{val:.1f}%"
                elif val < 0:
                    return f"{val:.1f}%"
                return "0.0%"
            
            # Determine tag
            if change > 0:
                tag = 'gain'
            elif change < 0:
                tag = 'loss'
            else:
                tag = ''
            
            tags = (tag, 'evenrow') if i % 2 == 0 else (tag,)
            
            self.tree.insert('', 'end', values=(
                symbol,
                name,
                f"₦{close:,.2f}",
                format_change(change),
                vol_str,
                format_change(perf_week),
                format_change(perf_month)
            ), tags=tags)
    
    def _on_stock_double_click(self, event):
        """Handle stock double-click."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        symbol = item['values'][0] if item['values'] else None
        
        if symbol and self.on_stock_click:
            # Find the stock data
            stock_data = next((s for s in self.stocks if s.get('symbol') == symbol), {})
            self.on_stock_click(stock_data)


def show_sector_detail(parent, sector: str, stocks: List[Dict[str, Any]], on_stock_click=None):
    """
    Show sector detail dialog.
    
    Args:
        parent: Parent window
        sector: Sector name
        stocks: List of stocks in this sector
        on_stock_click: Callback when a stock is clicked
    """
    dialog = SectorDetailDialog(parent, sector, stocks, on_stock_click)
    parent.wait_window(dialog.dialog)
