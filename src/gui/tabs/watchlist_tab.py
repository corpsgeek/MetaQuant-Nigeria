"""
Watchlist Tab for MetaQuant Nigeria.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Optional, Dict, Any
import logging

try:
    import ttkbootstrap as ttk_bs
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.gui.theme import COLORS, get_font, format_currency, format_percent


logger = logging.getLogger(__name__)


class WatchlistTab:
    """Watchlist management tab."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.current_watchlist_id: Optional[int] = None
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        self.refresh()
    
    def _setup_ui(self):
        """Setup UI."""
        # Selector
        sel = ttk.Frame(self.frame)
        sel.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(sel, text="Watchlist:", font=get_font('body_bold')).pack(side=tk.LEFT)
        
        self.watchlist_var = tk.StringVar()
        self.watchlist_combo = ttk.Combobox(sel, textvariable=self.watchlist_var, state="readonly", width=25)
        self.watchlist_combo.pack(side=tk.LEFT, padx=10)
        self.watchlist_combo.bind('<<ComboboxSelected>>', self._on_changed)
        
        ttk.Button(sel, text="+ New", command=self._create_watchlist).pack(side=tk.LEFT)
        
        # Main content
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Table
        cols = ('symbol', 'name', 'price', 'change', 'target', 'sector')
        self.tree = ttk.Treeview(main, columns=cols, show='headings', selectmode='browse')
        
        for c in cols:
            self.tree.heading(c, text=c.title())
        
        self.tree.column('symbol', width=80)
        self.tree.column('name', width=180)
        self.tree.column('price', width=100)
        self.tree.column('change', width=80)
        self.tree.column('target', width=100)
        self.tree.column('sector', width=120)
        
        scroll = ttk.Scrollbar(main, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<Button-3>', self._context_menu)
        
        # Add section
        add_frame = ttk.Frame(self.frame)
        add_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.search_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.search_var, width=20).pack(side=tk.LEFT)
        ttk.Button(add_frame, text="Search", command=self._search).pack(side=tk.LEFT, padx=5)
        ttk.Button(add_frame, text="Add", command=self._add_stock).pack(side=tk.LEFT)
        
        self.results_var = tk.StringVar()
        self.results_combo = ttk.Combobox(add_frame, textvariable=self.results_var, width=40)
        self.results_combo.pack(side=tk.LEFT, padx=10)
    
    def _load_lists(self):
        """Load watchlists."""
        lists = self.db.get_watchlists()
        names = [w['name'] for w in lists]
        self.watchlist_combo['values'] = names
        
        if lists and not self.current_watchlist_id:
            self.watchlist_var.set(lists[0]['name'])
            self.current_watchlist_id = lists[0]['id']
    
    def _on_changed(self, event=None):
        """Handle selection change."""
        name = self.watchlist_var.get()
        lists = self.db.get_watchlists()
        wl = next((w for w in lists if w['name'] == name), None)
        if wl:
            self.current_watchlist_id = wl['id']
            self._load_items()
    
    def _load_items(self):
        """Load watchlist items."""
        if not self.current_watchlist_id:
            return
        
        for i in self.tree.get_children():
            self.tree.delete(i)
        
        items = self.db.get_watchlist_items(self.current_watchlist_id)
        for item in items:
            self.tree.insert('', tk.END, values=(
                item.get('symbol', ''),
                item.get('name', '')[:25],
                format_currency(item.get('last_price') or 0),
                format_percent(item.get('change_percent') or 0),
                format_currency(item.get('target_price')) if item.get('target_price') else "—",
                item.get('sector', '')[:15]
            ))
    
    def _create_watchlist(self):
        """Create new watchlist."""
        name = simpledialog.askstring("New Watchlist", "Enter name:")
        if name:
            self.db.create_watchlist(name)
            self._load_lists()
            self.watchlist_var.set(name)
            self._on_changed()
    
    def _search(self):
        """Search stocks."""
        q = self.search_var.get().strip()
        if q:
            results = self.db.search_stocks(q)
            self.results_combo['values'] = [f"{s['symbol']} - {s['name'][:30]}" for s in results[:15]]
    
    def _add_stock(self):
        """Add stock to watchlist."""
        if not self.current_watchlist_id:
            return
        
        sel = self.results_var.get()
        if not sel:
            return
        
        symbol = sel.split(' - ')[0]
        stock = self.db.get_stock(symbol)
        if stock:
            self.db.add_to_watchlist(self.current_watchlist_id, stock['id'])
            self._load_items()
    
    def _context_menu(self, event):
        """Show context menu."""
        sel = self.tree.selection()
        if not sel:
            return
        
        menu = tk.Menu(self.frame, tearoff=0)
        menu.add_command(label="Set Target", command=self._set_target)
        menu.add_command(label="Remove", command=self._remove)
        menu.tk_popup(event.x_root, event.y_root)
    
    def _set_target(self):
        """Set target price."""
        sel = self.tree.selection()
        if not sel:
            return
        
        symbol = self.tree.item(sel[0])['values'][0]
        target = simpledialog.askfloat("Target", f"Target price for {symbol}:")
        if target:
            stock = self.db.get_stock(symbol)
            if stock:
                self.db.add_to_watchlist(self.current_watchlist_id, stock['id'], target_price=target)
                self._load_items()
    
    def _remove(self):
        """Remove from watchlist."""
        sel = self.tree.selection()
        if not sel:
            return
        
        symbol = self.tree.item(sel[0])['values'][0]
        self.db.conn.execute("""
            DELETE FROM watchlist_items 
            WHERE watchlist_id = ? AND stock_id = (SELECT id FROM stocks WHERE symbol = ?)
        """, [self.current_watchlist_id, symbol])
        self._load_items()
    
    def refresh(self):
        """Refresh data."""
        self._load_lists()
        if self.current_watchlist_id:
            self._load_items()
