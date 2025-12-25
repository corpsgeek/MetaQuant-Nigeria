"""
Watchlist Tab for MetaQuant Nigeria.
Manage stock watchlists with price alerts and targets.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Optional, List, Dict, Any
import logging

try:
    import ttkbootstrap as ttk_bs
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.gui.theme import COLORS, get_font

logger = logging.getLogger(__name__)


class WatchlistTab:
    """Watchlist tab for managing stock watchlists."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager, price_provider=None):
        self.parent = parent
        self.db = db
        self.price_provider = price_provider
        
        # State
        self.current_watchlist_id = None
        self.watchlist_items = []
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        
        # Load data
        self._load_watchlists()
        self._ensure_default_watchlist()
    
    def _setup_ui(self):
        """Setup the watchlist UI."""
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # === HEADER ===
        self._create_header(main)
        
        # === MAIN CONTENT ===
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left: Watchlist selector
        left = ttk.Frame(content, width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        self._create_watchlist_panel(left)
        
        # Right: Watchlist items
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._create_items_panel(right)
    
    def _create_header(self, parent):
        """Create header with title and add button."""
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header, text="⭐ Watchlists",
            font=get_font('heading'),
            foreground=COLORS['text_primary']
        ).pack(side=tk.LEFT)
        
        # Add stock button
        if TTKBOOTSTRAP_AVAILABLE:
            add_btn = ttk_bs.Button(header, text="➕ Add Stock", bootstyle="success",
                                   command=self._add_stock_dialog)
        else:
            add_btn = ttk.Button(header, text="➕ Add Stock", command=self._add_stock_dialog)
        add_btn.pack(side=tk.RIGHT)
        
        # Refresh button
        if TTKBOOTSTRAP_AVAILABLE:
            refresh_btn = ttk_bs.Button(header, text="🔄 Refresh", bootstyle="info-outline",
                                        command=self._refresh)
        else:
            refresh_btn = ttk.Button(header, text="🔄 Refresh", command=self._refresh)
        refresh_btn.pack(side=tk.RIGHT, padx=5)
    
    def _create_watchlist_panel(self, parent):
        """Create watchlist selection panel."""
        ttk.Label(parent, text="📋 My Watchlists", font=get_font('subheading'),
                 foreground=COLORS['text_primary']).pack(anchor='w', pady=(0, 5))
        
        # Watchlist listbox
        self.watchlist_listbox = tk.Listbox(parent, height=10, bg=COLORS['bg_dark'],
                                            fg=COLORS['text_primary'], selectbackground=COLORS['primary'])
        self.watchlist_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.watchlist_listbox.bind('<<ListboxSelect>>', self._on_watchlist_select)
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="➕ New", command=self._create_watchlist).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="✕ Delete", command=self._delete_watchlist).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    def _create_items_panel(self, parent):
        """Create watchlist items panel."""
        # Summary bar
        summary = ttk.Frame(parent)
        summary.pack(fill=tk.X, pady=(0, 10))
        
        self.summary_label = ttk.Label(summary, text="No watchlist selected",
                                       font=get_font('subheading'), foreground=COLORS['text_primary'])
        self.summary_label.pack(side=tk.LEFT)
        
        # Items table
        columns = ('symbol', 'name', 'price', 'change', 'target', 'alert', 'action')
        self.items_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        col_config = [
            ('symbol', 'Symbol', 80),
            ('name', 'Company', 180),
            ('price', 'Price ₦', 90),
            ('change', 'Change %', 85),
            ('target', 'Target ₦', 90),
            ('alert', 'Alert', 100),
            ('action', 'Action', 80)
        ]
        
        for col_id, col_text, width in col_config:
            self.items_tree.heading(col_id, text=col_text)
            self.items_tree.column(col_id, width=width, minwidth=width-10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.items_tree.yview)
        self.items_tree.configure(yscrollcommand=scrollbar.set)
        
        self.items_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags for styling
        self.items_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.items_tree.tag_configure('loss', foreground=COLORS['loss'])
        self.items_tree.tag_configure('alert_triggered', background='#2d1b1b')
        
        # Bind click for action column
        self.items_tree.bind('<Button-1>', self._on_item_click)
    
    # ==================== WATCHLIST MANAGEMENT ====================
    
    def _load_watchlists(self):
        """Load watchlists from database."""
        try:
            result = self.db.conn.execute(
                "SELECT id, name FROM watchlists ORDER BY name"
            ).fetchall()
            
            self.watchlist_listbox.delete(0, tk.END)
            for row in result:
                self.watchlist_listbox.insert(tk.END, row[1])
            
            if result:
                self.watchlist_listbox.select_set(0)
                self.current_watchlist_id = result[0][0]
                self._load_items()
        except Exception as e:
            logger.error(f"Failed to load watchlists: {e}")
    
    def _ensure_default_watchlist(self):
        """Ensure a default watchlist exists."""
        try:
            result = self.db.conn.execute("SELECT COUNT(*) FROM watchlists").fetchone()
            if result[0] == 0:
                self.db.conn.execute(
                    "INSERT INTO watchlists (id, name) VALUES (nextval('seq_watchlists'), 'My Watchlist')"
                )
                self.db.conn.commit()
                self._load_watchlists()
        except Exception as e:
            logger.error(f"Failed to create default watchlist: {e}")
    
    def _on_watchlist_select(self, event):
        """Handle watchlist selection."""
        selection = self.watchlist_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        name = self.watchlist_listbox.get(idx)
        
        try:
            result = self.db.conn.execute(
                "SELECT id FROM watchlists WHERE name = ?", [name]
            ).fetchone()
            if result:
                self.current_watchlist_id = result[0]
                self._load_items()
        except Exception as e:
            logger.error(f"Failed to select watchlist: {e}")
    
    def _create_watchlist(self):
        """Create a new watchlist."""
        name = simpledialog.askstring("New Watchlist", "Enter watchlist name:")
        if not name:
            return
        
        try:
            self.db.conn.execute(
                "INSERT INTO watchlists (id, name) VALUES (nextval('seq_watchlists'), ?)", [name]
            )
            self.db.conn.commit()
            self._load_watchlists()
            messagebox.showinfo("Success", f"Watchlist '{name}' created")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create watchlist: {e}")
    
    def _delete_watchlist(self):
        """Delete current watchlist."""
        if not self.current_watchlist_id:
            return
        
        if not messagebox.askyesno("Confirm", "Delete this watchlist?"):
            return
        
        try:
            self.db.conn.execute("DELETE FROM watchlist_items WHERE watchlist_id = ?", 
                                [self.current_watchlist_id])
            self.db.conn.execute("DELETE FROM watchlists WHERE id = ?", 
                                [self.current_watchlist_id])
            self.db.conn.commit()
            self.current_watchlist_id = None
            self._load_watchlists()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete watchlist: {e}")
    
    # ==================== ITEMS MANAGEMENT ====================
    
    def _load_items(self):
        """Load items for current watchlist."""
        if not self.current_watchlist_id:
            return
        
        # Clear
        for item in self.items_tree.get_children():
            self.items_tree.delete(item)
        
        try:
            # Only select columns that exist in the database
            result = self.db.conn.execute("""
                SELECT wi.id, s.symbol, s.name, s.last_price, s.change_percent,
                       wi.target_price
                FROM watchlist_items wi
                JOIN stocks s ON wi.stock_id = s.id
                WHERE wi.watchlist_id = ?
                ORDER BY s.symbol
            """, [self.current_watchlist_id]).fetchall()
            
            self.watchlist_items = []
            for row in result:
                item_id, symbol, name, price, change, target = row
                self.watchlist_items.append({
                    'id': item_id,
                    'symbol': symbol,
                    'price': price,
                    'target': target,
                })
                
                # Format values
                price = price or 0
                change = change or 0
                
                # Determine tag
                tag = 'gain' if change > 0 else ('loss' if change < 0 else '')
                
                # Change text
                if change > 0:
                    change_text = f"▲ +{change:.2f}%"
                elif change < 0:
                    change_text = f"▼ {change:.2f}%"
                else:
                    change_text = f"  {change:.2f}%"
                
                # Check if target hit
                alert_text = "-"
                if target and price >= target:
                    alert_text = "🎯 Target Hit!"
                    tag = 'gain'
                elif target:
                    pct_to_target = ((target - price) / price) * 100 if price > 0 else 0
                    alert_text = f"↗️ {pct_to_target:.1f}%"
                
                self.items_tree.insert('', tk.END, values=(
                    symbol,
                    name[:25] if name else '',
                    f"₦{price:,.2f}",
                    change_text,
                    f"₦{target:,.0f}" if target else "-",
                    alert_text,
                    "🗑️ Remove"
                ), tags=(tag,) if tag else ())
            
            # Update summary
            self.summary_label.config(text=f"📊 {len(result)} stocks in watchlist")
            
        except Exception as e:
            logger.error(f"Failed to load watchlist items: {e}")
    
    def _add_stock_dialog(self):
        """Open dialog to add stock."""
        if not self.current_watchlist_id:
            messagebox.showwarning("Watchlist", "Please select a watchlist first")
            return
        
        # Get available stocks
        try:
            result = self.db.conn.execute(
                "SELECT id, symbol, name FROM stocks WHERE is_active = TRUE ORDER BY symbol"
            ).fetchall()
            stocks = {f"{r[1]} - {r[2][:30]}": r[0] for r in result}
        except:
            stocks = {}
        
        if not stocks:
            messagebox.showwarning("Watchlist", "No stocks available")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.parent)
        dialog.title("Add Stock to Watchlist")
        dialog.geometry("350x150")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select Stock:").pack(pady=10)
        
        stock_var = tk.StringVar()
        stock_combo = ttk.Combobox(dialog, textvariable=stock_var, 
                                   values=list(stocks.keys()), width=40, state='readonly')
        stock_combo.pack(pady=5)
        
        ttk.Label(dialog, text="Target Price (optional):").pack(pady=5)
        target_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=target_var, width=20).pack()
        
        def add():
            selected = stock_var.get()
            if not selected:
                messagebox.showwarning("Add Stock", "Please select a stock")
                return
            
            stock_id = stocks.get(selected)
            target = float(target_var.get()) if target_var.get() else None
            
            try:
                self.db.conn.execute("""
                    INSERT INTO watchlist_items (id, watchlist_id, stock_id, target_price)
                    VALUES (nextval('seq_watchlist_items'), ?, ?, ?)
                """, [self.current_watchlist_id, stock_id, target])
                self.db.conn.commit()
                dialog.destroy()
                self._load_items()
                messagebox.showinfo("Success", "Stock added to watchlist")
            except Exception as e:
                if 'UNIQUE' in str(e):
                    messagebox.showwarning("Add Stock", "Stock already in watchlist")
                else:
                    messagebox.showerror("Error", f"Failed to add stock: {e}")
        
        ttk.Button(dialog, text="Add", command=add).pack(pady=15)
    
    def _on_item_click(self, event):
        """Handle click on items - check if remove button clicked."""
        region = self.items_tree.identify_region(event.x, event.y)
        if region == 'cell':
            column = self.items_tree.identify_column(event.x)
            if column == '#7':  # Action column
                item = self.items_tree.identify_row(event.y)
                if item:
                    values = self.items_tree.item(item, 'values')
                    symbol = values[0]
                    self._remove_stock(symbol)
    
    def _remove_stock(self, symbol: str):
        """Remove stock from watchlist."""
        if not messagebox.askyesno("Remove", f"Remove {symbol} from watchlist?"):
            return
        
        try:
            self.db.conn.execute("""
                DELETE FROM watchlist_items 
                WHERE watchlist_id = ? AND stock_id = (SELECT id FROM stocks WHERE symbol = ?)
            """, [self.current_watchlist_id, symbol])
            self.db.conn.commit()
            self._load_items()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove stock: {e}")
    
    def _refresh(self):
        """Refresh watchlist data."""
        self._load_items()
    
    def add_to_watchlist(self, symbol: str, watchlist_name: str = None):
        """Add stock to watchlist programmatically (called from other tabs)."""
        try:
            # Get stock ID
            stock = self.db.conn.execute(
                "SELECT id FROM stocks WHERE symbol = ?", [symbol]
            ).fetchone()
            if not stock:
                return False
            
            # Get or use default watchlist
            if watchlist_name:
                watchlist = self.db.conn.execute(
                    "SELECT id FROM watchlists WHERE name = ?", [watchlist_name]
                ).fetchone()
            else:
                watchlist = self.db.conn.execute(
                    "SELECT id FROM watchlists ORDER BY id LIMIT 1"
                ).fetchone()
            
            if not watchlist:
                return False
            
            # Add to watchlist
            self.db.conn.execute("""
                INSERT INTO watchlist_items (id, watchlist_id, stock_id)
                VALUES (nextval('seq_watchlist_items'), ?, ?)
                ON CONFLICT (watchlist_id, stock_id) DO NOTHING
            """, [watchlist[0], stock[0]])
            self.db.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Failed to add {symbol} to watchlist: {e}")
            return False
