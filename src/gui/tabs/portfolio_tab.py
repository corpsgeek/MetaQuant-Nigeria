"""
Portfolio Tab for MetaQuant Nigeria.
Provides portfolio management, position tracking, and performance analysis.
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
from src.portfolio.portfolio_manager import PortfolioManager
from src.gui.theme import COLORS, get_font, format_currency, format_percent, get_change_color


logger = logging.getLogger(__name__)


class PortfolioTab:
    """Portfolio management tab with positions and analytics."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.portfolio_manager = PortfolioManager(db)
        self.current_portfolio_id: Optional[int] = None
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        
        # Load data
        self.refresh()
    
    def _setup_ui(self):
        """Setup the portfolio UI."""
        # Top bar - Portfolio selector
        self._create_portfolio_selector()
        
        # Main content - split view
        self.paned = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Positions
        self._create_positions_panel()
        
        # Right panel - Summary and analytics
        self._create_summary_panel()
    
    def _create_portfolio_selector(self):
        """Create the portfolio selector bar."""
        selector_frame = ttk.Frame(self.frame)
        selector_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Label(
            selector_frame,
            text="Portfolio:",
            font=get_font('body_bold')
        ).pack(side=tk.LEFT)
        
        self.portfolio_var = tk.StringVar()
        self.portfolio_combo = ttk.Combobox(
            selector_frame,
            textvariable=self.portfolio_var,
            state="readonly",
            width=30
        )
        self.portfolio_combo.pack(side=tk.LEFT, padx=(10, 20))
        self.portfolio_combo.bind('<<ComboboxSelected>>', self._on_portfolio_changed)
        
        # Portfolio management buttons
        if TTKBOOTSTRAP_AVAILABLE:
            new_btn = ttk_bs.Button(
                selector_frame,
                text="+ New Portfolio",
                bootstyle="success-outline",
                command=self._create_portfolio
            )
        else:
            new_btn = ttk.Button(
                selector_frame,
                text="+ New Portfolio",
                command=self._create_portfolio
            )
        new_btn.pack(side=tk.LEFT, padx=5)
    
    def _create_positions_panel(self):
        """Create the positions panel."""
        positions_frame = ttk.Frame(self.paned)
        self.paned.add(positions_frame, weight=2)
        
        # Header with actions
        header_frame = ttk.Frame(positions_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame,
            text="Holdings",
            font=get_font('subheading')
        ).pack(side=tk.LEFT)
        
        # Add position button
        if TTKBOOTSTRAP_AVAILABLE:
            add_btn = ttk_bs.Button(
                header_frame,
                text="+ Add Position",
                bootstyle="success",
                command=self._add_position_dialog
            )
        else:
            add_btn = ttk.Button(
                header_frame,
                text="+ Add Position",
                command=self._add_position_dialog
            )
        add_btn.pack(side=tk.RIGHT)
        
        # Positions table
        table_frame = ttk.Frame(positions_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('symbol', 'name', 'quantity', 'avg_cost', 'current', 'value', 'pnl', 'return')
        
        self.positions_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        # Column headings
        self.positions_tree.heading('symbol', text='Symbol')
        self.positions_tree.heading('name', text='Company')
        self.positions_tree.heading('quantity', text='Qty')
        self.positions_tree.heading('avg_cost', text='Avg Cost')
        self.positions_tree.heading('current', text='Current')
        self.positions_tree.heading('value', text='Value')
        self.positions_tree.heading('pnl', text='P&L')
        self.positions_tree.heading('return', text='Return %')
        
        # Column widths
        self.positions_tree.column('symbol', width=80)
        self.positions_tree.column('name', width=150)
        self.positions_tree.column('quantity', width=70)
        self.positions_tree.column('avg_cost', width=90)
        self.positions_tree.column('current', width=90)
        self.positions_tree.column('value', width=100)
        self.positions_tree.column('pnl', width=100)
        self.positions_tree.column('return', width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=scrollbar.set)
        
        self.positions_tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Context menu
        self.positions_tree.bind('<Button-3>', self._show_position_menu)
    
    def _create_summary_panel(self):
        """Create the portfolio summary panel."""
        summary_frame = ttk.Frame(self.paned, width=300)
        self.paned.add(summary_frame, weight=1)
        
        # Portfolio summary card
        card_frame = ttk.LabelFrame(summary_frame, text="Portfolio Summary", padding=15)
        card_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Summary metrics
        self.total_value_label = self._create_metric_row(card_frame, "Total Value", "₦0.00")
        self.total_cost_label = self._create_metric_row(card_frame, "Total Cost", "₦0.00")
        self.total_pnl_label = self._create_metric_row(card_frame, "Unrealized P&L", "₦0.00")
        self.return_label = self._create_metric_row(card_frame, "Return", "0.00%")
        self.positions_count_label = self._create_metric_row(card_frame, "Positions", "0")
        
        # Top performers
        top_frame = ttk.LabelFrame(summary_frame, text="Top Performers", padding=10)
        top_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.top_performers_list = tk.Listbox(
            top_frame,
            height=5,
            bg=COLORS['bg_medium'],
            fg=COLORS['text_primary'],
            selectbackground=COLORS['primary'],
            font=get_font('small')
        )
        self.top_performers_list.pack(fill=tk.X)
        
        # Worst performers
        worst_frame = ttk.LabelFrame(summary_frame, text="Worst Performers", padding=10)
        worst_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.worst_performers_list = tk.Listbox(
            worst_frame,
            height=5,
            bg=COLORS['bg_medium'],
            fg=COLORS['text_primary'],
            selectbackground=COLORS['primary'],
            font=get_font('small')
        )
        self.worst_performers_list.pack(fill=tk.X)
    
    def _create_metric_row(self, parent, label: str, value: str) -> ttk.Label:
        """Create a metric row with label and value."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        
        ttk.Label(
            row, 
            text=label,
            font=get_font('small'),
            foreground=COLORS['text_secondary']
        ).pack(side=tk.LEFT)
        
        value_label = ttk.Label(
            row,
            text=value,
            font=get_font('body_bold')
        )
        value_label.pack(side=tk.RIGHT)
        
        return value_label
    
    def _load_portfolios(self):
        """Load portfolio list."""
        portfolios = self.portfolio_manager.get_all_portfolios()
        
        # Update combobox
        names = [p.name for p in portfolios]
        self.portfolio_combo['values'] = names
        
        # Select first portfolio if any
        if portfolios and not self.current_portfolio_id:
            self.portfolio_var.set(portfolios[0].name)
            self.current_portfolio_id = portfolios[0].id
        
        return portfolios
    
    def _on_portfolio_changed(self, event=None):
        """Handle portfolio selection change."""
        name = self.portfolio_var.get()
        portfolios = self.portfolio_manager.get_all_portfolios()
        
        portfolio = next((p for p in portfolios if p.name == name), None)
        if portfolio:
            self.current_portfolio_id = portfolio.id
            self._load_positions()
            self._update_summary()
    
    def _load_positions(self):
        """Load positions for current portfolio."""
        if not self.current_portfolio_id:
            return
        
        # Clear existing
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        # Get positions
        positions = self.db.get_portfolio_positions(self.current_portfolio_id)
        
        for pos in positions:
            qty = pos.get('quantity', 0)
            avg_cost = pos.get('avg_cost', 0)
            current = pos.get('last_price', 0) or 0
            value = pos.get('market_value', 0) or 0
            pnl = pos.get('unrealized_pnl', 0) or 0
            ret = pos.get('return_percent', 0) or 0
            
            self.positions_tree.insert('', tk.END, values=(
                pos.get('symbol', ''),
                pos.get('name', '')[:20],
                f"{qty:,.0f}",
                format_currency(avg_cost),
                format_currency(current),
                format_currency(value),
                format_currency(pnl),
                format_percent(ret)
            ), tags=('gain' if pnl >= 0 else 'loss',))
        
        # Configure tag colors
        self.positions_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.positions_tree.tag_configure('loss', foreground=COLORS['loss'])
    
    def _update_summary(self):
        """Update portfolio summary metrics."""
        if not self.current_portfolio_id:
            return
        
        summary = self.portfolio_manager.get_portfolio_summary(self.current_portfolio_id)
        
        # Update labels
        self.total_value_label.config(text=format_currency(summary.get('total_value', 0)))
        self.total_cost_label.config(text=format_currency(summary.get('total_cost', 0)))
        
        pnl = summary.get('unrealized_pnl', 0)
        self.total_pnl_label.config(
            text=format_currency(pnl),
            foreground=COLORS['gain'] if pnl >= 0 else COLORS['loss']
        )
        
        ret = summary.get('return_percent', 0)
        self.return_label.config(
            text=format_percent(ret),
            foreground=COLORS['gain'] if ret >= 0 else COLORS['loss']
        )
        
        self.positions_count_label.config(text=str(summary.get('position_count', 0)))
        
        # Update performers
        top = self.portfolio_manager.get_top_performers(self.current_portfolio_id, 5)
        self.top_performers_list.delete(0, tk.END)
        for p in top:
            self.top_performers_list.insert(
                tk.END,
                f"{p.get('symbol', '')}: {format_percent(p.get('return_percent', 0))}"
            )
        
        worst = self.portfolio_manager.get_worst_performers(self.current_portfolio_id, 5)
        self.worst_performers_list.delete(0, tk.END)
        for p in worst:
            self.worst_performers_list.insert(
                tk.END,
                f"{p.get('symbol', '')}: {format_percent(p.get('return_percent', 0))}"
            )
    
    def _create_portfolio(self):
        """Create a new portfolio."""
        name = simpledialog.askstring("New Portfolio", "Enter portfolio name:")
        if name:
            try:
                self.portfolio_manager.create_portfolio(name)
                self._load_portfolios()
                self.portfolio_var.set(name)
                self._on_portfolio_changed()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create portfolio: {e}")
    
    def _add_position_dialog(self):
        """Show dialog to add a new position."""
        if not self.current_portfolio_id:
            messagebox.showwarning("Warning", "Please select or create a portfolio first.")
            return
        
        # Simple dialog for now
        symbol = simpledialog.askstring("Add Position", "Enter stock symbol:")
        if not symbol:
            return
        
        quantity = simpledialog.askfloat("Add Position", "Enter quantity:")
        if not quantity:
            return
        
        price = simpledialog.askfloat("Add Position", "Enter purchase price:")
        if not price:
            return
        
        try:
            success = self.portfolio_manager.add_position(
                self.current_portfolio_id,
                symbol.upper(),
                quantity,
                price
            )
            if success:
                self._load_positions()
                self._update_summary()
                messagebox.showinfo("Success", f"Added {quantity} shares of {symbol}")
            else:
                messagebox.showerror("Error", f"Stock {symbol} not found in database")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add position: {e}")
    
    def _show_position_menu(self, event):
        """Show context menu for position."""
        selection = self.positions_tree.selection()
        if not selection:
            return
        
        menu = tk.Menu(self.frame, tearoff=0)
        menu.add_command(label="Sell Position", command=self._sell_position)
        menu.add_command(label="Edit Position", command=self._edit_position)
        menu.add_separator()
        menu.add_command(label="Remove Position", command=self._remove_position)
        
        menu.tk_popup(event.x_root, event.y_root)
    
    def _sell_position(self):
        """Sell shares from a position."""
        # Implementation for sell dialog
        pass
    
    def _edit_position(self):
        """Edit a position."""
        # Implementation for edit dialog
        pass
    
    def _remove_position(self):
        """Remove a position entirely."""
        selection = self.positions_tree.selection()
        if not selection:
            return
        
        item = self.positions_tree.item(selection[0])
        symbol = item['values'][0]
        
        if messagebox.askyesno("Confirm", f"Remove {symbol} from portfolio?"):
            stock = self.db.get_stock(symbol)
            if stock:
                self.db.delete_position(self.current_portfolio_id, stock['id'])
                self._load_positions()
                self._update_summary()
    
    def refresh(self):
        """Refresh portfolio data."""
        self._load_portfolios()
        if self.current_portfolio_id:
            self._load_positions()
            self._update_summary()
