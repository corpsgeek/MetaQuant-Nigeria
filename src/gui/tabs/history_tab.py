"""
History Tab for MetaQuant Nigeria.
View historical market data by selecting a date.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Dict
import logging

try:
    import ttkbootstrap as ttk_bs
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.gui.theme import COLORS, get_font, format_currency, format_percent


logger = logging.getLogger(__name__)


class HistoryTab:
    """Historical market view - see market performance on any past date."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        self._load_dates()
    
    def _setup_ui(self):
        """Setup UI."""
        # Header with date selector
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            header, 
            text="📅 Historical Market View", 
            font=get_font('subheading')
        ).pack(side=tk.LEFT)
        
        # Date selector
        date_frame = ttk.Frame(header)
        date_frame.pack(side=tk.RIGHT)
        
        ttk.Label(date_frame, text="Select Date:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.date_var = tk.StringVar()
        self.date_combo = ttk.Combobox(
            date_frame,
            textvariable=self.date_var,
            state="readonly",
            width=15
        )
        self.date_combo.pack(side=tk.LEFT)
        self.date_combo.bind('<<ComboboxSelected>>', lambda e: self._load_market_data())
        
        # Summary frame
        summary_frame = ttk.Frame(self.frame)
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Summary cards
        self.summary_labels = {}
        summary_items = [
            ('date', '📅 Date', '-'),
            ('gainers', '📈 Gainers', '0'),
            ('losers', '📉 Losers', '0'),
            ('volume', '📊 Volume', '0'),
        ]
        
        for key, label, default in summary_items:
            card = ttk.LabelFrame(summary_frame, text=label, padding=10)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            value_label = ttk.Label(
                card, 
                text=default, 
                font=get_font('subheading'),
                foreground=COLORS['primary']
            )
            value_label.pack()
            self.summary_labels[key] = value_label
        
        # Top movers frame
        movers_frame = ttk.Frame(self.frame)
        movers_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Top gainers
        gainers_frame = ttk.LabelFrame(movers_frame, text="🚀 Top Gainers", padding=10)
        gainers_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.gainers_list = ttk.Treeview(
            gainers_frame,
            columns=('symbol', 'price', 'change'),
            show='headings',
            height=5
        )
        self.gainers_list.heading('symbol', text='Symbol')
        self.gainers_list.heading('price', text='Close')
        self.gainers_list.heading('change', text='Change')
        self.gainers_list.column('symbol', width=80)
        self.gainers_list.column('price', width=80)
        self.gainers_list.column('change', width=80)
        self.gainers_list.tag_configure('gain', foreground=COLORS['gain'])
        self.gainers_list.pack(fill=tk.BOTH, expand=True)
        
        # Top losers
        losers_frame = ttk.LabelFrame(movers_frame, text="📉 Top Losers", padding=10)
        losers_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.losers_list = ttk.Treeview(
            losers_frame,
            columns=('symbol', 'price', 'change'),
            show='headings',
            height=5
        )
        self.losers_list.heading('symbol', text='Symbol')
        self.losers_list.heading('price', text='Close')
        self.losers_list.heading('change', text='Change')
        self.losers_list.column('symbol', width=80)
        self.losers_list.column('price', width=80)
        self.losers_list.column('change', width=80)
        self.losers_list.tag_configure('loss', foreground=COLORS['loss'])
        self.losers_list.pack(fill=tk.BOTH, expand=True)
        
        # Full market table
        table_frame = ttk.LabelFrame(self.frame, text="All Stocks on This Date", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ('symbol', 'name', 'sector', 'open', 'high', 'low', 'close', 'volume', 'change')
        self.market_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            selectmode='browse'
        )
        
        self.market_tree.heading('symbol', text='Symbol')
        self.market_tree.heading('name', text='Name')
        self.market_tree.heading('sector', text='Sector')
        self.market_tree.heading('open', text='Open')
        self.market_tree.heading('high', text='High')
        self.market_tree.heading('low', text='Low')
        self.market_tree.heading('close', text='Close')
        self.market_tree.heading('volume', text='Volume')
        self.market_tree.heading('change', text='Change%')
        
        self.market_tree.column('symbol', width=80)
        self.market_tree.column('name', width=200)
        self.market_tree.column('sector', width=120)
        self.market_tree.column('open', width=70)
        self.market_tree.column('high', width=70)
        self.market_tree.column('low', width=70)
        self.market_tree.column('close', width=70)
        self.market_tree.column('volume', width=100)
        self.market_tree.column('change', width=70)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=y_scroll.set)
        
        self.market_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags
        self.market_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.market_tree.tag_configure('loss', foreground=COLORS['loss'])
    
    def _load_dates(self):
        """Load available dates."""
        dates = self.db.get_price_history_dates()
        self.date_combo['values'] = dates
        
        if dates:
            self.date_combo.set(dates[0])
            self._load_market_data()
    
    def _load_market_data(self):
        """Load market data for selected date."""
        date = self.date_var.get()
        if not date:
            return
        
        # Get market data on this date
        market_data = self.db.get_market_on_date(date)
        
        if not market_data:
            return
        
        # Calculate changes (comparing close to open within same day)
        for stock in market_data:
            open_p = float(stock.get('open') or 0)
            close = float(stock.get('close') or 0)
            if open_p > 0:
                stock['change_pct'] = (close - open_p) / open_p * 100
            else:
                stock['change_pct'] = 0
        
        # Sort by change
        gainers = sorted([s for s in market_data if s['change_pct'] > 0], 
                        key=lambda x: -x['change_pct'])
        losers = sorted([s for s in market_data if s['change_pct'] < 0],
                       key=lambda x: x['change_pct'])
        
        # Update summary
        self.summary_labels['date'].config(text=date)
        self.summary_labels['gainers'].config(
            text=str(len(gainers)),
            foreground=COLORS['gain']
        )
        self.summary_labels['losers'].config(
            text=str(len(losers)),
            foreground=COLORS['loss']
        )
        
        total_volume = sum(int(s.get('volume') or 0) for s in market_data)
        self.summary_labels['volume'].config(text=f"{total_volume:,}")
        
        # Update gainers list
        for item in self.gainers_list.get_children():
            self.gainers_list.delete(item)
        for stock in gainers[:5]:
            self.gainers_list.insert('', tk.END, values=(
                stock['symbol'],
                format_currency(stock['close']),
                f"+{stock['change_pct']:.2f}%"
            ), tags=('gain',))
        
        # Update losers list
        for item in self.losers_list.get_children():
            self.losers_list.delete(item)
        for stock in losers[:5]:
            self.losers_list.insert('', tk.END, values=(
                stock['symbol'],
                format_currency(stock['close']),
                f"{stock['change_pct']:.2f}%"
            ), tags=('loss',))
        
        # Update main table
        for item in self.market_tree.get_children():
            self.market_tree.delete(item)
        
        for stock in sorted(market_data, key=lambda x: x['symbol']):
            change = stock['change_pct']
            tag = 'gain' if change > 0 else 'loss' if change < 0 else ''
            
            self.market_tree.insert('', tk.END, values=(
                stock.get('symbol', ''),
                stock.get('name', '')[:30],
                stock.get('sector', ''),
                format_currency(stock.get('open') or 0),
                format_currency(stock.get('high') or 0),
                format_currency(stock.get('low') or 0),
                format_currency(stock.get('close') or 0),
                f"{int(stock.get('volume') or 0):,}",
                f"{change:+.2f}%"
            ), tags=(tag,))
    
    def refresh(self):
        """Refresh data."""
        self._load_dates()
