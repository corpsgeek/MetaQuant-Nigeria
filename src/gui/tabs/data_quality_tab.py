"""
Data Quality Dashboard Tab for MetaQuant Nigeria.
Monitors data completeness, freshness, and identifies missing/stale data.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

logger = logging.getLogger(__name__)


class DataQualityTab:
    """
    Data Quality Dashboard - monitors data health across all stocks.
    
    Features:
    - Summary cards (coverage, freshness, ML-ready)
    - Data freshness table with status indicators
    - Refresh/fetch missing data actions
    """
    
    # Thresholds
    MIN_OHLCV_FOR_ML = 50  # Minimum rows needed for ML predictions
    STALE_DAYS_WARNING = 3  # Days before data considered stale
    STALE_DAYS_CRITICAL = 7  # Days before data critically stale
    
    def __init__(self, parent, db):
        self.parent = parent
        self.db = db
        self.data_stats: List[Dict] = []
        
        self._create_ui()
        self._load_data()
    
    def _create_ui(self):
        """Create the Data Quality Dashboard UI."""
        # Main container with padding
        main_frame = ttk.Frame(self.parent, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(
            header_frame,
            text="📊 Data Quality Dashboard",
            font=('Helvetica', 18, 'bold')
        ).pack(side='left')
        
        # Refresh button
        ttk.Button(
            header_frame,
            text="🔄 Refresh Analysis",
            command=self._load_data
        ).pack(side='right', padx=5)
        
        # Fetch missing button
        self.fetch_btn = ttk.Button(
            header_frame,
            text="📥 Fetch Missing Data",
            command=self._fetch_missing_data
        )
        self.fetch_btn.pack(side='right', padx=5)
        
        # ==================== SUMMARY CARDS ====================
        cards_frame = ttk.Frame(main_frame)
        cards_frame.pack(fill='x', pady=(0, 15))
        
        # Configure grid columns
        for i in range(5):
            cards_frame.columnconfigure(i, weight=1)
        
        self.cards = {}
        card_configs = [
            ('total', '📈 Total Stocks', '0', '#3498db'),
            ('ml_ready', '🤖 ML Ready', '0', '#27ae60'),
            ('coverage', '📊 Coverage', '0%', '#9b59b6'),
            ('stale', '⚠️ Stale Data', '0', '#f39c12'),
            ('critical', '❌ Critical', '0', '#e74c3c')
        ]
        
        for idx, (key, title, default, color) in enumerate(card_configs):
            card = self._create_card(cards_frame, title, default, color)
            card.grid(row=0, column=idx, padx=5, sticky='nsew')
            self.cards[key] = card
        
        # ==================== STATUS FILTER ====================
        filter_frame = ttk.Frame(main_frame)
        filter_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(filter_frame, text="Filter:").pack(side='left', padx=(0, 10))
        
        self.filter_var = tk.StringVar(value='all')
        filters = [
            ('All', 'all'),
            ('✅ Good', 'good'),
            ('⚠️ Stale', 'stale'),
            ('❌ Critical', 'critical'),
            ('🔴 Insufficient', 'insufficient')
        ]
        
        for label, value in filters:
            ttk.Radiobutton(
                filter_frame,
                text=label,
                variable=self.filter_var,
                value=value,
                command=self._apply_filter
            ).pack(side='left', padx=5)
        
        # Search
        ttk.Label(filter_frame, text="Search:").pack(side='left', padx=(20, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self._apply_filter())
        ttk.Entry(filter_frame, textvariable=self.search_var, width=20).pack(side='left')
        
        # ==================== DATA TABLE ====================
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill='both', expand=True)
        
        # Treeview with scrollbar
        columns = ('symbol', 'name', 'sector', 'ohlcv_count', 'last_update', 'days_stale', 'status')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        
        # Column headings
        self.tree.heading('symbol', text='Symbol', command=lambda: self._sort_by('symbol'))
        self.tree.heading('name', text='Name', command=lambda: self._sort_by('name'))
        self.tree.heading('sector', text='Sector', command=lambda: self._sort_by('sector'))
        self.tree.heading('ohlcv_count', text='OHLCV Rows', command=lambda: self._sort_by('ohlcv_count'))
        self.tree.heading('last_update', text='Last Update', command=lambda: self._sort_by('last_update'))
        self.tree.heading('days_stale', text='Days Stale', command=lambda: self._sort_by('days_stale'))
        self.tree.heading('status', text='Status', command=lambda: self._sort_by('status'))
        
        # Column widths
        self.tree.column('symbol', width=100, anchor='w')
        self.tree.column('name', width=200, anchor='w')
        self.tree.column('sector', width=120, anchor='w')
        self.tree.column('ohlcv_count', width=100, anchor='center')
        self.tree.column('last_update', width=120, anchor='center')
        self.tree.column('days_stale', width=100, anchor='center')
        self.tree.column('status', width=100, anchor='center')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Configure row tags for coloring
        self.tree.tag_configure('good', background='#d5f5e3')
        self.tree.tag_configure('stale', background='#fdebd0')
        self.tree.tag_configure('critical', background='#fadbd8')
        self.tree.tag_configure('insufficient', background='#f5b7b1')
        
        # ==================== STATUS BAR ====================
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief='sunken')
        status_bar.pack(fill='x', pady=(10, 0))
    
    def _create_card(self, parent, title: str, value: str, color: str) -> ttk.Frame:
        """Create a summary card widget."""
        card = ttk.Frame(parent, relief='solid', borderwidth=1)
        card.configure(padding=10)
        
        ttk.Label(
            card,
            text=title,
            font=('Helvetica', 10)
        ).pack(anchor='w')
        
        value_label = ttk.Label(
            card,
            text=value,
            font=('Helvetica', 24, 'bold')
        )
        value_label.pack(anchor='w', pady=(5, 0))
        value_label.value_label = value_label  # Store reference
        
        card.value_label = value_label
        return card
    
    def _update_card(self, key: str, value: str):
        """Update a card's value."""
        if key in self.cards:
            self.cards[key].value_label.configure(text=value)
    
    def _load_data(self):
        """Load data quality statistics from database."""
        self.status_var.set("Loading data quality metrics...")
        
        def load():
            try:
                # Get all stocks
                stocks = self.db.conn.execute("""
                    SELECT symbol, name, sector FROM stocks
                    ORDER BY symbol
                """).fetchall()
                
                self.data_stats = []
                now = datetime.now()
                
                for symbol, name, sector in stocks:
                    # Get OHLCV count and last update
                    result = self.db.conn.execute("""
                        SELECT COUNT(*), MAX(datetime)
                        FROM intraday_ohlcv
                        WHERE symbol = ?
                    """, [symbol]).fetchone()
                    
                    # Handle case where result is None
                    if result is None:
                        ohlcv_count = 0
                        last_update = None
                    else:
                        ohlcv_count = result[0] or 0
                        last_update = result[1] if result[1] else None
                    
                    # Calculate staleness
                    if last_update:
                        try:
                            # Handle both datetime objects and strings
                            if isinstance(last_update, datetime):
                                last_dt = last_update
                            elif isinstance(last_update, str):
                                last_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00').replace('+00:00', ''))
                            else:
                                last_dt = datetime.now()
                            days_stale = (now - last_dt).days
                        except:
                            days_stale = 999
                    else:
                        days_stale = 999
                    
                    # Determine status
                    if ohlcv_count < self.MIN_OHLCV_FOR_ML:
                        status = 'insufficient'
                        status_icon = '🔴 Insufficient'
                    elif days_stale >= self.STALE_DAYS_CRITICAL:
                        status = 'critical'
                        status_icon = '❌ Critical'
                    elif days_stale >= self.STALE_DAYS_WARNING:
                        status = 'stale'
                        status_icon = '⚠️ Stale'
                    else:
                        status = 'good'
                        status_icon = '✅ Good'
                    
                    # Format last_update for display
                    if last_update:
                        if isinstance(last_update, datetime):
                            last_update_str = last_update.strftime('%Y-%m-%d')
                        elif isinstance(last_update, str):
                            last_update_str = last_update[:10]
                        else:
                            last_update_str = str(last_update)[:10]
                    else:
                        last_update_str = 'Never'
                    
                    self.data_stats.append({
                        'symbol': symbol,
                        'name': name or '',
                        'sector': sector or 'Unknown',
                        'ohlcv_count': ohlcv_count,
                        'last_update': last_update_str,
                        'days_stale': days_stale if days_stale < 999 else 'N/A',
                        'status': status,
                        'status_icon': status_icon
                    })
                
                # Update UI on main thread
                self.parent.after(0, self._update_ui)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error loading data quality: {error_msg}")
                self.parent.after(0, lambda msg=error_msg: self.status_var.set(f"Error: {msg}"))
        
        threading.Thread(target=load, daemon=True).start()
    
    def _update_ui(self):
        """Update UI with loaded data."""
        # Calculate summary stats
        total = len(self.data_stats)
        ml_ready = sum(1 for s in self.data_stats if s['ohlcv_count'] >= self.MIN_OHLCV_FOR_ML)
        coverage = (ml_ready / total * 100) if total > 0 else 0
        stale = sum(1 for s in self.data_stats if s['status'] == 'stale')
        critical = sum(1 for s in self.data_stats if s['status'] in ('critical', 'insufficient'))
        
        # Update cards
        self._update_card('total', str(total))
        self._update_card('ml_ready', str(ml_ready))
        self._update_card('coverage', f"{coverage:.0f}%")
        self._update_card('stale', str(stale))
        self._update_card('critical', str(critical))
        
        # Populate table
        self._apply_filter()
        
        self.status_var.set(f"Loaded {total} stocks | ML Ready: {ml_ready} | Stale: {stale} | Critical: {critical}")
    
    def _apply_filter(self):
        """Apply filter to the data table."""
        # Clear table
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        filter_value = self.filter_var.get()
        search_text = self.search_var.get().lower()
        
        for stat in self.data_stats:
            # Apply filter
            if filter_value != 'all' and stat['status'] != filter_value:
                continue
            
            # Apply search
            if search_text:
                if search_text not in stat['symbol'].lower() and \
                   search_text not in stat['name'].lower() and \
                   search_text not in stat['sector'].lower():
                    continue
            
            # Insert row
            self.tree.insert('', 'end', values=(
                stat['symbol'],
                stat['name'][:30],
                stat['sector'],
                stat['ohlcv_count'],
                stat['last_update'],
                stat['days_stale'],
                stat['status_icon']
            ), tags=(stat['status'],))
    
    def _sort_by(self, column: str):
        """Sort table by column."""
        # Get current sort direction
        reverse = getattr(self, f'_sort_{column}_reverse', False)
        setattr(self, f'_sort_{column}_reverse', not reverse)
        
        # Sort data
        if column == 'ohlcv_count':
            self.data_stats.sort(key=lambda x: x[column], reverse=reverse)
        elif column == 'days_stale':
            self.data_stats.sort(key=lambda x: x[column] if isinstance(x[column], int) else 9999, reverse=reverse)
        else:
            self.data_stats.sort(key=lambda x: str(x.get(column, '')).lower(), reverse=reverse)
        
        self._apply_filter()
    
    def _fetch_missing_data(self):
        """Trigger fetch for stocks with missing/insufficient data."""
        # Find stocks needing data
        missing_stocks = [s['symbol'] for s in self.data_stats 
                         if s['ohlcv_count'] < self.MIN_OHLCV_FOR_ML]
        
        if not missing_stocks:
            messagebox.showinfo("Data Quality", "All stocks have sufficient data!")
            return
        
        count = len(missing_stocks)
        if not messagebox.askyesno(
            "Fetch Missing Data",
            f"Found {count} stocks with insufficient data.\n\nFetch data for these stocks?\n\nThis may take a few minutes."
        ):
            return
        
        self.fetch_btn.configure(state='disabled')
        self.status_var.set(f"Fetching data for {count} stocks...")
        
        def fetch():
            try:
                # Import collector if available
                from src.collectors.intraday_collector import IntradayCollector
                collector = IntradayCollector(self.db)
                
                fetched = 0
                for i, symbol in enumerate(missing_stocks):
                    try:
                        # Use backfill_symbol which fetches daily data
                        result = collector.backfill_symbol(symbol, intervals=['1d'], n_bars=500)
                        if result and result.get('1d', 0) > 0:
                            fetched += 1
                        self.parent.after(0, lambda s=symbol, idx=i: 
                            self.status_var.set(f"Fetching {s}... ({idx+1}/{count})"))
                    except Exception as e:
                        logger.warning(f"Failed to fetch {symbol}: {e}")
                
                # Reload data
                self.parent.after(0, lambda: self._on_fetch_complete(fetched, count))
                
            except ImportError:
                self.parent.after(0, lambda: messagebox.showerror(
                    "Error", "Intraday collector not available"))
            except Exception as e:
                self.parent.after(0, lambda: messagebox.showerror(
                    "Error", f"Fetch failed: {e}"))
            finally:
                self.parent.after(0, lambda: self.fetch_btn.configure(state='normal'))
        
        threading.Thread(target=fetch, daemon=True).start()
    
    def _on_fetch_complete(self, fetched: int, total: int):
        """Handle fetch completion."""
        self.status_var.set(f"Fetch complete! Updated {fetched}/{total} stocks")
        messagebox.showinfo(
            "Fetch Complete",
            f"Successfully fetched data for {fetched}/{total} stocks.\n\nRefreshing analysis..."
        )
        self._load_data()
    
    def refresh(self):
        """Refresh the dashboard data."""
        self._load_data()
