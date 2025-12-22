"""
Live Market Tab for MetaQuant Nigeria.
Shows real-time market data with microstructure analysis.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Dict, Any
import logging
import threading
from datetime import datetime

try:
    import ttkbootstrap as ttk_bs
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.collectors.tradingview_collector import TradingViewCollector
from src.gui.theme import COLORS, get_font
from src.gui.components.stock_detail_dialog import show_stock_detail
from src.gui.components.sector_detail_dialog import show_sector_detail

logger = logging.getLogger(__name__)


# Sector color mapping
SECTOR_COLORS = {
    'Financial Services': '#3498db',
    'Oil & Gas': '#e74c3c',
    'Consumer Goods': '#2ecc71',
    'Industrial Goods': '#f39c12',
    'Agriculture': '#27ae60',
    'Healthcare': '#9b59b6',
    'ICT': '#1abc9c',
    'Natural Resources': '#d35400',
    'Construction/Real Estate': '#7f8c8d',
    'Services': '#34495e',
    'Conglomerates': '#95a5a6',
}


class LiveMarketTab:
    """Live market view with microstructure analysis."""
    
    REFRESH_INTERVAL_MS = 60000  # 1 minute
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.collector = TradingViewCollector()
        self.last_update = None
        self.all_stocks_data = []  # Store for click handlers
        self.sector_regions = []  # Store sector click regions
        self.sector_stocks = {}  # Store stocks by sector
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        
        # Initial load
        self._load_data()
    
    def _setup_ui(self):
        """Setup the live market UI."""
        # Header with controls
        self._create_header()
        
        # Breadth indicator bar
        self._create_breadth_bar()
        
        # Main content area
        content_frame = ttk.Frame(self.frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Top row: Sector Heatmap + Top Movers + Volume Leaders
        top_row = ttk.Frame(content_frame)
        top_row.pack(fill=tk.X, pady=(0, 10))
        
        # Sector heatmap (left)
        sector_frame = ttk.LabelFrame(top_row, text="📊 Sector Heatmap", padding=5)
        sector_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._create_sector_heatmap(sector_frame)
        
        # Top movers (middle)
        movers_frame = ttk.LabelFrame(top_row, text="📈 Top Movers", padding=5)
        movers_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self._create_movers_panel(movers_frame)
        
        # Volume leaders (right)
        volume_frame = ttk.LabelFrame(top_row, text="🔥 Volume Leaders", padding=5)
        volume_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._create_volume_leaders_panel(volume_frame)
        
        # Bottom: Most Active Table (clickable)
        active_frame = ttk.LabelFrame(content_frame, text="📋 Most Active (click for details)", padding=5)
        active_frame.pack(fill=tk.BOTH, expand=True)
        self._create_active_table(active_frame)
    
    def _create_header(self):
        """Create header with controls."""
        header_frame = ttk.Frame(self.frame)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Label(
            header_frame,
            text="🔴 Live Market",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        # Last update time
        self.update_label = ttk.Label(
            header_frame,
            text="Last update: --",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.update_label.pack(side=tk.LEFT, padx=20)
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_cb = ttk.Checkbutton(
            header_frame,
            text="Auto-refresh",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh
        )
        auto_cb.pack(side=tk.RIGHT, padx=10)
        
        # Refresh button
        if TTKBOOTSTRAP_AVAILABLE:
            refresh_btn = ttk_bs.Button(
                header_frame,
                text="↻ Refresh",
                bootstyle="success-outline",
                command=self._load_data
            )
        else:
            refresh_btn = ttk.Button(header_frame, text="↻ Refresh", command=self._load_data)
        refresh_btn.pack(side=tk.RIGHT)
    
    def _create_breadth_bar(self):
        """Create market breadth indicator bar."""
        breadth_frame = ttk.Frame(self.frame)
        breadth_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Stats labels
        stats_frame = ttk.Frame(breadth_frame)
        stats_frame.pack(fill=tk.X)
        
        self.breadth_labels = {}
        
        metrics = [
            ('ad', '📊 A/D:', '--'),
            ('above_sma', '📈 Above SMA:', '--%'),
            ('status', '⏰ Status:', '--'),
        ]
        
        for key, label, default in metrics:
            ttk.Label(stats_frame, text=label, font=get_font('small')).pack(side=tk.LEFT, padx=(0, 5))
            lbl = ttk.Label(stats_frame, text=default, font=get_font('small'),
                            foreground=COLORS['text_primary'])
            lbl.pack(side=tk.LEFT, padx=(0, 20))
            self.breadth_labels[key] = lbl
        
        # Visual breadth bar
        self.breadth_canvas = tk.Canvas(
            breadth_frame,
            height=20,
            bg=COLORS['bg_medium'],
            highlightthickness=0
        )
        self.breadth_canvas.pack(fill=tk.X, pady=(5, 0))
    
    def _create_sector_heatmap(self, parent):
        """Create sector performance heatmap."""
        self.sector_canvas = tk.Canvas(
            parent,
            height=140,
            bg=COLORS['bg_dark'],
            highlightthickness=0
        )
        self.sector_canvas.pack(fill=tk.BOTH, expand=True)
        self.sector_canvas.bind('<Configure>', self._on_sector_canvas_resize)
        self.sector_canvas.bind('<Button-1>', self._on_sector_click)
    
    def _on_sector_canvas_resize(self, event=None):
        """Handle sector canvas resize."""
        # Will be redrawn on data load
        pass
    
    def _on_sector_click(self, event):
        """Handle click on sector heatmap cell."""
        x, y = event.x, event.y
        
        # Find which sector was clicked
        for region in self.sector_regions:
            x1, y1, x2, y2, sector_name = region
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Found the sector - show detail dialog
                stocks = self.sector_stocks.get(sector_name, [])
                if stocks:
                    show_sector_detail(
                        self.frame.winfo_toplevel(),
                        sector_name,
                        stocks,
                        on_stock_click=self._show_stock_detail
                    )
                break
    
    def _create_movers_panel(self, parent):
        """Create top gainers/losers panel."""
        # Gainers
        ttk.Label(parent, text="▲ Gainers", font=get_font('small'),
                  foreground=COLORS['gain']).pack(anchor=tk.W)
        
        self.gainer_labels = []
        for i in range(5):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=1)
            
            sym_lbl = ttk.Label(row, text="--", width=10, font=get_font('small'),
                                cursor="hand2")
            sym_lbl.pack(side=tk.LEFT)
            
            chg_lbl = ttk.Label(row, text="--", font=get_font('small'),
                                foreground=COLORS['gain'])
            chg_lbl.pack(side=tk.RIGHT)
            
            self.gainer_labels.append((sym_lbl, chg_lbl))
        
        ttk.Separator(parent).pack(fill=tk.X, pady=5)
        
        # Losers
        ttk.Label(parent, text="▼ Losers", font=get_font('small'),
                  foreground=COLORS['loss']).pack(anchor=tk.W)
        
        self.loser_labels = []
        for i in range(5):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=1)
            
            sym_lbl = ttk.Label(row, text="--", width=10, font=get_font('small'),
                                cursor="hand2")
            sym_lbl.pack(side=tk.LEFT)
            
            chg_lbl = ttk.Label(row, text="--", font=get_font('small'),
                                foreground=COLORS['loss'])
            chg_lbl.pack(side=tk.RIGHT)
            
            self.loser_labels.append((sym_lbl, chg_lbl))
    
    def _create_volume_leaders_panel(self, parent):
        """Create volume leaders panel."""
        self.volume_labels = []
        
        for i in range(8):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            
            sym_lbl = ttk.Label(row, text="--", width=10, font=get_font('small'),
                                cursor="hand2")
            sym_lbl.pack(side=tk.LEFT)
            
            vol_lbl = ttk.Label(row, text="--", font=get_font('small'),
                                foreground=COLORS['text_secondary'])
            vol_lbl.pack(side=tk.RIGHT)
            
            rvol_lbl = ttk.Label(row, text="", font=get_font('small'),
                                 foreground=COLORS['warning'])
            rvol_lbl.pack(side=tk.RIGHT, padx=(0, 10))
            
            self.volume_labels.append((sym_lbl, rvol_lbl, vol_lbl))
    
    def _create_active_table(self, parent):
        """Create most active stocks table."""
        # Create Treeview
        columns = ('symbol', 'price', 'change', 'volume', 'rvol', 'momentum')
        
        self.active_tree = ttk.Treeview(parent, columns=columns, show='headings', height=10)
        
        # Define headings
        self.active_tree.heading('symbol', text='Symbol')
        self.active_tree.heading('price', text='Price')
        self.active_tree.heading('change', text='Change')
        self.active_tree.heading('volume', text='Volume')
        self.active_tree.heading('rvol', text='RVOL')
        self.active_tree.heading('momentum', text='Mom')
        
        # Column widths
        self.active_tree.column('symbol', width=100, anchor='w')
        self.active_tree.column('price', width=100, anchor='e')
        self.active_tree.column('change', width=80, anchor='e')
        self.active_tree.column('volume', width=100, anchor='e')
        self.active_tree.column('rvol', width=70, anchor='center')
        self.active_tree.column('momentum', width=70, anchor='center')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.active_tree.yview)
        self.active_tree.configure(yscrollcommand=scrollbar.set)
        
        self.active_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind click event
        self.active_tree.bind('<Double-1>', self._on_stock_click)
        
        # Row tags for coloring
        self.active_tree.tag_configure('gain', foreground=COLORS['gain'])
        self.active_tree.tag_configure('loss', foreground=COLORS['loss'])
        self.active_tree.tag_configure('evenrow', background=COLORS['bg_medium'])
    
    def _on_stock_click(self, event):
        """Handle stock row click to show detail dialog."""
        selection = self.active_tree.selection()
        if not selection:
            return
        
        item = self.active_tree.item(selection[0])
        symbol = item['values'][0] if item['values'] else None
        
        if symbol:
            # Find stock data
            stock_data = next(
                (s for s in self.all_stocks_data if s.get('symbol') == symbol),
                {}
            )
            
            # Show detail dialog
            show_stock_detail(self.frame.winfo_toplevel(), symbol, stock_data, self.db)
    
    def _load_data(self):
        """Load market data."""
        def fetch():
            try:
                # Get market snapshot
                snapshot = self.collector.get_market_snapshot()
                
                # Get all stocks data
                all_stocks = self.collector.get_all_stocks()
                stocks_list = all_stocks.to_dict('records') if not all_stocks.empty else []
                
                # Get sector mapping from database
                sector_map = self._get_sector_mapping()
                
                # Enrich stocks with sector data
                for stock in stocks_list:
                    symbol = stock.get('symbol', '')
                    stock['sector'] = sector_map.get(symbol, 'Other')
                
                self.frame.after(0, lambda: self._update_ui(snapshot, stocks_list))
            except Exception as e:
                logger.error(f"Failed to load market data: {e}")
                self.frame.after(0, lambda: self._show_error(str(e)))
        
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
        self.update_label.config(text="Loading...")
    
    def _get_sector_mapping(self) -> Dict[str, str]:
        """Get symbol to sector mapping from database."""
        try:
            results = self.db.conn.execute(
                "SELECT symbol, sector FROM stocks WHERE sector IS NOT NULL AND sector != ''"
            ).fetchall()
            return {row[0]: row[1] for row in results}
        except Exception as e:
            logger.error(f"Error fetching sector mapping: {e}")
            return {}
    
    def _update_ui(self, snapshot: Dict[str, Any], stocks_list: List[Dict]):
        """Update all UI components with new data."""
        if 'error' in snapshot:
            self._show_error(snapshot['error'])
            return
        
        self.all_stocks_data = stocks_list
        self.last_update = datetime.now()
        self.update_label.config(text=f"Last: {self.last_update.strftime('%H:%M:%S')}")
        
        # Update breadth bar
        self._update_breadth(snapshot)
        
        # Update sector heatmap
        self._update_sector_heatmap(stocks_list)
        
        # Update gainers/losers
        self._update_movers(snapshot)
        
        # Update volume leaders
        self._update_volume_leaders(stocks_list)
        
        # Update most active table
        self._update_active_table(stocks_list)
        
        # Schedule next refresh
        if self.auto_refresh_var.get():
            self.frame.after(self.REFRESH_INTERVAL_MS, self._load_data)
    
    def _update_breadth(self, snapshot: Dict[str, Any]):
        """Update breadth indicator bar."""
        gainers = snapshot.get('gainers', 0)
        losers = snapshot.get('losers', 0)
        total = snapshot.get('total_stocks', 0)
        
        # Update labels
        self.breadth_labels['ad'].config(
            text=f"{gainers}/{losers}",
            foreground=COLORS['gain'] if gainers > losers else COLORS['loss']
        )
        
        # Estimate % above SMA (simplified)
        pct_above = (gainers / total * 100) if total > 0 else 0
        self.breadth_labels['above_sma'].config(text=f"{pct_above:.0f}%")
        
        # Market status
        from src.gui.app import is_market_open
        status = "🟢 Open" if is_market_open() else "🔴 Closed"
        self.breadth_labels['status'].config(
            text=status,
            foreground=COLORS['gain'] if is_market_open() else COLORS['text_muted']
        )
        
        # Draw breadth bar
        self._draw_breadth_bar(gainers, losers, total - gainers - losers)
    
    def _draw_breadth_bar(self, gainers: int, losers: int, unchanged: int):
        """Draw the visual breadth bar."""
        total = gainers + losers + unchanged
        if total == 0:
            return
        
        self.breadth_canvas.delete("all")
        width = self.breadth_canvas.winfo_width()
        if width <= 1:
            width = 400
        height = 20
        
        g_width = int((gainers / total) * width)
        l_width = int((losers / total) * width)
        u_width = width - g_width - l_width
        
        x = 0
        if g_width > 0:
            self.breadth_canvas.create_rectangle(x, 0, x + g_width, height,
                                                  fill=COLORS['gain'], outline="")
            if g_width > 25:
                self.breadth_canvas.create_text(x + g_width/2, height/2,
                                                text=str(gainers), fill="white")
            x += g_width
        
        if u_width > 0:
            self.breadth_canvas.create_rectangle(x, 0, x + u_width, height,
                                                  fill=COLORS['text_muted'], outline="")
            x += u_width
        
        if l_width > 0:
            self.breadth_canvas.create_rectangle(x, 0, x + l_width, height,
                                                  fill=COLORS['loss'], outline="")
            if l_width > 25:
                self.breadth_canvas.create_text(x + l_width/2, height/2,
                                                text=str(losers), fill="white")
    
    def _update_sector_heatmap(self, stocks_list: List[Dict]):
        """Update sector heatmap visualization."""
        # Group stocks by sector
        sectors = {}
        self.sector_stocks = {}  # Reset sector stocks
        
        for stock in stocks_list:
            sector = stock.get('sector', 'Other') or 'Other'
            if sector not in sectors:
                sectors[sector] = {'changes': [], 'count': 0, 'stocks': []}
            sectors[sector]['changes'].append(stock.get('change', 0) or 0)
            sectors[sector]['count'] += 1
            sectors[sector]['stocks'].append(stock)
        
        # Store stocks by sector for click handling
        self.sector_stocks = {sector: data['stocks'] for sector, data in sectors.items()}
        
        # Calculate average changes
        sector_perf = {}
        for sector, data in sectors.items():
            avg_change = sum(data['changes']) / len(data['changes']) if data['changes'] else 0
            sector_perf[sector] = {
                'avg_change': avg_change,
                'count': data['count']
            }
        
        # Draw heatmap
        self._draw_sector_heatmap(sector_perf)
    
    def _draw_sector_heatmap(self, sector_perf: Dict[str, Dict]):
        """Draw the sector heatmap."""
        self.sector_canvas.delete("all")
        self.sector_regions = []  # Reset click regions
        
        width = self.sector_canvas.winfo_width()
        height = self.sector_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width, height = 250, 140
        
        sectors = list(sector_perf.items())
        if not sectors:
            return
        
        # Set cursor to hand
        self.sector_canvas.configure(cursor="hand2")
        
        # Grid layout
        cols = 3
        rows = (len(sectors) + cols - 1) // cols
        
        cell_w = width // cols
        cell_h = height // rows if rows > 0 else height
        
        for i, (sector, data) in enumerate(sectors):
            row = i // cols
            col = i % cols
            
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = x1 + cell_w - 2
            y2 = y1 + cell_h - 2
            
            # Store region for click detection
            self.sector_regions.append((x1, y1, x2, y2, sector))
            
            # Color based on change
            change = data['avg_change']
            if change > 2:
                color = '#27ae60'  # Strong green
            elif change > 0:
                color = '#2ecc71'  # Green
            elif change < -2:
                color = '#c0392b'  # Strong red
            elif change < 0:
                color = '#e74c3c'  # Red
            else:
                color = COLORS['text_muted']
            
            # Draw cell
            self.sector_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=COLORS['bg_dark'])
            
            # Sector label (truncated)
            short_name = sector[:10] + '..' if len(sector) > 12 else sector
            self.sector_canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2 - 8,
                text=short_name, fill="white", font=get_font('small')
            )
            
            # Change value
            self.sector_canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 8,
                text=f"{change:+.1f}%", fill="white", font=get_font('small')
            )
    
    def _update_movers(self, snapshot: Dict[str, Any]):
        """Update top gainers and losers."""
        # Gainers
        for i, stock in enumerate(snapshot.get('top_gainers', [])[:5]):
            if i < len(self.gainer_labels):
                sym, chg = self.gainer_labels[i]
                sym.config(text=stock['symbol'])
                chg.config(text=f"+{stock['change']:.1f}%")
                # Bind click
                sym.bind('<Button-1>', lambda e, s=stock: self._show_stock_detail(s))
        
        # Losers
        for i, stock in enumerate(snapshot.get('top_losers', [])[:5]):
            if i < len(self.loser_labels):
                sym, chg = self.loser_labels[i]
                sym.config(text=stock['symbol'])
                chg.config(text=f"{stock['change']:.1f}%")
                sym.bind('<Button-1>', lambda e, s=stock: self._show_stock_detail(s))
    
    def _update_volume_leaders(self, stocks_list: List[Dict]):
        """Update volume leaders panel."""
        # Sort by volume
        sorted_by_vol = sorted(stocks_list, key=lambda x: x.get('volume', 0) or 0, reverse=True)
        
        for i, stock in enumerate(sorted_by_vol[:8]):
            if i < len(self.volume_labels):
                sym, rvol, vol = self.volume_labels[i]
                
                sym.config(text=stock.get('symbol', '--'))
                
                # Volume formatting
                v = stock.get('volume', 0) or 0
                if v >= 1_000_000:
                    vol.config(text=f"{v/1_000_000:.1f}M")
                elif v >= 1_000:
                    vol.config(text=f"{v/1_000:.1f}K")
                else:
                    vol.config(text=str(v))
                
                # RVOL (placeholder - would need historical avg)
                rvol.config(text="")
                
                # Bind click
                sym.bind('<Button-1>', lambda e, s=stock: self._show_stock_detail(s))
    
    def _update_active_table(self, stocks_list: List[Dict]):
        """Update most active stocks table."""
        # Clear existing
        for item in self.active_tree.get_children():
            self.active_tree.delete(item)
        
        # Sort by volume
        sorted_stocks = sorted(stocks_list, key=lambda x: x.get('volume', 0) or 0, reverse=True)
        
        for i, stock in enumerate(sorted_stocks[:20]):
            symbol = stock.get('symbol', '')
            close = stock.get('close', 0) or 0
            change = stock.get('change', 0) or 0
            volume = stock.get('volume', 0) or 0
            
            # Format volume
            if volume >= 1_000_000:
                vol_str = f"{volume/1_000_000:.1f}M"
            elif volume >= 1_000:
                vol_str = f"{volume/1_000:.1f}K"
            else:
                vol_str = str(int(volume))
            
            # Change formatting
            if change > 0:
                chg_str = f"+{change:.2f}%"
                tag = 'gain'
            elif change < 0:
                chg_str = f"{change:.2f}%"
                tag = 'loss'
            else:
                chg_str = f"{change:.2f}%"
                tag = ''
            
            # Momentum indicator (simplified)
            if change > 3:
                mom = "↑↑"
            elif change > 0:
                mom = "↑"
            elif change < -3:
                mom = "↓↓"
            elif change < 0:
                mom = "↓"
            else:
                mom = "→"
            
            # Add row
            tags = (tag, 'evenrow') if i % 2 == 0 else (tag,)
            self.active_tree.insert('', 'end', values=(
                symbol,
                f"₦{close:,.2f}",
                chg_str,
                vol_str,
                "--",  # RVOL placeholder
                mom
            ), tags=tags)
    
    def _show_stock_detail(self, stock: Dict):
        """Show stock detail dialog."""
        symbol = stock.get('symbol', '')
        if symbol:
            show_stock_detail(self.frame.winfo_toplevel(), symbol, stock, self.db)
    
    def _show_error(self, error: str):
        """Show error message."""
        self.update_label.config(text=f"Error: {error}", foreground=COLORS['loss'])
    
    def _toggle_auto_refresh(self):
        """Toggle auto-refresh."""
        if self.auto_refresh_var.get():
            self._load_data()
    
    def refresh(self):
        """Manual refresh."""
        self._load_data()
