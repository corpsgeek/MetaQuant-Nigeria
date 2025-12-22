"""
Stock Detail Dialog for MetaQuant Nigeria.
Popup showing detailed technical analysis for a selected stock.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional
import logging
import threading

try:
    import ttkbootstrap as ttk_bs
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.gui.theme import COLORS, get_font
from src.collectors.tradingview_collector import TradingViewCollector

logger = logging.getLogger(__name__)


class StockDetailDialog:
    """
    Popup dialog showing detailed stock analysis.
    
    Features:
    - Price summary
    - Technical indicators (RSI, MACD, SMA)
    - Volume analysis
    - Buy/Sell/Neutral rating
    """
    
    def __init__(self, parent, symbol: str, stock_data: Dict[str, Any], db=None):
        """
        Initialize the stock detail dialog.
        
        Args:
            parent: Parent window
            symbol: Stock symbol
            stock_data: Current stock data dictionary
            db: Optional database manager
        """
        self.parent = parent
        self.symbol = symbol
        self.stock_data = stock_data
        self.db = db
        self.collector = TradingViewCollector()
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"📊 {symbol} - Stock Analysis")
        self.dialog.geometry("500x600")
        self.dialog.configure(bg=COLORS['bg_dark'])
        
        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center on parent
        self._center_window()
        
        # Setup UI
        self._setup_ui()
        
        # Load technical analysis in background
        self._load_technical_analysis()
    
    def _center_window(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        # Get parent position
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_w = self.parent.winfo_width()
        parent_h = self.parent.winfo_height()
        
        # Get dialog size
        dialog_w = 500
        dialog_h = 600
        
        # Calculate position
        x = parent_x + (parent_w - dialog_w) // 2
        y = parent_y + (parent_h - dialog_h) // 2
        
        self.dialog.geometry(f"{dialog_w}x{dialog_h}+{x}+{y}")
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            header_frame,
            text=self.symbol,
            font=get_font('heading'),
            foreground=COLORS['primary']
        ).pack(side=tk.LEFT)
        
        # Close button
        close_btn = ttk.Button(
            header_frame,
            text="✕",
            width=3,
            command=self.dialog.destroy
        )
        close_btn.pack(side=tk.RIGHT)
        
        # Price info
        price_frame = ttk.LabelFrame(main_frame, text="Price Summary", padding=10)
        price_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_price_section(price_frame)
        
        # Technical indicators
        tech_frame = ttk.LabelFrame(main_frame, text="Technical Indicators", padding=10)
        tech_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_technical_section(tech_frame)
        
        # Rating
        rating_frame = ttk.LabelFrame(main_frame, text="Analysis Rating", padding=10)
        rating_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_rating_section(rating_frame)
        
        # Volume analysis
        volume_frame = ttk.LabelFrame(main_frame, text="Volume Analysis", padding=10)
        volume_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_volume_section(volume_frame)
    
    def _create_price_section(self, parent):
        """Create price summary section."""
        current_price = self.stock_data.get('close', 0)
        change = self.stock_data.get('change', 0)
        
        # Current price
        price_row = ttk.Frame(parent)
        price_row.pack(fill=tk.X)
        
        ttk.Label(price_row, text="Current Price:", font=get_font('body')).pack(side=tk.LEFT)
        ttk.Label(
            price_row,
            text=f"₦{current_price:,.2f}",
            font=get_font('subheading'),
            foreground=COLORS['text_primary']
        ).pack(side=tk.RIGHT)
        
        # Change
        change_row = ttk.Frame(parent)
        change_row.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(change_row, text="Change:", font=get_font('body')).pack(side=tk.LEFT)
        
        change_color = COLORS['gain'] if change > 0 else COLORS['loss'] if change < 0 else COLORS['text_muted']
        change_text = f"+{change:.2f}%" if change > 0 else f"{change:.2f}%"
        
        ttk.Label(
            change_row,
            text=change_text,
            font=get_font('body_bold'),
            foreground=change_color
        ).pack(side=tk.RIGHT)
        
        # OHLC
        ohlc_row = ttk.Frame(parent)
        ohlc_row.pack(fill=tk.X, pady=(10, 0))
        
        open_p = self.stock_data.get('open', current_price)
        high = self.stock_data.get('high', current_price)
        low = self.stock_data.get('low', current_price)
        
        ttk.Label(ohlc_row, text=f"O: ₦{open_p:,.2f}", font=get_font('small'),
                  foreground=COLORS['text_secondary']).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(ohlc_row, text=f"H: ₦{high:,.2f}", font=get_font('small'),
                  foreground=COLORS['gain']).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(ohlc_row, text=f"L: ₦{low:,.2f}", font=get_font('small'),
                  foreground=COLORS['loss']).pack(side=tk.LEFT)
    
    def _create_technical_section(self, parent):
        """Create technical indicators section."""
        self.tech_labels = {}
        
        indicators = [
            ('rsi', 'RSI (14)', '--'),
            ('macd', 'MACD', '--'),
            ('sma_20', 'SMA (20)', '--'),
            ('sma_50', 'SMA (50)', '--'),
            ('ema_20', 'EMA (20)', '--'),
        ]
        
        for i, (key, label, default) in enumerate(indicators):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            
            ttk.Label(row, text=label, font=get_font('body')).pack(side=tk.LEFT)
            
            value_label = ttk.Label(
                row,
                text=default,
                font=get_font('mono'),
                foreground=COLORS['text_secondary']
            )
            value_label.pack(side=tk.RIGHT)
            self.tech_labels[key] = value_label
        
        # Loading indicator
        self.loading_label = ttk.Label(
            parent,
            text="⏳ Loading...",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.loading_label.pack(pady=(10, 0))
    
    def _create_rating_section(self, parent):
        """Create analysis rating section."""
        self.rating_labels = {}
        
        # Overall rating
        rating_row = ttk.Frame(parent)
        rating_row.pack(fill=tk.X)
        
        ttk.Label(rating_row, text="Overall:", font=get_font('body')).pack(side=tk.LEFT)
        
        self.rating_labels['overall'] = ttk.Label(
            rating_row,
            text="--",
            font=get_font('body_bold'),
            foreground=COLORS['text_secondary']
        )
        self.rating_labels['overall'].pack(side=tk.RIGHT)
        
        # Buy/Sell/Neutral counts
        counts_row = ttk.Frame(parent)
        counts_row.pack(fill=tk.X, pady=(10, 0))
        
        self.rating_labels['buy'] = ttk.Label(
            counts_row, text="Buy: --", font=get_font('small'),
            foreground=COLORS['gain']
        )
        self.rating_labels['buy'].pack(side=tk.LEFT, padx=(0, 20))
        
        self.rating_labels['neutral'] = ttk.Label(
            counts_row, text="Neutral: --", font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.rating_labels['neutral'].pack(side=tk.LEFT, padx=(0, 20))
        
        self.rating_labels['sell'] = ttk.Label(
            counts_row, text="Sell: --", font=get_font('small'),
            foreground=COLORS['loss']
        )
        self.rating_labels['sell'].pack(side=tk.LEFT)
    
    def _create_volume_section(self, parent):
        """Create volume analysis section."""
        volume = self.stock_data.get('volume', 0)
        
        # Format volume
        if volume >= 1_000_000:
            vol_text = f"{volume/1_000_000:.2f}M"
        elif volume >= 1_000:
            vol_text = f"{volume/1_000:.2f}K"
        else:
            vol_text = f"{volume:,.0f}"
        
        # Volume row
        vol_row = ttk.Frame(parent)
        vol_row.pack(fill=tk.X)
        
        ttk.Label(vol_row, text="Volume:", font=get_font('body')).pack(side=tk.LEFT)
        ttk.Label(
            vol_row,
            text=vol_text,
            font=get_font('mono'),
            foreground=COLORS['text_primary']
        ).pack(side=tk.RIGHT)
        
        # RVOL row
        rvol_row = ttk.Frame(parent)
        rvol_row.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(rvol_row, text="Relative Volume:", font=get_font('body')).pack(side=tk.LEFT)
        
        self.rvol_label = ttk.Label(
            rvol_row,
            text="--",
            font=get_font('mono'),
            foreground=COLORS['text_secondary']
        )
        self.rvol_label.pack(side=tk.RIGHT)
    
    def _load_technical_analysis(self):
        """Load technical analysis in background."""
        def fetch():
            try:
                ta_data = self.collector.get_technical_analysis(self.symbol)
                self.dialog.after(0, lambda: self._update_technical(ta_data))
            except Exception as e:
                logger.error(f"Error loading TA for {self.symbol}: {e}")
                self.dialog.after(0, lambda: self._show_ta_error(str(e)))
        
        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
    
    def _update_technical(self, ta_data: Optional[Dict[str, Any]]):
        """Update UI with technical analysis data."""
        self.loading_label.config(text="")
        
        if not ta_data:
            self.loading_label.config(text="❌ TA data unavailable")
            return
        
        indicators = ta_data.get('indicators', {})
        
        # Update indicator labels
        if 'rsi' in self.tech_labels:
            rsi = indicators.get('RSI')
            if rsi is not None:
                color = COLORS['loss'] if rsi > 70 else COLORS['gain'] if rsi < 30 else COLORS['text_primary']
                self.tech_labels['rsi'].config(text=f"{rsi:.1f}", foreground=color)
        
        if 'macd' in self.tech_labels:
            macd = indicators.get('MACD.macd')
            if macd is not None:
                color = COLORS['gain'] if macd > 0 else COLORS['loss']
                self.tech_labels['macd'].config(text=f"{macd:.2f}", foreground=color)
        
        if 'sma_20' in self.tech_labels:
            sma20 = indicators.get('SMA20')
            if sma20 is not None:
                self.tech_labels['sma_20'].config(text=f"₦{sma20:,.2f}")
        
        if 'sma_50' in self.tech_labels:
            sma50 = indicators.get('SMA50')
            if sma50 is not None:
                self.tech_labels['sma_50'].config(text=f"₦{sma50:,.2f}")
        
        if 'ema_20' in self.tech_labels:
            ema20 = indicators.get('EMA20')
            if ema20 is not None:
                self.tech_labels['ema_20'].config(text=f"₦{ema20:,.2f}")
        
        # Update ratings
        recommendation = ta_data.get('recommendation', 'NEUTRAL')
        summary = ta_data.get('summary', {})
        
        rating_colors = {
            'STRONG_BUY': COLORS['gain'],
            'BUY': COLORS['gain'],
            'NEUTRAL': COLORS['text_muted'],
            'SELL': COLORS['loss'],
            'STRONG_SELL': COLORS['loss'],
        }
        
        self.rating_labels['overall'].config(
            text=recommendation.replace('_', ' '),
            foreground=rating_colors.get(recommendation, COLORS['text_secondary'])
        )
        
        self.rating_labels['buy'].config(
            text=f"Buy: {summary.get('BUY', 0) + summary.get('STRONG_BUY', 0)}"
        )
        self.rating_labels['neutral'].config(
            text=f"Neutral: {summary.get('NEUTRAL', 0)}"
        )
        self.rating_labels['sell'].config(
            text=f"Sell: {summary.get('SELL', 0) + summary.get('STRONG_SELL', 0)}"
        )
    
    def _show_ta_error(self, error: str):
        """Show TA loading error."""
        self.loading_label.config(text=f"❌ {error}", foreground=COLORS['loss'])


def show_stock_detail(parent, symbol: str, stock_data: Dict[str, Any], db=None):
    """
    Show stock detail dialog.
    
    Args:
        parent: Parent window
        symbol: Stock symbol
        stock_data: Current stock data
        db: Optional database manager
    """
    dialog = StockDetailDialog(parent, symbol, stock_data, db)
    parent.wait_window(dialog.dialog)
