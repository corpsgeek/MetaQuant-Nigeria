"""
Deep Stock Analysis Modal for MetaQuant Nigeria.
A popup modal showing comprehensive stock analysis when clicking on any stock.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Optional, Any
from datetime import datetime

from src.database.db_manager import DatabaseManager
from src.gui.theme import COLORS, get_font

logger = logging.getLogger(__name__)


class StockAnalysisModal:
    """
    Modal dialog showing comprehensive stock analysis.
    Can be triggered from any tab/table by calling show(symbol).
    """
    
    def __init__(self, parent, db: DatabaseManager, ml_engine=None):
        self.parent = parent
        self.db = db
        self.ml_engine = ml_engine
        self.dialog = None
    
    def show(self, symbol: str):
        """Show the analysis modal for a given stock symbol."""
        if self.dialog and self.dialog.winfo_exists():
            self.dialog.destroy()
        
        # Create modal
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(f"📊 Stock Analysis: {symbol}")
        self.dialog.geometry("800x700")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Get stock data
        stock_data = self._get_stock_data(symbol)
        if not stock_data:
            ttk.Label(self.dialog, text=f"Stock {symbol} not found",
                     font=get_font('heading')).pack(pady=50)
            return
        
        # Create scrollable content
        canvas = tk.Canvas(self.dialog, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.dialog, orient=tk.VERTICAL, command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def on_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_configure)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        content = scrollable
        
        # === HEADER ===
        self._create_header(content, stock_data, symbol)
        
        # === PRICE INFO ===
        self._create_price_section(content, stock_data)
        
        # === TECHNICAL INDICATORS ===
        self._create_technicals_section(content, symbol)
        
        # === ML PREDICTION ===
        self._create_ml_section(content, symbol)
        
        # === FUNDAMENTALS ===
        self._create_fundamentals_section(content, stock_data)
        
        # === ACTIONS ===
        self._create_actions(content, symbol)
    
    def _get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get stock data from database."""
        try:
            result = self.db.conn.execute("""
                SELECT s.symbol, s.name, s.sector, s.last_price, s.change_percent,
                       s.volume, s.market_cap, s.pe_ratio, s.eps, s.dividend_yield,
                       s.week_52_high, s.week_52_low
                FROM stocks s
                WHERE s.symbol = ?
            """, [symbol]).fetchone()
            
            if result:
                return {
                    'symbol': result[0],
                    'name': result[1],
                    'sector': result[2],
                    'price': float(result[3]) if result[3] else 0,
                    'change_percent': float(result[4]) if result[4] else 0,
                    'volume': int(result[5]) if result[5] else 0,
                    'market_cap': float(result[6]) if result[6] else 0,
                    'pe_ratio': float(result[7]) if result[7] else None,
                    'eps': float(result[8]) if result[8] else None,
                    'dividend_yield': float(result[9]) if result[9] else None,
                    'week_52_high': float(result[10]) if result[10] else None,
                    'week_52_low': float(result[11]) if result[11] else None,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get stock data: {e}")
            return None
    
    def _create_header(self, parent, data: Dict, symbol: str):
        """Create header section with stock name and key info."""
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, padx=20, pady=15)
        
        # Left: Symbol and name
        left = ttk.Frame(header)
        left.pack(side=tk.LEFT)
        
        ttk.Label(left, text=symbol, font=('Helvetica', 28, 'bold'),
                 foreground=COLORS['primary']).pack(anchor='w')
        ttk.Label(left, text=data.get('name', ''), font=get_font('body'),
                 foreground=COLORS['text_muted']).pack(anchor='w')
        ttk.Label(left, text=f"Sector: {data.get('sector', 'Unknown')}", 
                 font=get_font('small'), foreground=COLORS['text_muted']).pack(anchor='w')
        
        # Right: Price and change
        right = ttk.Frame(header)
        right.pack(side=tk.RIGHT)
        
        price = data.get('price', 0)
        change = data.get('change_percent', 0)
        color = COLORS['gain'] if change >= 0 else COLORS['loss']
        
        ttk.Label(right, text=f"₦{price:,.2f}", font=('Helvetica', 28, 'bold'),
                 foreground=color).pack(anchor='e')
        ttk.Label(right, text=f"{'+' if change >= 0 else ''}{change:.2f}%",
                 font=get_font('body'), foreground=color).pack(anchor='e')
        
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=5)
    
    def _create_price_section(self, parent, data: Dict):
        """Create price metrics section."""
        section = ttk.LabelFrame(parent, text="📈 Price Metrics")
        section.pack(fill=tk.X, padx=20, pady=10)
        
        content = ttk.Frame(section)
        content.pack(fill=tk.X, padx=15, pady=10)
        
        price = data.get('price', 0)
        high_52 = data.get('week_52_high')
        low_52 = data.get('week_52_low')
        volume = data.get('volume', 0)
        
        metrics = [
            ('Today\'s Volume', f"{volume:,}"),
            ('52W High', f"₦{high_52:,.2f}" if high_52 else 'N/A'),
            ('52W Low', f"₦{low_52:,.2f}" if low_52 else 'N/A'),
            ('52W Range %', f"{((price - low_52) / (high_52 - low_52) * 100):.0f}%" if high_52 and low_52 and high_52 != low_52 else 'N/A'),
        ]
        
        for i, (label, value) in enumerate(metrics):
            col = ttk.Frame(content)
            col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            
            ttk.Label(col, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            ttk.Label(col, text=value, font=('Helvetica', 14, 'bold'),
                     foreground=COLORS['text_primary']).pack(anchor='center')
    
    def _create_technicals_section(self, parent, symbol: str):
        """Create technical indicators section."""
        section = ttk.LabelFrame(parent, text="📊 Technical Indicators")
        section.pack(fill=tk.X, padx=20, pady=10)
        
        content = ttk.Frame(section)
        content.pack(fill=tk.X, padx=15, pady=10)
        
        # Get price data for indicators
        indicators = self._calculate_indicators(symbol)
        
        # Row 1: Trend indicators
        row1 = ttk.Frame(content)
        row1.pack(fill=tk.X, pady=5)
        
        for key, label, value, signal in [
            ('rsi', 'RSI (14)', indicators.get('rsi', '--'), indicators.get('rsi_signal', '')),
            ('macd', 'MACD', indicators.get('macd', '--'), indicators.get('macd_signal', '')),
            ('sma_20', 'SMA 20', indicators.get('sma_20', '--'), ''),
            ('sma_50', 'SMA 50', indicators.get('sma_50', '--'), ''),
        ]:
            col = ttk.Frame(row1)
            col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            ttk.Label(col, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            
            signal_color = COLORS['gain'] if 'Bullish' in signal or 'Oversold' in signal else \
                          COLORS['loss'] if 'Bearish' in signal or 'Overbought' in signal else \
                          COLORS['text_primary']
            
            ttk.Label(col, text=str(value), font=('Helvetica', 12, 'bold'),
                     foreground=signal_color).pack(anchor='center')
            if signal:
                ttk.Label(col, text=signal, font=get_font('small'),
                         foreground=signal_color).pack(anchor='center')
        
        # Row 2: Support/Resistance
        row2 = ttk.Frame(content)
        row2.pack(fill=tk.X, pady=10)
        
        for key, label, value in [
            ('support', '📉 Support', indicators.get('support', '--')),
            ('resistance', '📈 Resistance', indicators.get('resistance', '--')),
            ('trend', '📊 Trend', indicators.get('trend', '--')),
            ('volatility', '📈 Volatility', indicators.get('volatility', '--')),
        ]:
            col = ttk.Frame(row2)
            col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            ttk.Label(col, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            ttk.Label(col, text=str(value), font=('Helvetica', 12, 'bold'),
                     foreground=COLORS['text_primary']).pack(anchor='center')
    
    def _calculate_indicators(self, symbol: str) -> Dict:
        """Calculate technical indicators for the stock."""
        indicators = {}
        
        try:
            # Get price history
            result = self.db.conn.execute("""
                SELECT close, high, low FROM price_history
                WHERE symbol = ? ORDER BY date DESC LIMIT 50
            """, [symbol]).fetchall()
            
            if len(result) < 14:
                return {'rsi': 'N/A', 'macd': 'N/A', 'sma_20': 'N/A', 'sma_50': 'N/A'}
            
            closes = [float(r[0]) for r in result]
            highs = [float(r[1]) for r in result]
            lows = [float(r[2]) for r in result]
            
            # RSI
            gains = []
            losses = []
            for i in range(1, min(15, len(closes))):
                diff = closes[i-1] - closes[i]  # reversed order
                if diff > 0:
                    gains.append(diff)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(diff))
            
            avg_gain = sum(gains) / 14 if gains else 0
            avg_loss = sum(losses) / 14 if losses else 0.0001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            indicators['rsi'] = f"{rsi:.1f}"
            if rsi > 70:
                indicators['rsi_signal'] = '⚠️ Overbought'
            elif rsi < 30:
                indicators['rsi_signal'] = '✅ Oversold'
            else:
                indicators['rsi_signal'] = 'Neutral'
            
            # SMAs
            if len(closes) >= 20:
                sma_20 = sum(closes[:20]) / 20
                indicators['sma_20'] = f"₦{sma_20:,.2f}"
            
            if len(closes) >= 50:
                sma_50 = sum(closes[:50]) / 50
                indicators['sma_50'] = f"₦{sma_50:,.2f}"
            else:
                indicators['sma_50'] = 'N/A'
            
            # MACD (simplified)
            if len(closes) >= 26:
                ema_12 = sum(closes[:12]) / 12
                ema_26 = sum(closes[:26]) / 26
                macd = ema_12 - ema_26
                indicators['macd'] = f"{macd:+.2f}"
                indicators['macd_signal'] = '📈 Bullish' if macd > 0 else '📉 Bearish'
            
            # Support/Resistance
            indicators['support'] = f"₦{min(lows[:20]):,.2f}"
            indicators['resistance'] = f"₦{max(highs[:20]):,.2f}"
            
            # Trend
            if closes[0] > closes[-1]:
                pct = ((closes[0] - closes[-1]) / closes[-1]) * 100
                indicators['trend'] = f"📈 +{pct:.1f}%"
            else:
                pct = ((closes[-1] - closes[0]) / closes[-1]) * 100
                indicators['trend'] = f"📉 -{pct:.1f}%"
            
            # Volatility
            import statistics
            if len(closes) >= 20:
                returns = [(closes[i] - closes[i+1]) / closes[i+1] for i in range(min(19, len(closes)-1))]
                vol = statistics.stdev(returns) * 100 if len(returns) > 1 else 0
                indicators['volatility'] = f"{vol:.1f}%"
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            indicators = {'rsi': 'N/A', 'macd': 'N/A', 'sma_20': 'N/A', 'sma_50': 'N/A'}
        
        return indicators
    
    def _create_ml_section(self, parent, symbol: str):
        """Create ML prediction section."""
        section = ttk.LabelFrame(parent, text="🤖 ML Prediction")
        section.pack(fill=tk.X, padx=20, pady=10)
        
        content = ttk.Frame(section)
        content.pack(fill=tk.X, padx=15, pady=10)
        
        prediction = self._get_ml_prediction(symbol)
        
        # Direction indicator
        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, padx=20)
        
        direction = prediction.get('direction', 'HOLD')
        direction_color = COLORS['gain'] if direction == 'UP' else \
                         COLORS['loss'] if direction == 'DOWN' else COLORS['warning']
        direction_icon = '📈' if direction == 'UP' else '📉' if direction == 'DOWN' else '➡️'
        
        ttk.Label(left, text=direction_icon, font=('Helvetica', 36)).pack()
        ttk.Label(left, text=direction, font=('Helvetica', 18, 'bold'),
                 foreground=direction_color).pack()
        
        # Details
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        details = [
            ('Predicted Change', f"{prediction.get('predicted_change', 0):+.2f}%"),
            ('Confidence', f"{prediction.get('confidence', 0):.0f}%"),
            ('Signal', prediction.get('signal', 'HOLD')),
        ]
        
        for label, value in details:
            row = ttk.Frame(right)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label + ":", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(side=tk.LEFT)
            ttk.Label(row, text=value, font=get_font('body'),
                     foreground=COLORS['text_primary']).pack(side=tk.RIGHT)
        
        # Feature importance if available
        importance = prediction.get('feature_importance', {})
        if importance:
            ttk.Separator(section, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=5)
            ttk.Label(section, text="Top Factors:", font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='w', padx=15)
            
            factors_frame = ttk.Frame(section)
            factors_frame.pack(fill=tk.X, padx=15, pady=5)
            
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, imp in sorted_imp:
                ttk.Label(factors_frame, text=f"• {feat}: {imp:.1f}%",
                         font=get_font('small')).pack(anchor='w')
    
    def _get_ml_prediction(self, symbol: str) -> Dict:
        """Get ML prediction for the stock."""
        if not self.ml_engine:
            return {'direction': 'HOLD', 'predicted_change': 0, 'confidence': 0, 'signal': 'N/A'}
        
        try:
            result = self.ml_engine.predict(symbol)
            if result and 'prediction' in result:
                pred = result['prediction']
                return {
                    'direction': pred.get('direction', 'HOLD'),
                    'predicted_change': pred.get('predicted_change', 0),
                    'confidence': pred.get('confidence', 0),
                    'signal': 'BUY' if pred.get('direction') == 'UP' else 
                             'SELL' if pred.get('direction') == 'DOWN' else 'HOLD',
                    'feature_importance': result.get('feature_importance', {})
                }
        except Exception as e:
            logger.error(f"Failed to get ML prediction: {e}")
        
        return {'direction': 'HOLD', 'predicted_change': 0, 'confidence': 0, 'signal': 'N/A'}
    
    def _create_fundamentals_section(self, parent, data: Dict):
        """Create fundamentals section."""
        section = ttk.LabelFrame(parent, text="💰 Fundamentals")
        section.pack(fill=tk.X, padx=20, pady=10)
        
        content = ttk.Frame(section)
        content.pack(fill=tk.X, padx=15, pady=10)
        
        pe = data.get('pe_ratio')
        eps = data.get('eps')
        div_yield = data.get('dividend_yield')
        market_cap = data.get('market_cap', 0)
        
        # Format market cap
        if market_cap >= 1e12:
            cap_str = f"₦{market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            cap_str = f"₦{market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            cap_str = f"₦{market_cap/1e6:.2f}M"
        else:
            cap_str = f"₦{market_cap:,.0f}"
        
        metrics = [
            ('Market Cap', cap_str),
            ('P/E Ratio', f"{pe:.2f}" if pe else 'N/A'),
            ('EPS', f"₦{eps:.2f}" if eps else 'N/A'),
            ('Dividend Yield', f"{div_yield:.2f}%" if div_yield else 'N/A'),
        ]
        
        for label, value in metrics:
            col = ttk.Frame(content)
            col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            
            ttk.Label(col, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='center')
            ttk.Label(col, text=value, font=('Helvetica', 14, 'bold'),
                     foreground=COLORS['text_primary']).pack(anchor='center')
    
    def _create_actions(self, parent, symbol: str):
        """Create action buttons."""
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Button(actions, text="➕ Add to Watchlist",
                  command=lambda: self._add_to_watchlist(symbol)).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="📝 Paper Trade",
                  command=lambda: self._paper_trade(symbol)).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="❌ Close",
                  command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _add_to_watchlist(self, symbol: str):
        """Add stock to default watchlist."""
        from tkinter import messagebox
        try:
            # Get default watchlist
            result = self.db.conn.execute(
                "SELECT id FROM watchlists ORDER BY id LIMIT 1"
            ).fetchone()
            
            if result:
                watchlist_id = result[0]
                stock = self.db.conn.execute(
                    "SELECT id FROM stocks WHERE symbol = ?", [symbol]
                ).fetchone()
                
                if stock:
                    self.db.conn.execute("""
                        INSERT INTO watchlist_items (id, watchlist_id, stock_id)
                        VALUES (nextval('seq_watchlist_items'), ?, ?)
                    """, [watchlist_id, stock[0]])
                    self.db.conn.commit()
                    messagebox.showinfo("Success", f"{symbol} added to watchlist!")
        except Exception as e:
            if 'UNIQUE' in str(e):
                messagebox.showinfo("Info", f"{symbol} is already in watchlist")
            else:
                messagebox.showerror("Error", f"Failed to add: {e}")
    
    def _paper_trade(self, symbol: str):
        """Placeholder for paper trade action."""
        from tkinter import messagebox
        messagebox.showinfo("Paper Trade", f"Navigate to Paper Trading tab to trade {symbol}")


# Global instance for easy access
_modal_instance = None

def show_stock_analysis(parent, db, symbol: str, ml_engine=None):
    """Convenience function to show stock analysis modal."""
    global _modal_instance
    if _modal_instance is None:
        _modal_instance = StockAnalysisModal(parent, db, ml_engine)
    _modal_instance.ml_engine = ml_engine
    _modal_instance.show(symbol)
