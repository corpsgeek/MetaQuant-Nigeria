"""
PCA Factor Analysis Tab - Visualize factor analysis and market regimes.

Provides:
- Factor returns chart
- Variance explained breakdown  
- Market regime indicator
- Stock-factor exposure heatmap
- Individual stock factor analysis
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Colors
COLORS = {
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'text': '#e8e8e8',
    'gain': '#00d26a',
    'loss': '#ff6b6b',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'accent': '#7c3aed'
}


class PCAAnalysisTab:
    """PCA Factor Analysis tab for MetaQuant."""
    
    def __init__(self, parent, db=None, ml_engine=None, price_provider=None):
        """
        Initialize PCA Analysis tab.
        
        Args:
            parent: Parent widget
            db: DatabaseManager
            ml_engine: ML Engine with PCA
            price_provider: Callable that returns price data dict
        """
        self.frame = ttk.Frame(parent)
        self.db = db
        self.ml_engine = ml_engine
        self.price_provider = price_provider
        
        self._price_data = {}
        self._selected_symbol = None
        
        self._create_ui()
        
        logger.info("PCA Analysis tab initialized")
    
    def _create_ui(self):
        """Create the main UI layout."""
        # Main container
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top controls
        self._create_controls(main_frame)
        
        # Content area - two columns
        content = ttk.Frame(main_frame)
        content.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel - Charts and regime
        left_panel = ttk.Frame(content)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self._create_regime_panel(left_panel)
        self._create_variance_panel(left_panel)
        self._create_factor_returns_panel(left_panel)
        
        # Right panel - Exposure heatmap and stock analysis
        right_panel = ttk.Frame(content)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self._create_exposure_panel(right_panel)
        self._create_stock_analysis_panel(right_panel)
    
    def _create_controls(self, parent):
        """Create control buttons."""
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X)
        
        ttk.Label(ctrl_frame, text="🔬 PCA Factor Analysis", 
                  font=('Helvetica', 14, 'bold')).pack(side=tk.LEFT)
        
        # Buttons
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="📥 Load & Fit PCA", 
                   command=self._fit_pca).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🔄 Refresh", 
                   command=self._refresh_display).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = ttk.Label(ctrl_frame, text="Ready", 
                                       foreground=COLORS['info'])
        self.status_label.pack(side=tk.LEFT, padx=20)
    
    def _create_regime_panel(self, parent):
        """Create market regime indicator panel."""
        regime_frame = ttk.LabelFrame(parent, text="📊 Market Regime")
        regime_frame.pack(fill=tk.X, pady=(0, 5))
        
        inner = ttk.Frame(regime_frame)
        inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Regime indicator
        ttk.Label(inner, text="Current Regime:", 
                  font=('Helvetica', 10)).pack(side=tk.LEFT)
        
        self.regime_label = ttk.Label(inner, text="Unknown", 
                                       font=('Helvetica', 14, 'bold'))
        self.regime_label.pack(side=tk.LEFT, padx=10)
        
        # Confidence
        ttk.Label(inner, text="Confidence:", 
                  font=('Helvetica', 10)).pack(side=tk.LEFT, padx=(20, 0))
        
        self.confidence_label = ttk.Label(inner, text="-", 
                                           font=('Helvetica', 10, 'bold'))
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        # Factor signals frame
        signals_frame = ttk.Frame(regime_frame)
        signals_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.factor_signal_labels = {}
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        for i, factor in enumerate(factors):
            lbl = ttk.Label(signals_frame, text=f"{factor}: -", 
                           font=('Helvetica', 9))
            lbl.grid(row=0, column=i, padx=10)
            self.factor_signal_labels[factor] = lbl
    
    def _create_variance_panel(self, parent):
        """Create variance explained panel."""
        var_frame = ttk.LabelFrame(parent, text="📈 Variance Explained")
        var_frame.pack(fill=tk.X, pady=5)
        
        inner = ttk.Frame(var_frame)
        inner.pack(fill=tk.X, padx=10, pady=10)
        
        self.variance_bars = {}
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        for i, factor in enumerate(factors):
            row_frame = ttk.Frame(inner)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=f"{factor}:", width=12,
                     font=('Helvetica', 9)).pack(side=tk.LEFT)
            
            # Progress bar for variance
            bar = ttk.Progressbar(row_frame, length=200, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            
            pct_lbl = ttk.Label(row_frame, text="0.0%", width=8,
                               font=('Helvetica', 9, 'bold'))
            pct_lbl.pack(side=tk.LEFT)
            
            self.variance_bars[factor] = (bar, pct_lbl)
    
    def _create_factor_returns_panel(self, parent):
        """Create factor returns display."""
        ret_frame = ttk.LabelFrame(parent, text="📉 Factor Returns (20-day)")
        ret_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        inner = ttk.Frame(ret_frame)
        inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.factor_return_labels = {}
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        for i, factor in enumerate(factors):
            frame = ttk.Frame(inner)
            frame.pack(fill=tk.X, pady=3)
            
            ttk.Label(frame, text=f"{factor}:", width=12,
                     font=('Helvetica', 10)).pack(side=tk.LEFT)
            
            ret_lbl = ttk.Label(frame, text="0.00%", width=10,
                               font=('Helvetica', 12, 'bold'))
            ret_lbl.pack(side=tk.LEFT, padx=10)
            
            self.factor_return_labels[factor] = ret_lbl
    
    def _create_exposure_panel(self, parent):
        """Create stock-factor exposure panel."""
        exp_frame = ttk.LabelFrame(parent, text="🎯 Top Factor Exposures")
        exp_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Treeview for exposures
        columns = ('Symbol', 'Market', 'Size', 'Value', 'Momentum', 'Vol')
        self.exposure_tree = ttk.Treeview(exp_frame, columns=columns, 
                                           show='headings', height=10)
        
        col_widths = {'Symbol': 70, 'Market': 60, 'Size': 60, 
                      'Value': 60, 'Momentum': 70, 'Vol': 60}
        
        for col in columns:
            self.exposure_tree.heading(col, text=col)
            self.exposure_tree.column(col, width=col_widths.get(col, 60))
        
        scrollbar = ttk.Scrollbar(exp_frame, orient=tk.VERTICAL,
                                   command=self.exposure_tree.yview)
        self.exposure_tree.configure(yscrollcommand=scrollbar.set)
        
        self.exposure_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection
        self.exposure_tree.bind('<<TreeviewSelect>>', self._on_stock_select)
        
        # Color tags
        self.exposure_tree.tag_configure('positive', foreground=COLORS['gain'])
        self.exposure_tree.tag_configure('negative', foreground=COLORS['loss'])
    
    def _create_stock_analysis_panel(self, parent):
        """Create individual stock analysis panel."""
        stock_frame = ttk.LabelFrame(parent, text="🔍 Stock Factor Analysis")
        stock_frame.pack(fill=tk.X, pady=5)
        
        inner = ttk.Frame(stock_frame)
        inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Symbol selector
        sel_frame = ttk.Frame(inner)
        sel_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(sel_frame, text="Symbol:").pack(side=tk.LEFT)
        
        self.symbol_var = tk.StringVar()
        self.symbol_combo = ttk.Combobox(sel_frame, textvariable=self.symbol_var,
                                          width=15)
        self.symbol_combo.pack(side=tk.LEFT, padx=5)
        self.symbol_combo.bind('<<ComboboxSelected>>', self._on_symbol_change)
        
        ttk.Button(sel_frame, text="Analyze",
                   command=self._analyze_stock).pack(side=tk.LEFT, padx=5)
        
        # Factor exposure details
        self.stock_exposure_labels = {}
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        for i, factor in enumerate(factors):
            row_frame = ttk.Frame(inner)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=f"{factor}:", width=12,
                     font=('Helvetica', 9)).pack(side=tk.LEFT)
            
            exp_lbl = ttk.Label(row_frame, text="-", width=8,
                               font=('Helvetica', 9, 'bold'))
            exp_lbl.pack(side=tk.LEFT, padx=5)
            
            # Interpretation
            interp_lbl = ttk.Label(row_frame, text="", 
                                   font=('Helvetica', 8), foreground='gray')
            interp_lbl.pack(side=tk.LEFT, padx=5)
            
            self.stock_exposure_labels[factor] = (exp_lbl, interp_lbl)
        
        # Factor alignment score
        align_frame = ttk.Frame(inner)
        align_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(align_frame, text="Factor Alignment:", 
                  font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        
        self.alignment_label = ttk.Label(align_frame, text="-",
                                          font=('Helvetica', 11, 'bold'))
        self.alignment_label.pack(side=tk.LEFT, padx=10)
    
    # ==================== Actions ====================
    
    def _fit_pca(self):
        """Load price data and fit PCA."""
        self.status_label.config(text="Loading price data...", 
                                  foreground=COLORS['warning'])
        self.frame.update()
        
        try:
            # Load price data
            if self.price_provider:
                self._price_data = self.price_provider()
            elif self.db:
                self._price_data = self._load_price_data()
            
            if not self._price_data:
                self.status_label.config(text="No price data available",
                                          foreground=COLORS['loss'])
                return
            
            self.status_label.config(text="Fitting PCA...", 
                                      foreground=COLORS['warning'])
            self.frame.update()
            
            # Fit PCA
            if self.ml_engine and hasattr(self.ml_engine, 'fit_pca'):
                self.ml_engine.fit_pca(self._price_data)
            
            # Update symbol list
            symbols = sorted(self._price_data.keys())
            self.symbol_combo['values'] = symbols
            if symbols:
                self.symbol_combo.current(0)
            
            self._refresh_display()
            
            self.status_label.config(
                text=f"PCA fitted on {len(self._price_data)} stocks",
                foreground=COLORS['gain']
            )
            
        except Exception as e:
            logger.error(f"PCA fit failed: {e}")
            self.status_label.config(text=f"Error: {e}", 
                                      foreground=COLORS['loss'])
    
    def _refresh_display(self):
        """Refresh all displays with current PCA data."""
        if not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        pca = self.ml_engine.pca_engine
        if not pca._is_fitted:
            return
        
        # Update regime
        regime_info = pca.get_market_regime()
        regime = regime_info.get('regime', 'Unknown')
        
        regime_colors = {
            'Risk-On': COLORS['gain'],
            'Risk-Off': COLORS['loss'],
            'Rotation': COLORS['warning']
        }
        
        self.regime_label.config(
            text=f"{regime} {'🟢' if regime == 'Risk-On' else '🔴' if regime == 'Risk-Off' else '⚡'}",
            foreground=regime_colors.get(regime, 'white')
        )
        
        self.confidence_label.config(
            text=f"{regime_info.get('confidence', 0):.0%}"
        )
        
        # Update factor signals
        factor_signals = regime_info.get('factor_signals', {})
        for factor, lbl in self.factor_signal_labels.items():
            signal = factor_signals.get(factor, 'Neutral')
            color = COLORS['gain'] if signal == 'Bullish' else COLORS['loss'] if signal == 'Bearish' else 'gray'
            lbl.config(text=f"{factor}: {signal}", foreground=color)
        
        # Update variance explained
        variance = pca.get_variance_explained()
        for factor, (bar, lbl) in self.variance_bars.items():
            pct = variance.get(factor, 0) * 100
            bar['value'] = pct
            lbl.config(text=f"{pct:.1f}%")
        
        # Update factor returns
        factor_returns = pca.get_factor_returns()
        if not factor_returns.empty and len(factor_returns) >= 20:
            recent = factor_returns.tail(20).mean() * 100
            for factor, lbl in self.factor_return_labels.items():
                ret = recent.get(factor, 0)
                color = COLORS['gain'] if ret > 0 else COLORS['loss']
                lbl.config(text=f"{ret:+.2f}%", foreground=color)
        
        # Update exposure table
        self._update_exposure_table()
    
    def _update_exposure_table(self):
        """Update the stock-factor exposure table."""
        # Clear existing
        for item in self.exposure_tree.get_children():
            self.exposure_tree.delete(item)
        
        if not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        pca = self.ml_engine.pca_engine
        if not pca._is_fitted:
            return
        
        exposures_df = pca.get_all_exposures()
        if exposures_df.empty:
            return
        
        # Sort by absolute market exposure
        exposures_df['abs_market'] = exposures_df['Market'].abs()
        exposures_df = exposures_df.sort_values('abs_market', ascending=False).head(30)
        
        for symbol in exposures_df.index:
            row = exposures_df.loc[symbol]
            values = (
                symbol,
                f"{row.get('Market', 0):+.3f}",
                f"{row.get('Size', 0):+.3f}",
                f"{row.get('Value', 0):+.3f}",
                f"{row.get('Momentum', 0):+.3f}",
                f"{row.get('Volatility', 0):+.3f}"
            )
            
            # Tag based on market exposure
            tag = 'positive' if row.get('Market', 0) > 0 else 'negative'
            self.exposure_tree.insert('', 'end', values=values, tags=(tag,))
    
    def _on_stock_select(self, event):
        """Handle stock selection in exposure table."""
        selection = self.exposure_tree.selection()
        if selection:
            item = self.exposure_tree.item(selection[0])
            symbol = item['values'][0]
            self.symbol_var.set(symbol)
            self._analyze_stock()
    
    def _on_symbol_change(self, event):
        """Handle symbol combo change."""
        self._analyze_stock()
    
    def _analyze_stock(self):
        """Analyze selected stock's factor exposure."""
        symbol = self.symbol_var.get()
        if not symbol:
            return
        
        if not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        pca = self.ml_engine.pca_engine
        if not pca._is_fitted:
            return
        
        exposures = pca.get_factor_exposures(symbol)
        
        interpretations = {
            'Market': lambda v: "High beta" if v > 0.3 else "Low beta" if v < -0.3 else "Neutral",
            'Size': lambda v: "Large-cap" if v > 0.3 else "Small-cap" if v < -0.3 else "Mid-cap",
            'Value': lambda v: "Value" if v > 0.3 else "Growth" if v < -0.3 else "Blend",
            'Momentum': lambda v: "High momentum" if v > 0.3 else "Mean-reverting" if v < -0.3 else "Moderate",
            'Volatility': lambda v: "High vol" if v > 0.3 else "Low vol" if v < -0.3 else "Normal vol"
        }
        
        for factor, (exp_lbl, interp_lbl) in self.stock_exposure_labels.items():
            value = exposures.get(factor, 0)
            color = COLORS['gain'] if value > 0 else COLORS['loss'] if value < 0 else 'gray'
            exp_lbl.config(text=f"{value:+.3f}", foreground=color)
            
            interp = interpretations.get(factor, lambda v: "")(value)
            interp_lbl.config(text=interp)
        
        # Factor alignment
        alignment = pca.calculate_factor_alignment(symbol)
        color = COLORS['gain'] if alignment > 0 else COLORS['loss']
        self.alignment_label.config(text=f"{alignment:+.2f}", foreground=color)
    
    def _load_price_data(self):
        """Load price data from database."""
        import pandas as pd
        
        try:
            stocks = self.db.conn.execute("""
                SELECT DISTINCT s.symbol, s.id 
                FROM stocks s 
                JOIN daily_prices dp ON s.id = dp.stock_id
            """).fetchall()
            
            price_data = {}
            
            for symbol, stock_id in stocks:
                prices = self.db.conn.execute("""
                    SELECT date, open, high, low, close, volume
                    FROM daily_prices
                    WHERE stock_id = ?
                    ORDER BY date
                """, [stock_id]).fetchall()
                
                if prices:
                    df = pd.DataFrame(prices, 
                        columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    price_data[symbol] = df
            
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return {}
