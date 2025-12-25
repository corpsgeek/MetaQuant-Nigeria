"""
PCA Factor Analysis Tab - Super Enhanced Version with Advanced Analytics.

Provides:
- Factor timing charts (matplotlib)
- Rolling regime timeline
- Factor-based stock screener
- AI factor recommendations
- Correlation matrix
- Interactive factor heatmap
- Stock factor analysis
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

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

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - charts will be disabled")


class PCAAnalysisTab:
    """Super Enhanced PCA Factor Analysis tab for MetaQuant."""
    
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
        self._canvas = None
        self._figure = None
        
        self._create_ui()
        
        logger.info("PCA Analysis tab initialized")
    
    def _create_ui(self):
        """Create the main UI layout with notebook for sections."""
        # Main container
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top controls
        self._create_controls(main_frame)
        
        # Create notebook for multiple tabs within PCA
        self.sub_notebook = ttk.Notebook(main_frame)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Tab 1: Overview (Regime + Variance + Factors)
        overview_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(overview_frame, text="📊 Overview")
        self._create_overview_tab(overview_frame)
        
        # Tab 2: Factor Charts
        charts_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(charts_frame, text="📈 Factor Charts")
        self._create_charts_tab(charts_frame)
        
        # Tab 3: Factor Screener
        screener_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(screener_frame, text="🔍 Screener")
        self._create_screener_tab(screener_frame)
        
        # Tab 4: AI Recommendations
        ai_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(ai_frame, text="🤖 AI Insights")
        self._create_ai_tab(ai_frame)
        
        # Tab 5: Stock Analysis
        stock_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(stock_frame, text="🔬 Stock Analysis")
        self._create_stock_tab(stock_frame)
    
    def _create_controls(self, parent):
        """Create control buttons."""
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(fill=tk.X)
        
        ttk.Label(ctrl_frame, text="🔮 PCA Factor Intelligence", 
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
    
    # ==================== OVERVIEW TAB ====================
    
    def _create_overview_tab(self, parent):
        """Create overview tab with regime and factors."""
        # Two columns
        left = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right = ttk.Frame(parent)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Left: Regime + Variance
        self._create_regime_panel(left)
        self._create_variance_panel(left)
        self._create_factor_returns_panel(left)
        
        # Right: Exposure heatmap
        self._create_exposure_panel(right)
    
    def _create_regime_panel(self, parent):
        """Create market regime indicator panel."""
        regime_frame = ttk.LabelFrame(parent, text="📊 Market Regime")
        regime_frame.pack(fill=tk.X, pady=(0, 5))
        
        inner = ttk.Frame(regime_frame)
        inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Regime indicator - large display
        regime_display = ttk.Frame(inner)
        regime_display.pack(fill=tk.X)
        
        ttk.Label(regime_display, text="Current:", 
                  font=('Helvetica', 10)).pack(side=tk.LEFT)
        
        self.regime_label = ttk.Label(regime_display, text="Unknown", 
                                       font=('Helvetica', 18, 'bold'))
        self.regime_label.pack(side=tk.LEFT, padx=10)
        
        self.confidence_label = ttk.Label(regime_display, text="", 
                                           font=('Helvetica', 10))
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        # Factor signals in grid
        signals_frame = ttk.Frame(regime_frame)
        signals_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        self.factor_signal_labels = {}
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        for i, factor in enumerate(factors):
            lbl = ttk.Label(signals_frame, text=f"{factor}: -", 
                           font=('Helvetica', 9))
            lbl.grid(row=0, column=i, padx=8)
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
            
            ttk.Label(row_frame, text=f"{factor}:", width=10,
                     font=('Helvetica', 9)).pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(row_frame, length=150, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            
            pct_lbl = ttk.Label(row_frame, text="0.0%", width=6,
                               font=('Helvetica', 9, 'bold'))
            pct_lbl.pack(side=tk.LEFT)
            
            self.variance_bars[factor] = (bar, pct_lbl)
    
    def _create_factor_returns_panel(self, parent):
        """Create factor returns display with multiple time periods."""
        ret_frame = ttk.LabelFrame(parent, text="📉 Factor Returns")
        ret_frame.pack(fill=tk.X, pady=5)
        
        inner = ttk.Frame(ret_frame)
        inner.pack(fill=tk.X, padx=10, pady=10)
        
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        # 1D, 1W, 1M, YTD, Annualized
        periods = [('1D', 1), ('1W', 5), ('1M', 20), ('YTD', -1), ('Ann.', 252)]
        
        # Header row
        ttk.Label(inner, text="Factor", width=8, font=('Helvetica', 8, 'bold')).grid(row=0, column=0)
        for i, (period_name, _) in enumerate(periods):
            ttk.Label(inner, text=period_name, width=6, font=('Helvetica', 8, 'bold')).grid(row=0, column=i+1)
        
        # Factor rows
        self.factor_return_labels = {}
        for row, factor in enumerate(factors, 1):
            ttk.Label(inner, text=factor, width=8, font=('Helvetica', 9)).grid(row=row, column=0, sticky='w')
            
            self.factor_return_labels[factor] = {}
            for col, (period_name, _) in enumerate(periods, 1):
                lbl = ttk.Label(inner, text="-", width=6, font=('Helvetica', 9, 'bold'))
                lbl.grid(row=row, column=col)
                self.factor_return_labels[factor][period_name] = lbl
    
    def _create_exposure_panel(self, parent):
        """Create stock-factor exposure panel."""
        exp_frame = ttk.LabelFrame(parent, text="🎯 Factor Exposure Heatmap")
        exp_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sort controls
        sort_frame = ttk.Frame(exp_frame)
        sort_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sort_frame, text="Sort by:").pack(side=tk.LEFT)
        self.sort_var = tk.StringVar(value='Market')
        sort_combo = ttk.Combobox(sort_frame, textvariable=self.sort_var,
                                   values=['Market', 'Size', 'Value', 'Momentum', 'Volatility'],
                                   width=10, state='readonly')
        sort_combo.pack(side=tk.LEFT, padx=5)
        sort_combo.bind('<<ComboboxSelected>>', lambda e: self._update_exposure_table())
        
        # Treeview
        columns = ('Symbol', 'Market', 'Size', 'Value', 'Mom', 'Vol')
        self.exposure_tree = ttk.Treeview(exp_frame, columns=columns, 
                                           show='headings', height=12)
        
        col_widths = {'Symbol': 65, 'Market': 55, 'Size': 55, 
                      'Value': 55, 'Mom': 55, 'Vol': 55}
        
        for col in columns:
            self.exposure_tree.heading(col, text=col,
                command=lambda c=col: self._sort_exposure_by(c))
            self.exposure_tree.column(col, width=col_widths.get(col, 55))
        
        scrollbar = ttk.Scrollbar(exp_frame, orient=tk.VERTICAL,
                                   command=self.exposure_tree.yview)
        self.exposure_tree.configure(yscrollcommand=scrollbar.set)
        
        self.exposure_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection
        self.exposure_tree.bind('<<TreeviewSelect>>', self._on_stock_select)
        
        # Color tags
        self.exposure_tree.tag_configure('strong_pos', foreground='#00ff88')
        self.exposure_tree.tag_configure('weak_pos', foreground='#88ffaa')
        self.exposure_tree.tag_configure('neutral', foreground='#aaaaaa')
        self.exposure_tree.tag_configure('weak_neg', foreground='#ffaa88')
        self.exposure_tree.tag_configure('strong_neg', foreground='#ff6666')
    
    # ==================== CHARTS TAB ====================
    
    def _create_charts_tab(self, parent):
        """Create factor timing charts tab."""
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(parent, text="Matplotlib not available. Install with: pip install matplotlib",
                     font=('Helvetica', 12)).pack(pady=50)
            return
        
        # Chart controls
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ctrl, text="Period:").pack(side=tk.LEFT)
        self.chart_window_var = tk.StringVar(value='5')
        chart_options = [('1D', '1'), ('1W', '5'), ('1M', '20'), ('YTD', 'YTD')]
        for label, value in chart_options:
            ttk.Radiobutton(ctrl, text=label, variable=self.chart_window_var, 
                           value=value, command=self._update_chart).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ctrl, text="📊 Update Chart", 
                   command=self._update_chart).pack(side=tk.RIGHT)
        
        # Chart area
        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self._figure = Figure(figsize=(10, 5), dpi=100, facecolor='#1a1a2e')
        self._canvas = FigureCanvasTkAgg(self._figure, master=chart_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _update_chart(self):
        """Update the factor timing chart."""
        if not MATPLOTLIB_AVAILABLE or not self._figure:
            return
        
        if not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        pca = self.ml_engine.pca_engine
        if not pca._is_fitted:
            return
        
        factor_returns = pca.get_factor_returns()
        if factor_returns.empty:
            return
        
        # Determine window size
        window_val = self.chart_window_var.get()
        if window_val == 'YTD':
            # Calculate days from start of year
            from datetime import datetime
            today = datetime.now()
            start_of_year = datetime(today.year, 1, 1)
            window = (today - start_of_year).days
            window = min(window, len(factor_returns))  # Cap at available data
            period_label = 'YTD'
        else:
            window = int(window_val)
            period_label = f'{window}-day'
        
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor('#1a1a2e')
        
        # Plot cumulative returns for each factor
        colors = ['#00ff88', '#00bfff', '#ffdd00', '#ff6688', '#aa77ff']
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        recent = factor_returns.tail(window).copy()
        
        # INDEXED RETURNS: Each factor starts at 100, shows relative growth
        all_values = []
        for i, factor in enumerate(factors):
            if factor in recent.columns:
                # Calculate cumulative wealth: 100 * (1 + r1) * (1 + r2) * ...
                cumulative = 100 * (1 + recent[factor]).cumprod()
                all_values.extend(cumulative.values)
                
                ax.plot(range(len(cumulative)), cumulative, 
                       label=f"{factor}", color=colors[i], linewidth=2.5,
                       marker='o' if window <= 20 else None, markersize=4)
        
        ax.axhline(y=100, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Days', color='white', fontsize=10)
        ax.set_ylabel('Indexed Value (Base=100)', color='white', fontsize=10)
        ax.set_title(f'Factor Performance ({period_label})', color='white', fontsize=12, fontweight='bold')
        
        # Better legend
        legend = ax.legend(loc='upper left', facecolor='#2d2d4e', 
                           labelcolor='white', fontsize=9, framealpha=0.9)
        legend.get_frame().set_edgecolor('white')
        
        ax.tick_params(colors='white', labelsize=9)
        ax.grid(True, alpha=0.25, color='white', linestyle=':')
        
        # Auto-scale y-axis with padding to show all data
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            padding = (y_max - y_min) * 0.1 or 2  # 10% padding or at least 2 points
            ax.set_ylim(bottom=y_min - padding, top=y_max + padding)
        
        # Add spines styling
        for spine in ax.spines.values():
            spine.set_color('#555555')
        
        self._figure.tight_layout()
        self._canvas.draw()
    
    # ==================== SCREENER TAB ====================
    
    def _create_screener_tab(self, parent):
        """Create factor-based stock screener."""
        # Filter controls
        filter_frame = ttk.LabelFrame(parent, text="🔍 Factor Filters")
        filter_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.filter_vars = {}
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        grid = ttk.Frame(filter_frame)
        grid.pack(fill=tk.X, padx=10, pady=10)
        
        for i, factor in enumerate(factors):
            row = i // 3
            col = (i % 3) * 4
            
            ttk.Label(grid, text=factor + ":").grid(row=row, column=col, padx=5)
            
            op_var = tk.StringVar(value='>')
            op_combo = ttk.Combobox(grid, textvariable=op_var, 
                                    values=['>', '<', '='], width=3, state='readonly')
            op_combo.grid(row=row, column=col+1)
            
            val_var = tk.StringVar(value='0.0')
            val_entry = ttk.Entry(grid, textvariable=val_var, width=6)
            val_entry.grid(row=row, column=col+2, padx=2)
            
            enabled_var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(grid, variable=enabled_var)
            chk.grid(row=row, column=col+3)
            
            self.filter_vars[factor] = (enabled_var, op_var, val_var)
        
        # Buttons
        btn_frame = ttk.Frame(filter_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(btn_frame, text="🔍 Screen Stocks", 
                   command=self._run_screener).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🔄 Clear Filters", 
                   command=self._clear_filters).pack(side=tk.LEFT, padx=5)
        
        self.screener_status = ttk.Label(btn_frame, text="")
        self.screener_status.pack(side=tk.RIGHT, padx=10)
        
        # Quick presets
        preset_frame = ttk.Frame(filter_frame)
        preset_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(preset_frame, text="Quick Presets:").pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="High Momentum", 
                   command=lambda: self._apply_preset('momentum')).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="Low Volatility", 
                   command=lambda: self._apply_preset('low_vol')).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="Value Stocks", 
                   command=lambda: self._apply_preset('value')).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="High Beta", 
                   command=lambda: self._apply_preset('high_beta')).pack(side=tk.LEFT, padx=3)
        
        # Results
        results_frame = ttk.LabelFrame(parent, text="📋 Screener Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ('Symbol', 'Market', 'Size', 'Value', 'Mom', 'Vol', 'Align')
        self.screener_tree = ttk.Treeview(results_frame, columns=columns, 
                                           show='headings', height=12)
        
        for col in columns:
            self.screener_tree.heading(col, text=col)
            self.screener_tree.column(col, width=60 if col != 'Symbol' else 70)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL,
                                   command=self.screener_tree.yview)
        self.screener_tree.configure(yscrollcommand=scrollbar.set)
        
        self.screener_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _apply_preset(self, preset):
        """Apply a filter preset."""
        self._clear_filters()
        
        presets = {
            'momentum': {'Momentum': (True, '>', '0.2')},
            'low_vol': {'Volatility': (True, '<', '-0.1')},
            'value': {'Value': (True, '>', '0.2')},
            'high_beta': {'Market': (True, '>', '0.3')}
        }
        
        if preset in presets:
            for factor, (enabled, op, val) in presets[preset].items():
                if factor in self.filter_vars:
                    en_var, op_var, val_var = self.filter_vars[factor]
                    en_var.set(enabled)
                    op_var.set(op)
                    val_var.set(val)
            self._run_screener()
    
    def _clear_filters(self):
        """Clear all filters."""
        for factor, (en_var, op_var, val_var) in self.filter_vars.items():
            en_var.set(False)
            val_var.set('0.0')
    
    def _run_screener(self):
        """Run the factor screener."""
        if not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        pca = self.ml_engine.pca_engine
        if not pca._is_fitted:
            return
        
        exposures = pca.get_all_exposures()
        if exposures.empty:
            return
        
        # Apply filters
        filtered = exposures.copy()
        
        for factor, (en_var, op_var, val_var) in self.filter_vars.items():
            if not en_var.get():
                continue
            
            try:
                threshold = float(val_var.get())
                op = op_var.get()
                
                if op == '>':
                    filtered = filtered[filtered[factor] > threshold]
                elif op == '<':
                    filtered = filtered[filtered[factor] < threshold]
                else:  # =
                    filtered = filtered[abs(filtered[factor] - threshold) < 0.1]
            except ValueError:
                continue
        
        # Update results
        for item in self.screener_tree.get_children():
            self.screener_tree.delete(item)
        
        for symbol in filtered.index:
            row = filtered.loc[symbol]
            alignment = pca.calculate_factor_alignment(symbol)
            
            values = (
                symbol,
                f"{row.get('Market', 0):+.2f}",
                f"{row.get('Size', 0):+.2f}",
                f"{row.get('Value', 0):+.2f}",
                f"{row.get('Momentum', 0):+.2f}",
                f"{row.get('Volatility', 0):+.2f}",
                f"{alignment:+.2f}"
            )
            self.screener_tree.insert('', 'end', values=values)
        
        self.screener_status.config(
            text=f"Found {len(filtered)} stocks",
            foreground=COLORS['gain'] if len(filtered) > 0 else COLORS['warning']
        )
    
    # ==================== AI TAB ====================
    
    def _create_ai_tab(self, parent):
        """Create AI recommendations tab."""
        # Regime-based recommendations
        rec_frame = ttk.LabelFrame(parent, text="🤖 AI Factor Recommendations")
        rec_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current regime display
        regime_display = ttk.Frame(rec_frame)
        regime_display.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(regime_display, text="Current Market Regime:", 
                  font=('Helvetica', 11)).pack(side=tk.LEFT)
        
        self.ai_regime_label = ttk.Label(regime_display, text="Unknown", 
                                          font=('Helvetica', 14, 'bold'))
        self.ai_regime_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(regime_display, text="🔄 Refresh Insights", 
                   command=self._generate_ai_insights).pack(side=tk.RIGHT)
        
        # Recommendations text
        rec_text_frame = ttk.Frame(rec_frame)
        rec_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.ai_text = tk.Text(rec_text_frame, height=20, width=60, 
                               wrap=tk.WORD, font=('Helvetica', 10),
                               bg='#2d2d4e', fg='white', insertbackground='white')
        self.ai_text.pack(fill=tk.BOTH, expand=True)
        
        # Insert default text
        self.ai_text.insert('1.0', "Click 'Refresh Insights' after fitting PCA to get AI recommendations...")
        self.ai_text.config(state=tk.DISABLED)
    
    def _generate_ai_insights(self):
        """Generate AI-powered factor insights."""
        if not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        pca = self.ml_engine.pca_engine
        if not pca._is_fitted:
            return
        
        regime_info = pca.get_market_regime()
        regime = regime_info.get('regime', 'Unknown')
        factor_signals = regime_info.get('factor_signals', {})
        
        # Update regime label
        color_map = {'Risk-On': COLORS['gain'], 'Risk-Off': COLORS['loss'], 'Rotation': COLORS['warning']}
        self.ai_regime_label.config(text=regime, foreground=color_map.get(regime, 'white'))
        
        # Generate recommendations
        recommendations = []
        recommendations.append(f"═══════════════════════════════════════")
        recommendations.append(f"  MARKET REGIME: {regime.upper()}")
        recommendations.append(f"  Confidence: {regime_info.get('confidence', 0):.0%}")
        recommendations.append(f"═══════════════════════════════════════\n")
        
        # Regime-specific advice
        if regime == 'Risk-On':
            recommendations.append("📈 BULLISH ENVIRONMENT DETECTED\n")
            recommendations.append("✅ RECOMMENDED TILTS:")
            recommendations.append("   • Increase Market beta exposure")
            recommendations.append("   • Favor high-momentum stocks")
            recommendations.append("   • Consider small-cap tilt")
            recommendations.append("\n⚠️ AVOID:")
            recommendations.append("   • Defensive/low-vol strategies")
            recommendations.append("   • Excessive hedging")
        elif regime == 'Risk-Off':
            recommendations.append("📉 DEFENSIVE ENVIRONMENT DETECTED\n")
            recommendations.append("✅ RECOMMENDED TILTS:")
            recommendations.append("   • Reduce Market beta exposure")
            recommendations.append("   • Favor low-volatility stocks")
            recommendations.append("   • Consider value tilt")
            recommendations.append("   • Increase cash allocation")
            recommendations.append("\n⚠️ AVOID:")
            recommendations.append("   • High-beta momentum plays")
            recommendations.append("   • Overconcentration in single factors")
        else:
            recommendations.append("⚡ ROTATION ENVIRONMENT DETECTED\n")
            recommendations.append("✅ RECOMMENDED TILTS:")
            recommendations.append("   • Diversify factor exposures")
            recommendations.append("   • Monitor for regime shift")
            recommendations.append("   • Balance momentum and value")
            recommendations.append("\n⚠️ CAUTION:")
            recommendations.append("   • High uncertainty period")
            recommendations.append("   • Avoid large factor bets")
        
        # Factor-specific signals
        recommendations.append("\n───────────────────────────────────────")
        recommendations.append("FACTOR SIGNALS:\n")
        
        for factor, signal in factor_signals.items():
            emoji = "🟢" if signal == 'Bullish' else "🔴" if signal == 'Bearish' else "⚪"
            recommendations.append(f"   {emoji} {factor}: {signal}")
        
        # Top picks based on alignment
        recommendations.append("\n───────────────────────────────────────")
        recommendations.append("TOP FACTOR-ALIGNED STOCKS:\n")
        
        exposures = pca.get_all_exposures()
        if not exposures.empty:
            alignments = {}
            for symbol in exposures.index:
                alignments[symbol] = pca.calculate_factor_alignment(symbol)
            
            top_aligned = sorted(alignments.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (symbol, align) in enumerate(top_aligned, 1):
                recommendations.append(f"   {i}. {symbol}: {align:+.3f}")
        
        # Update text widget
        self.ai_text.config(state=tk.NORMAL)
        self.ai_text.delete('1.0', tk.END)
        self.ai_text.insert('1.0', '\n'.join(recommendations))
        self.ai_text.config(state=tk.DISABLED)
    
    # ==================== STOCK ANALYSIS TAB (ENHANCED) ====================
    
    def _create_stock_tab(self, parent):
        """Create super enhanced stock analysis tab."""
        # Main scrollable frame
        main = ttk.Frame(parent)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top: Symbol selector
        self._create_stock_selector(main)
        
        # Create 3x2 grid of panels
        grid = ttk.Frame(main)
        grid.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Configure grid weights
        for i in range(3):
            grid.columnconfigure(i, weight=1)
        for i in range(2):
            grid.rowconfigure(i, weight=1)
        
        # Row 1: Radar Chart | Factor Bars | Similar Stocks
        self._create_radar_panel(grid, 0, 0)
        self._create_factor_bars_panel(grid, 0, 1)
        self._create_similar_stocks_panel(grid, 0, 2)
        
        # Row 2: Attribution | Regime Sensitivity | Risk Decomposition
        self._create_attribution_panel(grid, 1, 0)
        self._create_stock_regime_panel(grid, 1, 1)
        self._create_risk_panel(grid, 1, 2)
        
        # Bottom: AI Insight + What-If
        self._create_ai_insight_panel(main)
    
    def _create_stock_selector(self, parent):
        """Create stock selector row."""
        sel_frame = ttk.Frame(parent)
        sel_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(sel_frame, text="🔬 Stock:", font=('Helvetica', 11, 'bold')).pack(side=tk.LEFT)
        
        self.symbol_var = tk.StringVar()
        self.symbol_combo = ttk.Combobox(sel_frame, textvariable=self.symbol_var, width=12)
        self.symbol_combo.pack(side=tk.LEFT, padx=5)
        self.symbol_combo.bind('<<ComboboxSelected>>', self._on_symbol_change)
        
        ttk.Button(sel_frame, text="🔍 Analyze", command=self._analyze_stock).pack(side=tk.LEFT, padx=5)
        
        # Current stock info
        self.stock_info_label = ttk.Label(sel_frame, text="", font=('Helvetica', 10))
        self.stock_info_label.pack(side=tk.LEFT, padx=20)
    
    def _create_radar_panel(self, parent, row, col):
        """Create factor profile bar chart panel (replaced radar to avoid segfault)."""
        frame = ttk.LabelFrame(parent, text="📊 Factor Profile")
        frame.grid(row=row, column=col, sticky='nsew', padx=3, pady=3)
        
        if MATPLOTLIB_AVAILABLE:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Use regular bar chart instead of polar (avoids macOS segfault)
            self._radar_fig = Figure(figsize=(3, 2.5), dpi=80, facecolor='#1a1a2e')
            self._radar_canvas = FigureCanvasTkAgg(self._radar_fig, master=frame)
            self._radar_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(frame, text="Matplotlib required").pack()
    
    def _create_factor_bars_panel(self, parent, row, col):
        """Create factor exposure bar panel."""
        frame = ttk.LabelFrame(parent, text="🎯 Factor Exposures")
        frame.grid(row=row, column=col, sticky='nsew', padx=3, pady=3)
        
        inner = ttk.Frame(frame)
        inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stock_exposure_labels = {}
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        for i, factor in enumerate(factors):
            row_f = ttk.Frame(inner)
            row_f.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_f, text=factor[:3], width=4, font=('Helvetica', 8)).pack(side=tk.LEFT)
            bar = ttk.Progressbar(row_f, length=80, mode='determinate')
            bar.pack(side=tk.LEFT, padx=2)
            lbl = ttk.Label(row_f, text="-", width=6, font=('Helvetica', 9, 'bold'))
            lbl.pack(side=tk.LEFT)
            interp = ttk.Label(row_f, text="", font=('Helvetica', 8), foreground='gray')
            interp.pack(side=tk.LEFT)
            
            self.stock_exposure_labels[factor] = (bar, lbl, interp)
        
        # Alignment
        align_f = ttk.Frame(inner)
        align_f.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(align_f, text="Alignment:", font=('Helvetica', 9)).pack(side=tk.LEFT)
        self.alignment_label = ttk.Label(align_f, text="-", font=('Helvetica', 10, 'bold'))
        self.alignment_label.pack(side=tk.LEFT, padx=5)
        self.alignment_interp = ttk.Label(align_f, text="", font=('Helvetica', 8))
        self.alignment_interp.pack(side=tk.LEFT)
    
    def _create_similar_stocks_panel(self, parent, row, col):
        """Create similar stocks panel."""
        frame = ttk.LabelFrame(parent, text="👥 Similar Stocks")
        frame.grid(row=row, column=col, sticky='nsew', padx=3, pady=3)
        
        self.similar_stocks_list = tk.Listbox(frame, height=6, font=('Helvetica', 9),
                                               bg='#2d2d4e', fg='white', 
                                               selectbackground='#7c3aed')
        self.similar_stocks_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.similar_stocks_list.bind('<<ListboxSelect>>', self._on_similar_stock_click)
    
    def _create_attribution_panel(self, parent, row, col):
        """Create factor attribution panel."""
        frame = ttk.LabelFrame(parent, text="💰 1M Factor Attribution")
        frame.grid(row=row, column=col, sticky='nsew', padx=3, pady=3)
        
        inner = ttk.Frame(frame)
        inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.attribution_labels = {}
        items = ['Total', 'Market', 'Size', 'Value', 'Momentum', 'Alpha']
        
        for item in items:
            row_f = ttk.Frame(inner)
            row_f.pack(fill=tk.X, pady=1)
            
            prefix = "├─" if item != 'Alpha' else "└─"
            if item == 'Total':
                prefix = ""
                font = ('Helvetica', 9, 'bold')
            else:
                font = ('Helvetica', 8)
            
            ttk.Label(row_f, text=f"{prefix}{item}:", font=font, width=12).pack(side=tk.LEFT)
            lbl = ttk.Label(row_f, text="-", font=('Helvetica', 9, 'bold'), width=8)
            lbl.pack(side=tk.LEFT)
            self.attribution_labels[item] = lbl
    
    def _create_stock_regime_panel(self, parent, row, col):
        """Create stock regime sensitivity panel."""
        frame = ttk.LabelFrame(parent, text="⚡ Regime Sensitivity")
        frame.grid(row=row, column=col, sticky='nsew', padx=3, pady=3)
        
        inner = ttk.Frame(frame)
        inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.regime_performance_labels = {}
        regimes = ['Risk-On', 'Risk-Off', 'Rotation']
        
        for regime in regimes:
            row_f = ttk.Frame(inner)
            row_f.pack(fill=tk.X, pady=2)
            
            emoji = "🟢" if regime == 'Risk-On' else "🔴" if regime == 'Risk-Off' else "⚡"
            ttk.Label(row_f, text=f"{emoji} {regime}:", width=12, font=('Helvetica', 9)).pack(side=tk.LEFT)
            
            ret_lbl = ttk.Label(row_f, text="-", width=7, font=('Helvetica', 9, 'bold'))
            ret_lbl.pack(side=tk.LEFT)
            
            win_lbl = ttk.Label(row_f, text="", width=8, font=('Helvetica', 8), foreground='gray')
            win_lbl.pack(side=tk.LEFT)
            
            self.regime_performance_labels[regime] = (ret_lbl, win_lbl)
        
        # Best regime
        best_f = ttk.Frame(inner)
        best_f.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(best_f, text="Best in:", font=('Helvetica', 9)).pack(side=tk.LEFT)
        self.best_regime_label = ttk.Label(best_f, text="-", font=('Helvetica', 9, 'bold'))
        self.best_regime_label.pack(side=tk.LEFT, padx=5)
    
    def _create_risk_panel(self, parent, row, col):
        """Create risk decomposition panel."""
        frame = ttk.LabelFrame(parent, text="🎲 Risk Decomposition")
        frame.grid(row=row, column=col, sticky='nsew', padx=3, pady=3)
        
        inner = ttk.Frame(frame)
        inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.risk_decomp_labels = {}
        items = ['Market', 'Size', 'Value', 'Momentum', 'Idiosyncratic']
        
        for item in items:
            row_f = ttk.Frame(inner)
            row_f.pack(fill=tk.X, pady=1)
            
            ttk.Label(row_f, text=item[:4] + ":", width=5, font=('Helvetica', 8)).pack(side=tk.LEFT)
            bar = ttk.Progressbar(row_f, length=60, mode='determinate')
            bar.pack(side=tk.LEFT, padx=2)
            lbl = ttk.Label(row_f, text="-", width=5, font=('Helvetica', 8))
            lbl.pack(side=tk.LEFT)
            
            self.risk_decomp_labels[item] = (bar, lbl)
    
    def _create_ai_insight_panel(self, parent):
        """Create AI insight and what-if panel."""
        frame = ttk.LabelFrame(parent, text="🤖 AI Stock Insight")
        frame.pack(fill=tk.X, pady=(5, 0))
        
        inner = ttk.Frame(frame)
        inner.pack(fill=tk.X, padx=10, pady=10)
        
        # AI text
        self.stock_ai_text = ttk.Label(inner, text="Select a stock and click Analyze...",
                                        font=('Helvetica', 9), wraplength=600)
        self.stock_ai_text.pack(fill=tk.X)
        
        # What-if simulator
        whatif = ttk.Frame(inner)
        whatif.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(whatif, text="🎛️ What-If:", font=('Helvetica', 9, 'bold')).pack(side=tk.LEFT)
        ttk.Label(whatif, text="If Market", font=('Helvetica', 9)).pack(side=tk.LEFT, padx=(10, 0))
        
        self.whatif_var = tk.StringVar(value='5')
        whatif_entry = ttk.Entry(whatif, textvariable=self.whatif_var, width=4)
        whatif_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(whatif, text="% →", font=('Helvetica', 9)).pack(side=tk.LEFT)
        self.whatif_result = ttk.Label(whatif, text="Stock: -", font=('Helvetica', 9, 'bold'))
        self.whatif_result.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(whatif, text="Calculate", command=self._calculate_whatif).pack(side=tk.LEFT)
    
    # ==================== ACTIONS ====================
    
    def _fit_pca(self):
        """Load price data and fit PCA."""
        self.status_label.config(text="Loading price data...", foreground=COLORS['warning'])
        self.frame.update()
        
        try:
            if self.price_provider:
                self._price_data = self.price_provider()
            elif self.db:
                self._price_data = self._load_price_data()
            
            if not self._price_data:
                self.status_label.config(text="No price data", foreground=COLORS['loss'])
                return
            
            self.status_label.config(text="Fitting PCA...", foreground=COLORS['warning'])
            self.frame.update()
            
            if self.ml_engine and hasattr(self.ml_engine, 'fit_pca'):
                self.ml_engine.fit_pca(self._price_data)
            
            # Update symbol list
            symbols = sorted(self._price_data.keys())
            self.symbol_combo['values'] = symbols
            if symbols:
                self.symbol_combo.current(0)
            
            self._refresh_display()
            
            self.status_label.config(
                text=f"PCA fitted: {len(self._price_data)} stocks",
                foreground=COLORS['gain']
            )
            
        except Exception as e:
            logger.error(f"PCA fit failed: {e}")
            self.status_label.config(text=f"Error: {e}", foreground=COLORS['loss'])
    
    def _refresh_display(self):
        """Refresh all displays."""
        if not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        pca = self.ml_engine.pca_engine
        if not pca._is_fitted:
            return
        
        # Update regime
        regime_info = pca.get_market_regime()
        regime = regime_info.get('regime', 'Unknown')
        
        colors = {'Risk-On': COLORS['gain'], 'Risk-Off': COLORS['loss'], 'Rotation': COLORS['warning']}
        icons = {'Risk-On': '🟢', 'Risk-Off': '🔴', 'Rotation': '⚡'}
        
        self.regime_label.config(
            text=f"{regime} {icons.get(regime, '')}",
            foreground=colors.get(regime, 'white')
        )
        self.confidence_label.config(text=f"({regime_info.get('confidence', 0):.0%})")
        
        # Factor signals
        for factor, lbl in self.factor_signal_labels.items():
            signal = regime_info.get('factor_signals', {}).get(factor, 'Neutral')
            color = COLORS['gain'] if signal == 'Bullish' else COLORS['loss'] if signal == 'Bearish' else 'gray'
            lbl.config(text=f"{factor}: {signal}", foreground=color)
        
        # Variance
        variance = pca.get_variance_explained()
        for factor, (bar, lbl) in self.variance_bars.items():
            pct = variance.get(factor, 0) * 100
            bar['value'] = min(pct * 5, 100)  # Scale for visibility
            lbl.config(text=f"{pct:.1f}%")
        
        # Factor returns - multiple periods including YTD and annualized
        factor_returns = pca.get_factor_returns()
        if not factor_returns.empty:
            # Calculate YTD days
            from datetime import datetime
            today = datetime.now()
            ytd_days = (today - datetime(today.year, 1, 1)).days
            ytd_days = min(ytd_days, len(factor_returns))
            
            periods = [('1D', 1), ('1W', 5), ('1M', 20), ('YTD', ytd_days), ('Ann.', 252)]
            factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
            
            for factor in factors:
                if factor in self.factor_return_labels and factor in factor_returns.columns:
                    for period_name, days in periods:
                        if period_name in self.factor_return_labels[factor]:
                            lbl = self.factor_return_labels[factor][period_name]
                            
                            if period_name == 'Ann.':
                                # Annualized: mean daily return × 252
                                mean_daily = factor_returns[factor].mean()
                                ret = mean_daily * 252 * 100
                            else:
                                # Cumulative for period
                                if len(factor_returns) >= days and days > 0:
                                    ret = factor_returns[factor].tail(days).sum() * 100
                                else:
                                    ret = 0
                            
                            color = COLORS['gain'] if ret > 0 else COLORS['loss']
                            lbl.config(text=f"{ret:+.1f}%", foreground=color)
        
        # Exposure table
        self._update_exposure_table()
        
        # Update symbol dropdown for Stock Analysis tab
        if hasattr(pca, '_symbols') and pca._symbols:
            symbols = sorted(pca._symbols)
            self.symbol_combo['values'] = symbols
            if symbols and not self.symbol_var.get():
                self.symbol_combo.current(0)
        
        # Update chart
        if MATPLOTLIB_AVAILABLE:
            self._update_chart()
    
    def _update_exposure_table(self):
        """Update exposure table."""
        for item in self.exposure_tree.get_children():
            self.exposure_tree.delete(item)
        
        if not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        pca = self.ml_engine.pca_engine
        if not pca._is_fitted:
            return
        
        exposures = pca.get_all_exposures()
        if exposures.empty:
            return
        
        # Sort by selected factor
        sort_by = self.sort_var.get()
        exposures = exposures.sort_values(sort_by, ascending=False).head(30)
        
        for symbol in exposures.index:
            row = exposures.loc[symbol]
            market_val = row.get('Market', 0)
            
            # Determine tag based on market exposure
            if market_val > 0.3:
                tag = 'strong_pos'
            elif market_val > 0:
                tag = 'weak_pos'
            elif market_val > -0.3:
                tag = 'weak_neg'
            else:
                tag = 'strong_neg'
            
            values = (
                symbol,
                f"{row.get('Market', 0):+.2f}",
                f"{row.get('Size', 0):+.2f}",
                f"{row.get('Value', 0):+.2f}",
                f"{row.get('Momentum', 0):+.2f}",
                f"{row.get('Volatility', 0):+.2f}"
            )
            self.exposure_tree.insert('', 'end', values=values, tags=(tag,))
    
    def _sort_exposure_by(self, column):
        """Sort exposure table by column."""
        factor_map = {'Market': 'Market', 'Size': 'Size', 'Value': 'Value', 
                      'Mom': 'Momentum', 'Vol': 'Volatility'}
        if column in factor_map:
            self.sort_var.set(factor_map[column])
            self._update_exposure_table()
    
    def _on_stock_select(self, event):
        """Handle stock selection."""
        selection = self.exposure_tree.selection()
        if selection:
            item = self.exposure_tree.item(selection[0])
            symbol = item['values'][0]
            self.symbol_var.set(symbol)
            self._analyze_stock()
    
    def _on_symbol_change(self, event):
        """Handle symbol change."""
        self._analyze_stock()
    
    def _analyze_stock(self):
        """Analyze selected stock."""
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
            'Market': lambda v: "High β" if v > 0.3 else "Low β" if v < -0.3 else "Neutral",
            'Size': lambda v: "Large" if v > 0.3 else "Small" if v < -0.3 else "Mid",
            'Value': lambda v: "Value" if v > 0.3 else "Growth" if v < -0.3 else "Blend",
            'Momentum': lambda v: "Hi Mom" if v > 0.3 else "Mean-Rev" if v < -0.3 else "Mod",
            'Volatility': lambda v: "Hi Vol" if v > 0.3 else "Lo Vol" if v < -0.3 else "Norm"
        }
        
        # Update factor bars
        for factor, (bar, exp_lbl, interp_lbl) in self.stock_exposure_labels.items():
            value = exposures.get(factor, 0)
            color = COLORS['gain'] if value > 0 else COLORS['loss'] if value < 0 else 'gray'
            bar['value'] = (value + 1) * 50
            exp_lbl.config(text=f"{value:+.2f}", foreground=color)
            interp_lbl.config(text=interpretations.get(factor, lambda v: "")(value))
        
        # Alignment
        alignment = pca.calculate_factor_alignment(symbol)
        color = COLORS['gain'] if alignment > 0 else COLORS['loss']
        self.alignment_label.config(text=f"{alignment:+.2f}", foreground=color)
        interp = "✅ Aligned" if alignment > 0.2 else "⚠️ Misaligned" if alignment < -0.2 else "➖ Neutral"
        self.alignment_interp.config(text=interp)
        
        # Update radar chart
        self._update_radar_chart(symbol, exposures)
        
        # Similar stocks
        similar = pca.find_similar_stocks(symbol, top_n=5)
        self.similar_stocks_list.delete(0, tk.END)
        for sym, score in similar:
            self.similar_stocks_list.insert(tk.END, f"{sym}  ({score:.0%})")
        
        # Factor attribution
        attribution = pca.get_factor_attribution(symbol, days=20)
        for item, lbl in self.attribution_labels.items():
            if item == 'Volatility':
                continue  # Skip if not in attribution
            val = attribution.get(item, 0)
            color = COLORS['gain'] if val > 0 else COLORS['loss']
            lbl.config(text=f"{val:+.1f}%", foreground=color)
        
        # Regime performance
        regime_perf = pca.get_regime_performance(symbol)
        best_regime = None
        best_return = -999
        for regime, (ret_lbl, win_lbl) in self.regime_performance_labels.items():
            if regime in regime_perf:
                perf = regime_perf[regime]
                ret = perf.get('avg_return', 0)
                win = perf.get('win_rate', 50)
                color = COLORS['gain'] if ret > 0 else COLORS['loss']
                ret_lbl.config(text=f"{ret:+.1f}%", foreground=color)
                win_lbl.config(text=f"({win:.0f}% win)")
                if ret > best_return:
                    best_return = ret
                    best_regime = regime
        
        if best_regime:
            self.best_regime_label.config(text=best_regime, foreground=COLORS['gain'])
        
        # Risk decomposition
        risk_decomp = pca.get_risk_decomposition(symbol)
        for item, (bar, lbl) in self.risk_decomp_labels.items():
            pct = risk_decomp.get(item, 0)
            bar['value'] = min(pct, 100)
            lbl.config(text=f"{pct:.0f}%")
        
        # AI Insight
        self._generate_stock_ai_insight(symbol, exposures, alignment, attribution, regime_perf)
    
    def _update_radar_chart(self, symbol: str, exposures: dict):
        """Update the factor profile chart (horizontal bar, not polar)."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, '_radar_fig'):
            return
        
        import numpy as np
        
        try:
            self._radar_fig.clear()
            ax = self._radar_fig.add_subplot(111)
            ax.set_facecolor('#1a1a2e')
            
            factors = ['Mkt', 'Size', 'Val', 'Mom', 'Vol']
            full_factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
            stock_values = [exposures.get(f, 0) for f in full_factors]
            
            # Market average
            avg_values = [0] * 5
            if self.ml_engine and hasattr(self.ml_engine, 'pca_engine'):
                avg = self.ml_engine.pca_engine.get_market_average_exposures()
                avg_values = [avg.get(f, 0) for f in full_factors]
            
            y = np.arange(len(factors))
            height = 0.35
            
            # Horizontal bars
            bars1 = ax.barh(y - height/2, stock_values, height, label=symbol[:8], color='#00ff88')
            bars2 = ax.barh(y + height/2, avg_values, height, label='Market', color='#555555', alpha=0.7)
            
            ax.set_yticks(y)
            ax.set_yticklabels(factors, color='white', fontsize=8)
            ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Exposure', color='white', fontsize=8)
            ax.tick_params(colors='white', labelsize=7)
            ax.legend(loc='lower right', fontsize=7, facecolor='#2d2d4e', labelcolor='white')
            
            # Color bars by sign
            for bar, val in zip(bars1, stock_values):
                bar.set_color('#00ff88' if val >= 0 else '#ff6b6b')
            
            self._radar_fig.tight_layout()
            self._radar_canvas.draw()
        except Exception as e:
            logger.debug(f"Radar chart update failed: {e}")
    
    def _generate_stock_ai_insight(self, symbol: str, exposures: dict, 
                                    alignment: float, attribution: dict, regime_perf: dict):
        """Generate AI insight text for the stock."""
        # Build characteristics
        chars = []
        if exposures.get('Market', 0) > 0.3:
            chars.append("high-beta")
        elif exposures.get('Market', 0) < -0.3:
            chars.append("defensive")
        
        if exposures.get('Value', 0) > 0.3:
            chars.append("value-tilted")
        elif exposures.get('Value', 0) < -0.3:
            chars.append("growth-oriented")
        
        if exposures.get('Momentum', 0) > 0.3:
            chars.append("momentum-driven")
        
        if exposures.get('Volatility', 0) > 0.3:
            chars.append("high-volatility")
        elif exposures.get('Volatility', 0) < -0.3:
            chars.append("low-volatility")
        
        char_text = ", ".join(chars) if chars else "balanced"
        
        # Current regime assessment
        regime_info = self.ml_engine.pca_engine.get_market_regime()
        current_regime = regime_info.get('regime', 'Unknown')
        
        if current_regime in regime_perf:
            hist_ret = regime_perf[current_regime].get('avg_return', 0)
            if hist_ret > 5:
                regime_assessment = f"FAVORABLE. Historical {current_regime} return: {hist_ret:+.1f}%"
            elif hist_ret < -5:
                regime_assessment = f"UNFAVORABLE. Historical {current_regime} return: {hist_ret:+.1f}%"
            else:
                regime_assessment = f"NEUTRAL. Historical {current_regime} return: {hist_ret:+.1f}%"
        else:
            regime_assessment = "Insufficient data"
        
        # Alpha
        alpha = attribution.get('Alpha', 0)
        alpha_text = f"Recent alpha: {alpha:+.1f}%" if alpha else ""
        
        insight = f"{symbol} is a {char_text} stock. Current {current_regime} regime is {regime_assessment}. {alpha_text}"
        
        self.stock_ai_text.config(text=insight)
    
    def _on_similar_stock_click(self, event):
        """Handle click on similar stock to analyze it."""
        selection = self.similar_stocks_list.curselection()
        if selection:
            item = self.similar_stocks_list.get(selection[0])
            symbol = item.split()[0]  # Extract symbol from "SYMBOL (XX%)"
            self.symbol_var.set(symbol)
            self._analyze_stock()
    
    def _calculate_whatif(self):
        """Calculate what-if scenario."""
        symbol = self.symbol_var.get()
        if not symbol or not self.ml_engine or not hasattr(self.ml_engine, 'pca_engine'):
            return
        
        try:
            change = float(self.whatif_var.get())
            pca = self.ml_engine.pca_engine
            result = pca.calculate_what_if(symbol, 'Market', change)
            color = COLORS['gain'] if result > 0 else COLORS['loss']
            self.whatif_result.config(text=f"Stock: {result:+.1f}%", foreground=color)
        except ValueError:
            self.whatif_result.config(text="Invalid input")
    
    def _load_price_data(self):
        """Load price data from database."""
        import pandas as pd
        
        try:
            stocks = self.db.conn.execute("""
                SELECT DISTINCT s.symbol, s.id FROM stocks s 
                JOIN daily_prices dp ON s.id = dp.stock_id
            """).fetchall()
            
            price_data = {}
            for symbol, stock_id in stocks:
                prices = self.db.conn.execute("""
                    SELECT date, open, high, low, close, volume
                    FROM daily_prices WHERE stock_id = ? ORDER BY date
                """, [stock_id]).fetchall()
                
                if prices:
                    df = pd.DataFrame(prices, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    price_data[symbol] = df
            
            return price_data
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return {}

