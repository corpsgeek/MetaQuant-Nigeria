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
        periods = [('1D', 1), ('1W', 5), ('1M', 20), ('Ann.', 252)]  # Added Annualized
        
        # Header row
        ttk.Label(inner, text="Factor", width=9, font=('Helvetica', 8, 'bold')).grid(row=0, column=0)
        for i, (period_name, _) in enumerate(periods):
            ttk.Label(inner, text=period_name, width=7, font=('Helvetica', 8, 'bold')).grid(row=0, column=i+1)
        
        # Factor rows
        self.factor_return_labels = {}
        for row, factor in enumerate(factors, 1):
            ttk.Label(inner, text=factor, width=9, font=('Helvetica', 9)).grid(row=row, column=0, sticky='w')
            
            self.factor_return_labels[factor] = {}
            for col, (period_name, _) in enumerate(periods, 1):
                lbl = ttk.Label(inner, text="-", width=7, font=('Helvetica', 9, 'bold'))
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
        
        ttk.Label(ctrl, text="Window:").pack(side=tk.LEFT)
        self.chart_window_var = tk.StringVar(value='20')
        for days in ['20', '60', '90']:
            ttk.Radiobutton(ctrl, text=f"{days}d", variable=self.chart_window_var, 
                           value=days, command=self._update_chart).pack(side=tk.LEFT, padx=5)
        
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
        
        window = int(self.chart_window_var.get())
        
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
        ax.set_title(f'Factor Performance ({window}-day)', color='white', fontsize=12, fontweight='bold')
        
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
    
    # ==================== STOCK ANALYSIS TAB ====================
    
    def _create_stock_tab(self, parent):
        """Create stock analysis tab."""
        # Left: Stock selector and details
        left = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)
        
        # Selector
        sel_frame = ttk.LabelFrame(left, text="🔍 Select Stock")
        sel_frame.pack(fill=tk.X, pady=(0, 10))
        
        sel_inner = ttk.Frame(sel_frame)
        sel_inner.pack(fill=tk.X, padx=10, pady=10)
        
        self.symbol_var = tk.StringVar()
        self.symbol_combo = ttk.Combobox(sel_inner, textvariable=self.symbol_var, width=15)
        self.symbol_combo.pack(side=tk.LEFT, padx=5)
        self.symbol_combo.bind('<<ComboboxSelected>>', self._on_symbol_change)
        
        ttk.Button(sel_inner, text="Analyze", command=self._analyze_stock).pack(side=tk.LEFT, padx=5)
        
        # Factor exposures
        exp_frame = ttk.LabelFrame(left, text="📊 Factor Exposures")
        exp_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stock_exposure_labels = {}
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        
        for factor in factors:
            row_frame = ttk.Frame(exp_frame)
            row_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(row_frame, text=factor + ":", width=12, 
                     font=('Helvetica', 10)).pack(side=tk.LEFT)
            
            # Visual bar
            bar = ttk.Progressbar(row_frame, length=100, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            
            exp_lbl = ttk.Label(row_frame, text="-", width=8, 
                               font=('Helvetica', 10, 'bold'))
            exp_lbl.pack(side=tk.LEFT, padx=5)
            
            interp_lbl = ttk.Label(row_frame, text="", font=('Helvetica', 9), foreground='gray')
            interp_lbl.pack(side=tk.LEFT, padx=5)
            
            self.stock_exposure_labels[factor] = (bar, exp_lbl, interp_lbl)
        
        # Alignment score
        align_frame = ttk.Frame(exp_frame)
        align_frame.pack(fill=tk.X, padx=10, pady=15)
        
        ttk.Label(align_frame, text="Factor Alignment Score:", 
                  font=('Helvetica', 11, 'bold')).pack(side=tk.LEFT)
        
        self.alignment_label = ttk.Label(align_frame, text="-", 
                                          font=('Helvetica', 14, 'bold'))
        self.alignment_label.pack(side=tk.LEFT, padx=10)
        
        self.alignment_interp = ttk.Label(align_frame, text="", 
                                           font=('Helvetica', 10))
        self.alignment_interp.pack(side=tk.LEFT)
    
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
        
        # Factor returns - multiple periods including annualized
        factor_returns = pca.get_factor_returns()
        if not factor_returns.empty:
            periods = [('1D', 1), ('1W', 5), ('1M', 20), ('Ann.', 252)]
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
                                if len(factor_returns) >= days:
                                    ret = factor_returns[factor].tail(days).sum() * 100
                                else:
                                    ret = 0
                            
                            color = COLORS['gain'] if ret > 0 else COLORS['loss']
                            lbl.config(text=f"{ret:+.1f}%", foreground=color)
        
        # Exposure table
        self._update_exposure_table()
        
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
            'Market': lambda v: "High beta" if v > 0.3 else "Low beta" if v < -0.3 else "Neutral",
            'Size': lambda v: "Large-cap" if v > 0.3 else "Small-cap" if v < -0.3 else "Mid-cap",
            'Value': lambda v: "Value" if v > 0.3 else "Growth" if v < -0.3 else "Blend",
            'Momentum': lambda v: "High mom" if v > 0.3 else "Mean-revert" if v < -0.3 else "Moderate",
            'Volatility': lambda v: "High vol" if v > 0.3 else "Low vol" if v < -0.3 else "Normal"
        }
        
        for factor, (bar, exp_lbl, interp_lbl) in self.stock_exposure_labels.items():
            value = exposures.get(factor, 0)
            color = COLORS['gain'] if value > 0 else COLORS['loss'] if value < 0 else 'gray'
            
            # Update bar (scale -1 to 1 to 0 to 100)
            bar['value'] = (value + 1) * 50
            
            exp_lbl.config(text=f"{value:+.3f}", foreground=color)
            interp_lbl.config(text=interpretations.get(factor, lambda v: "")(value))
        
        # Alignment
        alignment = pca.calculate_factor_alignment(symbol)
        color = COLORS['gain'] if alignment > 0 else COLORS['loss']
        self.alignment_label.config(text=f"{alignment:+.3f}", foreground=color)
        
        if alignment > 0.2:
            interp = "✅ Well aligned with regime"
        elif alignment < -0.2:
            interp = "⚠️ Misaligned with regime"
        else:
            interp = "➖ Neutral alignment"
        self.alignment_interp.config(text=interp)
    
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
