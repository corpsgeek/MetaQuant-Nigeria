"""
Backtesting & Portfolio Optimization Tab for MetaQuant Nigeria.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import threading

import pandas as pd

from ..theme import COLORS, get_font
from ...database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Try to import backtesting modules
try:
    from ...backtesting import BacktestEngine, SignalScorer, PortfolioOptimizer, calculate_returns, ParameterOptimizer
    BACKTEST_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Backtesting not available: {e}")
    BACKTEST_AVAILABLE = False


class BacktestTab:
    """Backtesting and Portfolio Optimization Tab."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager, ml_engine=None):
        self.parent = parent
        self.db = db
        self.ml_engine = ml_engine  # For full signal computation
        
        # State
        self.all_stocks: List[Dict] = []
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.backtest_results: Optional[Dict] = None
        self.optimization_results: Optional[Dict] = None
        self.stock_params: Dict[str, Dict] = {}  # Per-stock SL/TP params
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        if not BACKTEST_AVAILABLE:
            self._create_unavailable_ui()
            return
        
        # Create UI
        self._create_ui()
        
        # Load data
        self.frame.after(1000, self._load_data)
    
    def _create_unavailable_ui(self):
        """Show message when backtesting not available."""
        container = ttk.Frame(self.frame)
        container.pack(expand=True)
        
        ttk.Label(
            container,
            text="📊 Backtesting Module Loading...",
            font=get_font('heading'),
            foreground=COLORS['warning']
        ).pack(pady=20)
        
        ttk.Label(
            container,
            text="Required: scipy",
            foreground=COLORS['text_muted']
        ).pack(pady=5)
    
    def _create_ui(self):
        """Create the backtesting UI."""
        # Create notebook for sub-tabs
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Backtest
        self.backtest_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.backtest_tab, text="📈 Strategy Backtest")
        self._create_backtest_ui()
        
        # Tab 2: Parameter Optimization
        self.param_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.param_tab, text="🎯 Stop/Target Optimizer")
        self._create_param_optimizer_ui()
        
        # Tab 3: Portfolio Optimization
        self.optimize_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.optimize_tab, text="⚖️ Portfolio Optimizer")
        self._create_optimizer_ui()
    
    def _create_param_optimizer_ui(self):
        """Create parameter optimizer sub-tab."""
        main = ttk.Frame(self.param_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=5)
        
        ttk.Label(header, text="🎯 Find Optimal Stop Loss & Take Profit Per Stock",
                  font=get_font('heading'), foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        # Description
        ttk.Label(main, text="Uses grid search to find the best SL/TP for each stock based on its volatility profile",
                  foreground=COLORS['text_muted']).pack(anchor='w', pady=5)
        
        # Controls
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.opt_params_btn = ttk.Button(btn_frame, text="⚡ Optimize All Stocks", 
                                          command=self._run_param_optimization)
        self.opt_params_btn.pack(side=tk.LEFT, padx=5)
        
        self.use_optimized_var = tk.BooleanVar(value=False)
        self.use_optimized_check = ttk.Checkbutton(btn_frame, text="Use optimized params in backtest",
                                                    variable=self.use_optimized_var)
        self.use_optimized_check.pack(side=tk.LEFT, padx=20)
        
        self.param_status = ttk.Label(btn_frame, text="Ready", foreground=COLORS['text_muted'])
        self.param_status.pack(side=tk.LEFT, padx=15)
        
        # Progress bar
        self.param_progress = ttk.Progressbar(main, mode='determinate', maximum=100)
        self.param_progress.pack(fill=tk.X, pady=5)
        
        # Results table
        results_frame = ttk.LabelFrame(main, text="📊 Optimized Parameters")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Symbol', 'Stop Loss %', 'Take Profit %', 'Win Rate %', 'Volatility %')
        self.param_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.param_tree.heading(col, text=col)
            self.param_tree.column(col, width=100)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.param_tree.yview)
        self.param_tree.configure(yscrollcommand=scrollbar.set)
        
        self.param_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_backtest_ui(self):
        """Create backtest sub-tab."""
        main = ttk.Frame(self.backtest_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ========== CONTROLS ==========
        controls = ttk.LabelFrame(main, text="⚙️ Backtest Settings")
        controls.pack(fill=tk.X, pady=5)
        
        controls_inner = ttk.Frame(controls)
        controls_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Row 1: Period and Capital
        row1 = ttk.Frame(controls_inner)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(row1, text="Period:").pack(side=tk.LEFT, padx=5)
        self.period_var = tk.StringVar(value="1Y")
        period_combo = ttk.Combobox(row1, textvariable=self.period_var, 
                                     values=["1M", "3M", "6M", "1Y", "2Y", "ALL"], 
                                     width=8, state='readonly')
        period_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Capital (₦):").pack(side=tk.LEFT, padx=15)
        self.capital_var = tk.StringVar(value="10,000,000")
        capital_entry = ttk.Entry(row1, textvariable=self.capital_var, width=15)
        capital_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Max Positions:").pack(side=tk.LEFT, padx=15)
        self.max_pos_var = tk.StringVar(value="10")
        max_pos_spin = ttk.Spinbox(row1, textvariable=self.max_pos_var, from_=1, to=20, width=5)
        max_pos_spin.pack(side=tk.LEFT, padx=5)
        
        # Row 2: Thresholds
        row2 = ttk.Frame(controls_inner)
        row2.pack(fill=tk.X, pady=5)
        
        ttk.Label(row2, text="Buy Threshold:").pack(side=tk.LEFT, padx=5)
        self.buy_thresh_var = tk.StringVar(value="0.3")
        ttk.Entry(row2, textvariable=self.buy_thresh_var, width=6).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text="Sell Threshold:").pack(side=tk.LEFT, padx=15)
        self.sell_thresh_var = tk.StringVar(value="-0.3")
        ttk.Entry(row2, textvariable=self.sell_thresh_var, width=6).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text="Stop Loss %:").pack(side=tk.LEFT, padx=15)
        self.stop_loss_var = tk.StringVar(value="5")
        ttk.Entry(row2, textvariable=self.stop_loss_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text="Take Profit %:").pack(side=tk.LEFT, padx=15)
        self.take_profit_var = tk.StringVar(value="15")
        ttk.Entry(row2, textvariable=self.take_profit_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Row 3: Signal mode
        row3 = ttk.Frame(controls_inner)
        row3.pack(fill=tk.X, pady=5)
        
        self.full_signals_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="Use Full Signals (ML + Fundamentals + Momentum)", 
                        variable=self.full_signals_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row3, text="Weights: Mom 35%, ML 25%, Fund 20%, Trend 10%, Anomaly 10%",
                  foreground=COLORS['text_muted'], font=get_font('small')).pack(side=tk.LEFT, padx=15)
        
        # Run button
        btn_frame = ttk.Frame(controls_inner)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.run_btn = ttk.Button(btn_frame, text="▶️ Run Backtest", command=self._run_backtest)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        self.bt_status = ttk.Label(btn_frame, text="Ready", foreground=COLORS['text_muted'])
        self.bt_status.pack(side=tk.LEFT, padx=15)
        
        # ========== RESULTS ==========
        results_frame = ttk.LabelFrame(main, text="📊 Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Metrics cards
        metrics_frame = ttk.Frame(results_frame)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.metric_cards = {}
        metrics = [
            ('total_return', 'Total Return', '0.0%'),
            ('sharpe', 'Sharpe Ratio', '0.00'),
            ('max_dd', 'Max Drawdown', '0.0%'),
            ('win_rate', 'Win Rate', '0.0%'),
            ('profit_factor', 'Profit Factor', '0.00'),
            ('trades', 'Total Trades', '0')
        ]
        
        for i, (key, label, default) in enumerate(metrics):
            card = ttk.Frame(metrics_frame, relief='ridge', borderwidth=1)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            
            ttk.Label(card, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(pady=2)
            
            value_label = ttk.Label(card, text=default, font=get_font('heading'))
            value_label.pack(pady=5)
            
            self.metric_cards[key] = value_label
        
        # Trade log
        log_frame = ttk.LabelFrame(results_frame, text="📋 Trade Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ('Symbol', 'Size (₦)', '% Port', 'Entry', 'Exit', 'P&L', 'Return', 'Days')
        self.trade_tree = ttk.Treeview(log_frame, columns=columns, show='headings', height=10)
        
        col_widths = {'Symbol': 70, 'Size (₦)': 90, '% Port': 50, 'Entry': 80, 'Exit': 80, 'P&L': 80, 'Return': 60, 'Days': 40}
        for col in columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=col_widths.get(col, 70))
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.trade_tree.yview)
        self.trade_tree.configure(yscrollcommand=scrollbar.set)
        
        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags for colors
        self.trade_tree.tag_configure('profit', foreground=COLORS['gain'])
        self.trade_tree.tag_configure('loss', foreground=COLORS['loss'])
    
    def _create_optimizer_ui(self):
        """Create portfolio optimizer sub-tab."""
        main = ttk.Frame(self.optimize_tab)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ========== CONTROLS ==========
        controls = ttk.LabelFrame(main, text="⚙️ Optimization Settings")
        controls.pack(fill=tk.X, pady=5)
        
        controls_inner = ttk.Frame(controls)
        controls_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Strategy selection
        row1 = ttk.Frame(controls_inner)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(row1, text="Strategy:").pack(side=tk.LEFT, padx=5)
        self.opt_strategy_var = tk.StringVar(value="MAX_SHARPE")
        strategies = [
            ("MAX_SHARPE", "Maximum Sharpe"),
            ("MIN_VOL", "Minimum Volatility"),
            ("RISK_PARITY", "Risk Parity"),
            ("EQUAL", "Equal Weight")
        ]
        for val, text in strategies:
            ttk.Radiobutton(row1, text=text, value=val, 
                           variable=self.opt_strategy_var).pack(side=tk.LEFT, padx=10)
        
        # Run button
        btn_frame = ttk.Frame(controls_inner)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.opt_btn = ttk.Button(btn_frame, text="⚖️ Optimize Portfolio", command=self._run_optimization)
        self.opt_btn.pack(side=tk.LEFT, padx=5)
        
        self.opt_status = ttk.Label(btn_frame, text="Ready", foreground=COLORS['text_muted'])
        self.opt_status.pack(side=tk.LEFT, padx=15)
        
        # ========== RESULTS ==========
        results_pane = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        results_pane.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left: Allocations
        alloc_frame = ttk.LabelFrame(results_pane, text="📊 Optimal Allocation")
        results_pane.add(alloc_frame, weight=1)
        
        # Portfolio metrics
        port_metrics = ttk.Frame(alloc_frame)
        port_metrics.pack(fill=tk.X, padx=10, pady=5)
        
        self.port_return = ttk.Label(port_metrics, text="Expected Return: --", font=get_font('normal'))
        self.port_return.pack(side=tk.LEFT, padx=10)
        
        self.port_vol = ttk.Label(port_metrics, text="Volatility: --", font=get_font('normal'))
        self.port_vol.pack(side=tk.LEFT, padx=10)
        
        self.port_sharpe = ttk.Label(port_metrics, text="Sharpe: --", font=get_font('normal'))
        self.port_sharpe.pack(side=tk.LEFT, padx=10)
        
        # Allocation tree
        alloc_tree_frame = ttk.Frame(alloc_frame)
        alloc_tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ('Symbol', 'Weight', 'Return', 'Volatility')
        self.alloc_tree = ttk.Treeview(alloc_tree_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.alloc_tree.heading(col, text=col)
            self.alloc_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(alloc_tree_frame, orient=tk.VERTICAL, command=self.alloc_tree.yview)
        self.alloc_tree.configure(yscrollcommand=scrollbar.set)
        
        self.alloc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right: Weight bars visualization
        bars_frame = ttk.LabelFrame(results_pane, text="📈 Weight Distribution")
        results_pane.add(bars_frame, weight=1)
        
        self.weight_bars_frame = ttk.Frame(bars_frame)
        self.weight_bars_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder for weight bars
        self.weight_bars = {}
    
    def _load_data(self):
        """Load stock data for backtesting."""
        try:
            # Load stocks
            self.all_stocks = self.db.get_all_stocks()
            
            # Load historical prices
            for stock in self.all_stocks[:50]:  # Top 50 for performance
                symbol = stock.get('symbol')
                stock_id = stock.get('id')
                
                if not stock_id:
                    continue
                
                history = self.db.get_price_history(stock_id, days=500)
                
                if history:
                    df = pd.DataFrame(history)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                    self.price_data[symbol] = df
            
            self.bt_status.config(text=f"Loaded {len(self.price_data)} stocks")
            self.opt_status.config(text=f"Loaded {len(self.price_data)} stocks")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.bt_status.config(text=f"Error: {e}", foreground=COLORS['loss'])
    
    def _run_backtest(self):
        """Run the backtest."""
        self.bt_status.config(text="Running...", foreground=COLORS['warning'])
        self.run_btn.state(['disabled'])
        
        def backtest():
            try:
                # Parse settings
                capital = float(self.capital_var.get().replace(',', ''))
                max_pos = int(self.max_pos_var.get())
                buy_thresh = float(self.buy_thresh_var.get())
                sell_thresh = float(self.sell_thresh_var.get())
                stop_loss = float(self.stop_loss_var.get()) / 100
                take_profit = float(self.take_profit_var.get()) / 100
                
                # Calculate date range
                period = self.period_var.get()
                days_map = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'ALL': 9999}
                days = days_map.get(period, 365)
                
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                # Check if using optimized per-stock parameters
                use_optimized = self.use_optimized_var.get() if hasattr(self, 'use_optimized_var') else False
                stock_params = self.stock_params if use_optimized and self.stock_params else None
                
                # Check if using full signals
                use_full_signals = self.full_signals_var.get() if hasattr(self, 'full_signals_var') else True
                
                if stock_params:
                    logger.info(f"Using optimized params for {len(stock_params)} stocks")
                
                logger.info(f"Using {'full multi-source' if use_full_signals else 'momentum-only'} signals")
                
                # Create engine with all data sources
                engine = BacktestEngine(
                    initial_capital=capital,
                    max_positions=max_pos,
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit,
                    buy_threshold=buy_thresh,
                    sell_threshold=sell_thresh,
                    stock_params=stock_params,
                    db=self.db,  # For fundamentals
                    ml_engine=self.ml_engine,  # For ML predictions
                    use_full_signals=use_full_signals
                )
                
                # Build signal data (empty - signals computed on-the-fly)
                signal_data = {}
                
                # Run
                symbols = list(self.price_data.keys())
                results = engine.run(
                    symbols=symbols,
                    price_data=self.price_data,
                    signal_data=signal_data,
                    start_date=start_date,
                    end_date=end_date
                )
                
                self.backtest_results = results
                self.frame.after(0, self._display_backtest_results)
                
            except Exception as e:
                logger.error(f"Backtest error: {e}")
                import traceback
                traceback.print_exc()
                self.frame.after(0, lambda: self.bt_status.config(
                    text=f"Error: {e}", foreground=COLORS['loss']))
            finally:
                self.frame.after(0, lambda: self.run_btn.state(['!disabled']))
        
        threading.Thread(target=backtest, daemon=True).start()
    
    def _display_backtest_results(self):
        """Display backtest results in UI."""
        if not self.backtest_results:
            return
        
        r = self.backtest_results
        m = r.get('metrics', {})
        
        # Update metrics cards
        ret = m.get('total_return_pct', 0)
        self.metric_cards['total_return'].config(
            text=f"{ret:+.1f}%",
            foreground=COLORS['gain'] if ret > 0 else COLORS['loss']
        )
        self.metric_cards['sharpe'].config(text=f"{m.get('sharpe_ratio', 0):.2f}")
        self.metric_cards['max_dd'].config(text=f"-{m.get('max_drawdown', 0):.1f}%")
        self.metric_cards['win_rate'].config(text=f"{m.get('win_rate', 0):.1f}%")
        self.metric_cards['profit_factor'].config(text=f"{m.get('profit_factor', 0):.2f}")
        self.metric_cards['trades'].config(text=str(m.get('total_trades', 0)))
        
        # Update trade log
        for item in self.trade_tree.get_children():
            self.trade_tree.delete(item)
        
        for t in r.get('trades', [])[:100]:  # Last 100 trades
            pnl = t.get('pnl', 0)
            tag = 'profit' if pnl > 0 else 'loss'
            
            # Calculate position size
            qty = t.get('quantity', 0)
            entry_price = t.get('entry_price', 0)
            position_value = qty * entry_price
            initial_capital = self.backtest_results.get('settings', {}).get('initial_capital', 1)
            pct_port = (position_value / initial_capital) * 100
            
            # Calculate contribution to portfolio return (P&L / initial capital)
            # This makes returns additive to total return
            contribution_return = (pnl / initial_capital) * 100
            
            self.trade_tree.insert('', 'end', values=(
                t.get('symbol', ''),
                f"₦{position_value:,.0f}",
                f"{pct_port:.1f}%",
                t.get('entry_date', '')[:10],
                t.get('exit_date', '')[:10],
                f"₦{pnl:,.0f}",
                f"{contribution_return:+.2f}%",
                t.get('holding_days', 0)
            ), tags=(tag,))
        
        # Calculate totals
        total_pnl = sum(t.get('pnl', 0) for t in r.get('trades', []))
        total_return_pct = m.get('total_return_pct', 0)
        unique_stocks = len(set(t.get('symbol', '') for t in r.get('trades', [])))
        total_trades = m.get('total_trades', 0)
        
        self.bt_status.config(
            text=f"✅ {total_trades} trades across {unique_stocks} stocks | Total P&L: ₦{total_pnl:,.0f} ({total_return_pct:+.2f}%)", 
            foreground=COLORS['gain'] if total_pnl >= 0 else COLORS['loss']
        )
    
    def _run_optimization(self):
        """Run portfolio optimization."""
        self.opt_status.config(text="Optimizing...", foreground=COLORS['warning'])
        self.opt_btn.state(['disabled'])
        
        def optimize():
            try:
                # Calculate returns
                returns_df = calculate_returns(self.price_data)
                
                if returns_df.empty or len(returns_df.columns) < 3:
                    raise ValueError("Need at least 3 stocks with return data")
                
                # Create optimizer
                optimizer = PortfolioOptimizer(returns_df)
                
                # Run selected strategy
                strategy = self.opt_strategy_var.get()
                
                if strategy == "MAX_SHARPE":
                    result = optimizer.optimize_max_sharpe()
                elif strategy == "MIN_VOL":
                    result = optimizer.optimize_min_volatility()
                elif strategy == "RISK_PARITY":
                    result = optimizer.optimize_risk_parity()
                else:
                    result = optimizer.equal_weight()
                
                self.optimization_results = result
                self.frame.after(0, self._display_optimization_results)
                
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.frame.after(0, lambda msg=error_msg: self.opt_status.config(
                    text=f"Error: {msg}", foreground=COLORS['loss']))
            finally:
                self.frame.after(0, lambda: self.opt_btn.state(['!disabled']))
        
        threading.Thread(target=optimize, daemon=True).start()
    
    def _display_optimization_results(self):
        """Display optimization results."""
        if not self.optimization_results:
            return
        
        r = self.optimization_results
        
        # Update portfolio metrics
        self.port_return.config(text=f"Expected Return: {r.get('expected_return', 0):.1f}%")
        self.port_vol.config(text=f"Volatility: {r.get('volatility', 0):.1f}%")
        self.port_sharpe.config(text=f"Sharpe: {r.get('sharpe_ratio', 0):.3f}")
        
        # Update allocation tree
        for item in self.alloc_tree.get_children():
            self.alloc_tree.delete(item)
        
        for alloc in r.get('allocations', []):
            self.alloc_tree.insert('', 'end', values=(
                alloc.get('symbol', ''),
                f"{alloc.get('weight', 0):.1f}%",
                f"{alloc.get('expected_return', 0):.1f}%",
                f"{alloc.get('volatility', 0):.1f}%"
            ))
        
        # Update weight bars
        for widget in self.weight_bars_frame.winfo_children():
            widget.destroy()
        
        for alloc in r.get('allocations', [])[:10]:  # Top 10
            bar_frame = ttk.Frame(self.weight_bars_frame)
            bar_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(bar_frame, text=alloc.get('symbol', ''), width=10).pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(bar_frame, length=200, mode='determinate',
                                   maximum=30)  # Max 30% per stock
            bar['value'] = alloc.get('weight', 0)
            bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            ttk.Label(bar_frame, text=f"{alloc.get('weight', 0):.1f}%", width=8).pack(side=tk.LEFT)
        
        self.opt_status.config(
            text=f"✅ {r.get('strategy', '')} - {r.get('n_assets', 0)} assets",
            foreground=COLORS['gain']
        )
    
    def _run_param_optimization(self):
        """Run parameter optimization for all stocks."""
        self.param_status.config(text="Optimizing...", foreground=COLORS['warning'])
        self.opt_params_btn.state(['disabled'])
        self.param_progress['value'] = 0
        
        def optimize():
            try:
                optimizer = ParameterOptimizer(min_trades=5)
                total = len(self.price_data)
                
                def update_progress(current, total_count, symbol):
                    pct = (current / total_count) * 100
                    self.frame.after(0, lambda: self._update_param_progress(pct, symbol))
                
                # Run optimization
                results = optimizer.optimize_all(self.price_data, update_progress)
                
                # Store results for use in backtest
                self.stock_params = optimizer.get_all_params()
                
                self.frame.after(0, lambda: self._display_param_results(optimizer))
                
            except Exception as e:
                logger.error(f"Parameter optimization error: {e}")
                import traceback
                traceback.print_exc()
                self.frame.after(0, lambda: self.param_status.config(
                    text=f"Error: {e}", foreground=COLORS['loss']))
            finally:
                self.frame.after(0, lambda: self.opt_params_btn.state(['!disabled']))
        
        threading.Thread(target=optimize, daemon=True).start()
    
    def _update_param_progress(self, pct: float, symbol: str):
        """Update parameter optimization progress."""
        self.param_progress['value'] = pct
        self.param_status.config(text=f"Optimizing: {symbol}...")
    
    def _display_param_results(self, optimizer):
        """Display parameter optimization results."""
        # Clear existing
        for item in self.param_tree.get_children():
            self.param_tree.delete(item)
        
        # Populate tree
        df = optimizer.to_dataframe()
        for _, row in df.iterrows():
            self.param_tree.insert('', 'end', values=(
                row['Symbol'],
                f"{row['Stop Loss %']:.1f}%",
                f"{row['Take Profit %']:.1f}%",
                f"{row['Win Rate %']:.1f}%",
                f"{row['Volatility %']:.1f}%"
            ))
        
        self.param_progress['value'] = 100
        self.param_status.config(
            text=f"✅ Optimized {len(self.stock_params)} stocks", 
            foreground=COLORS['gain']
        )
        
        # Auto-enable checkbox
        self.use_optimized_var.set(True)
    
    def refresh(self):
        """Refresh data."""
        self._load_data()

