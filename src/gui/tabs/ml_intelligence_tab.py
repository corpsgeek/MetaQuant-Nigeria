"""
ML Intelligence Tab for MetaQuant Nigeria.
Displays machine learning predictions, anomalies, and stock clustering.
"""

import tkinter as tk
from tkinter import ttk
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading

import pandas as pd

from src.gui.theme import COLORS, get_font
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Try to import ML components
try:
    from src.ml import MLEngine
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML module not available")


class MLIntelligenceTab:
    """
    ML Intelligence Tab - Displays ML predictions, anomalies, and clustering.
    
    Sub-tabs:
    1. Price Predictions - XGBoost predictions for selected stock
    2. Anomaly Detection - Real-time anomaly alerts
    3. Stock Clusters - K-Means clustering visualization
    4. Sector Rotation - Sector momentum analysis
    """
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager):
        """Initialize the ML Intelligence tab."""
        self.parent = parent
        self.db = db
        
        # Initialize ML Engine with database for sector rotation history
        if ML_AVAILABLE:
            self.ml_engine = MLEngine(db=db)
            logger.info(f"ML Engine Status: {self.ml_engine.get_status()}")
        else:
            self.ml_engine = None
        
        # State
        self.current_symbol = None
        self.all_stocks_data: List[Dict] = []
        self.historical_data: Optional[pd.DataFrame] = None
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        # Check ML availability
        if not ML_AVAILABLE or not self.ml_engine or not self.ml_engine.available:
            self._create_unavailable_ui()
            return
        
        # Create sub-notebook
        self._create_sub_notebook()
        
        # Initialize data in background
        self.frame.after(1000, self._initialize_ml)
    
    def _create_unavailable_ui(self):
        """Create UI when ML is not available."""
        container = ttk.Frame(self.frame)
        container.pack(expand=True)
        
        ttk.Label(
            container,
            text="🤖 ML Not Available",
            font=get_font('heading'),
            foreground=COLORS['warning']
        ).pack(pady=20)
        
        ttk.Label(
            container,
            text="Install required dependencies:",
            font=get_font('body'),
            foreground=COLORS['text_secondary']
        ).pack(pady=10)
        
        ttk.Label(
            container,
            text="pip install scikit-learn xgboost lightgbm",
            font=('Courier', 12),
            foreground=COLORS['primary']
        ).pack(pady=5)
        
        ttk.Label(
            container,
            text="brew install libomp  # Mac only",
            font=('Courier', 12),
            foreground=COLORS['primary']
        ).pack(pady=5)
    
    def _create_sub_notebook(self):
        """Create sub-tabs for ML features."""
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 0: Signal Overview (NEW - all stocks)
        self.overview_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.overview_tab, text="🎯 Signal Overview")
        self._create_overview_ui()
        
        # Tab 1: Price Predictions
        self.predictions_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.predictions_tab, text="📈 Price Predictions")
        self._create_predictions_ui()
        
        # Tab 2: Anomaly Detection
        self.anomaly_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.anomaly_tab, text="⚠️ Anomalies")
        self._create_anomaly_ui()
        
        # Tab 3: Stock Clusters
        self.clusters_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.clusters_tab, text="📊 Clusters")
        self._create_clusters_ui()
        
        # Tab 4: Sector Rotation
        self.rotation_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.rotation_tab, text="🔄 Sector Rotation")
        self._create_rotation_ui()
    
    def _create_overview_ui(self):
        """Create Signal Overview sub-tab with all stocks predictions."""
        main = self.overview_tab
        
        # Header
        header = ttk.Frame(main)
        header.pack(fill=tk.X, padx=15, pady=10)
        
        ttk.Label(header, text="🎯 ML Signal Overview",
                 font=get_font('heading'), foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        # Scan all Button
        scan_btn = ttk.Button(header, text="🔍 Scan All Stocks", command=self._scan_all_predictions)
        scan_btn.pack(side=tk.RIGHT)
        
        # === HERO CARDS ===
        cards_frame = ttk.Frame(main)
        cards_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.overview_cards = {}
        cards = [
            ('total', '📊 Total Scanned', '0', COLORS['text_primary']),
            ('buy', '🟢 BUY Signals', '0', COLORS['gain']),
            ('sell', '🔴 SELL Signals', '0', COLORS['loss']),
            ('hold', '🟡 HOLD Signals', '0', COLORS['warning']),
            ('accuracy', '🎯 Model Accuracy', '--%', COLORS['text_primary']),
        ]
        
        for key, label, default, color in cards:
            card = ttk.Frame(cards_frame, style='Card.TFrame')
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            inner = ttk.Frame(card)
            inner.pack(padx=12, pady=10)
            
            ttk.Label(inner, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='w')
            val = ttk.Label(inner, text=default, font=('Helvetica', 20, 'bold'),
                           foreground=color)
            val.pack(anchor='w')
            self.overview_cards[key] = val
        
        # === FILTER BAR ===
        filter_frame = ttk.Frame(main)
        filter_frame.pack(fill=tk.X, padx=15, pady=5)
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        
        self.overview_filter_var = tk.StringVar(value="ALL")
        for val, text in [("ALL", "All"), ("BUY", "🟢 BUY"), ("SELL", "🔴 SELL"), ("HOLD", "🟡 HOLD")]:
            ttk.Radiobutton(filter_frame, text=text, variable=self.overview_filter_var,
                           value=val, command=self._filter_overview_table).pack(side=tk.LEFT, padx=10)
        
        # Sector filter
        ttk.Label(filter_frame, text="  Sector:").pack(side=tk.LEFT, padx=(20, 5))
        self.overview_sector_var = tk.StringVar(value="ALL")
        self.overview_sector_combo = ttk.Combobox(filter_frame, textvariable=self.overview_sector_var,
                                                  width=15, state='readonly')
        self.overview_sector_combo['values'] = ["ALL"]
        self.overview_sector_combo.pack(side=tk.LEFT)
        self.overview_sector_combo.bind("<<ComboboxSelected>>", lambda e: self._filter_overview_table())
        
        # === PREDICTIONS TABLE ===
        table_frame = ttk.Frame(main)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        columns = ('rank', 'symbol', 'name', 'sector', 'signal', 'score', 'confidence', 'price', 'target')
        self.overview_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        
        col_config = [
            ('rank', '#', 40),
            ('symbol', 'Symbol', 80),
            ('name', 'Company', 150),
            ('sector', 'Sector', 100),
            ('signal', 'Signal', 70),
            ('score', 'Score', 70),
            ('confidence', 'Confidence', 85),
            ('price', 'Price ₦', 90),
            ('target', 'Target ₦', 90),
        ]
        
        for col_id, col_text, width in col_config:
            self.overview_tree.heading(col_id, text=col_text, command=lambda c=col_id: self._sort_overview(c))
            self.overview_tree.column(col_id, width=width, minwidth=width-10)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.overview_tree.yview)
        self.overview_tree.configure(yscrollcommand=scrollbar.set)
        
        self.overview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags for styling
        self.overview_tree.tag_configure('buy', foreground=COLORS['gain'])
        self.overview_tree.tag_configure('sell', foreground=COLORS['loss'])
        self.overview_tree.tag_configure('hold', foreground=COLORS['warning'])
        
        # Double-click to open analysis modal
        self.overview_tree.bind('<Double-1>', self._on_overview_double_click)
        
        # Store all predictions for filtering
        self.all_predictions = []
        self.overview_sort_column = 'score'
        self.overview_sort_reverse = True
    
    def _on_overview_double_click(self, event):
        """Open stock analysis modal on double-click."""
        selection = self.overview_tree.selection()
        if selection:
            item = self.overview_tree.item(selection[0])
            symbol = item['values'][1]  # Symbol is in column 1
            from src.gui.components.stock_analysis_modal import show_stock_analysis
            show_stock_analysis(self.parent, self.db, symbol, self.ml_engine)
    
    def _scan_all_predictions(self):
        """Scan all stocks for ML predictions."""
        if not self.ml_engine:
            return
        
        self.overview_cards['total'].config(text="Scanning...")
        
        def scan():
            try:
                # Get all stocks
                stocks = self.db.conn.execute(
                    "SELECT symbol, name, sector, last_price FROM stocks WHERE is_active = TRUE"
                ).fetchall()
                
                predictions = []
                buy_count = sell_count = hold_count = 0
                
                for symbol, name, sector, price in stocks:
                    try:
                        result = self.ml_engine.predict(symbol)
                        if result and 'prediction' in result:
                            pred = result['prediction']
                            direction = pred.get('direction', 'HOLD')
                            
                            if direction == 'UP':
                                signal = 'BUY'
                                buy_count += 1
                            elif direction == 'DOWN':
                                signal = 'SELL'
                                sell_count += 1
                            else:
                                signal = 'HOLD'
                                hold_count += 1
                            
                            predictions.append({
                                'symbol': symbol,
                                'name': name or '',
                                'sector': sector or 'Unknown',
                                'signal': signal,
                                'score': pred.get('predicted_change', 0),
                                'confidence': pred.get('confidence', 0),
                                'price': price or 0,
                                'target': price * (1 + pred.get('predicted_change', 0) / 100) if price else 0
                            })
                    except:
                        pass
                
                # Sort by score
                predictions.sort(key=lambda x: x['score'], reverse=True)
                
                # Add rank
                for i, p in enumerate(predictions):
                    p['rank'] = i + 1
                
                self.all_predictions = predictions
                
                # Update UI in main thread
                self.frame.after(0, lambda: self._display_overview_results(
                    predictions, buy_count, sell_count, hold_count
                ))
                
            except Exception as e:
                logger.error(f"Failed to scan predictions: {e}")
        
        import threading
        threading.Thread(target=scan, daemon=True).start()
    
    def _display_overview_results(self, predictions, buy_count, sell_count, hold_count):
        """Display overview scan results."""
        total = len(predictions)
        
        # Update cards
        self.overview_cards['total'].config(text=str(total))
        self.overview_cards['buy'].config(text=str(buy_count))
        self.overview_cards['sell'].config(text=str(sell_count))
        self.overview_cards['hold'].config(text=str(hold_count))
        
        # Update sector filter
        sectors = sorted(set(p['sector'] for p in predictions))
        self.overview_sector_combo['values'] = ["ALL"] + sectors
        
        # Populate table
        self._filter_overview_table()
    
    def _filter_overview_table(self):
        """Filter and display overview table."""
        # Clear
        for item in self.overview_tree.get_children():
            self.overview_tree.delete(item)
        
        filter_signal = self.overview_filter_var.get()
        filter_sector = self.overview_sector_var.get()
        
        filtered = [p for p in self.all_predictions
                   if (filter_signal == "ALL" or p['signal'] == filter_signal)
                   and (filter_sector == "ALL" or p['sector'] == filter_sector)]
        
        # Sort
        filtered.sort(key=lambda x: x.get(self.overview_sort_column, 0), 
                     reverse=self.overview_sort_reverse)
        
        for i, p in enumerate(filtered):
            tag = p['signal'].lower()
            self.overview_tree.insert('', tk.END, values=(
                i + 1,
                p['symbol'],
                p['name'][:20] if p['name'] else '',
                p['sector'][:15] if p['sector'] else '',
                f"{'🟢' if p['signal']=='BUY' else '🔴' if p['signal']=='SELL' else '🟡'} {p['signal']}",
                f"{p['score']:+.2f}%",
                f"{p['confidence']:.0f}%",
                f"₦{p['price']:,.2f}",
                f"₦{p['target']:,.2f}"
            ), tags=(tag,))
    
    def _sort_overview(self, column):
        """Sort overview table by column."""
        col_map = {'rank': 'rank', 'score': 'score', 'confidence': 'confidence', 
                   'price': 'price', 'target': 'target', 'symbol': 'symbol', 
                   'signal': 'signal', 'sector': 'sector'}
        
        if column in col_map:
            if self.overview_sort_column == col_map[column]:
                self.overview_sort_reverse = not self.overview_sort_reverse
            else:
                self.overview_sort_column = col_map[column]
                self.overview_sort_reverse = True
            self._filter_overview_table()

    def _create_predictions_ui(self):
        """Create super-enhanced price predictions sub-tab UI."""
        main = self.predictions_tab
        
        # Create scrollable canvas for all content
        canvas = tk.Canvas(main, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient=tk.VERTICAL, command=canvas.yview)
        self.pred_scrollable = ttk.Frame(canvas)
        
        self.pred_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=self.pred_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Make inner frame expand to full width
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        content = self.pred_scrollable
        
        # ========== HEADER ==========
        header = ttk.Frame(content)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="🤖 ML Price Prediction Engine", font=get_font('heading'),
                  foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        # Symbol Selector on right
        selector_frame = ttk.Frame(header)
        selector_frame.pack(side=tk.RIGHT)
        
        ttk.Label(selector_frame, text="Symbol:", font=get_font('body')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.pred_symbol_var = tk.StringVar(value="DANGCEM")
        self.pred_symbol_combo = ttk.Combobox(selector_frame, textvariable=self.pred_symbol_var, width=12)
        self.pred_symbol_combo.pack(side=tk.LEFT, padx=5)
        self.pred_symbol_combo.bind("<<ComboboxSelected>>", lambda e: self._update_prediction())
        
        predict_btn = ttk.Button(selector_frame, text="🔮 Predict", command=self._update_prediction)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        train_btn = ttk.Button(selector_frame, text="🎯 Train", command=self._train_model)
        train_btn.pack(side=tk.LEFT)
        
        # ========== MAIN PREDICTION SIGNAL ==========
        signal_frame = ttk.LabelFrame(content, text="📊 Prediction Signal")
        signal_frame.pack(fill=tk.X, padx=10, pady=5)
        
        signal_content = ttk.Frame(signal_frame)
        signal_content.pack(fill=tk.X, padx=15, pady=15)
        
        # Left: Big Direction Indicator
        dir_frame = ttk.Frame(signal_content)
        dir_frame.pack(side=tk.LEFT, padx=20)
        
        self.pred_results = {}
        
        ttk.Label(dir_frame, text="Direction", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['direction'] = ttk.Label(dir_frame, text="--", font=('Helvetica', 48, 'bold'))
        self.pred_results['direction'].pack()
        self.pred_results['direction_icon'] = ttk.Label(dir_frame, text="", font=('Helvetica', 24))
        self.pred_results['direction_icon'].pack()
        
        # Center: Probability Distribution
        prob_frame = ttk.Frame(signal_content)
        prob_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=20)
        
        ttk.Label(prob_frame, text="Probability Distribution", font=get_font('body'), 
                  foreground=COLORS['text_secondary']).pack(anchor=tk.W)
        
        # UP probability bar
        up_row = ttk.Frame(prob_frame)
        up_row.pack(fill=tk.X, pady=3)
        ttk.Label(up_row, text="📈 UP  ", font=get_font('small'), foreground=COLORS['gain'], width=8).pack(side=tk.LEFT)
        self.pred_results['prob_up_bar'] = ttk.Progressbar(up_row, length=200, mode='determinate', 
                                                           style='success.Horizontal.TProgressbar')
        self.pred_results['prob_up_bar'].pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.pred_results['prob_up'] = ttk.Label(up_row, text="0%", font=get_font('small'), 
                                                 foreground=COLORS['gain'], width=6)
        self.pred_results['prob_up'].pack(side=tk.RIGHT)
        
        # FLAT probability bar
        flat_row = ttk.Frame(prob_frame)
        flat_row.pack(fill=tk.X, pady=3)
        ttk.Label(flat_row, text="↔️ FLAT", font=get_font('small'), foreground=COLORS['warning'], width=8).pack(side=tk.LEFT)
        self.pred_results['prob_flat_bar'] = ttk.Progressbar(flat_row, length=200, mode='determinate',
                                                              style='warning.Horizontal.TProgressbar')
        self.pred_results['prob_flat_bar'].pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.pred_results['prob_flat'] = ttk.Label(flat_row, text="0%", font=get_font('small'),
                                                   foreground=COLORS['warning'], width=6)
        self.pred_results['prob_flat'].pack(side=tk.RIGHT)
        
        # DOWN probability bar
        down_row = ttk.Frame(prob_frame)
        down_row.pack(fill=tk.X, pady=3)
        ttk.Label(down_row, text="📉 DOWN", font=get_font('small'), foreground=COLORS['loss'], width=8).pack(side=tk.LEFT)
        self.pred_results['prob_down_bar'] = ttk.Progressbar(down_row, length=200, mode='determinate',
                                                              style='danger.Horizontal.TProgressbar')
        self.pred_results['prob_down_bar'].pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.pred_results['prob_down'] = ttk.Label(down_row, text="0%", font=get_font('small'),
                                                   foreground=COLORS['loss'], width=6)
        self.pred_results['prob_down'].pack(side=tk.RIGHT)
        
        # Right: Signal Strength Gauge
        gauge_frame = ttk.Frame(signal_content)
        gauge_frame.pack(side=tk.RIGHT, padx=20)
        
        ttk.Label(gauge_frame, text="Signal Strength", font=get_font('small'), 
                  foreground=COLORS['text_muted']).pack()
        self.pred_results['confidence'] = ttk.Label(gauge_frame, text="--", font=('Helvetica', 36, 'bold'),
                                                    foreground=COLORS['primary'])
        self.pred_results['confidence'].pack()
        self.pred_results['confidence_label'] = ttk.Label(gauge_frame, text="", font=get_font('small'),
                                                          foreground=COLORS['text_muted'])
        self.pred_results['confidence_label'].pack()
        
        # ========== PRICE PREDICTION ==========
        price_frame = ttk.LabelFrame(content, text="💰 Price Forecast")
        price_frame.pack(fill=tk.X, padx=10, pady=5)
        
        price_row = ttk.Frame(price_frame)
        price_row.pack(fill=tk.X, padx=10, pady=10)
        
        # Current Price
        curr_card = ttk.Frame(price_row, relief='ridge', borderwidth=1, padding=15)
        curr_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(curr_card, text="Current Price", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['current_price'] = ttk.Label(curr_card, text="₦--", font=get_font('heading'))
        self.pred_results['current_price'].pack()
        
        # Arrow
        arrow_frame = ttk.Frame(price_row, padding=10)
        arrow_frame.pack(side=tk.LEFT)
        self.pred_results['price_arrow'] = ttk.Label(arrow_frame, text="➡️", font=('Helvetica', 24))
        self.pred_results['price_arrow'].pack()
        
        # Predicted Price
        pred_card = ttk.Frame(price_row, relief='ridge', borderwidth=1, padding=15)
        pred_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(pred_card, text="Predicted Price", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['price'] = ttk.Label(pred_card, text="₦--", font=get_font('heading'))
        self.pred_results['price'].pack()
        
        # Expected Return
        ret_card = ttk.Frame(price_row, relief='ridge', borderwidth=1, padding=15)
        ret_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(ret_card, text="Expected Return", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['return'] = ttk.Label(ret_card, text="--", font=get_font('heading'))
        self.pred_results['return'].pack()
        
        # Model Accuracy
        acc_card = ttk.Frame(price_row, relief='ridge', borderwidth=1, padding=15)
        acc_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(acc_card, text="Model Accuracy", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['accuracy'] = ttk.Label(acc_card, text="--", font=get_font('heading'))
        self.pred_results['accuracy'].pack()
        
        # ========== TRADE RECOMMENDATION ==========
        trade_frame = ttk.LabelFrame(content, text="📈 Trade Recommendation")
        trade_frame.pack(fill=tk.X, padx=10, pady=5)
        
        trade_content = ttk.Frame(trade_frame)
        trade_content.pack(fill=tk.X, padx=15, pady=15)
        
        # Recommendation label
        self.pred_results['recommendation'] = ttk.Label(
            trade_content, text="Select a stock to get ML prediction",
            font=get_font('subheading'), foreground=COLORS['text_secondary']
        )
        self.pred_results['recommendation'].pack(anchor=tk.W)
        
        # Recommendation details
        self.pred_results['rec_details'] = ttk.Label(
            trade_content, text="",
            font=get_font('body'), foreground=COLORS['text_muted'], wraplength=800
        )
        self.pred_results['rec_details'].pack(anchor=tk.W, pady=(5, 0))
        
        # ========== TOP FEATURES ==========
        features_outer = ttk.LabelFrame(content, text="🔬 Top Contributing Factors")
        features_outer.pack(fill=tk.X, padx=10, pady=5)
        
        self.feature_bars_frame = ttk.Frame(features_outer)
        self.feature_bars_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Pre-create 5 feature bar rows
        self.feature_bars = []
        for i in range(5):
            row = ttk.Frame(self.feature_bars_frame)
            row.pack(fill=tk.X, pady=3)
            
            name_lbl = ttk.Label(row, text=f"Feature {i+1}", font=get_font('small'), width=25, anchor=tk.W)
            name_lbl.pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(row, length=150, mode='determinate')
            bar.pack(side=tk.LEFT, padx=10)
            
            val_lbl = ttk.Label(row, text="0.00", font=get_font('small'), foreground=COLORS['text_muted'], width=10)
            val_lbl.pack(side=tk.LEFT)
            
            imp_lbl = ttk.Label(row, text="0%", font=get_font('small'), foreground=COLORS['primary'], width=6)
            imp_lbl.pack(side=tk.RIGHT)
            
            self.feature_bars.append({'name': name_lbl, 'bar': bar, 'value': val_lbl, 'importance': imp_lbl})
        
        # ========== MODEL INFO ==========
        info_frame = ttk.LabelFrame(content, text="ℹ️ Model Information")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_row = ttk.Frame(info_frame)
        info_row.pack(fill=tk.X, padx=10, pady=10)
        
        info_items = [
            ('model_type', '🧠 Model', 'XGBoost'),
            ('features_count', '📊 Features', '77'),
            ('thresholds', '📏 Thresholds', '±1%'),
            ('last_trained', '🕐 Last Trained', '--'),
            ('prediction_time', '⏱️ Prediction', '--')
        ]
        
        for key, label, default in info_items:
            card = ttk.Frame(info_row, relief='ridge', borderwidth=1, padding=8)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)
            ttk.Label(card, text=label, font=get_font('small'), foreground=COLORS['text_muted']).pack()
            self.pred_results[key] = ttk.Label(card, text=default, font=get_font('small'))
            self.pred_results[key].pack()
        
        # ========== FULL FEATURE IMPORTANCE (Expandable) ==========
        features_frame = ttk.LabelFrame(content, text="📊 All Feature Importance")
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for features
        cols = ('Rank', 'Feature', 'Importance', 'Current Value')
        self.features_tree = ttk.Treeview(features_frame, columns=cols, show='headings', height=8)
        
        widths = {'Rank': 50, 'Feature': 200, 'Importance': 100, 'Current Value': 120}
        for col in cols:
            self.features_tree.heading(col, text=col)
            self.features_tree.column(col, width=widths.get(col, 100))
        
        scrollbar2 = ttk.Scrollbar(features_frame, orient=tk.VERTICAL, command=self.features_tree.yview)
        self.features_tree.configure(yscrollcommand=scrollbar2.set)
        
        self.features_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ========== STATUS ==========
        status_frame = ttk.Frame(content)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pred_status = ttk.Label(status_frame, text="🔮 Select a symbol and click Predict to get ML forecast",
                                     font=get_font('small'), foreground=COLORS['text_muted'])
        self.pred_status.pack(side=tk.LEFT)
    
    def _create_anomaly_ui(self):
        """Create super-enhanced anomaly detection sub-tab UI."""
        main = self.anomaly_tab
        
        # Create scrollable canvas
        canvas = tk.Canvas(main, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient=tk.VERTICAL, command=canvas.yview)
        self.anomaly_scrollable = ttk.Frame(canvas)
        
        self.anomaly_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=self.anomaly_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Make inner frame expand to full width
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        content = self.anomaly_scrollable
        
        # ========== HEADER ==========
        header = ttk.Frame(content)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="🚨 Anomaly Detection Engine", font=get_font('heading'),
                  foreground=COLORS['warning']).pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)
        
        scan_btn = ttk.Button(btn_frame, text="🔍 Scan All Stocks", command=self._scan_anomalies)
        scan_btn.pack(side=tk.LEFT, padx=5)
        
        # ========== THREAT LEVEL GAUGE ==========
        threat_frame = ttk.LabelFrame(content, text="🎯 Market Threat Level")
        threat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        threat_content = ttk.Frame(threat_frame)
        threat_content.pack(fill=tk.X, padx=15, pady=15)
        
        # Left: Big threat indicator
        threat_left = ttk.Frame(threat_content)
        threat_left.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(threat_left, text="Anomaly Level", font=get_font('small'), 
                  foreground=COLORS['text_muted']).pack()
        self.threat_level = ttk.Label(threat_left, text="LOW", font=('Helvetica', 36, 'bold'),
                                      foreground=COLORS['gain'])
        self.threat_level.pack()
        self.threat_description = ttk.Label(threat_left, text="No significant anomalies detected",
                                            font=get_font('small'), foreground=COLORS['text_muted'])
        self.threat_description.pack()
        
        # Right: Summary stats
        self.anomaly_summary = {}
        
        stats_frame = ttk.Frame(threat_content)
        stats_frame.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=20)
        
        summaries = [
            ('total', '⚠️ Total', COLORS['warning']),
            ('volume_spike', '📊 Volume', COLORS['primary']),
            ('price_jump', '💰 Price', COLORS['gain']),
            ('smart_money', '🧠 Smart $', COLORS['secondary']),
            ('volatility', '📈 Volatility', COLORS['loss'])
        ]
        
        stats_row = ttk.Frame(stats_frame)
        stats_row.pack(fill=tk.X)
        
        for key, title, color in summaries:
            card = ttk.Frame(stats_row, relief='ridge', borderwidth=1, padding=12)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)
            ttk.Label(card, text=title, font=get_font('small'), foreground=COLORS['text_muted']).pack()
            self.anomaly_summary[key] = ttk.Label(card, text="0", font=get_font('heading'), foreground=color)
            self.anomaly_summary[key].pack()
        
        # ========== ANOMALY TYPE BREAKDOWN ==========
        types_frame = ttk.LabelFrame(content, text="📊 Anomaly Types Detected")
        types_frame.pack(fill=tk.X, padx=10, pady=5)
        
        types_content = ttk.Frame(types_frame)
        types_content.pack(fill=tk.X, padx=10, pady=10)
        
        self.anomaly_type_bars = {}
        
        anomaly_types = [
            ('volume_spike', '📊 Volume Spike', 'Unusual trading volume detected'),
            ('price_jump', '💰 Price Jump', 'Significant price movement'),
            ('accumulation', '📈 Accumulation', 'Institutional buying pattern'),
            ('distribution', '📉 Distribution', 'Institutional selling pattern'),
            ('volatility_spike', '⚡ Volatility Spike', 'Heightened price volatility'),
            ('smart_money', '🧠 Smart Money', 'Unusual block trades detected')
        ]
        
        for key, title, desc in anomaly_types:
            row = ttk.Frame(types_content)
            row.pack(fill=tk.X, pady=4)
            
            ttk.Label(row, text=title, font=get_font('body'), width=18, anchor=tk.W).pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(row, length=200, mode='determinate')
            bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            count_lbl = ttk.Label(row, text="0", font=get_font('body'), foreground=COLORS['warning'], width=5)
            count_lbl.pack(side=tk.LEFT, padx=5)
            
            desc_lbl = ttk.Label(row, text=desc, font=get_font('small'), foreground=COLORS['text_muted'])
            desc_lbl.pack(side=tk.RIGHT)
            
            self.anomaly_type_bars[key] = {'bar': bar, 'count': count_lbl}
        
        # ========== REAL-TIME ALERTS ==========
        alerts_frame = ttk.LabelFrame(content, text="🔔 Active Alerts")
        alerts_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.alerts_container = ttk.Frame(alerts_frame)
        self.alerts_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Pre-create 3 alert slots
        self.alert_cards = []
        for i in range(3):
            card = ttk.Frame(self.alerts_container, relief='ridge', borderwidth=2, padding=10)
            card.pack(fill=tk.X, pady=3)
            
            icon_lbl = ttk.Label(card, text="⚪", font=('Helvetica', 20))
            icon_lbl.pack(side=tk.LEFT, padx=10)
            
            text_frame = ttk.Frame(card)
            text_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
            
            title_lbl = ttk.Label(text_frame, text="No alert", font=get_font('body'))
            title_lbl.pack(anchor=tk.W)
            
            desc_lbl = ttk.Label(text_frame, text="--", font=get_font('small'), foreground=COLORS['text_muted'])
            desc_lbl.pack(anchor=tk.W)
            
            severity_lbl = ttk.Label(card, text="--", font=get_font('body'), width=10)
            severity_lbl.pack(side=tk.RIGHT)
            
            self.alert_cards.append({
                'frame': card, 'icon': icon_lbl, 'title': title_lbl, 
                'desc': desc_lbl, 'severity': severity_lbl
            })
        
        # ========== DETAILED ANOMALY TABLE ==========
        table_frame = ttk.LabelFrame(content, text="📋 All Detected Anomalies")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        cols = ('Severity', 'Symbol', 'Type', 'Time', 'Description', 'Score')
        self.anomaly_tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=12)
        
        widths = {'Severity': 70, 'Symbol': 80, 'Type': 120, 'Time': 100, 'Description': 350, 'Score': 60}
        for col in cols:
            self.anomaly_tree.heading(col, text=col)
            self.anomaly_tree.column(col, width=widths.get(col, 100))
        
        # Tags for severity colors
        self.anomaly_tree.tag_configure('high', foreground=COLORS['loss'])
        self.anomaly_tree.tag_configure('medium', foreground=COLORS['warning'])
        self.anomaly_tree.tag_configure('low', foreground=COLORS['gain'])
        
        scrollbar2 = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.anomaly_tree.yview)
        self.anomaly_tree.configure(yscrollcommand=scrollbar2.set)
        
        self.anomaly_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ========== STATUS ==========
        status_frame = ttk.Frame(content)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.anomaly_status = ttk.Label(status_frame, text="🔍 Click 'Scan All Stocks' to detect anomalies",
                                        font=get_font('small'), foreground=COLORS['text_muted'])
        self.anomaly_status.pack(side=tk.LEFT)
    
    def _create_clusters_ui(self):
        """Create super-enhanced stock clusters sub-tab UI."""
        main = self.clusters_tab
        
        # Create scrollable canvas
        canvas = tk.Canvas(main, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient=tk.VERTICAL, command=canvas.yview)
        self.cluster_scrollable = ttk.Frame(canvas)
        
        self.cluster_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=self.cluster_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Make inner frame expand to full width
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        content = self.cluster_scrollable
        
        # ========== HEADER ==========
        header = ttk.Frame(content)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="📊 K-Means Stock Clustering Engine", font=get_font('heading'),
                  foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)
        
        cluster_btn = ttk.Button(btn_frame, text="🔄 Re-cluster Stocks", command=self._recluster_stocks)
        cluster_btn.pack(side=tk.LEFT, padx=5)
        
        # ========== CLUSTER OVERVIEW GRID ==========
        overview_frame = ttk.LabelFrame(content, text="📋 Cluster Overview (8 K-Means Clusters)")
        overview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Two rows of 4 cards each
        self.cluster_cards = {}
        
        cluster_labels = [
            ('High Growth', COLORS['gain']),
            ('Value Defensive', COLORS['primary']),
            ('Momentum Leaders', COLORS['secondary']),
            ('Dividend Champions', COLORS['gain']),
            ('Turnaround Plays', COLORS['warning']),
            ('Blue Chips', COLORS['primary']),
            ('Small Caps', COLORS['text_secondary']),
            ('Speculative', COLORS['loss'])
        ]
        
        for row_idx in range(2):
            row_frame = ttk.Frame(overview_frame)
            row_frame.pack(fill=tk.X, padx=10, pady=5)
            
            for col_idx in range(4):
                i = row_idx * 4 + col_idx
                default_label, default_color = cluster_labels[i]
                
                card = ttk.Frame(row_frame, relief='ridge', borderwidth=2, padding=12)
                card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=3)
                
                # Cluster number
                num_lbl = ttk.Label(card, text=f"#{i}", font=('Helvetica', 14, 'bold'), foreground=default_color)
                num_lbl.pack()
                
                # Label
                label_lbl = ttk.Label(card, text=default_label, font=get_font('body'), foreground=default_color)
                label_lbl.pack()
                
                # Count
                count_lbl = ttk.Label(card, text="0 stocks", font=get_font('small'), foreground=COLORS['text_muted'])
                count_lbl.pack()
                
                # Progress bar for relative size
                bar = ttk.Progressbar(card, length=80, mode='determinate')
                bar.pack(pady=3)
                
                self.cluster_cards[i] = {'num': num_lbl, 'label': label_lbl, 'count': count_lbl, 'bar': bar, 'color': default_color}
        
        # ========== CLUSTER CHARACTERISTICS ==========
        chars_frame = ttk.LabelFrame(content, text="📈 Cluster Characteristics")
        chars_frame.pack(fill=tk.X, padx=10, pady=5)
        
        chars_content = ttk.Frame(chars_frame)
        chars_content.pack(fill=tk.X, padx=10, pady=10)
        
        self.cluster_stats = {}
        
        stats_items = [
            ('total_stocks', '📊 Total Stocks', '0'),
            ('clusters_active', '🎯 Active Clusters', '8'),
            ('largest_cluster', '📈 Largest Cluster', '--'),
            ('smallest_cluster', '📉 Smallest Cluster', '--'),
            ('inertia', '🔬 Model Inertia', '--')
        ]
        
        for key, label, default in stats_items:
            card = ttk.Frame(chars_content, relief='ridge', borderwidth=1, padding=10)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
            ttk.Label(card, text=label, font=get_font('small'), foreground=COLORS['text_muted']).pack()
            self.cluster_stats[key] = ttk.Label(card, text=default, font=get_font('body'))
            self.cluster_stats[key].pack()
        
        # ========== SIMILAR STOCKS FINDER ==========
        finder_frame = ttk.LabelFrame(content, text="🔍 Find Similar Stocks")
        finder_frame.pack(fill=tk.X, padx=10, pady=5)
        
        finder_content = ttk.Frame(finder_frame)
        finder_content.pack(fill=tk.X, padx=15, pady=15)
        
        # Input row
        input_row = ttk.Frame(finder_content)
        input_row.pack(fill=tk.X)
        
        ttk.Label(input_row, text="Enter Symbol:", font=get_font('body')).pack(side=tk.LEFT)
        
        self.cluster_symbol_var = tk.StringVar(value="DANGCEM")
        self.cluster_symbol_entry = ttk.Entry(input_row, textvariable=self.cluster_symbol_var, width=15, font=get_font('body'))
        self.cluster_symbol_entry.pack(side=tk.LEFT, padx=10)
        self.cluster_symbol_entry.bind('<Return>', lambda e: self._find_similar())
        
        find_btn = ttk.Button(input_row, text="🔍 Find Similar", command=self._find_similar)
        find_btn.pack(side=tk.LEFT)
        
        # Results area
        results_frame = ttk.Frame(finder_content)
        results_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Stock's cluster info
        self.similar_cluster_label = ttk.Label(results_frame, text="", font=get_font('subheading'))
        self.similar_cluster_label.pack(anchor=tk.W)
        
        # Similar stocks list
        self.similar_stocks_label = ttk.Label(results_frame, text="Enter a symbol and click Find Similar",
                                              font=get_font('body'), foreground=COLORS['text_muted'])
        self.similar_stocks_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Similar stocks as buttons
        self.similar_stocks_frame = ttk.Frame(results_frame)
        self.similar_stocks_frame.pack(fill=tk.X, pady=(10, 0))
        
        # ========== CLUSTER MEMBERS TABLE ==========
        members_frame = ttk.LabelFrame(content, text="👥 All Cluster Members")
        members_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Filter row
        filter_row = ttk.Frame(members_frame)
        filter_row.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(filter_row, text="Filter by Cluster:", font=get_font('small')).pack(side=tk.LEFT)
        
        self.cluster_filter_var = tk.StringVar(value="All")
        cluster_filter = ttk.Combobox(filter_row, textvariable=self.cluster_filter_var, width=20,
                                      values=["All"] + [f"Cluster {i}" for i in range(8)])
        cluster_filter.pack(side=tk.LEFT, padx=10)
        cluster_filter.bind("<<ComboboxSelected>>", lambda e: self._filter_cluster_members())
        
        # Table
        cols = ('Symbol', 'Cluster #', 'Cluster Label', 'Sector', 'Market Cap')
        self.members_tree = ttk.Treeview(members_frame, columns=cols, show='headings', height=10)
        
        widths = {'Symbol': 100, 'Cluster #': 80, 'Cluster Label': 150, 'Sector': 120, 'Market Cap': 100}
        for col in cols:
            self.members_tree.heading(col, text=col)
            self.members_tree.column(col, width=widths.get(col, 100))
        
        # Color tags for clusters
        for i in range(8):
            self.members_tree.tag_configure(f'cluster_{i}', foreground=cluster_labels[i][1])
        
        scrollbar2 = ttk.Scrollbar(members_frame, orient=tk.VERTICAL, command=self.members_tree.yview)
        self.members_tree.configure(yscrollcommand=scrollbar2.set)
        
        self.members_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ========== STATUS ==========
        status_frame = ttk.Frame(content)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.cluster_status = ttk.Label(status_frame, text="📊 Loading clusters...",
                                        font=get_font('small'), foreground=COLORS['text_muted'])
        self.cluster_status.pack(side=tk.LEFT)
    
    def _create_rotation_ui(self):
        """Create super-enhanced sector rotation sub-tab UI."""
        main = self.rotation_tab
        
        # Create scrollable canvas
        canvas = tk.Canvas(main, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient=tk.VERTICAL, command=canvas.yview)
        self.rotation_scrollable = ttk.Frame(canvas)
        
        self.rotation_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=self.rotation_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Make inner frame expand to full width
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        content = self.rotation_scrollable
        
        # ========== HEADER ==========
        header = ttk.Frame(content)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="🔄 ML Sector Rotation Predictor", font=get_font('heading'),
                  foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)
        
        train_btn = ttk.Button(btn_frame, text="🧠 Train Model", command=self._train_sector_rotation)
        train_btn.pack(side=tk.LEFT, padx=5)
        
        refresh_btn = ttk.Button(btn_frame, text="↻ Refresh Analysis", command=self._update_rotation)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # ========== ML STATUS BAR ==========
        ml_status_frame = ttk.Frame(content)
        ml_status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Data points collected
        self.sector_data_points = ttk.Label(ml_status_frame, text="📊 Data: 0/30 days", 
                                             foreground=COLORS['warning'])
        self.sector_data_points.pack(side=tk.LEFT, padx=10)
        
        # ML Status
        self.sector_ml_status = ttk.Label(ml_status_frame, text="🔶 Statistical Mode", 
                                           foreground=COLORS['warning'])
        self.sector_ml_status.pack(side=tk.LEFT, padx=10)
        
        # Model accuracy
        self.sector_accuracy = ttk.Label(ml_status_frame, text="Accuracy: --", 
                                          foreground=COLORS['text_muted'])
        self.sector_accuracy.pack(side=tk.LEFT, padx=10)
        
        # ========== ROTATION CYCLE INDICATOR ==========
        cycle_frame = ttk.LabelFrame(content, text="📊 Economic Cycle Position")
        cycle_frame.pack(fill=tk.X, padx=10, pady=5)
        
        cycle_content = ttk.Frame(cycle_frame)
        cycle_content.pack(fill=tk.X, padx=15, pady=15)
        
        # Left: Big cycle indicator
        cycle_left = ttk.Frame(cycle_content)
        cycle_left.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(cycle_left, text="Current Cycle", font=get_font('small'), 
                  foreground=COLORS['text_muted']).pack()
        self.cycle_indicator = ttk.Label(cycle_left, text="EXPANSION", font=('Helvetica', 32, 'bold'),
                                         foreground=COLORS['gain'])
        self.cycle_indicator.pack()
        self.cycle_description = ttk.Label(cycle_left, text="Risk-on assets favored",
                                           font=get_font('body'), foreground=COLORS['text_muted'])
        self.cycle_description.pack()
        
        # Center: Rotation status cards
        self.rotation_status = {}
        
        status_frame = ttk.Frame(cycle_content)
        status_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=20)
        
        status_items = [
            ('prediction', '🎯 Rotation Signal', '--'),
            ('rotating_to', '📈 Rotating To', '--'),
            ('rotating_from', '📉 Rotating From', '--'),
            ('confidence', '💪 Confidence', '--')
        ]
        
        status_row = ttk.Frame(status_frame)
        status_row.pack(fill=tk.X)
        
        for key, label, default in status_items:
            card = ttk.Frame(status_row, relief='ridge', borderwidth=1, padding=12)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)
            ttk.Label(card, text=label, font=get_font('small'), foreground=COLORS['text_muted']).pack()
            self.rotation_status[key] = ttk.Label(card, text=default, font=get_font('body'))
            self.rotation_status[key].pack()
        
        # ========== SECTOR ALLOCATION RECOMMENDATION ==========
        alloc_frame = ttk.LabelFrame(content, text="📈 Sector Allocation Recommendation")
        alloc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        alloc_content = ttk.Frame(alloc_frame)
        alloc_content.pack(fill=tk.X, padx=10, pady=10)
        
        self.sector_allocation_label = ttk.Label(
            alloc_content, text="Click 'Refresh Analysis' to get sector rotation recommendations",
            font=get_font('subheading'), foreground=COLORS['text_secondary']
        )
        self.sector_allocation_label.pack(anchor=tk.W)
        
        self.sector_allocation_details = ttk.Label(
            alloc_content, text="",
            font=get_font('body'), foreground=COLORS['text_muted'], wraplength=800
        )
        self.sector_allocation_details.pack(anchor=tk.W, pady=(5, 0))
        
        # ========== SECTOR MOMENTUM BARS ==========
        momentum_bars_frame = ttk.LabelFrame(content, text="📊 Sector Momentum Overview")
        momentum_bars_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.sector_momentum_bars = {}
        
        sectors = [
            'Financial Services', 'Oil & Gas', 'Consumer Goods', 'Industrial Goods',
            'Insurance', 'Conglomerates', 'Healthcare', 'Agriculture', 
            'ICT', 'Utilities', 'Real Estate', 'Construction'
        ]
        
        # Two columns of sector bars
        bars_container = ttk.Frame(momentum_bars_frame)
        bars_container.pack(fill=tk.X, padx=10, pady=10)
        
        left_col = ttk.Frame(bars_container)
        left_col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        right_col = ttk.Frame(bars_container)
        right_col.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        for idx, sector in enumerate(sectors):
            parent = left_col if idx < 6 else right_col
            
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=3, padx=5)
            
            name_lbl = ttk.Label(row, text=sector, font=get_font('small'), width=18, anchor=tk.W)
            name_lbl.pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(row, length=120, mode='determinate')
            bar.pack(side=tk.LEFT, padx=5)
            
            value_lbl = ttk.Label(row, text="0.0%", font=get_font('small'), width=8)
            value_lbl.pack(side=tk.LEFT)
            
            status_lbl = ttk.Label(row, text="--", font=get_font('small'), width=10)
            status_lbl.pack(side=tk.RIGHT)
            
            self.sector_momentum_bars[sector] = {'bar': bar, 'value': value_lbl, 'status': status_lbl}
        
        # ========== LEADERS AND LAGGARDS ==========
        ll_frame = ttk.Frame(content)
        ll_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Leaders
        leaders_frame = ttk.LabelFrame(ll_frame, text="🏆 Leading Sectors (Overweight)")
        leaders_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        leaders_content = ttk.Frame(leaders_frame)
        leaders_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.leader_cards = []
        for i in range(3):
            card = ttk.Frame(leaders_content, relief='ridge', borderwidth=2, padding=10)
            card.pack(fill=tk.X, pady=3)
            
            rank_lbl = ttk.Label(card, text=f"#{i+1}", font=('Helvetica', 16, 'bold'), foreground=COLORS['gain'])
            rank_lbl.pack(side=tk.LEFT, padx=10)
            
            info_frame = ttk.Frame(card)
            info_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
            
            sector_lbl = ttk.Label(info_frame, text="--", font=get_font('body'))
            sector_lbl.pack(anchor=tk.W)
            
            change_lbl = ttk.Label(info_frame, text="--", font=get_font('small'), foreground=COLORS['gain'])
            change_lbl.pack(anchor=tk.W)
            
            self.leader_cards.append({'sector': sector_lbl, 'change': change_lbl})
        
        # Laggards
        laggards_frame = ttk.LabelFrame(ll_frame, text="📉 Lagging Sectors (Underweight)")
        laggards_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        laggards_content = ttk.Frame(laggards_frame)
        laggards_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.laggard_cards = []
        for i in range(3):
            card = ttk.Frame(laggards_content, relief='ridge', borderwidth=2, padding=10)
            card.pack(fill=tk.X, pady=3)
            
            rank_lbl = ttk.Label(card, text=f"#{i+1}", font=('Helvetica', 16, 'bold'), foreground=COLORS['loss'])
            rank_lbl.pack(side=tk.LEFT, padx=10)
            
            info_frame = ttk.Frame(card)
            info_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
            
            sector_lbl = ttk.Label(info_frame, text="--", font=get_font('body'))
            sector_lbl.pack(anchor=tk.W)
            
            change_lbl = ttk.Label(info_frame, text="--", font=get_font('small'), foreground=COLORS['loss'])
            change_lbl.pack(anchor=tk.W)
            
            self.laggard_cards.append({'sector': sector_lbl, 'change': change_lbl})
        
        # ========== SECTOR MOMENTUM TABLE ==========
        momentum_frame = ttk.LabelFrame(content, text="📋 Detailed Sector Momentum")
        momentum_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        cols = ('Rank', 'Sector', 'Momentum', '1W Change', '1M Change', 'Status', 'Recommendation')
        self.momentum_tree = ttk.Treeview(momentum_frame, columns=cols, show='headings', height=10)
        
        widths = {'Rank': 50, 'Sector': 150, 'Momentum': 80, '1W Change': 80, '1M Change': 80, 'Status': 80, 'Recommendation': 100}
        for col in cols:
            self.momentum_tree.heading(col, text=col)
            self.momentum_tree.column(col, width=widths.get(col, 80))
        
        # Tags for momentum colors
        self.momentum_tree.tag_configure('bullish', foreground=COLORS['gain'])
        self.momentum_tree.tag_configure('bearish', foreground=COLORS['loss'])
        self.momentum_tree.tag_configure('neutral', foreground=COLORS['warning'])
        
        scrollbar2 = ttk.Scrollbar(momentum_frame, orient=tk.VERTICAL, command=self.momentum_tree.yview)
        self.momentum_tree.configure(yscrollcommand=scrollbar2.set)
        
        self.momentum_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ========== STATUS ==========
        status_frame = ttk.Frame(content)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.rotation_label = ttk.Label(status_frame, text="🔄 Tracking sector momentum...",
                                        font=get_font('small'), foreground=COLORS['text_muted'])
        self.rotation_label.pack(side=tk.LEFT)
    
    # ========== DATA METHODS ==========
    
    def _initialize_ml(self):
        """Initialize ML models in background."""
        if not self.ml_engine:
            return
        
        def init():
            try:
                # Get all stocks data
                from src.collectors.tradingview_collector import TradingViewCollector
                collector = TradingViewCollector()
                stocks = collector.get_all_stocks()
                
                if not stocks.empty:
                    self.all_stocks_data = stocks.to_dict('records')
                    
                    # Populate symbol combos
                    symbols = sorted([s.get('symbol', '') for s in self.all_stocks_data if s.get('symbol')])
                    self.frame.after(0, lambda: self._update_symbol_combos(symbols))
                    
                    # Cluster stocks
                    result = self.ml_engine.cluster_stocks(self.all_stocks_data)
                    if result.get('success'):
                        self.frame.after(0, self._update_cluster_display)
                    
                logger.info(f"ML initialized with {len(self.all_stocks_data)} stocks")
                
            except Exception as e:
                logger.error(f"ML initialization error: {e}")
        
        threading.Thread(target=init, daemon=True).start()
    
    def _update_symbol_combos(self, symbols: List[str]):
        """Update symbol combo boxes."""
        self.pred_symbol_combo['values'] = symbols
        if symbols:
            self.pred_symbol_var.set(symbols[0])
    
    def _update_prediction(self):
        """Update price prediction for selected symbol."""
        if not self.ml_engine:
            return
        
        symbol = self.pred_symbol_var.get()
        if not symbol:
            return
        
        self.pred_status.config(text=f"Predicting for {symbol}...")
        
        def predict():
            try:
                # Get historical data using IntradayCollector
                from src.collectors.intraday_collector import IntradayCollector
                collector = IntradayCollector(self.db)
                records = collector.fetch_history(symbol, interval='1d', n_bars=200)
                
                if not records:
                    self.frame.after(0, lambda: self.pred_status.config(text="No data available"))
                    return
                
                # Convert to DataFrame
                df = pd.DataFrame(records)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').sort_index()
                
                # Make prediction
                result = self.ml_engine.predict_price(df, symbol)
                
                # Update UI
                self.frame.after(0, lambda r=result: self._display_prediction(r))
                
            except Exception as ex:
                error_msg = str(ex)
                logger.error(f"Prediction error: {error_msg}")
                self.frame.after(0, lambda msg=error_msg: self.pred_status.config(text=f"Error: {msg}"))
        
        threading.Thread(target=predict, daemon=True).start()
    
    def _display_prediction(self, result: Dict):
        """Display prediction results in enhanced UI."""
        logger.info(f"Displaying prediction result: {result}")
        
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown')
            logger.error(f"Prediction not successful: {error_msg}")
            self.pred_status.config(text=f"❌ Prediction failed: {error_msg}")
            return
        
        try:
            symbol = result.get('symbol', '')
            
            # ========== DIRECTION ==========
            direction = result.get('direction', '--')
            dir_color = COLORS['gain'] if direction == 'UP' else COLORS['loss'] if direction == 'DOWN' else COLORS['warning']
            dir_icon = "📈" if direction == 'UP' else "📉" if direction == 'DOWN' else "↔️"
            
            self.pred_results['direction'].config(text=direction, foreground=dir_color)
            self.pred_results['direction_icon'].config(text=dir_icon)
            
            # ========== PROBABILITY DISTRIBUTION ==========
            probs = result.get('probabilities', {})
            prob_up = probs.get('up', 0) * 100
            prob_flat = probs.get('flat', 0) * 100
            prob_down = probs.get('down', 0) * 100
            
            self.pred_results['prob_up_bar']['value'] = prob_up
            self.pred_results['prob_up'].config(text=f"{prob_up:.1f}%")
            
            self.pred_results['prob_flat_bar']['value'] = prob_flat
            self.pred_results['prob_flat'].config(text=f"{prob_flat:.1f}%")
            
            self.pred_results['prob_down_bar']['value'] = prob_down
            self.pred_results['prob_down'].config(text=f"{prob_down:.1f}%")
            
            # ========== SIGNAL STRENGTH ==========
            conf = result.get('confidence', 0)
            conf_color = COLORS['gain'] if conf > 60 else COLORS['warning'] if conf > 40 else COLORS['loss']
            conf_label = "STRONG" if conf > 70 else "MODERATE" if conf > 50 else "WEAK"
            
            self.pred_results['confidence'].config(text=f"{conf:.1f}%", foreground=conf_color)
            self.pred_results['confidence_label'].config(text=conf_label)
            
            # ========== PRICE FORECAST ==========
            current_price = result.get('current_price', 0)
            pred_price = result.get('predicted_price', 0)
            expected_ret = result.get('expected_return', 0)
            
            self.pred_results['current_price'].config(text=f"₦{current_price:,.2f}")
            self.pred_results['price'].config(text=f"₦{pred_price:,.2f}", foreground=dir_color)
            
            # Price arrow
            arrow = "📈" if expected_ret > 0 else "📉" if expected_ret < 0 else "➡️"
            self.pred_results['price_arrow'].config(text=arrow)
            
            # Return
            ret_color = COLORS['gain'] if expected_ret > 0 else COLORS['loss'] if expected_ret < 0 else COLORS['warning']
            self.pred_results['return'].config(text=f"{expected_ret:+.2f}%", foreground=ret_color)
            
            # Model accuracy
            acc = result.get('model_accuracy', 0)
            acc_color = COLORS['gain'] if acc > 60 else COLORS['warning'] if acc > 45 else COLORS['loss']
            self.pred_results['accuracy'].config(text=f"{acc:.1f}%", foreground=acc_color)
            
            # ========== TRADE RECOMMENDATION ==========
            if direction == 'UP' and conf > 60:
                rec_text = f"🟢 BULLISH - Consider BUYING {symbol}"
                rec_details = f"Model predicts {expected_ret:+.2f}% upside with {conf:.1f}% confidence. Price target: ₦{pred_price:,.2f}"
                rec_color = COLORS['gain']
            elif direction == 'DOWN' and conf > 60:
                rec_text = f"🔴 BEARISH - Consider SELLING {symbol}"
                rec_details = f"Model predicts {abs(expected_ret):.2f}% downside with {conf:.1f}% confidence. Potential risk: ₦{pred_price:,.2f}"
                rec_color = COLORS['loss']
            elif conf < 40:
                rec_text = f"⚪ LOW CONFIDENCE - Wait for clearer signal"
                rec_details = f"Model confidence is only {conf:.1f}%. Consider waiting for a stronger signal before trading."
                rec_color = COLORS['text_muted']
            else:
                rec_text = f"🟡 NEUTRAL - {symbol} expected to trade sideways"
                rec_details = f"Model expects minimal price movement ({expected_ret:+.2f}%). Consider theta strategies or wait."
                rec_color = COLORS['warning']
            
            self.pred_results['recommendation'].config(text=rec_text, foreground=rec_color)
            self.pred_results['rec_details'].config(text=rec_details)
            
            # ========== TOP FEATURES (Visual bars) ==========
            top_features = result.get('top_features', [])
            max_importance = max([f.get('importance', 0) for f in top_features], default=1)
            
            for i, bar_row in enumerate(self.feature_bars):
                if i < len(top_features):
                    feat = top_features[i]
                    name = feat.get('name', '').replace('_', ' ').title()
                    value = feat.get('value', 0)
                    importance = feat.get('importance', 0)
                    
                    bar_row['name'].config(text=name[:25])
                    bar_row['bar']['value'] = (importance / max_importance * 100) if max_importance > 0 else 0
                    bar_row['value'].config(text=f"{value:.2f}")
                    bar_row['importance'].config(text=f"{importance*100:.1f}%")
                else:
                    bar_row['name'].config(text="--")
                    bar_row['bar']['value'] = 0
                    bar_row['value'].config(text="--")
                    bar_row['importance'].config(text="--")
            
            # ========== MODEL INFO ==========
            pred_time = result.get('prediction_time', '')
            if pred_time:
                try:
                    pred_dt = datetime.fromisoformat(pred_time)
                    self.pred_results['prediction_time'].config(text=pred_dt.strftime('%H:%M:%S'))
                except:
                    self.pred_results['prediction_time'].config(text='--')
            
            # ========== FEATURE IMPORTANCE TABLE ==========
            importance = self.ml_engine.get_feature_importance(symbol)
            self._display_feature_importance(importance, result.get('top_features', []))
            
            self.pred_status.config(text=f"✅ {symbol} prediction complete at {datetime.now().strftime('%H:%M:%S')}")
            logger.info("Prediction display complete")
            
        except Exception as e:
            logger.error(f"Error displaying prediction: {e}")
            import traceback
            traceback.print_exc()
            self.pred_status.config(text=f"Display error: {e}")
    
    def _display_feature_importance(self, importance: Dict[str, float], top_features: List[Dict] = None):
        """Display feature importance in tree with extended info."""
        # Clear existing
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)
        
        # Create value lookup from top_features
        value_lookup = {}
        if top_features:
            for f in top_features:
                value_lookup[f.get('name', '')] = f.get('value', 0)
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        for rank, (feature, imp) in enumerate(sorted_features, 1):
            value = value_lookup.get(feature, '--')
            value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
            self.features_tree.insert('', 'end', values=(
                f"#{rank}",
                feature.replace('_', ' ').title(),
                f"{imp*100:.2f}%",
                value_str
            ))
    
    def _train_model(self):
        """Train model for selected symbol."""
        if not self.ml_engine:
            return
        
        symbol = self.pred_symbol_var.get()
        if not symbol:
            return
        
        self.pred_status.config(text=f"Training model for {symbol}...")
        
        def train():
            try:
                from src.collectors.intraday_collector import IntradayCollector
                collector = IntradayCollector(self.db)
                records = collector.fetch_history(symbol, interval='1d', n_bars=500)
                
                if not records:
                    self.frame.after(0, lambda: self.pred_status.config(text="No data for training"))
                    return
                
                # Convert to DataFrame
                df = pd.DataFrame(records)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').sort_index()
                
                result = self.ml_engine.train_predictor(df, symbol)
                
                msg = f"Training complete! Accuracy: {result.get('direction_accuracy', 0)*100:.1f}%" if result.get('success') else f"Training failed: {result.get('error')}"
                self.frame.after(0, lambda m=msg: self.pred_status.config(text=m))
                
            except Exception as ex:
                error_msg = str(ex)
                logger.error(f"Training error: {error_msg}")
                self.frame.after(0, lambda msg=error_msg: self.pred_status.config(text=f"Error: {msg}"))
        
        threading.Thread(target=train, daemon=True).start()
    
    def _scan_anomalies(self):
        """Scan all stocks for anomalies."""
        if not self.ml_engine:
            return
        
        self.anomaly_status.config(text="🔄 Starting anomaly scan...")
        
        def scan():
            try:
                from src.collectors.intraday_collector import IntradayCollector
                collector = IntradayCollector(self.db)
                
                all_anomalies = []
                counts = {'volume_spike': 0, 'price_jump': 0, 'accumulation': 0, 'distribution': 0}
                
                # Scan top 30 stocks
                stocks_to_scan = self.all_stocks_data[:30]
                total = len(stocks_to_scan)
                
                for i, stock in enumerate(stocks_to_scan):
                    symbol = stock.get('symbol', '')
                    if not symbol:
                        continue
                    
                    # Update progress in GUI
                    self.frame.after(0, lambda s=symbol, i=i, t=total: self.anomaly_status.config(
                        text=f"🔍 Scanning {s} ({i+1}/{t})..."))
                    
                    try:
                        records = collector.fetch_history(symbol, interval='1d', n_bars=50)
                        if records:
                            df = pd.DataFrame(records)
                            df['datetime'] = pd.to_datetime(df['datetime'])
                            df = df.set_index('datetime').sort_index()
                            
                            anomalies = self.ml_engine.detect_anomalies(df, symbol)
                            
                            for a in anomalies:
                                all_anomalies.append(a)
                                atype = a.get('type', '')
                                if atype in counts:
                                    counts[atype] += 1
                    except:
                        continue
                
                self.frame.after(0, lambda a=all_anomalies, c=counts: self._display_anomalies(a, c))
                
            except Exception as ex:
                error_msg = str(ex)
                logger.error(f"Anomaly scan error: {error_msg}")
                self.frame.after(0, lambda msg=error_msg: self.anomaly_status.config(text=f"❌ Error: {msg}"))
        
        threading.Thread(target=scan, daemon=True).start()
    
    def _display_anomalies(self, anomalies: List[Dict], counts: Dict):
        """Display detected anomalies in enhanced UI."""
        total = len(anomalies)
        
        # Update threat level
        if total > 20:
            self.threat_level.config(text="HIGH", foreground=COLORS['loss'])
            self.threat_description.config(text="Multiple significant anomalies detected across market")
        elif total > 10:
            self.threat_level.config(text="MODERATE", foreground=COLORS['warning'])
            self.threat_description.config(text="Several notable anomalies detected")
        elif total > 0:
            self.threat_level.config(text="LOW", foreground=COLORS['gain'])
            self.threat_description.config(text="Minor anomalies within normal parameters")
        else:
            self.threat_level.config(text="NONE", foreground=COLORS['gain'])
            self.threat_description.config(text="No significant anomalies detected")
        
        # Update summary cards
        if 'total' in self.anomaly_summary:
            self.anomaly_summary['total'].config(text=str(total))
        if 'volume_spike' in self.anomaly_summary:
            self.anomaly_summary['volume_spike'].config(text=str(counts.get('volume_spike', 0)))
        if 'price_jump' in self.anomaly_summary:
            self.anomaly_summary['price_jump'].config(text=str(counts.get('price_jump', 0)))
        if 'smart_money' in self.anomaly_summary:
            smart = counts.get('accumulation', 0) + counts.get('distribution', 0)
            self.anomaly_summary['smart_money'].config(text=str(smart))
        if 'volatility' in self.anomaly_summary:
            self.anomaly_summary['volatility'].config(text=str(counts.get('volatility_spike', 0)))
        
        # Update type bars
        max_count = max(counts.values()) if counts else 1
        for key, bar_data in self.anomaly_type_bars.items():
            count = counts.get(key, 0)
            bar_data['count'].config(text=str(count))
            bar_data['bar']['value'] = (count / max_count * 100) if max_count > 0 else 0
        
        # Update alert cards (top 3 most severe)
        sorted_anomalies = sorted(anomalies, key=lambda x: x.get('severity', 0), reverse=True)
        for i, card in enumerate(self.alert_cards):
            if i < len(sorted_anomalies):
                a = sorted_anomalies[i]
                severity = a.get('severity', 0)
                sev_color = COLORS['loss'] if severity > 70 else COLORS['warning'] if severity > 40 else COLORS['gain']
                icon = "🔴" if severity > 70 else "🟡" if severity > 40 else "🟢"
                
                card['icon'].config(text=icon)
                card['title'].config(text=f"{a.get('symbol', '')} - {a.get('type', '').replace('_', ' ').title()}")
                card['desc'].config(text=a.get('description', '')[:80])
                card['severity'].config(text=f"{severity:.0f}%", foreground=sev_color)
            else:
                card['icon'].config(text="⚪")
                card['title'].config(text="No alert")
                card['desc'].config(text="--")
                card['severity'].config(text="--")
        
        # Clear and populate tree
        for item in self.anomaly_tree.get_children():
            self.anomaly_tree.delete(item)
        
        for a in sorted_anomalies[:50]:
            severity = a.get('severity', 0)
            sev_tag = 'high' if severity > 70 else 'medium' if severity > 40 else 'low'
            sev_label = "🔴 HIGH" if severity > 70 else "🟡 MED" if severity > 40 else "🟢 LOW"
            time_str = a.get('detected_at', '')[:19].replace('T', ' ') if a.get('detected_at') else '--'
            
            self.anomaly_tree.insert('', 'end', values=(
                sev_label,
                a.get('symbol', ''),
                a.get('type', '').replace('_', ' ').title(),
                time_str,
                a.get('description', '')[:60],
                f"{severity:.0f}%"
            ), tags=(sev_tag,))
        
        self.anomaly_status.config(text=f"✅ Found {total} anomalies at {datetime.now().strftime('%H:%M:%S')}")
    
    def _recluster_stocks(self):
        """Re-cluster all stocks."""
        if not self.ml_engine or not self.all_stocks_data:
            return
        
        self.cluster_status.config(text="🔄 Clustering stocks... please wait")
        
        def cluster():
            self.frame.after(0, lambda: self.cluster_status.config(
                text=f"🔄 Clustering {len(self.all_stocks_data)} stocks..."))
            result = self.ml_engine.cluster_stocks(self.all_stocks_data)
            if result.get('success'):
                self.frame.after(0, self._update_cluster_display)
            else:
                self.frame.after(0, lambda: self.cluster_status.config(text=f"❌ Error: {result.get('error')}"))
        
        threading.Thread(target=cluster, daemon=True).start()
    
    def _update_cluster_display(self):
        """Update cluster display for enhanced UI."""
        if not self.ml_engine:
            return
        
        clusters = self.ml_engine.get_all_clusters()
        total_stocks = sum(info.get('size', 0) for info in clusters.values())
        max_size = max((info.get('size', 0) for info in clusters.values()), default=1)
        
        # Update cluster cards with bars
        for i, card in self.cluster_cards.items():
            if i in clusters:
                info = clusters[i]
                size = info.get('size', 0)
                card['label'].config(text=info.get('label', f'Cluster {i}'))
                card['count'].config(text=f"{size} stocks")
                if 'bar' in card:
                    card['bar']['value'] = (size / max_size * 100) if max_size > 0 else 0
            else:
                card['label'].config(text=f"Cluster {i}")
                card['count'].config(text="0 stocks")
                if 'bar' in card:
                    card['bar']['value'] = 0
        
        # Update stats
        if hasattr(self, 'cluster_stats'):
            self.cluster_stats['total_stocks'].config(text=str(total_stocks))
            self.cluster_stats['clusters_active'].config(text=str(len([c for c in clusters.values() if c.get('size', 0) > 0])))
            
            if clusters:
                largest = max(clusters.items(), key=lambda x: x[1].get('size', 0))
                smallest = min(clusters.items(), key=lambda x: x[1].get('size', 0))
                self.cluster_stats['largest_cluster'].config(text=f"#{largest[0]} ({largest[1].get('size', 0)})")
                self.cluster_stats['smallest_cluster'].config(text=f"#{smallest[0]} ({smallest[1].get('size', 0)})")
        
        # Update tree with all members
        for item in self.members_tree.get_children():
            self.members_tree.delete(item)
        
        for cluster_id, info in clusters.items():
            for symbol in info.get('stocks', []):
                self.members_tree.insert('', 'end', values=(
                    symbol, 
                    f"#{cluster_id}", 
                    info.get('label', ''),
                    '--',  # Sector placeholder
                    '--'   # Market cap placeholder
                ), tags=(f'cluster_{cluster_id}',))
        
        self.cluster_status.config(text=f"✅ Clustered {total_stocks} stocks into {len(clusters)} clusters at {datetime.now().strftime('%H:%M:%S')}")
    
    def _filter_cluster_members(self):
        """Filter cluster members table by selected cluster."""
        filter_val = self.cluster_filter_var.get()
        
        if not self.ml_engine:
            return
        
        clusters = self.ml_engine.get_all_clusters()
        
        # Clear tree
        for item in self.members_tree.get_children():
            self.members_tree.delete(item)
        
        # Filter and populate
        for cluster_id, info in clusters.items():
            if filter_val != "All" and f"Cluster {cluster_id}" != filter_val:
                continue
            for symbol in info.get('stocks', []):
                self.members_tree.insert('', 'end', values=(
                    symbol,
                    f"#{cluster_id}",
                    info.get('label', ''),
                    '--',
                    '--'
                ), tags=(f'cluster_{cluster_id}',))
    
    def _find_similar(self):
        """Find similar stocks with enhanced display."""
        if not self.ml_engine:
            return
        
        symbol = self.cluster_symbol_var.get().upper()
        cluster_info = self.ml_engine.get_cluster(symbol)
        
        similar = cluster_info.get('similar_stocks', [])
        label = cluster_info.get('label', 'Unknown')
        cluster_id = cluster_info.get('cluster', -1)
        
        # Update cluster label
        if hasattr(self, 'similar_cluster_label'):
            self.similar_cluster_label.config(
                text=f"📊 {symbol} is in Cluster #{cluster_id}: {label}",
                foreground=COLORS['primary']
            )
        
        # Update similar stocks label
        if hasattr(self, 'similar_stocks_label'):
            if similar:
                self.similar_stocks_label.config(
                    text=f"✅ Found {len(similar)} similar stocks: {', '.join(similar[:8])}{'...' if len(similar) > 8 else ''}",
                    foreground=COLORS['gain']
                )
            else:
                self.similar_stocks_label.config(
                    text="❌ No similar stocks found or symbol not in clusters",
                    foreground=COLORS['loss']
                )
    
    def _update_rotation(self):
        """Update sector rotation analysis with enhanced UI."""
        if not self.ml_engine or not self.all_stocks_data:
            self.rotation_label.config(text="❌ No data available for rotation analysis")
            return
        
        self.rotation_label.config(text="🔄 Analyzing sector rotation...")
        
        # Track current performance (stores to database for ML training)
        self.ml_engine.track_sector_performance(self.all_stocks_data)
        
        # Get prediction from ML predictor
        rotation = self.ml_engine.predict_sector_rotation()
        
        # Update ML status bar
        data_points = rotation.get('data_points', 0)
        ml_active = rotation.get('ml_active', False)
        accuracy = rotation.get('model_accuracy', 0)
        
        if hasattr(self, 'sector_data_points'):
            color = COLORS['gain'] if data_points >= 30 else COLORS['warning']
            self.sector_data_points.config(text=f"📊 Data: {data_points}/30 days", foreground=color)
        
        if hasattr(self, 'sector_ml_status'):
            if ml_active:
                self.sector_ml_status.config(text="🟢 ML Active", foreground=COLORS['gain'])
            else:
                self.sector_ml_status.config(text="🔶 Statistical Mode", foreground=COLORS['warning'])
        
        if hasattr(self, 'sector_accuracy'):
            if accuracy > 0:
                self.sector_accuracy.config(text=f"Accuracy: {accuracy:.1f}%", 
                                             foreground=COLORS['gain'] if accuracy > 50 else COLORS['warning'])
            else:
                self.sector_accuracy.config(text="Accuracy: --", foreground=COLORS['text_muted'])
        
        # Update cycle indicator
        leading = rotation.get('leading', [])
        lagging = rotation.get('lagging', [])
        
        if hasattr(self, 'cycle_indicator'):
            if len(leading) > len(lagging):
                self.cycle_indicator.config(text="EXPANSION", foreground=COLORS['gain'])
                self.cycle_description.config(text="Risk-on assets favored • Cyclicals outperforming")
            elif len(lagging) > len(leading):
                self.cycle_indicator.config(text="CONTRACTION", foreground=COLORS['loss'])
                self.cycle_description.config(text="Defensive positioning recommended • Rotate to safety")
            else:
                self.cycle_indicator.config(text="TRANSITION", foreground=COLORS['warning'])
                self.cycle_description.config(text="Mixed signals • Watch for rotation confirmation")
        
        # Update status cards
        self.rotation_status['prediction'].config(text=rotation.get('prediction', 'ANALYZING'))
        
        rotating_to = rotation.get('rotating_to')
        if rotating_to:
            self.rotation_status['rotating_to'].config(text=rotating_to, foreground=COLORS['gain'])
        else:
            self.rotation_status['rotating_to'].config(text="--")
        
        if 'rotating_from' in self.rotation_status:
            rotating_from = lagging[0] if lagging else '--'
            self.rotation_status['rotating_from'].config(text=rotating_from, foreground=COLORS['loss'])
        
        if 'confidence' in self.rotation_status:
            conf = rotation.get('confidence', 50)
            self.rotation_status['confidence'].config(text=f"{conf:.0f}%")
        
        # Update allocation recommendation
        if hasattr(self, 'sector_allocation_label'):
            if leading:
                self.sector_allocation_label.config(
                    text=f"🟢 OVERWEIGHT: {', '.join(leading[:3])}",
                    foreground=COLORS['gain']
                )
                if lagging:
                    self.sector_allocation_details.config(
                        text=f"Consider reducing exposure to: {', '.join(lagging[:3])}. Rotation signal suggests moving capital from lagging to leading sectors."
                    )
            else:
                self.sector_allocation_label.config(
                    text="🟡 NEUTRAL - No clear rotation signal",
                    foreground=COLORS['warning']
                )
        
        # Update sector momentum bars
        if hasattr(self, 'sector_momentum_bars'):
            momentum = rotation.get('momentum', {})
            max_abs = max([abs(m) for m in momentum.values()], default=1) or 1
            
            for sector_name, bar_data in self.sector_momentum_bars.items():
                mom = momentum.get(sector_name, 0)
                bar_data['bar']['value'] = min(100, (abs(mom) / max_abs * 100))
                bar_data['value'].config(text=f"{mom:+.2f}%")
                
                if mom > 1:
                    bar_data['status'].config(text="🟢 Leading", foreground=COLORS['gain'])
                elif mom < -1:
                    bar_data['status'].config(text="🔴 Lagging", foreground=COLORS['loss'])
                else:
                    bar_data['status'].config(text="🟡 Neutral", foreground=COLORS['warning'])
        
        # Update leader cards
        if hasattr(self, 'leader_cards'):
            momentum = rotation.get('momentum', {})
            sorted_sectors = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
            for i, card in enumerate(self.leader_cards):
                if i < len(sorted_sectors):
                    sector, mom = sorted_sectors[i]
                    card['sector'].config(text=sector)
                    card['change'].config(text=f"+{mom:.2f}%")
                else:
                    card['sector'].config(text="--")
                    card['change'].config(text="--")
        
        # Update laggard cards
        if hasattr(self, 'laggard_cards'):
            momentum = rotation.get('momentum', {})
            sorted_sectors = sorted(momentum.items(), key=lambda x: x[1])
            for i, card in enumerate(self.laggard_cards):
                if i < len(sorted_sectors):
                    sector, mom = sorted_sectors[i]
                    card['sector'].config(text=sector)
                    card['change'].config(text=f"{mom:.2f}%")
                else:
                    card['sector'].config(text="--")
                    card['change'].config(text="--")
        
        # Update momentum tree
        for item in self.momentum_tree.get_children():
            self.momentum_tree.delete(item)
        
        momentum = rotation.get('momentum', {})
        for rank, (sector, mom) in enumerate(sorted(momentum.items(), key=lambda x: x[1], reverse=True), 1):
            status = "🟢 Leading" if mom > 1 else "🔴 Lagging" if mom < -1 else "🟡 Neutral"
            tag = 'bullish' if mom > 1 else 'bearish' if mom < -1 else 'neutral'
            rec = "OVERWEIGHT" if mom > 2 else "UNDERWEIGHT" if mom < -2 else "NEUTRAL"
            self.momentum_tree.insert('', 'end', values=(
                f"#{rank}", sector, f"{mom:+.2f}%", "--", "--", status, rec
            ), tags=(tag,))
        
        self.rotation_label.config(text=f"✅ Updated at {datetime.now().strftime('%H:%M:%S')}")
    
    def _train_sector_rotation(self):
        """Train the sector rotation ML model."""
        if not self.ml_engine:
            return
        
        self.rotation_label.config(text="🧠 Training ML model...")
        
        def train():
            result = self.ml_engine.train_sector_rotation()
            self.frame.after(0, lambda r=result: self._show_training_result(r))
        
        import threading
        threading.Thread(target=train, daemon=True).start()
    
    def _show_training_result(self, result: Dict):
        """Show result of model training."""
        if result.get('success'):
            accuracy = result.get('accuracy', 0)
            samples = result.get('training_samples', 0)
            self.rotation_label.config(
                text=f"✅ Model trained! Accuracy: {accuracy:.1f}% on {samples} samples"
            )
            self._update_rotation()  # Refresh to show new ML status
        else:
            error = result.get('error', 'Unknown error')
            self.rotation_label.config(
                text=f"⚠️ Training: {error}",
            )
    
    def refresh(self):
        """Refresh all ML data."""
        self._initialize_ml()
