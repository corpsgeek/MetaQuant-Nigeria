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
        
        # Initialize ML Engine
        if ML_AVAILABLE:
            self.ml_engine = MLEngine()
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
    
    def _create_predictions_ui(self):
        """Create price predictions sub-tab UI."""
        main = self.predictions_tab
        
        # ========== HEADER ==========
        header = ttk.Frame(main)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="📈 ML Price Predictions", font=get_font('subheading'),
                  foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        # Symbol Selector
        ttk.Label(header, text="Symbol:", font=get_font('body')).pack(side=tk.LEFT, padx=(30, 5))
        
        self.pred_symbol_var = tk.StringVar(value="DANGCEM")
        self.pred_symbol_combo = ttk.Combobox(header, textvariable=self.pred_symbol_var, width=15)
        self.pred_symbol_combo.pack(side=tk.LEFT, padx=5)
        self.pred_symbol_combo.bind("<<ComboboxSelected>>", lambda e: self._update_prediction())
        
        predict_btn = ttk.Button(header, text="🔮 Predict", command=self._update_prediction)
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        train_btn = ttk.Button(header, text="🎯 Train Model", command=self._train_model)
        train_btn.pack(side=tk.LEFT)
        
        # ========== PREDICTION RESULTS ==========
        results_frame = ttk.LabelFrame(main, text="🎯 Prediction Results")
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        results_row = ttk.Frame(results_frame)
        results_row.pack(fill=tk.X, padx=10, pady=10)
        
        self.pred_results = {}
        
        # Direction Card
        card1 = ttk.Frame(results_row, relief='ridge', borderwidth=1, padding=10)
        card1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(card1, text="📊 Direction", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['direction'] = ttk.Label(card1, text="--", font=get_font('heading'))
        self.pred_results['direction'].pack()
        
        # Confidence Card
        card2 = ttk.Frame(results_row, relief='ridge', borderwidth=1, padding=10)
        card2.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(card2, text="💪 Confidence", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['confidence'] = ttk.Label(card2, text="--", font=get_font('subheading'))
        self.pred_results['confidence'].pack()
        
        # Expected Return Card
        card3 = ttk.Frame(results_row, relief='ridge', borderwidth=1, padding=10)
        card3.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(card3, text="📈 Expected Return", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['return'] = ttk.Label(card3, text="--", font=get_font('subheading'))
        self.pred_results['return'].pack()
        
        # Predicted Price Card
        card4 = ttk.Frame(results_row, relief='ridge', borderwidth=1, padding=10)
        card4.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(card4, text="💰 Predicted Price", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['price'] = ttk.Label(card4, text="--", font=get_font('subheading'))
        self.pred_results['price'].pack()
        
        # Model Accuracy Card
        card5 = ttk.Frame(results_row, relief='ridge', borderwidth=1, padding=10)
        card5.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(card5, text="🎯 Model Accuracy", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.pred_results['accuracy'] = ttk.Label(card5, text="--", font=get_font('subheading'))
        self.pred_results['accuracy'].pack()
        
        # ========== FEATURE IMPORTANCE ==========
        features_frame = ttk.LabelFrame(main, text="📊 Top Feature Importance")
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for features
        cols = ('Feature', 'Importance')
        self.features_tree = ttk.Treeview(features_frame, columns=cols, show='headings', height=10)
        
        for col in cols:
            self.features_tree.heading(col, text=col)
            self.features_tree.column(col, width=200 if col == 'Feature' else 100)
        
        scrollbar = ttk.Scrollbar(features_frame, orient=tk.VERTICAL, command=self.features_tree.yview)
        self.features_tree.configure(yscrollcommand=scrollbar.set)
        
        self.features_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ========== STATUS ==========
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pred_status = ttk.Label(status_frame, text="Select a symbol and click Predict",
                                     font=get_font('small'), foreground=COLORS['text_muted'])
        self.pred_status.pack(side=tk.LEFT)
    
    def _create_anomaly_ui(self):
        """Create anomaly detection sub-tab UI."""
        main = self.anomaly_tab
        
        # ========== HEADER ==========
        header = ttk.Frame(main)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="⚠️ Anomaly Detection", font=get_font('subheading'),
                  foreground=COLORS['warning']).pack(side=tk.LEFT)
        
        scan_btn = ttk.Button(header, text="🔍 Scan All Stocks", command=self._scan_anomalies)
        scan_btn.pack(side=tk.RIGHT)
        
        # ========== SUMMARY CARDS ==========
        summary_frame = ttk.LabelFrame(main, text="📊 Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        summary_row = ttk.Frame(summary_frame)
        summary_row.pack(fill=tk.X, padx=10, pady=10)
        
        self.anomaly_summary = {}
        
        summaries = [
            ('total', '⚠️ Total Anomalies', COLORS['warning']),
            ('volume', '📊 Volume Spikes', COLORS['primary']),
            ('price', '💰 Price Jumps', COLORS['gain']),
            ('accumulation', '📈 Accumulation', COLORS['gain']),
            ('distribution', '📉 Distribution', COLORS['loss'])
        ]
        
        for key, title, color in summaries:
            card = ttk.Frame(summary_row, relief='ridge', borderwidth=1, padding=10)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
            ttk.Label(card, text=title, font=get_font('small'), foreground=COLORS['text_muted']).pack()
            self.anomaly_summary[key] = ttk.Label(card, text="0", font=get_font('subheading'), foreground=color)
            self.anomaly_summary[key].pack()
        
        # ========== ANOMALY LIST ==========
        list_frame = ttk.LabelFrame(main, text="🚨 Recent Anomalies")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        cols = ('Time', 'Symbol', 'Type', 'Severity', 'Description')
        self.anomaly_tree = ttk.Treeview(list_frame, columns=cols, show='headings', height=15)
        
        widths = {'Time': 100, 'Symbol': 80, 'Type': 120, 'Severity': 80, 'Description': 400}
        for col in cols:
            self.anomaly_tree.heading(col, text=col)
            self.anomaly_tree.column(col, width=widths.get(col, 100))
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.anomaly_tree.yview)
        self.anomaly_tree.configure(yscrollcommand=scrollbar.set)
        
        self.anomaly_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status
        self.anomaly_status = ttk.Label(main, text="Click 'Scan All Stocks' to detect anomalies",
                                        font=get_font('small'), foreground=COLORS['text_muted'])
        self.anomaly_status.pack(pady=5)
    
    def _create_clusters_ui(self):
        """Create stock clusters sub-tab UI."""
        main = self.clusters_tab
        
        # ========== HEADER ==========
        header = ttk.Frame(main)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="📊 Stock Clusters", font=get_font('subheading'),
                  foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        cluster_btn = ttk.Button(header, text="🔄 Re-cluster Stocks", command=self._recluster_stocks)
        cluster_btn.pack(side=tk.RIGHT)
        
        # ========== CLUSTER OVERVIEW ==========
        overview_frame = ttk.LabelFrame(main, text="📋 Cluster Overview")
        overview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.cluster_cards_frame = ttk.Frame(overview_frame)
        self.cluster_cards_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.cluster_cards = {}
        
        # Create 8 cluster cards
        for i in range(8):
            card = ttk.Frame(self.cluster_cards_frame, relief='ridge', borderwidth=1, padding=8)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)
            
            label_lbl = ttk.Label(card, text=f"Cluster {i}", font=get_font('small'), foreground=COLORS['primary'])
            label_lbl.pack()
            count_lbl = ttk.Label(card, text="0 stocks", font=get_font('small'), foreground=COLORS['text_muted'])
            count_lbl.pack()
            
            self.cluster_cards[i] = {'label': label_lbl, 'count': count_lbl}
        
        # ========== SIMILAR STOCKS FINDER ==========
        finder_frame = ttk.LabelFrame(main, text="🔍 Find Similar Stocks")
        finder_frame.pack(fill=tk.X, padx=10, pady=5)
        
        finder_row = ttk.Frame(finder_frame)
        finder_row.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(finder_row, text="Symbol:", font=get_font('body')).pack(side=tk.LEFT)
        
        self.cluster_symbol_var = tk.StringVar(value="DANGCEM")
        self.cluster_symbol_entry = ttk.Entry(finder_row, textvariable=self.cluster_symbol_var, width=15)
        self.cluster_symbol_entry.pack(side=tk.LEFT, padx=10)
        
        find_btn = ttk.Button(finder_row, text="🔍 Find Similar", command=self._find_similar)
        find_btn.pack(side=tk.LEFT)
        
        # Results
        self.similar_results = ttk.Label(finder_row, text="", font=get_font('body'))
        self.similar_results.pack(side=tk.LEFT, padx=20)
        
        # ========== CLUSTER MEMBERS ==========
        members_frame = ttk.LabelFrame(main, text="👥 Cluster Members")
        members_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        cols = ('Symbol', 'Cluster', 'Label')
        self.members_tree = ttk.Treeview(members_frame, columns=cols, show='headings', height=12)
        
        for col in cols:
            self.members_tree.heading(col, text=col)
            self.members_tree.column(col, width=150 if col == 'Label' else 100)
        
        scrollbar = ttk.Scrollbar(members_frame, orient=tk.VERTICAL, command=self.members_tree.yview)
        self.members_tree.configure(yscrollcommand=scrollbar.set)
        
        self.members_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status
        self.cluster_status = ttk.Label(main, text="Loading clusters...",
                                        font=get_font('small'), foreground=COLORS['text_muted'])
        self.cluster_status.pack(pady=5)
    
    def _create_rotation_ui(self):
        """Create sector rotation sub-tab UI."""
        main = self.rotation_tab
        
        # ========== HEADER ==========
        header = ttk.Frame(main)
        header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header, text="🔄 Sector Rotation Analysis", font=get_font('subheading'),
                  foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        refresh_btn = ttk.Button(header, text="↻ Refresh", command=self._update_rotation)
        refresh_btn.pack(side=tk.RIGHT)
        
        # ========== ROTATION STATUS ==========
        status_frame = ttk.LabelFrame(main, text="📊 Rotation Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        status_row = ttk.Frame(status_frame)
        status_row.pack(fill=tk.X, padx=10, pady=10)
        
        self.rotation_status = {}
        
        # Prediction card
        card1 = ttk.Frame(status_row, relief='ridge', borderwidth=1, padding=15)
        card1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(card1, text="🎯 Prediction", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.rotation_status['prediction'] = ttk.Label(card1, text="--", font=get_font('subheading'))
        self.rotation_status['prediction'].pack()
        
        # Rotating To card
        card2 = ttk.Frame(status_row, relief='ridge', borderwidth=1, padding=15)
        card2.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(card2, text="🔄 Rotating To", font=get_font('small'), foreground=COLORS['text_muted']).pack()
        self.rotation_status['rotating_to'] = ttk.Label(card2, text="--", font=get_font('subheading'), foreground=COLORS['gain'])
        self.rotation_status['rotating_to'].pack()
        
        # ========== SECTOR LEADERS/LAGGARDS ==========
        ll_frame = ttk.Frame(main)
        ll_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Leaders
        leaders_frame = ttk.LabelFrame(ll_frame, text="🏆 Leading Sectors")
        leaders_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.leaders_list = tk.Listbox(leaders_frame, height=6, font=get_font('body'),
                                       bg=COLORS['bg_medium'], fg=COLORS['gain'])
        self.leaders_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Laggards
        laggards_frame = ttk.LabelFrame(ll_frame, text="📉 Lagging Sectors")
        laggards_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.laggards_list = tk.Listbox(laggards_frame, height=6, font=get_font('body'),
                                        bg=COLORS['bg_medium'], fg=COLORS['loss'])
        self.laggards_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ========== SECTOR MOMENTUM ==========
        momentum_frame = ttk.LabelFrame(main, text="📊 Sector Momentum")
        momentum_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        cols = ('Sector', 'Momentum', 'Status')
        self.momentum_tree = ttk.Treeview(momentum_frame, columns=cols, show='headings', height=10)
        
        for col in cols:
            self.momentum_tree.heading(col, text=col)
            self.momentum_tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(momentum_frame, orient=tk.VERTICAL, command=self.momentum_tree.yview)
        self.momentum_tree.configure(yscrollcommand=scrollbar.set)
        
        self.momentum_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status
        self.rotation_label = ttk.Label(main, text="Tracking sector momentum...",
                                        font=get_font('small'), foreground=COLORS['text_muted'])
        self.rotation_label.pack(pady=5)
    
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
                # Get historical data
                from src.collectors.tradingview_collector import TradingViewCollector
                collector = TradingViewCollector()
                df = collector.fetch_bars(symbol, interval='1D', n_bars=200)
                
                if df is None or df.empty:
                    self.frame.after(0, lambda: self.pred_status.config(text="No data available"))
                    return
                
                # Make prediction
                result = self.ml_engine.predict_price(df, symbol)
                
                # Update UI
                self.frame.after(0, lambda: self._display_prediction(result))
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                self.frame.after(0, lambda: self.pred_status.config(text=f"Error: {e}"))
        
        threading.Thread(target=predict, daemon=True).start()
    
    def _display_prediction(self, result: Dict):
        """Display prediction results."""
        if not result.get('success'):
            self.pred_status.config(text=f"Prediction failed: {result.get('error', 'Unknown')}")
            return
        
        # Direction
        direction = result.get('direction', '--')
        dir_color = COLORS['gain'] if direction == 'UP' else COLORS['loss'] if direction == 'DOWN' else COLORS['warning']
        self.pred_results['direction'].config(text=direction, foreground=dir_color)
        
        # Confidence
        conf = result.get('confidence', 0)
        self.pred_results['confidence'].config(text=f"{conf:.1f}%")
        
        # Expected return
        ret = result.get('expected_return', 0)
        ret_color = COLORS['gain'] if ret > 0 else COLORS['loss'] if ret < 0 else COLORS['text_primary']
        self.pred_results['return'].config(text=f"{ret:+.2f}%", foreground=ret_color)
        
        # Predicted price
        pred_price = result.get('predicted_price', 0)
        self.pred_results['price'].config(text=f"₦{pred_price:,.2f}")
        
        # Model accuracy
        acc = result.get('model_accuracy', 0)
        self.pred_results['accuracy'].config(text=f"{acc:.1f}%")
        
        # Feature importance
        importance = self.ml_engine.get_feature_importance(result.get('symbol', ''))
        self._display_feature_importance(importance)
        
        self.pred_status.config(text=f"Prediction complete at {datetime.now().strftime('%H:%M:%S')}")
    
    def _display_feature_importance(self, importance: Dict[str, float]):
        """Display feature importance in tree."""
        # Clear existing
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        for feature, imp in sorted_features:
            self.features_tree.insert('', 'end', values=(feature, f"{imp:.4f}"))
    
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
                from src.collectors.tradingview_collector import TradingViewCollector
                collector = TradingViewCollector()
                df = collector.fetch_bars(symbol, interval='1D', n_bars=500)
                
                if df is None or df.empty:
                    self.frame.after(0, lambda: self.pred_status.config(text="No data for training"))
                    return
                
                result = self.ml_engine.train_predictor(df, symbol)
                
                msg = f"Training complete! Accuracy: {result.get('direction_accuracy', 0)*100:.1f}%" if result.get('success') else f"Training failed: {result.get('error')}"
                self.frame.after(0, lambda: self.pred_status.config(text=msg))
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                self.frame.after(0, lambda: self.pred_status.config(text=f"Error: {e}"))
        
        threading.Thread(target=train, daemon=True).start()
    
    def _scan_anomalies(self):
        """Scan all stocks for anomalies."""
        if not self.ml_engine:
            return
        
        self.anomaly_status.config(text="Scanning for anomalies...")
        
        def scan():
            try:
                from src.collectors.tradingview_collector import TradingViewCollector
                collector = TradingViewCollector()
                
                all_anomalies = []
                counts = {'volume_spike': 0, 'price_jump': 0, 'accumulation': 0, 'distribution': 0}
                
                # Scan top 30 stocks
                for stock in self.all_stocks_data[:30]:
                    symbol = stock.get('symbol', '')
                    if not symbol:
                        continue
                    
                    try:
                        df = collector.fetch_bars(symbol, interval='1D', n_bars=50)
                        if df is not None and not df.empty:
                            anomalies = self.ml_engine.detect_anomalies(df, symbol)
                            
                            for a in anomalies:
                                all_anomalies.append(a)
                                atype = a.get('type', '')
                                if atype in counts:
                                    counts[atype] += 1
                    except:
                        continue
                
                self.frame.after(0, lambda: self._display_anomalies(all_anomalies, counts))
                
            except Exception as e:
                logger.error(f"Anomaly scan error: {e}")
                self.frame.after(0, lambda: self.anomaly_status.config(text=f"Error: {e}"))
        
        threading.Thread(target=scan, daemon=True).start()
    
    def _display_anomalies(self, anomalies: List[Dict], counts: Dict):
        """Display detected anomalies."""
        # Update summary
        self.anomaly_summary['total'].config(text=str(len(anomalies)))
        self.anomaly_summary['volume'].config(text=str(counts.get('volume_spike', 0)))
        self.anomaly_summary['price'].config(text=str(counts.get('price_jump', 0)))
        self.anomaly_summary['accumulation'].config(text=str(counts.get('accumulation', 0)))
        self.anomaly_summary['distribution'].config(text=str(counts.get('distribution', 0)))
        
        # Clear tree
        for item in self.anomaly_tree.get_children():
            self.anomaly_tree.delete(item)
        
        # Sort by severity
        sorted_anomalies = sorted(anomalies, key=lambda x: x.get('severity', 0), reverse=True)
        
        for a in sorted_anomalies[:50]:
            time_str = a.get('detected_at', '')[:19].replace('T', ' ')
            self.anomaly_tree.insert('', 'end', values=(
                time_str,
                a.get('symbol', ''),
                a.get('type', '').replace('_', ' ').title(),
                f"{a.get('severity', 0):.0f}%",
                a.get('description', '')
            ))
        
        self.anomaly_status.config(text=f"Found {len(anomalies)} anomalies at {datetime.now().strftime('%H:%M:%S')}")
    
    def _recluster_stocks(self):
        """Re-cluster all stocks."""
        if not self.ml_engine or not self.all_stocks_data:
            return
        
        self.cluster_status.config(text="Clustering stocks...")
        
        def cluster():
            result = self.ml_engine.cluster_stocks(self.all_stocks_data)
            if result.get('success'):
                self.frame.after(0, self._update_cluster_display)
            else:
                self.frame.after(0, lambda: self.cluster_status.config(text=f"Error: {result.get('error')}"))
        
        threading.Thread(target=cluster, daemon=True).start()
    
    def _update_cluster_display(self):
        """Update cluster display."""
        if not self.ml_engine:
            return
        
        clusters = self.ml_engine.get_all_clusters()
        
        # Update cards
        for i, card in self.cluster_cards.items():
            if i in clusters:
                info = clusters[i]
                card['label'].config(text=info.get('label', f'Cluster {i}'))
                card['count'].config(text=f"{info.get('size', 0)} stocks")
        
        # Update tree
        for item in self.members_tree.get_children():
            self.members_tree.delete(item)
        
        for cluster_id, info in clusters.items():
            for symbol in info.get('stocks', []):
                self.members_tree.insert('', 'end', values=(symbol, cluster_id, info.get('label', '')))
        
        self.cluster_status.config(text=f"Clustered {len(self.ml_engine.clusterer.stock_clusters)} stocks at {datetime.now().strftime('%H:%M:%S')}")
    
    def _find_similar(self):
        """Find similar stocks."""
        if not self.ml_engine:
            return
        
        symbol = self.cluster_symbol_var.get().upper()
        cluster_info = self.ml_engine.get_cluster(symbol)
        
        similar = cluster_info.get('similar_stocks', [])
        label = cluster_info.get('label', 'Unknown')
        
        if similar:
            self.similar_results.config(
                text=f"Cluster: {label} | Similar: {', '.join(similar[:5])}",
                foreground=COLORS['gain']
            )
        else:
            self.similar_results.config(text="No similar stocks found", foreground=COLORS['loss'])
    
    def _update_rotation(self):
        """Update sector rotation analysis."""
        if not self.ml_engine or not self.all_stocks_data:
            return
        
        # Track current performance
        self.ml_engine.track_sector_performance(self.all_stocks_data)
        
        # Get prediction
        rotation = self.ml_engine.predict_sector_rotation()
        
        # Update UI
        self.rotation_status['prediction'].config(text=rotation.get('prediction', 'UNKNOWN'))
        
        rotating_to = rotation.get('rotating_to')
        if rotating_to:
            self.rotation_status['rotating_to'].config(text=rotating_to, foreground=COLORS['gain'])
        else:
            self.rotation_status['rotating_to'].config(text="--", foreground=COLORS['text_muted'])
        
        # Update leaders
        self.leaders_list.delete(0, tk.END)
        for sector in rotation.get('leading', []):
            self.leaders_list.insert(tk.END, f"  ▲ {sector}")
        
        # Update laggards
        self.laggards_list.delete(0, tk.END)
        for sector in rotation.get('lagging', []):
            self.laggards_list.insert(tk.END, f"  ▼ {sector}")
        
        # Update momentum tree
        for item in self.momentum_tree.get_children():
            self.momentum_tree.delete(item)
        
        momentum = rotation.get('momentum', {})
        for sector, mom in sorted(momentum.items(), key=lambda x: x[1], reverse=True):
            status = "Leading" if mom > 1 else "Lagging" if mom < -1 else "Neutral"
            self.momentum_tree.insert('', 'end', values=(sector, f"{mom:+.2f}%", status))
        
        self.rotation_label.config(text=f"Updated at {datetime.now().strftime('%H:%M:%S')}")
    
    def refresh(self):
        """Refresh all ML data."""
        self._initialize_ml()
