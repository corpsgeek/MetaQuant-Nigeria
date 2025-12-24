"""
ML-Based Sector Rotation Predictor for MetaQuant Nigeria.
Uses XGBoost to predict which sector will lead based on historical patterns.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available. ML sector rotation disabled.")


class SectorRotationPredictor:
    """
    ML-based sector rotation prediction using XGBoost.
    
    Features:
    - Stores daily sector performance to database
    - Computes momentum features (1d, 5d, 20d)
    - Trains classifier to predict next leading sector
    - Provides confidence scores for predictions
    """
    
    # Standard NGX sectors
    SECTORS = [
        'Financial Services', 'Oil & Gas', 'Consumer Goods', 'Industrial Goods',
        'Insurance', 'Conglomerates', 'Healthcare', 'Agriculture', 
        'ICT', 'Utilities', 'Real Estate', 'Construction'
    ]
    
    MIN_TRAINING_DAYS = 30  # Minimum days of data before ML kicks in
    
    def __init__(self, db=None, model_dir: Optional[str] = None):
        """
        Initialize the sector rotation predictor.
        
        Args:
            db: DatabaseManager instance for storing history
            model_dir: Directory to save/load trained models
        """
        self.available = XGB_AVAILABLE
        self.db = db
        
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model
        self.model: Optional[XGBClassifier] = None
        self.sector_to_idx: Dict[str, int] = {s: i for i, s in enumerate(self.SECTORS)}
        self.idx_to_sector: Dict[int, str] = {i: s for i, s in enumerate(self.SECTORS)}
        
        # Training stats
        self.model_accuracy: float = 0.0
        self.training_date: Optional[datetime] = None
        self.data_points: int = 0
        
        # In-memory history (for quick access)
        self.sector_history: List[Dict] = []
        
        # Initialize database table
        self._init_db_table()
        
        # Load existing model
        self._load_model()
        
        # Load history from database
        self._load_history_from_db()
    
    def _init_db_table(self):
        """Create sector_history table if it doesn't exist."""
        if not self.db:
            return
        
        try:
            # Create sequence if needed
            self.db.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_sector_history START 1")
            
            # Create sector history table
            self.db.conn.execute("""
                CREATE TABLE IF NOT EXISTS sector_history (
                    id INTEGER PRIMARY KEY DEFAULT nextval('seq_sector_history'),
                    date DATE NOT NULL,
                    sector VARCHAR NOT NULL,
                    avg_change DOUBLE,
                    total_volume DOUBLE,
                    n_stocks INTEGER,
                    leading_rank INTEGER,
                    market_avg_change DOUBLE,
                    relative_strength DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, sector)
                )
            """)
            
            # Create index
            self.db.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sector_history_date 
                ON sector_history(date)
            """)
            
            logger.info("Sector history table initialized")
            
        except Exception as e:
            logger.error(f"Error creating sector_history table: {e}")
    
    def _load_history_from_db(self):
        """Load sector history from database."""
        if not self.db:
            return
        
        try:
            results = self.db.conn.execute("""
                SELECT date, sector, avg_change, total_volume, n_stocks, 
                       leading_rank, market_avg_change, relative_strength
                FROM sector_history
                ORDER BY date DESC
                LIMIT 500
            """).fetchall()
            
            # Group by date
            by_date = defaultdict(dict)
            for row in results:
                date, sector, avg_change, volume, n, rank, mkt_avg, rel_str = row
                by_date[str(date)][sector] = {
                    'avg_change': avg_change or 0,
                    'volume': volume or 0,
                    'n_stocks': n or 0,
                    'rank': rank or 0,
                    'market_avg': mkt_avg or 0,
                    'relative_strength': rel_str or 0
                }
            
            # Convert to list sorted by date
            self.sector_history = [
                {'date': d, 'sectors': by_date[d]} 
                for d in sorted(by_date.keys())
            ]
            
            self.data_points = len(self.sector_history)
            logger.info(f"Loaded {self.data_points} days of sector history from database")
            
        except Exception as e:
            logger.error(f"Error loading sector history: {e}")
    
    def store_daily_performance(self, stocks: List[Dict], date: Optional[str] = None):
        """
        Store daily sector performance snapshot.
        
        Args:
            stocks: List of stock data dictionaries with sector and change
            date: Date string (YYYY-MM-DD), defaults to today
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Aggregate by sector
            sector_data = defaultdict(lambda: {'changes': [], 'volumes': []})
            
            for stock in stocks:
                sector = stock.get('sector', 'Unknown')
                if sector and sector != 'Unknown':
                    change = stock.get('change', 0) or 0
                    volume = stock.get('volume', 0) or 0
                    sector_data[sector]['changes'].append(change)
                    sector_data[sector]['volumes'].append(volume)
            
            if not sector_data:
                logger.warning("No sector data to store")
                return
            
            # Calculate market average
            all_changes = [c for s in sector_data.values() for c in s['changes'] if c]
            market_avg = np.mean(all_changes) if all_changes else 0
            
            # Calculate sector metrics
            sector_metrics = []
            for sector, data in sector_data.items():
                avg_change = np.mean(data['changes']) if data['changes'] else 0
                total_volume = sum(data['volumes'])
                n_stocks = len(data['changes'])
                relative_strength = avg_change - market_avg
                
                sector_metrics.append({
                    'sector': sector,
                    'avg_change': avg_change,
                    'volume': total_volume,
                    'n_stocks': n_stocks,
                    'market_avg': market_avg,
                    'relative_strength': relative_strength
                })
            
            # Rank by performance
            sector_metrics.sort(key=lambda x: x['avg_change'], reverse=True)
            for rank, m in enumerate(sector_metrics, 1):
                m['rank'] = rank
            
            # Store to database
            if self.db:
                for m in sector_metrics:
                    self.db.conn.execute("""
                        INSERT INTO sector_history (
                            date, sector, avg_change, total_volume, n_stocks,
                            leading_rank, market_avg_change, relative_strength
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (date, sector) DO UPDATE SET
                            avg_change = EXCLUDED.avg_change,
                            total_volume = EXCLUDED.total_volume,
                            n_stocks = EXCLUDED.n_stocks,
                            leading_rank = EXCLUDED.leading_rank,
                            market_avg_change = EXCLUDED.market_avg_change,
                            relative_strength = EXCLUDED.relative_strength
                    """, [
                        date, m['sector'], m['avg_change'], m['volume'],
                        m['n_stocks'], m['rank'], m['market_avg'], m['relative_strength']
                    ])
            
            # Update in-memory history
            self.sector_history.append({
                'date': date,
                'sectors': {m['sector']: m for m in sector_metrics}
            })
            
            # Keep last 500 days
            if len(self.sector_history) > 500:
                self.sector_history = self.sector_history[-500:]
            
            self.data_points = len(self.sector_history)
            logger.info(f"Stored sector performance for {date}, {len(sector_metrics)} sectors")
            
        except Exception as e:
            logger.error(f"Error storing sector performance: {e}")
            import traceback
            traceback.print_exc()
    
    def compute_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ML features from sector history.
        
        Returns:
            X: Feature matrix (samples x features)
            y: Target labels (next day's leading sector index)
        """
        if len(self.sector_history) < self.MIN_TRAINING_DAYS:
            return np.array([]), np.array([])
        
        X_list = []
        y_list = []
        
        # We predict tomorrow's leader from today's features
        for i in range(20, len(self.sector_history) - 1):
            today = self.sector_history[i]
            tomorrow = self.sector_history[i + 1]
            
            features = []
            
            for sector in self.SECTORS:
                sector_data = today['sectors'].get(sector, {})
                
                # Today's metrics
                change_1d = sector_data.get('avg_change', 0)
                rel_strength = sector_data.get('relative_strength', 0)
                rank = sector_data.get('rank', 6)  # Default mid-rank
                
                # 5-day momentum
                if i >= 5:
                    changes_5d = []
                    for j in range(i-4, i+1):
                        s_data = self.sector_history[j]['sectors'].get(sector, {})
                        changes_5d.append(s_data.get('avg_change', 0))
                    momentum_5d = np.mean(changes_5d) if changes_5d else 0
                else:
                    momentum_5d = change_1d
                
                # 20-day momentum
                if i >= 20:
                    changes_20d = []
                    for j in range(i-19, i+1):
                        s_data = self.sector_history[j]['sectors'].get(sector, {})
                        changes_20d.append(s_data.get('avg_change', 0))
                    momentum_20d = np.mean(changes_20d) if changes_20d else 0
                else:
                    momentum_20d = momentum_5d
                
                # Volatility (std of last 10 days)
                if i >= 10:
                    changes_10d = []
                    for j in range(i-9, i+1):
                        s_data = self.sector_history[j]['sectors'].get(sector, {})
                        changes_10d.append(s_data.get('avg_change', 0))
                    volatility = np.std(changes_10d) if changes_10d else 0
                else:
                    volatility = 0
                
                # Add features for this sector
                features.extend([
                    change_1d, momentum_5d, momentum_20d, 
                    rel_strength, rank, volatility
                ])
            
            X_list.append(features)
            
            # Target: which sector led tomorrow?
            tomorrow_sectors = tomorrow['sectors']
            leading_sector = min(
                tomorrow_sectors.items(), 
                key=lambda x: x[1].get('rank', 99)
            )[0] if tomorrow_sectors else 'Financial Services'
            
            y_idx = self.sector_to_idx.get(leading_sector, 0)
            y_list.append(y_idx)
        
        return np.array(X_list), np.array(y_list)
    
    def train(self) -> Dict[str, Any]:
        """
        Train the XGBoost classifier on historical sector data.
        
        Returns:
            Dictionary with training results
        """
        if not self.available:
            return {'success': False, 'error': 'XGBoost not available'}
        
        if self.data_points < self.MIN_TRAINING_DAYS:
            return {
                'success': False, 
                'error': f'Insufficient data. Need {self.MIN_TRAINING_DAYS} days, have {self.data_points}'
            }
        
        try:
            logger.info(f"Training sector rotation model on {self.data_points} days of data...")
            
            X, y = self.compute_features()
            
            if len(X) < 10:
                return {'success': False, 'error': 'Insufficient training samples'}
            
            # Train/test split (80/20, time-based)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train XGBoost
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            if len(X_test) > 0:
                y_pred = self.model.predict(X_test)
                self.model_accuracy = (y_pred == y_test).mean() * 100
            else:
                self.model_accuracy = 0
            
            self.training_date = datetime.now()
            
            # Save model
            self._save_model()
            
            logger.info(f"Model trained. Accuracy: {self.model_accuracy:.1f}%")
            
            return {
                'success': True,
                'accuracy': self.model_accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'data_points': self.data_points
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def predict(self) -> Dict[str, Any]:
        """
        Predict which sector will lead next.
        
        Returns:
            Dictionary with prediction, confidence, and analysis
        """
        # Fallback to statistical analysis if ML not ready
        if not self.model or self.data_points < self.MIN_TRAINING_DAYS:
            return self._statistical_analysis()
        
        try:
            # Get recent history for features
            if len(self.sector_history) < 20:
                return self._statistical_analysis()
            
            # Compute features for latest day
            i = len(self.sector_history) - 1
            today = self.sector_history[i]
            
            features = []
            for sector in self.SECTORS:
                sector_data = today['sectors'].get(sector, {})
                
                change_1d = sector_data.get('avg_change', 0)
                rel_strength = sector_data.get('relative_strength', 0)
                rank = sector_data.get('rank', 6)
                
                # 5-day momentum
                changes_5d = []
                for j in range(max(0, i-4), i+1):
                    s_data = self.sector_history[j]['sectors'].get(sector, {})
                    changes_5d.append(s_data.get('avg_change', 0))
                momentum_5d = np.mean(changes_5d) if changes_5d else 0
                
                # 20-day momentum
                changes_20d = []
                for j in range(max(0, i-19), i+1):
                    s_data = self.sector_history[j]['sectors'].get(sector, {})
                    changes_20d.append(s_data.get('avg_change', 0))
                momentum_20d = np.mean(changes_20d) if changes_20d else 0
                
                # Volatility
                changes_10d = []
                for j in range(max(0, i-9), i+1):
                    s_data = self.sector_history[j]['sectors'].get(sector, {})
                    changes_10d.append(s_data.get('avg_change', 0))
                volatility = np.std(changes_10d) if changes_10d else 0
                
                features.extend([
                    change_1d, momentum_5d, momentum_20d,
                    rel_strength, rank, volatility
                ])
            
            X = np.array([features])
            
            # Predict probabilities
            probs = self.model.predict_proba(X)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx] * 100
            
            predicted_sector = self.idx_to_sector.get(pred_idx, 'Unknown')
            
            # Get current rankings
            current = today['sectors']
            sorted_sectors = sorted(current.items(), key=lambda x: x[1].get('rank', 99))
            leading = [s[0] for s in sorted_sectors[:3]]
            lagging = [s[0] for s in sorted_sectors[-3:]]
            
            # Build momentum dict
            momentum = {s: current.get(s, {}).get('avg_change', 0) for s in current}
            
            return {
                'prediction': 'ML_ROTATION',
                'predicted_sector': predicted_sector,
                'confidence': confidence,
                'model_accuracy': self.model_accuracy,
                'data_points': self.data_points,
                'leading': leading,
                'lagging': lagging,
                'rotating_to': predicted_sector,
                'momentum': momentum,
                'ml_active': True
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._statistical_analysis()
    
    def _statistical_analysis(self) -> Dict[str, Any]:
        """Fallback statistical analysis when ML not ready."""
        if not self.sector_history:
            return {
                'prediction': 'NO_DATA',
                'leading': [],
                'lagging': [],
                'momentum': {},
                'ml_active': False,
                'data_points': 0,
                'confidence': 0
            }
        
        latest = self.sector_history[-1]['sectors']
        sorted_sectors = sorted(latest.items(), key=lambda x: x[1].get('rank', 99))
        
        leading = [s[0] for s in sorted_sectors[:3]]
        lagging = [s[0] for s in sorted_sectors[-3:]]
        momentum = {s: d.get('avg_change', 0) for s, d in latest.items()}
        
        return {
            'prediction': 'STATISTICAL',
            'leading': leading,
            'lagging': lagging,
            'rotating_to': leading[0] if leading else None,
            'momentum': momentum,
            'ml_active': False,
            'data_points': self.data_points,
            'confidence': 40  # Lower confidence for statistical
        }
    
    def _save_model(self):
        """Save model to disk."""
        try:
            model_path = self.model_dir / "sector_rotation_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'accuracy': self.model_accuracy,
                    'training_date': self.training_date,
                    'data_points': self.data_points
                }, f)
            logger.info(f"Sector rotation model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _load_model(self):
        """Load model from disk."""
        try:
            model_path = self.model_dir / "sector_rotation_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data.get('model')
                self.model_accuracy = data.get('accuracy', 0)
                self.training_date = data.get('training_date')
                logger.info(f"Loaded sector rotation model (accuracy: {self.model_accuracy:.1f}%)")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the predictor."""
        return {
            'available': self.available,
            'model_trained': self.model is not None,
            'model_accuracy': self.model_accuracy,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'data_points': self.data_points,
            'min_required': self.MIN_TRAINING_DAYS,
            'ml_ready': self.data_points >= self.MIN_TRAINING_DAYS
        }
