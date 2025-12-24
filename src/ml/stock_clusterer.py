"""
Stock Clustering and Sector Rotation Prediction for MetaQuant Nigeria.
Uses K-Means clustering to group similar stocks and predict sector rotation.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Stock clustering disabled.")


class StockClusterer:
    """
    Clusters stocks by behavioral and fundamental characteristics.
    
    Features:
    - K-Means clustering based on volatility, returns, volume, fundamentals
    - Identifies similar stocks
    - Tracks sector rotation patterns
    - Predicts which clusters will outperform
    """
    
    # Standard sectors for NGX
    SECTORS = [
        'Financial Services', 'Industrial Goods', 'Consumer Goods',
        'Oil & Gas', 'Agriculture', 'Healthcare', 'Services',
        'Technology', 'Utilities', 'Real Estate'
    ]
    
    def __init__(self, n_clusters: int = 8, model_dir: Optional[str] = None):
        """
        Initialize the stock clusterer.
        
        Args:
            n_clusters: Number of clusters to create
            model_dir: Directory to save/load trained models
        """
        self.available = SKLEARN_AVAILABLE
        self.n_clusters = n_clusters
        
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        
        # Cluster data
        self.stock_clusters: Dict[str, int] = {}
        self.cluster_centers: Optional[np.ndarray] = None
        self.feature_columns: List[str] = []
        
        # Cluster labels (descriptive names)
        self.cluster_labels: Dict[int, str] = {
            0: "High Growth",
            1: "Value Defensive",
            2: "High Yield",
            3: "Momentum Leaders",
            4: "Blue Chips",
            5: "Small Caps",
            6: "Turnaround",
            7: "Speculative"
        }
        
        # Sector rotation tracking
        self.sector_performance_history: List[Dict[str, float]] = []
        self.cluster_performance_history: List[Dict[int, float]] = []
        
    def compute_features(self, stocks: List[Dict]) -> pd.DataFrame:
        """
        Compute clustering features from stock data.
        
        Args:
            stocks: List of stock data dictionaries
            
        Returns:
            DataFrame with computed features
        """
        if not stocks:
            return pd.DataFrame()
        
        try:
            rows = []
            
            for stock in stocks:
                symbol = stock.get('symbol', '')
                if not symbol:
                    continue
                
                # Extract features with fallbacks
                row = {
                    'symbol': symbol,
                    
                    # Price performance
                    'change_1d': stock.get('change', 0) or 0,
                    'change_1w': stock.get('Perf.W', 0) or stock.get('perf_w', 0) or 0,
                    'change_1m': stock.get('Perf.1M', 0) or stock.get('perf_1m', 0) or 0,
                    'change_ytd': stock.get('Perf.YTD', 0) or stock.get('perf_ytd', 0) or 0,
                    
                    # Volatility
                    'volatility': stock.get('Volatility.D', 0) or stock.get('volatility', 0) or 0,
                    'atr': stock.get('ATR', 0) or stock.get('atr', 0) or 0,
                    
                    # Volume
                    'volume': stock.get('volume', 0) or 0,
                    'rel_volume': stock.get('relative_volume', 0) or stock.get('Relative Volume', 1) or 1,
                    
                    # Fundamentals
                    'pe_ratio': stock.get('price_earnings_ttm', 0) or stock.get('pe_ratio', 0) or 0,
                    'pb_ratio': stock.get('price_book_fq', 0) or stock.get('pb_ratio', 0) or 0,
                    'dividend_yield': stock.get('dividend_yield_recent', 0) or stock.get('dividend_yield', 0) or 0,
                    'market_cap': stock.get('market_cap_basic', 0) or stock.get('market_cap', 0) or 0,
                    
                    # Technical
                    'rsi': stock.get('RSI', 50) or stock.get('rsi', 50) or 50,
                    'adx': stock.get('ADX', 20) or stock.get('adx', 20) or 20,
                    
                    # Sector (encoded later)
                    'sector': stock.get('sector', 'Unknown') or 'Unknown'
                }
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            if df.empty:
                return pd.DataFrame()
            
            # Set symbol as index
            df = df.set_index('symbol')
            
            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))
            
            # Log transform market cap (wide range)
            df['market_cap_log'] = np.log1p(df['market_cap'].clip(lower=0))
            
            # Sector encoding (one-hot)
            sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
            
            # Drop original sector and market_cap
            df = df.drop(columns=['sector', 'market_cap'])
            
            # Combine
            df = pd.concat([df, sector_dummies], axis=1)
            
            # Store feature columns
            self.feature_columns = df.columns.tolist()
            
            return df
            
        except Exception as e:
            logger.error(f"Error computing clustering features: {e}")
            return pd.DataFrame()
    
    def fit_clusters(self, stocks: List[Dict]) -> Dict[str, Any]:
        """
        Fit K-Means clustering on stock data.
        
        Args:
            stocks: List of stock data dictionaries
            
        Returns:
            Dictionary with clustering results
        """
        if not self.available:
            return {'success': False, 'error': 'scikit-learn not available'}
        
        try:
            # Compute features
            features = self.compute_features(stocks)
            
            if features.empty or len(features) < self.n_clusters:
                return {'success': False, 'error': 'Insufficient data for clustering'}
            
            logger.info(f"Clustering {len(features)} stocks into {self.n_clusters} clusters...")
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features.values)
            
            # Optional: PCA for dimensionality reduction
            if X_scaled.shape[1] > 10:
                self.pca = PCA(n_components=min(10, X_scaled.shape[1]))
                X_pca = self.pca.fit_transform(X_scaled)
            else:
                X_pca = X_scaled
            
            # Fit K-Means
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            clusters = self.kmeans.fit_predict(X_pca)
            
            # Store cluster assignments
            self.stock_clusters = dict(zip(features.index, clusters))
            self.cluster_centers = self.kmeans.cluster_centers_
            
            # Analyze clusters
            cluster_stats = self._analyze_clusters(features, clusters)
            
            # Update cluster labels based on characteristics
            self._update_cluster_labels(cluster_stats)
            
            # Save model
            self._save_model()
            
            logger.info(f"Clustering complete. Inertia: {self.kmeans.inertia_:.2f}")
            
            return {
                'success': True,
                'n_stocks': len(features),
                'n_clusters': self.n_clusters,
                'inertia': self.kmeans.inertia_,
                'cluster_sizes': {i: int(np.sum(clusters == i)) for i in range(self.n_clusters)},
                'cluster_stats': cluster_stats
            }
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _analyze_clusters(self, features: pd.DataFrame, clusters: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Analyze characteristics of each cluster."""
        stats = {}
        
        for i in range(self.n_clusters):
            mask = clusters == i
            if not np.any(mask):
                continue
            
            cluster_data = features[mask]
            
            stats[i] = {
                'size': int(np.sum(mask)),
                'avg_change_1d': float(cluster_data['change_1d'].mean()),
                'avg_change_1w': float(cluster_data['change_1w'].mean()),
                'avg_volatility': float(cluster_data['volatility'].mean()),
                'avg_pe': float(cluster_data['pe_ratio'].mean()),
                'avg_dividend': float(cluster_data['dividend_yield'].mean()),
                'avg_rsi': float(cluster_data['rsi'].mean())
            }
        
        return stats
    
    def _update_cluster_labels(self, cluster_stats: Dict[int, Dict[str, float]]):
        """Update cluster labels based on characteristics."""
        for cluster_id, stats in cluster_stats.items():
            if stats['avg_dividend'] > 5:
                self.cluster_labels[cluster_id] = "High Yield"
            elif stats['avg_pe'] < 5 and stats['avg_pe'] > 0:
                self.cluster_labels[cluster_id] = "Deep Value"
            elif stats['avg_change_1w'] > 5:
                self.cluster_labels[cluster_id] = "Momentum Leaders"
            elif stats['avg_volatility'] < 1:
                self.cluster_labels[cluster_id] = "Low Volatility"
            elif stats['avg_volatility'] > 5:
                self.cluster_labels[cluster_id] = "High Beta"
            elif stats['avg_rsi'] > 70:
                self.cluster_labels[cluster_id] = "Overbought"
            elif stats['avg_rsi'] < 30:
                self.cluster_labels[cluster_id] = "Oversold"
    
    def get_cluster(self, symbol: str) -> Optional[int]:
        """Get cluster ID for a stock."""
        return self.stock_clusters.get(symbol.upper())
    
    def get_cluster_label(self, cluster_id: int) -> str:
        """Get descriptive label for a cluster."""
        return self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
    
    def get_cluster_stocks(self, cluster_id: int) -> List[str]:
        """Get all stocks in a cluster."""
        return [s for s, c in self.stock_clusters.items() if c == cluster_id]
    
    def get_similar_stocks(self, symbol: str, limit: int = 5) -> List[str]:
        """Get stocks similar to the given symbol."""
        cluster_id = self.get_cluster(symbol.upper())
        if cluster_id is None:
            return []
        
        similar = [s for s in self.get_cluster_stocks(cluster_id) if s != symbol.upper()]
        return similar[:limit]
    
    def track_sector_performance(self, stocks: List[Dict]):
        """
        Track sector performance for rotation analysis.
        
        Args:
            stocks: List of stock data with sector and performance info
        """
        try:
            sector_perf = defaultdict(list)
            
            for stock in stocks:
                sector = stock.get('sector', 'Unknown')
                change = stock.get('change', 0) or 0
                sector_perf[sector].append(change)
            
            # Average performance per sector
            avg_perf = {s: np.mean(p) for s, p in sector_perf.items() if p}
            
            # Store with timestamp
            self.sector_performance_history.append({
                'timestamp': datetime.now(),
                'performance': avg_perf
            })
            
            # Keep last 50 snapshots
            if len(self.sector_performance_history) > 50:
                self.sector_performance_history = self.sector_performance_history[-50:]
                
        except Exception as e:
            logger.error(f"Error tracking sector performance: {e}")
    
    def predict_sector_rotation(self) -> Dict[str, Any]:
        """
        Predict sector rotation based on recent performance trends.
        
        Returns:
            Dictionary with rotation predictions
        """
        if len(self.sector_performance_history) < 5:
            return {
                'prediction': 'INSUFFICIENT_DATA',
                'leading': [],
                'lagging': [],
                'rotating_to': None
            }
        
        try:
            # Get recent performance
            recent = self.sector_performance_history[-5:]
            
            # Calculate momentum for each sector
            sector_momentum = defaultdict(list)
            for snapshot in recent:
                for sector, perf in snapshot.get('performance', {}).items():
                    sector_momentum[sector].append(perf)
            
            # Average momentum
            avg_momentum = {s: np.mean(m) for s, m in sector_momentum.items() if m}
            
            # Sort by momentum
            sorted_sectors = sorted(avg_momentum.items(), key=lambda x: x[1], reverse=True)
            
            leading = [s[0] for s in sorted_sectors[:3] if s[1] > 0]
            lagging = [s[0] for s in sorted_sectors[-3:] if s[1] < 0]
            
            # Detect rotation (lagging sector gaining momentum)
            rotation_candidates = []
            for snapshot in recent[-3:]:
                for sector, perf in snapshot.get('performance', {}).items():
                    if sector in [s[0] for s in sorted_sectors[-5:]]:  # Was lagging
                        if perf > 0:  # Now positive
                            rotation_candidates.append(sector)
            
            rotating_to = max(set(rotation_candidates), key=rotation_candidates.count) if rotation_candidates else None
            
            return {
                'prediction': 'ROTATION_DETECTED' if rotating_to else 'STABLE',
                'leading': leading,
                'lagging': lagging,
                'rotating_to': rotating_to,
                'momentum': avg_momentum
            }
            
        except Exception as e:
            logger.error(f"Sector rotation prediction failed: {e}")
            return {'prediction': 'ERROR', 'leading': [], 'lagging': []}
    
    def get_cluster_performance(self) -> Dict[int, float]:
        """Get average performance by cluster."""
        if not self.stock_clusters:
            return {}
        
        # This would need current stock data to compute
        # For now return placeholder
        return {i: 0.0 for i in range(self.n_clusters)}
    
    def _save_model(self):
        """Save clustering model to disk."""
        try:
            model_path = self.model_dir / "cluster_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'kmeans': self.kmeans,
                    'scaler': self.scaler,
                    'pca': self.pca,
                    'stock_clusters': self.stock_clusters,
                    'cluster_labels': self.cluster_labels,
                    'feature_columns': self.feature_columns
                }, f)
            logger.info(f"Cluster model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self) -> bool:
        """Load clustering model from disk."""
        try:
            model_path = self.model_dir / "cluster_model.pkl"
            if not model_path.exists():
                return False
            
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.kmeans = data['kmeans']
            self.scaler = data['scaler']
            self.pca = data['pca']
            self.stock_clusters = data['stock_clusters']
            self.cluster_labels = data['cluster_labels']
            self.feature_columns = data['feature_columns']
            
            logger.info(f"Cluster model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
