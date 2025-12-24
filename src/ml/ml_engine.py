"""
ML Engine - Main coordinator for all ML models in MetaQuant Nigeria.
Provides unified interface for price prediction, anomaly detection, and clustering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

import pandas as pd

from .xgb_predictor import XGBPredictor
from .anomaly_detector import AnomalyDetector, Anomaly, AnomalyType
from .stock_clusterer import StockClusterer

logger = logging.getLogger(__name__)


class MLEngine:
    """
    Coordinates all ML models for MetaQuant Nigeria.
    
    Provides:
    - Price prediction (XGBoost)
    - Anomaly detection (Isolation Forest + statistical)
    - Stock clustering (K-Means)
    - Sector rotation prediction
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the ML Engine.
        
        Args:
            model_dir: Directory for storing trained models
        """
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-engines
        self.predictor = XGBPredictor(model_dir=str(self.model_dir))
        self.anomaly_detector = AnomalyDetector(model_dir=str(self.model_dir))
        self.clusterer = StockClusterer(model_dir=str(self.model_dir))
        
        # Status tracking
        self.initialized = False
        self.last_training_time: Optional[datetime] = None
        self.training_in_progress = False
        
        # Cache
        self._prediction_cache: Dict[str, Dict] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        
        logger.info("ML Engine initialized")
    
    @property
    def available(self) -> bool:
        """Check if ML capabilities are available."""
        return self.predictor.available or self.anomaly_detector.available
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all ML components."""
        return {
            'available': self.available,
            'predictor_available': self.predictor.available,
            'anomaly_detector_available': self.anomaly_detector.available,
            'clusterer_available': self.clusterer.available,
            'initialized': self.initialized,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_in_progress': self.training_in_progress,
            'cached_predictions': len(self._prediction_cache),
            'cluster_count': len(self.clusterer.stock_clusters)
        }
    
    # ========== PRICE PREDICTION ==========
    
    def predict_price(self, df: pd.DataFrame, symbol: str, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Predict next-day price direction and magnitude.
        
        Args:
            df: Historical OHLCV data
            symbol: Stock symbol
            force_retrain: Force model retraining
            
        Returns:
            Dictionary with prediction results
        """
        if not self.predictor.available:
            return {
                'success': False,
                'error': 'XGBoost not available',
                'symbol': symbol
            }
        
        # Check cache
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
        if not force_retrain and cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            cached['from_cache'] = True
            return cached
        
        # Make prediction
        result = self.predictor.predict(df, symbol)
        
        if result.get('success'):
            self._prediction_cache[cache_key] = result
            result['from_cache'] = False
        
        return result
    
    def train_predictor(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Train the price predictor for a specific symbol.
        
        Args:
            df: Historical OHLCV data
            symbol: Stock symbol
            
        Returns:
            Training results
        """
        if not self.predictor.available:
            return {'success': False, 'error': 'XGBoost not available'}
        
        self.training_in_progress = True
        try:
            result = self.predictor.train(df, symbol)
            if result.get('success'):
                self.last_training_time = datetime.now()
            return result
        finally:
            self.training_in_progress = False
    
    def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """Get feature importance for a trained model."""
        return self.predictor.get_feature_importance()
    
    # ========== ANOMALY DETECTION ==========
    
    def detect_anomalies(self, df: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Detect trading anomalies in the data.
        
        Args:
            df: Recent OHLCV data
            symbol: Stock symbol
            
        Returns:
            List of detected anomalies as dictionaries
        """
        anomalies = self.anomaly_detector.detect_anomalies(df, symbol)
        
        # Convert to serializable format
        return [
            {
                'symbol': a.symbol,
                'type': a.anomaly_type.value,
                'severity': a.severity,
                'description': a.description,
                'detected_at': a.detected_at.isoformat(),
                'data': a.data
            }
            for a in anomalies
        ]
    
    def detect_accumulation_distribution(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Detect accumulation/distribution patterns.
        
        Args:
            df: OHLCV data
            symbol: Stock symbol
            
        Returns:
            A/D analysis results
        """
        return self.anomaly_detector.detect_accumulation_distribution(df, symbol)
    
    def get_recent_anomalies(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently detected anomalies."""
        anomalies = self.anomaly_detector.get_recent_anomalies(symbol, limit)
        
        return [
            {
                'symbol': a.symbol,
                'type': a.anomaly_type.value,
                'severity': a.severity,
                'description': a.description,
                'detected_at': a.detected_at.isoformat(),
                'data': a.data
            }
            for a in anomalies
        ]
    
    def train_anomaly_detector(self, df: pd.DataFrame) -> bool:
        """
        Train the Isolation Forest anomaly detector.
        
        Args:
            df: Historical data for training
            
        Returns:
            True if training successful
        """
        self.training_in_progress = True
        try:
            success = self.anomaly_detector.train_isolation_forest(df)
            if success:
                self.anomaly_detector.save_model()
                self.last_training_time = datetime.now()
            return success
        finally:
            self.training_in_progress = False
    
    # ========== STOCK CLUSTERING ==========
    
    def cluster_stocks(self, stocks: List[Dict]) -> Dict[str, Any]:
        """
        Cluster stocks by behavioral/fundamental characteristics.
        
        Args:
            stocks: List of stock data dictionaries
            
        Returns:
            Clustering results
        """
        if not self.clusterer.available:
            return {'success': False, 'error': 'sklearn not available'}
        
        self.training_in_progress = True
        try:
            result = self.clusterer.fit_clusters(stocks)
            if result.get('success'):
                self.initialized = True
                self.last_training_time = datetime.now()
            return result
        finally:
            self.training_in_progress = False
    
    def get_cluster(self, symbol: str) -> Dict[str, Any]:
        """
        Get cluster information for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Cluster info with label and similar stocks
        """
        cluster_id = self.clusterer.get_cluster(symbol)
        
        if cluster_id is None:
            return {
                'symbol': symbol,
                'cluster': None,
                'label': 'Unknown',
                'similar_stocks': []
            }
        
        return {
            'symbol': symbol,
            'cluster': cluster_id,
            'label': self.clusterer.get_cluster_label(cluster_id),
            'similar_stocks': self.clusterer.get_similar_stocks(symbol)
        }
    
    def get_all_clusters(self) -> Dict[int, Dict[str, Any]]:
        """Get all cluster information."""
        result = {}
        
        for cluster_id in range(self.clusterer.n_clusters):
            stocks = self.clusterer.get_cluster_stocks(cluster_id)
            result[cluster_id] = {
                'label': self.clusterer.get_cluster_label(cluster_id),
                'size': len(stocks),
                'stocks': stocks[:10],  # First 10
                'has_more': len(stocks) > 10
            }
        
        return result
    
    # ========== SECTOR ROTATION ==========
    
    def track_sector_performance(self, stocks: List[Dict]):
        """Track sector performance for rotation analysis."""
        self.clusterer.track_sector_performance(stocks)
    
    def predict_sector_rotation(self) -> Dict[str, Any]:
        """Predict sector rotation."""
        return self.clusterer.predict_sector_rotation()
    
    # ========== COMPREHENSIVE ANALYSIS ==========
    
    def get_comprehensive_analysis(self, df: pd.DataFrame, symbol: str, 
                                    all_stocks: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Get comprehensive ML analysis for a stock.
        
        Args:
            df: OHLCV data for the symbol
            symbol: Stock symbol
            all_stocks: All stock data for clustering context
            
        Returns:
            Complete ML analysis
        """
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'ml_available': self.available
        }
        
        # Price prediction
        if self.predictor.available:
            prediction = self.predict_price(df, symbol)
            analysis['prediction'] = prediction
        else:
            analysis['prediction'] = {'available': False}
        
        # Anomaly detection
        if self.anomaly_detector.available:
            anomalies = self.detect_anomalies(df, symbol)
            ad_pattern = self.detect_accumulation_distribution(df, symbol)
            analysis['anomalies'] = anomalies
            analysis['accumulation_distribution'] = ad_pattern
        else:
            analysis['anomalies'] = []
            analysis['accumulation_distribution'] = {'pattern': 'UNAVAILABLE'}
        
        # Clustering
        if self.clusterer.available:
            cluster_info = self.get_cluster(symbol)
            analysis['cluster'] = cluster_info
        else:
            analysis['cluster'] = {'available': False}
        
        # Sector rotation (if we have all stocks)
        if all_stocks:
            self.track_sector_performance(all_stocks)
            analysis['sector_rotation'] = self.predict_sector_rotation()
        
        return analysis
    
    def initialize(self, historical_data: pd.DataFrame, all_stocks: List[Dict]) -> Dict[str, Any]:
        """
        Initialize all ML models with data.
        
        Args:
            historical_data: Historical OHLCV data
            all_stocks: All current stock data
            
        Returns:
            Initialization results
        """
        results = {
            'anomaly_training': False,
            'clustering': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Train anomaly detector
        if self.anomaly_detector.available and not historical_data.empty:
            results['anomaly_training'] = self.train_anomaly_detector(historical_data)
        
        # Cluster stocks
        if self.clusterer.available and all_stocks:
            results['clustering'] = self.cluster_stocks(all_stocks)
        
        self.initialized = True
        self.last_training_time = datetime.now()
        
        return results
    
    def clear_cache(self):
        """Clear prediction cache."""
        self._prediction_cache.clear()
